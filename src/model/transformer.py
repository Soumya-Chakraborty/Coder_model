import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int = 50304  # ByteScale vocabulary size
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # For GQA
    num_hidden_layers: int = 32
    num_experts: int = 8  # For MoE
    experts_per_token: int = 2
    max_position_embeddings: int = 32768
    sliding_window: int = 4096  # For SWA
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layernorm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

class MultiheadLatentAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.sliding_window = config.sliding_window
        
        # GQA: Reduced KV heads for efficiency
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, 
                               self.head_dim * config.num_key_value_heads, 
                               bias=False)
        self.v_proj = nn.Linear(config.hidden_size, 
                               self.head_dim * config.num_key_value_heads, 
                               bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        query_states = query_states.view(
            batch_size, seq_length, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        # Sliding Window Attention
        query_states = self._sliding_chunks_query_key_matmul(
            query_states, key_states, self.sliding_window
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
            attention_mask = attention_mask.masked_fill(
                attention_mask == 0, float("-inf")
            )
            query_states = query_states + attention_mask
        
        # Compute attention weights and dropout
        attention_weights = F.softmax(query_states / math.sqrt(self.head_dim), dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute attention output
        attention_output = torch.matmul(attention_weights, value_states)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, seq_length, self.hidden_size
        )
        
        # Project output
        attention_output = self.o_proj(attention_output)
        
        return attention_output, past_key_value

    def _sliding_chunks_query_key_matmul(
        self, query: torch.Tensor, key: torch.Tensor, window_overlap: int
    ) -> torch.Tensor:
        """Implementation of sliding window attention."""
        batch_size, num_heads, seq_length, head_dim = query.size()
        chunks_count = seq_length // window_overlap - 1

        # Group batch_size and num_heads dimensions into one
        query = query.transpose(1, 2).reshape(batch_size * seq_length, num_heads, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * seq_length, num_heads, head_dim)

        # Compute attention scores for each window
        query_expanded = query.unsqueeze(-2)
        key_expanded = key.unsqueeze(-3)
        
        # Compute attention scores
        attention_scores = torch.matmul(
            query_expanded, key_expanded.transpose(-2, -1)
        ) / math.sqrt(head_dim)

        # Mask out attention scores outside the sliding window
        window_mask = torch.ones(
            (window_overlap, window_overlap), device=query.device
        ).tril(diagonal=window_overlap-1)
        attention_scores = attention_scores.masked_fill(
            window_mask == 0, float("-inf")
        )

        return attention_scores

class MixtureOfExperts(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size)
            ) for _ in range(self.num_experts)
        ])
        
        # Router for selecting experts
        self.router = nn.Linear(config.hidden_size, self.num_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Get router logits and probabilities
        router_logits = self.router(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.experts_per_token, dim=-1
        )
        
        # Normalize probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        for i in range(self.experts_per_token):
            expert_idx = top_k_indices[..., i]
            prob = top_k_probs[..., i].unsqueeze(-1)
            
            # Gather expert outputs
            for j in range(self.num_experts):
                mask = (expert_idx == j).unsqueeze(-1)
                if mask.any():
                    expert_output = self.experts[j](
                        hidden_states[mask.squeeze(-1)]
                    )
                    expert_outputs[mask] += prob[mask] * expert_output
        
        return expert_outputs

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # Pre-LN architecture
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.attention = MultiheadLatentAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.moe = MixtureOfExperts(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # Pre-LN attention block
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attention_output, past_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(attention_output)
        
        # Pre-LN MoE block
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        moe_output = self.moe(hidden_states)
        hidden_states = residual + self.dropout(moe_output)
        
        return hidden_states, past_key_value

class CodingTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings with ByteScale
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        
        # Output projection
        self.output_projection = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor]]]]:
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        hidden_states = self.token_embeddings(input_ids)
        
        # Add positional embeddings
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        hidden_states = hidden_states + self.position_embeddings(position_ids)
        
        # Store new key values
        new_key_values = [] if use_cache else None
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                new_key_values.append(key_value)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Project to vocabulary
        logits = self.output_projection(hidden_states)
        
        return logits, new_key_values

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate text using nucleus (top-p) sampling."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        cur_len = input_ids.shape[1]
        past_key_values = None
        
        while cur_len < max_length:
            # Forward pass
            logits, past_key_values = self.forward(
                input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token
            input_ids = torch.cat([input_ids, next_token], dim=1)
            cur_len += 1
            
            # Check if we've generated an EOS token
            if (next_token == self.config.eos_token_id).any():
                break
        
        return input_ids