import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from torch.utils.data import DataLoader
import numpy as np

@dataclass
class RLHFConfig:
    # Model parameters
    kl_coef: float = 0.1
    clip_range_ratio: float = 0.2
    clip_range_value: float = 10.0
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.01
    
    # Constitutional AI parameters
    constitution_rules: List[str] = None
    rule_weights: List[float] = None
    max_rule_violations: int = 3
    
    # DAPO parameters
    decoupled_clip_eps: float = 0.2
    dynamic_sampling_alpha: float = 0.5
    policy_clip_range: float = 0.2
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden_states)

class SafeRLHFTrainer:
    def __init__(
        self,
        policy_model,
        ref_model,
        reward_model,
        tokenizer,
        config: RLHFConfig,
        constitution_checker=None
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.constitution_checker = constitution_checker
        
        # Initialize value head for policy model
        self.value_head = ValueHead(policy_model.config.hidden_size)
        
        # Setup optimizers
        self.optimizer = torch.optim.AdamW(
            list(policy_model.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate
        )

    def compute_rewards(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Get base rewards from reward model
        base_rewards = self.reward_model(sequences, attention_mask=attention_mask)
        
        # Apply constitutional checks if available
        if self.constitution_checker is not None:
            const_penalties = self.constitution_checker(
                sequences,
                self.config.constitution_rules,
                self.config.rule_weights
            )
            rewards = base_rewards - const_penalties
        else:
            rewards = base_rewards
        
        return rewards

    def compute_policy_values(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute advantages using GAE (Generalized Advantage Estimation)
        advantages = rewards - self.value_head(self.policy_model.hidden_states)
        
        # Compute KL penalty
        kl_penalty = (logprobs - ref_logprobs) * mask
        kl_penalty = kl_penalty.sum(dim=1) / mask.sum(dim=1)
        
        # Apply KL penalty to advantages
        advantages = advantages - self.config.kl_coef * kl_penalty
        
        return advantages, kl_penalty

    def dapo_policy_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        # Compute probability ratio
        ratio = torch.exp(logprobs - old_logprobs)
        
        # Compute clipped ratio
        ratio_clipped = torch.clamp(
            ratio,
            1.0 - self.config.decoupled_clip_eps,
            1.0 + self.config.decoupled_clip_eps
        )
        
        # Dynamic sampling weights
        sampling_weights = torch.pow(
            advantages.abs(),
            self.config.dynamic_sampling_alpha
        )
        sampling_weights = sampling_weights / sampling_weights.sum()
        
        # Compute loss components
        policy_loss = -advantages * ratio * sampling_weights
        policy_loss_clipped = -advantages * ratio_clipped * sampling_weights
        
        # Take maximum loss
        policy_loss = torch.max(policy_loss, policy_loss_clipped)
        policy_loss = (policy_loss * mask).sum() / mask.sum()
        
        return policy_loss

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        # Unpack batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        old_logprobs = batch["logprobs"]
        
        # Forward pass through policy model
        policy_outputs = self.policy_model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = policy_outputs.logits
        logprobs = F.log_softmax(logits, dim=-1)
        
        # Get rewards
        rewards = self.compute_rewards(input_ids, attention_mask)
        
        # Get reference model logprobs
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids,
                attention_mask=attention_mask
            ).logits
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        # Compute advantages and KL penalty
        advantages, kl_penalty = self.compute_policy_values(
            logprobs,
            ref_logprobs,
            rewards,
            attention_mask
        )
        
        # Compute DAPO policy loss
        policy_loss = self.dapo_policy_loss(
            logprobs,
            old_logprobs,
            advantages,
            attention_mask
        )
        
        # Compute value loss
        value_pred = self.value_head(policy_outputs.hidden_states)
        value_loss = F.mse_loss(value_pred, rewards)
        
        # Compute entropy loss for exploration
        entropy_loss = -(logprobs.exp() * logprobs).sum(dim=-1).mean()
        
        # Combine losses
        total_loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss -
            self.config.entropy_coef * entropy_loss
        )
        
        # Backward pass and optimization
        total_loss = total_loss / self.config.gradient_accumulation_steps
        total_loss.backward()
        
        if (self.train_step_counter + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Return metrics
        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "kl_penalty": kl_penalty.mean().item(),
            "mean_reward": rewards.mean().item()
        }

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        self.train_step_counter = 0
        metrics_history = {
            "train_loss": [],
            "train_reward": [],
            "eval_loss": [],
            "eval_reward": []
        }
        
        for epoch in range(self.config.num_epochs):
            # Training loop
            self.policy_model.train()
            epoch_losses = []
            epoch_rewards = []
            
            for batch in train_dataloader:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics["total_loss"])
                epoch_rewards.append(metrics["mean_reward"])
                self.train_step_counter += 1
            
            metrics_history["train_loss"].append(np.mean(epoch_losses))
            metrics_history["train_reward"].append(np.mean(epoch_rewards))
            
            # Evaluation loop
            if eval_dataloader is not None:
                self.policy_model.eval()
                eval_losses = []
                eval_rewards = []
                
                with torch.no_grad():
                    for batch in eval_dataloader:
                        metrics = self.train_step(batch)
                        eval_losses.append(metrics["total_loss"])
                        eval_rewards.append(metrics["mean_reward"])
                
                metrics_history["eval_loss"].append(np.mean(eval_losses))
                metrics_history["eval_reward"].append(np.mean(eval_rewards))
        
        return metrics_history

class ConstitutionalChecker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def check_rule(
        self,
        text: str,
        rule: str,
        weight: float = 1.0
    ) -> float:
        """Check a single constitutional rule and return a penalty score."""
        # Implement rule checking logic here
        # This is a placeholder - actual implementation would depend on specific rules
        violation_count = 0
        penalty = violation_count * weight
        return min(penalty, self.config.max_rule_violations * weight)
    
    def __call__(
        self,
        sequences: torch.Tensor,
        rules: List[str],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if weights is None:
            weights = [1.0] * len(rules)
        
        # Decode sequences to text
        texts = self.tokenizer.batch_decode(sequences)
        
        # Check each rule for each sequence
        penalties = torch.zeros(len(texts), device=sequences.device)
        for text_idx, text in enumerate(texts):
            text_penalty = 0
            for rule, weight in zip(rules, weights):
                text_penalty += self.check_rule(text, rule, weight)
            penalties[text_idx] = text_penalty
        
        return penalties