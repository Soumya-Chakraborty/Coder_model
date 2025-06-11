from dataclasses import dataclass
from typing import Optional, List, Union
from enum import Enum

class OptimizationType(Enum):
    MUON = "muon"
    ADAM = "adam"
    ADAMW = "adamw"
    LION = "lion"

@dataclass
class MuonConfig:
    """Configuration for Muon optimizer"""
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    momentum_decay: float = 0.1
    variance_reduction: bool = True
    gradient_centralization: bool = True
    adaptive_momentum: bool = True
    warm_restart_cycles: int = 1

@dataclass
class ModelConfig:
    """Configuration for the coding-focused transformer model"""
    # Architecture
    vocab_size: int = 50304  # ByteScale vocabulary size
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # For GQA
    sliding_window_size: int = 4096  # For SWA
    
    # Mixture of Experts
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    router_jitter_noise: float = 0.1
    router_z_loss_coef: float = 0.01
    
    # Multi-head Latent Attention
    num_latent_channels: int = 64
    latent_dim: int = 256
    use_rotary_embeddings: bool = True
    
    # Layer configuration
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    
    # Positional encoding
    max_position_embeddings: int = 32768
    position_embedding_type: str = "rotary"
    
    # ByteScale tokenization
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_bytescale: bool = True
    
    # Training
    max_seq_length: int = 4096
    gradient_checkpointing: bool = False
    tie_word_embeddings: bool = True
    
    # Optimization
    optimizer_type: OptimizationType = OptimizationType.MUON
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    scheduler_type: str = "cosine"
    
    # Parallel training
    use_hybrid_data_parallel: bool = True
    dp_degree: int = 8
    pp_degree: int = 1
    mp_degree: int = 4
    
    # RLHF
    use_rlhf: bool = True
    reward_model_hidden_size: int = 1024
    kl_coef: float = 0.1
    clip_range_ratio: float = 0.2
    value_loss_coef: float = 0.1
    
    # Constitutional AI
    constitution_rules: Optional[List[str]] = None
    rule_weights: Optional[List[float]] = None
    max_rule_violations: int = 3
    
    # SWE-RL
    use_swe_rl: bool = True
    swe_rl_reward_scale: float = 1.0
    swe_rl_horizon: int = 100
    swe_rl_gamma: float = 0.99
    
    def __post_init__(self):
        """Validate and adjust configurations after initialization"""
        # Validate GQA configuration
        assert self.num_key_value_heads <= self.num_attention_heads, \
            "Number of KV heads must be <= number of attention heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "Number of attention heads must be divisible by number of KV heads"
            
        # Validate MoE configuration
        assert self.num_experts_per_token <= self.num_experts, \
            "Number of experts per token must be <= total number of experts"
            
        # Validate parallel training configuration
        assert self.dp_degree * self.pp_degree * self.mp_degree >= 1, \
            "Invalid parallel training configuration"
            
        # ByteScale requires specific vocab size
        if self.use_bytescale:
            assert self.vocab_size == 50304, \
                "ByteScale requires vocab_size of 50304"

        # Validate position embedding configuration
        valid_pos_emb_types = ["rotary", "absolute", "relative"]
        assert self.position_embedding_type in valid_pos_emb_types, \
            f"Position embedding type must be one of {valid_pos_emb_types}"

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Basic training params
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_steps: Optional[int] = None
    
    # Optimizer settings
    optimizer_type: OptimizationType = OptimizationType.MUON
    muon_config: Optional[MuonConfig] = None
    
    # Learning rate schedule
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 2000
    warmup_ratio: float = 0.01
    lr_scheduler_type: str = "cosine"
    
    # Mixed precision training
    fp16: bool = True
    bf16: bool = False
    mixed_precision_dtype: str = "float16"
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    
    # Regularization
    weight_decay: float = 0.01
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Distributed training
    local_rank: int = -1
    dataloader_num_workers: int = 4
    
    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: Optional[int] = 5
    
    # Logging
    logging_strategy: str = "steps"
    logging_steps: int = 100
    log_level: str = "info"
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    def __post_init__(self):
        """Initialize configurations and validate settings"""
        # Set Muon optimizer config if not provided
        if self.optimizer_type == OptimizationType.MUON and not self.muon_config:
            self.muon_config = MuonConfig()
            
        # Validate mixed precision settings
        assert not (self.fp16 and self.bf16), \
            "Cannot use both FP16 and BF16 mixed precision"
            
        # Validate learning rate schedule
        assert 0.0 <= self.warmup_ratio <= 1.0, \
            "Warmup ratio must be between 0 and 1"
        assert self.min_learning_rate <= self.learning_rate, \
            "Minimum learning rate must be <= maximum learning rate"
            
        # Validate parallel processing
        if self.dataloader_num_workers > 0:
            assert self.dataloader_num_workers <= 16, \
                "Number of dataloader workers should not exceed 16"