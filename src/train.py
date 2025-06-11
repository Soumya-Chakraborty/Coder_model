import os
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
from transformers import PreTrainedTokenizer, AutoTokenizer

from model.transformer import CodingTransformer
from model.config import ModelConfig, TrainingConfig, OptimizationType, MuonConfig
from data.dataset_manager import DatasetManager, DatasetConfig
from training.rlhf import SafeRLHFTrainer, RLHFConfig, ConstitutionalChecker

def setup_logging(config: TrainingConfig):
    """Setup logging configuration"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=config.log_level.upper(),
    )

def setup_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizer:
    """Initialize ByteScale tokenizer"""
    if model_config.use_bytescale:
        # Initialize ByteScale tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",  # Using Phi-2's tokenizer as base
            trust_remote_code=True
        )
        # Extend vocabulary for ByteScale
        tokenizer.add_special_tokens({
            "pad_token": "[PAD]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
        })
        return tokenizer
    else:
        raise ValueError("Only ByteScale tokenizer is supported")

def setup_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig
) -> torch.optim.Optimizer:
    """Initialize Muon optimizer or fallback to AdamW"""
    if config.optimizer_type == OptimizationType.MUON:
        try:
            from muon_optimizer import MuonOptimizer
            return MuonOptimizer(
                model.parameters(),
                lr=config.muon_config.learning_rate,
                beta1=config.muon_config.beta1,
                beta2=config.muon_config.beta2,
                eps=config.muon_config.eps,
                weight_decay=config.muon_config.weight_decay,
                momentum_decay=config.muon_config.momentum_decay,
                variance_reduction=config.muon_config.variance_reduction,
                gradient_centralization=config.muon_config.gradient_centralization,
                adaptive_momentum=config.muon_config.adaptive_momentum,
                warm_restart_cycles=config.muon_config.warm_restart_cycles
            )
        except ImportError:
            logging.warning("Muon optimizer not available, falling back to AdamW")
            config.optimizer_type = OptimizationType.ADAMW

    if config.optimizer_type == OptimizationType.ADAMW:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    num_training_steps: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Setup learning rate scheduler"""
    if config.lr_scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=config.min_learning_rate
        )
    elif config.lr_scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.min_learning_rate / config.learning_rate,
            total_iters=num_training_steps
        )
    return None

def train(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_config: DatasetConfig,
    rlhf_config: RLHFConfig,
    output_dir: str
):
    """Main training function"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = setup_tokenizer(model_config)
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(dataset_config, tokenizer)
    train_loader, val_loader = dataset_manager.load_training_data()
    eval_loaders = dataset_manager.get_evaluation_dataloaders()
    
    # Initialize model
    model = CodingTransformer(model_config).to(device)
    
    # Setup parallel training if enabled
    if model_config.use_hybrid_data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[training_config.local_rank],
            output_device=training_config.local_rank
        )
    
    # Initialize optimizer and scheduler
    optimizer = setup_optimizer(model, training_config)
    num_training_steps = (
        len(train_loader) * training_config.num_train_epochs //
        training_config.gradient_accumulation_steps
    )
    scheduler = setup_scheduler(optimizer, training_config, num_training_steps)
    
    # Initialize RLHF components if enabled
    if model_config.use_rlhf:
        # Create reference model (frozen copy of initial model)
        ref_model = CodingTransformer(model_config).to(device)
        ref_model.load_state_dict(model.state_dict())
        for param in ref_model.parameters():
            param.requires_grad = False
            
        # Initialize constitutional checker
        const_checker = ConstitutionalChecker(tokenizer)
        
        # Initialize RLHF trainer
        rlhf_trainer = SafeRLHFTrainer(
            policy_model=model,
            ref_model=ref_model,
            reward_model=None,  # Will be initialized during training
            tokenizer=tokenizer,
            config=rlhf_config,
            constitution_checker=const_checker
        )
    
    # Training loop
    global_step = 0
    best_eval_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(training_config.num_train_epochs):
        model.train()
        epoch_loss = 0
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / training_config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            epoch_loss += loss.item()
            
            # Update weights if gradient accumulation is complete
            if (step + 1) % training_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training_config.max_grad_norm
                )
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log progress
                if global_step % training_config.logging_steps == 0:
                    logging.info(
                        f"Epoch {epoch+1}/{training_config.num_train_epochs} "
                        f"Step {global_step}/{num_training_steps} "
                        f"Loss {loss.item():.4f}"
                    )
                
                # Evaluate if needed
                if (global_step % training_config.eval_steps == 0 and
                    training_config.evaluation_strategy == "steps"):
                    eval_loss = evaluate(model, val_loader, device)
                    model.train()
                    
                    # Early stopping check
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        early_stopping_counter = 0
                        
                        # Save best model
                        if training_config.save_strategy == "steps":
                            save_model(
                                model,
                                tokenizer,
                                os.path.join(output_dir, "best_model")
                            )
                    else:
                        early_stopping_counter += 1
                        if (early_stopping_counter >=
                            training_config.early_stopping_patience):
                            logging.info("Early stopping triggered")
                            return
                
                # Save checkpoint if needed
                if (global_step % training_config.save_steps == 0 and
                    training_config.save_strategy == "steps"):
                    save_model(
                        model,
                        tokenizer,
                        os.path.join(output_dir, f"checkpoint-{global_step}")
                    )
        
        # Epoch-end evaluation
        if training_config.evaluation_strategy == "epoch":
            eval_loss = evaluate(model, val_loader, device)
            model.train()
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_model(
                    model,
                    tokenizer,
                    os.path.join(output_dir, "best_model")
                )
    
    # Final evaluation on test sets
    model.eval()
    test_results = {}
    for name, loader in eval_loaders.items():
        if loader is not None:
            test_results[name] = evaluate(model, loader, device)
    
    logging.info("Final evaluation results:")
    for name, score in test_results.items():
        logging.info(f"{name}: {score:.4f}")
    
    # Save final model
    save_model(model, tokenizer, os.path.join(output_dir, "final_model"))

def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """Evaluate model on given dataloader"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)

def save_model(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    output_dir: str
):
    """Save model and tokenizer"""
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save config
    model.config.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--training_config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--rlhf_config", type=str, required=True)
    args = parser.parse_args()
    
    # Load configurations
    model_config = ModelConfig.from_json_file(args.model_config)
    training_config = TrainingConfig.from_json_file(args.training_config)
    dataset_config = DatasetConfig.from_json_file(args.dataset_config)
    rlhf_config = RLHFConfig.from_json_file(args.rlhf_config)
    
    # Setup logging
    setup_logging(training_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(
        model_config,
        training_config,
        dataset_config,
        rlhf_config,
        args.output_dir
    )

if __name__ == "__main__":
    main()