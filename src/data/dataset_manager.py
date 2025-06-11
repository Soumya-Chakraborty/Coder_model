import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np
from transformers import PreTrainedTokenizer

@dataclass
class DatasetConfig:
    # GitHub Code dataset
    github_code_path: str
    github_code_languages: List[str] = None
    github_max_files: int = 1000000
    
    # ConDefects dataset
    condefects_path: str
    condefects_languages: List[str] = None
    
    # LeetCode dataset
    leetcode_path: str
    leetcode_difficulty: List[str] = None  # ["easy", "medium", "hard"]
    
    # Case2Code dataset
    case2code_path: str
    
    # Evaluation datasets
    humaneval_path: Optional[str] = None
    mbpp_path: Optional[str] = None
    swe_bench_path: Optional[str] = None
    research_codebench_path: Optional[str] = None
    aime_path: Optional[str] = None
    livecode_path: Optional[str] = None
    
    # General settings
    max_sequence_length: int = 2048
    train_val_split: float = 0.95
    seed: int = 42

class CodeDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        is_training: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Process code and comments/documentation
        code = item.get("code", "")
        comments = item.get("comments", "")
        prompt = item.get("prompt", "")
        
        # Combine text elements
        if prompt:
            combined_text = f"{prompt}\n{code}"
        else:
            combined_text = f"{comments}\n{code}" if comments else code
        
        # Tokenize
        encodings = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert to tensors
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Add labels for training
        if self.is_training:
            labels = input_ids.clone()
            # Mask padding tokens
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

class GithubCodeDataset(CodeDataset):
    @staticmethod
    def load_data(
        data_path: str,
        languages: Optional[List[str]] = None,
        max_files: int = 1000000
    ) -> List[Dict]:
        data = []
        for root, _, files in os.walk(data_path):
            if len(data) >= max_files:
                break
                
            for file in files:
                if languages:
                    ext = Path(file).suffix.lstrip(".")
                    if ext not in languages:
                        continue
                
                try:
                    with open(os.path.join(root, file), "r") as f:
                        content = f.read()
                    data.append({"code": content})
                except Exception as e:
                    logging.warning(f"Error loading file {file}: {e}")
        
        return data[:max_files]

class ConDefectsDataset(CodeDataset):
    @staticmethod
    def load_data(
        data_path: str,
        languages: Optional[List[str]] = None
    ) -> List[Dict]:
        data = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if not file.endswith(".json"):
                    continue
                    
                try:
                    with open(os.path.join(root, file), "r") as f:
                        item = json.load(f)
                    
                    if languages and item.get("language") not in languages:
                        continue
                        
                    data.append({
                        "code": item["code"],
                        "defect": item.get("defect"),
                        "fix": item.get("fix")
                    })
                except Exception as e:
                    logging.warning(f"Error loading file {file}: {e}")
        
        return data

class LeetCodeDataset(CodeDataset):
    @staticmethod
    def load_data(
        data_path: str,
        difficulty: Optional[List[str]] = None
    ) -> List[Dict]:
        data = []
        try:
            with open(data_path, "r") as f:
                problems = json.load(f)
            
            for problem in problems:
                if difficulty and problem["difficulty"].lower() not in difficulty:
                    continue
                    
                data.append({
                    "prompt": problem["description"],
                    "code": problem.get("solution", ""),
                    "test_cases": problem.get("test_cases", [])
                })
        except Exception as e:
            logging.error(f"Error loading LeetCode dataset: {e}")
        
        return data

class Case2CodeDataset(CodeDataset):
    @staticmethod
    def load_data(data_path: str) -> List[Dict]:
        data = []
        try:
            with open(data_path, "r") as f:
                cases = json.load(f)
            
            for case in cases:
                data.append({
                    "prompt": case["description"],
                    "code": case["implementation"],
                    "test_cases": case.get("tests", [])
                })
        except Exception as e:
            logging.error(f"Error loading Case2Code dataset: {e}")
        
        return data

class DatasetManager:
    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: PreTrainedTokenizer
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.rng = np.random.RandomState(config.seed)
        
    def load_training_data(self) -> Tuple[DataLoader, DataLoader]:
        datasets = []
        
        # Load GitHub Code data
        github_data = GithubCodeDataset.load_data(
            self.config.github_code_path,
            self.config.github_code_languages,
            self.config.github_max_files
        )
        datasets.append(
            CodeDataset(
                github_data,
                self.tokenizer,
                self.config.max_sequence_length
            )
        )
        
        # Load ConDefects data
        condefects_data = ConDefectsDataset.load_data(
            self.config.condefects_path,
            self.config.condefects_languages
        )
        datasets.append(
            CodeDataset(
                condefects_data,
                self.tokenizer,
                self.config.max_sequence_length
            )
        )
        
        # Load LeetCode data
        leetcode_data = LeetCodeDataset.load_data(
            self.config.leetcode_path,
            self.config.leetcode_difficulty
        )
        datasets.append(
            CodeDataset(
                leetcode_data,
                self.tokenizer,
                self.config.max_sequence_length
            )
        )
        
        # Load Case2Code data
        case2code_data = Case2CodeDataset.load_data(
            self.config.case2code_path
        )
        datasets.append(
            CodeDataset(
                case2code_data,
                self.tokenizer,
                self.config.max_sequence_length
            )
        )
        
        # Combine datasets
        combined_dataset = ConcatDataset(datasets)
        
        # Split into train and validation
        train_size = int(len(combined_dataset) * self.config.train_val_split)
        val_size = len(combined_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            combined_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def load_evaluation_dataset(
        self,
        dataset_path: str,
        dataset_type: str
    ) -> DataLoader:
        """Load evaluation datasets like HumanEval, MBPP, etc."""
        try:
            with open(dataset_path, "r") as f:
                eval_data = json.load(f)
            
            dataset = CodeDataset(
                eval_data,
                self.tokenizer,
                self.config.max_sequence_length,
                is_training=False
            )
            
            return DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
        except Exception as e:
            logging.error(f"Error loading {dataset_type} dataset: {e}")
            return None
    
    def get_evaluation_dataloaders(self) -> Dict[str, DataLoader]:
        eval_loaders = {}
        
        # Load HumanEval
        if self.config.humaneval_path:
            eval_loaders["humaneval"] = self.load_evaluation_dataset(
                self.config.humaneval_path,
                "HumanEval"
            )
        
        # Load MBPP
        if self.config.mbpp_path:
            eval_loaders["mbpp"] = self.load_evaluation_dataset(
                self.config.mbpp_path,
                "MBPP"
            )
        
        # Load SWE-Bench
        if self.config.swe_bench_path:
            eval_loaders["swe_bench"] = self.load_evaluation_dataset(
                self.config.swe_bench_path,
                "SWE-Bench"
            )
        
        # Load ResearchCodeBench
        if self.config.research_codebench_path:
            eval_loaders["research_codebench"] = self.load_evaluation_dataset(
                self.config.research_codebench_path,
                "ResearchCodeBench"
            )
        
        # Load AIME
        if self.config.aime_path:
            eval_loaders["aime"] = self.load_evaluation_dataset(
                self.config.aime_path,
                "AIME"
            )
        
        # Load LiveCode
        if self.config.livecode_path:
            eval_loaders["livecode"] = self.load_evaluation_dataset(
                self.config.livecode_path,
                "LiveCode"
            )
        
        return eval_loaders