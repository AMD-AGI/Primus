###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
SFT Dataset for Megatron-LM based supervised fine-tuning.

This module provides a universal dataset interface for SFT training that:
1. Supports HuggingFace datasets as data source
2. Supports local JSONL files for offline training
3. Handles various conversation formats (extensible)
4. Implements proper loss masking for instruction tuning
5. Follows Megatron-LM's dataset provider pattern
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Import logging utility
try:
    from primus.modules.module_utils import log_rank_0
except ImportError:
    # Fallback for testing without full Primus installation
    def log_rank_0(msg):
        print(msg)


def load_jsonl_file(file_path: str) -> List[Dict]:
    """
    Load data from a JSONL (JSON Lines) file.
    
    Each line in the file should be a valid JSON object.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If a line is not valid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON on line {line_num} in {file_path}: {e.msg}",
                        e.doc,
                        e.pos
                    )
    
    log_rank_0(f"Loaded {len(data)} samples from {file_path}")
    return data


class ConversationFormatter:
    """
    Base class for conversation formatting.
    
    Different conversation formats (ChatML, Alpaca, ShareGPT, etc.) can be 
    implemented by subclassing this and providing format-specific logic.
    """
    
    def format_conversation(
        self, 
        instruction: str, 
        response: str, 
        input_text: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Format a conversation into a training sample.
        
        Args:
            instruction: The instruction/question text
            response: The response/answer text
            input_text: Optional additional input context
            system_prompt: Optional system prompt
            
        Returns:
            Tuple of (formatted_text, instruction_length)
            instruction_length is used to determine where to mask loss
        """
        raise NotImplementedError
    
    def get_special_tokens(self) -> Dict[str, str]:
        """Return special tokens used in this format."""
        return {}


class AlpacaFormatter(ConversationFormatter):
    """
    Alpaca conversation format.
    
    Format:
        Below is an instruction that describes a task. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Response:
        {response}
    """
    
    def format_conversation(
        self, 
        instruction: str, 
        response: str, 
        input_text: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, int]:
        """Format conversation in Alpaca style."""
        prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        
        if system_prompt:
            prompt_template = system_prompt + "\n\n"
            
        instruction_part = f"### Instruction:\n{instruction}\n\n"
        
        if input_text:
            instruction_part += f"### Input:\n{input_text}\n\n"
            
        instruction_part += "### Response:\n"
        
        full_instruction = prompt_template + instruction_part
        full_text = full_instruction + response
        
        return full_text, len(full_instruction)


class ChatMLFormatter(ConversationFormatter):
    """
    ChatML conversation format.
    
    Format:
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {instruction}<|im_end|>
        <|im_start|>assistant
        {response}<|im_end|>
    """
    
    def format_conversation(
        self, 
        instruction: str, 
        response: str, 
        input_text: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, int]:
        """Format conversation in ChatML style."""
        parts = []
        
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
            
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
            
        parts.append(f"<|im_start|>user\n{user_content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        
        instruction_text = "".join(parts)
        full_text = instruction_text + response + "<|im_end|>"
        
        return full_text, len(instruction_text)
    
    def get_special_tokens(self) -> Dict[str, str]:
        """Return ChatML special tokens."""
        return {
            "im_start": "<|im_start|>",
            "im_end": "<|im_end|>"
        }


class SFTDataset(Dataset):
    """
    Universal SFT dataset that supports HuggingFace datasets and local JSONL files.
    
    This dataset:
    1. Loads data from HuggingFace datasets OR local JSONL files
    2. Formats conversations using specified formatter
    3. Tokenizes text
    4. Creates loss masks (only compute loss on response tokens)
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_seq_length: int,
        split: str = "train",
        formatter: str = "alpaca",
        seed: int = 1234,
        **kwargs
    ):
        """
        Initialize SFT dataset.
        
        Args:
            dataset_name: HuggingFace dataset name, local file path, or JSONL file path.
                         Examples:
                         - "tatsu-lab/alpaca" (HuggingFace Hub)
                         - "/path/to/data.jsonl" (local JSONL file)
                         - "/path/to/data.json" (local JSON file)
            tokenizer: Megatron tokenizer instance
            max_seq_length: Maximum sequence length
            split: Dataset split (train/validation/test) - only used for HuggingFace datasets
            formatter: Conversation format type ("alpaca", "chatml")
            seed: Random seed for dataset shuffling
            **kwargs: Additional arguments passed to load_dataset (for HuggingFace only)
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.seed = seed
        
        # Initialize formatter
        if formatter == "alpaca":
            self.formatter = AlpacaFormatter()
        elif formatter == "chatml":
            self.formatter = ChatMLFormatter()
        else:
            raise ValueError(f"Unknown formatter: {formatter}. Supported: alpaca, chatml")
        
        # Determine if this is a local file or HuggingFace dataset
        is_local_file = (
            dataset_name.endswith('.jsonl') or 
            dataset_name.endswith('.json') or 
            os.path.isfile(dataset_name)
        )
        
        if is_local_file:
            # Load from local JSONL/JSON file
            log_rank_0(f"Loading dataset from local file: {dataset_name}")
            
            # Load data from file
            if dataset_name.endswith('.jsonl'):
                data = load_jsonl_file(dataset_name)
            elif dataset_name.endswith('.json'):
                # Also support single JSON file with array of objects
                with open(dataset_name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"JSON file must contain a list of objects, got {type(data)}")
                log_rank_0(f"Loaded {len(data)} samples from {dataset_name}")
            else:
                # Try to detect format by reading the file
                try:
                    data = load_jsonl_file(dataset_name)
                except json.JSONDecodeError:
                    # If JSONL fails, try as single JSON
                    with open(dataset_name, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError(f"File must contain a list of objects or be JSONL format")
                    log_rank_0(f"Loaded {len(data)} samples from {dataset_name}")
            
            # Convert to HuggingFace Dataset format for compatibility
            try:
                from datasets import Dataset as HFDataset
            except ImportError:
                raise ImportError(
                    "HuggingFace datasets library is required. "
                    "Install with: pip install datasets"
                )
            
            self.dataset = HFDataset.from_list(data)
            log_rank_0(f"Created dataset with {len(self.dataset)} samples")
            
        else:
            # Load dataset from HuggingFace Hub
            log_rank_0(f"Loading dataset from HuggingFace Hub: {dataset_name}, split: {split}")
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "HuggingFace datasets library is required. "
                    "Install with: pip install datasets"
                )
            
            self.dataset = load_dataset(dataset_name, split=split, **kwargs)
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)
    
    def _tokenize_and_mask(
        self, 
        text: str, 
        instruction_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize text and create loss mask.
        
        Args:
            text: Full conversation text
            instruction_length: Character length of instruction part
            
        Returns:
            Tuple of (input_ids, labels, loss_mask)
            
        Note:
            This function tokenizes the instruction and response separately to determine
            the loss mask boundary. Some tokenizers (like BPE) may produce slightly different
            results when tokenizing substrings vs full text due to boundary effects.
            In practice, this approach works well for most tokenizers, but if you encounter
            issues with incorrect masking, consider tokenizing instruction and response
            separately and concatenating the token IDs directly.
        """
        # Tokenize full text
        try:
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        except AttributeError as e:
            raise TypeError(
                f"Tokenizer must have 'tokenize' and 'convert_tokens_to_ids' methods. "
                f"Got tokenizer of type: {type(self.tokenizer)}. Error: {e}"
            )
        
        # Tokenize instruction part to find where to start computing loss
        instruction_text = text[:instruction_length]
        instruction_tokens = self.tokenizer.tokenize(instruction_text)
        instruction_token_ids = self.tokenizer.convert_tokens_to_ids(instruction_tokens)
        instruction_len = len(instruction_token_ids)
        
        # Truncate if needed
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        
        # Create loss mask (0 for instruction, 1 for response)
        # Note: Sequences shorter than max_seq_length are not padded here.
        # The training loop should handle variable-length sequences or apply padding as needed.
        loss_mask = np.zeros(len(token_ids), dtype=np.int64)
        if instruction_len < len(token_ids):
            loss_mask[instruction_len:] = 1
        
        # Convert to tensors
        input_ids = torch.tensor(token_ids, dtype=torch.int64)
        labels = input_ids.clone()
        loss_mask = torch.tensor(loss_mask, dtype=torch.int64)
        
        return input_ids, labels, loss_mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            Dictionary containing:
                - input_ids: Token IDs for input
                - labels: Token IDs for labels (same as input_ids for causal LM)
                - loss_mask: Binary mask (1 where loss should be computed)
        """
        sample = self.dataset[idx]
        
        # Extract fields (assuming common field names)
        # Support multiple common field name conventions
        # Use explicit None checks to handle empty strings correctly
        instruction = sample.get("instruction")
        if instruction is None:
            instruction = sample.get("prompt")
        if instruction is None:
            instruction = sample.get("question", "")
            
        response = sample.get("response")
        if response is None:
            response = sample.get("output")
        if response is None:
            response = sample.get("answer", "")
            
        input_text = sample.get("input", None)
        system_prompt = sample.get("system", None)
        
        # Format conversation
        full_text, instruction_length = self.formatter.format_conversation(
            instruction=instruction,
            response=response,
            input_text=input_text,
            system_prompt=system_prompt
        )
        
        # Tokenize and create masks
        input_ids, labels, loss_mask = self._tokenize_and_mask(full_text, instruction_length)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }


def build_train_valid_test_datasets(
    dataset_name: str,
    tokenizer,
    max_seq_length: int,
    train_val_test_num_samples: List[int],
    formatter: str = "alpaca",
    seed: int = 1234,
    **kwargs
) -> Tuple[Optional[SFTDataset], Optional[SFTDataset], Optional[SFTDataset]]:
    """
    Build train, validation, and test datasets for SFT.
    
    This follows Megatron-LM's dataset provider pattern.
    
    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Megatron tokenizer
        max_seq_length: Maximum sequence length
        train_val_test_num_samples: List of [train_samples, val_samples, test_samples]
        formatter: Conversation format type
        seed: Random seed
        **kwargs: Additional arguments for load_dataset
        
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
        Any can be None if num_samples is 0
    """
    train_samples, valid_samples, test_samples = train_val_test_num_samples
    
    train_ds = None
    valid_ds = None
    test_ds = None
    
    if train_samples > 0:
        train_ds = SFTDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split="train",
            formatter=formatter,
            seed=seed,
            **kwargs
        )
    
    if valid_samples > 0:
        try:
            valid_ds = SFTDataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                split="validation",
                formatter=formatter,
                seed=seed,
                **kwargs
            )
        except (ValueError, KeyError) as e:
            # Some datasets don't have validation split or have different split names
            log_rank_0(f"Validation split not available: {e}")
            valid_ds = None
    
    if test_samples > 0:
        try:
            test_ds = SFTDataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                split="test",
                formatter=formatter,
                seed=seed,
                **kwargs
            )
        except (ValueError, KeyError) as e:
            # Some datasets don't have test split or have different split names
            log_rank_0(f"Test split not available: {e}")
            test_ds = None
    
    return train_ds, valid_ds, test_ds
