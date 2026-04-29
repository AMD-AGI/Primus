# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""
GPT SFT Chat Dataset - Adapted from Megatron-Bridge

This module provides a complete SFT dataset implementation adapted from Megatron-Bridge,
but implemented entirely within Primus without importing Megatron-Bridge code.
"""

import logging
import re
from typing import Any, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Regex to detect if chat template has generation prompt support
GENERATION_REGEX = re.compile(r"{%\s*if\s+add_generation_prompt\s*%}")

# Default Llama3 chat template (used when tokenizer doesn't have one)
LLAMA3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] | trim }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] | trim }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% generation %}"
    "{{ message['content'] | trim }}"
    "{% endgeneration %}"
    "<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)


def _convert_to_openai_messages(source: dict) -> list[dict]:
    """
    Convert input to OpenAI messages format.

    Supports both ShareGPT and OpenAI formats.
    """
    if isinstance(source, dict):
        if source.get("conversations"):
            # ShareGPT format: {"conversations": [{"from": "User/Assistant", "value": ""}]}
            # Convert to OpenAI format
            chat = [
                {"role": convo["from"].lower(), "content": convo["value"]}
                for convo in source["conversations"]
            ]
            if source.get("system"):
                chat.insert(0, {"role": "system", "content": source["system"]})
        elif source.get("messages"):
            # Already in OpenAI format
            chat = source.get("messages")
        else:
            raise ValueError(f"Unknown source format: {source}")
    else:
        chat = source

    return chat


def _chat_preprocess(source: dict, tokenizer, tool_schemas: Optional[list[Any]] = None) -> dict:
    """
    Preprocess messages to apply chat template and tokenize.

    This is the core function adapted from Megatron-Bridge's implementation.

    Args:
        source: Input data in ShareGPT or OpenAI format
        tokenizer: Megatron tokenizer with HuggingFace tokenizer backend
        tool_schemas: Optional tool schemas for function calling

    Returns:
        Dictionary with input_ids, loss_mask, context_ids, answer_ids
    """
    # Validate tokenizer
    if not hasattr(tokenizer, "_tokenizer") or not hasattr(tokenizer._tokenizer, "apply_chat_template"):
        raise ValueError(
            "Cannot apply chat template with tokenizer that is not a HuggingFace AutoTokenizer. "
            "The tokenizer must have a '_tokenizer' attribute with an 'apply_chat_template' method."
        )

    # Convert to OpenAI format
    chat = _convert_to_openai_messages(source)

    # Extract tools if present
    tools = None
    if isinstance(source, dict):
        tools = source.get("tools") or tool_schemas
    else:
        tools = tool_schemas

    # Get the underlying HuggingFace tokenizer
    hf_tokenizer = tokenizer._tokenizer

    # Check if template supports generation prompt
    chat_template = getattr(hf_tokenizer, "chat_template", None)
    if not chat_template:
        # Use default Llama3 template if none is set
        chat_template = LLAMA3_CHAT_TEMPLATE
        logger.info("No chat template found, using default Llama3 template")

    template_has_generation_kwd = GENERATION_REGEX.search(chat_template) is not None

    # Tokenize the chat - always pass chat_template explicitly
    try:
        tokenized_chat = hf_tokenizer.apply_chat_template(
            chat,
            tools=tools if tools else None,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=template_has_generation_kwd,
            chat_template=chat_template,  # Always pass template explicitly
        )
    except Exception as e:
        # Fallback if return_dict or return_assistant_tokens_mask not supported
        logger.warning(
            f"Error applying chat template with return_dict: {e}. Falling back to simple tokenization."
        )
        input_ids = hf_tokenizer.apply_chat_template(
            chat,
            tools=tools if tools else None,
            tokenize=True,
            chat_template=chat_template,  # Always pass template explicitly
        )
        tokenized_chat = {"input_ids": input_ids}
        template_has_generation_kwd = False

    # Extract input_ids and mask
    input_ids = tokenized_chat.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    if template_has_generation_kwd and "assistant_masks" in tokenized_chat:
        mask = tokenized_chat["assistant_masks"]
        if isinstance(mask, torch.Tensor):
            mask = mask.tolist()
    else:
        # No mask available, assume all tokens are for training
        mask = [1] * len(input_ids)

    # Add EOS token if not present
    eos_id = getattr(tokenizer, "eod", None) or getattr(hf_tokenizer, "eos_token_id", None)
    if eos_id and input_ids[-1] != eos_id:
        input_ids += [eos_id]
        mask += [1]

    # Find context end (last masked token position)
    if 0 in mask:
        # Traverse backward to find first masked token
        context_end_idx = len(mask) - mask[::-1].index(0)
    else:
        context_end_idx = len(mask)

    # Split into context and answer
    context_ids = input_ids[:context_end_idx]
    answer_ids = input_ids[context_end_idx:]

    return dict(
        input_ids=torch.LongTensor(input_ids),
        loss_mask=torch.BoolTensor(mask),
        context_ids=torch.LongTensor(context_ids),
        answer_ids=torch.LongTensor(answer_ids),
    )


class _JSONLMemMapDataset:
    """Simple memory-mapped JSONL dataset loader."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class GPTSFTChatDataset(Dataset):
    """
    SFT Chat Dataset - Adapted from Megatron-Bridge

    This dataset handles conversational data in ShareGPT or OpenAI formats,
    applies chat templates, and prepares data for SFT training.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_seq_length: int = 4096,
        pad_to_max_length: bool = False,
        tool_schemas: Optional[list[Any]] = None,
        max_num_samples: Optional[int] = None,
        seed: int = 1234,
        get_attention_mask_from_fusion: bool = True,
    ):
        """
        Initialize GPTSFTChatDataset.

        Args:
            file_path: Path to JSONL file with conversations
            tokenizer: Megatron tokenizer instance
            max_seq_length: Maximum sequence length
            pad_to_max_length: Whether to pad to max length
            tool_schemas: Optional tool schemas for function calling
            max_num_samples: Maximum number of samples to use
            seed: Random seed
            get_attention_mask_from_fusion: Whether to let attention kernel handle masking
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length
        self.tool_schemas = tool_schemas
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.get_attention_mask_from_fusion = get_attention_mask_from_fusion

        # Load dataset
        logger.info(f"Loading SFT dataset from {file_path}")
        self.indexed_dataset = _JSONLMemMapDataset(file_path)

        # Build samples mapping if max_num_samples is specified
        if self.max_num_samples is not None:
            np.random.seed(self.seed)
            dataset_len = len(self.indexed_dataset)
            if self.max_num_samples > dataset_len:
                # Oversample
                max_num_epochs = np.ceil(self.max_num_samples / dataset_len)
                indices = np.arange(dataset_len)[None, :].repeat(int(max_num_epochs), axis=0)
                for epoch_indices in indices:
                    np.random.shuffle(epoch_indices)
                self.samples_mapping = indices.reshape(-1)[: self.max_num_samples]
            else:
                # Subsample
                self.samples_mapping = np.random.choice(dataset_len, self.max_num_samples, replace=False)
        else:
            self.samples_mapping = None

        # Validate and report dataset statistics
        self.validate_and_report()

    def __len__(self):
        if self.samples_mapping is not None:
            return len(self.samples_mapping)
        return len(self.indexed_dataset)

    def validate_and_report(self):
        """
        Validate dataset and report statistics.
        
        This method samples up to 100 examples from the dataset to compute
        and report useful statistics about sequence lengths, loss ratios,
        and data quality.
        """
        try:
            total_samples = len(self)
            
            if total_samples == 0:
                logger.warning("Dataset is empty!")
                return
            
            # Sample up to 100 examples for statistics
            sample_size = min(100, total_samples)
            sample_indices = np.random.choice(total_samples, sample_size, replace=False)
            
            seq_lengths = []
            loss_ratios = []
            context_lengths = []
            answer_lengths = []
            errors = []
            
            logger.info(f"Validating dataset with {sample_size} samples...")
            
            for idx in sample_indices:
                try:
                    sample = self[int(idx)]
                    
                    # Collect statistics
                    tokens_len = len(sample['tokens'])
                    seq_lengths.append(tokens_len)
                    
                    # Calculate loss ratio (percentage of tokens that contribute to loss)
                    loss_mask = sample['loss_mask']
                    if tokens_len > 0:
                        loss_ratio = loss_mask.sum().item() / tokens_len
                        loss_ratios.append(loss_ratio)
                    
                    # Try to get context and answer lengths if available
                    if 'position_ids' in sample:
                        context_lengths.append(len(sample['position_ids']))
                    
                except Exception as e:
                    errors.append(f"Sample {idx}: {str(e)}")
            
            # Report statistics
            logger.info("=" * 80)
            logger.info("Dataset Validation Report")
            logger.info("=" * 80)
            logger.info(f"Total samples: {total_samples:,}")
            logger.info(f"Samples validated: {len(seq_lengths)}/{sample_size}")
            
            if seq_lengths:
                logger.info(f"\nSequence Length Statistics:")
                logger.info(f"  Average: {np.mean(seq_lengths):.1f} tokens")
                logger.info(f"  Median: {np.median(seq_lengths):.1f} tokens")
                logger.info(f"  Min: {np.min(seq_lengths)} tokens")
                logger.info(f"  Max: {np.max(seq_lengths)} tokens")
                logger.info(f"  Std Dev: {np.std(seq_lengths):.1f} tokens")
                logger.info(f"  Max allowed: {self.max_seq_length} tokens")
                
                # Check truncation rate
                truncated = sum(1 for l in seq_lengths if l >= self.max_seq_length)
                if truncated > 0:
                    logger.warning(
                        f"  {truncated}/{len(seq_lengths)} samples ({truncated/len(seq_lengths)*100:.1f}%) "
                        f"will be truncated!"
                    )
            
            if loss_ratios:
                logger.info(f"\nLoss Mask Statistics:")
                logger.info(f"  Average loss ratio: {np.mean(loss_ratios):.2%}")
                logger.info(f"  Median loss ratio: {np.median(loss_ratios):.2%}")
                logger.info(f"  Min loss ratio: {np.min(loss_ratios):.2%}")
                logger.info(f"  Max loss ratio: {np.max(loss_ratios):.2%}")
                
                # Warn about potential issues
                avg_loss_ratio = np.mean(loss_ratios)
                if avg_loss_ratio < 0.1:
                    logger.warning(
                        f"  Average loss ratio is very low ({avg_loss_ratio:.2%}). "
                        "This means most tokens are masked and won't contribute to training!"
                    )
                elif avg_loss_ratio > 0.9:
                    logger.warning(
                        f"  Average loss ratio is very high ({avg_loss_ratio:.2%}). "
                        "Consider if instruction tokens should be masked."
                    )
            
            if errors:
                logger.warning(f"\nEncountered {len(errors)} errors during validation:")
                for error in errors[:5]:  # Show first 5 errors
                    logger.warning(f"  {error}")
                if len(errors) > 5:
                    logger.warning(f"  ... and {len(errors) - 5} more errors")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error during dataset validation: {e}")
            logger.error("Continuing without validation...")

    def __getitem__(self, idx):
        # Map index if needed
        if self.samples_mapping is not None:
            idx = self.samples_mapping[idx]
            if isinstance(idx, (np.int32, np.int64)):
                idx = idx.item()

        # Get example
        example = self.indexed_dataset[idx]

        # Preprocess using chat template
        try:
            result = _chat_preprocess(example, self.tokenizer, self.tool_schemas)
        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            logger.error(f"Example: {example}")
            raise

        # Extract fields
        input_ids = result["input_ids"]  # Full sequence
        loss_mask = result["loss_mask"]  # Boolean mask

        # Create tokens and labels for next-token prediction
        # tokens[i] predicts labels[i]
        tokens = input_ids[:-1]  # Remove last token
        labels = input_ids[1:]  # Shift left by 1
        loss_mask = loss_mask[1:].float()  # Shift mask accordingly

        # Get sequence length
        seq_len = len(tokens)

        # Pad or truncate to max_seq_length
        if seq_len > self.max_seq_length:
            tokens = tokens[: self.max_seq_length]
            labels = labels[: self.max_seq_length]
            loss_mask = loss_mask[: self.max_seq_length]
            seq_len = self.max_seq_length

        if self.pad_to_max_length and seq_len < self.max_seq_length:
            pad_len = self.max_seq_length - seq_len
            pad_id = getattr(self.tokenizer, "eod", 0)
            tokens = torch.cat([tokens, torch.full((pad_len,), pad_id, dtype=tokens.dtype)])
            labels = torch.cat([labels, torch.full((pad_len,), pad_id, dtype=labels.dtype)])
            loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype)])

        # Create position ids (simple sequential)
        position_ids = torch.arange(len(tokens), dtype=torch.long)

        # Return in Megatron format
        return {
            "tokens": tokens,  # Input tokens
            "labels": labels,  # Target labels (shifted)
            "loss_mask": loss_mask,  # Mask for loss computation
            "position_ids": position_ids,  # Position IDs
        }

    def _create_attention_mask(self, max_length):
        """Creates an upper-triangular causal attention mask."""
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask

    def _collate_item(self, item, max_length, pad_id):
        """Collate and pad items to max length."""
        if isinstance(item[0], torch.Tensor):
            item = [x.tolist() for x in item]
        return [x + [pad_id] * (max_length - len(x)) for x in item]

    def collate_fn(self, batch):
        """
        Collate a batch of samples.

        This function prepares batched tensors for model input.
        """
        # Extract fields
        input_ids = [item["input_ids"][:-1].tolist() for item in batch]
        labels = [item["input_ids"][1:].tolist() for item in batch]
        contexts = [item["context_ids"].tolist() for item in batch]
        answers = [item["answer_ids"].tolist() for item in batch]
        loss_mask = [item["loss_mask"][1:].tolist() for item in batch]
        metadata = [item["metadata"] for item in batch]

        # Calculate max length
        max_length = max(len(x) for x in input_ids)

        # Truncate if needed
        if max_length > self.max_seq_length:
            input_ids = [x[: self.max_seq_length] for x in input_ids]
            labels = [x[: self.max_seq_length] for x in labels]
            loss_mask = [x[: self.max_seq_length] for x in loss_mask]

            # Safety check: warn if truncation removed all trainable tokens
            for i, mask in enumerate(loss_mask):
                if sum(mask) == 0:
                    logger.warning(
                        f"Sample {i}: Truncation removed all trainable tokens. Setting loss_mask to all ones."
                    )
                    loss_mask[i] = [1] * self.max_seq_length

            contexts = [x[: self.max_seq_length] for x in contexts]
            answers = [x[: self.max_seq_length] for x in answers]

        # Pad to max length or nearest multiple of 16
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, ((max_length + 15) // 16) * 16)

        # Create position IDs
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)

        # Pad all sequences
        eos_id = getattr(self.tokenizer, "eod", 0)
        input_ids = torch.LongTensor(self._collate_item(input_ids, max_length, eos_id))
        labels = torch.LongTensor(self._collate_item(labels, max_length, eos_id))
        loss_mask = torch.FloatTensor(self._collate_item(loss_mask, max_length, 0))
        context_lengths = torch.LongTensor([len(x) for x in contexts])
        contexts = torch.LongTensor(self._collate_item(contexts, max_length, eos_id))
        answers = torch.LongTensor(self._collate_item(answers, max_length, eos_id))

        # Build result
        processed_batch = {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "contexts": contexts,
            "context_lengths": context_lengths,
            "answers": answers,
            "metadata": metadata,
        }

        # Add attention mask if needed
        if not self.get_attention_mask_from_fusion:
            attention_mask = [self._create_attention_mask(max_length) for _ in batch]
            processed_batch["attention_mask"] = torch.stack(attention_mask)

        return processed_batch
