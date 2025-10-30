###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect
from functools import wraps

import numpy as np
from datasets import Dataset


def _create_mock_text_dataset(num_samples: int = 128) -> Dataset:
    """Create a lightweight text dataset for validation mock."""
    texts = [f"validation sample {i}" for i in range(num_samples)]
    return Dataset.from_dict({"text": texts})


def _create_mock_token_dataset(
    seq_len: int = 2048,
    vocab_size: int = 32000,
    num_samples: int = 256,
) -> Dataset:
    """
    Create fake tokenized text dataset (Titan-compatible).

    Each "text" field is a string of roughly `seq_len // 8` space-separated integers.
    Titan's tokenizer.encode() will parse these into tokens and reconstruct
    proper seq_len-sized sequences from multiple samples if needed.

    This lightweight mock simulates a streaming dataset and avoids heavy memory usage.
    """
    rng = np.random.default_rng(42)
    token_per_sample = seq_len  # shorter text, Titan will concatenate internally

    samples = []
    for _ in range(num_samples):
        token_ids = rng.integers(0, vocab_size, size=token_per_sample, dtype=np.int32)
        text = " ".join(map(str, token_ids))
        samples.append({"text": text})

    return Dataset.from_list(samples)


def patch_mock_hf_dataset() -> None:
    from primus.core.utils.logger import _logger as logger

    try:
        import datasets

        def mock_load_dataset(path: str, *args, **kwargs) -> Dataset:
            """
            Replacement for datasets.load_dataset().
            Intercepts Titan calls like load_dataset('allenai/c4', ...).
            Returns a fake Dataset of text samples.
            """
            logger.warning(f"[PrimusPatch][MockDataset] load_dataset('{path}') is mocked.")
            # Shorter dataset for validation split
            if "validation" in path.lower():
                return _create_mock_text_dataset(num_samples=32)
            else:
                return _create_mock_token_dataset(seq_len=8192, vocab_size=32000, num_samples=256)

        datasets.load_dataset = mock_load_dataset
        logger.warning("[PrimusPatch][Dataset] Patched datasets.load_dataset successfully.")

    except Exception as e:
        logger.error(f"[PrimusPatch][Dataset] Failed to patch datasets.load_dataset: {e}")


def apply_patch_checkpoint_wrapper():
    """
    Patch torch.distributed.algorithms._checkpoint.checkpoint_wrapper
    to ignore unsupported kwargs such as `early_stop`.
    """
    from primus.core.utils.logger import _logger as logger

    try:
        import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as ckpt_mod

        orig_fn = ckpt_mod.checkpoint_wrapper

        @wraps(orig_fn)
        def safe_checkpoint_wrapper(*args, **kwargs):
            sig = inspect.signature(orig_fn)
            valid = set(sig.parameters.keys())
            dropped = []
            for k in list(kwargs.keys()):
                if k not in valid:
                    kwargs.pop(k)
                    dropped.append(k)
            return orig_fn(*args, **kwargs)

        ckpt_mod.checkpoint_wrapper = safe_checkpoint_wrapper
        logger.warning("[PrimusPatch][Checkpoint] checkpoint_wrapper patched successfully")
    except Exception as e:
        logger.warning(f"[PrimusPatch][Checkpoint] Failed to patch checkpoint_wrapper: {e}")
