###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Mock HF Dataset Patch

This patch mirrors ``patch_mock_hf_dataset`` from
``primus.modules.trainer.torchtitan.patch_utils`` using the generic Primus
patch system, so that HF dataset mocking can be enabled via config without
tightly coupling it to the trainer implementation.
"""

from datasets import Dataset
from datasets.search import np

from primus.core.patches import PatchContext, get_param, register_patch
from primus.modules.module_utils import log_rank_0


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


@register_patch(
    "torchtitan.training.mock_hf_dataset",
    backend="torchtitan",
    phase="setup",
    description="Enable mock HuggingFace dataset mode for TorchTitan",
    condition=lambda ctx: get_param(ctx, "training.mock_data", False),
)
def patch_mock_hf_dataset(ctx: PatchContext) -> None:  # noqa: ARG001
    """
    Patch HF datasets.load_dataset with a lightweight mock implementation.

    This patch replaces ``datasets.load_dataset`` with a mock that returns
    fake tokenized datasets, avoiding the need to download real HuggingFace
    datasets during testing or validation runs.
    """
    try:
        import datasets

        log_rank_0(
            "[Patch:torchtitan.training.mock_hf_dataset] " "Enabling mock HuggingFace dataset mode.",
        )

        def mock_load_dataset(path: str, *args, **kwargs) -> Dataset:
            """
            Replacement for datasets.load_dataset().
            Intercepts Titan calls like load_dataset('allenai/c4', ...).
            Returns a fake Dataset of text samples.
            """
            log_rank_0(
                "[Patch:torchtitan.training.mock_hf_dataset] " f"load_dataset('{path}') is mocked.",
            )
            # Shorter dataset for validation split
            if "validation" in path.lower():
                return _create_mock_text_dataset(num_samples=32)
            else:
                return _create_mock_token_dataset(seq_len=8192, vocab_size=32000, num_samples=256)

        datasets.load_dataset = mock_load_dataset
        log_rank_0(
            "[Patch:torchtitan.training.mock_hf_dataset] " "Patched datasets.load_dataset successfully.",
        )

    except Exception as e:
        log_rank_0(
            "[Patch:torchtitan.training.mock_hf_dataset][ERROR] "
            f"Failed to patch datasets.load_dataset: {e}",
        )
