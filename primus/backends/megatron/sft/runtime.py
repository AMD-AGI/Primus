###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Runtime wiring helpers for Megatron-native SFT."""

from collections.abc import Callable
import inspect
from typing import Any, Optional

from primus.backends.megatron.sft.dataset import build_train_valid_test_datasets
from primus.modules.module_utils import log_rank_0


def _safe_signature(fn: Callable[..., Any]) -> inspect.Signature | None:
    """Best-effort signature lookup for dynamic wrappers."""
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _supports_kwarg(sig: inspect.Signature, name: str) -> bool:
    """Return whether a callable signature safely accepts the given keyword."""
    if name in sig.parameters:
        return True

    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in sig.parameters.values()
    )


def create_sft_datasets_provider() -> Callable:
    """Create Megatron-compatible SFT dataset provider."""

    def train_valid_test_datasets_provider(
        train_val_test_num_samples: list[int],
        vp_stage: Optional[int] = None,
    ):
        """Build train/valid/test datasets for SFT."""
        del vp_stage

        from megatron.training import get_args, get_tokenizer

        args = get_args()
        tokenizer = get_tokenizer()

        dataset_name = getattr(args, "sft_dataset_name", "tatsu-lab/alpaca")
        conversation_format = getattr(args, "sft_conversation_format", "alpaca")

        log_rank_0(f"Building SFT datasets from: {dataset_name}")
        log_rank_0(f"Using conversation format: {conversation_format}")

        return build_train_valid_test_datasets(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_seq_length=args.seq_length,
            train_val_test_num_samples=train_val_test_num_samples,
            formatter=conversation_format,
            seed=args.seed,
        )

    # Required by Megatron pretrain dataset setup path.
    train_valid_test_datasets_provider.is_distributed = True
    return train_valid_test_datasets_provider


def run_sft_pretrain(
    *,
    pretrain_fn: Callable[..., Any],
    datasets_provider: Callable[..., Any],
    model_provider: Callable[..., Any],
    forward_step: Callable[..., Any],
) -> None:
    """Run Megatron pretrain entrypoint for SFT with Megatron API compatibility."""
    from megatron.core.enums import ModelType

    wrapped_pretrain = pretrain_fn
    store = None
    try:
        from megatron.training import inprocess_restart

        if hasattr(inprocess_restart, "maybe_wrap_for_inprocess_restart"):
            wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain_fn)
    except (ImportError, AttributeError) as exc:
        log_rank_0(f"Inprocess restart not available for SFT pretrain: {exc}")

    # Prefer wrapped signature when available, then fall back to the original
    # pretrain function if a dynamic wrapper hides introspection metadata.
    sig = _safe_signature(wrapped_pretrain) or _safe_signature(pretrain_fn)
    kwargs = {}
    if sig is not None:
        if _supports_kwarg(sig, "args_defaults"):
            kwargs["args_defaults"] = {"tokenizer_type": "GPT2BPETokenizer"}
        if _supports_kwarg(sig, "extra_args_provider"):
            kwargs["extra_args_provider"] = None
        if _supports_kwarg(sig, "store"):
            kwargs["store"] = store

    wrapped_pretrain(
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        **kwargs,
    )
