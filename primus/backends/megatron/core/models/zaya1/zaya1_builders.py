###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ZAYA1 hooks for the shared builder (``MODEL_FAMILIES["zaya1"]`` in ``registry.py``)."""

from functools import partial

from primus.backends.megatron.core.models.builder import build_model, model_provider

__all__ = ["zaya1_builder", "model_provider", "prepare_zaya1_args", "zaya1_model_kwargs"]


def _assert_data_parallel_only(args) -> None:
    """CCA head-parallel TP, conv/value-shift CP, and expert-parallel dispatch are not implemented."""
    blocked = []
    for name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
    ):
        size = int(getattr(args, name, 1) or 1)
        if size != 1:
            blocked.append(f"{name}={size}")
    if getattr(args, "sequence_parallel", False):
        blocked.append("sequence_parallel=True")
    if blocked:
        raise NotImplementedError(
            "ZAYA1 pretraining currently runs with data parallel only "
            "(the base paper trained DP + ZeRO-1). Not implemented: " + ", ".join(blocked) + "."
        )


def prepare_zaya1_args(args) -> None:
    _assert_data_parallel_only(args)
    assert not getattr(
        args, "multi_latent_attention", False
    ), "ZAYA1 must leave multi_latent_attention false."
    if args.position_embedding_type != "rope":
        raise ValueError(
            "ZAYA1 applies partial RoPE inside CCA. "
            f"position_embedding_type must be 'rope', got {args.position_embedding_type!r}."
        )


def zaya1_model_kwargs(args) -> dict:
    return {
        "position_embedding_type": args.position_embedding_type,
        "scatter_embedding_sequence_parallel": False,
    }


zaya1_builder = partial(build_model, "zaya1")
