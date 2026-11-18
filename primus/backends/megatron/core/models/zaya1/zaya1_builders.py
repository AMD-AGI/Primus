###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ZAYA1 hooks for the shared builder (``MODEL_FAMILIES["zaya1"]`` in ``registry.py``)."""

from functools import partial

from primus.backends.megatron.core.models.builder import (
    build_model,
    family_model_provider,
)

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


def _assert_overlap_grad_reduce_supported(args) -> None:
    """Megatron's overlap_grad_reduce stops training when a parameter gets no gradient in a step.

    Runs before the config is built, so _turbo_experts reads fp8 and fp8_recipe from the arguments.
    """
    from primus.backends.megatron.core.models.zaya1.zaya1_modules import _turbo_experts

    if getattr(args, "overlap_grad_reduce", False) and not _turbo_experts(args)[0]:
        raise ValueError(
            "overlap_grad_reduce needs every parameter to get a gradient each step. ZAYA1's split "
            "experts leave an expert that gets no tokens without one, and the first MoE stage's "
            "router_states_scale too. Set overlap_grad_reduce: false, or run the experts as Primus "
            "Turbo grouped GEMMs (enable_primus_turbo and use_turbo_grouped_gemm, and under FP8 "
            "fp8_recipe: tensorwise)."
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
    _assert_overlap_grad_reduce_supported(args)


def zaya1_model_kwargs(args) -> dict:
    return {
        "position_embedding_type": args.position_embedding_type,
        "scatter_embedding_sequence_parallel": False,
    }


zaya1_builder = partial(build_model, "zaya1")
model_provider = family_model_provider(zaya1_builder)
