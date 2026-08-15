###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan FSDP2 weight-tying patch.

Qwen3 enables ``enable_weight_tying``, sharing ``tok_embeddings.weight`` with
``output.weight``. Upstream ``apply_fsdp`` wraps those modules in separate
FSDP groups, which PyTorch 2.12 FSDP2 rejects at lazy init:

    ValueError: Parameter 'None' is shared with a parameter already managed by
    another FSDP group. For shared/tied parameters, use
    fully_shard([module_a, module_b]) to place them in the same FSDP group.

This patch wraps upstream ``apply_fsdp`` and temporarily redirects the two
embedding/output ``fully_shard`` calls without modifying third_party/torchtitan.
"""

from __future__ import annotations

from typing import Any, Callable

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _normalize_modules(target: Any) -> list[Any]:
    if isinstance(target, (list, tuple)):
        return list(target)
    return [target]


def _make_apply_fsdp_wrapper(
    orig_apply_fsdp: Callable[..., Any],
    parallelize_mod: Any,
) -> Callable[..., Any]:
    orig_fully_shard = parallelize_mod.fully_shard

    def apply_fsdp(model, *args, **kwargs):
        if not getattr(model, "enable_weight_tying", False):
            return orig_apply_fsdp(model, *args, **kwargs)

        def fully_shard_proxy(target, **fsdp_kwargs):
            modules = _normalize_modules(target)
            if modules == [model.tok_embeddings] and model.output is not None:
                return orig_fully_shard(
                    [model.tok_embeddings, model.output],
                    **fsdp_kwargs,
                )
            if (
                model.norm is not None
                and model.output is not None
                and set(modules) == {model.norm, model.output}
            ):
                return orig_fully_shard(model.norm, **fsdp_kwargs)
            return orig_fully_shard(target, **fsdp_kwargs)

        parallelize_mod.fully_shard = fully_shard_proxy
        try:
            return orig_apply_fsdp(model, *args, **kwargs)
        finally:
            parallelize_mod.fully_shard = orig_fully_shard

    return apply_fsdp


@register_patch(
    "torchtitan.fsdp.weight_tying",
    backend="torchtitan",
    phase="setup",
    description=(
        "Group tied tok_embeddings/output weights in one FSDP2 shard group for "
        "models with enable_weight_tying (Qwen3 on PyTorch 2.12+)."
    ),
)
def patch_torchtitan_fsdp_weight_tying(ctx: PatchContext) -> None:  # noqa: ARG001
    import importlib

    patched = []
    try:
        llama4_mod = importlib.import_module("torchtitan.models.llama4.infra.parallelize")
        wrapped = _make_apply_fsdp_wrapper(llama4_mod.apply_fsdp, llama4_mod)
        llama4_mod.apply_fsdp = wrapped
        patched.append("llama4")

        qwen3_mod = importlib.import_module("torchtitan.models.qwen3.infra.parallelize")
        qwen3_mod.apply_fsdp = wrapped
        patched.append("qwen3")
    except ImportError as exc:
        log_rank_0(
            "[Patch:torchtitan.fsdp.weight_tying] "
            f"Skipped: torchtitan parallelize modules unavailable ({exc})",
        )
        return

    log_rank_0(
        "[Patch:torchtitan.fsdp.weight_tying] "
        f"Patched apply_fsdp for weight tying in: {', '.join(patched)}",
    )
