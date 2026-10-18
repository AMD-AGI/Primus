###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Fused (prob-weighted) SwiGLU (pinned Triton)
============================================

With ``use_primus_fused_swiglu: true`` the SwiGLU entry points that run under
``bias_activation_fusion`` are replaced by one Triton kernel per direction
(``primus.backends.megatron.core.fusions.triton_swiglu``; BLOCK_N 1024,
num_warps 4, num_stages 1, no autotune). Without torch.compile (``jit_fuser``
off, e.g. ``TORCH_COMPILE_DISABLE=1``) Megatron's SwiGLU functions run as
eager chains of 4 forward / ~16 backward kernels with fp32 temporaries. The
probs gradient of the weighted variant (a row sum) is computed in the same
backward pass.

Patched names (module globals, so callers that bind them by ``from ... import``
at call time or at import time are both covered):

  primus.backends.megatron.core.fusions.fused_bias_swiglu.weighted_bias_swiglu_impl   PrimusGroupedMLP
  megatron.core.fusions.fused_bias_swiglu.{weighted_bias_swiglu_impl, bias_swiglu_impl}
  megatron.core.transformer.mlp.{weighted_bias_swiglu_impl, bias_swiglu_impl}          dense / shared expert
  megatron.core.transformer.moe.shared_experts.bias_swiglu_impl                       overlapped shared expert
  megatron.core.transformer.moe.experts.weighted_bias_swiglu_impl                     TEGroupedMLP

Calls outside the fused contract (bias, clamp, fp8 input store, CPU offload,
non-CUDA or non-16-bit input) run the original function. The Turbo path
(``use_turbo_fused_act_with_probs``) bypasses these names and is unaffected.
"""

from __future__ import annotations

import importlib

from primus.backends.megatron.patches._gfx1250_notice import warn_if_not_gfx1250
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.activation.fused_swiglu"
_ARG = "use_primus_fused_swiglu"

_PRIMUS = "primus.backends.megatron.core.fusions.fused_bias_swiglu"
_MFUS = "megatron.core.fusions.fused_bias_swiglu"
# module -> [(attribute, kind)]
SITES = {
    _PRIMUS: [("weighted_bias_swiglu_impl", "primus_weighted")],
    _MFUS: [("weighted_bias_swiglu_impl", "megatron_weighted"), ("bias_swiglu_impl", "megatron_plain")],
    "megatron.core.transformer.mlp": [
        ("weighted_bias_swiglu_impl", "megatron_weighted"),
        ("bias_swiglu_impl", "megatron_plain"),
    ],
    "megatron.core.transformer.moe.shared_experts": [("bias_swiglu_impl", "megatron_plain")],
    "megatron.core.transformer.moe.experts": [("weighted_bias_swiglu_impl", "megatron_weighted")],
}


def patch_sites() -> dict:
    """Patch every site; returns {"module.attr": True | "absent" | error string}."""
    from primus.backends.megatron.core.fusions import triton_swiglu as ops

    factory = {
        "primus_weighted": ops.make_primus_weighted,
        "megatron_weighted": ops.make_megatron_weighted,
        "megatron_plain": ops.make_megatron_plain,
    }
    result = {}
    for modname, attrs in SITES.items():
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001 - optional site
            for attr, _ in attrs:
                result[f"{modname}.{attr}"] = f"import failed: {exc!r}"
            continue
        for attr, kind in attrs:
            key = f"{modname}.{attr}"
            cur = getattr(mod, attr, None)
            if cur is None:
                result[key] = "absent"
                continue
            if getattr(cur, ops.ORIGINAL_ATTR, None) is not None:
                result[key] = True
                continue
            setattr(mod, attr, factory[kind](cur))
            result[key] = True
    return result


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Fused (prob-weighted) SwiGLU fwd/bwd with one pinned Triton config",
    condition=lambda ctx: bool(getattr(get_args(ctx), _ARG, False)),
    tags=["activation", "moe", "triton"],
)
def patch_fused_swiglu(ctx: PatchContext) -> None:
    warn_if_not_gfx1250(_PATCH_KEY, _ARG)
    args = get_args(ctx)
    # turbo_sync_free_moe_stage 3 turns use_turbo_fused_act_with_probs on later, in validate_args.
    if (
        getattr(args, "use_turbo_fused_act_with_probs", False)
        or (getattr(args, "turbo_sync_free_moe_stage", 0) or 0) >= 3
    ):
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] use_turbo_fused_act_with_probs (or turbo_sync_free_moe_stage 3) is set: "
            "routed experts take the Turbo path and do not reach the patched names."
        )
    result = patch_sites()
    bad = {k: v for k, v in result.items() if v is not True and v != "absent"}
    log_rank_0(f"[Patch:{_PATCH_KEY}] patched {sorted(k for k, v in result.items() if v is True)}")
    if bad:
        warning_rank_0(f"[Patch:{_PATCH_KEY}] not patched: {bad}")
