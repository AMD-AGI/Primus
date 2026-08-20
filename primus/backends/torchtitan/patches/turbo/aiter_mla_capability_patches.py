###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route the failing gfx942 DeepSeek MLA backward directly to aiter CK."""

from typing import Any

from primus.backends.torchtitan.patches.turbo.attention_safety import (
    requires_ck_mla_backward,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_LOG_PREFIX = "[Patch:torchtitan.primus_turbo.aiter_mla_capability]"
_WRAPPED_ATTR = "_primus_gfx942_mla_ck_backward"


def _can_patch(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return requires_ck_mla_backward(ctx.model_name, args.primus_turbo)


def _is_gfx942_tensor(tensor) -> bool:
    try:
        import torch

        arch = torch.cuda.get_device_properties(tensor.device).gcnArchName or ""
        return "gfx942" in arch
    except Exception:  # pragma: no cover - defensive
        return False


def _should_use_ck(kwargs: dict[str, Any]) -> bool:
    q = kwargs["q"]
    v = kwargs["v"]
    return bool(
        kwargs.get("sink") is None
        and kwargs.get("qkv_format", "bshd") == "bshd"
        and q.ndim == 4
        and v.ndim == 4
        and q.shape[-1] in (128, 192)
        and v.shape[-1] == 128
        and kwargs["dropout_p"] == 0.0
        and kwargs.get("bias") is None
        and kwargs.get("alibi_slopes") is None
        and kwargs.get("dbias") is None
        and _is_gfx942_tensor(q)
    )


def _run_ck_backward(mha, kwargs: dict[str, Any]):
    softmax_d = mha.mha_bwd(
        kwargs["dout"],
        kwargs["q"],
        kwargs["k"],
        kwargs["v"],
        kwargs["out"],
        kwargs["softmax_lse"],
        kwargs["dropout_p"],
        kwargs["softmax_scale"],
        kwargs["causal"],
        kwargs["window_size_left"],
        kwargs["window_size_right"],
        kwargs["deterministic"],
        kwargs["dq"],
        kwargs["dk"],
        kwargs["dv"],
        kwargs["dbias"],
        kwargs["bias"],
        kwargs["alibi_slopes"],
        kwargs["rng_state"],
        None,
        kwargs["sink"],
        kwargs["dsink"],
    )
    return (
        softmax_d,
        kwargs["dq"],
        kwargs["dk"],
        kwargs["dv"],
        kwargs["dbias"],
        kwargs["dsink"],
    )


def _make_execute_wrapper(original, mha):
    def execute(*args, **kwargs):
        if not args and _should_use_ck(kwargs):
            return _run_ck_backward(mha, kwargs)
        return original(*args, **kwargs)

    setattr(execute, _WRAPPED_ATTR, True)
    return execute


@register_patch(
    "torchtitan.primus_turbo.aiter_mla_capability",
    backend="torchtitan",
    phase="setup",
    description="Route failing gfx942 DeepSeek MLA backward from v3 to aiter CK",
    condition=_can_patch,
    priority=51,
)
def patch_aiter_mla_capability(ctx: PatchContext) -> None:
    import aiter.ops.mha as mha
    from primus_turbo.pytorch.kernels.attention import attention_aiter_impl

    backend = attention_aiter_impl.AttnBwdAiterBackend
    original = backend.execute
    if getattr(original, _WRAPPED_ATTR, False):
        return

    backend.execute = staticmethod(_make_execute_wrapper(original, mha))
    log_rank_0(f"{_LOG_PREFIX} routing gfx942 DeepSeek MLA backward directly to aiter CK")
