###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Route gfx942 known-bad attention shapes to TE, avoiding the aiter FMHA backward crash.

On gfx942 (MI300X/MI325X), aiter's FMHA v3/CK *backward* aborts on the first step for a known
family of attention shapes ("illegal memory access" / "hipModuleLaunchKernel invalid argument"
/ CK "invalid configuration argument", 8-rank SIGABRT, zero iters). Confirmed bad: standard MHA
head_dim==128, MLA head_dim==128 (kernel sees qk=192 / v=128, e.g. DeepSeek-V2-Lite), and
standard GQA head_dim==256 (e.g. qwen3_5_35B). For those shapes only, reroute the attention to
Megatron ``TEDotProductAttention`` (a known-good substitute on gfx942); every other shape keeps
``PrimusTurboAttention`` so it still benefits from Turbo attention.

Runtime monkey-patch of ``PrimusTurboSpecProvider.core_attention`` in the ``before_train`` phase;
no edits to Primus-Turbo/aiter sources, no-op off gfx942. OFF by default; enable explicitly with
``PRIMUS_TURBO_ATTN_HD128_FALLBACK_TE=1`` (CI sets it, see .github/workflows/ci.yaml). Temporary
workaround: remove together with this patch once the upstream aiter fix (ROCm/aiter#1332, same
family) lands. The "hd128" in the name is historical (the first shape we hit); the known-bad set
now also covers hd256, and the name is kept un-renamed to minimize churn.
"""

import functools
import os
from typing import Any, Optional

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

_LOG_PREFIX = "[Patch:megatron.turbo.attn_hd128_te_fallback]"

# Env switch (fallback is OFF by default; CI sets it to 1).
_ENV_FALLBACK_TE = "PRIMUS_TURBO_ATTN_HD128_FALLBACK_TE"

# Standard MHA/GQA head_dims confirmed to crash aiter's FMHA backward on gfx942.
_KNOWN_BAD_HEAD_DIMS = (128, 256)


def _coerce_bool(value: Any, default: bool) -> bool:
    """Best-effort string/bool -> bool, falling back to ``default``."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in ("1", "true", "yes", "on"):
        return True
    if token in ("0", "false", "no", "off"):
        return False
    return default


def _fallback_enabled() -> bool:
    """Read the env switch; preventive fallback is OFF unless explicitly enabled."""
    return _coerce_bool(os.getenv(_ENV_FALLBACK_TE), False)


def _is_gfx942() -> bool:
    """True only on gfx942 (MI300X/MI325X family), where the crash reproduces."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        arch = torch.cuda.get_device_properties(0).gcnArchName or ""
        return "gfx942" in arch
    except Exception:  # pragma: no cover - defensive, never block training
        return False


def _as_int(value: Any) -> Optional[int]:
    """Best-effort int cast, returning ``None`` on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_mha_head_dim(cfg: Any) -> Optional[int]:
    """Standard-MHA head_dim: prefer ``kv_channels``, else hidden_size // num_heads."""
    head_dim = _as_int(getattr(cfg, "kv_channels", None))
    if head_dim:
        return head_dim
    hidden_size = _as_int(getattr(cfg, "hidden_size", None))
    num_heads = _as_int(getattr(cfg, "num_attention_heads", None))
    if hidden_size and num_heads:
        return hidden_size // num_heads
    return None


def _describe_hd128(cfg: Any) -> str:
    """Diagnostic shape label for the hit log only (MLA vs standard MHA); not a gate."""
    if bool(getattr(cfg, "multi_latent_attention", False)):
        return (
            f"MLA, qk_head_dim={_as_int(getattr(cfg, 'qk_head_dim', None))}, "
            f"v_head_dim={_as_int(getattr(cfg, 'v_head_dim', None))}"
        )
    return f"standard MHA, head_dim={_get_mha_head_dim(cfg)}"


def _is_mla_hd128(cfg: Any) -> bool:
    """MLA known-bad: per-head qk and v dims are both 128 (kernel sees qk=192 / v=128)."""
    return (
        _as_int(getattr(cfg, "qk_head_dim", None)) == 128 and _as_int(getattr(cfg, "v_head_dim", None)) == 128
    )


def _is_known_bad(cfg: Any) -> bool:
    """True for the gfx942 aiter-FMHA-backward crash shapes (MHA hd128/hd256, MLA hd128)."""
    if cfg is None:
        return False
    if bool(getattr(cfg, "multi_latent_attention", False)):
        return _is_mla_hd128(cfg)
    return _get_mha_head_dim(cfg) in _KNOWN_BAD_HEAD_DIMS


def _can_patch_attn_hd128_te_fallback(ctx: PatchContext) -> bool:
    """Apply only on gfx942 + usable Turbo attention + known-bad shape + switch enabled."""
    args = get_args(ctx)

    if not bool(getattr(args, "use_turbo_attention", False)):
        return False

    if not is_primus_turbo_can_patch(ctx):
        return False

    if not _is_gfx942():
        log_rank_0(f"{_LOG_PREFIX} device is not gfx942; TE attention fallback not needed.")
        return False

    if not _is_known_bad(args):
        log_rank_0(
            f"{_LOG_PREFIX} not a known-bad shape (gfx942 aiter FMHA backward crashes on "
            "MHA hd128/hd256 or MLA hd128); keeping PrimusTurboAttention."
        )
        return False

    if not _fallback_enabled():
        log_rank_0(
            f"{_LOG_PREFIX} disabled (default off; set {_ENV_FALLBACK_TE}=1 to enable); "
            "keeping PrimusTurboAttention."
        )
        return False

    return True


@register_patch(
    "megatron.turbo.attn_hd128_te_fallback",
    backend="megatron",
    phase="before_train",
    description=(
        "On gfx942, route known-bad attention shapes (MHA hd128/hd256, MLA hd128) to "
        "TEDotProductAttention at spec-build time to avoid the aiter FMHA v3/CK backward crash; "
        "other shapes keep PrimusTurboAttention. Off by default; CI enables it via "
        "PRIMUS_TURBO_ATTN_HD128_FALLBACK_TE=1. Temporary workaround (ROCm/aiter#1332)."
    ),
    condition=_can_patch_attn_hd128_te_fallback,
    # Run after te_spec_provider (priority 41), which installs PrimusTurboSpecProvider as the
    # active spec provider; we then wrap its ``core_attention`` method.
    priority=42,
)
def patch_attn_hd128_te_fallback(ctx: PatchContext) -> None:
    """Wrap ``core_attention`` to return ``TEDotProductAttention`` for the known-bad shapes."""
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    original = PrimusTurboSpecProvider.core_attention
    if getattr(original, "_primus_hd128_te_fallback_wrapped", False):
        log_rank_0(f"{_LOG_PREFIX} core_attention already patched; skipping.")
        return

    # Log only the first known-bad hit to avoid per-layer log spam.
    state = {"hit_logged": False}

    @functools.wraps(original)
    def _core_attention_with_te_fallback(self):
        cfg = getattr(self, "cfg", None)
        if cfg is not None and bool(getattr(cfg, "use_turbo_attention", False)) and _is_known_bad(cfg):
            if not state["hit_logged"]:
                state["hit_logged"] = True
                log_rank_0(
                    f"{_LOG_PREFIX} known-bad shape hit (gfx942, {_describe_hd128(cfg)}): "
                    "using TEDotProductAttention to avoid the aiter FMHA backward crash."
                )
            return TEDotProductAttention
        return original(self)

    _core_attention_with_te_fallback._primus_hd128_te_fallback_wrapped = True
    _core_attention_with_te_fallback._primus_original = original
    PrimusTurboSpecProvider.core_attention = _core_attention_with_te_fallback

    log_rank_0(
        f"{_LOG_PREFIX} Patched PrimusTurboSpecProvider.core_attention "
        f"(gfx942: known-bad shapes [MHA hd128/hd256, MLA hd128] -> TEDotProductAttention; "
        f"other shapes keep PrimusTurboAttention). Enabled via {_ENV_FALLBACK_TE}=1 (default off)."
    )
