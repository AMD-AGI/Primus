###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Runtime safety gate for TorchTitan Primus-Turbo attention."""

from typing import Any


def _model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    return str(getattr(model, "name", model) or "")


def _is_gfx942() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return "gfx942" in (torch.cuda.get_device_properties(0).gcnArchName or "")
    except Exception:  # pragma: no cover - fail open outside supported GPU runs
        return False


def requires_ck_mla_backward(model: Any, turbo_config: Any) -> bool:
    """Whether gfx942 MLA must reject the unsupported fmha_v3 backward path."""
    is_deepseek = "deepseek" in _model_name(model).lower()
    is_fp8_recipe = bool(
        getattr(turbo_config, "use_turbo_float8_linear", False) or getattr(turbo_config, "use_moe_fp8", False)
    )
    uses_nonclassic_attention = not bool(getattr(turbo_config, "use_classic_attention", False))
    return is_deepseek and is_fp8_recipe and uses_nonclassic_attention and _is_gfx942()


def should_use_turbo_attention(model: Any, turbo_config: Any) -> bool:
    return bool(
        getattr(turbo_config, "enable_primus_turbo", False)
        and getattr(turbo_config, "use_turbo_attention", False)
    )
