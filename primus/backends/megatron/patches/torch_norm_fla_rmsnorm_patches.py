###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
WrappedTorchNorm FLA RMSNorm Patch
====================================

Routes ``WrappedTorchNorm``'s RMSNorm construction through
``fla.modules.RMSNorm`` (flash-linear-attention's Triton-fused kernel)
instead of ``torch.nn.RMSNorm``, to match FLA's normalization numerics
exactly for GDN/KDA/Mamba hybrid parity training.

Toggle: ``args.use_fla_fused_rmsnorm`` (resolved by ``fla_runtime_patches.py``
from ``PRIMUS_FLA_NORM`` / YAML ``use_fla_fused_rmsnorm``, default False).

``WrappedTorchNorm.__new__`` directly returns the constructed norm module
(no plain ``__init__`` path), so this is a simple function-wrapping patch --
no source rewrite needed.
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.torch_norm.fla_rmsnorm"


def _install_fla_rmsnorm_patch() -> None:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm
    from megatron.training import get_args as _get_args

    if is_patched(WrappedTorchNorm, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] WrappedTorchNorm already patched; skipping.")
        return

    original_new = WrappedTorchNorm.__new__

    def patched_new(
        cls,
        config,
        hidden_size,
        eps=1e-5,
        persist_layer_norm=False,
        zero_centered_gamma=False,
        normalization="LayerNorm",
    ):
        if config.normalization == "RMSNorm" and getattr(_get_args(), "use_fla_fused_rmsnorm", False):
            from fla.modules import RMSNorm as FLARMSNorm

            return FLARMSNorm(hidden_size=hidden_size, eps=eps)
        return original_new(
            cls,
            config,
            hidden_size,
            eps=eps,
            persist_layer_norm=persist_layer_norm,
            zero_centered_gamma=zero_centered_gamma,
            normalization=normalization,
        )

    WrappedTorchNorm.__new__ = patched_new
    mark_patched(WrappedTorchNorm, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] Patched WrappedTorchNorm.__new__ to use fla.modules.RMSNorm "
        "when use_fla_fused_rmsnorm is set."
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Route WrappedTorchNorm's RMSNorm construction through fla.modules.RMSNorm "
        "to match FLA's normalization kernel exactly."
    ),
    # Runs after fla_runtime_knobs (priority=-100) has resolved args.use_fla_fused_rmsnorm.
    priority=50,
    condition=lambda ctx: getattr(get_args(ctx), "use_fla_fused_rmsnorm", False),
)
def patch_torch_norm_fla_rmsnorm(ctx: PatchContext) -> None:
    _install_fla_rmsnorm_patch()
