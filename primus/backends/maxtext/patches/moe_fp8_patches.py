###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MoE FP8 Scan Patch

Patches ``RoutedMoE.get_einsum`` so that FP8 quantized MoE expert matmuls
fall back to ``jnp.einsum`` (bf16) when ``scan_layers`` is enabled.

The upstream ``Fp8Einsum`` module allocates Flax variables (amax_history,
scales) inside ``setup()``. When a MoE layer runs inside ``jax.lax.scan``
(e.g. gemma4's scanned local layers), those variable allocations leak JAX
tracers and crash with ``UnexpectedTracerError``. Bypassing ``Fp8Einsum``
for the MoE einsum path avoids the leak while keeping FP8 active for
dense layers via ``dot_general_cls``.
"""

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


@register_patch(
    patch_id="maxtext.moe_fp8_scan",
    backend="maxtext",
    phase="setup",
    description="Bypass Fp8Einsum in MoE when scan_layers is true (tracer leak fix)",
    condition=lambda ctx: True,
)
def patch_moe_fp8_scan(ctx: PatchContext) -> None:
    """Monkey-patch RoutedMoE.get_einsum to skip Fp8Einsum under scan."""
    try:
        from maxtext.layers import moe, quantizations
        import jax.numpy as jnp
    except ImportError as e:
        warning_rank_0(
            f"[Patch:moe_fp8_scan] Could not import maxtext.layers.moe; skipping: {e}"
        )
        return

    RoutedMoE = moe.RoutedMoE
    _original_get_einsum = RoutedMoE.get_einsum

    def _patched_get_einsum(self, rhs_mesh_axes=(), einsum_name=None):
        if self.config.model_call_mode == "inference" and einsum_name in (
            moe.DISPATCH,
            moe.COMBINE,
        ):
            return jnp.einsum

        if self.quant:
            is_fp8 = isinstance(
                self.quant,
                (quantizations.Fp8Quantization, quantizations.NANOOFp8Quantization),
            )
            if is_fp8 and self.config.scan_layers:
                return jnp.einsum

        return _original_get_einsum(self, rhs_mesh_axes=rhs_mesh_axes, einsum_name=einsum_name)

    RoutedMoE.get_einsum = _patched_get_einsum
    log_rank_0("[Patch:moe_fp8_scan] Patched RoutedMoE.get_einsum for FP8+scan compatibility.")
