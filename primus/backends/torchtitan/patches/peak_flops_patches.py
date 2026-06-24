###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Peak FLOPS Patch
===========================

Teach ``torchtitan.tools.utils.get_peak_flops`` about the AMD MI350 series
(gfx950) so MFU (``num_flops_per_token * tps / gpu_peak_flops``) is correct.

The vendored TorchTitan does not know these devices and silently falls back to
the A100 value (312 TFLOPS), making MFU ~8x too large BF16 dense
peaks below come from AMD's product pages; MI355X matches upstream TorchTitan.
"""

from primus.core.patches import PatchContext, register_patch

# AMD CDNA4 (gfx950) BF16 dense matrix peak, in FLOP/s.
_PEAK_FLOPS = {
    "MI355X": 2500e12,  # liquid-cooled, 2400 MHz; identical to upstream TorchTitan
    "MI350X": 2300e12,  # air-cooled, 2200 MHz; not covered upstream
}


@register_patch(
    patch_id="torchtitan.peak_flops",
    backend="torchtitan",
    phase="setup",  # before MetricsProcessor caches gpu_peak_flops
    description="Add MI350X/MI355X peak FLOPS so MFU is computed correctly",
    condition=lambda ctx: True,
)
def patch_torchtitan_peak_flops(ctx: PatchContext) -> None:
    import torchtitan.tools.utils as titan_utils

    original = titan_utils.get_peak_flops

    def get_peak_flops(device_name: str) -> float:
        for name, flops in _PEAK_FLOPS.items():
            if name in (device_name or ""):
                return flops
        return original(device_name)

    titan_utils.get_peak_flops = get_peak_flops
