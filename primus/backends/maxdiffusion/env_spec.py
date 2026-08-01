###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxDiffusion backend environment specification (single source of truth).

This is the MaxDiffusion counterpart of ``primus/backends/maxtext/env_spec.py``.
It declares the architecture environment that Primus is responsible for applying
before JAX/XLA is imported, consumed by :class:`MaxDiffusionAdapter` via
``env_defaults()`` and applied by the base adapter through the shared
``primus.core.backend.env_registry`` mechanism.

Unlike MaxText, MaxDiffusion keeps all of its JAX/XLA/NVTE performance tuning in
the per-config top-level ``env:`` block of each wrapper config
(``examples/maxdiffusion/configs/**``), which TrainRuntime applies before JAX
init. The adapter therefore only needs to own the single arch-gated knob that is
not (and should not be) hard-coded per config:

  * gfx950 (MI350X/MI355X): ``RCCL_WARP_SPEED_AUTO=0`` — WarpSpeed is default-on in
    gfx950 RCCL builds and can cause NaN losses / hangs; harmless on gfx942.

Precedence (see env_registry): per-config ``env:`` > outer/shell env > this
default > image-baked.
"""

from __future__ import annotations

from typing import List

from primus.core.backend.env_registry import ARCH_GFX950, EnvVar


def maxdiffusion_env_defaults() -> List[EnvVar]:
    """Return the declarative MaxDiffusion arch env defaults for the current run."""
    return [
        EnvVar(
            "RCCL_WARP_SPEED_AUTO",
            "0",
            arch=ARCH_GFX950,
            note="gfx950 WarpSpeed default-on can cause NaN losses / hangs",
        ),
    ]
