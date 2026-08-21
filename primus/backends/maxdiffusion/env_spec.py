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

Unlike MaxText, MaxDiffusion keeps *all* of its JAX/XLA/NVTE/RCCL/HIP tuning in
the per-config top-level ``env:`` block of each wrapper config
(``examples/maxdiffusion/configs/**``), which TrainRuntime applies before JAX
init. Those blocks are a faithful, complete copy of the verified-good
``scripts/jax-maxdiffusion/env_scripts/base_*_env.sh`` from the
``clairlee/feat/maxdiffusion_support`` baseline.

The adapter therefore contributes *no* environment of its own. In particular it
must NOT inject ``RCCL_WARP_SPEED_AUTO`` — the known-good MaxDiffusion baseline
never sets it, and forcing it here diverged from that baseline (contributing to
an RCCL init hang on gfx950). Any arch-specific diffusion knob belongs in the
per-config ``env:`` block alongside the rest of the tuning.

Precedence (see env_registry): per-config ``env:`` > outer/shell env > image-baked.
"""

from __future__ import annotations

from typing import List

from primus.core.backend.env_registry import EnvVar


def maxdiffusion_env_defaults() -> List[EnvVar]:
    """MaxDiffusion owns no adapter-level env; the per-config ``env:`` block is the
    single source of truth (see module docstring)."""
    return []
