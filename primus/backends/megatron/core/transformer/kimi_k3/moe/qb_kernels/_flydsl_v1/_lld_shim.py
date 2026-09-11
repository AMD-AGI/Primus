###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Re-export of the ROCm ``ld.lld`` shim. Deliberately not a second copy.

See ``kda_kernels/_flydsl_v1/_lld_shim.py`` for the defect (a 26 KB relocation
trampoline that resolves the linker from ``argv[0]``, which MLIR spawns bare) and
for the two plausible fixes that do not work. The fix is process-global and
idempotent, so importing it here means the shadow ROCm toolkit is built at most
once per process no matter which kernel family compiles first.
"""

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._lld_shim import (
    ensure_usable_lld,
)

__all__ = ["ensure_usable_lld"]
