###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Re-export of the ROCm ``ld.lld`` shim. Deliberately not a second copy.

We found that FlyDSL JIT was broken image-wide: the ``ld.lld`` in the
``_rocm_sdk_devel`` wheel that ships this image's toolchain is a 26 KB
relocation trampoline that resolves the real linker from ``argv[0]``, and MLIR's
ROCDL ``gpu-module-to-binary`` pass spawns it with a bare ``argv[0] = "ld.lld"``.
Every FlyDSL kernel therefore failed to compile, down to a trivial vector add.
The fix — a shadow ROCm toolkit symlinking every entry of the real
``$ROCM_PATH`` with only ``ld.lld`` replaced by a wrapper that execs
``lld -flavor gnu`` by absolute path — lives in
``kda_kernels/_flydsl_v1/_lld_shim.py`` and is process-global and idempotent.

Importing it here rather than duplicating it means the shadow toolkit is built
at most once per process no matter which kernel family compiles first, and that
a future toolchain fix has exactly one place to land. Two ways of "fixing" it
that do not work: a symlink (preserves ``argv[0]``) and prepending to ``PATH``
(MLIR looks inside the ROCm *toolkit* directory, not on ``PATH``).
"""

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._lld_shim import (
    ensure_usable_lld,
)

__all__ = ["ensure_usable_lld"]
