###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL's ``buffer_ops`` for the MSA kernels, on either FlyDSL line.

FlyDSL 0.3 dropped ``flydsl.expr.buffer_ops``, and Primus-Turbo vendors the
0.2.4 module for it. Primus-Turbo builds that predate the vendored copy -- the
one in ``rocm/primus:v26.8`` among them -- ship FlyDSL 0.2.x, which still has
the original.
"""

try:
    from primus_turbo.flydsl.utils import buffer_ops
except ImportError:
    from flydsl.expr import buffer_ops

__all__ = ["buffer_ops"]
