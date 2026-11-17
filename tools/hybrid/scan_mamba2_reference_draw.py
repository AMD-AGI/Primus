#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Find the compute-unit count that rebuilds a reference all-Mamba-2 student's random init.

The random Mamba-2 tensors (z and dt rows of ``in_proj``, the convolution, ``A_log``,
``dt_bias``) were drawn on a GPU, and PyTorch's GPU generator maps its stream to elements by
the GPU's compute-unit count. This scan rebuilds the draw for each candidate count on the CPU
and reports the count whose tensors match the reference's trained ones (which barely move in
Enhanced-ILD)::

    python tools/hybrid/scan_mamba2_reference_draw.py \\
        --reference /path/to/reference/model.safetensors

Then write the draw into an upcycled checkpoint with ``write_mamba2_reference_init.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if __name__ == "__main__":
    from primus.backends.megatron.checkpoint.mamba2_reference_draw import scan_main

    scan_main()
