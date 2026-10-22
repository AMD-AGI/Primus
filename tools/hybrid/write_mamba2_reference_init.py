#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Write a reference all-Mamba-2 student's random init into an upcycled checkpoint.

Copies ``--template`` (the output of ``upcycle_dense_to_hybrid.py``) and replaces each mixer's
z and dt rows of ``in_proj``, convolution weight and bias, ``A_log``, and ``dt_bias`` with the
values the reference drew, rebuilt for the count ``scan_mamba2_reference_draw.py`` reported.
The from-teacher rows, ``D``, the norms, and the MLPs stay as the upcycler wrote them::

    python tools/hybrid/write_mamba2_reference_init.py \\
        --reference /path/to/reference/model.safetensors \\
        --template /path/to/all-m2-upcycled \\
        --compute-units <count from the scan> \\
        --output-dir /path/to/all-m2-reference-init

It prints each rebuilt tensor's correlation with the reference's trained tensor, writes
``iter_0000000`` and ``reference_draw_report.json``, and is loaded like any upcycled checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if __name__ == "__main__":
    from primus.backends.megatron.checkpoint.mamba2_reference_draw import write_main

    write_main()
