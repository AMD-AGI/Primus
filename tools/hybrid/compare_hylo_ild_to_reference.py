#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Compare an Enhanced-ILD student with a Zebra-Llama reference stage-1 student, tensor by tensor.

Prints, per tensor group (median over layers), the cosine and relative distance to the
reference, how far the student and the reference moved from the student's upcycled init, the
cosine and norm ratio of the two updates, and how close the init is to its teacher source::

    python tools/hybrid/compare_hylo_ild_to_reference.py --kind m2 \\
        --ours /path/to/run/checkpoints --iteration 6934 \\
        --init /path/to/all-m2-upcycled \\
        --reference /path/to/reference/model.safetensors \\
        --dense /path/to/Llama-3.2-1B-Instruct/model.safetensors \\
        --json m2-it6934.json

Add ``--shared-random-init`` when the student started from the reference's own Mamba-2 draw
(``write_mamba2_reference_init.py``), so every tensor is compared rather than only the
statistics of the random ones. Runs on the CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if __name__ == "__main__":
    from primus.backends.megatron.checkpoint.hylo_reference_compare import compare_main

    compare_main()
