#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Write a Zebra-Llama reference stage-1 student as a Megatron checkpoint of the student's layout.

The template (a student checkpoint of the same layout, such as the upcycled init) supplies the
keys, shapes, dtypes, and args; every tensor is replaced by the reference's value. The result
loads like an upcycled checkpoint, so ``run_hylo_ild_eval.sh`` can score the reference against
the same teacher and test data as a Primus run::

    python tools/hybrid/convert_hylo_reference_to_megatron.py --kind mla \\
        --template /path/to/all-mla-upcycled \\
        --reference /path/to/reference/model.safetensors \\
        --output-dir /path/to/reference-as-megatron
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if __name__ == "__main__":
    from primus.backends.megatron.checkpoint.hylo_reference_compare import convert_main

    convert_main()
