#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Upcycle a dense Megatron checkpoint into a non-MoE hybrid checkpoint.

Embeddings, the final norm, and every MLP are copied from the dense model.
MLA, GDN, and Mamba sublayers are initialized with the HyLo from-teacher
recipes (SVD for MLA, Q/K/V/O copy for GDN and Mamba) when the checkpoint
args carry those widths. KDA mixers keep the initialization in
``--hybrid-init-checkpoint``.

The hybrid init checkpoint is a legacy ``ckpt_format: torch`` save of the
target hybrid config at TP=PP=1. A one-iteration mock run with ``lr: 0``
writes constructor weights::

    lr: 0.0
    min_lr: 0.0
    train_iters: 1
    save_interval: 1
    mock_data: true
    ckpt_format: torch
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1

Then::

    python tools/hybrid/upcycle_dense_to_hybrid.py \\
        --dense-checkpoint /path/to/llama/checkpoints \\
        --hybrid-init-checkpoint /path/to/hybrid-init \\
        --output-dir /path/to/upcycled \\
        --hybrid-pattern '*-M-M-M-*-M-M-M-'

Point the hybrid pretrain YAML at the output directory::

    load: /path/to/upcycled
    finetune: true
    no_load_optim: true
    no_load_rng: true
    auto_continue_train: false
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if __name__ == "__main__":
    from primus.backends.megatron.checkpoint.hybrid_upcycle import main

    main()
