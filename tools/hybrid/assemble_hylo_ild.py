#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Assemble a stage-2 HyLo checkpoint from GDN and MLA Enhanced-ILD outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from primus.backends.megatron.checkpoint.hybrid_upcycle import (  # noqa: E402
    UpcycleReport,
    assemble_ild_state_dict,
    build_upcycle_checkpoint,
    load_checkpoint,
    save_upcycle_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-base-checkpoint", type=Path, required=True)
    parser.add_argument("--gdn-ild-checkpoint", type=Path, required=True)
    parser.add_argument("--mla-ild-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-pattern", default="M-*-M-M-M-*-M-M-M-M-*-M-M-M-*-M-")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base, base_envelope = load_checkpoint(args.hybrid_base_checkpoint)
    gdn, _ = load_checkpoint(args.gdn_ild_checkpoint)
    mla, _ = load_checkpoint(args.mla_ild_checkpoint)
    assembled = assemble_ild_state_dict(base, gdn, mla, args.hybrid_pattern)
    checkpoint = build_upcycle_checkpoint(base_envelope, assembled)
    report = UpcycleReport(
        pattern=args.hybrid_pattern,
        num_dense_layers=len(args.hybrid_pattern) // 2,
        copied=sorted(key for key in assembled if key.startswith("decoder.layers.")),
    )
    output = save_upcycle_checkpoint(checkpoint, args.output_dir, report)
    print(f"wrote assembled HyLo stage-2 checkpoint to {output}")


if __name__ == "__main__":
    main()
