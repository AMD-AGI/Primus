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
    pattern_from_checkpoint_args,
    save_upcycle_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-base-checkpoint", type=Path, required=True)
    parser.add_argument("--gdn-ild-checkpoint", type=Path, required=True)
    parser.add_argument("--mla-ild-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-pattern", help="defaults to the pattern the base checkpoint was saved with")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base, base_envelope = load_checkpoint(args.hybrid_base_checkpoint)
    saved = pattern_from_checkpoint_args(base_envelope.get("args"))
    pattern = "".join((args.hybrid_pattern or saved or "").split())
    if not pattern:
        parser.error("the base checkpoint records no hybrid pattern; pass --hybrid-pattern")
    if saved and pattern != saved:
        parser.error(f"--hybrid-pattern {pattern} differs from the base checkpoint's {saved}")
    gdn, _ = load_checkpoint(args.gdn_ild_checkpoint)
    mla, _ = load_checkpoint(args.mla_ild_checkpoint)
    assembled = assemble_ild_state_dict(base, gdn, mla, pattern)
    checkpoint = build_upcycle_checkpoint(base_envelope, assembled)
    report = UpcycleReport(
        pattern=pattern,
        num_dense_layers=len(pattern) // 2,
        copied=sorted(key for key in assembled if key.startswith("decoder.layers.")),
    )
    output = save_upcycle_checkpoint(checkpoint, args.output_dir, report)
    print(f"wrote assembled HyLo stage-2 checkpoint to {output}")


if __name__ == "__main__":
    main()
