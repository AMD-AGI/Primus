#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Mean Enhanced-ILD loss per evaluated checkpoint, from ``run_hylo_ild_eval.sh`` logs.

Reads Megatron's iteration lines (``iteration N/ M | ... | hylo ild: ... |``). The evaluation
recipes log every step and train one micro-batch a step, so each line is one evaluated batch.
Pass one ``name=log`` pair per checkpoint::

    python tools/hybrid/summarize_hylo_ild_eval.py init=eval-init.log it6934=eval-it6934.log \\
        reference=eval-reference.log
"""

from __future__ import annotations

import re
import statistics
import sys
from typing import Dict, List, Sequence

_ITER = re.compile(r"iteration\s+\d+/\s*\d+ \|(.*)")
_NUM = re.compile(r"\s*([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
_KEYS = ("hylo ild", "hylo hidden ild", "hylo mixer ild")


def parse(path: str) -> Dict[str, List[float]]:
    """Per-step values of each loss term in one evaluation log."""
    values: Dict[str, List[float]] = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = _ITER.search(line)
            if not m:
                continue
            for part in m.group(1).split("|"):
                key, _, value = part.partition(":")
                number = _NUM.match(value)
                if number and key.strip() in _KEYS:
                    values.setdefault(key.strip(), []).append(float(number.group(1)))
    return values


def summarize(values: Dict[str, List[float]]) -> Dict[str, float]:
    total = values.get("hylo ild", [])
    if not total:
        return {}
    out = {"n": len(total), "median": statistics.median(total), "first": total[0], "last": total[-1]}
    for key in _KEYS:
        out[key] = statistics.fmean(values[key]) if values.get(key) else float("nan")
    return out


def main(argv: Sequence[str] = None) -> List[tuple]:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.exit(__doc__)
    rows = []
    print(f"{'checkpoint':18s} {'n':>4s} {'ild mean':>9s} {'median':>9s} {'hidden':>9s} {'mixer':>9s}")
    for arg in argv:
        name, path = arg.split("=", 1)
        s = summarize(parse(path))
        rows.append((name, s))
        if not s:
            print(f"{name:18s} no iteration lines with 'hylo ild'")
            continue
        print(
            f"{name:18s} {s['n']:4d} {s['hylo ild']:9.3f} {s['median']:9.3f} "
            f"{s['hylo hidden ild']:9.3f} {s['hylo mixer ild']:9.3f}"
        )
    return rows


if __name__ == "__main__":
    main()
