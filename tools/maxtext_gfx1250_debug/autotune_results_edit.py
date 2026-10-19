#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inspect / splice XLA autotune result textprotos (entries keyed by hlo_fingerprint).

  autotune_results_edit.py show A.txt                       fingerprint -> algorithm (0 = default)
  autotune_results_edit.py diff A.txt B.txt                 entries whose algorithm differs
  autotune_results_edit.py splice BASE.txt DONOR.txt OUT.txt fp1,fp2,...   BASE with DONOR's config for the listed fingerprints
"""
import re
import sys

ENTRY = re.compile(r"^entries \{\n.*?^\}\n", re.S | re.M)
FP = re.compile(r'hlo_fingerprint: "([0-9a-f]+)"')
ALG = re.compile(r"algorithm: (\d+)")
BACKEND = re.compile(r"backend: (\w+)")


def load(path):
    text = open(path).read()
    out = {}
    for m in ENTRY.finditer(text):
        e = m.group(0)
        a = ALG.search(e)
        b = BACKEND.search(e)
        out[FP.search(e).group(1)] = (e, int(a.group(1)) if a else 0, b.group(1) if b else "?")
    return out


def main():
    cmd = sys.argv[1]
    if cmd == "show":
        for fp, (_, alg, be) in sorted(load(sys.argv[2]).items()):
            print(fp, be, alg)
    elif cmd == "diff":
        a, b = load(sys.argv[2]), load(sys.argv[3])
        for fp in sorted(set(a) | set(b)):
            x = a.get(fp, (None, None, None))[1]
            y = b.get(fp, (None, None, None))[1]
            if x != y:
                print(fp, x, y)
    elif cmd == "splice":
        base, donor = load(sys.argv[2]), load(sys.argv[3])
        fps = [f for f in sys.argv[5].split(",") if f]
        for fp in fps:
            base[fp] = donor[fp]
        with open(sys.argv[4], "w") as f:
            f.write("".join(e for e, _, _ in base.values()))
        print(f"wrote {sys.argv[4]} ({len(base)} entries, {len(fps)} spliced)")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
