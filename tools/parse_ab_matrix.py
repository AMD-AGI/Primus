"""Summarize the 2x2 MegaMoE-vs-baseline A/B (bf16 pair, mxfp8 pair) from one run directory.

Logs are named <precision>.<arm>.log, so the pairing is read off the filenames rather than passed in.

Throughput means start at --skip (default 20 of 50): the first iterations carry FlyDSL JIT
compilation and allocator warmup, and including them would credit the baseline arm for not having a
JIT. The within-run spread is printed next to each mean because a difference smaller than it is not
a result.

Loss is compared per iteration, not just at the end: a monotonic gap with no sign flip means the two
arms compute different functions, while a gap that changes sign is quantization jitter. Those two
readings support opposite conclusions from the same final number.
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ITER = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*?per GPU \(TFLOP/s/GPU\):\s*([\d.]+)"
)
LOSS = re.compile(r"iteration\s+(\d+)/\s*\d+ .*?lm loss:\s*([\d.E+-]+)")
MEM = re.compile(r"max reserved: ([\d.]+)")
# The dots are what distinguishes Megatron's argument dump from the launcher echoing a command line
# that contains the same flag names.
FLAGS = {
    k: re.compile(rf"\b{k} \.{{3,}} (\S+)")
    for k in ("use_turbo_mega_moe", "turbo_mega_moe_precision", "fp8", "fp8_recipe", "seed")
}


class Run:
    def __init__(self, path: Path, skip: int):
        text = ANSI.sub("", path.read_text(errors="replace"))
        self.name = path.name[: -len(".log")]
        self.rows = [(int(i), float(ms), float(tf)) for i, ms, tf in ITER.findall(text)]
        self.timed = [r for r in self.rows if r[0] > skip]
        self.loss = {int(i): float(v) for i, v in LOSS.findall(text)}
        mem = MEM.findall(text)  # megatron prints MB
        self.mem = max(float(m) for m in mem) / 1024 if mem else None
        self.flags = {k: (m.group(1) if (m := p.search(text)) else "-") for k, p in FLAGS.items()}

    @property
    def ok(self):
        return bool(self.timed)

    def mean_ms(self):
        return statistics.mean(ms for _, ms, _ in self.timed)

    def mean_tf(self):
        return statistics.mean(tf for _, _, tf in self.timed)

    def spread(self):
        ms = [m for _, m, _ in self.timed]
        return (max(ms) - min(ms)) / statistics.mean(ms) * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--skip", type=int, default=20, help="ignore iterations <= this (JIT + warmup)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    runs = {}
    # Matched by name rather than by "every .log": the run directory also holds the driver's own log
    # and the per-arm launcher captures, none of which are training runs.
    for prec in ("bf16", "mxfp8"):
        for arm in ("mega", "baseline"):
            log = out / f"{prec}.{arm}.log"
            if log.exists():
                runs[f"{prec}.{arm}"] = Run(log, args.skip)
    if not runs:
        print(f"no arm logs under {out}", file=sys.stderr)
        return 1

    print("# MegaMoE vs baseline MoE, 1 node x 8 MI355X\n")
    print((out / "launch.txt").read_text().rstrip() if (out / "launch.txt").exists() else "")
    print("\n## Per-arm\n")
    print(f"Means over iterations >{args.skip} (JIT compile and warmup excluded).\n")
    print("| arm | iters | ms/iter | spread | TFLOP/s/GPU | max mem (GB) | loss@last |")
    print("|---|---|---|---|---|---|---|")
    for name, r in runs.items():
        if not r.ok:
            print(f"| {name} | 0 | — | — | — | — | **no iterations, run failed** |")
            continue
        last = max(r.loss) if r.loss else None
        print(
            f"| {name} | {len(r.timed)}/{len(r.rows)} | {r.mean_ms():.1f} | ±{r.spread() / 2:.1f}% | "
            f"{r.mean_tf():.1f} | {r.mem:.1f} | {r.loss[last]:.4f} @{last} |"
        )

    print("\n## Pairs\n")
    print("| precision | mega ms/iter | baseline ms/iter | speedup | mega TFLOP/s | baseline TFLOP/s |")
    print("|---|---|---|---|---|---|")
    for prec in ("bf16", "mxfp8"):
        m, s = runs.get(f"{prec}.mega"), runs.get(f"{prec}.baseline")
        if not (m and s and m.ok and s.ok):
            print(f"| {prec} | — | — | incomplete pair | — | — |")
            continue
        print(
            f"| {prec} | {m.mean_ms():.1f} | {s.mean_ms():.1f} | **{s.mean_ms() / m.mean_ms():.3f}x** | "
            f"{m.mean_tf():.1f} | {s.mean_tf():.1f} |"
        )

    for prec in ("bf16", "mxfp8"):
        m, s = runs.get(f"{prec}.mega"), runs.get(f"{prec}.baseline")
        if not (m and s and m.loss and s.loss):
            continue
        common = [k for k in sorted(m.loss) if k in s.loss]
        if not common:
            continue
        diffs = [s.loss[k] - m.loss[k] for k in common]
        print(f"\n### {prec} loss: baseline - mega\n")
        cols = common[::10] + ([common[-1]] if common[-1] not in common[::10] else [])
        print("| iteration | " + " | ".join(str(k) for k in cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        print("| mega | " + " | ".join(f"{m.loss[k]:.3f}" for k in cols) + " |")
        print("| baseline | " + " | ".join(f"{s.loss[k]:.3f}" for k in cols) + " |")
        print("| baseline-mega | " + " | ".join(f"{s.loss[k] - m.loss[k]:+.3f}" for k in cols) + " |")
        # Flips in the first iterations are counted separately: there the gap is ~0 and its sign is
        # float noise, which would otherwise read as "oscillating" on a curve that then diverges.
        settled = [s.loss[k] - m.loss[k] for k in common if k > args.skip]
        flips = sum(1 for a, b in zip(settled, settled[1:]) if (a > 0) != (b > 0))
        above = sum(1 for d in diffs if d > 0)
        print(
            f"\nbaseline above mega in {above}/{len(common)} iterations, "
            f"sign flips {flips}x after iteration {args.skip}, "
            f"final gap {diffs[-1]:+.4f}, max |gap| {max(abs(d) for d in diffs):.4f}, "
            f"mean |gap| {statistics.mean(abs(d) for d in diffs):.4f}. "
            f"Iteration {common[0]}: mega {m.loss[common[0]]:.5f} / baseline {s.loss[common[0]]:.5f}."
        )

    print("\n## Flags as the run actually saw them\n")
    print("| arm | use_turbo_mega_moe | precision | fp8 | fp8_recipe | seed |")
    print("|---|---|---|---|---|---|")
    for name, r in runs.items():
        f = r.flags
        print(
            f"| {name} | {f['use_turbo_mega_moe']} | {f['turbo_mega_moe_precision']} | "
            f"{f['fp8']} | {f['fp8_recipe']} | {f['seed']} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
