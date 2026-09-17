"""Plot the loss trajectories of one 2x2 MegaMoE-vs-baseline run directory.

Two panels rather than one: the curves alone hide the thing that matters, because at this scale a
0.7 gap and a 5.2 gap both look like "close to the others". The right panel plots baseline - mega
per precision, where a monotone curve means the two arms compute different functions and a curve
that crosses zero means quantization noise.

Pairs share a colour and the arm picks the line style, so a reader compares within a precision
without consulting the legend.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOSS = re.compile(r"iteration\s+(\d+)/\s*\d+ .*?lm loss:\s*([\d.E+-]+)")
STYLE = {
    ("bf16", "mega"): ("#1f77b4", "-", "bf16 · MegaMoE"),
    ("bf16", "baseline"): ("#1f77b4", "--", "bf16 · baseline"),
    ("mxfp8", "mega"): ("#d62728", "-", "mxfp8 · MegaMoE"),
    ("mxfp8", "baseline"): ("#d62728", "--", "mxfp8 · baseline"),
}


def traj(path: Path) -> dict[int, float]:
    text = ANSI.sub("", path.read_text(errors="replace"))
    return {int(i): float(v) for i, v in LOSS.findall(text)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = Path(args.out) if args.out else run / "loss_curves.png"

    curves = {k: traj(run / f"{k[0]}.{k[1]}.log") for k in STYLE if (run / f"{k[0]}.{k[1]}.log").exists()}
    if not curves:
        raise SystemExit(f"no arm logs under {run}")

    fig, (ax, axg) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})

    for key, c in curves.items():
        color, ls, label = STYLE[key]
        it = sorted(c)
        ax.plot(it, [c[i] for i in it], color=color, ls=ls, lw=1.9, label=label)
        ax.annotate(
            f"{c[it[-1]]:.2f}",
            (it[-1], c[it[-1]]),
            textcoords="offset points",
            xytext=(6, -3),
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("lm loss")
    ax.set_title("DeepSeek-V3, 4 layers, EP8, 1 node x 8 MI355X\nsame seed, same data, mock data")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    for prec in ("bf16", "mxfp8"):
        m, b = curves.get((prec, "mega")), curves.get((prec, "baseline"))
        if not (m and b):
            continue
        common = [i for i in sorted(m) if i in b]
        axg.plot(
            common,
            [b[i] - m[i] for i in common],
            color=STYLE[(prec, "mega")][0],
            lw=2.0,
            label=f"{prec}: baseline − MegaMoE",
        )
    axg.axhline(0, color="k", lw=0.8)
    axg.set_xlabel("iteration")
    axg.set_ylabel("loss gap (baseline − MegaMoE)")
    axg.set_title("Gap within each precision pair\nmonotone = different function, crossing = noise")
    axg.legend(frameon=False)
    axg.grid(alpha=0.25)

    meta = (run / "launch.txt").read_text().strip().replace("\n", "  |  ") if (run / "launch.txt").exists() else ""
    if meta:
        fig.text(0.5, 0.005, meta, ha="center", fontsize=6.5, color="#555")

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
