#!/usr/bin/env python3
"""Three-way mamba2 loss comparison to prove the cause of the +10% offset.

Columns:
  FLA ref            : FLA's mamba2_300M_hybrid.json run (loss summed over 8
                       DeepSpeed ranks -> divided by 8 to match Megatron mean).
  Primus production  : zebra_llama_300M_mamba_hybrid (GDN-matched arch, 12 MLPs).
  Primus FLA-exact   : zebra_llama_300M_mamba_hybrid_flaexact (this experiment;
                       byte-for-byte replica of FLA's arch, 3 MLPs).

Interpretation:
  - production  vs FLA  ~= +10%   (the offset we are explaining)
  - FLA-exact   vs FLA  ~= +1-2%  (no-FLA-init band, same as GDN hybrid)
    => proves the offset is architecture/hyperparameter, not a bug.
"""
import re
from pathlib import Path

FLA_REF = Path(
    "/home/vanbhati@amd.com/flash-linear-attention/legacy/training/"
    "train_mamba2_hybrid_300M.log"
)
PROD = Path(
    "/home/vanbhati@amd.com/Primus/sanity_before_push/logs/300M_mamba_hybrid.log"
)
FLAEXACT = Path(
    "/home/vanbhati@amd.com/Primus/sanity_before_push/mamba_parity/logs/"
    "300M_mamba_hybrid_flaexact.log"
)
FLAINIT = Path(
    "/home/vanbhati@amd.com/Primus/sanity_before_push/mamba_parity/logs/"
    "300M_mamba_hybrid_flaexact_flainit.log"
)

MILESTONES = [10, 100, 200, 300, 400, 500, 600, 700]

_PRIMUS_STEP = re.compile(r"iteration\s+(\d+)/")
_PRIMUS_LOSS = re.compile(r"lm loss: ([0-9.E+-]+)")
_FLA_LOSS = re.compile(rb"'loss': ([0-9.]+)")
_PARAMS = re.compile(r"number of parameters on .*?: (\d+)")


def parse_primus(path: Path):
    """step -> per-token mean loss."""
    out = {}
    if not path.is_file():
        return out
    with open(path, errors="ignore") as f:
        for line in f:
            ms, ml = _PRIMUS_STEP.search(line), _PRIMUS_LOSS.search(line)
            if ms and ml:
                out[int(ms.group(1))] = float(ml.group(1))
    return out


def primus_params(path: Path):
    if not path.is_file():
        return None
    with open(path, errors="ignore") as f:
        for line in f:
            m = _PARAMS.search(line)
            if m:
                return int(m.group(1))
    return None


def parse_fla(path: Path):
    """Map each FLA 'loss' dict to the step from its preceding tqdm bar
    (logging-interval agnostic). FLA sums loss over 8 ranks -> divide by 8."""
    out = {}
    if not path.is_file():
        return out
    content = path.read_bytes()
    for m in _FLA_LOSS.finditer(content):
        before = content[max(0, m.start() - 300):m.start()]
        steps = re.findall(rb"(\d+)/\d+ \[", before)
        if steps:
            out[int(steps[-1])] = float(m.group(1)) / 8.0
    return out


def nearest(d, step):
    if step in d:
        return step, d[step]
    cands = [s for s in d if abs(s - step) <= 5]
    if not cands:
        return None, None
    s = min(cands, key=lambda s: abs(s - step))
    return s, d[s]


def pct(a, b):
    return f"{(a - b) / b * 100:+.2f}%" if (a is not None and b) else "—"


def fmt(x):
    return f"{x:.4f}" if x is not None else "n/a"


def main():
    fla = parse_fla(FLA_REF)
    prod = parse_primus(PROD)
    exact = parse_primus(FLAEXACT)
    flainit = parse_primus(FLAINIT)

    print("\n=== Mamba2 loss-parity experiment ===\n")
    print("Param counts (lower-is-not-better; this is about WHERE params go):")
    print(f"  FLA mamba2 ref          : 301,317,922  (hidden 1216, state 128, n_groups 1, 3 MLPs)")
    pp = primus_params(PROD)
    pe = primus_params(FLAEXACT)
    print(f"  Primus production       : {pp:>11,}  (hidden 1024, state 64,  n_groups 8, 12 MLPs)"
          if pp else "  Primus production       : n/a")
    print(f"  Primus FLA-exact (this) : {pe:>11,}  (hidden 1216, state 128, n_groups 1, 3 MLPs)"
          if pe else "  Primus FLA-exact (this) : n/a  (run not finished)")

    have_init = bool(flainit)
    print("\nLoss at milestones (FLA divided by 8):\n")
    if have_init:
        hdr = (f"{'step':>5} | {'FLA ref':>9} | {'prod':>8} | {'prodΔ%':>7} | "
               f"{'exact':>8} | {'exactΔ%':>7} | {'FLA-init':>8} | {'initΔ%':>7}")
    else:
        hdr = (f"{'step':>5} | {'FLA ref':>9} | {'Primus prod':>11} | {'prod Δ%':>8} | "
               f"{'FLA-exact':>9} | {'exact Δ%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in MILESTONES:
        _, fv = nearest(fla, s)
        _, pv = nearest(prod, s)
        _, ev = nearest(exact, s)
        if have_init:
            _, iv = nearest(flainit, s)
            print(f"{s:>5} | {fmt(fv):>9} | {fmt(pv):>8} | {pct(pv, fv):>7} | "
                  f"{fmt(ev):>8} | {pct(ev, fv):>7} | {fmt(iv):>8} | {pct(iv, fv):>7}")
        else:
            print(f"{s:>5} | {fmt(fv):>9} | {fmt(pv):>11} | {pct(pv, fv):>8} | "
                  f"{fmt(ev):>9} | {pct(ev, fv):>8}")

    print("\nVerdict:")
    _, fv = nearest(fla, 700)
    _, ev = nearest(exact, 700)
    # iter-1 sanity: with correct FLA-init, iter-10 loss must match FLA closely.
    if have_init:
        _, fi10 = nearest(fla, 10)
        _, ii10 = nearest(flainit, 10)
        if ii10 is not None and fi10:
            print(f"  [mapping check] FLA-init iter-10 = {ii10:.4f} vs FLA {fi10:.4f} "
                  f"({pct(ii10, fi10)}) — should be ~0% if the converter (incl. MLA) is correct.")
        _, iv = nearest(flainit, 700)
        if iv is not None and fv:
            d = abs((iv - fv) / fv * 100)
            if d <= 2.5:
                print(f"  FLA-init lands within {d:.2f}% of FLA at step 700.")
                print("  => PROVEN: the offset was INITIALIZATION; the Mamba2 mixer is correct.")
            else:
                print(f"  FLA-init still off by {d:.2f}% at step 700 with FLA's exact weights.")
                print("  => If the iter-10 mapping check above is ~0%, this is a REAL Mamba2 "
                      "mixer/numerics bug. If iter-10 is off, fix the converter (likely MLA) first.")
        return
    if ev is None:
        print("  FLA-exact run not finished yet — rerun after it reaches step 700.")
    elif fv:
        d = abs((ev - fv) / fv * 100)
        if d <= 2.5:
            print(f"  FLA-exact replica is within {d:.2f}% of FLA at step 700 (no-FLA-init band).")
            print("  => the offset is architecture/hyperparameter.")
        else:
            print(f"  FLA-exact replica still off by {d:.2f}% at step 700 (> no-init band).")
            print("  => Config does NOT explain it. Run the FLA-init test "
                  "(run_mamba_parity.sh flainit) to separate init vs mixer bug.")


if __name__ == "__main__":
    main()
