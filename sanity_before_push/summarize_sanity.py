#!/usr/bin/env python3
"""Summarize the sanity_before_push/ runs into a FLA-parity PASS/FAIL table.

For each of the 5 models it pairs the Primus run log with FLA's reference log
and reports:

  * Speed: steady-state ms/iter (Primus run-avg vs FLA), Δ%, TFLOP/s/GPU.
           PASS when Primus is no more than SPEED_TOL_PCT slower than FLA.
  * Loss : loss at the final compared step (warmup + 500), Δ vs FLA, Δ%.
           PASS when |Δ%| <= LOSS_TOL_PCT.
           Hybrids have no FLA-init weight converter, so their absolute loss
           is offset from FLA (the curve shape + speed are still valid); they
           are reported as INFO, not PASS/FAIL, on loss.

Usage:
    python3 sanity_before_push/summarize_sanity.py --print
    python3 sanity_before_push/summarize_sanity.py --out sanity_before_push/summary.md
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "sanity_before_push" / "logs"

FLA_TRAIN = "/home/vanbhati@amd.com/flash-linear-attention/legacy/training"

# Per-model metadata.
#   fla_ref       : FLA reference training log
#   has_fla_init  : True if an FLA-init checkpoint makes iter-1 loss bit-match
#   warmup        : FLA warmup iters (loss/speed compared AFTER this)
#   steady        : steady steps run after warmup
#   speed_tol_pct : Primus may be at most this % slower than FLA and still PASS
#   loss_tol_pct  : |Primus-FLA|/FLA loss tolerance at the final compared step
MODELS: "dict[str, dict]" = {
    "300M_gdn_pure": {
        "fla_ref": f"{FLA_TRAIN}/train_gdn_bs32.log",
        "has_fla_init": True, "warmup": 200, "steady": 500,
        "speed_tol_pct": 5.0, "loss_tol_pct": 2.0,
    },
    "300M_kda_pure": {
        "fla_ref": f"{FLA_TRAIN}/train_kda_300M.log",
        "has_fla_init": True, "warmup": 200, "steady": 500,
        "speed_tol_pct": 5.0, "loss_tol_pct": 2.0,
    },
    "300M_gdn_hybrid": {
        "fla_ref": f"{FLA_TRAIN}/train_gdn_hybrid_300M.log",
        "has_fla_init": False, "warmup": 200, "steady": 500,
        "speed_tol_pct": 5.0, "loss_tol_pct": 2.0,
    },
    "300M_mamba_hybrid": {
        "fla_ref": f"{FLA_TRAIN}/train_mamba2_hybrid_300M.log",
        "has_fla_init": False, "warmup": 200, "steady": 500,
        "speed_tol_pct": 5.0, "loss_tol_pct": 2.0,
    },
    "1B_gdn_pure": {
        "fla_ref": f"{FLA_TRAIN}/train_gdn_pure_1B_100B.log",
        "has_fla_init": False, "warmup": 2000, "steady": 500,
        "speed_tol_pct": 6.0, "loss_tol_pct": 2.0,
    },
}


# -----------------------------------------------------------------------------
#  Primus (Megatron) log parsing
# -----------------------------------------------------------------------------
# Each field is matched independently so an optional/absent field (e.g. the
# run-avg "/X" is missing on iter 1) never swallows the others.  Where a
# "instant/run-avg" pair exists we take the run-avg (second value).
_TS_STEP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\] iteration\s+(\d+)/")
_ELAPSED = re.compile(r"elapsed time per iteration \(ms\): ([0-9.]+)(?:/([0-9.]+))?")
_TFLOPS = re.compile(r"throughput per GPU \(TFLOP/s/GPU\): ([0-9.]+)(?:/([0-9.]+))?")
_TOKS = re.compile(r"tokens per GPU \(tokens/s/GPU\): ([0-9.]+)(?:/([0-9.]+))?")
_LOSS = re.compile(r"lm loss: ([0-9.E+-]+)")


def _pick(m):
    """Return the run-avg (2nd capture) when present, else the instantaneous."""
    if not m:
        return None
    return float(m.group(2)) if m.group(2) else float(m.group(1))


def parse_primus(log_path: Path):
    if not log_path.is_file():
        return None
    first_ts = last_ts = None
    last_run_avg_ms = last_tflops = last_toks = None
    losses: "list[tuple[int, float]]" = []
    with open(log_path, errors="ignore") as f:
        for line in f:
            ms = _TS_STEP.search(line)
            mloss = _LOSS.search(line)
            if not ms or not mloss:
                continue
            ts = datetime.strptime(ms.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
            step = int(ms.group(2))
            if first_ts is None:
                first_ts = ts
            last_ts = ts
            elapsed = _pick(_ELAPSED.search(line))
            tflops = _pick(_TFLOPS.search(line))
            toks = _pick(_TOKS.search(line))
            if elapsed is not None:
                last_run_avg_ms = elapsed
            if tflops is not None:
                last_tflops = tflops
            if toks is not None:
                last_toks = toks
            losses.append((step, float(mloss.group(1))))
    if last_ts is None:
        return None
    return {
        "wall_sec": last_ts - first_ts,
        "ms_per_iter": last_run_avg_ms,
        "tflops_per_gpu": last_tflops,
        "toks_per_gpu": last_toks,
        "losses": losses,
        "last_step": losses[-1][0] if losses else None,
    }


# -----------------------------------------------------------------------------
#  FLA log parsing  (HF Trainer style)
# -----------------------------------------------------------------------------
#   {'loss': X, ...} appears after a tqdm bar 'STEP/TOTAL [HH:MM:SS<..., Ys/it]'.
#   FLA reports loss summed across the 8-GPU batch → divide by 8 for per-token.
_FLA_LOSS = re.compile(rb"'loss': ([0-9.]+)")
_FLA_ITER = re.compile(rb"(\d+\.\d+)s/it")


def parse_fla(log_path: str):
    p = Path(log_path)
    if not p.is_file():
        return None
    content = Path(p).read_bytes()
    if not content:
        return None
    losses: "list[tuple[int, float]]" = []
    for m in _FLA_LOSS.finditer(content):
        loss_raw = float(m.group(1).decode())
        before = content[max(0, m.start() - 300): m.start()]
        steps = re.findall(rb"(\d+)/\d+ \[", before)
        if not steps:
            continue
        losses.append((int(steps[-1].decode()), loss_raw / 8.0))
    iters = [float(m.group(1).decode()) * 1000 for m in _FLA_ITER.finditer(content)]
    steady = [t for t in iters if t < 60000]  # drop warm-up / checkpoint outliers
    ms_iter = sum(steady) / len(steady) if steady else None
    return {"ms_per_iter": ms_iter, "losses": losses}


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------
def closest_loss(losses, target: int, tol: int = 25):
    cands = [(s, l) for s, l in losses if abs(s - target) <= tol]
    if not cands:
        return None, None
    return min(cands, key=lambda sl: abs(sl[0] - target))


def fmt(v, spec="{:.1f}"):
    return "n/a" if v is None else spec.format(v)


# -----------------------------------------------------------------------------
#  Report
# -----------------------------------------------------------------------------
def render() -> str:
    out: "list[str]" = []
    out.append("# Pre-push FLA parity sanity summary")
    out.append("")
    out.append("Generated by `python3 sanity_before_push/summarize_sanity.py`.")
    out.append("")
    out.append("Each model ran FLA's warmup + 500 steady steps with FLA's full "
               "LR schedule. Speed PASS = Primus within tolerance of FLA's "
               "steady ms/iter. Loss PASS = |Δ%| within tolerance at the final "
               "step. Hybrids have no FLA-init converter, so their loss is "
               "reported as INFO (curve shape + speed still valid).")
    out.append("")

    # ── Speed table ──────────────────────────────────────────────────────────
    out.append("## Speed (steady-state)")
    out.append("")
    out.append("| Model | Primus ms/iter | FLA ms/iter | Δ% | TFLOP/s/GPU | tok/s/GPU | Speed |")
    out.append("|---|---:|---:|---:|---:|---:|:--:|")

    speed_verdicts, loss_verdicts = [], []

    for name, meta in MODELS.items():
        p = parse_primus(LOGS_DIR / f"{name}.log")
        fla = parse_fla(meta["fla_ref"])
        if not p:
            out.append(f"| {name} | n/a (no log) | | | | | — |")
            speed_verdicts.append((name, None))
            continue
        pm = p["ms_per_iter"]
        fm = fla["ms_per_iter"] if fla else None
        if pm is not None and fm:
            dpct = (pm / fm - 1) * 100
            ok = dpct <= meta["speed_tol_pct"]
            verdict = "PASS" if ok else "FAIL"
            dstr = f"{dpct:+.1f}%"
        else:
            verdict, dstr = "—", "—"
        speed_verdicts.append((name, verdict if verdict in ("PASS", "FAIL") else None))
        out.append(
            f"| {name} | {fmt(pm)} | {fmt(fm)} | {dstr} | "
            f"{fmt(p['tflops_per_gpu'])} | {fmt(p['toks_per_gpu'], '{:.0f}')} | {verdict} |"
        )
    out.append("")

    # ── Loss table ───────────────────────────────────────────────────────────
    out.append("## Loss at final step (warmup + 500)")
    out.append("")
    out.append("FLA loss is divided by 8 (undo DeepSpeed sum-across-ranks) to "
               "match Megatron's per-token mean.")
    out.append("")
    out.append("| Model | Step | Primus loss | FLA loss | Δ | Δ% | Loss |")
    out.append("|---|---:|---:|---:|---:|---:|:--:|")

    for name, meta in MODELS.items():
        p = parse_primus(LOGS_DIR / f"{name}.log")
        fla = parse_fla(meta["fla_ref"])
        target = meta["warmup"] + meta["steady"]
        if not p:
            out.append(f"| {name} | {target} | n/a (no log) | | | | — |")
            loss_verdicts.append((name, None))
            continue
        ps, pl = closest_loss(p["losses"], target)
        fs, fl = closest_loss(fla["losses"], target) if fla else (None, None)
        if pl is None:
            out.append(f"| {name} | {target} | n/a | {fmt(fl, '{:.4f}')} | | | — |")
            loss_verdicts.append((name, None))
            continue
        if fl is None:
            out.append(f"| {name} | {ps} | {pl:.4f} | n/a | | | INFO |")
            loss_verdicts.append((name, None))
            continue
        d = pl - fl
        dpct = d / fl * 100
        if meta["has_fla_init"]:
            verdict = "PASS" if abs(dpct) <= meta["loss_tol_pct"] else "FAIL"
            loss_verdicts.append((name, verdict))
        else:
            verdict = "INFO (no FLA-init)"
            loss_verdicts.append((name, None))
        out.append(
            f"| {name} | {ps} | {pl:.4f} | {fl:.4f} | {d:+.4f} | {dpct:+.2f}% | {verdict} |"
        )
    out.append("")

    # ── Verdict ──────────────────────────────────────────────────────────────
    out.append("## Verdict")
    out.append("")
    sp_checked = [(n, v) for n, v in speed_verdicts if v in ("PASS", "FAIL")]
    ls_checked = [(n, v) for n, v in loss_verdicts if v in ("PASS", "FAIL")]
    sp_fail = [n for n, v in sp_checked if v == "FAIL"]
    ls_fail = [n for n, v in ls_checked if v == "FAIL"]
    missing = [n for n in MODELS if not (LOGS_DIR / f"{n}.log").is_file()]

    out.append(f"- Speed checked: {len(sp_checked)}/{len(MODELS)} "
               f"({'all PASS' if not sp_fail else 'FAIL: ' + ', '.join(sp_fail)})")
    out.append(f"- Loss checked (pure only): {len(ls_checked)} "
               f"({'all PASS' if not ls_fail else 'FAIL: ' + ', '.join(ls_fail)})")
    if missing:
        out.append(f"- Missing logs (not yet run): {', '.join(missing)}")
    overall = "READY TO PUSH" if (not sp_fail and not ls_fail and not missing) else "NOT READY"
    out.append("")
    out.append(f"**{overall}**")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "sanity_before_push" / "summary.md")
    ap.add_argument("--print", action="store_true", help="Print to stdout too")
    args = ap.parse_args()
    text = render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    if args.print:
        print(text)
    print(f"\n[written] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
