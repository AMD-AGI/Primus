"""Regime-aware anchor selection for TP/EP restore.

A single low-GPU benchmark anchor extrapolates well to sharded targets *only*
when the model has no heavy single-GPU runtime overhead. When one GPU is
overhead-saturated (kernel launch / dispatch / scheduling across many
layers x experts), sharding relieves it **super-linearly** — per-GPU decode
throughput *rises* when you split the model. A bandwidth/compute roofline
assumes each GPU already runs at peak, so it cannot extrapolate that relief from
the saturated anchor (this is exactly the MiniMax 1-GPU -> TP2/EP2 gap).

Detection is a cheap measured probe, not an architecture heuristic: compare
per-GPU decode throughput at 1 GPU vs 2 GPUs. If it rises, the anchor must be
taken in-regime (>=2 GPUs). A measured-vs-sim "excess slope" was evaluated and
rejected — it false-positives on models whose sharding gives no relief.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# per-GPU decode throughput must rise by more than this (1 GPU -> 2 GPU) for a
# model to count as single-GPU-overhead-saturated. ~5% clears run-to-run noise.
SUPERLINEAR_THRESH = 1.05


def _decode_ms(path) -> tuple[dict, dict]:
    d = json.load(open(path))
    ms = {int(e["batch"]): float(e["decode_ms"]) for e in d["sweep"] if e.get("decode_ms")}
    return ms, d.get("meta", {})


def _per_gpu_tput(ms: dict, batch: int, gpus: int) -> Optional[float]:
    return batch * 1000.0 / ms[batch] / gpus if ms.get(batch) else None


def superlinear_relief(anchor_1gpu, anchor_2gpu, batch: int = 32) -> Optional[float]:
    """Per-GPU decode throughput ratio (2-GPU / 1-GPU) at ``batch``.

    >1 means sharding makes each GPU *faster* (heavy single-GPU overhead that a
    roofline restore from the 1-GPU anchor cannot capture). None if either
    anchor lacks the batch point.
    """
    m1, meta1 = _decode_ms(anchor_1gpu)
    m2, meta2 = _decode_ms(anchor_2gpu)
    g1 = int(meta1.get("tp", 1)) * int(meta1.get("pp", 1))
    g2 = int(meta2.get("tp", 1)) * int(meta2.get("pp", 1))
    t1 = _per_gpu_tput(m1, batch, g1)
    t2 = _per_gpu_tput(m2, batch, g2)
    return (t2 / t1) if (t1 and t2) else None


def select_anchor(target_gpus: int, anchor_1gpu, anchor_2gpu=None, batch: int = 32):
    """Choose the anchor to restore a sharded target from.

    Returns ``(anchor_path, in_regime, reason)``. ``in_regime`` is False only
    when a sharded target must fall back to an unverified 1-GPU anchor (no
    2-GPU probe available) — the caller should treat that restore as low
    confidence.
    """
    if target_gpus <= 1:
        return anchor_1gpu, True, "single-GPU target"
    if anchor_2gpu is None:
        return anchor_1gpu, False, "no 2-GPU probe; 1-GPU restore is unverified"
    r = superlinear_relief(anchor_1gpu, anchor_2gpu, batch)
    if r is not None and r > SUPERLINEAR_THRESH:
        return anchor_2gpu, True, f"super-linear relief {r:.2f}x -> in-regime 2-GPU anchor"
    rtxt = f"{r:.2f}x" if r is not None else "n/a"
    return anchor_1gpu, True, f"sub-linear ({rtxt}) -> 1-GPU anchor sufficient"


# --- Confidence ladder: climb GPU count until per-GPU decode throughput flattens
# The one-shot 1->2 probe above only checks the first doubling. A model can stay
# overhead-saturated past 2 GPUs (per-GPU throughput still rising at 2->4), so a
# 2-GPU anchor is *also* out-of-regime for a larger target. The ladder walks the
# available rungs (1,2,4,...) and returns the smallest anchor whose relief to the
# next rung has flattened to within CONF_TOL -- that anchor restores the target
# accurately with no measured floor / cap crutch. If no available rung has
# flattened, it recommends the next GPU count to benchmark.
CONF_TOL = 0.05  # per-GPU throughput ratio within +/-5% between rungs => in-regime


def relief_between(anchor_lo, anchor_hi, batch: int = 32) -> Optional[float]:
    """Per-GPU decode-throughput ratio ``hi/lo`` at ``batch`` for two rungs.

    Generalises :func:`superlinear_relief` to any adjacent GPU counts. ``>1``
    means the larger rung still extracts more per-GPU throughput (the smaller
    rung is still overhead-saturated / out-of-regime).
    """
    m_lo, meta_lo = _decode_ms(anchor_lo)
    m_hi, meta_hi = _decode_ms(anchor_hi)
    g_lo = int(meta_lo.get("tp", 1)) * int(meta_lo.get("pp", 1))
    g_hi = int(meta_hi.get("tp", 1)) * int(meta_hi.get("pp", 1))
    t_lo = _per_gpu_tput(m_lo, batch, g_lo)
    t_hi = _per_gpu_tput(m_hi, batch, g_hi)
    return (t_hi / t_lo) if (t_lo and t_hi) else None


def confidence_ladder(target_gpus: int, anchors_by_gpu: dict, batch: int = 32,
                      tol: float = CONF_TOL) -> dict:
    """Pick the cheapest in-regime anchor for ``target_gpus`` from a rung set.

    ``anchors_by_gpu`` maps ``gpu_count -> anchor_path`` (e.g.
    ``{1: a1, 2: a2, 4: a4}``). Walks rungs in ascending order and returns the
    smallest rung whose per-GPU throughput has flattened (relief to the next
    available rung within ``tol``): restoring the target from that rung needs no
    floor. Convergence is guaranteed because a rung ``>= target_gpus`` is exact.

    Returns a dict::

        {
          "anchor": <path or None>,     # chosen anchor (None if none available)
          "gpus": <int or None>,        # its GPU count
          "confidence": "high"|"low",   # high => trustworthy restore
          "converged": bool,            # a flattened (or >=target) rung was found
          "next_gpus": <int or None>,   # GPU count to benchmark next if not converged
          "relief": <float or None>,    # relief measured into the chosen rung
          "reason": str,
        }
    """
    rungs = sorted(g for g in anchors_by_gpu if anchors_by_gpu.get(g))
    if not rungs:
        return {"anchor": None, "gpus": None, "confidence": "low", "converged": False,
                "next_gpus": 1, "relief": None, "reason": "no anchors available"}

    if target_gpus <= 1:
        g = rungs[0]
        return {"anchor": anchors_by_gpu[g], "gpus": g, "confidence": "high",
                "converged": True, "next_gpus": None, "relief": None,
                "reason": "single-GPU target"}

    # A measured rung at or above the target is exact (interpolation, not restore).
    at_or_above = [g for g in rungs if g >= target_gpus]
    if at_or_above:
        g = min(at_or_above)
        return {"anchor": anchors_by_gpu[g], "gpus": g, "confidence": "high",
                "converged": True, "next_gpus": None, "relief": None,
                "reason": f"anchor at {g} GPUs >= target {target_gpus} (exact)"}

    # A rung ``g`` is *verified in-regime* when the per-GPU throughput is FLAT
    # across the adjacent measured pair ending at ``g`` (relief within +/-tol on
    # BOTH sides -- rising is super-linear overhead the roofline can't see;
    # falling is a sharding penalty the roofline over-credits). A flat curve up
    # to ``g`` may be extrapolated ONE doubling, so ``g`` certifies a restore only
    # when ``2*g >= target`` -- i.e. the flat evidence sits within one doubling of
    # the target. A flat pair far below the target (e.g. gpt-oss 1->2 flat but the
    # EP all-to-all collapses by 8 GPUs) is NOT trusted; the ladder climbs instead.
    verified = []  # (g, relief) where the pair (prev, g) is flat
    for i in range(1, len(rungs)):
        r = relief_between(anchors_by_gpu[rungs[i - 1]], anchors_by_gpu[rungs[i]], batch)
        if r is not None and abs(r - 1.0) <= tol:
            verified.append((rungs[i], r))

    certifying = [(g, r) for (g, r) in verified if g * 2 >= target_gpus]
    if certifying:
        g, r = min(certifying, key=lambda t: t[0])
        return {"anchor": anchors_by_gpu[g], "gpus": g, "confidence": "high",
                "converged": True, "next_gpus": None, "relief": r,
                "reason": f"per-GPU throughput flat within one doubling of target "
                          f"(relief into {g} GPUs {r:.2f}x, {g}*2>={target_gpus}) -> "
                          f"{g}-GPU anchor certifies the restore"}

    # Not certified: either no flat adjacent pair, or the only flat pair is >1
    # doubling below the target. Climb -- benchmark the next rung toward target.
    top = rungs[-1]
    r_top = relief_between(anchors_by_gpu[rungs[-2]], anchors_by_gpu[top], batch) if len(rungs) >= 2 else None
    nxt = min(target_gpus, top * 2)
    rtxt = f"{r_top:.2f}x" if r_top is not None else "n/a"
    if verified:
        why = (f"flat only at {max(g for g, _ in verified)} GPUs, >1 doubling below "
               f"target {target_gpus} (high-scale comm/imbalance unprobed)")
    else:
        why = f"no flat adjacent pair yet (relief into top rung {rtxt})"
    return {"anchor": anchors_by_gpu[top], "gpus": top, "confidence": "low",
            "converged": False, "next_gpus": nxt, "relief": r_top,
            "reason": f"{why}; benchmark at {nxt} GPUs to converge"}


def climb_anchor_ladder(target_gpus: int, bench_fn, max_gpus: Optional[int] = None,
                        batch: int = 32, tol: float = CONF_TOL, start_gpus: int = 1) -> dict:
    """Iteratively benchmark rungs 1,2,4,... until confidence is high.

    ``bench_fn(gpus) -> anchor_path`` runs (or fetches) the decode benchmark at
    ``gpus`` GPUs and returns its anchor file. The loop doubles the GPU count,
    re-evaluates :func:`confidence_ladder`, and stops when it converges or hits
    ``max_gpus``/``target_gpus``. Returns the final ``confidence_ladder`` dict
    augmented with ``"rungs_measured"``.
    """
    cap = min(target_gpus, max_gpus) if max_gpus else target_gpus
    anchors: dict = {}
    g = max(1, start_gpus)
    rungs_measured = []
    while True:
        if g not in anchors:
            path = bench_fn(g)
            if path:
                anchors[g] = path
                rungs_measured.append(g)
        verdict = confidence_ladder(target_gpus, anchors, batch, tol)
        verdict["rungs_measured"] = list(rungs_measured)
        if verdict["converged"] or g >= cap:
            return verdict
        g = min(cap, g * 2)
