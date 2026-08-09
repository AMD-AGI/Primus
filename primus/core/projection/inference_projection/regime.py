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
