#!/usr/bin/env python
"""Phase 5 probe: FP4 CSA-indexer QK selection-quality + grad-flow check.

Runs the V4 Indexer in BF16 vs MXFP4-QK (PRIMUS_INDEXER_FP4) on identical
weights/inputs and reports:
  * top-k selection OVERLAP (paper's quality metric: do we pick the same
    compressed positions?), per query, averaged.
  * score cosine (numerical drift of I_{t,s}).
  * STE backward sanity (grads finite, non-zero through the FP4 quant).

Usage:
  PRIMUS_INDEXER_FP4=0 not needed — the probe toggles the env itself.
  python phase5_fp4_indexer_probe.py
"""
import os

import torch

# Import after we can toggle env per-call.
from primus.backends.megatron.core.transformer import indexer as idx_mod
from primus.backends.megatron.core.transformer.indexer import Indexer

torch.manual_seed(0)

DEV = "cuda"
DT = torch.bfloat16

B, S, D = 2, 2048, 2048
COMPRESS_RATIO = 4
INDEX_HEAD_DIM = 128
INDEX_N_HEADS = 64
INDEX_TOPK = 64


def build():
    torch.manual_seed(0)
    m = Indexer(
        hidden_size=D,
        index_head_dim=INDEX_HEAD_DIM,
        index_n_heads=INDEX_N_HEADS,
        index_topk=INDEX_TOPK,
        compress_ratio=COMPRESS_RATIO,
    ).to(DEV, DT)
    return m


def run(m, hidden, fp4: bool):
    os.environ["PRIMUS_INDEXER_FP4"] = "1" if fp4 else "0"
    assert idx_mod._indexer_fp4_enabled() is fp4
    h = hidden.clone().requires_grad_(True)
    topk_idxs, topk_scores = m(h)
    return topk_idxs, topk_scores, h


def main():
    assert torch.cuda.is_available(), "needs MI355X (gfx950) for MXFP4"
    m = build()
    hidden = torch.randn(B, S, D, device=DEV, dtype=DT) * 0.5

    idx_bf16, sc_bf16, _ = run(m, hidden, fp4=False)
    idx_fp4, sc_fp4, h_fp4 = run(m, hidden, fp4=True)

    # ---- selection overlap (ignore -1 padding/masked slots) ----
    valid = idx_bf16 >= 0  # [B,S,K]
    overlaps = []
    Bn, Sn, K = idx_bf16.shape
    a = idx_bf16.reshape(-1, K)
    b = idx_fp4.reshape(-1, K)
    v = valid.reshape(-1, K)
    for i in range(a.shape[0]):
        sa = set(a[i][v[i]].tolist())
        if not sa:
            continue
        sb = set(b[i][b[i] >= 0].tolist())
        overlaps.append(len(sa & sb) / len(sa))
    overlap = sum(overlaps) / max(len(overlaps), 1)

    # ---- score cosine on valid (finite) entries ----
    fa = sc_bf16.reshape(-1).float()
    fb = sc_fp4.reshape(-1).float()
    fin = torch.isfinite(fa) & torch.isfinite(fb)
    cos = torch.nn.functional.cosine_similarity(fa[fin], fb[fin], dim=0).item()

    # ---- STE backward sanity through the FP4 quant ----
    loss = sc_fp4[torch.isfinite(sc_fp4)].sum()
    loss.backward()
    g = h_fp4.grad
    grad_ok = bool(torch.isfinite(g).all()) and g.abs().sum().item() > 0

    print(f"[phase5] shapes: idx {tuple(idx_bf16.shape)}  P={S // COMPRESS_RATIO}  topk={INDEX_TOPK}")
    print(f"[phase5] top-k selection overlap (FP4 vs BF16): {overlap:.4f}")
    print(f"[phase5] score cosine (FP4 vs BF16):            {cos:.4f}")
    print(f"[phase5] STE grad through FP4 QK finite+nonzero: {grad_ok}")
    # Selection quality is the gate; the indexer only needs the same top-k.
    print(f"[phase5] VERDICT: {'PASS' if overlap > 0.9 and grad_ok else 'CHECK'}")


if __name__ == "__main__":
    main()
