# 06 — Calibration (single-node, measured)

The projection is anchored to a real single-node run so its iteration time and
TFLOP/s line up with what Megatron reports.

## FLOPs: ported V4 closed-form (exact)

`tools/v4_flops.py` ports Megatron's `deepseek_v4_flops_patches` closed form.
Self-test against the measured flash 16-layer run (GBS64, seq4096):

| component | analytic (TFLOP/gb) | measured (TFLOP/gb) |
|---|---:|---:|
| attn_qkv_o | 10766.4 | 10766.4 |
| attn_scores | 2476.0 | 2476.0 |
| compressor | 527.8 | 527.8 |
| indexer | 1650.9 | 1650.9 |
| moe | 17838.5 | 17818.7 |
| logits | 832.9 | 833.7 |
| **TOTAL** | **34112** | **34093** (0.05%) |

The site uses these analytic FLOPs (per cr-layer, B=1, capture seq) for TFLOP/s,
matching Megatron's convention (fwd+bwd × FMA = 6×, recompute excluded). The
breakdown JSON carries them in `analytic_flops`.

## Iteration time: single-layer → full-model bias

Measured single-node anchor (`script/_calibrate_flash.sh`, full flash):

| knob | value |
|---|---|
| layers / cr | 16, cr=[0×3, 4×6, 128×7] |
| parallel | PP1 / EP8 / DP8 (world 8), TP1/CP1 |
| GBS / GA / MBS / seq | 64 / 8 / 1 / 4096 |
| recompute | full (uniform, 1) |
| optimizer | adam + distributed optimizer |

**Measured**: iter ≈ 6665 ms, 636 TFLOP/s/GPU, ~4917 tokens/s/GPU.

**Projection (raw, calibFactor=1.0)**: iter 7177 ms (+7.7%), 586 TFLOP/s/GPU
(−7.9%). The per-layer time captured from the single-layer profile runs ~7-8%
high vs a layer inside the full model (single-layer capture has no neighbour-
layer overlap / cache reuse, and per-launch overhead is a larger share). This is
a systematic, near-constant bias.

**Calibration**: a single `calibFactor = 0.93` on the pipeline compute time
brings it in line:

| metric | measured | projection (calibFactor 0.93) |
|---|---:|---:|
| iter time | 6665 ms | ~6680 ms (+0.2%) |
| TFLOP/s/GPU | 636 | ~630 (−1%) |
| tokens/s/GPU | 4917 | ~4900 (−0.4%) |

`calibFactor` is a site control (default 0.93).

## Caveats

- `calibFactor` is from one anchor (flash, 16L, PP1). Pro / other parallel
  layouts may want a slightly different value; re-anchor with another
  `_calibrate_*` run if precision matters.
- analytic FLOPs are evaluated at the capture seq (4096); changing seq in the
  UI does not re-derive them (attention FLOPs are seq-dependent).
- Optimizer step is analytic (per-rank params / HBM-BW); DP/PP comm assumed
  hidden (A2/A4).

## Reproduce

```bash
# measured anchor (single node, full model)
LAYERS=16 GBS=64 bash deepseek-v4/projection/script/_calibrate_flash.sh   # (helper; not committed)
# analytic flops self-test
python3 deepseek-v4/projection/tools/v4_flops.py
```
