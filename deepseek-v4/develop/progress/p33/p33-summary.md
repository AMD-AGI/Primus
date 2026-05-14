# Plan-6 P33 — V4 TFLOP/s closed-form correction (SWA visible-pair + HC fn matmul)

> Phase summary written at P33 close. P33 is the plan-6 opener and is
> a **reporting-only fix**: no training runtime is touched, only the
> closed-form FLOP/s formula in
> `primus.backends.megatron.patches.deepseek_v4_flops_patches.compute_v4_flops`.

---

## 1. Objective

Plan-3 P20's `compute_v4_flops` was the canonical closed form for V4
TFLOP/s reporting since `deepseek-v4/develop/plan-3/02-phase-details.md`
shipped, but it carried two known gaps once plan-5 P30+ made the actual
attention kernels honour SWA pruning and once the V4 hybrid-layer HC
matmuls became measurable:

* **(a) SWA over-count** — the `attn_scores` term used Megatron's
  full-causal upper bound `B * n * d * S_eff^2` (== ``2 * n * d *
  S_eff^2/2`` after the FMA expansion) for the local branch.  Plan-5
  P30's SWA K-loop pruning made every dense / HCA local layer cost
  scale with the SWA-visible pair count instead of the full causal
  triangle; the over-count factor is
  ``S_eff / swa_window ≈ 128x`` at the V4-Flash proxy shape
  (``S_eff = S * hc_mult = 16384, swa_window = 128``).

* **(b) HyperConnection `fn.weight` matmul never counted** — the
  V4 hybrid layer runs two ``HyperMixer.fn`` projections per layer
  (``[B, S, K*D] → [B, S, (2+K)*K]``, ``K == hc_mult``) and one
  ``HyperHead.fn`` at the trunk end (and one per MTP depth).  None
  of these were in `compute_v4_flops`.

Together the two gaps inflate the reported TFLOP/s/GPU at the V4-Flash
EP=8 proxy from a real ~444 TFLOP/s/GPU to a reported ~1134 TFLOP/s/GPU
— a 2.55x optimistic headline.  P33 fixes both so plan-6 perf
comparisons (P34 GroupedMLP layout fusion, P35 RoPE Triton, P36
Sinkhorn Triton, …) measure progress against the honest denominator.

---

## 2. What changed

| component | path | change |
|---|---|---|
| visible-pair helper | `primus/backends/megatron/patches/deepseek_v4_flops_patches.py` | New `_visible_pairs` + `_local_visible_pairs` + `_pool_visible_pairs` closed forms.  `_local_visible_pairs` is the SWA-pruned local triangle (`swa * S - swa * (swa-1) // 2` when `0 < swa < S`, else full causal `S * (S+1) // 2`).  `_pool_visible_pairs` is the causal-visible HCA pool count `c * n * (n-1) // 2 + n * (S - c*n + 1)` with `n = S // c`.  Both match the proxy-shape numbers in `develop/perf/attention_perf.md` ("Test Shape And Counting"). |
| score FMAC | same | `_attn_scores_fmac_per_layer` rewritten as `2 * num_heads * head_dim * visible_pairs`.  SWA-aware via new `swa_window` kwarg threaded through `compute_v4_flops` from `args.attn_sliding_window`. |
| HC matmul | same | Two new helpers — `_hc_mixer_fmac_per_layer` (`2 * B * S * (K*D) * ((2+K)*K)` per layer; leading axis is `seq_len`, NOT `seq_len_eff`, because the mixer runs on the un-packed K streams) and `_hc_head_fmac` (`(1 + mtp_num_layers) * B * S * (K*D) * K`).  Wired into `compute_v4_flops` over every decoder + MTP layer, accumulated into the new `hc` field of `_V4FlopsBreakdown`. |
| breakdown log | same | `_emit_breakdown` now prints an `hc` row after `logits` so the per-component log line passes a grep-stability check.  `_V4FlopsBreakdown.total_fmac()` adds `self.hc` to the sum. |
| tests | `tests/unit_tests/backends/megatron/test_deepseek_v4_flops_patches.py` | (i) Updated G16 `_hand_attn_scores` to the new visible-pair convention so the patch and the hand-reference move together; (ii) **G36** — `TestG36SWAVisiblePairs` covers exhaustive sum-over-query parity, the three attention_perf.md proxy-shape numbers (`516160`, `516160 + 63520`, `516160 + 512*4096`), `swa=128` strict-monotone reduction vs `swa=0` on every `compress_ratio` × `hc_mult` cell, and the legacy-vs-new over-count ratio guard (`>= 100x` at proxy shape); (iii) **G36a** — `TestG36aHCMatmulAccounting` pins `br.hc` against the closed form `B * S * K^2 * D * (2*(L+M)*(2+K) + (1+M))` and verifies the per-axis ratios (doubling `seq_length` doubles `hc`; doubling `hc_mult` follows the K^2*(2+K)+K^2 scaling). |
| perf row | `deepseek-v4/develop/perf/proxy_ep8.md` | Adds a P33 row at the same iter time as P32 final (603.3 ms / iter) with the recomputed TFLOP/s/GPU (**444.2**, down from the legacy 1134.3) plus a long-form note explaining the two corrections and why the `vs baseline` column stays `14.64x` (iter-time speedup unchanged). |

---

## 3. Gates

| gate | status | numbers |
|---|---|---|
| **G36** SWA visible-pair | **GREEN** | `pytest -q tests/unit_tests/backends/megatron/test_deepseek_v4_flops_patches.py::TestG36SWAVisiblePairs` — covers `_visible_pairs` exhaustive sum parity across `swa ∈ {0, 64, 128, 4096, 8192}` and the three proxy-shape values from `attention_perf.md`. |
| **G36a** HC fn matmul | **GREEN** | `pytest -q tests/unit_tests/backends/megatron/test_deepseek_v4_flops_patches.py::TestG36aHCMatmulAccounting` — covers `hc_mult ∈ {2, 4, 8}` × `mtp_num_layers ∈ {0, 1, 2}` against the closed form and the per-axis scaling. |
| **G16 closed-form parity** | **GREEN (updated)** | Pre-existing G16 test class re-runs; `_hand_attn_scores` updated in lockstep so a regression in either side fails. |
| **No runtime change** | **N/A** | P33 is a reporting-only fix; no kernel / model / config code paths touched, so no proxy iter-time / loss ratchet needed.  The P32 final proxy run (`tas-mi355x-20260514/p32_final_postropefix_defaults`) remains the perf baseline going into P34. |

---

## 4. Performance delta

| metric | P32 final (legacy formula) | P33 (corrected formula) | delta |
|---|---:|---:|---:|
| iter time | 603.3 ms | 603.3 ms | unchanged (reporting-only) |
| total FLOP / iter (GBS=8) | 5468 TFLOP | 2144 TFLOP | **-60.8 %** |
| reported TFLOP/s/GPU | 1134.3 | **444.2** | -60.8 % |
| `vs P28 baseline` | 14.64x | 14.64x | unchanged (iter-time-based) |

Per-component breakdown at V4-Flash proxy shape
(`B=8, S=4096, hc_mult=4, S_eff=16384, swa=128, 8 layers, compress_ratios=[0,0,4,128,4,128,4,0]`):

| component | P32 final (legacy, TFLOP/iter) | P33 (corrected, TFLOP/iter) | note |
|---|---:|---:|---|
| `attn_qkv_o` | 672.9 | 672.9 | unchanged (projection FMAC only) |
| `attn_scores` | ~3500 | **138.3** | -25x; SWA prunes local pairs from `S_eff^2/2` to ~2.09M |
| `compressor` | 26.4 | 26.4 | unchanged |
| `indexer` | 88.4 | 88.4 | unchanged (proj + scoring at S_eff) |
| `moe` | 1112.4 | 1112.4 | unchanged |
| `mtp` (eh_proj) | 0.0 | 0.0 | (mtp_num_layers = 0 at proxy) |
| `logits` | 104.1 | 104.1 | unchanged |
| `hc` (new) | **0.0** | **1.25** | 2 HyperMixers × 8 layers + 1 HyperHead, all at seq_len (NOT seq_len_eff) |
| **TOTAL** | **5468.0** | **2143.8** | -60.8 % |

The 1.25 TFLOP/iter HC contribution is small in absolute terms but is
the only counted matmul whose cost is not at `seq_len_eff` — pinning
it now lets P37 (HyperConnection elemwise fusion) report a sensible
`hc`-relative speedup later.

---

## 5. Hand-off to P34

Plan-6 P34 (`PrimusTurboGroupedMLP._stack_grouped_linear_weight` Triton
FWD/BWD) is the next phase.  The P32 final trace shows
`hipMemcpyWithStream` at **289.6 ms / 32 calls** sourced from the two
`_stack_grouped_linear_weight` calls inside `PrimusTurboGroupedMLP`
(one per `gemm1` / `gemm2`).  P34 fuses those layout transforms into a
single Triton kernel that produces the
`[E, N_in, N_out]` packed weight directly from the
`num_local_experts` separate `[N_in, N_out]` tensors, removing the
intermediate `torch.stack` allocation entirely.  Expected proxy win:
~150-250 ms / iter at the V4-Flash EP=8 shape (rough upper bound
assuming the two memcpys account for ~280 ms of the 603 ms iter, and
~30 % of that is exposed not overlapped).

The honest TFLOP/s/GPU after P34 should land at ~570-630 against the
P33 denominator, vs. the optimistic ~1450-1600 the legacy formula
would have shown.
