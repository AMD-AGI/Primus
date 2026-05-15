# Plan-8 — Tilelang-backed V4 attention kernels (cr ∈ {0, 4, 128})

Plan-8 re-implements the three V4 attention families using
[tilelang](../../../tilelang/) (a tile-level DSL on top of TVM) and
ships them behind a new dispatcher knob (`use_v4_tilelang_attention`)
that defaults OFF until each kernel's release-tier parity gate +
EP=8 proxy A/B confirm a positive delta vs the plan-4 P25 / P26
Triton kernels.

The three compress-ratio families correspond to V4-Flash's
`compress_ratios = [0, 0, 4, 128, 4, 128, 4, 0]` layer schedule:

| compress_ratio | layer kind | kernel today | tilelang target |
|---:|---|---|---|
| `0` | dense (full attention + per-head sink + optional SWA) | `_triton/v4_attention_fwd.py` + `_triton/v4_attention_bwd.py` (plan-4 P25 + plan-5 P32 split BWD) | **P50 / P51** |
| `128` | HCA — local SWA + compressed-pool with joint softmax + sink | same Triton kernel above, called twice in split-mask mode | **P52 / P53** |
| `4` | CSA — local SWA + sparse top-K compressed-pool gather + sink | `_triton/v4_csa_attention_fwd.py` + `_triton/v4_csa_attention_bwd.py` (plan-4 P26 + plan-5 P31b split BWD + plan-5 P32 sparse pool + segreduce) | **P54 / P55** |

## At a glance

| phase | scope | rough est savings vs Triton |
|---|---|---:|
| **P49** | Tilelang infra (env knob `PRIMUS_V4_TILELANG_ATTN`, dispatcher, build / autotune cache layout, ratchet check) | enables P50–P55 |
| **P50** | Dense FWD (cr=0) tilelang kernel + G50 parity + microbench | -5 to -10 ms / iter |
| **P51** | Dense BWD (cr=0) — dQ / dKV split à la plan-5 P32 + G51 parity | -10 to -15 ms / iter |
| **P52** | HCA FWD (cr=128) — split-mask local SWA + pool branch + G52 parity | -3 to -5 ms / iter |
| **P53** | HCA BWD (cr=128) + G53 parity | -3 to -5 ms / iter |
| **P54** | CSA FWD (cr=4) — local SWA + sparse top-K + sink fused + G54 parity | -5 to -8 ms / iter |
| **P55** | CSA BWD (cr=4) — gather + atomic_add or segreduce, mirroring plan-5 P32 + G55 parity | -10 to -15 ms / iter |
| **P56** | Plan-8 close-out — cumulative EP=8 proxy bake-off + perf docs + status pinning | -- |

**End-of-plan-8 EP=8 proxy steady-iter target: ≤ 470 ms / iter, ≥
570 TFLOP/s/GPU** (P33-corrected denominator) — ~`1.09×` over the
plan-7 P48 anchor of 510.6 ms.  The target is **best-effort, not
a contract**; any phase that regresses vs the corresponding Triton
kernel ships with its env default flipped to `0` (R9.1 precedent).

## Why tilelang

The plan-4 P25 / P26 Triton kernels are mature and at the
**single-row sparse-tile** + **WMMA-aware** local optimum for
the V4 shape envelope (head_dim=512, MQA, joint-softmax sink).
Tilelang opens three concrete wins the Triton form cannot reach:

1. **MFMA scheduling control** — `tl.gemm(...)`'s `policy=GemmWarpPolicy.FullRow`
   + `k_pack` hints map directly onto MI355's MFMA 16×16×16 /
   32×32×8 schedules.  The current Triton kernel's `tl.dot` chooses
   schedules per-shape and lands on suboptimal layouts at
   `head_dim=512` (see plan-5 P31b BWD split notes).
2. **First-class pipelining** — `T.Pipelined(loop, num_stages=N)`
   gives explicit control over the load-compute pipeline depth.
   The plan-4 P25 kernel uses Triton's implicit autotune; tilelang
   moves this into the per-kernel autotuning loop.
3. **Shared-memory layout primitives** — `T.alloc_shared` +
   `T.alloc_fragment` + `T.use_swizzle(panel_size, ...)` expose
   the LDS swizzle pattern as a tunable knob; the existing
   Triton kernel pays for default LDS bank conflicts at
   `head_dim=512`.

The `tilelang/examples/amd/example_amd_flash_attn_{fwd,bwd}.py`
reference (MI355X-tuned FlashAttention v2 with autotune) and the
`tilelang/examples/attention_sink/example_mha_sink_fwd_bhsd.py`
(sink + SWA together) are the load-bearing borrowable references.
For CSA the `tilelang/examples/dsa_sparse_finetune/sparse_mla_{fwd,bwd}.py`
+ `tilelang/examples/deepseek_v4/sparse_attn_fwd_sm90.py` cover
the sparse-gather pattern.

## Out of scope (plan-8)

- **Optimizer-step fusion** — plan-7's territory; the plan-7 P45
  prototype (multi_tensor_add) is the seed for a future plan-9
  fused-Adam Triton kernel.
- **Model-arch change / new compress ratios** — plan-8 stays at the
  V4-Flash `[0, 4, 128]` schedule.
- **FP8 / FP4 / mxfp4** — separate plan.
- **Long-context (1M-token) / multi-node EP** — same as plan-5/6/7.
- **HF state-dict adapter** — same as plan-5/6/7.

## Files

- `01-roadmap.md` — phase overview, dependency graph, milestones, top risks.
- `02-phase-details.md` — per-phase task breakdown + design notes.
- `03-test-strategy.md` — gate matrix (G50..G56).
