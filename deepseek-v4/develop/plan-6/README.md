# Plan-6 — DeepSeek-V4 in Primus: elementwise / layout fusion + TFLOP/s correction

> Plan-6 picks up where plan-5 P32 final closed (`603 ms / iter, 1134 TFLOP/s/GPU` at V4-Flash EP=8 single-node proxy, `14.64×` over
> P28 baseline) and attacks the remaining elemwise / small-op tail
> that the P32 final chrome trace exposes. The post-P32 attention
> kernels (`_v4_attention_{fwd,bwd_dq,bwd_dkv}_kernel`,
> `_v4_csa_attention_pool_{sparse_fwd,sparse_bwd_partial}_kernel`)
> are out of scope here — they were the plan-5 deliverable and
> already meet their kernel budgets. Plan-6 is strictly scoped to
> the **elemwise / layout-transform tail** that is now the
> dominant share of `iter time − attention`:
>
> 1. **TFLOP/s closed-form correction** — `compute_v4_flops` in
> `primus/backends/megatron/patches/deepseek_v4_flops_patches.py`
> misses two terms that became material after plan-5 P30 SWA
> pruning + plan-5 P14 mHC streams: (a) SWA visible-pair pruning
> makes the `attn_scores` term over-count by 5-8× on dense and
> HCA layers; (b) the HyperConnection `fn.weight` matmul (twice
> per layer + once per HyperHead) was never counted.
> 2. **Six elemwise / layout fusion targets** ranked by trace-
> measured GPU time:
>   - `PrimusTurboGroupedMLP._stack_grouped_linear_weight` (single
>   biggest line item — `hipMemcpyWithStream` 289.6 ms / 32 calls
>   in the P32 final trace, almost half of iter wall time)
>   - `apply_interleaved_partial_rope` (cat + multiply + stack
>   chain, ~50 ms across `CatArrayBatchedCopy_contig` and
>   `elementwise_manual_unroll<128, 8>` buckets)
>   - HyperConnection `compute_weights` / `collapse` / `expand`
>   post-linear elemwise glue (sigmoid + scale + base + softmax
>   chain × 16 calls per iter)
>   - `sinkhorn_normalize` — plan-5 P29 wrapped it in
>   `torch.compile(fullgraph=True, dynamic=True)`; replace with
>   a hand-rolled Triton FWD/BWD that does the 20-iter alternating
>   row/col normalize in one kernel, no `torch.compile` cold-start
>   cost and no Inductor `CompiledFunctionBackward` overhead.
>   - `Indexer.forward` scoring chain (`einsum + relu + mul w + sum + causal_mask add` — keep `topk` as a separate op)
>   - V4 Router post-logits chain shared between
>   `DeepseekV4LearnedRouter` and `DeepseekV4HashRouter`
>   (`score_fn + topk + gather + denom + scatter` × 2 scatters)
>
> The "fuse everything elemwise" mandate is **measurement-driven**:
> each phase ships its Triton FWD/BWD kernel behind a single
> `PRIMUS_<NAME>_TRITON=1` env (default `1`, fallback to eager
> by setting to `0`). If a phase regresses end-to-end iter time —
> even when the standalone microbench wins — the env flag flips to
> default `0` for shipping and the regression goes into the phase's
> "failed / negative probes" section. Plan-5 P32 RoPE bf16 cast is
> the load-bearing precedent: microbench wins do not guarantee
> proxy wins, and the proxy A/B is the source of truth.

## Scope


| In scope                                                                                                                                                                                                                                                                                                         | Out of scope                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compute_v4_flops` (TFLOP/s patch) — add SWA visible-pair pruning + HyperConnection fn matmul terms; **no runtime change**, only the reported TFLOP/s number moves                                                                                                                                               | Re-deriving any other Megatron FLOPs term (logits / mtp / moe / compressor / indexer / qkv-o are already correct post-plan-3 P20)                                                                                                           |
| `_stack_grouped_linear_weight` Triton FWD/BWD fusion in `primus/backends/megatron/core/extensions/primus_turbo.py::PrimusTurboGroupedMLP` (single Triton kernel does stack + transpose + contiguous in one memcpy-with-permute; BWD scatters back to per-expert `weight{i}.grad`)                                | Refactoring grouped-MLP to allocate one contiguous `[E, K, N]` `nn.Parameter` up front (eliminates the stack entirely but changes state-dict + autograd surface; tracked as a plan-6 follow-up if the Triton fusion does not close the gap) |
| `apply_interleaved_partial_rope` Triton FWD/BWD in `primus/backends/megatron/core/transformer/dual_rope.py` (fuse `slice → reshape → 4 multiplies → stack → cat` into one kernel; partial-RoPE `nope` prefix copied through, `rotary_dim` suffix rotated in place)                                               | Re-deriving RoPE math (the plan-5 P32 RoPE bf16 cast fix is the canonical math; plan-6 only changes the kernel boundary)                                                                                                                    |
| `sinkhorn_normalize` Triton FWD/BWD in `primus/backends/megatron/core/transformer/hyper_connection.py` (replaces P29 `torch.compile` path; K=4 fits in registers, 20-iter row/col loop unrolled at compile time)                                                                                                 | Reducing `hc_sinkhorn_iters` from 20 (model-quality decision; out of scope per plan-5 P29 design note)                                                                                                                                      |
| HyperConnection elemwise Triton FWD/BWD for `compute_weights` post-linear (slice + scale + base + sigmoid + softmax), `collapse` (multiply + reduce), `expand` (outer-product + add); the GEMM inside `compute_weights._packed_logits` stays as-is (matmul is GPU-bound, not elemwise)                           | Rewriting the HyperConnection numerics (matmul / RMS-norm semantics unchanged)                                                                                                                                                              |
| `Indexer.forward` scoring Triton FWD/BWD — fuse `einsum(q_i, k_icomp) → relu → mul(w_i.unsqueeze) → sum(-2) → causal_mask add`; `topk` and the `where(isneginf, -1, ...)` + `pad cat` tail stay on host-side (rare branch)                                                                                       | Re-deriving the Indexer math (techblog §1.4) — fusion only changes the kernel boundary                                                                                                                                                      |
| V4 Router post-logits Triton FWD/BWD shared between `DeepseekV4LearnedRouter._compute_route` (`v4_topk_router.py`) and `DeepseekV4HashRouter.forward` (`v4_hash_router.py`); fuses `score_fn + [expert_bias add] + topk + gather + denom + scale + scatter (probs) + scatter (routing_map)` into a single kernel | `tid2eid[token_ids]` lookup in the hash router (host-side, cannot be fused without changing the parameter layout)                                                                                                                           |
| Per-phase unit tests under `tests/unit_tests/megatron/transformer/deepseek_v4/` (or `tests/unit_tests/megatron/extensions/` for P34) — FWD output parity vs eager + BWD gradcheck (fp32) + bf16 atol/rtol contract; release-tier shapes `pytest.mark.slow`                                                       | New gate categories — plan-6 reuses plan-4 / plan-5 ratchets and adds G36..G42 (one per phase)                                                                                                                                              |
| Per-phase microbench scripts under `deepseek-v4/develop/progress/p3X/bench_<name>.py` — standalone op timing on the EP8 shape                                                                                                                                                                                    | A new common bench harness (reuse plan-5 P31 / P32 conventions; each phase's bench is a small argparse wrapper following the same pattern)                                                                                                  |
| Per-phase EP8 proxy A/B trace + post-phase profile report under `develop/profile/profile-after-p3X-ep8-<YYYYMMDD>.{md,html}`                                                                                                                                                                                     | New profile-report tooling — plan-6 reuses `develop/profile/_tools/render_baseline_report.py` from plan-5 P28                                                                                                                               |
| `develop/perf/elem_fusion.md` (new file) — one row per phase, format `<ms> ms | <tflops or throughput>` per the standing R2.5 convention                                                                                                                                                                         | Re-formatting `attention_perf.md` / `proxy_ep8.md` (frozen plan-5 P31b-onwards)                                                                                                                                                             |


## Why each fusion is in plan-6 and not plan-5

Plan-5 was strictly scoped to **kernel performance** — the in-tree
V4 Triton attention kernels (P30 / P31 / P32) and the dominant
`aten::sum` fp32 reduce (P29 RESCOPED). The plan-5 P32 RoPE bf16
cast fix already unlocked the kernels' microbench wins to land in
the proxy, but it left the elemwise / layout tail intact. Plan-5
explicitly de-scoped small-op fusion at P28 close (CPU-bound floor
0.3 %, GPU 99.7 % active — the kernel-launch overhead rule said
"don't fuse"), but the post-P32 trace shows a different bottleneck
shape: the GPU is still 99.7 % active, but the elemwise tail moved
from being kernel-launch-overhead-bound to being HBM-bandwidth-bound
(`hipMemcpyWithStream` at 289 ms / 32 calls dominates, not the
kernel-launch tail). The plan-6 fusion targets are picked from the
P32 final trace, not from a-priori "small kernels look slow"
heuristic.

## Phase Map (added under Phase 32 in `../progress/status.md`)


| #       | Theme                                                                                                                      | Predecessor phase             |
| ------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **P33** | TFLOP/s closed-form correction — SWA visible-pair pruning + HyperConnection fn matmul; no runtime change                   | plan-3 P20 patch              |
| **P34** | `_stack_grouped_linear_weight` Triton FWD/BWD fusion (BIGGEST expected win, ~200 ms / iter on the EP8 proxy)               | plan-3 P23 grouped MLP wiring |
| **P35** | `apply_interleaved_partial_rope` Triton FWD/BWD fusion (per-q-k per-layer × 8 layers = 16 fused calls / iter)              | plan-5 P32 RoPE bf16 cast fix |
| **P36** | `sinkhorn_normalize` Triton FWD/BWD (replaces plan-5 P29 `torch.compile` path; eliminates `CompiledFunction*` overhead)    | plan-5 P29                    |
| **P37** | HyperConnection elemwise Triton fusion — `compute_weights` post-linear / `collapse` / `expand` outer-product               | plan-2 P14 mHC + plan-5 P29   |
| **P38** | `Indexer.forward` scoring Triton FWD/BWD — fuse `einsum + relu + mul + sum + causal_mask`; `topk` stays standalone         | plan-2 P14 Indexer            |
| **P39** | V4 Router post-logits Triton FWD/BWD — shared between learned (`v4_topk_router.py`) and hash (`v4_hash_router.py`) routers | plan-2 P14 routers            |
| **P40** | Plan-6 close-out — `elem_fusion.md` finalised, cumulative `proxy_ep8.md` row, status pinning, p3X-summary.md per R2.1      | plan-6 P33..P39               |


> **Caveat — plan-6 is trace-driven, not a-priori "small kernels
> look slow".** Each phase opens with a task-list-refinement pass
> against the P32 final trace + the post-previous-phase trace, and
> a phase may be **descoped or reordered** if its trace row is < 5 %
> of step time. The plan-5 P32 RoPE bug taught us that microbench
> wins do not guarantee proxy wins; the per-phase EP8 proxy A/B
> trace is the source of truth, and every fusion ships behind a
> default-on env that flips to default-off if the proxy A/B regresses.

## References

- Plan-5 P32 final state:
`../plan-5/02-phase-details.md` (P32 hand-off block) +
`../progress/status.md` (Phase 32 row) +
`../progress/p32/p32-summary.md` (RoPE bf16 cast addendum +
final-defaults perf table).
- Plan-5 P32 final EP8 trace (the input to every plan-6 phase):
`output/amd/tas-mi355x-20260514/p32_final_profile_ropefix_split_segreduce_pp1_ep8_seq4096/tensorboard/primus-megatron-exp_p32_final_trace_1778730863159449851.pt.trace.json.tgz`
- Plan-3 P20 V4 FLOPs patch (P33 amends this):
`primus/backends/megatron/patches/deepseek_v4_flops_patches.py`.
- Plan-5 P28 profile-report tooling (plan-6 reuses):
`develop/profile/_tools/render_baseline_report.py`.
- Plan-5 perf anchors:
  - `develop/perf/attention_perf.md` (frozen — plan-6 does not edit).
  - `develop/perf/proxy_ep8.md` (plan-6 appends one row per phase).

## Documents

- `[01-roadmap.md](./01-roadmap.md)` — phase overview, dependency
graph, milestones, exit criteria, top risks, out-of-scope.
- `[02-phase-details.md](./02-phase-details.md)` — phase-by-phase
task breakdown, design notes, edge cases.
- `[03-test-strategy.md](./03-test-strategy.md)` — gate matrix
(G36..G42), correctness ratchet (every plan-6 phase MUST keep
plan-4 G23 / G24 / G26 / G27 + plan-5 G32 / G34 / G34b / G35
green), and the perf-budget contract.

## Reporting hand-off

Plan-6 closes after P40 when every phase row in
`../progress/status.md` is checked, the post-P40 EP8 proxy trace
matches the per-phase trace predictions, `develop/perf/elem_fusion.md`
holds one row per shipped fusion, `proxy_ep8.md` holds the
`P40 final` row, and every plan-4 / plan-5 unit-test gate plus the
plan-6 G36..G42 gates are green at the final commit.

## Out of scope (plan-6)

- **Attention kernel work** — plan-5 owns V4 / CSA attention; plan-6
does not touch any `v4_attention_kernels/_triton/` file except
to plumb the RoPE Triton entry point.
- **FP8 / FP4 / mxfp4 quantised forward** — separate plan; plan-6
stays at BF16.
- **Reducing `hc_sinkhorn_iters`** — model-quality decision (plan-5
P29 design note).
- **State-dict layout change for grouped MLP** (`[E, K, N]` single
contiguous parameter) — tracked as a plan-6 follow-up only if the
P34 Triton-fuse path does not close the gap.
- **Convergence run / long-context / multi-node EP** — same as plan-5.
- **HF state-dict adapter** — plan-2 deferred to "Phase 22+".
