# Plan-5 — DeepSeek-V4 in Primus: end-to-end performance optimisation

> Plan-5 picks up where plan-4 closed (in-tree V4 Triton kernels for
> dense / HCA / CSA attention shipped behind `use_v4_triton_attention`
> and `use_v4_triton_csa_attention`, both default `True` after the
> 2026-05-08 `run_deepseek_v4.sh` flip; turbo DeepEP + turbo grouped
> GEMM also default-on for the EP8 single-node smoke). Plan-5 is
> **strictly scoped** to one outcome:
>
> Take the V4-Flash EP=8 single-node training step from its current
> 17 TFLOP/s/GPU steady-state (plan-4 P27 G30 smoke at
> `seq_length=128`) up the throughput curve at production-shape
> sequence lengths, by attacking the bottlenecks visible in a real
> torch.profiler trace — first the kernel-launch / small-op tail that
> dominates CPU idle time, then the V4 Triton attention kernels
> themselves once the small-op tail is gone.
>
> Why now: plan-4 G30 confirmed the kernels are numerically correct
> and the pipeline is end-to-end stable, but G30 ran at
> `seq_length=128` to fit under the kernel-launch overhead floor.
> At production `seq_length` (target `4096` for V4-Flash) the
> attention quadratic + the per-layer small-op chain (single-latent
> KV projections, per-head q_rms, partial-RoPE, grouped low-rank O,
> Compressor / Indexer, MoE router + permute) collide in a way that
> the smoke-time profile cannot surface. Plan-5 ships the proxy
> training config that lets us measure that collision, the baseline
> trace, and the optimisation phases that pay off the trace.

## Scope

| In scope | Out of scope |
| --- | --- |
| `run_deepseek_v4_flash_proxy.sh` — V4-Flash proxy (8 layers, full V4-Flash widths, all four perf knobs on: `use_v4_triton_attention`, `use_v4_triton_csa_attention`, `use_turbo_deepep`, `use_turbo_grouped_mlp`) tuned to fit one MI355X node at EP=8 | FP8 / FP4 / mxfp4 quantised forward (separate plan; numerics-first, not perf-first) |
| EP=8 baseline torch.profiler trace at the proxy config + a baseline analysis report (Markdown + HTML) under `deepseek-v4/develop/profile/profile-baseline-*` | Convergence run + long-context (1M-token) bring-up + multi-node EP scaling smokes (separate plan) |
| Small-op fusion (P29) — torch.compile and / or hand-written Triton fusions for the attention pre / post-projection chain, Compressor / Indexer, partial-RoPE, MoE router gather / scatter — concrete fusion targets get picked from the P28 trace | HF state-dict adapter, V4-Pro release-tier perf, V3 / V2 backports of the Triton fusions |
| V4 Triton attention kernel perf tuning (P30) — autotune `BLOCK_M / BLOCK_N / num_warps / num_stages` per-shape, persistent kernel for FWD, in-kernel SWA, HCA LSE-merge variant (was a plan-4 follow-up) | Re-implementing V4 attention as TE / aiter callouts (plan-4 owns that fallback path; plan-5 only optimises the in-tree Triton kernels) |
| V4 Triton CSA kernel perf tuning (P31) — in-kernel `topk_idxs` gather to drop the wrapper-side `[B, H, Sq, K, D]` materialisation, better K-tile prefetching, dense local + sparse head-block BWD split | A new CSA kernel design (radically different math); plan-5 stays in the per-row design that plan-4 P26 shipped |
| Operator-microbenchmark-driven attention kernel speed-ups (P32) — `bench_v4_attention_ep8.py` + targeted kernel rewrites to hit CSA FWD ≤ 6 ms, V4 attention BWD ≤ 15 ms, CSA BWD ≤ 15 ms on the proxy EP8 shape | Algorithmic changes to attention (e.g. different softmax / sparsity), FP8 / FP4 numerics work, autograd-graph rewriting outside the V4 attention modules |

## Why each phase is in plan-5 and not plan-4

Plan-4 was strictly scoped to **kernel correctness** at production V4
shapes; the perf bar plan-4 cleared was "don't regress the eager
baseline" (plan-4 P27 G30 closed at +37 % vs the P22 eager baseline
at `seq_length=128`). Plan-5 owns everything that requires the trace
as input — fusion candidate selection and per-shape autotune — because
none of that can be picked correctly without measurement, and the
measurement infrastructure (the proxy + the baseline report) is itself
plan-5's first deliverable.

## Phase Map (added under Phase 27 in `../progress/status.md`)

| #       | Theme                                                                                                          | Plan-4 leftovers folded in |
| ------- | -------------------------------------------------------------------------------------------------------------- | -------------------------- |
| **P28** | V4-Flash proxy script + EP=8 baseline trace + bottleneck analysis report (md + html in `develop/profile/profile-baseline-*`) | (none — kick-off phase)    |
| **P29** | Small-op fusion targets picked from the P28 trace (torch.compile and / or in-tree Triton); attention pre / post-projection chain + Compressor / Indexer + MoE router as the prime candidates | P22 follow-up: "the Compressor / Indexer / partial-RoPE small-op chain runs as eager Python on every layer" |
| **P30** | V4 Triton attention kernel perf tuning — per-shape autotune, persistent kernel, in-kernel SWA, HCA LSE-merge variant | Plan-4 P25 follow-up: "HCA single-kernel-with-additive-bias is simpler and good enough for plan-4; LSE-merge is a future perf optimisation" |
| **P31** | V4 Triton CSA kernel perf tuning — in-kernel `topk_idxs` gather + tile / prefetch tuning + split CSA BWD | Plan-4 P26 follow-up: "wrapper-side gather materialises 2–4 GiB / microbatch; in-kernel `topk_idxs`-driven `tl.load` is left for a future perf plan" |
| **P32** | Operator-microbench-driven attention kernel speed-ups — `bench_v4_attention_ep8.py` lands; CSA FWD multi-row tile + LSE merge, V4 attention BWD split into dQ + dK/dV kernels, CSA BWD sparse tuning. Single-kernel targets: CSA FWD ≤ 6 ms, V4 attention BWD ≤ 15 ms, CSA BWD ≤ 15 ms | P31b follow-up: "the remaining V4 attention BWD / CSA FWD / CSA BWD slice is the residual proxy-trace bottleneck; tightening it via the standalone operator microbenchmark avoids the proxy-train round-trip" |

> **Caveat — P29..P31 are trace-driven; P32 is microbench-driven.**
> The exact fusion / autotune targets for P29..P31 get **picked** in
> P28's analysis report. P32 takes the post-P31b shape baseline forward
> with the CSA microbench plus a new V4 attention microbench so kernel
> iteration is not gated on a full EP8 training round-trip. Plan-5
> commits to delivering P28; the P29..P32 task lists below are seeded
> from the user's optimisation hints and from known plan-4 / P31b
> follow-ups, but each phase opens with a "task list refinement" pass
> that revises the breakdown against the latest trace / bench data
> before kernel work starts.

## References

- Plan-4 wrap-up: `../plan-4/02-phase-details.md` (P27 hand-off block)
  + `../progress/status.md` (Phase 27 row).
- Plan-3 P22 / P23 traces (existing baselines for delta reporting):
  - `../progress/p19/run_profile_ep8.sh` — eager attention, no DeepEP, EP=8.
  - `../progress/p23/run_profile_deepep_on_ep8.sh` — eager attention, DeepEP, EP=8.
  - `../progress/p25/run_profile_v4_triton_attention_ep8.sh` — V4 Triton dense kernel, eager CSA, no DeepEP, EP=8.
- V4 attention class (the small-op surface plan-5 optimises against):
  `primus/backends/megatron/core/transformer/deepseek_v4_attention.py`.
- V4 attention kernels (the Triton kernels plan-5 tunes):
  `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/`.
- V4-Flash production yaml:
  `primus/configs/models/megatron/deepseek_v4_flash.yaml` (43 layers,
  `hidden_size=4096`, `H=64`, `head_dim=512`, `num_experts=256`,
  `moe_router_topk=6`, `moe_ffn_hidden_size=2048`,
  `compress_ratios=[0,0,4,128,...,4,0]` — every layer kind exercised).

## Documents

- [`01-roadmap.md`](./01-roadmap.md) — phase overview, dependency
  graph, milestones, exit criteria, top risks, out-of-scope.
- [`02-phase-details.md`](./02-phase-details.md) — phase-by-phase
  task breakdown, design notes, edge cases.
- [`03-test-strategy.md`](./03-test-strategy.md) — gate matrix
  (G31..G34b), correctness ratchet (every perf phase MUST keep G23 /
  G24 / G26 / G27 / G29 / G30 green), and the perf-budget contract.

## Reporting hand-off

Plan-5 closes after P32 when every remaining phase row in
`../progress/status.md` is checked, the latest EP=8 proxy trace/report
plus the operator microbenchmarks (`progress/p31/bench_csa_attention_ep8.py`
and `progress/p32/bench_v4_attention_ep8.py`) show the per-kernel
targets met, and every plan-4 unit-test gate
(G23 / G24 / G25 / G26 / G27 / G28 / G29) plus the plan-5 smoke /
perf gates through G35 are green at the final commit.

## Out of scope (plan-5)

- **FP8 / FP4 / mxfp4 quantised forward** — separate plan.
  Plan-5 stays at BF16; the V4 Triton kernels' FP8 hooks (P25 has
  the `__init__.py` placeholder) are wired but not exercised here.
- **Convergence run** — plan-5 runs 10–50-iter smokes for trace
  capture and perf gating; convergence to a target loss is a separate
  plan that owns dataset prep + the multi-day training run.
- **Long-context (1M-token) bring-up** — V4 supports 1M tokens in
  the released checkpoint, but the Primus pretrain target is
  `seq_length=4096`; long-context perf lives in a future plan.
- **Multi-node EP scaling** — plan-5 is single-node EP=8. Multi-node
  EP=64 / EP=128 is a separate plan that owns the inter-node
  RoCE / IB tuning.
- **HF state-dict adapter** — plan-2 deferred to "Phase 22+".
  Plan-5 does not unblock it.
