# 02 — Plan-5 Phase Details

> Each phase below lists (a) the user request that motivates it, (b)
> the concrete tasks, (c) the design notes that the implementer must
> keep in mind, and (d) the edge cases / risks. Test gates live in
> `03-test-strategy.md`. P29 / P30 / P31 / P32 task lists are
> **seeded**; each phase opens with a "task list refinement" pass
> that revises the breakdown against the latest trace / bench data.

## Phase 28 — V4-Flash proxy + EP=8 baseline trace + bottleneck report

> "基于run_deepseek_v4.sh创建一个run_deepseek_v4_flash_proxy.sh，
> 基于v4-flash模型配置，先确定一个proxy模型，调整layer数量，例如8层，
> 保证能够EP8单机跑起来训练（把v4 triton attention和v4 triton
> csa attention, deepep, turbo grouped gemm都打开）。如果显存还是不够，
> 就调整seq length等其他参数。接着，生成一个EP8的trace，这个作为一个
> baseline，然后分析trace里面瓶颈，把分析报告md格式和html格式放到
> deepseek-v4/develop/profile目录下面，命名是profile-baseline开头。" —
> user, plan-5 kick-off.

P28 is the foundation for every other plan-5 phase: it defines the
proxy that lets us measure V4-Flash perf on a single MI355X node, the
baseline trace that names the bottlenecks, and the report that picks
which optimisations are in scope for P29 / P30 / P31. Without
P28, every perf phase would pick fusion / autotune targets blind, and
the "did this phase actually help" check would have nothing to compare
against.

### Tasks

1. **Proxy script** — create `run_deepseek_v4_flash_proxy.sh` next to
   `run_deepseek_v4.sh`. The proxy:
   - Exports `PRIMUS_TOTAL_LAYERS=8` (vs V4-Flash production's 43);
   - Exports `PRIMUS_NUM_EXPERTS=256` (full V4-Flash MoE width);
   - Exports `PRIMUS_MOE_TOPK=6` (full V4-Flash router top-k);
   - Exports `PRIMUS_MOE_FFN_HIDDEN_SIZE=2048` (full V4-Flash MoE FFN);
   - Exports `PRIMUS_INDEX_TOPK=512` (full V4-Flash CSA top-K);
   - Exports `PRIMUS_COMPRESS_RATIOS="[0,0,4,128,4,128,4,0]"` —
     8-layer slice that exercises every layer kind (3 × cr=0, 3 × cr=4,
     2 × cr=128) without changing the cr=0 endpoints (V4-Flash has
     dense first/last layers);
   - Exports `PRIMUS_TP=1 PRIMUS_PP=1 PRIMUS_EP=8` (single-node EP=8);
   - Exports `MBS=1 GBS=8` (one micro-batch per data-parallel rank;
     V4-Flash production GBS scales linearly with DP, EP=8 → DP=1 here);
   - Exports `PRIMUS_SEQ_LENGTH=4096`,
     `PRIMUS_MAX_POSITION_EMBEDDINGS=4096` (V4 pretrain target seq);
   - Exports `USE_V4_TRITON_ATTENTION=True`,
     `USE_V4_TRITON_CSA_ATTENTION=True`,
     `USE_TURBO_DEEPEP=True`, `TURBO_USE_GROUPED_MLP=True`,
     `USE_TURBO_ATTENTION=False` (Turbo would override the V4 Triton
     dense path; we want the V4 Triton kernel to actually run);
   - Defaults `TRAIN_ITERS=10` for the smoke pass, but accepts an
     override (the trace pass uses `TRAIN_ITERS=10` with profiler
     window iter 6 → iter 7 to keep the artefact size bounded);
   - Calibrates `PRIMUS_SEQ_LENGTH` if the first run OOMs: fall back
     to `2048` → `1024` → `512` and document the chosen value in the
     report. The CSA wrapper-side `[B, H, Sq, K_topk, D]` gather is
     the dominant memory cost (`64 GiB` at `Sq=4096, H=64, K=512,
     D=512` in bf16); plan-4 P26 documents this. P31's in-kernel gather
     is the structural fix.
2. **Baseline trace capture** — create
   `deepseek-v4/develop/progress/p28/run_baseline_trace_ep8.sh` that
   wraps the proxy with `PROFILE=True --profile_step_start 6
   --profile_step_end 7` so torch.profiler emits the chrome-trace
   JSON for one steady-iter window. Trace JSON lands at
   `output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/tensorboard/...
   .pt.trace.json`. The progress dir ships a `.gitignore` matching the
   plan-4 P27 pattern (`*.log`, `log_*.txt`, `*.tgz`, `*.json`,
   `debug.log`) so the trace JSON does not land in git. The proxy
   script + the trace-capture script + the report are the only files
   committed for P28.
3. **Bottleneck analysis report (md + html)** — create
   `deepseek-v4/develop/profile/profile-baseline-ep8-<YYYYMMDD>.md`
   and `.html`. The report covers:
   - **Run config provenance** — the resolved
     `PRIMUS_*` env vars, the resolved YAML overrides, the commit
     SHA, the host (`mi355-gpu-XX`), the container tag, the chosen
     `PRIMUS_SEQ_LENGTH` (with rationale if downscaled);
   - **Per-iter wall time** — cold (iter 0..2), warm (iter 3..5),
     steady (iter 6..9). TFLOP/s/GPU under the V4-aware FLOPS
     formula (plan-3 P20 patch);
   - **GPU vs CPU active / idle %** — total stream-0 active time
     vs total iter time; CPU thread active time vs idle. This is
     the headline metric for "kernel-launch overhead is high, CPU
     idle is high";
   - **Top-N kernels by total time** — `key_averages()` table
     (top 30) with name, count, self time, total time, % of step,
     average time per call. Triton kernels (`v4_attention_fwd`,
     `v4_attention_bwd_preprocess`, `v4_attention_bwd`,
     `v4_csa_attention_fwd`, `v4_csa_attention_bwd_preprocess`,
     `v4_csa_attention_bwd`) called out separately; eager small-op
     kernels (`elementwise_kernel`, `index_select_kernel`,
     `gather_kernel`, `softmax_kernel`, `addmm_kernel`,
     `index_kernel`, `mul_kernel`, `add_kernel`, `cat_kernel`,
     `slice_kernel`) called out separately;
   - **Kernel launch count + average launch interval** — total
     kernels launched per iter, histogram of inter-launch intervals
     (10 µs / 50 µs / 100 µs buckets), the launch-rate floor
     ("every kernel launch takes ≥ X µs of CPU; if the average GPU
     kernel takes < X µs, you are CPU-bound");
   - **Module-level CPU time attribution** — chrome-trace
     traceEvents grouped by `cat=cpu_op` and split by qualified
     name. The output is "DeepseekV4Attention.forward took N ms",
     "DeepseekV4Attention._attention_forward took M ms",
     "Compressor.forward took P ms", "Indexer.forward took Q ms",
     "DeepseekV4MoE.forward took R ms" — i.e. which Python module
     contributes which slice of the kernel-launch tail;
   - **Comm time** — DeepEP `dispatch` / `combine` total time +
     ratio of comm to compute. `c10d::*` ops separated;
   - **Ranked bottleneck list** — three rows minimum: `[1] eager
     small-op chain in <module>`, `[2] <attention kernel> at
     <shape>`, `[3] <comm or other>`. Each row carries an
     improvement budget (e.g. "current: 18.5 ms / iter (= 32 % of
     step), target: 10 ms / iter (= 18 % of step)") that becomes
     the perf-budget contract for the corresponding plan-5 phase;
   - **Per-phase improvement budgets** — set the X / Y / Z
     numbers from `01-roadmap.md` against the actual baseline.
     If the trace shows the small-op tail is < 10 % of step time,
     P29 gets de-scoped; if attention kernels are < 10 %, P30 / P31
     get de-scoped. The report owns the de-scope decision in writing.
4. **Report tooling** — the report-generation code (a small Python
   script that consumes the chrome-trace JSON) lives at
   `deepseek-v4/develop/profile/_tools/render_baseline_report.py`.
   It parses `traceEvents`, builds the kernel / module / comm
   tables, renders the markdown report, then renders an HTML
   dashboard (Plotly tables + bar charts; no chrome-trace embed).
   The tool is committed and reused by post-phase profile reports.

### Design notes

- **Proxy seq length calibration** is the load-bearing decision in
  P28. The wrapper-side CSA gather grows as `O(B * H * Sq * K * D)`
  and dominates HBM; at V4-Flash dims with `B=1, H=64, K=512, D=512,
  bf16` it is `64 KiB * Sq` per microbatch. At `Sq=4096` that is
  `256 MiB` per microbatch... wait, recompute: `1 * 64 * 4096 * 512
  * 512 * 2 / (1024^3) = 64 GiB`. Yes 64 GiB. Plus the
  `[B, S, K, D]` pool = `1 * 4096 * 512 * 512 * 2 / GiB = 2 GiB`.
  Plus expert weights (`32 experts/rank * 25 M params * 2 B = 1.6 GiB
  per layer * 6 MoE layers = 9.6 GiB`), KV cache, optimizer state,
  activations. Total comfortably blows 192 GiB at `Sq=4096` if EP=8
  (DP=1). Realistic landing: the proxy will likely calibrate to
  `Sq=1024` or `Sq=2048`. The report owns picking and documenting
  the value.
- **Trace window** uses one steady-iter (`profile_step_start=6
  profile_step_end=7`) to keep the trace JSON under ~500 MB.
  Multi-iter windows are not necessary for bottleneck analysis;
  iter-to-iter variance is captured by per-iter wall time stats
  (above) which read out of the trace's `traceEvents` for every iter.
- **Report tooling** stays minimal (Plotly tables, no chrome-trace
  viewer embed) so the HTML dashboard is committable (~50–200 KB).
  An offline chrome-trace viewer is a nice-to-have but a separate
  asset.
- **Bottleneck-list de-scope rule** is in writing in the report so
  that P29 / P30 / P31 cannot accidentally take work that the
  trace says is < 10 % of the step. Plan-5 is **measurement-driven**;
  every optimisation phase has a baseline cost it is buying down.
- **Banned-warning ratchet** — the proxy MUST run with zero
  `submodule init failed` / `fallback to nn.Linear` /
  `unsupported dispatcher module` / `c10d::allreduce` / `using local
  Compressor|Indexer` / `fallback to alltoall` warnings (plan-3
  P20 / P21 / P23 + plan-4 P27 ratchet). G31 enforces this on the
  P28 smoke.

### Edge cases

- **Cold-iter cost** — torch.compile / Triton autotune trigger at
  the first call; iter 0..2 wall time is **not** representative.
  The report partitions cold / warm / steady explicitly so plan-5
  perf claims always cite the steady number.
- **MoE expert load imbalance** — at EP=8 with 256 experts and
  random-init router, some experts can see 0 tokens / iter. DeepEP's
  empty-bucket handling has historically masked overhead at low
  experts-per-rank; the proxy keeps `num_experts=256` so the
  experts-per-rank count is realistic (32 / rank).
- **First-iter NaN sanity** — V4 trains stably with random init at
  the plan-4 G30 smoke shape; at the proxy shape (longer seq, full
  MoE width) re-verify on the cold iter so we do not trace a NaN
  step.

---

## Phase 29 — Sinkhorn fp32 reduce: kill the dominant 7.6 s kernel (RESCOPED)

> P28 (commit `afd7ea59`) report at
> `develop/profile/profile-baseline-ep8-20260508.md` named the dominant
> `aten::sum` fp32 reduce as the #1 bottleneck and KEEP-RESCOPED P29:
> "P29 KEEP — RESCOPE. CPU-bound floor is 0.3 % (≪ 10 % rule), so the
> original P29 mandate (small-op kernel-launch fusion via torch.compile
> or Triton-fused Compressor / Indexer / MoE-router chains) is
> de-scoped. P29 is redirected to root-cause + eliminate the dominant
> aten::sum fp32 reduce (87.5 % of step, 87 % of Σ kernel dur)."
>
> The original "small-op fusion" kick-off framing
> ("我之前看到trace里面，有很多小kernel, kernel launch开销很大") does NOT
> hold at V4-Flash production widths — the P28 trace shows the GPU is
> 99.7 % active and the CPU-bound floor is 0.3 %. Kernel-launch tail is
> not the bottleneck.

### What was de-scoped (and why it stays in tree as a follow-up note)

The seeded plan-5 P29 candidates (a) `v4_fused_q_proj`,
(b) `v4_fused_kv_proj`, (c) `v4_fused_o_proj`, (d) `v4_fused_compressor`
+ `v4_fused_indexer`, (e) `v4_fused_moe_router` are all targeted at the
small-op kernel-launch tail. The P28 bottleneck table shows that tail
is rank #4 at 0.7 % of step (`small-op kernel-launch tail`), well below
the 10 % keep-rule. They become **plan-5 follow-ups** for revisit only
if a future trace shows the small-op tail re-emerging at a different
configuration (e.g. multi-node EP with cross-node activation
reshuffling, or a much smaller per-rank batch).

### Forensic root cause (P29 task 1 deliverable)

`progress/p29/refinement.md` (the task-list refinement document for
this rescope) walks every dominant
`reduce_kernel<512, 1, ReduceOp<float, sum_functor<float, float, float>>>`
launch back to `primus/backends/megatron/core/transformer/
hyper_connection.py:47 sinkhorn_normalize`. **624 of 717 launches**
(96 % by count, 99.95 % by time) are `aten::sum` calls on shape
`(1, 4096, 4, 4) → (1, 4096, 4, 1)` with `dim=[-1] keepdim=True` —
the 64-K-element fp32 reduce that the Sinkhorn-Knopp doubly-stochastic
projection issues 39 times per call (1 priming column-norm + 19
alternating row/col cycles × 2 sums). Eight V4-Flash hybrid layers per
iter × 39 sums per call × 2 (FWD + BWD chain) ≈ 624 launches.

The kernel runs at **~250× over memory-bound floor** (12.19 ms observed
vs. 51 µs floor) because the HIP / ROCm dispatcher chose a 512-thread-
block reduce kernel with one thread per output for an inner dim of
size 4 → effective occupancy ≈ 12.5 %, plus 624 × 5 µs launch overhead.
We can't fix the dispatcher; we **must avoid issuing 624 of these
calls per iter in the first place**.

### Tasks

1. **Task-list refinement (P29 task 1)** — `progress/p29/refinement.md`
   pins the forensic call-site attribution + the chosen fix
   (`torch.compile(fullgraph=True, dynamic=False)` wrapping
   `sinkhorn_normalize`) + the fall-back (hand-Triton fused-Sinkhorn
   kernel) + the X1 perf budget (≥ 50 % drop in `aten::sum` kernel
   time, ≥ 35 % drop in steady iter wall time, ≥ 60 % gain in
   TFLOP/s/GPU).
2. **Config flag** — add `use_v4_compiled_sinkhorn: bool` (default
   `False`) on `DeepSeekV4TransformerConfig`; plumb through
   `examples/megatron/configs/MI355X/deepseek_v4_*-BF16-pretrain.yaml`,
   `run_deepseek_v4.sh`, and the proxy script
   `run_deepseek_v4_flash_proxy.sh`.
3. **Fix implementation** — in `hyper_connection.py`:
   - module-level cached compile (`_compiled_sinkhorn_cache: dict[(int,
     float), Callable]`) so each `(n_iters, eps)` combination compiles
     exactly once per process;
   - `sinkhorn_normalize` accepts a new `use_compiled: bool = False`
     keyword that switches between the eager loop (today's path) and
     the cached compiled path;
   - `HyperMixer.__init__` accepts `use_compiled_sinkhorn: bool = False`
     and forwards it at every `compute_weights` call;
   - `DeepseekV4HybridLayer` reads `config.use_v4_compiled_sinkhorn`
     and passes it through.
4. **G32 — equivalence test** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/
   test_v4_p29_compiled_sinkhorn.py`:
   - fast tier: `B=2 S=64 K=4` random fp32 inputs, FWD + BWD parity
     vs eager within `atol=1e-5, rtol=1e-5` (Sinkhorn is bit-stable in
     fp32 for K=4);
   - release tier (`pytest.mark.slow`): `B=1 S=4096 K=4` (V4-Flash
     production input shape), same parity budget;
   - cold-compile time recorded in the test as a `print()` line.
5. **G33a — proxy smoke** —
   `progress/p29/run_smoke_compiled_sinkhorn_ep8.sh`: 10-iter EP=8 run
   under the proxy with the flag on; assert plan-4 ratchet
   (G23..G30) stays green, no banned warnings (plan-3 / plan-4 grep
   set), `lm_loss` after 10 iters within 5e-2 of the P28 baseline at
   the same fixed seed.
6. **G33b — proxy trace + post-P29 report** —
   `progress/p29/run_baseline_trace_ep8_p29.sh` (mirrors
   `progress/p28/run_baseline_trace_ep8.sh`) captures iter 6 → 7 with
   the flag on; render
   `develop/profile/profile-after-p29-ep8-<YYYYMMDD>.{md,html}` reusing
   `develop/profile/_tools/render_baseline_report.py`. Assert
   `aten::sum` fp32 reduce kernel time drops by ≥ 50 % vs P28 baseline
   (budget X1).
7. **Conditional escalation** — if the post-P29 trace shows a
   `< 50 %` drop, ship the fall-back hand-Triton fused-Sinkhorn kernel
   (`primus/backends/megatron/core/transformer/v4_attention_kernels/
   _triton/v4_sinkhorn.py`) under the same `use_v4_compiled_sinkhorn`
   flag; document in `progress/p29/post_compile_results.md`.
8. **Default flip** — once G32 + G33a + G33b are green, flip
   `use_v4_compiled_sinkhorn` default to `True` in YAML +
   `run_deepseek_v4.sh` + the proxy. The plan-5 baseline TFLOP/s/GPU
   pinned in P28 is rolled to the post-P29 value for P30 / P31 to
   measure against.

### Design notes

- **Why `torch.compile(fullgraph=True, dynamic=False)` is the right
  tool.** The 39 sums + 39 divides + 39 broadcasts in
  `sinkhorn_normalize` are pure-function shape-stable
  (`(1, 4096, 4, 4)` fp32 in, same fp32 out) — exactly the case
  Inductor handles best. Inductor unrolls the Python `for` loop at
  compile time (since `n_iters` is captured in the closure), produces
  one big graph, and fuses the whole thing into one (or maybe two:
  FWD + BWD) Triton kernel(s). No algorithmic change → bit-equivalent
  math (modulo non-deterministic reduction order which the test
  tolerance accommodates). AOT-autograd compiles BWD too.
- **Why we did NOT pick "reduce `n_iters` from 20 to 5".** That is a
  model-quality decision that affects pretraining convergence
  (Sinkhorn-Knopp converges quadratically; 5 iters is usually enough
  for `K=4`, but the V4 release pinned `hc_sinkhorn_iters=20`). Out
  of plan-5 scope.
- **Why we did NOT pick "cast `sinkhorn_normalize` to bf16".** The
  function explicitly casts up to fp32 (techblog §2.2 pitfall #3 —
  the doubly-stochastic projection is sensitive to dtype underflow on
  the iterative `m / m.sum()` chain); a cast-to-bf16 hedge is its own
  experiment, not a perf optimisation, and would need its own
  numerical-stability gate.
- **Cold-compile cost.** First-iter compile time for
  `torch.compile(fullgraph=True, dynamic=False)` on a 39-op pure
  function is typically 5 – 20 s; subsequent calls are ~0 latency.
  The proxy's iter 0 will reflect this. The post-P29 report uses
  steady-iter throughput; total wall-time-to-first-iter is
  documented but does not gate.
- **Why `dynamic=False`.** The Sinkhorn input shape is statically
  `[B, S, K, K]` per V4 layer with `K = hc_mult = 4`. Locking shapes
  via `dynamic=False` avoids guard-failure recompiles.
- **Why module-level cache (`_compiled_sinkhorn_cache`).** Without
  the cache, each `HyperMixer` instance compiles its own copy → 8
  copies per V4-Flash trunk + N more per MTP head. With the cache
  keyed on `(n_iters, eps)`, exactly one compilation happens per
  process for all layers.

### Edge cases

- **`torch.compile` ROCm pin.** Inductor's HIP backend has had
  occasional miscompiles on minor torch versions. The `use_v4_
  compiled_sinkhorn` switch defaults to `False` for the P29-shipping
  commit; the default flip happens only after G33b's perf delta is
  pinned and the equivalence test is green at release tier.
- **Recompilation guard.** If a future config change varies
  `hc_sinkhorn_iters` per layer (currently uniform), the cache key
  picks that up. If shape varies (e.g. when the K-stream
  representation changes between PP stages), the proxy script must
  pin one `seq_length` per run.
- **Numerical determinism.** `torch.compile` may reorder the inner
  reductions into a different order than eager (e.g. tree-reduction
  vs sequential reduction); the FWD output is bit-identical for
  `K = 4` (only 6 reduction orderings exist on 4 floats and pairwise
  tree reduction is the canonical choice for Inductor) but the BWD
  may differ at the ULP level. The G32 tolerance (`atol=1e-5,
  rtol=1e-5`) accommodates this.
- **Plan-2 P17 state-dict compatibility.** The fix touches no weight
  tensors (Sinkhorn is parameter-free); state-dict keys are
  unchanged.

---

## Phase 30 — V4 Triton attention kernel perf tuning

> "如果小kernel问题解决了，大概率attention可能是后续的瓶颈之一。可以
> 制定一些关于v4 triton的两个attention算子的性能优化。" — user,
> plan-5 kick-off (optimisation hint #2).

P30 tunes the in-tree dense / HCA `v4_attention` Triton kernel. The
post-P29 trace shows the wall-time critical path moved to V4 Triton
attention BWD, with `_v4_attention_bwd_kernel` at **3.18 s / 36.8 %**
of the steady step and `_v4_attention_fwd_kernel` at **641 ms / 7.4 %**.

The first P30 optimisation is deliberately narrower than the seeded
autotune / LSE-merge plan: **SWA K-loop range pruning** for dense
`compress_ratio == 0` and HCA `compress_ratio == 128` layers. The
plan-4 dense wrapper was still passing a materialised local additive mask
into the kernel, and the first P30 cut only moved dense onto the
kernel-native `swa_window` path. Trace review then showed the two
remaining 600 ms+ `_v4_attention_bwd_kernel` launches were exactly the
HCA layers, where the joint full additive mask still forced a full
local-key scan. P30 therefore adds an HCA split-mask mode: local keys use
the same pruned SWA loop, while the compressed-pool suffix uses the
pool-only additive mask.

### Tasks

0. **Task list refinement** — read the post-P29 report's attention
   rows. Because dense / HCA attention totals **3.82 s / 44.2 %** of
   the step, P30 stays in scope. Choose SWA K-loop pruning as the
   first cut because it removes mathematically dead tiles with no new
   user-facing flag, no autotune warmup, and no dtype-contract change.
1. **FWD SWA K-loop pruning** — in
   `_triton/v4_attention_fwd.py`, compute `n_loop_start` from the
   earliest query row in the program:
   `max(0, pid_m * BLOCK_M - SWA_WINDOW + 1)`, rounded down to
   `BLOCK_N`. Iterate `range(n_loop_start, n_loop_end, BLOCK_N)`.
   `HAS_ADD_MASK` still forces the full key axis because caller-provided
   additive masks can express arbitrary visibility.
2. **BWD SWA K-loop pruning** — apply the same loop-start logic in
   `_triton/v4_attention_bwd.py`. The recompute path must visit exactly
   the same visible K tiles as FWD so FWD/BWD parity remains inside the
   existing bf16 tolerance budget.
3. **Dense wrapper dispatch** — in `deepseek_v4_attention.py`, route
   `compress_ratio == 0` + V4 Triton attention through
   `v4_attention(..., additive_mask=None, swa_window=attn_sliding_window)`.
   Eager dense layers continue building `_local_mask`; HCA
   (`compress_ratio == 128`) passes `extra_mask` (`[S, P]`) plus
   `hca_local_seqlen=S` so the kernel can split local SWA and pool
   visibility under one joint softmax. Eager HCA still builds the full
   concatenated mask.
4. **HCA split-mask kernel mode** — extend `v4_attention` with optional
   `hca_local_seqlen` (default `0`). When set, the FWD/BWD kernels run
   two K loops: a pruned local-SWA loop over `[0, S)` and a short
   pool-suffix loop over `[S, S+P)` using the pool-only additive mask.
5. **G34 — SWA-prune equivalence + release-shape gate** — re-run
   `test_v4_p25_v4_attention_{fwd,bwd}.py` at fast and slow tiers.
   This covers dense SWA and HCA additive-mask shapes, including
   `head_dim=512`, sink, and bf16 BWD tolerances.
6. **G34a — EP8 smoke + trace delta** — run a 10-iter EP=8 proxy with
   P29 compiled Sinkhorn and P30 SWA pruning on, then capture a
   torch.profiler trace and render
   `profile-after-p30-ep8-<YYYYMMDD>.{md,html}`. The P30 success metric
   is measured against post-P29: lower `_v4_attention_bwd_kernel` time,
   stable loss, no banned warnings, and positive steady TFLOP/s/GPU.
7. **Deferred structural work** — per-shape autotune, persistent FWD,
   and the HCA LSE-merge variant stay as follow-ups. They require a
   larger correctness surface and are better justified after the P30
   trace shows the residual dense/HCA cost.

### Design notes

- **No new user knob.** SWA K-loop pruning is a strict implementation
  improvement under the existing `use_v4_triton_attention` path. If
  `additive_mask` is present, pruning is disabled because arbitrary
  additive masks are not guaranteed to be sliding windows.
- **Exact-mask equivalence.** For dense layers the additive local mask
  and the kernel-native `swa_window` mask describe the same set of
  visible keys. The only intended numerical difference is the absence
  of work on fully masked tiles; output and gradients stay within the
  existing plan-4 tolerances.
- **HCA split-mask is still one joint softmax.** The implementation does
  not approximate HCA as two independent attentions. It keeps the same
  online softmax state across the pruned local loop, pool loop, and sink
  update, matching the original full-mask math.
- **HCA LSE-merge remains deferred.** Split-mask removes the dead local
  tiles while preserving the single-kernel structure. The LSE-merge
  variant is still the future route if we want to run local and pool
  branches as independent flash kernels and merge their LSEs.

### Edge cases

- **MQA broadcast** — V4 K / V have only `num_query_groups=1`; the
  kernel broadcasts across H Q-heads via the in-tile reuse pattern
  (plan-4 P25). The autotune grid keeps this; configs that allocate
  H copies of K / V in SMEM (Inductor sometimes does this) fail the
  SMEM heuristic and get pruned.
- **`dsink` atomic-add jitter** — autotune may pick a config that
  reorders the BWD `dsink` reduction; bf16 `dsink atol=5e-2`
  (release-tier) absorbs typical jitter. If the new config breaches
  it, the config gets dropped or the budget gets re-derived (with
  written rationale).

---

## Phase 31 — V4 Triton CSA kernel perf tuning

> "如果小kernel问题解决了，大概率attention可能是后续的瓶颈之一。" —
> user, plan-5 kick-off (optimisation hint #2, applied to CSA).
> Plus plan-4 P26 follow-up: "wrapper-side gather is a known
> limitation; in-kernel `topk_idxs` gather is left for a future
> perf plan."

P31 ships the in-kernel `topk_idxs` gather that plan-4 P26 deferred,
plus K-tile prefetching. The wrapper-side gather is the dominant HBM
cost at production V4-Flash dims (64 GiB / microbatch at `Sq=4096`),
so the in-kernel gather is also the structural OOM fix that lets P28's
proxy eventually go to `Sq=4096` without downscaling.

### Tasks

0. **Task list refinement** — read the P28 report's CSA-kernel row +
   peak HBM row; if CSA-kernel time is < 10 % of step time AND HBM
   headroom is sufficient at the chosen proxy seq length, P31 gets
   de-scoped to a "K-tile prefetch only" pass.
1. **In-kernel `topk_idxs` gather** — replace the wrapper-side
   `gathered = pool[..., topk_idxs, :]` materialisation with a
   `tl.load` on `pool` driven by `topk_idxs` inside the K-tile loop
   in `_triton/v4_csa_attention_fwd.py`. The kernel signature
   becomes `(q, k_local, v_local, pool_k, pool_v, topk_idxs, sink,
   ...)` — `pool_k / pool_v` are the un-gathered compressed pool
   tensors of shape `[B, P, head_dim]`. The wrapper-side gather
   stays in tree as the eager fallback (controlled by a new
   `use_v4_csa_in_kernel_gather` switch defaulting to `False`).
2. **Backward in-kernel scatter-add** — `dgathered`'s wrapper-side
   `torch.gather` BWD becomes an in-kernel `tl.atomic_add` driven
   by `topk_idxs` into `dpool`. Kernel returns `dpool` directly;
   no wrapper-side scatter.
3. **K-tile prefetching** — for the per-row design (one program per
   `(b, qhid, m)`) the K-tile load is the dominant HBM read. Add
   software pipelining (`tl.advance` + double-buffered `tl.load`).
4. **Equivalence gates (plan-4 G26 / G27 release tier)** — re-run
   `pytest --run-slow` on
   `test_v4_p26_v4_csa_attention_{fwd,bwd}.py` with the new switch
   on; assert FWD + BWD numerical equivalence within the existing
   bf16 release-tier budget (`fwd atol=5e-2`,
   `bwd dq/dk_local/dv_local/dpool atol=2e-1`,
   `dsink atol=5e-2`). The `dgathered` leaf disappears (replaced by
   `dpool`); the test file gains a `dpool` assertion path.
5. **End-to-end smoke (G33 re-run)** — proxy with the in-kernel gather
   on; assert (a) CSA-kernel time drops by ≥ Z %; (b) wrapper gather
   peak HBM ≈ 0; (c) ≥ +X % cumulative TFLOP/s/GPU vs P28 baseline.
   Z and X come from the P28 report.

### Design notes

- **`pool_k / pool_v` alignment for `tl.load`** — `topk_idxs` is
  a `[B, Sq, K]` int32 tensor; `pool_k` is `[B, P, D]`. The
  in-kernel load pattern is `tl.load(pool_k_ptr + topk_idxs[b, q,
  k_idx] * D + d_offsets)` per K-tile. Coalesced load requires
  `topk_idxs` to be sorted within a tile; if it is not, the load
  is gathered and slow. Plan-4 P26's `_topk_with_dropout` does NOT
  sort; P31 owns adding a per-tile sort pass IF the trace shows the
  scatter is bandwidth-bound.
- **HBM savings** — at V4-Flash dims `(B=1, S=4096, K=512, D=512)`
  bf16 the gather drops `B * S * K * D * 2 = 2 GiB` / microbatch
  for the gathered tensor (CSA layer count × micro-batch count);
  at full proxy this is ~12 GiB headroom across 3 cr=4 layers. At
  V4-Pro `(K=1024)` the savings double.

### Edge cases

- **Atomic-add into `dpool`** — multiple `(b, q, k)` tuples can
  point at the same pool slot (the top-K with replacement option in
  V4 is on); the BWD must atomic-add to handle overlap. fp32
  accumulator inside `dpool`, cast back to input dtype on return.
- **`topk_idxs == -1` slot mask** — sparse top-K pads short
  sequences with `-1`; the kernel checks `topk_idxs >= 0` before
  every `tl.load` (plan-4 P26 already takes this contract).

### P31 close-out note

P31 shipped the in-kernel gather/scatter path as a new
`v4_csa_attention_from_pool` API. `DeepseekV4Attention._csa_forward`
now passes the compressed `pool` and `topk_idxs` directly when
`use_v4_triton_csa_attention=True`; the old `gathered` API remains in
tree for eager fallback and P26 ratchet tests.

Correctness:

- P31 fast pool/topk tests: `8 passed`.
- P31 release pool/topk tests: `8 passed`.
- P26 CSA release ratchet: `16 passed`.
- P26/P27 fast CSA + dispatch ratchet: `43 passed`.

Performance on the V4-Flash EP8 proxy:

- 10-iter smoke: `lm_loss[10] = 9.259875E+00`, no NaN / Inf, steady
  iter 10 `4312.3/4331.7 ms`, `158.7/158.0 TFLOP/s/GPU`.
- Trace: steady window `4317.0 ms`, `158.5 TFLOP/s/GPU`, vs P30b
  `4943.4 ms`, `138.4 TFLOP/s/GPU` (`-12.7 %` step time, `+14.5 %`
  throughput).
- CSA BWD drops from P30b `4.04 s` to P31 `3.50 s` (`-13.5 %`);
  CSA FWD drops from ~`153 ms` to `123.5 ms`.

The original `BLOCK_K=64` tuning experiment compiled and passed the
P31 fast/release tests, but smoke throughput regressed to ~`155`
TFLOP/s/GPU, so the shipped pool kernels keep `BLOCK_K=32`. K-tile
prefetch remains deferred; the remaining P31 bottleneck is the per-row
CSA BWD design itself, especially sparse-branch bandwidth and atomic
pressure into `dpool`.

### P31 follow-up — CSA BWD deep optimization

After the first close-out, CSA BWD was still too slow. P31 was reopened
with a stricter target: drive the CSA backward kernel toward **< 50 ms**
at the proxy EP8 shape.

New tasks:

1. **Standalone EP8-shape CSA benchmark** — add
   `progress/p31/bench_csa_attention_ep8.py` so candidate kernel changes
   can be measured on one GPU without launching full EP8 training. The
   default shape is `B=1, H=64, S=4096, D=512, P=1024, K_topk=512,
   swa_window=128, bf16, sink=on`.
2. **Optimization log** — record reference scans, failed tuning
   experiments, benchmark numbers, and conclusions in
   `progress/p31/csa_bwd_optimization_log.md`.
3. **Quick contention experiments** — test per-head `dpool` staging and
   launch-parameter tuning (`num_warps`, `BLOCK_K`) under the existing
   correctness tests before using the benchmark.
4. **Sparse BWD redesign plan** — if the benchmark shows the current
   per-row sparse branch cannot approach target, pivot to a split sparse
   BWD design: separate `dQ` from `dpool`, and investigate pool-owned
   reduction / inverted-index accumulation to remove random `dpool`
   atomics.

Current benchmark outcome: the first corrected BWD-only baseline was
~`1433 ms`. A split redesign now runs the local branch through the dense
`_v4_attention_bwd_kernel` with CSA's joint `lse/D`, then runs a new
head-block sparse kernel for `dq + dpool`. The standalone EP8-shape
benchmark reports **35.43 ms** BWD-only, meeting the <50 ms kernel target.
The same pass also fixed the benchmark so the BWD number excludes forward
execution.

---

## Phase 32 — Operator-microbenchmark-driven attention kernel speed-ups

> "添加phase32，继续优化v4 triton attention和v4 triton csa attention的
> 性能。优化目标：v4 csa attention forward，当前在41ms左右，需要优化到
> 6ms以内。v4 attention backward，当前在31ms左右，需要优化到15ms以内。
> v4 csa attention backward，当前在26ms左右，需要优化到15ms以内。优化
> 可以使用类似 `progress/p31/bench_csa_attention_ep8.py` 单独的算子
> benchmark来进行测试，可以再添加一个attention的版本，不用每次都跑proxy
> model train来测试算子性能，这样比较快。" — user, plan-5 P32 kick-off.

P32 closes the remaining V4 Triton attention slice of the EP8 proxy
trace. Post-P31b, three kernel families still dominate the attention
budget at the production V4-Flash EP8 shape (`B=1, H=64, S=4096,
D=512, swa=128, sink=on, bf16`):

- **CSA FWD (`_v4_csa_attention_pool_fwd_kernel`)** — the per-row
  design (one program per `(b, qhid, m)`, `tl.sum(k * q, axis=1)`)
  cannot reach tensor-core utilisation because there is only one
  query row per program. Benchmark on
  `progress/p31/bench_csa_attention_ep8.py`: **~41 ms** per CSA layer.
  Target: **< 6 ms**.
- **V4 attention BWD (`_v4_attention_bwd_kernel`)** — monolithic
  single-kernel design parallelises over m-blocks and atomic-adds
  `dK` / `dV` across `H` heads × multiple m-blocks. EP8 trace shows
  each of the five launches at **~31 ms**. Target: **< 15 ms**.
- **CSA BWD (`_v4_csa_attention_pool_sparse_bwd_kernel` +
  `_v4_attention_bwd_kernel` local split)** — already split after
  P31b; remaining cost is sparse-branch arithmetic intensity + local
  branch atomic contention. Benchmark (EP8 shape): **~26 ms**.
  Target: **< 15 ms**.

The cure for all three is the same FlashAttention-2 lesson:
multi-row tiles + `tl.dot` + remove cross-program atomics. P32 lands
the kernel rewrites that put those wins in tree, plus a new V4
attention microbenchmark so kernel iteration does not require a full
EP8 training round-trip.

### Tasks

0. **Task list refinement** — read the post-P31b CSA / V4 attention
   shape data from `progress/p31/bench_csa_attention_ep8.py` and the
   latest EP8 trace; pin the three single-kernel targets (CSA FWD
   < 6 ms, V4 attention BWD < 15 ms, CSA BWD < 15 ms) and the per-
   target hypothesis (multi-row tile / split BWD / sparse tuning).
   Refinement note lands at `progress/p32/refinement.md`.
1. **`progress/p32/bench_v4_attention_ep8.py`** — mirror
   `progress/p31/bench_csa_attention_ep8.py` for the dense / HCA
   `v4_attention` Triton path. Covers FWD + BWD-only timing for both
   `compress_ratio=0` (dense SWA) and `compress_ratio=128` (HCA
   split-mask) at the proxy EP8 shape. Argparse interface mirrors the
   CSA bench (`--batch`, `--heads`, `--seq-len`, `--head-dim`,
   `--swa-window`, `--dtype`, `--iters`, `--warmup`, `--profile`,
   `--trace-dir`, `--json-out`, `--no-sink`, `--mode {dense, hca}`).
2. **CSA FWD multi-row / tensor-core tile rewrite** — replace
   `_v4_csa_attention_pool_fwd_kernel`'s per-row design with a
   FlashAttention-2 style multi-row tile for the local SWA branch
   (`BLOCK_M=32`, `tl.dot(Q, K.T)`), plus either:

   - **(2a) single-kernel join** — keep the existing program grid
     `(cdiv(Sq, BLOCK_M), B * HQ)`, run the local SWA tile loop
     first, then walk the per-row `topk_idxs` for the sparse branch
     under the same online softmax. Per-row sparse load stays as the
     `tl.load(POOL, topk_idx)` scatter; only the local tile gets
     multi-row treatment. Maintenance cost: low (one kernel).
   - **(2b) LSE-merge split** — run a dense Triton attention for the
     local branch and a separate head-block sparse kernel for the
     pool branch (head-block sharing the per-query top-K gather
     across all `H` heads), then merge their `(out, lse)` via a
     small online-softmax merge kernel. Sink applied to the merged
     result. Maintenance cost: higher (three kernels) but unlocks
     head-block tile shared-pool reads.

   P32 ships (2a) first; if benchmark falls short of the < 6 ms
   target, escalate to (2b). Both keep the existing
   `v4_csa_attention_from_pool` Python API; the kernel rewrites stay
   behind that boundary.
3. **V4 attention BWD split-kernel rewrite** — split
   `_v4_attention_bwd_kernel` into two parallelisation modes:

   - **dQ kernel** — parallelise over m-blocks (same as today),
     re-materialise `P = exp(qk - lse)` per-tile, accumulate `dQ`
     in registers, write at end. Computes `dsink` here (it reuses
     the same `P_sink = exp(sink_h - lse)`). No atomics for `dQ`.
   - **dK/dV kernel** — parallelise over n-blocks (one program per
     `n_block × batch × head_k`), iterate m-blocks per program,
     re-materialise `P` and `dS`, accumulate `dK` / `dV` in
     registers, write at end. No atomics for `dK` / `dV`.

   Total compute doubles (`P` re-materialised twice per `(m, n)`
   tile) but atomic contention disappears. On MI355 atomic adds are
   slow relative to register accumulation, so the split design is a
   net win for `H=64` × SWA-window-pruned tile counts. Both kernels
   keep the existing SWA K-loop pruning + HCA split-mask mode from
   P30. The pre-pass `D` kernel stays as-is.
4. **CSA BWD picks up the V4 attention BWD split for its local
   branch** — the CSA BWD launcher already routes the local SWA
   contribution through `_v4_attention_bwd_kernel`; pointing it at
   the new split kernels also lowers CSA BWD. The sparse branch
   stays on `_v4_csa_attention_pool_sparse_bwd_kernel` with
   `BLOCK_K` / `BLOCK_H` re-tuned against the new benchmark.
5. **Unit-test ratchet** — re-run plan-4 G23..G28 fast + release
   tiers, P31 G34b fast + release pool/topk tests, and P25
   dispatch/log tests after each kernel change. Every kernel rewrite
   is committed only after both fast and release tiers stay green.
6. **EP8 proxy trace + profile report** — once all three targets are
   met on the microbenchmarks, capture an EP8 proxy trace with
   `progress/p32/run_baseline_trace_ep8_p32.sh` and render
   `develop/profile/profile-after-p32-ep8-<YYYYMMDD>.{md,html}`.
   Update `develop/perf/attention_perf.md` + `proxy_ep8.md`.
7. **P32 summary** — `progress/p32/p32-summary.md` follows the
   project-wide eight-section per-phase summary format (rule R2.1).

### Design notes

- **Why benchmark first.** Running the proxy EP8 training for every
  kernel experiment costs ~10 minutes per iteration. The
  microbenchmarks (CSA + V4 attention) finish each measurement in
  seconds, so the optimisation loop is bound only by kernel
  compilation time. Trace capture only happens at the end to confirm
  the proxy-level delta.
- **Multi-row tile vs SMEM at `head_dim=512`.** The local SWA
  branch with `BLOCK_M=BLOCK_N=32`, bf16 inputs uses
  `Q + K + V = 3 × 32 × 512 × 2 = 96 KiB` per program plus
  `[BLOCK_M, BLOCK_N]` qk / p tiles in fp32 — comfortably under
  MI355's 160 KiB SMEM budget. The sparse branch, if it stays on
  the per-row design, only allocates one `[BLOCK_K=32, D=512]` pool
  tile (32 KiB).
- **CSA FWD per-row → multi-row migration.** The `tl.sum(k * q,
  axis=1)` per-row dot product on AMD ROCm Triton cannot use the
  tensor cores. Switching to `tl.dot(Q, tl.trans(K))` with
  `BLOCK_M=32, BLOCK_N=32, head_dim=512` gives the same 4096
  visible local pairs at full MFMA throughput.
- **BWD split atomics cost vs recompute cost.** Re-materialising
  `P` twice per `(m, n)` tile costs one extra `tl.dot(Q, K.T)` and
  one `tl.exp` per tile. For SWA window = 128 with `BLOCK_M=BLOCK_N=32`,
  each `(m, n)` tile is hit by both kernels exactly once → 2x
  `Q @ K.T` work. Atomic-add into a 64 KiB `dK` / `dV` tile across
  `H=64` heads × `4` m-blocks per K position is ~256 atomic ops per
  K position; on MI355 each atomic_add is ~10 cycles, and these are
  serialised inside the cache, so removing them is a strict win.
- **Sparse BWD ceiling.** The sparse branch already uses head-block
  tiling and `tl.dot` (P31 follow-up). Remaining knobs are
  `BLOCK_K`, `BLOCK_H`, `num_warps`, and (if memory budget allows)
  software pipelined K-tile loads.

### Edge cases

- **Numerics: bf16 BWD tolerance.** The split-BWD design re-issues
  the QK^T matmul; bit-for-bit drift is expected within the existing
  G24 / G27 bf16 release-tier budget (`dq / dk / dv atol = 2e-1`,
  `dsink atol = 5e-2`). Tests must pass the existing budget; no
  budget relaxation in P32.
- **HCA split-mask BWD.** The dK/dV kernel must iterate the same
  `(local SWA window, pool keys)` set the FWD kernel uses; the P30
  split-mask FWD path runs two K loops (pruned local SWA + pool
  suffix). The new dK/dV kernel mirrors this.
- **CSA FWD sink interaction.** The single-kernel join (2a) shares
  the sink contribution with the local branch's online softmax. If
  P32 escalates to the LSE-merge variant (2b), sink is applied
  AFTER the merge so the final softmax denominator is correct.

### P32 close-out hand-off

Closes when:

1. `progress/p31/bench_csa_attention_ep8.py` reports FWD ≤ 6 ms and
   BWD ≤ 15 ms at the EP8 production shape.
2. `progress/p32/bench_v4_attention_ep8.py` reports BWD ≤ 15 ms for
   both cr=0 and cr=128 shapes.
3. Plan-4 G23..G28 + P31 G34b fast + release tiers all pass.
4. EP8 proxy trace + report show positive headline TFLOP/s/GPU vs
   P31b (`709.3` baseline from `develop/perf/proxy_ep8.md`) and no
   banned warnings.
5. `develop/perf/attention_perf.md` + `proxy_ep8.md` updated with
   the P32 row.
