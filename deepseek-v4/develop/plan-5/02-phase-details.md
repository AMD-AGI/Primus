# 02 — Plan-5 Phase Details

> Each phase below lists (a) the user request that motivates it, (b)
> the concrete tasks, (c) the design notes that the implementer must
> keep in mind, and (d) the edge cases / risks. Test gates live in
> `03-test-strategy.md`. P29 / P30 / P31 / P32 task lists are
> **seeded**; each phase opens with a "task list refinement" pass
> that revises the breakdown against the P28 trace.

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
which optimisations are in scope for P29 / P30 / P31 / P32. Without
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
   - **Per-phase improvement budgets** — set the X / Y / Z / W
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
   The tool is committed; it is reused for P32's
   `profile-final-ep8-<YYYYMMDD>.{md,html}`.

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
  that P29 / P30 / P31 / P32 cannot accidentally take work that the
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

---

## Phase 32 — Pipeline / comm / optimizer overlap + recompute knobs

> "也可以发挥你的分析能力，看看还有没有其他的优化可以做。" — user,
> plan-5 kick-off (optimisation hint #3).
> Plus plan-1 P6 follow-up: `--overlap_grad_reduce False
> --overlap_param_gather False` in `run_deepseek_v4.sh` was a
> plan-2 stability hedge that plan-4 G30 obsoleted.

P32 closes plan-5 with the comm / overlap re-enables surfaced by the
P28 trace, plus recompute granularity tuning if HBM headroom permits.
Lands last because comm overlap multiplies the gain from kernel-side
improvements; running it before the kernel phases would conflate
kernel-time wins with overlap wins on the per-phase delta.

### Tasks

0. **Task list refinement** — read the P28 report's comm row + HBM
   row; pick the overlap targets visible in the trace.
1. **Re-enable `--overlap_grad_reduce True --overlap_param_gather
   True`** in `run_deepseek_v4.sh` and the proxy. Plan-4 G30
   obsoletes the plan-2 stability hedge; G31 + G33 (post-P29 / P30 /
   P31) re-verify the overlap is correct.
2. **MoE shared-expert overlap** — currently `moe_shared_expert_overlap=
   False` (DeepEP contract from plan-3 P23). Investigate whether a
   stream-side overlap is feasible without violating the DeepEP
   contract; if not, leave as-is and document.
3. **Recompute granularity tuning** — current `recompute_num_layers=0,
   recompute_granularity=full, recompute_method=block`. If P31's
   in-kernel gather frees enough HBM headroom, drop recompute
   (`recompute_granularity=null`); the trace + the new memory
   floor in the P32 report show the trade-off.
4. **Final EP=8 trace** — at all P29 / P30 / P31 / P32 optimisations
   on; report at `develop/profile/profile-final-ep8-<YYYYMMDD>.{md,html}`
   reusing the P28 report tooling.
5. **Plan-5 hand-off note** — appended to `plan-5/02-phase-details.md`
   summarising commit chain P28..P32, gate totals, baseline → final
   TFLOP/s/GPU delta, follow-ups (FP8, multi-node EP, long-context,
   convergence).

### Design notes

- **Overlap re-enable is a smoke gate, not a kernel gate.** G31
  (smoke) catches overlap regressions; the existing plan-4 unit
  tests (G23 / G24 / G26 / G27 / G29) are TP=1 PP=1 EP=1 mock-tests
  and do not exercise the overlap.
- **Recompute trade-off** — dropping recompute frees HBM but doubles
  the activation memory inside the layer. P32's report owns the
  per-iter memory floor table so the trade-off is visible.

### Edge cases

- **DeepEP comm-stream interaction** — DeepEP runs its own comm
  stream when `turbo_deepep_use_comm_stream=True`; default is
  `False` (plan-3 P23). If P32 re-enables `moe_shared_expert_overlap`,
  the DeepEP comm-stream contract has to be re-verified; G31 + G33
  catch contract violations end-to-end.
- **Overlap + V4 hash-router PP broadcast** — plan-2 P19 added
  `pp_token_pre_broadcast` to push `input_ids` to PP rank > 0 ahead
  of the schedule. With `--overlap_grad_reduce True`, the broadcast
  has to land before the first `recv_forward.wait()` on PP rank > 0;
  the plan-2 P19 patch already enforces this (broadcast in the
  schedule wrapper, before any send/recv). G31 confirms.

---

## Plan-5 hand-off note (status — closes plan-5)

> _(Filled in at P32 commit time, mirrors the plan-4 P27 hand-off
> block.)_

- Commit chain P28 → P32: `TBD-p28` → `TBD-p29` → `TBD-p30` →
  `TBD-p31` → `TBD-p32`.
- Gate totals: G31 / G32.{a..e} / G33 / G34 / G35 + plan-4 ratchet
  (G23 / G24 / G25 / G26 / G27 / G28 / G29 / G30) all green at the
  final commit.
- Baseline (P28) → final (P32) steady-iter TFLOP/s/GPU delta:
  `TBD-baseline` → `TBD-final` (= `+TBD %`); per-phase share of the
  delta tabulated in the P32 report.
- Follow-ups (out of scope, owned by future plans): FP8 / FP4 /
  mxfp4 quantised forward; multi-node EP scaling; long-context
  (1M-token) bring-up; convergence run; HF state-dict adapter.
