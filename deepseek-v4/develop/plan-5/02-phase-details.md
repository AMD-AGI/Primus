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

## Phase 29 — Small-op fusion targets picked from the P28 trace

> "我之前看到trace里面，有很多小kernel, kernel launch开销很大，cpu
> idle占比很高。因此，这个是一个优化重点，可以查看是哪些模块的小算子，
> 是否可以通过torch.compile或者triton来开发fused版本。" — user, plan-5
> kick-off (optimisation hint #1).

P29 attacks the kernel-launch / CPU-idle tail named in the P28 report.
The seeded targets below match the V4 attention small-op chain that
runs eager Python today; they get **picked or dropped** in the P29
"task list refinement" pass against the actual P28 trace. A target
that the trace shows is < 1 % of step time is dropped before
implementation.

### Tasks

0. **Task list refinement** — read the P28 report's bottleneck list,
   pick the top-K small-op chains by total CPU time (typical: K = 3),
   add or drop targets in the seeded list below. The chosen list is
   recorded as a `### P29 task list refinement` block at the top of
   the corresponding `progress/p29/` notes.
1. **Q-projection chain (candidate a)** — fuse
   `linear_q_down → q_layernorm → linear_q_up → _per_head_rms_norm
   → apply_interleaved_partial_rope` into a single callable
   `v4_fused_q_proj(hidden, q_down_w, q_up_w, q_norm_w, q_pe_freqs,
   eps) -> (q_nope, q_pe)`. Two implementation options; pick at
   refinement time:
   - **torch.compile** — wrap the pure function with `torch.compile`
     and rely on the Inductor backend to fuse pointwise ops + emit
     fused matmul + RoPE kernels.
   - **In-tree Triton** — write a single-program-per-row kernel that
     does the matmul / RMSNorm / partial-RoPE in registers (analogous
     to the per-row design in plan-4 P26).
2. **KV-projection chain (candidate b)** — fuse
   `linear_kv → kv_layernorm → split_qk_pe` (single-latent KV projection
   plus the position-embedding axis split). Same two implementation
   options as candidate (a).
3. **O-projection group (candidate c)** — fuse
   `attn_out → reshape → linear_o_a → linear_o_b` (grouped low-rank
   output projection — currently two separate matmuls + an explicit
   reshape). Lower priority because matmuls are big; but if the
   reshape contributes a hot pointwise kernel in the P28 trace, the
   fusion picks up the reshape + linear_o_a chain.
4. **Compressor + Indexer pre-attention chain (candidate d)** —
   for cr=4 / cr=128 layers, the Compressor (small linear + softmax
   + scatter) and Indexer (small linear + topk + gather) launch
   ~6–10 small kernels per layer per step. Fuse the compressor's
   linear + softmax into a single Triton kernel; fuse the indexer's
   linear + topk into a single Triton kernel.
5. **MoE router (candidate e)** — fuse the gate-softmax + topk +
   permute-mask chain inside the MoE router (NOT touching the
   DeepEP dispatch boundary). Asserts `moe_router_dtype=fp32` is
   preserved (DeepEP contract from plan-3 P23).
6. **Per-target switches** — every fusion lands behind its own
   boolean (`use_v4_fused_q_proj`, `use_v4_fused_kv_proj`,
   `use_v4_fused_o_proj`, `use_v4_fused_compressor`,
   `use_v4_fused_indexer`, `use_v4_fused_moe_router`) defaulting
   to `False`. The proxy script (P28) flips them to `True` after
   the corresponding G32.{a..e} gate is green.
7. **Per-target equivalence test (G32.{a..e})** — for each chosen
   fusion, add a `tests/unit_tests/megatron/transformer/deepseek_v4/
   test_v4_p29_<target>.py` with forward + backward equivalence
   (eager small-op vs fused) at V4-Flash + V4-Pro shapes, fast tier
   + release tier (release marked `pytest.mark.slow`). Reuses the
   plan-4 `compare_fwd_bwd` harness.
8. **End-to-end smoke (G33)** — re-run the proxy with all fusions
   on; assert ≥ +X % TFLOP/s/GPU vs P28 baseline. X comes from the
   P28 report.

### Design notes

- **Functional fusion, not module fusion.** Fuse pure functions, not
  `nn.Module.forward`. `torch.compile`-wrapping `DeepseekV4Attention.forward`
  closes over `self._submodules`, `self._rope`, `pg_collection`,
  `self.layer_number`, etc., which Megatron's spec walker rebinds
  at construction time → recompile-on-every-rebuild or silent miscompile.
  Functional sub-chains (e.g. `q_proj_chain(hidden, q_down_w,
  q_up_w, q_norm_w, q_pe_freqs)`) close over only the leaf tensors,
  so they compile once and stay valid for the whole run.
- **torch.compile vs hand-written Triton.** The decision is per-target,
  picked at refinement time:
  - `torch.compile`: lower author cost, no autotune, but Inductor
    sometimes emits suboptimal kernels at `head_dim=512` (long
    register pressure, large SMEM); the cost is the perf delta
    vs hand-tuning. Fast to land, easy to revert.
  - Hand-written Triton: higher author cost, autotune-able, but a
    ground-truth kernel that the team owns end-to-end (no Inductor
    version drift). Slow to land, hard to revert.
  Defaults: torch.compile for candidates (b), (c), (d), (e); hand
  Triton for (a) — Q-projection has both the partial-RoPE and the
  per-head RMSNorm, which Inductor historically struggles to fuse
  cleanly at `head_dim=512`.
- **Cold-compile cost** — `torch.compile` first-iter compile time
  can run minutes per fused chain; the proxy's iter 0 cost will
  jump materially. The report uses steady-iter throughput, but
  total wall-time-to-first-iter matters for short jobs. Document
  the cold-compile cost in the per-target G32 status row.
- **Backward path** — every fusion MUST also fuse the BACKWARD,
  not just the forward. `torch.compile` handles this via Inductor's
  AOT autograd; hand-Triton fusions ship a matching `_bwd` kernel
  alongside.
- **No shared-state mutation.** A fused chain that writes into
  `self.<some buffer>` (e.g. an `attn_dropout_mask`) breaks the
  pure-function contract. The plan-5 fusions all run on stateless
  pure-tensor inputs/outputs; if the trace surfaces a fusion
  candidate that mutates module state, it gets refactored to a pure
  function first.

### Edge cases

- **Per-layer compress_ratio dispatch** — candidates (d) and (e) only
  apply to a subset of layers (cr=4 / cr=128 for compressor+indexer;
  every MoE layer for router). The fusion switch is checked per layer
  in `DeepseekV4Attention.__init__` and auto-disabled for the wrong
  layer kind (mirrors the plan-4 `_use_v4_triton_attention` /
  `_use_v4_triton_csa_attention` auto-disable pattern).
- **State-dict compatibility** — fused weights MUST keep the same
  state-dict keys as the unfused chain (`linear_q_down_proj.weight`
  + `linear_q_up_proj.weight` + `q_layernorm.weight` etc.) so plan-2
  P17's checkpoint-load path stays unbroken. The fused functions
  read the unfused weights at call time; they do NOT introduce a
  fused weight tensor.

---

## Phase 30 — V4 Triton attention kernel perf tuning

> "如果小kernel问题解决了，大概率attention可能是后续的瓶颈之一。可以
> 制定一些关于v4 triton的两个attention算子的性能优化。" — user,
> plan-5 kick-off (optimisation hint #2).

P30 tunes the in-tree dense / HCA `v4_attention` Triton kernel. The
plan-4 kernel landed at conservative `BLOCK_M=BLOCK_N=32, num_warps=8,
num_stages=1` to fit MI355's 160 KiB SMEM budget at `head_dim=512`;
P30 explores per-shape autotune for the configurations that fit, plus
the structural HCA LSE-merge variant that plan-4 deferred.

### Tasks

0. **Task list refinement** — read the P28 report's attention-kernel
   row; if attention time is < 10 % of step time at the post-P29
   measurement, P30 gets de-scoped to "autotune table only" (no
   structural redesign).
1. **Per-shape autotune table (FWD)** — extend
   `_triton/v4_attention_fwd.py` with a `triton.autotune` decorator
   keyed on `(H, head_dim, swa_window, has_add_mask, has_sink)` over
   a config grid: `BLOCK_M ∈ {32, 64, 128}, BLOCK_N ∈ {32, 64,
   128}, num_warps ∈ {4, 8}, num_stages ∈ {1, 2}`. SMEM budget
   constraint encoded: configs whose runtime SMEM exceeds 160 KiB
   are pruned at compile time via `@triton.heuristics`.
2. **Per-shape autotune table (BWD)** — extend
   `_triton/v4_attention_bwd.py` analogously; tighter SMEM budget
   because BWD also holds `dout`.
3. **Persistent FWD kernel** — drop the per-`m`-tile launch by
   running one program per `(B, HQ)` and looping over `m`-tiles
   in-kernel. Pays off at long sequence lengths where launch
   overhead is a non-trivial fraction of kernel time.
4. **HCA LSE-merge variant** — implement a second forward path
   (`v4_attention_lse_merge`) that runs SWA over `[Sq, Sq]` and the
   compressed-pool branch over `[Sq, P]` as two flash kernels and
   merges via online softmax. Avoids materialising the
   `[Sq, Sk] = [Sq, Sq + P]` additive bias tensor (plan-4 P25
   takes this materialised path because it is simpler). The LSE-merge
   variant is the structural HCA optimisation.
5. **`use_v4_attention_lse_merge` switch** (default `False`) +
   gate G34 (FWD + BWD equivalence between the two variants
   within the bf16 tolerance budget).
6. **In-kernel SWA mask path** stays where it is (plan-4 P25 already
   took it). P30 only revisits if the P28 trace shows SWA-mask CPU
   cost as a hot spot.

### Design notes

- **Autotune cache pinning** — the autotune cache lives in-process
  per `(shape, dtype)` key and gets re-evaluated on every fresh
  process. For long training jobs the cost amortises; for short
  jobs it compounds. Plan-5 documents the autotune warmup cost in
  the G34 / G33 status rows.
- **HCA LSE-merge correctness gate** — the LSE-merge variant has
  to **exactly** match the single-kernel-with-additive-bias
  variant (within bf16 tol) on the bf16 path. The merge step uses
  `m_merged = max(m_swa, m_pool); l_merged = l_swa * exp(m_swa -
  m_merged) + l_pool * exp(m_pool - m_merged); o_merged =
  (o_swa * exp(m_swa - m_merged) * l_swa + o_pool * exp(m_pool -
  m_merged) * l_pool) / l_merged`. Sink contributes only to
  `l_merged` (virtual key with zero V) — the BWD propagates `dsink`
  per the plan-4 P25 contract.
- **Persistent kernel + autotune interaction** — persistent kernels
  benefit less from per-shape `BLOCK_N` autotune (the m-loop is
  inside the kernel, so `BLOCK_N` mostly controls tile time, not
  launch count). Autotune the persistent kernel separately;
  document the chosen tile in the G34 status row.

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
