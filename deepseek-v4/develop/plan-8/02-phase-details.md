# 02 — Plan-8 Phase Details

## Phase 49 — Tilelang infra + dispatcher

> Sourced from the plan-8 README §3 ("Why tilelang") and the user
> kick-off note: "在容器里面，我已经编译了 tilelang，可以直接使用".

### Tasks

1. **Pinned tilelang version + venv probe.**  Read
   `tilelang/VERSION` (currently `0.1.9+cuda.gitbcb2da33`) and pin
   it in `progress/p49/p49-summary.md`.  Add a one-shot import
   probe to
   `primus/backends/megatron/core/transformer/v4_attention_kernels/_tilelang/__init__.py`
   that raises `ImportError` with the pinned version if the
   installed tilelang ≠ pin.  Tilelang is already installed inside
   `dev_primus_wenx_693` (per the user); no `pip install` step in
   the build script.

2. **Stub module +
   `is_tilelang_path_enabled()` predicate.**
   `_tilelang/__init__.py` exposes:

   ```python
   def is_tilelang_path_enabled() -> bool:
       """True iff PRIMUS_V4_TILELANG_ATTN == '1'.  Default OFF."""
       return os.environ.get("PRIMUS_V4_TILELANG_ATTN", "0") == "1"
   ```

   Plus stubs `v4_attention_fwd_tilelang`,
   `v4_attention_bwd_tilelang`,
   `v4_csa_attention_fwd_tilelang`,
   `v4_csa_attention_bwd_tilelang` that raise
   `NotImplementedError` until the corresponding plan-8 phase
   lands them.

3. **Dispatch precedence wiring.**  Update
   `DeepseekV4Attention.forward` (and the two helpers
   `_attention_forward_via_triton` /
   `_attention_forward_via_csa_triton`) so:

   * **cr=0 / cr=128**: `use_turbo_attention >
     PRIMUS_V4_TILELANG_ATTN > use_v4_triton_attention > eager`.
   * **cr=4**: `PRIMUS_V4_TILELANG_ATTN >
     use_v4_triton_csa_attention > eager`.

   The dispatcher emits a one-time rank-0 log line per kernel
   kind when tilelang is engaged ("v4_attention engaged via
   tilelang FWD/BWD"), mirroring the plan-4 P27 startup-log
   convention.  Banned-warning ratchet stays clean.

4. **Cache directory layout.**  Tilelang autotune caches per-shape
   kernel binaries under `output/.tilelang_cache/v4/`.  Add a
   `.gitignore` entry under `output/` plus an env override
   `PRIMUS_V4_TILELANG_CACHE_DIR` (defaults to the above).  The
   wrapper sets `TILELANG_CACHE_DIR` / equivalent before
   importing tilelang.

5. **Build script.**
   `deepseek-v4/develop/progress/p49/build_tilelang_kernels.sh`
   AOT-compiles every shape variant the plan-8 kernels can hit:

   * dense FWD/BWD: `(B=1, H=64, Sq=Sk=4096, D=512, has_sink ∈
     {True, False}, swa_window ∈ {128, 0}, hca_local_seqlen ∈
     {0, 4096})`.
   * CSA FWD/BWD: `(B=1, H=64, Sq=4096, K_topk=512, D=512,
     swa_window=128)`.

   Skips phases whose kernels haven't landed yet (no-op for
   `NotImplementedError`).

6. **G49 ratchet check.**  `pytest -q
   tests/unit_tests/megatron/transformer/deepseek_v4/` (with
   `--run-slow` for release-tier gates) before + after the P49
   wiring lands.  Pass count must not drop.  Plan-4..plan-7
   banned-warning grep returns 0.

### Design notes

- **Why a single env knob, not per-family.**  The user's intent
  (one tilelang kick-off) reads as "ship all three families
  behind one switch".  Per-family knobs (`PRIMUS_V4_TILELANG_DENSE
  / _HCA / _CSA`) add 3 default-off bits + 3 banned-warning
  exemptions.  P49 keeps one knob; future phases can split if a
  family regresses while another wins.
- **Why dispatcher precedence above the Triton flag.**  Once a
  tilelang kernel ships + parity passes + proxy A/B wins, the
  natural production state is "tilelang on, Triton flag still
  available as a debug fallback".  The dispatcher orders tilelang
  above Triton so the env knob is sufficient to flip back.

### Edge cases

- **`tilelang` not importable** — the stub raises a clear
  `ImportError` with the installed-vs-pinned versions printed.
  Plan-8 phases that the user runs without tilelang installed
  produce a single rank-0 warning + fall through to Triton.
- **`compress_ratio` not in {0, 4, 128}** — the dispatcher
  refuses to route through tilelang and falls back to Triton
  (mirrors the plan-4 P27 precedent for unsupported cr values).

---

## Phase 50 — Dense FWD tilelang (cr=0)

> Sourced from `tilelang/examples/amd/example_amd_flash_attn_fwd.py`
> (MI355X-tuned FlashAttention v2) and
> `tilelang/examples/attention_sink/example_mha_sink_fwd_bhsd.py`
> (sink fusion at end of softmax).

### Tasks

1. **Kernel +
   wrapper.**  New `_tilelang/v4_attention_fwd_tilelang.py`:

   - `@tilelang.autotune(configs=...)` over `block_M ∈ {16, 32}`,
     `block_N ∈ {16, 32, 64}`, `num_stages ∈ {0, 1, 2}`,
     `threads ∈ {128, 256}`, `k_pack ∈ {1, 2}`, `panel_size ∈
     {7, 8}`.  Conservative starting set; SMEM-budget-limited at
     `head_dim=512`.
   - `@tilelang.jit(out_idx=[3, 4])` returning `(out, lse)`.
   - Inputs: `Q [B, H, Sq, D]`, `K [B, K_H, Sk, D]`, `V [B, K_H,
     Sk, D]` (K_H ∈ {1, H} for MQA / MHA), optional `Sink [H]`,
     optional `AddMask [Sq, Sk]`, scalars `(scale, swa_window,
     hca_local_seqlen)`.
   - Per-program tile: `(b_split, byz_combined)` so each program
     covers `BLOCK_M` queries × the full Sk loop pipelined over
     `BLOCK_N`.
   - Online softmax over `acc_s = Q @ K^T`; track `m_running`,
     `l_running`, `acc`; apply optional SWA + additive bias;
     apply sink as a virtual key column at end-of-K (same
     formulation as the existing attention_sink example).
   - Output: `out` (q.dtype), `lse` (fp32).

2. **Wrapper API.**  `v4_attention_fwd_tilelang(q, k, v, *, sink,
   swa_window, additive_mask, scale, hca_local_seqlen=0)`
   matches the Triton launcher's signature for drop-in
   replacement.  Inside the existing `V4AttentionFn` autograd
   Function, the FWD dispatcher chooses tilelang vs Triton based
   on the env knob.

3. **Microbench
   `progress/p50/bench_v4_attention_fwd_tilelang.py`** — V4-Flash
   widths + smoke shape.  Reports `<ms> ms | <effective TFLOP/s>`
   per the R2.5 cell format.  Compares tilelang vs Triton FWD at
   identical shapes.

4. **G50 unit tests** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_p50_v4_attention_fwd_tilelang.py`:

   - FWD `out` + `lse` parity vs the plan-4 G23 eager reference
     (`reference.py::eager_v4_attention`) within bf16 `atol=2e-3
     rtol=2e-3` at fast tier (`B=2, H=4, Sq=Sk=64, D=64`) and
     release tier (`B=1, H=64, Sq=Sk=4096, D=512,
     pytest.mark.slow`).
   - Parametrise `(MQA, MHA) × (sink, no_sink) × (SWA, full) ×
     (additive_mask, no_mask)`.

5. **Smoke EP=8.**
   `progress/p50/run_smoke_p50_dense_ep8.sh` with
   `PRIMUS_V4_TILELANG_ATTN=1` for cr=0 layers only; HCA / CSA
   still on Triton.  10 iters, lm_loss within `5e-2` of the
   plan-7 P48 baseline.

### Design notes

- **MQA broadcast.**  When `K_H == 1`, the kernel indexes `K /
  V` with `by // groups` (where `groups = H / K_H = H`).  The
  shared K/V head is repeated across query heads via the per-
  program index calculation, not via `K.repeat_interleave` (which
  would materialise a full `[B, H, Sk, D]` copy — same trick the
  Triton kernel uses).

- **Sink + virtual key column.**  Following the
  `attention_sink` example:

  ```python
  for i in T.Parallel(block_M):
      logsum[i] += T.exp2(sinks[i] * 1.44269504 - scores_max[i] * scale)
  ```

  This adds the sink to the denominator only — its "value" is 0,
  so no extra term is added to `acc_o`.  Matches the existing
  Triton kernel formulation.

- **`exp2` vs `exp`.**  The
  `attention_sink/example_mha_sink_fwd_bhsd.py` reference uses
  `exp2(x * 1.44269504 - max * scale)` to map to FFMA on MFMA;
  P50 follows.  The plan-4 P25 Triton kernel uses `exp` for
  parity-readability reasons; the bf16 numerical contract
  (`atol=2e-3`) covers either.

### Edge cases

- **`head_dim=512` SMEM budget.**  At `block_M=32, block_N=32, D=512,
  dtype=bf16`: Q-shared = 32 * 512 * 2 = 32 KiB; K-shared / V-
  shared each 32 KiB; acc-shared 4 KiB.  Cumulative ~100 KiB, fits
  under the 160 KiB MI355 budget.  `block_M=64` would push past
  that; the autotuner is hand-bounded.

- **`hca_local_seqlen` parameter.**  P50 ships the parameter but
  the HCA split-mask path is engaged only by P52.  Default `0`
  in the dense FWD; passing it through preserves the wrapper
  signature.

---

## Phase 51 — Dense BWD tilelang (cr=0)

> Borrows from `tilelang/examples/amd/example_amd_flash_attn_bwd.py`
> (MI355X-tuned FlashAttention v2 BWD) + the plan-5 P32 final
> Triton split-BWD design.

### Tasks

1. **Preprocess kernel** —
   `_tilelang/v4_attention_bwd_tilelang.py::preprocess_kernel`
   computes `Delta = (O * dO).sum(-1)` per `(b, h, m)` per the
   FlashAttention BWD recipe.  One Triton-equivalent loop with
   `T.reduce_sum`.

2. **`dq_kernel`** — one program per `(b, h, m_tile)`; loads `Q,
   dO, O, LSE, sink` + iterates over k_tile; re-materialises `P
   = exp(qk * scale - lse)` from saved `LSE`; computes `dP = dO @
   V^T`; back-prop softmax: `dS = P * (dP - delta[:, None])`;
   accumulates `dQ = dS @ K * scale` in fp32.  At end-of-loop:
   write `dQ` cast to bf16.

3. **`dkv_kernel`** — one program per `(b, k_h, n_tile)`; loops
   over m_tile; re-materialises `P` from saved `LSE`; computes
   `dV = P^T @ dO` and `dK = dS^T @ Q * scale`.  MQA case
   accumulates across query heads inside the program (no
   cross-block atomic_add needed).

4. **`dsink_kernel`** (only when `has_sink`) — one program per
   `h`; reduces sink gradient via `dsink[h] = sum_b sum_m
   exp(sink[h] - lse[b, h, m]) * delta[b, h, m] * scale`.  Tiny;
   one launch.

5. **Wrapper** — `v4_attention_bwd_tilelang(...)` matches the
   Triton launcher signature; returns `(dQ, dK, dV, dSink)`.
   Used by `V4AttentionFn.backward` when the env knob is set.

6. **Microbench
   `progress/p51/bench_v4_attention_bwd_tilelang.py`** at
   V4-Flash widths.

7. **G51 unit tests**:

   - FWD-then-BWD parity vs `reference.py::eager_v4_attention`
     within bf16 `atol=5e-3 rtol=5e-3` (FWD `atol=2e-3 rtol=2e-3`,
     BWD `atol=5e-3 rtol=5e-3`).
   - `torch.autograd.gradcheck` at fast-tier fp32 small shape.
   - Release-tier slow at V4-Flash widths.

### Design notes

- **Split BWD vs monolithic BWD.**  Plan-5 P32 final shipped the
  split BWD on the Triton side as the production default — the
  monolithic BWD's cross-block atomic_add traffic on `dK / dV`
  cost ~190 ms / iter at V4-Flash widths.  P51 ships split BWD
  from day one (no monolithic variant).

- **`Delta` precompute as a separate kernel.**  Tilelang's
  `example_amd_flash_attn_bwd.py` does the same — the preprocess
  is a tiny single-launch reduction that keeps the dQ / dKV
  kernels simple.

### Edge cases

- **Recomputed `P` vs saved `P`.**  FlashAttention recipe stores
  only `LSE [B, H, Sq]` and re-materialises `P [BLOCK_M,
  BLOCK_N]` per-tile.  Saves `H * Sq * Sk * 2 bytes ≈ 4 GiB` of
  HBM at V4-Flash widths.

- **Sink BWD edge case** — when `has_sink` but the eager
  reference's sink grad is zero (`sink[h]` is at -inf, no
  contribution), the tilelang kernel emits exactly zero too.

---

## Phase 52 — HCA FWD tilelang (cr=128)

> Reuses P50 dense FWD with `hca_local_seqlen` engaged + an
> additional `[Sq, Sk_pool]` additive bias.

### Tasks

1. **Kernel extension.**  P52 is **not** a new kernel — it
   parametrises P50's FWD kernel with `hca_local_seqlen >
   0`:

   - First `hca_local_seqlen` queries hit kernel-native SWA on
     `k_local` (the first `hca_local_seqlen` entries of `K /
     V`).
   - Remaining `Sq - hca_local_seqlen` queries hit the compressed
     pool half of `K / V` (the trailing `Sk - hca_local_seqlen`
     entries) with the caller-supplied
     `additive_mask [Sq, Sk_pool]`.
   - Joint softmax merges all candidates plus the per-head sink.

   The plan-4 P25 Triton kernel does this via a constexpr branch
   on `HCA_LOCAL_SEQLEN`; P52 mirrors with a `T.if_then_else`.

2. **Wrapper.**  No new API — same
   `v4_attention_fwd_tilelang(...)`, just pass
   `hca_local_seqlen > 0` + `additive_mask` (pool-only `[Sq_local,
   Sk_pool]`).

3. **G52 unit tests** — extend the P50 G50 release tier with
   parametrisations for `(hca_local_seqlen ∈ {0, 4096})` + the
   pool-only additive mask path.  Fast + release-tier slow.

### Design notes

- **Pool-only additive mask shape.**  Convention from plan-4
  P25: `additive_mask [Sq_local, Sk_pool]` not `[Sq, Sk]`.  The
  kernel computes `local_pos < hca_local_seqlen` to determine
  which side of the split a query belongs to.

### Edge cases

- **Degenerate `hca_local_seqlen == 0`** — collapses to dense
  attention with optional `[Sq, Sk]` additive mask (P50 path).
- **Degenerate `hca_local_seqlen == Sq`** — collapses to dense
  SWA attention (no pool branch).  Matches the plan-5 P30b
  precedent.

---

## Phase 53 — HCA BWD tilelang (cr=128)

> Extends P51 split BWD with the HCA split-mask BWD path.

### Tasks

1. **Kernel extension.**  Similar to P52, P53 is not a new
   kernel — it parametrises the P51 BWD kernels with
   `hca_local_seqlen > 0`:

   - `dq_kernel` covers both local + pool branches per query
     (the joint softmax means `dQ` accumulates both `dP_local @
     K_local + dP_pool @ K_pool`).
   - `dkv_kernel` for `dk_local / dv_local` covers only the local
     part of `K / V` (entries `[0 : hca_local_seqlen)`).
   - A second `dkv_kernel` invocation covers `dk_pool / dv_pool`
     for the trailing part (entries `[hca_local_seqlen : Sk)`).
   - Sink BWD reuses P51.

2. **G53 unit tests** — parity vs `reference.py::eager_v4_attention`
   HCA path within bf16 `atol=5e-3 rtol=5e-3`.  `gradcheck`
   fp32 fast tier.

### Design notes

- **Two-pass dKV.**  The pool keys' BWD reads the trailing half
  of `K / V` — running it as a second kernel invocation (with a
  different `Sk_offset`) keeps both passes' SMEM footprint
  bounded.

---

## Phase 54 — CSA FWD tilelang (cr=4)

> Sourced from `tilelang/examples/dsa_sparse_finetune/sparse_mla_fwd.py`
> (sparse-gather pattern at MLA scale).  Plan-4 P26 + plan-5 P31
> design notes carry over for the joint local + sparse + sink
> softmax structure.

### Tasks

1. **Kernel
   `_tilelang/v4_csa_attention_fwd_tilelang.py`.**

   - Inputs: `Q [B, H, Sq, D]`, `K_local [B, H, Sq, D]`,
     `V_local [B, H, Sq, D]`, `Gathered [B, Sq, K_topk, D]`,
     `SparseMask [B, Sq, K_topk]`, `Sink [H]`, scalars `(scale,
     swa_window)`.
   - Grid: `(seq_len, batch * heads)` (one program per `(b, h,
     m)` row tile — single-row sparse tile to keep SMEM under
     32 KiB at `head_dim=512`).
   - Inner loop 1 (local SWA): pipeline over k_tile in
     `[max(0, m - swa_window) // block_N, ceildiv(m + 1,
     block_N))`; per-tile softmax update.
   - Inner loop 2 (sparse top-K): pipeline over `K_topk //
     block_N`; for each tile, gather K/V rows from `Gathered` (or
     from `pool` indirectly via `topk_idxs` — see the
     `v4_csa_attention_from_pool` wrapper); apply `sparse_mask` as
     additive bias; per-tile softmax update.
   - Final: apply sink as virtual key column; normalise; cast
     output to bf16.

2. **Wrappers.**  Two public entry points matching the existing
   Triton API:

   - `v4_csa_attention_fwd_tilelang(q, k_local, v_local, gathered,
     *, sink, swa_window, sparse_mask, scale)` — caller has
     pre-gathered.
   - `v4_csa_attention_fwd_tilelang_from_pool(q, k_local, v_local,
     pool, *, topk_idxs, sink, swa_window, scale)` — kernel
     gathers in-loop from `pool[B, P, D]` (plan-5 P31 in-kernel
     gather pattern; saves the pre-gather pass).

3. **Microbench
   `progress/p54/bench_v4_csa_attention_fwd_tilelang.py`** at
   V4-Flash CSA widths.

4. **G54 unit tests** — parity vs
   `reference.py::eager_v4_csa_attention`, `atol=2e-3 rtol=2e-3`.

### Design notes

- **In-kernel gather vs pre-gathered.**  Plan-5 P31 shipped the
  in-kernel gather as the default Triton path (saves the
  `[B, Sq, K_topk, D]` intermediate at ~32 GiB).  P54 ships both
  variants behind the same env knob; the wrapper picks the
  in-kernel-gather variant when the caller passes `pool` +
  `topk_idxs` (the production path), and the pre-gathered
  variant for unit-test friendliness.

- **Single-row sparse tile.**  Plan-5 P31 forensics established
  that per-row programs are the only fit at `(head_dim=512,
  K_topk=512)` — multi-row tiles balloon SMEM to ~1 MiB which is
  over MI355's budget.  P54 follows.

### Edge cases

- **`K_topk == 0`** — wrapper short-circuits to
  `v4_attention_fwd_tilelang(..., additive_mask=None)` (the dense
  + sink + SWA path collapses to CSA's local-only branch when
  no sparse keys exist).  Mirrors the plan-5 P31 Triton wrapper.
- **`topk_idxs == -1`** — kernel uses `sparse_mask = -inf` for
  the masked entries; their probability mass goes to zero.

---

## Phase 55 — CSA BWD tilelang (cr=4)

> Extends P51 split BWD design with the CSA segreduce pool BWD
> from plan-5 P32 final.

### Tasks

1. **`_tilelang/v4_csa_attention_bwd_tilelang.py`** — 3-kernel
   pipeline:

   - **`dq_kernel`** — one program per `(b, h, m_tile)`;
     re-materialises `P` for both branches from saved `LSE`;
     emits `dQ`.
   - **`dkv_local_kernel`** — one program per `(b, k_h,
     n_local_tile)`; emits `dK_local / dV_local`.  MQA case
     accumulates across query heads inside the program (no
     atomic_add).
   - **`dpool_segreduce_kernel`** — segreduce variant for
     `dpool` (the `[B, P, D]` pool's gradient that the in-kernel
     gather scattered into).  Uses the **sorted-inverse-index**
     trick from plan-5 P32 final (`PRIMUS_V4_CSA_BWD_SEGREDUCE`):
     scatter `dK_pool` partials into a 4 GiB partial buffer
     sorted by `topk_idxs`, then run a single tilelang reduce
     pass.  Avoids the per-element atomic_add at the sparse-write
     site.

2. **Sink BWD** — reuses P51's `dsink_kernel`.

3. **Microbench + G55 unit tests** — parity vs
   `reference.py::eager_v4_csa_attention`, `atol=5e-3 rtol=5e-3`.
   `gradcheck` fp32.

### Design notes

- **Sparse BWD memory layout.**  Plan-5 P32 final established
  that the segreduce pool-BWD path lost to the gather + atomic
  pool-BWD on EP=8 proxy (40 ms regression because the 4 GiB
  partial buffer's HBM traffic competes with MoE work).  P55
  ships **both** variants behind env sub-knobs
  (`PRIMUS_V4_TILELANG_CSA_BWD_SEGREDUCE=1` for the segreduce
  path).  The microbench wins are real (`35.43 → 16.31 ms` at
  V4-Flash widths per plan-5 P32 final §1.2.5); the wrapper
  defaults to the gather + atomic path until P56 close-out's
  bake-off confirms a different default.

### Edge cases

- **`K_topk == 0`** — wrapper short-circuits to the dense BWD
  (same as the FWD path).
- **`topk_idxs == -1`** for some entries — the segreduce path
  filters them out via the sorted-inverse index (entries with
  `idx == -1` map to a sentinel position that's discarded).

---

## Phase 56 — Plan-8 close-out

> Hand-off phase.  No new kernels.

### Tasks

1. **`develop/perf/attention_perf.md`** — append plan-8 rows
   (one per kernel × FWD / BWD).  Cell format per R2.5:
   `<ms> ms | <tflops>`.  Header references plan-4 P25 / P26
   anchors for FWD / BWD comparison.

2. **`develop/perf/proxy_ep8.md`** — append `P50..P55 individual`
   rows + `P56 final` cumulative row pinned to the 15-iter clean
   bake-off steady iter.

3. **`progress/p49/p49-summary.md` ... `progress/p56/p56-summary.md`** —
   one R2.1 eight-section summary per phase, written at the time
   each phase commits.  P56's summary doubles as the plan-8
   hand-off note.

4. **Status pinning per R2.4** — every `[x]` row in Phase 49..56
   of `progress/status.md` gets the commit SHA + date pinned.

5. **`run_deepseek_v4_flash_proxy.sh`** — surface the
   `PRIMUS_V4_TILELANG_ATTN` env knob.  Header note explains the
   precedence (tilelang > Triton > eager).  Default `"1"` only if
   the P56 bake-off confirms ≥ 30 ms / iter saved vs the P48
   anchor; else default `"0"`.

6. **15-iter clean bake-off** —
   `progress/p56/run_smoke_p56_bakeoff.sh` with all three
   tilelang kernel families on.  Steady iter time + TFLOP/s/GPU
   per iter; report in `progress/p56/p56-summary.md` §3.

7. **Plan-8 close-out commit** — final commit message follows
   the plan-7 P48 convention: `docs(deepseek-v4)[plan-8][P56]:
   plan-8 close-out — attention_perf + proxy_ep8 rows + status
   pinning`.

### Design notes

- **Why one knob, not one per family.**  Production state is
  "all tilelang on" or "all tilelang off" — partial state (e.g.
  dense on, CSA off) is reserved for debugging / regression
  diagnosis, achievable via the per-family Triton flags
  (`use_v4_triton_attention=False` for the dense path, etc.).

### Edge cases

- **A plan-8 phase regresses end-to-end.**  Per the plan-5 P32
  RoPE bug precedent + plan-7 P45 microbench evidence: the
  phase ships with `PRIMUS_V4_TILELANG_ATTN=0` and the phase row
  in `status.md` notes the regression.  The
  `attention_perf.md` row still shows the microbench number (the
  bench is honest); the `proxy_ep8.md` row records the proxy
  regression.

- **A plan-8 phase descopes at task-list refinement.**  The
  phase row in `status.md` is marked `[-]` per R2.2.  The
  `p4X-summary.md` documents why (e.g. "tilelang SMEM budget
  exceeded at head_dim=512 even with block_M=16").

---

## Phase 57 — Triton V4 attention kernel perf push (cr=0/4/128 FWD + BWD)

> Sourced from the user's post-plan-8 perf-push directive:
> "优化 cr=0 BWD (7.65 ms → 3 ms), cr=4 FWD (3.18 ms → 1.5 ms),
> cr=4 BWD (16.29 ms → 5 ms), cr=128 BWD (11.89 ms → 3 ms).
> 可以开启多个 subagent 并发做."

### Targets (V4-Flash widths, `B=1, H=64, Sq=4096, D=512, swa=128, sink=True, bf16`)

| Kernel family | Current (P32 RoPE-fix final) | P57 target | Speedup |
|---|---:|---:|---:|
| `cr=0` BWD (dense + SWA + sink)                 |  7.66 ms (22.11 TF) | ≤ 3.0 ms | **2.55×** |
| `cr=4` FWD (CSA: local SWA + sparse + sink)     |  3.18 ms (108.4 TF) | ≤ 1.5 ms | **2.12×** |
| `cr=4` BWD (CSA dq + dkv_local + dpool sparse)  | 16.29 ms (52.50 TF) | ≤ 5.0 ms | **3.26×** |
| `cr=128` BWD (HCA split-mask BWD)               | 11.89 ms (15.95 TF) | ≤ 3.0 ms | **3.96×** |

### Optimisation surface — per-kernel ideas

* **cr=0 BWD** (`_triton/v4_attention_bwd.py`, currently split dQ + dKV
  with `PRIMUS_V4_ATTN_BWD_USE_SPLIT=1`):
  - Cooperative groups for shared Q/K/V load between dQ + dKV passes.
  - Persistent kernel layout that fuses both passes (one HBM read of
    Q/K/V/dO instead of two).
  - Better MFMA scheduling (`k_pack=2`, swizzle, num_stages tuning).
  - Larger BLOCK_M / BLOCK_N when SMEM permits at `head_dim=512`.
  - Tail-loop pruning: SWA + causal mask let many BWD tiles short-circuit.

* **cr=4 FWD** (`_triton/v4_csa_attention_fwd.py`):
  - Joint local SWA + sparse top-K load into shared (avoid re-reading
    Q for the sparse branch).
  - Larger top-K tile (K_topk read in fewer chunks).
  - Vectorised gather for the sparse branch.
  - In-register top-K reordering to avoid uncoalesced HBM gathers.

* **cr=4 BWD** (`_triton/v4_csa_attention_bwd.py`):
  - In-register dpool partial accumulation (avoid the 4 GiB partial
    buffer; segreduce stays as a fallback).
  - Sorted-inverse-index reordering to defeat HBM scatter contention.
  - Cooperative groups for dq / dk_local / dpool sharing Q/K/V loads.
  - Single-pass dq + dk_local + dpool kernel (vs current 3-kernel
    pipeline) when SMEM permits.

* **cr=128 BWD** (HCA split-mask, shares `v4_attention_bwd.py`):
  - HCA-specific BWD split (separate kernel from dense BWD); local
    branch reuses the dense BWD kernel, pool branch is a tiny pool-
    only dKV kernel.
  - Joint local + pool dq accumulation (avoid two passes over Q).
  - Eliminate the cross-block dpool atomic_add via sorted-inverse-
    index (same pattern as cr=4 segreduce).

### Methodology — parallel best-of-N

P57 runs as a parallel best-of-N optimisation pass:

1. The parent agent spawns **N subagents in isolated git worktrees**
   (via `best-of-n-runner`), one per target × optimisation angle.
   Each subagent has:
   - Read-only access to the existing kernel + bench harness.
   - A clear target wall-clock + parity gate.
   - License to modify ONLY its target kernel file in its worktree.
2. Each subagent iterates: design → implement → microbench → parity
   check → report. Returns its best result (wall-clock, parity
   status, branch / patch) to the parent.
3. The parent picks the **fastest result that passes parity** per
   target and cherry-picks / re-applies into the main branch.
4. If a target misses, the parent spawns a follow-up round with
   refined angles.
5. Final integration: `attention_perf.md` row + `progress/p57/p57-summary.md`
   + status-pin commit.

### Tasks (parent agent)

1. **Baseline lock** — run
   `progress/p32/bench_v4_attention_ep8.py --mode dense --warmup 3
   --iters 10` (cr=0), `--mode hca` (cr=128), and
   `progress/p31/bench_csa_attention_ep8.py --warmup 3 --iters 10`
   (cr=4 FWD+BWD).  Pin baselines in `progress/p57/baseline.md`.

2. **Subagent dispatch** — launch `best-of-n-runner` subagents in
   parallel, one per optimisation angle.  Subagent prompt includes:
   - Target wall-clock + parity tolerance.
   - File scope (single kernel file in `_triton/`).
   - Bench command + parity test command.
   - Iteration budget (compile-fail / parity-fail / perf-regress
     all count; subagent must converge in N tries).

3. **Result collection** — each subagent returns a one-page
   report: wall-clock numbers, parity-test result, branch name,
   summary of what changed.

4. **Integration** — apply winning patches via `git cherry-pick` /
   `git apply --3way` into the main branch.  Resolve conflicts when
   two targets share a file (e.g. cr=0 BWD + cr=128 BWD both touch
   `_triton/v4_attention_bwd.py`).

5. **G57 — parity ratchet** — run
   `pytest -q tests/unit_tests/megatron/transformer/deepseek_v4/`
   (with `--run-slow` for release-tier) and the plan-8 P49 G49
   tests.  Pass count must not drop.

6. **G57a — EP=8 proxy smoke** — 10-iter run with the new kernels
   default-on; `lm_loss` within `5e-2` of the P48 baseline; no
   banned warnings.

7. **G57b — `attention_perf.md`** — append a P57 row with the
   landed wall-clock numbers in the R2.5 cell format.

8. **`progress/p57/p57-summary.md`** — eight-section per-phase
   summary per R2.1.  Document per-target winner + descope rationale
   for any miss.

### Coordination notes (multi-subagent safety)

* Each `best-of-n-runner` subagent uses its own git worktree under
  `.worktrees/p57-<target>-<angle>/`.  Their commits land on
  per-subagent branches (`dev/wenx/p57-<target>-<angle>`); the
  main `dev/wenx/deepseek-v4` branch is **NOT** modified by the
  subagents.
* The parent agent picks the winner and re-applies via
  `git cherry-pick` (or manual edits if the subagent's branch
  diverged too far).
* Conflict resolution: `cr=0 BWD` and `cr=128 BWD` both touch
  `_triton/v4_attention_bwd.py`.  The parent merges them serially
  (cr=0 first, then cr=128 layered on top), running parity after
  each step.

### Edge cases

* **A target misses by < 10 %.**  Ship the kernel as-is (best
  result wins); document the gap in `p57-summary.md` §"failed /
  negative probes".
* **A target misses by > 10 %.**  Re-spawn a second round of
  subagents with refined optimisation angles.  Iterate until
  the target is met OR the budget is exhausted (≤ 3 rounds).
* **A parity test fails** (numerical regression).  The
  optimisation is rejected; the kernel stays on the P32 RoPE-fix
  final implementation.  Same precedent as P38 / P39 / P50.
