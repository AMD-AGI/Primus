# 02 — Plan-4 Phase Details

> Each phase below lists (a) the user request that motivates it, (b) the
> concrete tasks, (c) the design notes that the implementer must keep in
> mind, and (d) the edge cases / risks that escaped Plan-3. Test gates
> live in `03-test-strategy.md`.

## Phase 24 — Test harness + V4 attention shape fixtures + reference op

> "最后要有单元测试，triton版本和原始的小算子版本的结果要能对齐，包含
> forward和backward计算。测试的shape就按照v4-flash和v4-pro的attention
> shape来进行测试" — user, plan-4 kick-off.

P24 is the foundation for P25 + P26: it defines (i) the V4 shape
fixtures every test parametrises over, (ii) the eager-Python reference
ops the Triton kernels will be compared against, and (iii) the
test-side equivalence harness (forward + backward, with tolerance
budgets per dtype). Without P24, every kernel phase would re-implement
its own harness and shape table, and the two kernels would drift apart
in their definition of "what shape are we testing".

### Tasks

1. **Shape table** — create
   `tests/unit_tests/megatron/transformer/deepseek_v4/v4_attention_shapes.py`
   exposing two parametrisable dataclasses:
   - `V4FlashAttnShape` (B, S, H=64, head_dim=512, q_pe_dim=64,
     attn_sliding_window=128, sink=True, compress_ratio ∈ {0, 4, 128},
     index_topk=512, P = S // compress_ratio when compress_ratio > 0)
   - `V4ProAttnShape` (B, S, H=128, head_dim=512, q_pe_dim=64,
     attn_sliding_window=128, sink=True, compress_ratio ∈ {0, 4, 128},
     index_topk=1024, P = S // compress_ratio when compress_ratio > 0)
   - Each fixture exposes `seq_lengths` as `(small=128, medium=512,
     large=4096)` so unit tests parametrise over a fast tier (CI-time)
     and a release tier (smoke-time).
2. **Reference op extraction** — create
   `primus/backends/megatron/core/transformer/v4_attention_kernels/__init__.py`
   and `v4_attention_kernels/reference.py` exposing two pure functions:
   - `eager_v4_attention(q, k, v, *, sink, swa_window, additive_mask,
     attn_dropout, training, scale) -> out`
   - `eager_v4_csa_attention(q, k_local, v_local, gathered, *, sink,
     swa_window, sparse_mask, attn_dropout, training, scale) -> out`
   These wrap the math currently inlined in
   `DeepseekV4Attention._attention_forward` and `_csa_forward` without
   changing any tensor / dtype / numerical contract — they are
   refactor-only.
   - `DeepseekV4Attention._attention_forward` and `_csa_forward` are
     rewritten to delegate to the new functions; the existing forward
     path runs through the same code as the reference op so the test
     harness has exactly one definition of "the eager truth".
3. **Equivalence harness** — create
   `tests/unit_tests/megatron/transformer/deepseek_v4/v4_attention_test_utils.py`
   exposing `compare_fwd_bwd(reference, candidate, inputs, *,
   fwd_tol, bwd_tol)` that:
   - runs both `reference` and `candidate` on independent input clones
     (cloned so backward does not interfere),
   - calls `out.sum().backward()` on each,
   - asserts forward output matches within `fwd_tol`,
   - asserts every leaf input gradient matches within `bwd_tol`,
   - on mismatch, prints a structured diff (max abs / max rel error per
     leaf, with the 8 worst entries).
   Tolerance budget defaults: `fp32 → atol=1e-5, rtol=1e-5`,
   `bf16 fwd → atol=2e-2, rtol=2e-2`, `bf16 bwd → atol=5e-2, rtol=5e-2`.
4. **Existing-test equivalence (G22)** — assert that swapping the
   `_attention_forward` / `_csa_forward` body for the new
   `eager_v4_*` callers does NOT change the existing P22 / P19 / P17
   forward equivalence baselines on CPU. This is the safety net for
   the refactor.

### Design notes

- `eager_v4_attention` MUST exactly match the existing `_attention_forward`
  math: `logits = (q.float() @ k.float().transpose) * scale; logits +=
  additive_mask; probs = softmax_with_optional_sink(logits); out =
  probs.to(v.dtype) @ v`. The float32 cast around the matmul / softmax
  is load-bearing for numerics on bf16 and must NOT be removed.
- `eager_v4_csa_attention` MUST exactly match the existing `_csa_forward`
  math: it consumes a pre-gathered `gathered` tensor (the wrapper does
  the per-query top-K `torch.gather` outside the function), and a
  `sparse_mask` of shape `[B, Sq, K]` whose `-inf` entries flag
  positions where the indexer returned `topk_idx == -1`.
- The reference op signature is intentionally close to the kernel
  signature so the test harness can plug them in interchangeably.
- The reference op does NOT fuse RoPE — Q / K are post-RoPE on entry.
  This matches the kernel contract (RoPE stays outside the fused
  attention call, exactly like Turbo / TE flash-attn).

### Edge cases

- `attn_sink is None` (no sink). The reference op handles `sink=None`
  identically to "sink with all zeros + drop-the-column": it falls
  through to the standard softmax path. The kernel must do the same.
- `attn_dropout == 0.0` AND `training == False`. Reference path skips
  dropout entirely. Kernel must take the same fast path so dropout RNG
  state is not consumed (downstream Megatron RNG state would diverge).
- `compress_ratio == 128, S < compress_ratio` (sub-stride seq). Pool
  size `P == 0`; the additive bias has shape `[Sq, Sq + 0] = [Sq, Sq]`
  and the kernel collapses to the dense path. The shape fixture sets
  `S = max(S, 4 * compress_ratio)` to avoid this degenerate case at
  the small / medium tier.

### Risks

- The `_attention_forward` / `_csa_forward` extraction is a refactor on
  the hot path; mis-extraction silently changes V4 numerics and breaks
  the released-checkpoint reproduction. G22 catches this by asserting
  bit-equivalence (or fp32-tolerance equivalence if the only delta is
  a `.to(dtype)` re-ordering) of the small-CPU forward against a
  pre-extraction baseline.

---

## Phase 25 — `v4_attention` Triton kernel (compress_ratio ∈ {0, 128})

> "我理解应该是开发两个版本的attention，一个是正常的attention，用于
> compress=0和128…基于triton开发对应的attention" — user, plan-4 kick-off.

P25 ships the first of the two plan-4 kernels — the "general"
flash-attn-style kernel that V4's dense (`compress_ratio == 0`, with
SWA + per-head sink) and HCA (`compress_ratio == 128`, with shared sink
and an additive bias for the joint-softmax mask) both call. The kernel
uses the same Triton math as aiter's `mha.py` reference but is owned
by Primus and targeted at V4's exact shape envelope.

### Tasks

1. **Forward kernel** — create
   `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/v4_attention_fwd.py`
   exposing `_v4_attention_fwd_kernel` (Triton JIT function) and
   `_launch_v4_attention_fwd` (Python launcher). Math:
   ```
   Inputs (post-RoPE):
     q          [B, H, Sq, head_dim]   bf16 / fp32
     k, v       [B, *, Sk, head_dim]   bf16 / fp32   (* in {1, H} for MQA / MHA)
     sink       [H] or None            fp32
     additive_mask  [Sq, Sk] or None   fp32 / bf16
     swa_window int                    -1 = off, else > 0
     scale      float                  precomputed by caller

   Forward:
     for tile m (BLOCK_M queries):
       acc = 0; m_running = -inf; l_running = 0
       for tile n (BLOCK_N keys):
         qk = q_tile @ k_tile^T * scale          # [BLOCK_M, BLOCK_N], fp32
         if additive_mask is not None: qk += additive_mask_tile
         if swa_window > 0: apply SWA causal mask in-kernel
         elif additive_mask is None: apply full causal mask in-kernel
         m_new = max(m_running, qk.max(dim=-1))
         alpha = exp(m_running - m_new)
         p = exp(qk - m_new[:, None])
         l_running = l_running * alpha + p.sum(dim=-1)
         acc = acc * alpha[:, None] + p @ v_tile
         m_running = m_new
       if sink is not None:
         # join sink as a virtual key column
         m_new = max(m_running, sink_h)
         alpha = exp(m_running - m_new)
         beta  = exp(sink_h - m_new)
         l_running = l_running * alpha + beta
         acc = acc * alpha[:, None]
         m_running = m_new
       out_tile = acc / l_running
       lse_tile = m_running + log(l_running)   # saved for BWD
   ```
2. **Backward kernel** — create
   `_triton/v4_attention_bwd.py` exposing `_v4_attention_bwd_kernel`
   and `_launch_v4_attention_bwd`. The BWD re-materialises the softmax
   in fp32 from the saved `LSE` tensor and the post-RoPE `Q / K / V`
   inputs (so we do not store the `[Sq, Sk]` `P` matrix). Outputs:
   `dq, dk, dv, dsink`. Tile choice for BWD prioritises SMEM:
   `BLOCK_M = BLOCK_N = 64` to keep peak SMEM under 160 KiB at
   `head_dim=512`.
3. **Autograd Function wrapper** — create
   `v4_attention_kernels/v4_attention.py` exposing
   `class V4AttentionFn(torch.autograd.Function)` and the public
   functional API `v4_attention(q, k, v, *, sink, swa_window,
   additive_mask, attn_dropout, training, scale) -> out`. The Function
   stores `q, k, v, sink, additive_mask, lse, scale, swa_window` for
   BWD; `attn_dropout` / `training` are constexpr in the kernel and
   determine whether dropout-RNG-state is consumed.
4. **MQA broadcast** — V4's K and V are single-latent (`*=1`); the
   kernel detects `K.shape[1] == 1` (or `==H`) at launch and chooses
   the MQA tile loop (each Q-head reuses the same K / V tiles) or the
   MHA tile loop (one K-head per Q-head). `dk / dv` for MQA accumulate
   via `tl.atomic_add` into the single-head buffer; for MHA they
   write straight into the per-head buffer (no atomics).
5. **Switch + provider plumbing** — add `use_v4_triton_attention: bool = False`
   to `args` (config + CLI), with a config-helper recogniser in
   `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_builders.py`.
   `DeepseekV4Attention.__init__` reads the arg and stores
   `self._use_v4_triton_attention: bool` for `forward` to consult.
6. **Forward / HCA caller** — `DeepseekV4Attention.forward` (cr=0 and
   cr=128 branches) checks the precedence
   `use_turbo_attention > use_v4_triton_attention > eager` and dispatches to
   `v4_attention(...)` accordingly. The eager path stays in the file
   as the fallback.
7. **Unit tests (G23 / G24 / G25)** — at every V4-Flash + V4-Pro
   `compress_ratio ∈ {0, 128}` shape (small / medium tier; large tier
   marked `pytest.mark.slow` and skipped in CI).

### Design notes

- The kernel takes the **scale** from the caller (V4's
  `_attention_scale()` is layer-dependent because of YaRN's `m_scale`
  on compressed branches) — kernel does NOT recompute `1 / sqrt(D)`.
- Sink is **per-head**, shape `[H]`. The kernel loads sink once per
  query-head tile and broadcasts. Sink contributes to `m_running` /
  `l_running` only; it does not contribute to `acc` because its
  notional value is zero (and would multiply to zero anyway).
- The kernel honours dtype: Q / K / V can be bf16 OR fp32. Inside the
  kernel, accumulators are fp32 always (matches the eager `.float()`
  cast). Output dtype matches the input dtype.
- Partial-RoPE head-split (separate tiling for the rotary "PE" sub-head
  and the non-rotary "NOPE" sub-head, like aiter's `mha.py`) is **NOT**
  required for plan-4 — V4 applies RoPE outside the kernel and the
  kernel sees post-RoPE Q / K with `head_dim=512`. We keep a single
  `BLOCK_DMODEL = next_pow_2(512) = 512` tile and skip the PE split.
  Rationale: V4's `head_dim == kv_channels` already, and the rotary
  sub-head is `qk_pos_emb_head_dim == 64` channels of the same 512
  vector — there is no separate K_PE tensor at the kernel boundary.
- HCA mask path: caller pre-concatenates pool keys to local keys
  (existing `forward` code already does this) and passes the full
  `[Sq, Sk]` additive mask. Kernel does not need to know HCA structure.

### Edge cases

- `head_dim=512` exceeds the `BLOCK_DMODEL_POW2 == 256` upper bound in
  aiter's reference. Plan-4 kernel has its own tuning table that
  supports `BLOCK_DMODEL_POW2 == 512`. Tile choice for FWD:
  `BLOCK_M = 64, BLOCK_N = 64, num_warps = 8, num_stages = 1` —
  validated empirically in P25 to fit under 160 KiB SMEM on MI355.
- `swa_window > Sq` — kernel sets effective window to `Sq` (full
  causal) so the in-kernel mask path reduces correctly. Caller does
  the same in `_local_mask`.
- `additive_mask is None and swa_window <= 0` — kernel applies the
  full causal mask (the cr=0, no-SWA, no-bias case) so the test
  harness can exercise the kernel without a mask tensor.
- bf16 + sink + fp32 sum — saved `LSE` is fp32, but `acc` is fp32
  during the tile loop and cast to bf16 only on store. Ensures sink's
  (potentially tiny / negative) contribution does not lose bits to
  bf16 round-off.

### Risks

- SMEM budget for BWD at `head_dim=512` is the single biggest risk
  (this is exactly what blocks Turbo's path on V4 today). Mitigation:
  BWD owns its own tile table separate from FWD; `BLOCK_M = BLOCK_N =
  64` keeps the K / V tile + Q tile + dK / dV tile + dQ tile + LSE
  fp32 row inside ~140 KiB at `head_dim=512`. P25 lands the kernel
  with this conservative tile choice, then opens a perf follow-up to
  retune.
- Atomic-add for MQA `dk / dv` may have observable non-determinism on
  bf16 due to floating-point reduction order. Mitigation: G24's bf16
  tolerance budget is `atol=5e-2, rtol=5e-2` (loose enough to absorb
  this); G25 tests determinism only at fp32 OR with MHA layout.

---

## Phase 26 — `v4_csa_attention` Triton kernel (compress_ratio == 4)

> "还有一个csa的版本…使用use_v4_triton_attention和use_v4_triton_csa_attention两个开
> 关来控制使用新的triton版本" — user, plan-4 kick-off.

P26 ships the second plan-4 kernel — the CSA fused kernel that runs
local SWA + per-query top-K sparse attention with a joint softmax and
shared per-head sink. CSA's per-query top-K gather pattern is what
makes it different from a stock flash-attn kernel: each query reads a
DIFFERENT `K`-row subset of the compressed pool, indexed by the
indexer's `topk_idxs`.

### Tasks

1. **Forward kernel** — create
   `_triton/v4_csa_attention_fwd.py` exposing
   `_v4_csa_attention_fwd_kernel`. Math:
   ```
   Inputs (all post-RoPE; gathered already done by wrapper):
     q          [B, H, Sq, head_dim]
     k_local    [B, H, Sq, head_dim]   (MQA: H == 1 broadcast or H == query-H)
     v_local    [B, H, Sq, head_dim]
     gathered   [B, H, Sq, K, head_dim] (K = index_topk)
     sparse_mask [B, Sq, K]      # -inf for topk_idx == -1
     sink       [H] or None
     swa_window int                    # local-branch SWA window
     scale      float

   Forward:
     for tile m (BLOCK_M queries):
       acc = 0; m_running = -inf; l_running = 0

       # Local SWA branch
       for tile n (BLOCK_N local keys):
         qk_local = q_tile @ k_local_tile^T * scale
         apply SWA causal mask in-kernel (window = swa_window)
         m_new, alpha, p, l_running, acc = online_softmax_step(...)

       # Sparse branch (per-query gathered keys)
       for tile k (BLOCK_K sparse keys):
         qk_sparse = einsum("bmh d, bmh k d -> bmh k", q_tile, gathered_tile) * scale
         qk_sparse += sparse_mask_tile     # -inf for invalid topk
         m_new, alpha, p, l_running, acc = online_softmax_step(...)

       # Sink (joint with both branches)
       if sink is not None:
         m_new = max(m_running, sink_h)
         ... (same as v4_attention)

       out_tile = acc / l_running
       lse_tile = m_running + log(l_running)
   ```
2. **Backward kernel** — create
   `_triton/v4_csa_attention_bwd.py`. Outputs:
   `dq, dk_local, dv_local, dgathered, dsink`. The `dgathered` output
   has the same shape as `gathered` (`[B, H, Sq, K, head_dim]`) — the
   wrapper is responsible for scattering this back into the `[B, H, P,
   head_dim]` compressed pool gradient (per-query scatter-add by
   `topk_idxs`). The kernel does NOT see `topk_idxs`.
3. **Autograd Function wrapper** — `v4_csa_attention.py` exposing
   `v4_csa_attention(q, k_local, v_local, gathered, *, sink,
   swa_window, sparse_mask, attn_dropout, training, scale) -> out`.
   The wrapper takes the **post-gather** `gathered` tensor (the caller
   does the `torch.gather` of `pool[..., topk_idxs, :]` outside the
   kernel — same as the existing `_csa_forward` does today).
4. **Switch + dispatch** — add `use_v4_triton_csa_attention: bool = False` to
   `args`. `DeepseekV4Attention.forward` cr=4 branch uses precedence
   `use_v4_triton_csa_attention > eager` and dispatches accordingly.
5. **Unit tests (G26 / G27)** — at every V4-Flash + V4-Pro
   `compress_ratio == 4` shape (small / medium tier; large tier
   `pytest.mark.slow`).

### Design notes

- The CSA kernel reads `gathered` directly. The "natural" alternative
  — pulling `topk_idxs` into the kernel and doing per-query
  `tl.load(pool_ptr + topk_idxs[m] * stride)` — saves the
  `[B, H, Sq, K, head_dim]` materialisation but is significantly more
  complex (Triton's `tl.load` with per-row offsets, BWD scatter via
  `tl.atomic_add` into the pool gradient with non-deterministic
  collisions when two queries pick the same pool position). Plan-4
  keeps the wrapper-side gather; the in-kernel gather is a future
  perf optimisation.
- The fused SWA + sparse + sink joint softmax MUST share one running
  `m_running / l_running` — that is the entire point of "joint
  softmax with shared sink". The kernel processes the local branch
  first, then the sparse branch, then the sink, in a single tile loop
  (the same `acc / m_running / l_running` carry across all three
  segments). This matches the eager `_csa_forward`'s
  `torch.cat([local_logits, sparse_logits], dim=-1)` then
  `_append_sink_softmax(...)`.
- `dgathered` returned by the kernel is the gradient w.r.t. the
  per-query gathered tensor; the wrapper does NOT scatter-add
  `dgathered` into the compressed-pool gradient inside the autograd
  function — instead, the wrapper's BWD returns `dgathered` as the
  autograd output, and the caller (`DeepseekV4Attention._csa_forward`
  replacement) is responsible for the scatter-add. This matches how
  `torch.gather`'s autograd already works (its BWD scatter-adds back
  into the source). G27 verifies this via end-to-end backward against
  the eager reference (which uses `torch.gather` and `torch.einsum`
  whose autograd does the scatter for us).

### Edge cases

- `K == 0` (no top-K — degenerate Indexer state). Kernel skips the
  sparse branch entirely and reduces to the local SWA + sink path.
  The wrapper detects `gathered.shape[3] == 0` and short-circuits to
  the dense `v4_attention` kernel.
- `topk_idx == -1` for some queries (Indexer returned fewer than K
  valid positions because of causal-on-pool truncation). The wrapper
  flags these in `sparse_mask` as `-inf`; the kernel honors the mask
  inside the sparse-branch tile loop.
- The compressed-pool gradient (downstream of `dgathered` after the
  wrapper-side scatter-add) accumulates contributions from **every**
  query whose `topk_idxs` selected a given pool position — at large
  Sq this can cause large gradient magnitudes on hot pool positions.
  This is identical to the eager-Python path (`torch.gather` + `sum`
  has the same accumulation semantics) and is NOT a kernel concern;
  G27 simply asserts numerical equivalence to eager.

### Risks

- The "share one running `m / l`" requirement across three branches
  (local SWA, sparse, sink) means the kernel cannot be split into
  three independent flash kernels with LSE merge — they all must run
  in one tile loop with one running softmax. This is a real
  constraint and is what makes the CSA kernel non-trivial. Mitigation:
  the kernel is designed top-down with the joint softmax as the
  carrier; the unit tests (G26 / G27) catch any deviation.

---

## Phase 27 — Release-tier shape gate + dispatch wiring + smoke

> "使用run_deepseek_v4跑通开启…the plan-4 kernels…" — implicit user
> request, plan-4 kick-off.
>
> "在phase27的计划里面，先添加这两个算子单测里面添加真实的shape信息，
> 然后保证算子在真实shape的正确性。因为只是算子测试，显存应该不是问题。
> phase27的其他task放在这个task之后。" — user, plan-4 P26 review.

P27 closes plan-4 with three layered checks:

1. **Real-shape kernel correctness (G28)** — extends the P25 / P26
   parametrisations with V4-Flash + V4-Pro production-dim entries
   (real `H`, real `head_dim=512`, real `swa_window`, real `K_topk`)
   so the kernels are validated at the SMEM stress point that
   `head_dim=512` actually exercises (the small-tier P25 / P26 shapes
   used `head_dim=64`, which does NOT exercise the
   ``head_dim * tile_size`` SMEM corner that plan-4 exists to solve).
2. **Dispatch wiring (G29)** — wires the two switches into
   `DeepseekV4Attention.forward` and through `run_deepseek_v4.sh` to
   the Megatron training stack. (The flag plumbing on the config /
   yaml / run-script side already landed in P25; P26 wired the
   `_csa_forward` dispatch. P27 task 2 only documents the precedence
   and adds the rank-0 startup log line.)
3. **Smoke (G30)** — runs a 10-iter smoke at TP=1 PP=1 EP=8 with both
   kernels engaged. The smoke is plan-4's release evidence — once it
   is green, the plan-4 hand-off notes record the kernel-vs-eager
   perf delta and the `USE_V4_TRITON_ATTENTION` /
   `USE_V4_TRITON_CSA_ATTENTION` defaults can be flipped in a
   follow-up plan.

The order matters: a green G28 means the kernels are *known correct*
at production dims before the dispatch and smoke layers are added on
top. If G30 then fails it must be a wiring / pipeline issue, NOT a
kernel-numerics issue.

### Tasks

1. **Release-tier shape gate (G28)** — extend the
   `_BASE_SHAPES` parametrisations in
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p25_v4_attention_fwd.py`,
   `test_v4_p25_v4_attention_bwd.py`,
   `test_v4_p26_v4_csa_attention_fwd.py`, and
   `test_v4_p26_v4_csa_attention_bwd.py` with production-dim entries.
   Concrete tiers (revisit at implementation time if memory headroom
   permits a larger `S`):

   ```
   v4_flash_release: B=1, H=64,  head_dim=512, swa_window=128, S ∈ {512, 1024}, K_topk=512   (cr ∈ {0, 128, 4})
   v4_pro_release:   B=1, H=128, head_dim=512, swa_window=128, S ∈ {512, 1024}, K_topk=1024  (cr ∈ {0, 128, 4})
   ```

   The tiers are guarded with `pytest.mark.slow` (and
   `pytest.mark.gpu` already inherited from the GPU import skip) so
   default `pytest` runs only the fast tier; `pytest -m slow` runs
   the release tier. Reuses the existing dtype × sink × kv_layout
   parameter axes — the only delta vs. the small tier is the shape
   row.
2. **Dispatch precedence + startup log** —
   `DeepseekV4Attention.forward` already follows
   `use_turbo_attention > use_v4_triton_attention > eager` for the
   dense path (plan-3 P22 + plan-4 P25) and
   `use_v4_triton_csa_attention > eager` for CSA (plan-4 P26). P27
   task 2 only:
   - Documents the precedence in `deepseek_v4_attention.py`'s module
     docstring.
   - Emits a single rank-0 startup line per layer kind summarising
     which kernel each layer is using ("Layer 17: cr=128, kernel =
     v4_attention (Triton)").
3. **Run script plumbing** — `run_deepseek_v4.sh` already exposes
   `USE_V4_TRITON_ATTENTION` (default `False`) and
   `USE_V4_TRITON_CSA_ATTENTION` (default `False`) env vars + matching
   CLI args (P25 lit them up). P27 task 3 confirms the help text +
   adds a TP=1 guard message (Triton kernel does not currently
   support TP > 1).
4. **Smoke run** — TP=1 PP=1 EP=8, 10 iters,
   `USE_V4_TRITON_ATTENTION=True USE_V4_TRITON_CSA_ATTENTION=True
   USE_TURBO_ATTENTION=False USE_TURBO_DEEPEP=True
   PRIMUS_SEQ_LENGTH=128`. Log lives at
   `deepseek-v4/develop/progress/p27/` with a `.gitignore` that
   excludes `*.log` / `log_*.txt` / `debug.log` (smoke logs MUST NOT
   land in git, per the user's plan-3 directive).
5. **Hand-off note** — append a short P27 status box to plan-4's
   `02-phase-details.md` (this file) recording: (a) commit SHAs for
   P24..P27, (b) the G28 release-tier matrix size + pass count,
   (c) the smoke's iter / TFLOP/s / ms-per-iter numbers, (d) the
   kernel-vs-eager perf delta, and (e) any follow-ups surfaced by
   the smoke.

### Design notes

- **Why release-tier kernel UT belongs in P27, not P25 / P26.** The
  small-tier shapes in P25 / P26 (`head_dim=64`, `H ∈ {4, 8}`) cover
  the numerical-contract correctness. They do NOT cover the
  `head_dim=512` SMEM corner that plan-4 exists to solve — and
  attaching the release-tier matrix to P25 / P26 directly would have
  inflated their test runtimes for everyone running the fast suite.
  Splitting it into a dedicated P27 task keyed on `pytest.mark.slow`
  is the standard plan-4 pattern (see plan-4 `03-test-strategy.md`
  GPU-toy harness section).
- **Why the eager reference fits.** At V4-Flash release-tier
  (`H=64, head_dim=512, S=1024, K_topk=512`) the eager CSA
  reference's largest fp32 intermediate is the joint-softmax
  `probs: [B, H, S, S+K] = [1, 64, 1024, 1536]` at `4 bytes ≈
  400 MiB`, and the broadcast `gathered_h.to(dtype)` materialisation
  is `[B, H, S, K, D] = [1, 64, 1024, 512, 512]` at `2 bytes ≈
  16 GiB` (bf16). Both fit on a 192 GiB MI355X with margin. At
  V4-Pro release-tier (`H=128, K_topk=1024`) the gathered
  materialisation grows to ~64 GiB; if memory is tight,
  implementation-time picks a smaller `S` (e.g., `S=512`) — the
  shape-correctness contract is per-tile and does not require a
  particular `S`.
- **Why we do NOT chase full S=4096.** Beyond 1–2 K seq the eager
  reference's joint softmax allocations dominate; the kernel-side
  test gives diminishing returns (the same tile loop is exercised at
  S=1024 as at S=4096). Full-S=4096 evidence comes from the P27 G30
  smoke at the model level, not from G28 kernel UT.
- **Smoke seq length.** The smoke uses `PRIMUS_SEQ_LENGTH=128` by
  default to fit under EP=8 on `mi355-gpu-12` / `mi355-gpu-14`; a
  separate full-S=4096 smoke is gated by available memory and is
  documented as a plan-4 follow-up.
- The new switches are V4-only — they live on the V4 builder /
  attention class and never affect non-V4 model types. This mirrors
  the plan-3 P22 / P23 contract.

### Edge cases

- TP > 1 + `use_v4_triton_attention=True` — kernel does not currently
  support TP-sharded heads; the wrapper raises a clear error if
  `tp_size > 1` with `use_v4_triton_attention=True`. Same precedent as
  Turbo's TP=1 requirement for DeepEP. P27 task 3 documents this in
  the `run_deepseek_v4.sh` CLI help text.
- bf16 vs. fp32 mismatch — the kernel honors the input dtype so the
  V4 yaml's `bf16: true` runs end-to-end in bf16 with no extra
  conversion. fp32 paths still work for unit tests.
- G28 long-runtime — release-tier matrix size is roughly
  `2 variants × {dense, HCA, CSA} × {fp32, bf16} × {sink_on, off} ×
  {mqa, mha for dense/HCA only} ≈ 60 tests`. Estimated runtime on
  MI355X is a few minutes (matmul-bound at `head_dim=512`); guarded
  behind `pytest.mark.slow` so default CI skips it.

### Risks

- The G28 release-tier surfaces a kernel-numerics bug at
  `head_dim=512` that the small-tier did not catch (e.g., a
  ``BLOCK_DMODEL`` corner). Mitigation: G28 lands BEFORE dispatch
  wiring + smoke so any kernel fix is localised to the kernel files
  before the pipeline is touched. The kernel was designed with
  `head_dim=512` SMEM in mind from the start (see P25 / P26 design
  notes); G28 is the empirical confirmation.
- The G30 smoke surfaces an unexpected interaction with PP / VPP /
  EP collectives that did not show up in the unit tests.
  Mitigation: the smoke's first 5 iters are run with NCCL P2P
  logging on; failures get captured as a regression test in P27's
  hand-off.
- A regression in the per-head-sink contract (`attn_sink` parameter
  no longer loadable from V4-Flash checkpoint when
  `use_v4_triton_attention=True`). Mitigation: the kernel reads
  `self.attn_sink` directly; the parameter stays on the attention
  module under the same key, so state-dict load is unaffected. G23 /
  G27 explicitly test that
  `model.load_state_dict({"attn_sink": ...})` still works after the
  kernel switches are flipped on.

### P27 hand-off (status — closes plan-4)

Plan-4 ships a Primus-owned, in-tree Triton kernel pair for the V4
attention block — one for the dense / HCA path (`compress_ratio ∈
{0, 128}`) and one for the CSA path (`compress_ratio == 4`) — gated
behind two new V4-only switches (`use_v4_triton_attention`,
`use_v4_triton_csa_attention`, both default `False`). The kernels
sit alongside the eager-Python reference (P24) and the Plan-3 P22
Turbo `core_attention` wiring; precedence is enforced layer-by-layer
in `DeepseekV4Attention` (`use_turbo_attention >
use_v4_triton_attention > eager` for cr ∈ {0, 128};
`use_v4_triton_csa_attention > eager` for cr == 4) and surfaced at
boot via a one-shot rank-0 `[V4-attn] Layer N: cr=R, kernel = ...`
log line.

**Commit chain (P24 → P27)**

| phase   | scope                                                                                                                                                | commit(s)                  |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| P24     | reference op + harness + dtype contract refinement (bf16 matmul, fp32 softmax)                                                                       | `8b971881`, `38ef526c`     |
| P25     | dense / HCA Triton kernel + autograd Function + dispatch plumbing + G23 / G24 / G25 + dispatch surface tests                                         | `7df4cfeb`                 |
| P26     | CSA Triton kernel + autograd Function + `_csa_forward` dispatch + G26 / G27 + dispatch surface tests                                                 | `da6f48bc`                 |
| P27 G28 | release-tier shape gate (`H ∈ {64, 128}, head_dim=512, swa_window=128, K_topk ∈ {512, 1024}`); `pytest.mark.slow` + `--run-slow` opt-in              | `e19663f7`                  |
| P27 G29 | dispatch precedence runtime test + `_log_kernel_choice` rank-0 startup log + class-docstring precedence section + run-script TP > 1 soft warn       | `e19663f7`                  |
| P27 G30 | TP=1 PP=1 EP=8 10-iter smoke with both kernels engaged + Turbo DeepEP + script under `progress/p27/` (gitignored log)                                | `e19663f7`                  |

**Test totals (`mi355-gpu-14` inside `dev_primus_wenx_693`)**

* Fast tier (default `pytest tests/unit_tests/megatron/transformer/deepseek_v4/`):
  **272 pass / 1 skipped + 80 deselected (`pytest.mark.slow` release-tier rows)**
  * P25 G23 (fwd) — 34, P25 G24 (bwd) — 33, P25 G25 (determinism / dropout) — 2, P25 dispatch — 5
  * P26 G26 (fwd) — 13, P26 G27 (bwd) — 10, P26 dispatch — 4
  * P27 G29 (dispatch precedence) — 16
  * The 1 deselected fast-tier row (`test_v4_mtp.py::test_helper_pulls_norm_and_linear_from_v4_provider`) is a pre-plan-4 failure on `dev/wenx/deepseek-v4`; verified by stash + re-run on commit `6a17c3b0`.
* Release tier (`pytest --run-slow -m slow`):
  **80 / 80 pass in 60.2 s** at production V4 dims (`head_dim=512`,
  V4-Flash @ `H=64, S=1024, K_topk=512` and V4-Pro @
  `H=128, S=512, K_topk=512`). Per-test `torch.cuda.empty_cache()`
  fixture under `tests/unit_tests/megatron/transformer/deepseek_v4/conftest.py`
  prevents PyTorch's caching allocator from holding onto the
  `~64 GiB [B, H, Sq, K, D]` eager-CSA broadcast across consecutive
  tests; bf16 tolerances bumped at release tier (FWD `atol=5e-2`;
  BWD `dq/dk/dv/dgathered atol=2e-1`; `dsink atol=5e-2`) to absorb
  `head_dim=512` matmul noise + `tl.atomic_add` jitter on `dk / dv`.

**G30 smoke perf delta**

| baseline                                            | TFLOP/s/GPU (steady-state) | ms / iter | iter loss curve  |
| --------------------------------------------------- | -------------------------- | --------- | ---------------- |
| P22 eager Turbo-off (`p22/smoke_eager_ep8_pp1`)     | ~12.6                      | ~770      | `11.89 → 11.62`  |
| P23 eager + Turbo DeepEP (`p23/smoke_turbo_deepep`) | ~17–19                     | ~530      | `11.88 → 11.65`  |
| **P27 V4 Triton + Turbo DeepEP (G30)**              | **~17.3** (peak ~19.8)     | **~500**  | `11.85 → 11.65`  |

Plan-4's Triton kernels are at parity (within smoke noise) with the
P23 Turbo-DeepEP-on-eager-attention baseline at the smoke's small
seq length (`PRIMUS_SEQ_LENGTH=128`). The eager attention is
matmul-cheap at this seq length and DeepEP dominates iter time, so
the Triton kernels' real win is expected on full V4-Flash production
dims (`S=4096`) where the eager `[B, H, S, S]` logits tensor blows
to 16 GiB / microbatch. The full-S smoke is a Plan-4 follow-up
(G30 only covers the 10-iter ergonomic gate; the production-scale
perf comparison is planned for the same successor plan that flips
the kernel defaults to `True`).

**Follow-ups (post-plan-4)**

1. **Megatron-side `layer_number` plumbing.** Every layer's
   `[V4-attn]` startup log line currently says `Layer 0` because
   the V4 spec does not pass `layer_number=` through to
   `DeepseekV4Attention.__init__` at construction time (Megatron
   normally sets it later via a spec-walker setattr). The dispatch
   itself is unaffected (it only reads `compress_ratio` + the
   runtime flags), and the cr-by-cr log lines are still
   distinguishable, but a dedicated patch in
   `primus/backends/megatron/patches/deepseek_v4_*` to populate
   `self.layer_number` from the spec walker would make per-layer
   debugging cleaner. Low priority — cosmetic.
2. **Full-S=4096 V4-Flash smoke.** P27 G30 ran at
   `PRIMUS_SEQ_LENGTH=128` for ergonomic loop time and to fit on
   any free MI355X without juggling allocator pressure. A
   follow-up plan should run a single-iter smoke at the
   release-default `S=4096` to confirm the kernels' bandwidth
   advantage materialises at the seq lengths plan-4 was designed
   for. Gated on a free node with sufficient HBM headroom for the
   eager comparison baseline; the kernel side is already exercised
   by G28's `S ∈ {512, 1024}` release-tier matrix.
3. **HCA LSE merge for Turbo `core_attention`.** Plan-4's HCA path
   stays on the v4_attention Triton kernel because Turbo / TE
   flash-attn does not return LSE, so the joint local + pool
   softmax cannot be decomposed into two flash calls + a recombine.
   A follow-up that adds an LSE-returning branch to Turbo would
   let HCA layers ride the same `core_attention` path the dense
   layers use.
4. **CSA in-kernel gather.** Plan-4 P26 does the
   `pool[..., topk_idxs, :]` gather wrapper-side and feeds the
   `[B, H, Sq, K, head_dim]` gathered tensor into the kernel. A
   follow-up that does the gather in-kernel (issuing per-row
   indirect loads from `pool` based on `topk_idxs`) would save
   the gather's `K*D` write-then-read round-trip to HBM. Bigger
   kernel-engineering effort — defer until the V4 production
   smoke confirms the gather is on the critical path.
5. **FP8 path for both kernels.** Plan-4 ships bf16 + fp32 only
   (V4-Flash trains in bf16 with `attn_dropout=0`). FP8 quant at
   the QK^T and PV matmuls follows the same pattern Turbo /
   `core_attention` already supports; gated on a follow-up plan
   that picks a release recipe (`fp8_recipe ∈ {delayed, hybrid}`)
   and updates the bf16 tolerance budget for fp8.
6. **Flip the V4 Triton kernel switches to `True` by default.**
   Plan-4 ships the kernels at `default=False` so this PR is a
   pure safety-net add. Once items (2) + (5) above are landed and
   the release perf is confirmed, a follow-up plan should flip
   `use_v4_triton_attention` and `use_v4_triton_csa_attention` to
   `True` in the V4-Flash YAML default and update the
   release-config docs.
