# 02 — Plan-6 Phase Details

> Each phase below lists (a) the user request that motivates it, (b)
> the concrete tasks, (c) the design notes that the implementer must
> keep in mind, and (d) the edge cases / risks. Test gates live in
> `03-test-strategy.md`. Every plan-6 phase task list is **seeded**
> from the user's optimisation hints + the plan-5 P32 final trace
> (`output/amd/tas-mi355x-20260514/p32_final_profile_ropefix_split_segreduce_pp1_ep8_seq4096/.../*.pt.trace.json.tgz`),
> and each phase opens with a "task list refinement" pass that
> revises the breakdown against the latest trace / bench data before
> kernel work starts.

## Phase 33 — V4 TFLOP/s closed-form correction (SWA visible-pair + HC fn matmul)

> "重新review一下，当前deepseek v4的tflops计算逻辑是否有问题。如果
> 有问题修复一下。" — user, plan-6 kick-off (correction hint #1).

P33 closes two known gaps in plan-3 P20's `compute_v4_flops` closed
form. The patch is correct on its original mandate (V4-aware Q LoRA +
single-latent KV + grouped low-rank O + Compressor / Indexer + hash
router + MTP + LM head) but two terms became material after plan-5
landed: SWA visible-pair pruning makes the local-attention FMAC count
over-count by 5–8× on dense and HCA layers (the closed form uses
`S_eff^2 / 2` for the local pair count, but the actual visible-pair
count at `swa=128, S_eff=4096` is `516,160` — about `15.5×` smaller
than `4096^2 / 2 = 8,388,608`), and the HyperConnection `fn.weight`
matmul was never counted.

### Tasks

0. **Task list refinement** — re-read
   `develop/perf/attention_perf.md` "Test Shape And Counting" block
   for the canonical `visible_pairs` table, and re-confirm by reading
   `_v4_p25_v4_attention_fwd.py` / `_v4_p26_v4_csa_attention_fwd.py`
   how the per-layer SWA mask interacts with the `S_eff = S * hc_mult`
   sequence axis. Pin the SWA visible-pair formulas as a function of
   `(swa_window, compress_ratio, index_topk, S_eff)`.
1. **SWA visible-pair helper** — in
   `primus/backends/megatron/patches/deepseek_v4_flops_patches.py`,
   add `_visible_pairs(*, swa_window, compress_ratio, index_topk,
   seq_len_eff) -> int` that returns:
   - dense `cr == 0`: SWA-pruned local pair count
     `swa_window * seq_len_eff - swa_window * (swa_window - 1) // 2`
     (the triangular causal correction).
   - HCA `cr == 128`: SWA local pair count (above) + pool-visible
     pair count `pool * seq_len_eff` where `pool = seq_len_eff //
     compress_ratio`.
   - CSA `cr == 4`: SWA local pair count (above) + sparse top-K
     pair count `min(index_topk, pool) * seq_len_eff`.
   The helper has a docstring that cites `develop/perf/attention_perf.md`
   for the formula.
2. **`_attn_scores_fmac_per_layer` rewrite** — replace the
   `S_eff^2 / 2` local term with `2 * num_heads * head_dim *
   _visible_pairs(...)` (the `2 *` is QK + PV combined; matches
   Megatron's `n * d * S_eff^2` convention reduced to visible pairs).
   HCA / CSA sparse terms already use the visible-pair convention;
   leave them but update the docstring + the inline comment that
   says "Megatron's `/2 causal` convention" to call out that local
   is now SWA-visible.
3. **`_hc_matmul_fmac_per_layer` (new function)** — for each
   decoder layer, count the `HyperMixer.fn.weight` matmul:
   `(K*D) -> (2+K)*K` per token per layer, times two mixers per
   layer (pre-attn + pre-FFN). For `HyperHead` at the trunk end
   and at each MTP depth: `(K*D) -> K` per token. Total per global
   batch: `batch * seq_len * (2 * num_layers * K * D * (2+K) * K +
   (1 + mtp_num_layers) * K * D * K)`.
4. **`_V4FlopsBreakdown` gains `hc: int`** — added after `logits`
   so the existing dataclass field order stays stable; `total_fmac`
   updated; `_log_breakdown` prints the new term as the last row
   (preserves existing log-line grep for `attn_qkv_o`, `attn_scores`,
   ..., `logits`).
5. **`compute_v4_flops` wires the new term** — calls
   `_hc_matmul_fmac_per_layer` once for the trunk (with
   `mtp_num_layers` argument) and adds the result into the new
   `hc` field.
6. **Unit tests (G36 + G36a)** —
   `tests/unit_tests/backends/megatron/test_deepseek_v4_flops_patches.py`
   gains:
   - **G36 (SWA visible-pair):** a parametrised case for
     `(swa=128, S=4096, cr ∈ {0, 4, 128}, hc_mult ∈ {1, 4})`. The
     expected value is computed by a small reference Python loop
     over `(t, s)` pairs in the test (independent path), not by
     calling the patch internals, so the test cannot trivially
     agree with itself.
   - **G36a (HC matmul):** asserts the new `hc` field equals a
     hand-computed reference for the V4-Flash 8-layer proxy
     (`B=8, S=4096, L=8, K=4, D=4096, mtp=0`), within int64 byte-
     equal (no float rounding in the closed form).
   - Existing G16 / G17 byte-for-byte tests stay green (no change
     to non-V4 dispatch fall-through; no change to fields other
     than `attn_scores` and `hc`).
7. **`proxy_ep8.md` `P33 corrected TFLOP/s` row** — re-compute the
   P32 final TFLOP/s using the new denominator and pin it as a new
   row (do **not** back-fill historic rows; rules R2.5 freeze them).
   Include a footnote that explains the denominator change.

### Design notes

- **Why not also count Sinkhorn / score_fn elemwise?** They are
  pointwise non-matmul; Megatron's TFLOP/s convention only counts
  matmul FMAC. The user explicitly clarified this in the plan-6
  kick-off ("c里面如果没有matmul，就忽略").
- **Why HC matmul matters numerically.** At V4-Flash widths
  `(K=4, D=4096, L=8, S=4096)` the HC trunk matmul contributes
  about `1.3 TFLOP / global batch` — small but not negligible
  vs the `~9.7 TFLOP / global batch` `attn_qkv_o + attn_scores`
  total at the 8-layer proxy. Skipping it would systematically
  under-report mFP4 / mFP6 future runs.
- **Why SWA pruning matters numerically.** At `swa=128, S_eff=4096`
  the local visible-pair count is `516,160` vs `S_eff^2 / 2 =
  8,388,608` — a `16.25×` over-count. Across 8 layers × 8
  micro-batches × `n * d * 2` this is a `~7 TFLOP / global batch`
  over-count, more than the entire HC term.
- **No runtime change.** P33 only changes the reported FLOPs
  denominator; the actual training loop is byte-identical. The
  P32 final iter time of `603 ms` stays put; only the TFLOP/s
  number moves.

### Edge cases

- **`swa_window == 0`** (turned off): `_visible_pairs` falls back
  to `S_eff^2 / 2` (the pre-P33 behavior). Test parametrises this
  case to lock the contract.
- **`compress_ratios` shorter than `num_layers`**: the existing
  `_normalize_layer_ratios` helper handles this; P33 does not
  touch it.
- **MTP depth > 0**: `mtp_num_layers` is plumbed into both the
  existing MTP closed form AND the new HC matmul (MTP layers run
  a full V4 hybrid layer inside, so each MTP depth pays one
  pre-attn + one pre-FFN mixer matmul).

---

## Phase 34 — `_stack_grouped_linear_weight` Triton FWD/BWD fusion

> "PrimusTurboGroupedMLP里面两个_stack_grouped_linear_weight开销
> 比较大，也是elemwise" — user, plan-6 kick-off (fusion hint #6).

P34 is the single biggest expected wall-clock win in plan-6. The
P32 final trace shows `hipMemcpyWithStream` at **289.6 ms / 32
calls** — about half the iter wall time. Of those 32 calls, 16 are
FWD (`linear_fc1.stack + linear_fc2.stack` per layer × 8 layers)
and 16 are BWD (the autograd path of the same stack op runs once
per layer in the backward pass via `torch.stack` + `transpose +
contiguous`'s VJP). Each call writes the full grouped weight
tensor (`[E, K, N]` at V4-Flash EP=8: `E=32, K=4096, N=2048`,
bf16, ~512 MiB per call). Plan-6 P34 collapses each call into a
single Triton kernel that does a fused **memcpy-with-permute**
from `E` per-expert `[K, N]` weight tensors to one `[E, N, K]`
contiguous buffer (the `transpose + contiguous` of the eager path
becomes implicit in the kernel's write address calculation), and
the BWD does the reverse permute-and-scatter back to each
`weight{i}.grad`.

### Tasks

0. **Task list refinement** — confirm against the latest EP8 proxy
   trace that `hipMemcpyWithStream` is still the top GPU-time line
   item after plan-5 P32 final. Re-measure call count + total time;
   if the number has shifted materially, re-pin the P34 success
   metric to the new baseline.
1. **`primus/backends/megatron/core/extensions/_triton/stack_grouped_weight.py`** —
   new file with two Triton kernels and a `torch.autograd.Function`:
   - `_stack_grouped_weight_fwd_kernel(weight_ptrs, out_ptr, E, K,
     N, ...)`: each program processes a `[BLOCK_K, BLOCK_N]` tile
     of one expert; reads from `weight_ptrs[expert_idx] + k * N + n`
     and writes to `out_ptr + expert_idx * (N * K) + n * K + k`
     (the implicit transpose). `weight_ptrs` is a `[E]` int64
     pointer tensor staged from the per-expert `nn.Parameter`
     `.data_ptr()` calls.
   - `_stack_grouped_weight_bwd_kernel(dW_ptr, dweight_ptrs, E, K,
     N, ...)`: symmetric — each program reads a `[BLOCK_K, BLOCK_N]`
     tile of `dW [E, N, K]` and writes to `dweight_ptrs[expert_idx]
     + k * N + n`. No atomic-add needed — the layout is a bijection
     so each `(e, k, n)` triple is touched by exactly one program.
   - `StackGroupedWeightFn(torch.autograd.Function)` wraps the
     two kernels and is the API entry point.
2. **`PrimusTurboGroupedMLP._stack_grouped_linear_weight` wiring** —
   gate on `os.environ.get("PRIMUS_STACK_GROUPED_WEIGHT_TRITON",
   "1") == "1"`; when on, call `StackGroupedWeightFn.apply(*[getattr(
   module, f"weight{i}") for i in range(self.num_local_experts)])`;
   when off, fall back to the eager `torch.stack + transpose +
   contiguous`. The Triton path returns the same `[E, N, K]`
   contiguous tensor (note the `transpose(1, 2)` in the eager
   path's final result).
3. **Microbench `progress/p34/bench_stack_grouped_weight.py`** —
   mirrors the plan-5 P31 / P32 bench conventions. Default
   shape: `E=32, K=4096, N=2048, bf16` (V4-Flash EP=8 proxy widths).
   `--mode {fc1, fc2}` swaps `(K, N)`. Reports `<ms> ms |
   <GB/s effective bandwidth>` per call.
4. **EP8 proxy A/B trace** —
   `progress/p34/run_baseline_trace_ep8_p34.sh` (mirrors plan-5
   P32 final) with `PRIMUS_STACK_GROUPED_WEIGHT_TRITON=1`;
   render `develop/profile/profile-after-p34-ep8-<YYYYMMDD>.{md,html}`.
5. **G37 — unit tests** —
   `tests/unit_tests/megatron/extensions/test_stack_grouped_weight_triton.py`:
   - FWD bit-equal vs `torch.stack(weights).transpose(1, 2).contiguous()`
     at fast (`E=4, K=8, N=8, fp32`) and release (`E=32, K=4096,
     N=2048, bf16`) tiers.
   - BWD `torch.autograd.gradcheck` at fast tier (fp32 small shape).
   - BWD bit-equal vs eager at release tier (bf16, `gradcheck`
     not supported at bf16 — use a hand-derived reference).
6. **Default flip note** — `PRIMUS_STACK_GROUPED_WEIGHT_TRITON`
   default is `"1"` at landing (the EP8 proxy A/B determines this).
   If the proxy A/B regresses, the default flips to `"0"` and the
   reason goes in the phase-summary "failed / negative probes"
   section.

### Design notes

- **Why the bijection matters.** The kernel writes each output
  element exactly once and reads each input element exactly once;
  no atomics needed. The BWD also has the bijection (inverse
  permute is also a bijection), so no atomics in BWD either.
- **Why not refactor to a single contiguous `[E, K, N]`
  `nn.Parameter`?** That would eliminate the stack entirely but
  requires a state-dict adapter (HF checkpoint stores per-expert
  weights as `weight{i}` keys) and changes autograd surface
  (`grad` accumulators move). User selected the conservative path
  ("triton_kernel") in plan-6 plan-mode; the structural refactor
  stays as a plan-6 follow-up if this Triton fusion does not
  fully close the `hipMemcpyWithStream` gap.
- **Kernel layout choice.** The Triton kernel does `[E, K, N] ->
  [E, N, K]` via per-program tile-level transpose; `BLOCK_K = 64,
  BLOCK_N = 64` keeps the LDS footprint small (8 KiB / program in
  bf16) and lets each program write a contiguous strip of the
  output. The autotune grid only needs to scan `(BLOCK_K,
  BLOCK_N) ∈ {(64, 64), (128, 64), (64, 128), (128, 128)}` since
  the kernel is bandwidth-bound.

### Edge cases

- **`num_local_experts` not divisible by `BLOCK_E`** — the kernel's
  outer-most program axis is `expert_idx`; no tile dimension on
  `E`, so any `num_local_experts >= 1` works.
- **`weight{i}.data_ptr()` aliasing** — Python-side asserts that
  all `E` pointers are distinct (a debug-only check; release builds
  skip it).
- **`weight{i}` not contiguous** — Megatron's parameter allocator
  always returns contiguous; the FWD kernel asserts
  `weight{i}.is_contiguous()` defensively.

---

## Phase 35 — `apply_interleaved_partial_rope` Triton FWD/BWD fusion

> "rope 需要fusion" — user, plan-6 kick-off (fusion hint #1).

P35 fuses the 9-op chain
(`slice → reshape → reshape → multiply ×4 → stack → reshape → cat`)
in `apply_interleaved_partial_rope` into a single Triton kernel.
The P32 final trace attributes `10.0 ms / 24 calls` of
`CatArrayBatchedCopy_contig` plus a non-trivial share of
`elementwise_kernel_manual_unroll<128, 8>` (61.4 ms / 693 calls)
to this function — at 16 invocations per iter (q + k per dual-RoPE
call × 8 layers) the per-call cost is ~3-5 ms.

### Tasks

0. **Task list refinement** — re-confirm the per-call cost on the
   current trace; if the actual call count or per-call cost has
   shifted post-P34 (P34 changes the GPU-time mix), re-pin the
   P35 success metric.
1. **`primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/rope_interleaved_partial.py`** —
   new file with `_apply_rope_fwd_kernel`,
   `_apply_rope_bwd_kernel`, and `RoPEInterleavedPartialFn`:
   - FWD: reads `x [..., H, head_dim]` (any leading shape; flattened
     to `[N, H, head_dim]` for the kernel), `cos / sin
     [..., 1, rd/2]` broadcast against the leading shape; writes
     `out [..., H, head_dim]` in-place over a pre-allocated buffer.
     The leading `nope = head_dim - rotary_dim` channels are copied
     through; the trailing `rotary_dim` channels are rotated.
     Pair access: `even = x[..., 2k]`, `odd = x[..., 2k+1]`,
     `out[..., 2k] = even * cos - odd * sin`,
     `out[..., 2k+1] = even * sin + odd * cos`.
   - BWD: takes `dout [..., H, head_dim]`, returns
     `dx [..., H, head_dim]`. Cos / sin treated as constants (they
     are buffers, not Parameters). The BWD math is the transpose
     of the FWD rotation matrix
     (`cos, sin / -sin, cos -> cos, -sin / sin, cos`).
2. **`dual_rope.py::apply_interleaved_partial_rope` routing** —
   gate on `os.environ.get("PRIMUS_ROPE_TRITON", "1") == "1"`;
   when on, call `RoPEInterleavedPartialFn.apply(x, cos, sin,
   rotary_dim)`; when off, fall back to the existing eager body.
   The eager body stays in tree as the fallback (and as the
   reference for the unit test).
3. **Microbench `progress/p35/bench_rope_triton.py`** — covers
   Q (`[B=1, S=4096, H=64, head_dim=512, rd=64]`) and K
   (`[B=1, S=4096, H=1, head_dim=64, rd=64]`) shapes; reports
   `<ms> ms | <GB/s effective bandwidth>` per call.
4. **EP8 proxy A/B trace** —
   `progress/p35/run_baseline_trace_ep8_p35.sh` with
   `PRIMUS_ROPE_TRITON=1`; render
   `develop/profile/profile-after-p35-ep8-<YYYYMMDD>.{md,html}`.
   Expect `CatArrayBatchedCopy_contig` to drop to ≈ 0.
5. **G38 — unit tests** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_p35_rope_triton.py`:
   - FWD parity vs eager `apply_interleaved_partial_rope` within
     `atol=1e-3 rtol=1e-3` for bf16; `atol=1e-6 rtol=1e-6` for fp32.
   - BWD `gradcheck` at fast tier (fp32 small shape).
   - Release-tier shape (`B=1, S=4096, H=64, head_dim=512, rd=64`)
     marked `pytest.mark.slow`, bf16 FWD + BWD parity within plan-5
     P32 ratchet tolerance.
   - Parametrised over `rd ∈ {0, 16, 64}` (rd=0 is the no-op
     branch — the kernel must early-return; rd=16 covers a
     non-V4 case for general robustness).
6. **Default flip note** — `PRIMUS_ROPE_TRITON` default is `"1"`
   at landing. The plan-5 P32 RoPE bf16 cast fix is the source of
   truth for cos/sin dtype handling; the Triton kernel inherits
   that contract (cos/sin cast to `x.dtype` inside the kernel via
   a `tl.cast` if needed).

### Design notes

- **Why `torch.autograd.Function` and not `triton.heuristics`-only.**
  The kernel needs to participate in autograd; `Function` is the
  canonical wrapper. The FWD saves `cos / sin / rotary_dim` for
  BWD (not `x` — BWD does not need the input).
- **Layout.** The kernel's program grid is `(N // BLOCK_N, H //
  BLOCK_H)` where `BLOCK_N = 16, BLOCK_H = 8` keeps the LDS
  footprint under 8 KiB per program at `head_dim=512`. Each
  program iterates over `head_dim` in `BLOCK_D = 64` chunks.
  Cos/sin are broadcast against the leading axis (loaded once
  per program).
- **The `nope` prefix copy.** The leading `head_dim - rotary_dim`
  channels are copied verbatim; the kernel reads them as a
  vectorised `tl.load` and writes them as a `tl.store` — no
  multiplication. This avoids a separate `torch.cat` op in the
  caller. For V4-Flash (`head_dim=512, rotary_dim=64`) the nope
  copy is `448 / 512 = 87.5 %` of the data; making it part of
  the kernel removes the `CatArrayBatchedCopy_contig` op.

### Edge cases

- **`rotary_dim == 0`** — caller has an early return; the Triton
  path mirrors it (`return x.contiguous()`).
- **`rotary_dim > head_dim`** — eager path raises `ValueError`;
  Triton path asserts the same precondition.
- **Non-contiguous `x`** — Triton kernel requires contiguous input;
  the wrapper calls `.contiguous()` defensively (no-op if already
  contiguous).
- **bf16 cos / sin dtype mismatch with x.dtype** — plan-5 P32 RoPE
  bf16 cast fix already lives in the caller (`cos.unsqueeze(-2).to(
  orig_dtype)`); the Triton kernel can assume `cos.dtype ==
  sin.dtype == x.dtype`. A defensive `tl.cast` inside the kernel
  guards against future bugs.

---

## Phase 36 — `sinkhorn_normalize` Triton FWD/BWD (replaces plan-5 P29 `torch.compile`)

> "sinkhorn_normalize是使用torch.compile实现的，但是forward和
> backward中有好多小的fused kernel。把sinkhorn的部分改成triton
> 实现fwd/bwd" — user, plan-6 kick-off (fusion hint #3).

Plan-5 P29 wrapped `sinkhorn_normalize` in
`torch.compile(fullgraph=True, dynamic=True)`. The P32 final trace
shows `Torch-Compiled Region` at `~21 ms / 16 calls` plus
`CompiledFunctionBackward` at `~41 ms / 16 calls` — `torch.compile`
collapsed the 39 small reductions but its CPU-side Dynamo overhead
and Inductor's BWD path are still non-trivial. P36 replaces both with
a single hand-rolled Triton kernel that loads the `[..., K, K]` matrix
into registers (K=4 at V4-Flash → 16 fp32 registers / row), runs the
20-iter row/col normalize entirely in registers, and writes the
result out. BWD uses the implicit-function-theorem closed form (the
doubly-stochastic projection has an analytic VJP that fits in
registers at K=4).

### Tasks

0. **Task list refinement** — re-confirm against the current trace
   that `Torch-Compiled Region` + `CompiledFunctionBackward` are
   still material. If P34 / P35 shifted the GPU-time mix, re-pin
   the P36 success metric.
1. **`primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/sinkhorn.py`** —
   new file with `_sinkhorn_fwd_kernel`, `_sinkhorn_bwd_kernel`,
   `SinkhornNormalizeFn`:
   - FWD kernel: one program per leading-axis tile (`BLOCK_LEADING =
     128`); loads `K * K` fp32 elements per row into registers,
     runs 20-iter alternating `row_sum / col_sum` + division
     entirely in registers, writes the result back as the caller's
     input dtype.
   - BWD kernel: analytic VJP. For K=4 the per-element Jacobian
     is closed-form; the kernel materialises a `[K, K] x [K, K]`
     Jacobian per row in registers and computes
     `dlogits = J^T * dnorm`.
2. **`hyper_connection.py::sinkhorn_normalize` routing** — accept
   a new keyword `use_triton: bool = False`; `HyperMixer.__init__`
   gains `use_triton_sinkhorn: bool = False`; `DeepseekV4HybridLayer`
   reads `config.use_v4_triton_sinkhorn`. Env knob
   `PRIMUS_SINKHORN_TRITON=1` (default `1`) flips the runtime
   path. The `use_compiled` (plan-5 P29) path stays in tree as the
   secondary fallback. Routing precedence at call site:
   `use_triton > use_compiled > eager`.
3. **Microbench `progress/p36/bench_sinkhorn.py`** — covers
   `K ∈ {4, 8}, leading ∈ {1024, 4096, 8192}` for FWD + BWD; reports
   `<ms> ms | <GB/s>` per call.
4. **EP8 proxy A/B trace** —
   `progress/p36/run_baseline_trace_ep8_p36.sh` with
   `PRIMUS_SINKHORN_TRITON=1`; render
   `develop/profile/profile-after-p36-ep8-<YYYYMMDD>.{md,html}`.
   Expect `Torch-Compiled Region` and `CompiledFunctionBackward`
   trace buckets to drop to ≈ 0.
5. **G39 — unit tests** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_p36_sinkhorn_triton.py`:
   - FWD parity vs eager and vs P29 compiled within `atol=1e-5
     rtol=1e-5` fp32 / `atol=1e-3 rtol=1e-3` bf16 at fast tier
     (`B=2, S=64, K=4`).
   - BWD `gradcheck` at fast tier fp32 small shape.
   - Release-tier shape (`B=1, S=4096, K=4`) marked
     `pytest.mark.slow`, FWD + BWD parity vs eager and vs P29
     within the same tolerance.
   - Parametrised over `n_iters ∈ {5, 20}` so the unrolled-loop
     compile path is exercised.
   - **Doubly-stochastic property check** — assert
     `row_sum.allclose(1, atol=eps*K)` and `col_sum.allclose(1,
     atol=eps*K)` on the FWD output. This is a model-quality
     contract independent of the eager path.
6. **Default flip note** — `PRIMUS_SINKHORN_TRITON` default is
   `"1"` at landing. The plan-5 P29 `use_compiled` knob stays as
   a secondary fallback; the `use_eager` no-flag baseline stays
   reachable via `PRIMUS_SINKHORN_TRITON=0 USE_V4_COMPILED_SINKHORN=False`.

### Design notes

- **Why K=4 fits in registers.** Each row's K×K = 16 fp32
  elements = 64 bytes ≈ 16 registers; 20 iterations × 2 K-element
  reductions = 40 reductions in registers. Total per-row register
  pressure (one row per warp lane): ~32 GP registers. On MI355
  each program has 256 registers / warp, so ~8 rows per program
  fit comfortably.
- **BWD analytic VJP.** The Sinkhorn iteration is a sequence of
  row-normalize + col-normalize steps. Each step's VJP is
  `J_row = (I/r) - (m * 1^T) / r^2` (or symmetric for col) where
  `r` is the row sum. Chain-rule through 39 steps fits in
  registers at K=4. For K=8 the register footprint doubles but
  still fits.
- **Why not unroll inside `torch.compile`.** Inductor unrolls
  Python loops at compile time, but its generated Triton kernel
  still launches separate kernels for FWD and BWD, and the
  Dynamo-side bookkeeping (`Torch-Compiled Region` event) is
  non-trivial overhead on every call. A hand-rolled Triton path
  emits one FWD kernel and one BWD kernel, with no Dynamo
  bookkeeping per call.
- **Cache key for FWD/BWD kernel.** Keyed on `(K, n_iters, eps,
  dtype)` — same as P29 but `shape` is not in the key (the kernel
  is shape-generic). For V4-Flash only one combination ever runs
  (`K=4, n_iters=20, eps=1e-6, dtype=bf16`), so the cache is a
  single entry.

### Edge cases

- **`K = 1`** — trivial case (matrix is `[1, 1]`, doubly-stochastic
  = 1.0). The kernel early-returns the input.
- **`K > 16`** — register pressure explodes; the kernel falls back
  to using shared memory for the K×K matrix (slow path; warns at
  rank 0). V4-Flash never hits this.
- **Plan-5 P29 cache key collision** — `PRIMUS_SINKHORN_TRITON=1`
  bypasses the `_compiled_sinkhorn_cache`; setting both
  `PRIMUS_SINKHORN_TRITON=0 USE_V4_COMPILED_SINKHORN=True`
  preserves the plan-5 P29 path.

---

## Phase 37 — HyperConnection elemwise Triton fusion

> "Hyper connection 里面有很多小算子，需要fusion" — user, plan-6
> kick-off (fusion hint #2).

P37 fuses the elemwise glue around `HyperMixer.compute_weights`,
`collapse`, and `expand`. The matmul inside `_packed_logits` stays
as a `torch.nn.functional.linear` call (GEMM is already
GPU-bound), but the **3 slices + 4 sigmoid/softmax + scale + base
chain** after the linear, plus the `(pre.unsqueeze * x).sum`
reduce in `collapse`, plus the `post.unsqueeze * out.unsqueeze`
outer-product in `expand`, are all small elemwise that the trace
buckets `elementwise_kernel_manual_unroll<128, 8>` (61.4 ms / 693
calls) and `vectorized_elementwise_kernel<8, CUDAFunctor_add>`
(13.6 ms / 933 calls). At 16 mixer calls per iter + 1 trunk-end
HyperHead per iter, the per-call cost is ~3-5 ms across the
elementwise chain.

### Tasks

0. **Task list refinement** — re-confirm the elementwise buckets'
   per-call cost post-P36 (P36's Sinkhorn replacement reduces the
   total elemwise launch count; the residual is what P37 targets).
   If the residual drops below 5 ms / iter, P37 is descoped.
1. **`primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/hyper_connection_glue.py`** —
   new file with three FWD + three BWD kernels and three
   `torch.autograd.Function`s:
   - `_hc_post_linear_glue_fwd_kernel`: takes `logits [..., (2+K)*K]`
     (fp32), `scale [3]` (fp32), `base [(2+K)*K]` (fp32); writes:
     * `pre_logit [..., K] = logits[..., :K] * scale[0] +
       base[:K]`
     * `post [..., K] = 2 * sigmoid(logits[..., K:2K] * scale[1] +
       base[K:2K])`
     * `comb_pre_sinkhorn [..., K, K] = softmax(logits[..., 2K:].view(
       ..., K, K) * scale[2] + base[2K:].view(K, K)) + eps`
     Output `pre = sigmoid(pre_logit) + eps` written too.
   - `_hc_collapse_fwd_kernel`: takes `x [..., K, D]`, `pre [..., K]`;
     writes `out [..., D] = (pre.unsqueeze(-1) * x).sum(-2)`.
   - `_hc_expand_outer_fwd_kernel`: takes `out [..., D]`,
     `post [..., K]`; writes `write [..., K, D] = post.unsqueeze(-1)
     * out.unsqueeze(-2)`. The `+ matmul(comb, x)` portion stays
     as a separate `torch.matmul` call; the trailing add is a
     fourth small kernel (`_hc_expand_add_fwd_kernel`) or a single
     `torch.add` (decided by microbench).
   - BWD kernels mirror the chain.
2. **`HyperMixer.compute_weights` routing** — gate on
   `os.environ.get("PRIMUS_HC_TRITON", "1") == "1"`; when on, call
   the three Triton kernels in order (post-linear glue → Sinkhorn
   → collapse / expand are called by the surrounding block).
   `HyperMixer.collapse / expand` and `HyperHead.forward` get the
   same gating.
3. **Microbench `progress/p37/bench_hyper_connection.py`** — covers
   `[B=1, S=4096, K=4, D=4096]` for each sub-kernel separately +
   end-to-end mixer call. Reports `<ms> ms | <GB/s>` per call.
4. **EP8 proxy A/B trace** —
   `progress/p37/run_baseline_trace_ep8_p37.sh` with
   `PRIMUS_HC_TRITON=1`; render
   `develop/profile/profile-after-p37-ep8-<YYYYMMDD>.{md,html}`.
5. **G40 — unit tests** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_p37_hyper_connection_glue_triton.py`:
   - Per-sub-kernel FWD parity vs the eager body within bf16
     `atol=1e-3 rtol=1e-3` (eager body extracted from
     `HyperMixer.compute_weights` / `collapse` / `expand`).
   - Composed end-to-end mixer / head FWD parity vs the existing
     eager class within the same tolerance (the existing
     plan-2 P14 HyperMixer / HyperHead unit tests
     `test_v4_p14_hyper_mixer.py` provide reference outputs;
     plan-6 P37 reuses the same fixtures).
   - BWD `gradcheck` at fast tier fp32 small shape.
   - Release-tier shape (`B=1, S=4096, K=4, D=4096`) marked
     `pytest.mark.slow`.
   - **Dtype-contract parametrise** — `(in_dtype, compute_dtype,
     out_dtype) ∈ {(bf16, fp32, bf16), (fp32, fp32, fp32)}` to
     verify the fp32-internal chain matches the existing eager
     contract.
6. **Default flip note** — `PRIMUS_HC_TRITON` default is `"1"` at
   landing.

### Design notes

- **The matmul stays.** `HyperMixer._packed_logits` ends in
  `F.linear(flat32 * rsqrt, fn.weight)` — a `[N, K*D] x [K*D,
  (2+K)*K]` GEMM. That goes to cuBLAS / Triton GEMM at peak;
  fusing it into the elemwise chain would re-implement a GEMM
  badly. The pre-linear `flat * rsqrt` is also separated from
  the linear because rsqrt is a small kernel and Inductor or
  cuBLAS does not fuse it across the matmul boundary.
- **fp32-internal contract.** The plan-2 P14 invariant is that
  `compute_weights` runs in fp32 and casts to bf16 only at the
  very end. The Triton kernels accept an explicit `compute_dtype`
  kwarg and enforce this; G40 parametrises on it.
- **`HyperHead.forward` reuse.** It is structurally similar to
  `HyperMixer.compute_weights` (rsqrt + linear + scale + base +
  sigmoid) but without the Sinkhorn step. P37 ships a thin
  wrapper that reuses `_hc_post_linear_glue_fwd_kernel` with
  `K_steps=1` (post-only, no comb).
- **Why `expand` is split (outer-product Triton + matmul torch +
  trailing-add torch).** The matmul `comb @ x` is `[K, K] x [K,
  D]` per row — a small GEMM but tensor-core-bound. The outer-
  product `post * out` is purely elemwise. The trailing add is
  one elemwise op; fusing it with `comb @ x` is hard (Inductor
  does not generally fuse across matmul). P37's first cut keeps
  the trailing add as `torch.add` and re-evaluates in the
  microbench.

### Edge cases

- **K=1 mixer** — degenerate case (only one parallel stream);
  the kernel must still produce a valid scalar Sinkhorn output
  (just the input through). G40 parametrises K=1 as a smoke.
- **`scale` / `base` dtype** — Parameters are fp32 (plan-2 P14
  invariant); the kernel casts to `compute_dtype` if different.

---

## Phase 38 — `Indexer.forward` scoring Triton FWD/BWD

> "Indexer 里面也有很多小算子，帮我融合成triton kernel" — user,
> plan-6 kick-off (fusion hint #4).

P38 fuses the `einsum + relu + mul + sum + causal_mask` scoring
chain in `Indexer.forward`. The chain runs once per CSA layer
(`compress_ratio == 4`) — 3 layers in the proxy. At the V4-Flash
shape `[B=1, S=4096, P=1024, H=8, Hd=128]` the chain currently
issues ~7 GPU ops (einsum + relu + mul + sum + mask alloc + mask
add) totalling roughly 5-8 ms per CSA layer in the residual trace.
The `topk` call and the post-`topk` `where(isneginf, -1, ...) + pad
cat` tail stay on host-side — `topk` is heavy GPU compute on its
own and benefits from being its own kernel.

### Tasks

0. **Task list refinement** — measure the per-layer Indexer
   scoring cost on the post-P37 trace; if it is < 3 ms / iter,
   P38 is descoped.
1. **`primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/indexer_score.py`** —
   new file with `_indexer_score_fwd_kernel`,
   `_indexer_score_bwd_kernel`, `IndexerScoreFn`:
   - FWD: takes `q_i [B, S, H, Hd]`, `k_icomp [B, P, Hd]`,
     `w_i [B, S, H]`, plus scalars `(compress_ratio, S_eff,
     P_eff)`; writes `scores [B, S, P]`. Internally:
     * Per program: load a `[BLOCK_S, BLOCK_P]` tile of
       `(s_tile, p_tile)`; for each `(s, p)` compute
       `dot_h = sum_d q_i[b, s, h, d] * k_icomp[b, p, d]` per
       head; apply `relu(dot_h)`; multiply by `w_i[b, s, h]`;
       sum over `h`; apply causal mask
       `tl.where((p+1)*compress_ratio - 1 <= s, 0, -inf)`;
       write `scores[b, s, p]`.
   - BWD: takes `dscores [B, S, P]`, recomputes the `relu` mask
     and walks back through the chain, accumulating
     `dq_i / dk_icomp / dw_i`.
2. **`Indexer.forward` routing** — gate on
   `os.environ.get("PRIMUS_INDEXER_TRITON", "1") == "1"`; when
   on, call `IndexerScoreFn.apply(q_i, k_icomp, w_i,
   compress_ratio)`; `topk` and the trailing tail stay as eager
   `torch.topk` / `torch.where` / `torch.cat`. When off, the
   eager body runs unchanged.
3. **Microbench `progress/p38/bench_indexer.py`** — covers
   `[B=1, S=4096, P=1024, H=8, Hd=128]` (V4-Flash) and a smaller
   smoke shape; reports FWD + BWD `<ms> ms | <effective FLOP/s>`.
4. **EP8 proxy A/B trace** —
   `progress/p38/run_baseline_trace_ep8_p38.sh` with
   `PRIMUS_INDEXER_TRITON=1`; render
   `develop/profile/profile-after-p38-ep8-<YYYYMMDD>.{md,html}`.
5. **G41 — unit tests** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_p38_indexer_triton.py`:
   - FWD `scores` parity vs eager `Indexer.forward` (extracted
     pre-`topk`) within bf16 `atol=5e-3 rtol=5e-3`.
   - Post-`topk` `topk_idxs` bit-equal vs the eager full chain
     (this is the load-bearing contract — sparse top-K
     indices must match exactly).
   - BWD `gradcheck` at fast tier fp32 small shape.
   - Release-tier shape `pytest.mark.slow`.
6. **Default flip note** — `PRIMUS_INDEXER_TRITON` default is
   `"1"` at landing.

### Design notes

- **Why keep `topk` on host-side.** `topk` is heavy GPU compute
  (`O(P log K)` per query); a stable, well-tuned `torch.topk`
  kernel exists. Re-implementing it in Triton inside the score
  kernel would conflate two unrelated kernels and bloat the
  register pressure. The plan-5 P31 CSA in-kernel `topk_idxs`
  gather already showed that `topk` lives best as its own kernel.
- **Causal mask materialised inline.** The eager path allocates
  a `[S, P]` mask tensor with `torch.where`; the Triton path
  materialises the same condition inline via
  `tl.where((p+1)*compress_ratio - 1 <= s, 0, -inf)`. No mask
  tensor, no HBM traffic.
- **`relu` recompute in BWD.** Standard FlashAttention-style
  trick: recompute the per-tile `relu` mask in BWD instead of
  saving it. Saves `~H * S * P` bits of HBM per CSA layer.

### Edge cases

- **Sequences shorter than `compress_ratio * topk`** — eager path
  returns `-1` sentinels in `topk_idxs` and pads with
  `pad_idxs`; the Triton score path produces `-inf` for invalid
  `(s, p)` positions and the host-side `topk + where + cat` tail
  handles the sentinel substitution.
- **`compress_ratio != 4`** — `Indexer` is currently only invoked
  with `compress_ratio == 4` (CSA layers), but the kernel accepts
  arbitrary `compress_ratio` as a scalar arg for forward-
  compatibility.

---

## Phase 39 — V4 Router post-logits Triton FWD/BWD (hash + topk shared)

> "v4_hash_router.py 生成logits之后，有很多小算子，融合成triton
> kernel" — user, plan-6 kick-off (fusion hint #5).

P39 fuses the post-logits chain shared between
`DeepseekV4LearnedRouter._compute_route` (`v4_topk_router.py`) and
`DeepseekV4HashRouter.forward` (`v4_hash_router.py`). Both routers
produce `logits [N, E]` (with `N = batch * seq_len`); the post-
logits chain is `score_fn → [+ expert_bias →] topk-K → gather →
denom → scale → sparse scatter (probs) + sparse scatter (routing_map)`.
Each pass is ~9 GPU ops per call; at V4-Flash 8 layers (3 hash + 5
learned) × 2 (FWD + BWD) the chain runs ~16 times per iter, ~40-50
total elementwise launches per iter.

### Tasks

0. **Task list refinement** — measure the per-layer router post-
   logits cost on the post-P38 trace; if it is < 2 ms / iter, P39
   is descoped.
1. **`primus/backends/megatron/core/transformer/moe/_triton/v4_router_post.py`** —
   new file with `_v4_router_post_fwd_kernel`,
   `_v4_router_post_bwd_kernel`, `V4RouterPostFn`:
   - FWD: takes `logits [N, E]` (fp32), `expert_bias [E]` (optional,
     fp32), `score_function` (enum: `0=softmax, 1=sigmoid,
     2=sqrtsoftplus`), `topk_indices [N, K]` (optional —
     hash-router pre-computes; learned-router computes inside the
     kernel via in-register tournament), `topk_scaling_factor`
     (fp32). Writes `probs [N, E]` sparse (mostly 0) and
     `routing_map [N, E] bool`. Algorithm:
     1. Apply `score_fn` to `logits` per row.
     2. If `topk_indices is None`: `[+ expert_bias]` then in-
        register tournament for top-K (K ≤ 8).
     3. Gather K weights from un-biased scores at the K indices.
     4. If `score_fn != softmax`: denom = `weights.sum()`;
        normalize; multiply by `topk_scaling_factor`.
     5. Sparse scatter `probs[n, indices] = weights` and
        `routing_map[n, indices] = True`.
   - BWD: reverses the chain. The scatter-VJP gathers
     `dprobs[n, indices]`; the denom-VJP is closed-form; the
     `score_fn`-VJP is the analytic derivative of softmax /
     sigmoid / sqrtsoftplus.
2. **`v4_topk_router.py::_compute_route` routing** — gate on
   `os.environ.get("PRIMUS_V4_ROUTER_TRITON", "1") == "1"`; when
   on, call `V4RouterPostFn.apply(logits, expert_bias,
   score_function_enum, None, topk, topk_scaling_factor)`. When
   off, the eager body runs unchanged.
3. **`v4_hash_router.py::DeepseekV4HashRouter.forward` routing** —
   same env knob; pre-computes `indices = tid2eid[flat_ids].long()`
   host-side (the hash table is parameter-free and the lookup is
   embarrassingly parallel; not worth fusing) and passes
   `topk_indices=indices` into the same kernel so the kernel
   skips its internal tournament.
4. **Microbench `progress/p39/bench_router_post.py`** — covers
   `[N=4096, E=256, K=6]` × `score_function ∈ {softmax, sigmoid,
   sqrtsoftplus}` × `{with, without bias}`. Reports
   `<ms> ms | <GB/s>` per call.
5. **EP8 proxy A/B trace** —
   `progress/p39/run_baseline_trace_ep8_p39.sh` with
   `PRIMUS_V4_ROUTER_TRITON=1`; render
   `develop/profile/profile-after-p39-ep8-<YYYYMMDD>.{md,html}`.
6. **G42 — unit tests** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_p39_router_post_triton.py`:
   - FWD `probs` / `routing_map` bit-equal vs eager across:
     * 3 score functions × {with / without bias} × {hash / topk}
       = 12 cases at fast tier (`N=128, E=32, K=4`).
     * Same 12 cases at release tier (`pytest.mark.slow`,
       `N=4096, E=256, K=6`).
   - BWD `gradcheck` at fast tier fp32 small shape.
   - Sparse output comparison: `(probs.nonzero(), routing_map.nonzero())`
     bit-equal between Triton and eager. This is the load-
     bearing contract; downstream `MoEDispatch` reads `probs` and
     `routing_map` exactly, so any sentinel / sort-order mismatch
     breaks dispatch.
7. **Default flip note** — `PRIMUS_V4_ROUTER_TRITON` default is
   `"1"` at landing.

### Design notes

- **Why share the kernel between hash and learned routers.** The
  post-logits chain is character-for-character the same; only the
  `topk_indices` source differs (kernel-internal tournament vs
  pre-computed table). Sharing avoids two parallel BWD VJPs.
- **`score_function` as enum.** The kernel parametrises on
  `score_function_enum: tl.constexpr` and emits three specialised
  binaries at compile time. At V4-Flash only `sqrtsoftplus` is
  exercised; the other two binaries are paid once per process
  (compile-time hits a cold-cache only).
- **In-register top-K tournament.** For `K ≤ 8` and `E ≤ 256`,
  the tournament is `O(E log K)` per row in registers; on MI355
  with 256 registers / warp this fits comfortably. For larger E
  (V4-Pro might be 384 experts), the kernel falls back to a
  shared-memory tournament; G42 parametrises both.
- **Sparse scatter contract.** `probs.scatter_(1, indices,
  weights)` and `routing_map.scatter_(1, indices, True)` are
  bijective writes (each `(n, e)` written at most once per row).
  The kernel uses `tl.store` (no atomics needed) for both. The
  output buffers are zero-init'd by the caller (via
  `torch.zeros_like` allocation); the kernel only writes the
  selected positions.

### Edge cases

- **`expert_bias = None`** — kernel parametrises on
  `HAS_BIAS: tl.constexpr` so the bias-add is conditionally
  compiled out for learned routers without bias.
- **`topk_scaling_factor == 1.0`** — kernel skips the final
  multiply (eager path's same shortcut).
- **`vocab_size` mismatch in hash router** — host-side bounds
  check on `flat_ids.max() < vocab_size` stays in the wrapper
  (the kernel only sees pre-validated `topk_indices`).
- **`flat_ids.numel() == 0`** — the wrapper short-circuits and
  returns empty `probs / routing_map`; the kernel is not invoked.

---

## Phase 40 — Plan-6 close-out

> Hand-off phase. No new kernels.

### Tasks

1. **`develop/perf/elem_fusion.md` (new file)** — one row per
   shipped fusion (P34..P39). Header references:
   - Cell format: `<ms> ms | <tflops or throughput>` per R2.5.
   - Columns: `Phase | Target op | Eager baseline (ms / throughput) |
     Triton-fused (ms / throughput) | Speedup | Source bench | EP8
     proxy delta (ms / iter)`.
   - One row per phase; the "EP8 proxy delta" cell pulls the
     end-of-phase EP8 proxy A/B number from
     `develop/profile/profile-after-p3X-ep8-<YYYYMMDD>.md`.
2. **`develop/perf/proxy_ep8.md` — append plan-6 rows.** Each
   phase commits adds its own row at the end of the table
   (the rule R2.5 frozen-history convention applies). The last
   row is `P40 final` with the cumulative iter time + corrected
   TFLOP/s/GPU + `vs P28 baseline` (which is the perpetual
   anchor) + `vs P32 final` (the plan-6 starting point).
3. **`progress/p33/p33-summary.md` ... `progress/p40/p40-summary.md`** —
   one R2.1 eight-section summary per phase, written at the time
   each phase commits. P40's summary doubles as the plan-6
   hand-off note (references all per-phase SHAs + the cumulative
   speedup).
4. **`run_deepseek_v4_flash_proxy.sh` — surface new env knobs.**
   The proxy script gains an explicit `${VAR:-1}` for each of
   the six new envs (`PRIMUS_STACK_GROUPED_WEIGHT_TRITON`,
   `PRIMUS_ROPE_TRITON`, `PRIMUS_SINKHORN_TRITON`, `PRIMUS_HC_TRITON`,
   `PRIMUS_INDEXER_TRITON`, `PRIMUS_V4_ROUTER_TRITON`). The
   header gains a "Plan-6 elemwise fusion knobs" section
   mirroring the plan-5 P32 final precedent (perf anchor +
   per-knob A/B fallback note).
5. **Status pinning (R2.4)** — every `[x]` row in Phase 33..40
   of `progress/status.md` gets the commit SHA + date pinned
   (the rule says the SHA goes in only after the phase commits;
   the row stays `[ ]` until then).
6. **Plan-6 close-out commit** — final commit message follows
   the plan-5 P32 convention: `docs(deepseek-v4)[plan-6][P40]:
   plan-6 close-out — elem_fusion.md + cumulative proxy_ep8 row +
   status pinning`.

### Design notes

- **Why a separate `elem_fusion.md` instead of appending to
  `attention_perf.md`.** `attention_perf.md` is plan-5-frozen
  (per R2.5 historical-record rule); mixing elemwise fusion
  rows into it would break the cell-format header (attention
  uses `visible-pair`-derived TFLOP/s, elemwise uses GB/s or
  effective throughput). Separate file = separate header
  contract.
- **R2.5 cell format.** Every cell in `elem_fusion.md` follows
  `<ms> ms | <tflops or throughput>` — even if the throughput
  metric is GB/s for memory-bandwidth-bound kernels. The header
  documents the per-row metric choice.

### Edge cases

- **A phase regresses end-to-end.** Per the plan-5 P32 RoPE bug
  precedent, the phase ships with its env default flipped to
  `0` and the phase row in `status.md` notes the regression in
  the "note" column. The phase's row in `elem_fusion.md` still
  shows the microbench win (the bench is honest); the
  `proxy_ep8.md` row records the proxy regression.
- **A phase is descoped at task-list refinement.** The phase
  row in `status.md` is marked `[-]` (mirrors the plan-5 P31b
  `BLOCK_K=64` revert convention); the corresponding
  `p3X-summary.md` documents why.

---

## Phase 41 — `Indexer.forward` post-einsum tail Triton fusion + plan-7 candidate inventory

> "关于 PRIMUS_INDEXER_TRITON，不需要把最开始的 einsum matmul
> 也融合到 triton kernel 里面，而是把后面的小算子融合。然后再
> 分析一下当前的 trace，我看到还有好多 elementwise / reduce 等
> 类型的 kernel，找到更多可以使用 triton 融合的优化。" — user,
> 2026-05-15 (post-P40 trace review).

P41 reopens plan-6 after the interim close-out at P40 to do two
things:

1. **Re-attempt the Indexer scoring fusion that was descoped at
   P38**, this time keeping the `einsum` matmul eager (cuBLAS /
   hipBLASLt wins at V4-Flash widths per the P38 descope analysis)
   and fusing **only** the post-einsum tail
   (`relu → mul(w_i) → sum(H) → + causal mask`).  The mask
   materialises inline via `tl.where` so no `[S, P]` host-side
   mask tensor is allocated.

2. **Seed plan-7 from the P40 trace**.  The four default-on
   plan-6 fusions reduced the elementwise/reduce residual to
   ~30 ms of in-model elementwise launches per iter; the next
   tier (≥ 200 ms of optimizer / grad-clip elementwise) lives
   outside the model and needs its own plan.  P41 inventories
   both buckets in `progress/p41/p41-candidates.md` with
   per-bucket trace evidence + estimated savings + proposed
   phase id.

### Tasks

0. **Task list refinement** — re-measure the per-CSA-layer
   Indexer scoring cost on the post-P40 trace.  If the chain is
   < 1 ms / iter (i.e. it disappeared after P34..P37 absorbed
   neighbouring elementwise launches), P41 ships only the
   plan-7 inventory and the Indexer kernel work is descoped.
1. **`primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/indexer_score_post.py`** —
   new file with `_indexer_score_post_fwd_kernel`,
   `_indexer_score_post_bwd_kernel`, `IndexerScorePostFn`:
   - FWD: takes `dot [B, S, H, P]` (output of the eager
     `torch.einsum("bshd,bpd->bshp", q_i, k_icomp)`),
     `w_i [B, S, H]`, plus scalars `(compress_ratio, S_eff,
     P_eff)`; writes `scores [B, S, P]`.  Per program:
     * Load a `[BLOCK_S, BLOCK_P]` tile of `(s_tile, p_tile)`;
     * Inner unrolled loop over heads (`H: tl.constexpr`):
       acc += `relu(dot[b, s, h, p]) * w_i[b, s, h]`;
     * Apply causal mask inline
       (`tl.where((p+1)*compress_ratio - 1 <= s, acc, -inf)`);
     * Store as `OUT_DTYPE`.
   - BWD: takes `d_scores [B, S, P]` + saved `dot [B, S, H, P]`
     + `w_i [B, S, H]`; emits `d_dot [B, S, H, P]` and
     `d_w_i [B, S, H]`:
     * `d_dot[b, s, h, p] = d_scores[b, s, p] * w_i[b, s, h]`
       where `dot[b, s, h, p] > 0`, else 0;
     * `d_w_i[b, s, h] = sum_p(d_scores[b, s, p] * relu(dot[b, s, h, p]))`.
     Saved `dot` is the same `dot` returned by the eager FWD
     einsum (no extra HBM round-trip vs the P38 design).
2. **`Indexer.forward` re-routing** — gate on the **re-purposed**
   `PRIMUS_INDEXER_TRITON` env (default `"1"`):
   * `"1"` (new default) → `dot = torch.einsum(q_i, k_icomp)`
     stays eager; `IndexerScorePostFn.apply(dot, w_i,
     compress_ratio)` runs the tail.
   * `"0"` → full eager body (current default after P38 descope).
   * Legacy P38 full-fuse path moves to a separate switch
     `PRIMUS_INDEXER_TRITON_FULL` (default `"0"`).  Stays in tree
     for small-shape paths and future tuning.
3. **Microbench `progress/p41/bench_indexer_tail.py`** — V4-Flash
   shape (`B=1, S=4096, P=1024, H=8, Hd=128, bf16`) + a smoke
   shape; three paths bench-compared: eager tail (extracted from
   `Indexer.forward`) / P41 Triton tail / legacy P38 full-fuse.
   `iters=20, warmup=5, n_input_copies=4, l2_flush_mb=512`.
4. **EP8 proxy A/B trace** —
   `progress/p41/run_baseline_trace_ep8_p41.sh` with
   `PRIMUS_INDEXER_TRITON=1`; render
   `develop/profile/profile-after-p41-ep8-<YYYYMMDD>.{md,html}`.
   A side is `PRIMUS_INDEXER_TRITON=0` (P40 production).
5. **G43 — unit tests** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_p41_indexer_tail_triton.py`:
   - FWD `scores` parity vs the eager **tail** (extracted into
     a helper inside the test, identical math) within bf16
     `atol=5e-3 rtol=5e-3`.
   - Post-`topk` `topk_idxs` bit-equal vs the eager full chain
     (the load-bearing contract from G41 carries over —
     sparse top-K indices must match exactly).
   - BWD `gradcheck` fp32 fast tier small shape.
   - Release-tier shape `pytest.mark.slow`.
   - Parametrise `compress_ratio ∈ {1, 4, 16}` so the causal
     mask geometry is exercised across the boundary cases.
6. **`progress/p41/p41-candidates.md`** — trace-driven inventory
   of additional fusion targets (see "Plan-7 candidate
   inventory" below).
7. **Default flip note** — `PRIMUS_INDEXER_TRITON` default is
   `"1"` ONLY if the EP=8 proxy A/B confirms a positive
   delta (≥ 1 ms / iter at the same `lm_loss`).  If the proxy
   A/B is within ± 1 ms (noise), the default stays at `"0"` and
   the kernel ships as opt-in (mirrors P38 / P39 descope
   precedent).

### Design notes

- **Why keep the einsum eager.** P38 fused the einsum + tail
  into one Triton kernel and lost to cuBLAS / hipBLASLt at
  V4-Flash widths (eager `einsum` 28 TFLOP/s vs Triton 20 TFLOP/s;
  see `progress/p38/p38-summary.md` §4.3 for the tensor-core
  utilisation comparison).  P41 only fuses the post-matmul tail
  which is bandwidth-bound (not compute-bound), so the
  comparison flips: Triton wins because every elementwise op
  costs a full HBM round-trip.  The matmul half stays at cuBLAS
  peak.
- **Why save `dot` for BWD instead of recomputing `relu` mask.**
  P38's "recompute the relu mask in BWD" trick saved
  `H * S * P` bits of HBM but cost `BLOCK_S * BLOCK_P * H * Hd`
  bf16 loads of `q_i` + `k_icomp` per program plus `H` `tl.dot`
  invocations.  P41 already has `dot` in HBM (it's the eager
  einsum output, which is needed for the FWD anyway); the BWD
  reads `dot` once per `(b, s, p)` tile and derives the relu
  mask in registers via `dot > 0`.  Net: one HBM read of
  `dot [B, S, H, P]` bf16 = 67 MiB at V4-Flash widths, vs
  P38's `q_i [B, S, H, Hd] + k_icomp [B, P, Hd]` = ~16 MiB +
  `H` Triton tensor-core dots ≈ 8.6 GFLOP.  Despite the larger
  HBM footprint, P41 BWD avoids the tensor-core under-utilisation
  that killed P38 BWD.
- **Causal mask materialised inline.** Same as P38 — no `[S, P]`
  mask tensor.  The condition
  `(p + 1) * compress_ratio - 1 <= s` is exact and cheap (one
  fmadd + compare per element).
- **No new state vs P38.** The eager einsum output `dot` is
  the same activation `Indexer.forward` already saves for
  backward (PyTorch's autograd captures it implicitly via the
  einsum call).  P41 passes `dot` explicitly into
  `IndexerScorePostFn.apply` so the saved tensor is identical
  to the eager path and the autograd graph is unchanged.

### Edge cases

- **`compress_ratio == 1`** — degenerate case where every pool
  position is "1 query token wide"; the causal mask collapses
  to `p <= s`.  The G43 parametrisation exercises this.
- **Sequences shorter than `compress_ratio * topk`** — same as
  P38 (the trailing `topk + where + cat` host-side tail handles
  sentinel substitution unchanged).
- **`out_dtype` cast contract** — FWD writes `scores` as
  `hidden.dtype` (the surrounding decoder layer's working dtype,
  typically bf16); the BWD reads `d_scores` in `hidden.dtype`
  and writes `d_dot` in bf16 + `d_w_i` in bf16 (the eager BWD
  also casts to bf16 before propagating to the einsum BWD).
- **`H` not a power of two** — V4-Flash uses `H = 8` which is
  `tl.constexpr`-friendly; the kernel asserts `H` is a power of
  two ≤ 16 and falls back to eager otherwise (mirrors P38's
  `is_triton_kernel_supported` guard).

### Plan-7 candidate inventory — sourced from the P40 trace

The post-P40 chrome trace (`output/amd/tas-mi355x-20260514/
p40_profile_plan6_close_pp1_ep8_seq4096/tensorboard/...
1778800838095839437.pt.trace.json`) shows the steady-iter window
at 523.67 ms with the following residual elementwise / reduce
buckets — anything not already named "_v4_*" / "_sinkhorn_*" /
"_hc_*" / "_apply_*" / "_stack_*" is still unfused.  P41 enrols
these into `progress/p41/p41-candidates.md` with proposed phase
ids (P42..P45 inside this plan if cheap, plan-7 otherwise):

| trace bucket | total / iter | launches | proposed phase | in-model? | rationale |
| --- | ---: | ---: | --- | --- | --- |
| `vec_elem<add_bf16>` (Adam ε-add) | 170.99 ms | 743 | **plan-7 P0a** | optimizer | TE / Apex `AdamFunctorMasterParamRemainder` calls the BF16 ε-add as a separate functor (743 launches).  Folding ε-add into the master Adam multi-tensor kernel saves ~150 ms / iter.  Out-of-model, needs Apex / TE coordination. |
| `multi_tensor<adam_master>` (TE fused Adam) | 45.92 ms | 321 | plan-7 P0b | optimizer | Already multi-tensor; further fusion needs a custom Triton optimizer kernel. |
| `vec_elem<bf16_copy>` (`.contiguous()` after permute) | 24.63 ms | 1303 | **P42** | model | Highest-count residual.  Sourced by `permute(0, 3, 1, 2, 4).contiguous()` materialising `gathered_k_v` for CSA's sparse top-K branch + a similar permute in the V4 attention output projection.  Folding these into the consumer Triton kernels (CSA FWD/BWD already in plan-5) absorbs the copies. |
| `vec_elem<bf16->fp32>` (pre-reduce promotion) | 20.99 ms | 1215 | plan-7 P1 | mixed | bf16 -> fp32 dtype promotion before `reduce_kernel`.  Mostly inside grad-norm clipping (`reduce<l2norm_bf16>` 7.76 ms is the consumer); the rest are inside the eager router post-logits path (P39 descope leftovers) and the dispatch / combine post-processing (TE / DeepEP). |
| `multi_tensor<scale>` (grad-scaling pre-allreduce) | 10.96 ms | 321 | plan-7 P0c | optimizer | TE-owned; out-of-model. |
| `vec_elem<mul_fp32>` (fp32 broadcast mul) | 9.59 ms | 109 | **P43** | model | Sourced by the eager V4 router post-logits chain (P39 descope leftovers — `softmax(logits) * scale -> scatter`) + `attn_sink` per-head scale broadcast + a few elementwise muls inside the V4 attention output projection.  P43 re-attempts P39 with a longer A/B (50-iter) to defeat NCCL noise and a measurement methodology that aggregates over multiple proxy runs. |
| `elem_unroll<mul_bf16>` (manual-unroll bf16 mul) | 8.31 ms | 122 | P43 | model | Same source as above (`elementwise_kernel_manual_unroll<128, 8>` is the BF16-mul variant used by V4 router + post-softmax scale).  Captured by the same fusion in P43. |
| `reduce<l2norm_bf16>` (grad-norm clipping) | 7.76 ms | 12 | plan-7 P0d | optimizer | TE multi-tensor + reduce; out-of-model.  Could fuse with `multi_tensor<l2norm>` (6.72 ms) → ~14 ms savings. |
| `multi_tensor<l2norm>` (per-param L2) | 6.72 ms | 321 | plan-7 P0d | optimizer | Same as above. |
| `elem_unroll<mul_fp32>` (manual-unroll fp32 mul) | 6.51 ms | 121 | P43 | model | Same as `vec_elem<mul_fp32>`. |
| `vec_elem<mul_bf16>` (AUnary scalar mul) | 5.71 ms | 12 | **P44** | model | Sourced by the V4 attention output projection's `out * scale` per-head broadcast.  Currently 12 launches × 0.48 ms — one per V4 attention call.  Foldable into the V4 attention FWD epilogue. |
| `vec_elem<add_fp32>` (fp32 add) | 5.20 ms | 36 | P43 | model | Mostly inside the eager router post-logits chain. |
| `elem_unroll<copy>` (bf16 direct-copy) | 5.78 ms | 47 | P42 | model | Same source as `vec_elem<bf16_copy>` — implicit `.contiguous()` after a permute.  Captured by the same fusion. |

**Plan-7 starter set (top-3 in-model targets):**

| phase | target | est. savings | scope |
| --- | --- | ---: | --- |
| **P42** | Fold `.contiguous()` + permute into V4 attention / CSA FWD inputs (eliminate `bf16_copy` + `elem_unroll<copy>`) | ~30 ms / iter | extend plan-5 V4 attention kernels |
| **P43** | V4 router post-logits with 50-iter A/B + per-`score_fn` specialised tile sizes (re-attempt P39 with better measurement) | ~5-10 ms / iter | extend plan-6 P39 |
| **P44** | V4 attention output projection epilogue (`out * scale + sinks`) folded into FWD kernel | ~3-5 ms / iter | extend plan-5 V4 attention kernels |

**Plan-7 starter set (top-2 optimizer-step targets):**

| phase | target | est. savings | scope |
| --- | --- | ---: | --- |
| **plan-7 P45** | Custom Triton fused Adam kernel that absorbs the BF16 ε-add into the master functor | ~150 ms / iter | wrap TE call site; no third_party edits |
| **plan-7 P47** | Fused grad-norm clip kernel (L2-norm reduce + max + scale) | ~14 ms / iter | wrap TE call site |

---

## Phase 42 — Permute + `.contiguous()` absorption into V4 attention / CSA FWD inputs

> Sourced from `progress/p41/p41-candidates.md` row #1 (P40 trace top
> residual at 24.63 ms / 1303 launches).

P42 folds the `.contiguous()` + permute chain that materialises
`gathered_k_v` / `q.contiguous()` / `kv.contiguous()` into the
existing plan-5 V4 attention FWD kernels.  No new Python kernel —
extend `_v4_attention_fwd_kernel`, `_v4_csa_attention_pool_sparse_fwd_kernel`,
and their BWD partners to read directly from the strided / permuted
source tensors so the explicit `.contiguous()` round-trip
disappears.

### Tasks

1. **Kernel patch — strided input loads.** Add a
   `PERMUTE_PATTERN: tl.constexpr` enum + strided-load helpers to
   `_triton/v4_attention.py`, `_triton/v4_csa_attention.py`, and
   `_triton/v4_csa_attention_pool_sparse.py`.  Each kernel accepts
   the original (pre-permute) tensor + a strides tuple and emits
   the per-program tile-load with the appropriate stride pattern.
2. **Python call-site cleanup.** Remove the explicit
   `.contiguous()` / `.permute(...).contiguous()` chains from:
   - `DeepseekV4Attention._attention_forward_via_triton` (Q/KV
     pre-pack for dense + HCA);
   - `csa_attention.py::CsaAttention.forward` (gathered KV
     materialisation);
   - `compressor.py::Compressor.forward` (output reshape).
3. **Env gate.** New `PRIMUS_V4_ATTN_FUSED_PERMUTE` env, default
   `"1"`.  Eager fallback keeps the explicit `.contiguous()` chain.
4. **Microbench
   `progress/p42/bench_v4_attention_strided_input.py`** — FWD/BWD
   wall-clock at V4-Flash widths (`[B=1, S=4096, H=64,
   head_dim=512]`) with strided vs contiguous inputs across the 3
   compress-ratio branches.
5. **G44 unit tests** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_p42_strided_v4_attention.py`:
   FWD parity vs eager `.contiguous() + kernel` within bf16
   `atol=1e-3 rtol=1e-3`; BWD `gradcheck` fast tier fp32;
   release-tier shape `pytest.mark.slow`.
6. **EP8 proxy A/B trace + report.**

### Design notes

- The strided-load path uses `tl.load(ptr, mask, other=0,
  cache_modifier=".cv")` so the L2 hits on the source tensor are
  preserved.  Tile loads with a stride pattern that isn't unit-row
  still hit at ~80% of contiguous bandwidth on MI355X.
- The biggest win comes from the CSA `gathered_k_v` materialisation
  which is ~13 ms / iter on its own (out of the 24.63 ms bucket).
  Folding the gather into the kernel's per-program top-K select is
  out of scope (that's a plan-5 P31 follow-up); P42 only absorbs
  the post-gather `.permute(0, 3, 1, 2, 4).contiguous()`.

---

## Phase 43 — V4 router post-logits + Compressor APE elementwise Triton fusion

> Sourced from `progress/p41/p41-candidates.md` row #2 (P40 trace
> ~30 ms residual across `vec_elem<mul_fp32>` 9.59 ms +
> `elem_unroll<mul_bf16>` 8.31 ms + `vec_elem<add_fp32>` 5.20 ms +
> `elem_unroll<mul_fp32>` 6.51 ms).

P43 is a two-pronged phase: (a) re-attempt the P39 router
post-logits fusion with better measurement methodology, and (b)
fuse the Compressor APE (per-position-encoding) elementwise chain.

### Tasks

1. **Router post-logits re-attempt.** Re-use the
   `_v4_router_post_fwd_kernel` / `_v4_router_post_bwd_kernel`
   pair from P39.  Add a per-`score_fn` tile-shape autotune table
   that picks `(BLOCK_N, BLOCK_E)` based on the score function
   constexpr.  Run a 50-iter EP8 proxy A/B with 3 independent runs
   to defeat the ±1-3 ms NCCL noise floor that swamped P39.
   Default `PRIMUS_V4_ROUTER_TRITON=1` if the aggregated mean
   shows a positive delta.
2. **Compressor APE Triton kernel.** New file
   `primus/backends/megatron/core/transformer/_triton/compressor_ape.py`
   with `_compressor_ape_fwd_kernel`, `_compressor_ape_bwd_kernel`,
   `CompressorAPEFn`:
   - FWD: takes `x [B, S, D]` + `ape [ratio, D]` + `bias [D]`;
     reshapes inline to `[B, P, ratio, D]`, multiplies by `ape`,
     reduces over `ratio`, adds `bias`; emits `out [B, P, D]`.
   - BWD: `d_x` via broadcast multiply (no reduce); `d_ape` via
     `sum` over `(b, p)` (tl.atomic_add for cross-block); `d_bias`
     via `sum` over `(b, p)`.
3. **Routing.** `Compressor.forward` gates on
   `PRIMUS_COMPRESSOR_APE_TRITON=1` (default `"1"`).
4. **Microbench
   `progress/p43/bench_router_post_v2.py`** — 50-iter, 3-run
   aggregate for the router on V4 production shape +
   `progress/p43/bench_compressor_ape.py` for V4-Flash
   `[B=1, S=4096, ratio=4, D=4096]`.
5. **G45 unit tests** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_p43_compressor_ape_triton.py`:
   FWD parity vs eager `reshape + mul + sum + add` within bf16
   `atol=1e-3 rtol=1e-3`; BWD `gradcheck`; release-tier slow.
   G42 (P39 router parity) carried green at all 3 score functions.
6. **EP8 proxy A/B trace + report.**

### Design notes

- The Compressor APE chain runs **twice per CSA layer** (once for
  the main pool, once for the Indexer's mini-pool) — 6 calls per
  iter at V4-Flash.  Combined elementwise budget: ~3-5 ms / iter.
- The P39 router re-attempt's success hinges on the 3-run
  aggregate methodology.  If the 3-run mean is still within the
  ±1 ms band, the router default stays at `"0"` (mirrors the
  original P39 descope precedent).

---

## Phase 44 — V4 attention FWD epilogue (`out * scale + sinks`) absorbed into kernel

> Sourced from `progress/p41/p41-candidates.md` row #3 (P40 trace
> 5.71 ms × 12 launches `vec_elem<mul_bf16>` AUnary).

P44 absorbs the per-head `out * scale + sinks` chain into the
V4 attention FWD kernel epilogue.  The eager path issues a separate
`vec_elem<mul_bf16>` (output scaling) + `vec_elem<add_bf16>` (sink
addition) per V4 attention call; both are absorbed.

### Tasks

1. **Kernel patch.** Extend `_v4_attention_fwd_kernel` to accept
   an optional `attn_sink [H]` argument; when provided, apply
   `out = out * scale + attn_sink[h]` in the FWD epilogue (after
   the softmax + V matmul).  Extend `_v4_attention_bwd_kernel` to
   compute `d_attn_sink` via `tl.atomic_add` over the head axis.
2. **Python call-site cleanup.** Strip the `out * scale + sinks`
   chain from
   `primus/backends/megatron/core/transformer/deepseek_v4_attention.py::_attention_forward_via_triton`
   and `attn_sink.py::AttentionSinkApplier.forward`.
3. **Env gate.** `PRIMUS_V4_ATTN_FUSED_SINK=1` (default `"1"`).
4. **Microbench
   `progress/p44/bench_v4_attention_sink_epilogue.py`** at
   V4-Flash widths for the FWD epilogue alone vs the full FWD.
5. **G46 unit tests** —
   `tests/unit_tests/megatron/transformer/deepseek_v4/test_p44_attn_sink_epilogue.py`:
   FWD parity vs eager `kernel + out * scale + sinks` within bf16
   `atol=1e-3 rtol=1e-3`; BWD `gradcheck`; release-tier slow.
6. **EP8 proxy A/B trace + report.**

### Design notes

- The `attn_sink` parameter is per-head (`[H]`) so absorbing it
  costs zero extra HBM traffic per program (one fp32 broadcast in
  registers).  The wins come from eliminating two separate kernel
  launches per V4 attention call.
- HCA + dense V4 attention paths both call the same kernel; CSA
  has its own kernel and is **not** modified by P44 (it already
  fuses the epilogue per the plan-5 P31 design).
