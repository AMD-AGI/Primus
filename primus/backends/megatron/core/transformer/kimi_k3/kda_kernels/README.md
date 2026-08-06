# Kimi Delta Attention kernels

This package holds every compute backend for Kimi Delta Attention, used by
`KimiDeltaAttention`
(`primus/backends/megatron/core/transformer/kimi_k3/kimi_delta_attention.py`).
`__init__.py` is the single entry point: `resolve_kda_backend(name)` maps a
backend name to its functional entry, and the attention module imports only
from there. The layout mirrors `../../v4_attention_kernels/`.

## What KDA is

Gated DeltaNet with a **per-channel** forget gate instead of a per-head
scalar one. With state `S ∈ R^{K×V}`, per-channel retention `α_t = exp(g_t)`
and per-head write strength `β_t`:

```
S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t−1} + β_t k_t v_tᵀ
o_t = S_tᵀ q_t
```

Two consequences drive the implementations:

- The decay is applied **first**, so the transition is
  `Diag(α_t) − β_t k_t (α_t ⊙ k_t)ᵀ` — diagonal-plus-rank-1 with both
  low-rank vectors tied to `k`. That tying is the efficiency win over a
  general DPLR transition.
- `o_t` reads the **post-update** state, so the chunked form's intra-chunk
  attention matrix **retains its diagonal** (mask `triu(diagonal=1)`, not
  `triu(diagonal=0)`). This is the single detail most likely to be got wrong.

Setting `g` channel-constant reduces KDA exactly to Gated DeltaNet, which is
how the implementations are validated — see
`tests/unit_tests/megatron/transformer/kimi_k3/test_kda_collapse_to_gated_delta_rule.py`,
which asserts agreement with Megatron's own
`megatron.core.ssm.gated_delta_net.torch_chunk_gated_delta_rule`.

## Backends

| Name | Folder | Entry point | Notes |
| --- | --- | --- | --- |
| `eager` | `_eager/` | `eager_chunk_kda` | **The reference.** Pure-PyTorch chunkwise-parallel form; always importable, differentiable, device-agnostic. What every other backend is validated against. |
| `eager_recurrent` | `_eager/` | `eager_recurrent_kda` | Literal `O(T)` transcription of the recurrence. Correct by inspection; the oracle for the chunked form. Far too slow for training. |
| `fla` | `_fla/` | `fla_chunk_kda` | `flash-linear-attention`'s fused Triton `chunk_kda`. Lazily imported — `fla` is an optional dependency. **The production backend today**, and the speed baseline. |
| `flydsl` | `_flydsl_v1/` | `flydsl_chunk_kda` | Native FlyDSL kernel (WP9), **gfx950 / CDNA4 only**. Forward and backward both work; a `@flyc.kernel` computes the intra-chunk score matrices, the rest is batched torch GEMMs. Currently **slower than `fla`** — see "What the FlyDSL backend does and does not accelerate". |

`fla` and `flydsl` are loaded on demand (`load_fla_kda_backend`,
`load_flydsl_kda_backend`) so that `import ...kda_kernels` never fails
because of a dependency the caller did not select. `load_flydsl_kda_backend`
additionally requires gfx950 via `_require_gfx950()`, so selecting it on the
wrong hardware fails while the model is being built rather than inside a
kernel compile.

## Shared signature

Every backend is call-compatible, so the reference and a fused kernel can be
swapped at the call site:

```python
o, final_state = backend(
    q, k, v, g, beta,
    scale=None,                     # defaults to K ** -0.5, applied to q
    initial_state=None,             # [B, H, K, V]
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    chunk_size=64,
)
```

| Tensor | Shape | Notes |
| --- | --- | --- |
| `q`, `k` | `[B, T, H, K]` | |
| `v` | `[B, T, H, V]` | |
| `g` | `[B, T, H, K]` | **log** decay, `g ≤ 0` |
| `beta` | `[B, T, H]` | **already sigmoid-activated** |
| state | `[B, H, K, V]` | accumulation dtype (≥ fp32) |
| output | `[B, T, H, V]` | `v.dtype` |

The gate (`A_log` / `dt_bias` / lower bound) and the optional `q`/`k` L2 norm
are applied by the **caller** via `kda_gate` and `kda_l2norm`, not inside a
backend. That keeps one auditable copy of the gate math instead of one per
backend, and makes the eager and fused paths comparable term by term.

## Two `fla` version hazards

Both are handled inside `_fla/` rather than at the call site.

- **`use_beta_sigmoid_in_kernel`.** The HF `modeling_kimi_linear.py` passes
  this flag as `True` together with a raw `b_proj(x)`. No released `fla`
  (0.4.2 included) declares the keyword, so `chunk_kda` swallows it via
  `**kwargs` and leaves `beta` un-activated — silently changing the write
  strength's range from `(0, 1)` to all of `R`. The adapter therefore takes an
  already-activated `beta` and never forwards the flag.
  `test_kda_vs_fla.py::test_fla_chunk_kda_does_not_activate_beta_itself`
  pins this down empirically, so a future `fla` that *does* honour the flag
  will fail loudly rather than change numerics quietly.
- **`transpose_state_layout`.** A deprecated alias for `state_v_first` that
  emits a `DeprecationWarning`. Not forwarded; the returned state keeps
  `chunk_kda`'s documented `[N, H, K, V]` layout.

## Why the eager chunked form loops over columns

The intra-chunk score matrices need `exp(cum_g_r − cum_g_c)` per channel. The
algebraically equivalent two-matmul form `(K ⊙ Γ)(K / Γ)ᵀ` would be far
faster, but `1 / Γ` reaches `exp(320)` at `C = 64` with the bounded gate
(`g ≥ −5`), which overflows fp32. Building the matrix one column at a time
keeps every exponent `≤ 0` and holds peak memory at `[B, H, NC, C, K]` per
column instead of a `[B, H, NC, C, C, K]` intermediate.

The exponent is also zeroed on entries the caller is about to mask away.
Those entries have a large **positive** exponent, and while the forward
hides them, `exp` overflowing to `inf` there would make the backward
`inf * 0 = nan`. Megatron's per-head reference achieves the same with
`.tril()` on both sides of `.exp()`.

Removing both costs is the job of the FlyDSL backend below.

## What the FlyDSL backend does and does not accelerate

`_flydsl_v1/` replaces the two stages the eager form is deliberately slow at,
and leaves the rest to hipBLAS. Read this table before assuming a measurement
says something about the kernel.

| Stage | Where it runs | Cost at `B=1, T=4096, H=96, K=V=128`, bf16 |
| --- | --- | --- |
| within-chunk cumsum of `g` | torch | 111 µs |
| **`Aqk`, `Akk` — the two `[C, C]` score matrices** | **`@flyc.kernel`** | **338 µs** |
| `(I − L)^{-1}` UT transform | torch, Neumann doubling | 748 µs |
| `W = M(Γ⊙K)`, `U = MV` | torch batched GEMM | 218 µs |
| inter-chunk state sweep | torch, `NC`-step Python loop | dominant |

The kernel itself is *not* the bottleneck; the serial state sweep is. It is
`NC = 64` iterations of GEMMs far too small to fill an MI355X, so its cost is
launch latency, not arithmetic. `fla` avoids this by running the whole sweep
inside one Triton kernel, and a fused FlyDSL sweep kernel is the next thing to
write. Numbers, ratios and the full breakdown are in `wp9/RESULTS.md`.

### How the `1/Γ` overflow is avoided in the tiled form

The kernel *does* use the fast two-matmul form, but assembles each `[C, C]`
matrix from `16 × 16` blocks and picks a different reference row for the
cumulative log-decay in each block. For a block at row-block `i`, column-block
`j`, reference row `n`:

```
A[r, c] = Σ_d (q[r,d]·exp(cg[r,d] − cg[n,d])) · (k[c,d]·exp(cg[n,d] − cg[c,d]))
```

which is exact for **any** `n`, so `n` is free to be chosen for numerics:

- `j < i` → `n` is the **first row of row-block `i`**. `cg` is non-increasing in
  the row index, so both exponents are `≤ 0`. Nothing can overflow.
- `j == i` → `n` is the block **midpoint**, bounding `|exponent|` by
  `(16/2)·5 = 40`. The above-diagonal entries that the mask discards reach at
  most `exp((16−1)·5) = exp(75) ≈ 3.7e32`, inside fp32's `3.4e38`.

`16` is the largest sub-block with a *provable* margin — `32` would permit
`exp(155)` and survives only by luck of the data, and `64` is the naive form
and returns `nan`. This is the same device as `fla`'s `safe_gate` secondary
16-tile. `test_kda_flydsl_scores_kernel_survives_a_saturated_gate` pins it with
`g = −5` everywhere, where the naive factor really is `inf`.

### gfx950 toolchain repair

`_flydsl_v1/_lld_shim.py` exists because `ld.lld` in this image's ROCm SDK
wheel is a 26 KB trampoline that resolves the real linker from `argv[0]`, and
MLIR spawns it with a bare `argv[0]`. Without the repair **no** FlyDSL kernel
compiles at all — not this one and not the DeepSeek-V4 ones in
`../../v4_attention_kernels/_flydsl_v1/`.

The repair is a wrapper script that hands off to `lld -flavor gnu` by absolute
path. Putting it on `PATH` is **not** sufficient — measured: MLIR looks for the
tool inside the ROCm *toolkit* directory, so a `PATH`-only wrapper is never
consulted. It therefore goes into a shadow toolkit that symlinks every entry of
the real `$ROCM_PATH` (so `amdgcn/bitcode` and friends still resolve) with only
`ld.lld` replaced, and repoints `ROCM_PATH` at it. `ensure_usable_lld()` probes
for the defect the way MLIR triggers it and leaves a healthy toolchain
untouched.
