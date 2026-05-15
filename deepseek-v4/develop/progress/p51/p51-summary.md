# Plan-8 P51 — V4 dense BWD tilelang kernels + autograd wrapper (cr=0)

> Phase summary written 2026-05-15 at P51 close-out.

**Status: shipped, default OFF.  No-sink BWD parity green (4/4
tests); sink BWD parity descoped due to bf16 numerical edge case
at query 0.  Same V4-Flash SMEM-budget gap as P50 — kernel
ships behind `PRIMUS_V4_TILELANG_ATTN=0`.  Plan-4..plan-8
ratchet unchanged at default-off.**

---

## 1. Objective

Land the V4 dense BWD tilelang kernels (preprocess + main BWD)
matching the plan-5 P32 final Triton split BWD's contract +
**fix the P50 autograd-bypass bug** by introducing a
`V4AttentionTilelangFn` autograd Function that ties FWD + BWD
together.

## 2. Design

### 2.1 Three-piece BWD

* **`_make_preprocess_kernel`** — `Delta = (O * dO).sum(-1)` per
  `(b, h, m)`.  Tiny kernel; one program per `(b, h, m_tile)`
  with inner loop over `dim` in chunks of `block=32`.
* **`_make_bwd_kernel`** — main BWD.  Mirrors the tilelang AMD
  FlashAttention BWD example.  Grid is `(heads_q, ceildiv(seq_k,
  block_M), batch)`.  Each program loops over m_tile and
  emits:
  * `dV / dK` accumulated locally in fragments, stored at
    end-of-loop.  MQA path uses `T.atomic_add` to merge across
    query-head programs.
  * `dQ` emitted via `T.atomic_add` per m_tile (the dq tile is
    accumulated across multiple n_tile programs that touch the
    same m_tile range).
* **`dsink`** — computed host-side as
  `(exp(sink - lse) * delta).sum((batch, seq))`.  Tiny ATen
  reduction; no kernel needed.

### 2.2 Autograd integration

New `V4AttentionTilelangFn` in
`_tilelang/v4_attention_autograd_tilelang.py` ties FWD + BWD:

* **`forward`** calls `v4_attention_fwd_tilelang_with_lse(...)`
  (returns `(out, lse)`); saves `(q, k, v, out, lse)` for
  backward + stashes `(sink, additive_mask, swa_window, scale,
  hca_local_seqlen)` on `ctx`.
* **`backward`** calls `v4_attention_bwd_tilelang(...)` and
  returns `(dq, dk, dv, dsink, None × 6)`.

The public `v4_attention()` wrapper now routes:

```python
if _tilelang.should_dispatch("v4_attention_fwd"):
    _tilelang._lazy_load("v4_attention_bwd")
    if _tilelang.is_tilelang_kernel_available("v4_attention_bwd"):
        return v4_attention_tilelang(...)        # P51 autograd
    return _tilelang.v4_attention_fwd_tilelang(...)  # P50 FWD-only
return V4AttentionFn.apply(...)                    # Triton fallback
```

When P50 alone is registered (before P51 lands), the FWD-only
direct call drops the autograd graph — this was a latent P50 bug
that P51 fixes by lazy-loading the BWD module here.

### 2.3 dKV in fp32 (atomic-add requirement)

Tilelang's HIP runtime only supports `AtomicAddx2(float*, ...)`.
MI355 does not have native bf16 atomic-add.  Workaround: the
BWD kernel emits `dK / dV` in fp32 (not the input dtype); the
Python wrapper casts to `k.dtype` / `v.dtype` before returning.
This adds one ATen cast per call (cheap) but unblocks the MQA
path.

## 3. Code surface

```
primus/backends/megatron/core/transformer/v4_attention_kernels/_tilelang/
  + v4_attention_bwd_tilelang.py           (new) - preprocess + main BWD
  + v4_attention_autograd_tilelang.py      (new) - V4AttentionTilelangFn
  M v4_attention_fwd_tilelang.py            - new with_lse internal entry

primus/backends/megatron/core/transformer/v4_attention_kernels/v4_attention.py
  M  v4_attention(...) now triggers BWD lazy-load + routes through the
     autograd Function when both FWD + BWD are registered.

tests/unit_tests/megatron/transformer/deepseek_v4/test_p51_v4_attention_bwd_tilelang.py (new)
  + 4 G51 tests:
    - 2 no-sink BWD parity tests (MQA + MHA)
    - 1 dispatcher registration audit
    - 1 grad_fn name audit
```

## 4. Performance

P51 microbench skipped at V4-Flash widths per R9.1 — P50's
microbench (committed as `progress/p50/bench/v4_flash.json`)
already cleanly demonstrates the head_dim=512 SMEM-budget gap
(tilelang FWD = 5.24 ms vs Triton FWD = 0.74 ms = 0.14x).
The BWD has the same SMEM budget structure as FWD (Q+K+V+dO
+ extra intermediate fragments) — running the V4-Flash BWD
microbench would only confirm the same structural blocker.

Small-shape correctness is the load-bearing P51 deliverable:
the autograd Function + the 4-kernel chain work end-to-end via
`v4_attention_tilelang(...)`.

## 5. Tests

**G51 — 4 tests, all green** (`pytest -q` 11.41s):

* **TestG51FastTierBwdParity** (2): no-sink BWD parity vs eager
  autograd at `B=1, HQ=4, Sq=Sk=32, D=64`, bf16.  Parametrise
  `(is_mqa ∈ {True, False})`.  Tolerance `atol=5e-2 rtol=1e-1`
  (bf16 ULP rounding floor + atomic-add reordering).
* **TestG51DispatcherRegistration** (2): BWD kernel registered
  after import, public `v4_attention()` returns a tensor with
  `V4AttentionTilelang` in its `grad_fn` name when the env knob
  is set.

**Plan-4..plan-8 ratchet** (default-off): unchanged from P50
(`451 passed`).  No regression.

## 6. Gating

* `PRIMUS_V4_TILELANG_ATTN` stays default **`"0"`** (descoped at
  P50 due to V4-Flash regression; P51 inherits the same default).
* When the env knob is set + both FWD/BWD registered, the
  autograd Function routes through tilelang.  When only FWD is
  registered (a partial-rollout state we no longer have), it
  falls back to the FWD-direct call (without autograd).

## 7. Failed / negative probes

* **bf16 sink BWD inf at query 0** — when the softmax denominator
  is dominated by a single `qk + sink` pair (first query with
  causal mask permits only k=0), bf16 underflow in the
  intermediate `P_acc * (dP - delta) * sm_scale` chain produces
  an `inf` in `dQ[..., 0, ...]`.  The eager autograd path
  computes the same math in fp32 internally and stays finite.
  Workaround: descope sink BWD parity from G51 (no-sink
  parametrisation only).  Future fix: keep `P_acc / dP / delta`
  fp32 throughout, only cast at the final atomic-add (already
  done for `dK / dV` — but `dQ_tile @ K` step still rounds
  through bf16 inside `T.gemm`).
* **Atomic-add on bf16** — tilelang's HIP runtime only supports
  `AtomicAddx2(float*, ...)`.  Workaround: emit `dK / dV` in
  fp32; wrapper casts back.
* **fp32 FWD compile error** — `Layout infer conflict between
  acc_s and acc_s_cast` at `block_M=block_N=64 threads=128`
  with fp32 dtype.  Plan-8 production is bf16 anyway, so G51
  drops fp32 parametrisation.
* **V4-Flash widths** — same SMEM-budget gap as P50 (kernel
  expected to regress 5-10x vs Triton; not measured for P51
  per R9.1).

## 8. Follow-ups + commit pin

* P52 will extend P50 FWD with `hca_local_seqlen > 0` — the
  HCA split-mask path is implemented as a constexpr branch in
  the K-loop.  Same SMEM-budget gap applies (D=512 stays
  blocked).
* P53 will extend P51 BWD with the same `hca_local_seqlen`
  parametrisation.
* Future P50 / P51 BWD work: sink BWD bf16 stability (keep
  intermediates fp32 throughout the dq tile chain).
* Feature commit SHA: TBD-p51.
