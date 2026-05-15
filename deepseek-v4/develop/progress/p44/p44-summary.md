# Plan-6 P44 — V4 attention FWD epilogue absorption (descoped at task-list refinement)

> Phase summary written 2026-05-15 at P44 close-out.

**Status: descoped at task-list refinement (R9.1 / R9.3).  The
originally-scoped target — folding `out * scale + sinks` into the
V4 attention FWD kernel epilogue — does not map to existing code.
The `attn_sink` is already absorbed into the softmax via a virtual
row in `_v4_attention_fwd_kernel` (lines 289-295) and the
`softmax_scale` is already a kernel parameter (line 580 of
`deepseek_v4_attention.py`).  The trace bucket `vec_elem<mul_bf16>`
AUnary 12 launches × 0.48 ms = 5.71 ms is a different (still
unidentified) call site that requires R9.3 forensic External-id
attribution before a fusion target can be chosen.**

---

## 1. Objective (originally scoped)

Fold the per-head `out * scale + sinks` chain into the
`_v4_attention_fwd_kernel` epilogue based on the trace evidence
that `vec_elem<mul_bf16>` AUnary contributed 5.71 ms / 12 launches
per iter at the P40 anchor.

## 2. Why the descope

Three independent observations drive the descope:

### 2.1 Sink + scale are already fused in the kernel

Reading `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/v4_attention_fwd.py`:

```python
# Lines 289-295
# uses sink as a candidate row maximum; sink contributes to
# the softmax via a virtual row.
if HAS_SINK:
    sink_h = tl.load(SINK + qhid).to(tl.float32)
    m_new = tl.maximum(m_i, sink_h)
    ...
    beta = tl.exp(sink_h - m_new)
```

The plan-4 P25 kernel design already takes `sink: Optional[torch.Tensor]`
as a parameter and integrates it via the FlashAttention-style
"virtual row" trick — no post-attention `out + sinks` chain exists
to fuse.  The `softmax_scale` is similarly passed as a kernel
parameter (`deepseek_v4_attention.py:580`).

### 2.2 The trace bucket source is unattributed

The P40 trace shows:

```
12 launches x 476.0 us = 5.71 ms (1.1%)  vec_elem<mul_bf16> (AUnary)
```

`AUnary` is a unary functor that multiplies by a scalar.  The
per-launch cost (476 µs) and the 12-launch count don't cleanly
map to any V4-specific call site:

* V4 attention FWD has 8 calls per iter (one per layer), not 12.
* MoE residual add is `+`, not `*`, so wouldn't go through this
  functor.
* Token-dispatch combine output scale (`combine * topk_weights`)
  is 16 calls / iter (8 layers × 2 dirs), still not 12.

Identifying the exact source needs the R9.3 forensic helper
(`_forensics_p44.py`); writing that helper is itself a separate
deliverable.

### 2.3 The win budget is below the R9.1 cut-off

5.71 ms / iter at the P40 anchor of 510 ms = **1.1 %**, well
below the R9.1 10 % cut-off.  Per the rule, the phase de-scopes.

## 3. Code surface

No code change.  Documentation-only deliverable:

```
deepseek-v4/develop/progress/p44/
  + p44-summary.md (this file)
```

The status row in `progress/status.md` is marked `[-]` per the
R2.2 de-scope convention.

## 4. Follow-ups + commit pin

* Future re-attempts of P44 should land the R9.3 forensic helper
  first.
* Plan-8 attention re-work is the natural home for any remaining
  attention-side elementwise residual cleanup.
* Plan-7 P45..P48 now target the dominant residual instead (Adam
  optimizer step at ~242 ms / iter; 42× larger than the P44
  budget).
* Feature commit SHA: TBD-p44.
