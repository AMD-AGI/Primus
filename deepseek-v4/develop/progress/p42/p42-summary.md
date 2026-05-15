# Plan-6 P42 — Permute + `.contiguous()` absorption (descoped at task-list refinement)

> Phase summary written 2026-05-15 at P42 close-out (task-list
> refinement → descope per R9.1 / R9.3).

**Status: descoped at task-list refinement.  Ships the trace-
attribution analysis + eliminate-candidates inventory; the
strided-input kernel modification is deferred to a future plan.**

---

## 1. Objective

Fold the `.contiguous()` + permute chain that the P40 trace
surfaces (`vec_elem<bf16_copy>` 24.63 ms × 1303 launches +
`elem_unroll<copy>` 5.78 ms × 47 launches = ~30 ms / iter) into
the plan-5 V4 attention FWD kernels by adding strided-input load
helpers.

## 2. Descope rationale

The 1303 `bf16_copy` launches are spread across **many** call
sites (mean 19 µs per launch).  Three independent observations
drive the descope:

### 2.1 The R9.3 forensic-attribution requirement

R9.3 ("Forensic attribution before fix") requires that every perf
fix attribute its target to a single Python source line via the
chrome trace's `External id` correlation.  P42 has 1303 launches
to attribute, which is itself a non-trivial deliverable (the
forensic helper for P29 covered 717 launches and was a 200-line
helper script).  Doing P42 properly is "build the forensic
helper" then "modify the plan-5 kernels".

### 2.2 The plan-5 kernel modification is high-risk

The plan-5 V4 attention kernels (`_v4_attention_fwd_kernel`,
`_v4_csa_attention_pool_sparse_fwd_kernel`, plus the matching
BWD kernels) are mature production code that ship with full
G23 / G24 / G26 / G27 / G29 ratchets at the V4-Flash release
tier (`pytest.mark.slow`).  Adding a `PERMUTE_PATTERN:
tl.constexpr` enum + per-pattern strided-load helpers means
re-emitting the per-program tile-load logic for **every**
kernel × every permute pattern.  At V4-Flash widths the new
ratchet would need re-running on every layout combination
(dense Q, dense KV, HCA Q, CSA gathered), and any regression
breaks the plan-5 release contract.

### 2.3 The win budget is below the noise floor

The R9.1 ("10 % de-scope rule") gives a per-bottleneck cutoff
of ~50 ms / iter (10 % of 510 ms steady iter).  The eliminate-
candidate inventory (§3) sums to ~30 ms; the 1303 launches are
distributed (no single bucket > 10 ms).  Per the rule, each
sub-target is below the de-scope threshold individually.

## 3. Trace-driven eliminate-candidate inventory

Sourced from the post-P40 trace `1778800838095839437`.  Each
row pins a specific Python source line + estimated savings if
the `.contiguous()` is eliminated.

| call site | trace bucket | est. launches / iter | est. savings (ms) | risk |
| --- | --- | ---: | ---: | --- |
| `deepseek_v4_attention.py:944,945` — `q_sbhd = q.transpose(0,1).contiguous()` + `kv_sbhd = kv.transpose(0,1).contiguous()` (core_attention dispatch) | `vec_elem<bf16_copy>` | ~16 (8 layers × 2 dirs) | ~0.3 | medium (needs core_attention to accept BSHD) |
| `deepseek_v4_attention.py:959` — `out = out.view(S,B,H,Dh).permute(1,2,0,3).contiguous()` (core_attention output) | `vec_elem<bf16_copy>` | ~8 | ~0.15 | medium (downstream consumer accepts BHSD?) |
| `deepseek_v4_attention.py:1173,1247` — `out = out_bh.transpose(1,2).contiguous()` (V4 Triton output unpack) | `vec_elem<bf16_copy>` | ~8 | ~0.15 | low (already inside V4 dispatch) |
| `v4_csa_attention_bwd.py:1230,1236` — `flat_topk = topk_idxs.contiguous().view(...)` + `queries.unsqueeze(0).expand(...).contiguous()` | `elem_unroll<copy>` | ~6 (3 CSA layers × 2 ops) | ~0.07 | low (no kernel change, just elide) |
| `rope_interleaved_partial.py:374,375` — `cos_flat = cos.contiguous().reshape(...)` + `sin_flat = sin.contiguous().reshape(...)` | `vec_elem<bf16_copy>` | ~16 (8 layers × 2) | ~0.05 | low (already inside RoPE Triton wrapper) |
| **Top-5 candidate sum** |  | ~54 | ~0.72 | mixed |
| (remaining ~1245 launches) | various | ~1245 | ~23.6 | high (need per-call-site forensic) |

The top-5 candidates account for **~3 %** of the `bf16_copy`
bucket and would save **~0.7 ms / iter** if all are eliminated
— below the proxy A/B noise floor (~±1 ms).

The remaining **~97 %** of launches come from Megatron / TE
pipeline machinery (Apex's `multi_tensor_apply`, TE's `cast`
shims, DeepEP's dispatch / combine) — these are out-of-model
and need TE / Apex coordination, not V4 source edits.

## 4. Decision

P42 is **descoped at task-list refinement**.  The strided-input
kernel modification is deferred to a future plan (plan-8 or
plan-9 attention re-work scope).  The plan-7 optimizer-step
fusion sweep (P45..P47) targets the dominant residual instead
(`Adam ε-add` 170.99 ms / iter; `grad-norm clip` 14.48 ms / iter;
`grad-scale` 10.96 ms / iter — collectively **15× larger** than
the P42 budget).

Future re-attempts of P42 should:

1. Land the forensic helper (`_forensics_p42.py` per R9.3) first.
2. Pick the **top-5 candidates above** as the initial scope —
   each is a single `.contiguous()` deletion + downstream consumer
   check.  Total budget ~0.7 ms / iter; likely below the proxy
   noise floor but useful to remove the per-launch overhead
   creep before plan-8.
3. Treat the strided-input kernel modification as a separate
   **plan-8** kick-off, not a one-phase change.

## 5. Code surface

No code change.  Documentation-only deliverable:

```
deepseek-v4/develop/progress/p42/
  + p42-summary.md (this file)
```

The status row in `progress/status.md` is marked `[-]` per the
R2.2 de-scope convention (mirrors the plan-5 P31b BLOCK_K=64
revert precedent).

## 6. Follow-ups + commit pin

* P43 (next): V4 router post-logits re-attempt with 50-iter A/B
  + Compressor APE elementwise fusion.
* P44: V4 attention FWD epilogue (`out * scale + sinks`) absorbed
  into kernel.
* Plan-7 P45..P48: optimizer-step fusion sweep.
* Feature commit SHA: TBD-p42.
