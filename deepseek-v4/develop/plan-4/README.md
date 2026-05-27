# Plan-4 — DeepSeek-V4 in Primus: in-tree Triton-fused attention kernels

> Plan-4 picks up where plan-3 left off (Turbo flash-attn + Turbo DeepEP
> wiring landed, see `../plan-3/`) and is **strictly scoped** to one
> outcome:
>
> Replace V4's eager-Python attention paths (`_attention_forward` and
> `_csa_forward` inside `DeepseekV4Attention`) with **Primus-owned Triton
> kernels** that we control end-to-end. The eager-Python paths and the
> `core_attention` (Turbo flash-attn) path stay in tree as references and
> as fallbacks; the Triton kernels land behind two new boolean switches
> (`use_v4_triton_attention`, `use_v4_triton_csa_attention`) that default to `False`.
>
> Why now: plan-3 P22 confirmed that Turbo's underlying aiter Triton
> backend cannot run V4's `head_dim=512 + sink + SWA` shape today (160 KiB
> SMEM hardware limit on MI355 vs. 256 KiB required by the aiter backward
> kernel; SWA also unsupported in aiter Triton at all). That blocks
> `use_turbo_attention=True` on V4 indefinitely until aiter ships an
> upgrade. Rather than wait, plan-4 ships our own kernels, tuned for V4's
> exact shape envelope (`head_dim=512`, `num_heads ∈ {64, 128}`, MQA
> single-latent KV, optional SWA, optional per-head learned sink, and the
> CSA per-query top-K gather pattern).
>
> Note (correction to the plan-4 brief): all three V4 attention layer
> kinds map onto **two** kernels, not three. `compress_ratio == 0` (dense
>
> - SWA + sink) and `compress_ratio == 128` (HCA — local SWA concatenated
> with compressed pool, shared sink, joint softmax) both call the SAME
> primitive in the eager code today (`_attention_forward(q, k, v, mask)`)
> — they only differ in (a) the K / V tensor lengths and (b) the additive
> mask, both of which the new kernel takes as inputs. So one Triton
> kernel covers `compress_ratio ∈ {0, 128}`. `compress_ratio == 4` (CSA —
> per-query top-K gather over a compressed pool, joint softmax with the
> SWA local branch) is a different math shape and gets its own kernel.
>
> Every other follow-up (FP8, full-Flash convergence run, long-context,
> HF state-dict adapter, MoE perf, Turbo / aiter SMEM upgrades) stays
> out of scope and is owned by a future plan.

## References

- Plan-3 wrap-up: `../plan-3/02-phase-details.md` (P22 status box + P23
status box)
- Plan-3 P22 smoke + SMEM analysis: `../progress/p22/`
- Aiter Triton attention reference (read-only):
`../../aiter/aiter/ops/triton/attention/`
  - `mha.py` + `_triton_kernels/attention/mha.py` — fwd kernel skeleton
  (sink-LSE handling, partial-RoPE head split, fp8 hooks)
  - `unified_attention.py` — vLLM-style sliding-window forward (paged-KV
  inference; not directly reusable for training but sets the SWA mask
  pattern)
  - `unified_attention_sparse_mla.py` — sparse-MLA decode kernel that
  reads a per-query `topk_indices` table; closest existing reference
  to the CSA pattern (decode-only — no backward — but the indexed-load
  structure is the model)
  - `mha_onekernel_bwd.py` / `mha_fused_bwd.py` — backward kernel
  skeletons (one-kernel = no atomics; fused = atomics)
- V4 attention class (eager-Python paths to align against):
`primus/backends/megatron/core/transformer/deepseek_v4_attention.py`
  - `_attention_forward(q, k, v, mask)` — dense + HCA primitive
  - `_csa_forward(...)` — joint local SWA + per-query top-K
- V4 attention shape source-of-truth (test fixtures):
  - `primus/configs/models/megatron/deepseek_v4_flash.yaml`
  (`hidden_size=4096, num_attention_heads=64, kv_channels=512, qk_pos_emb_head_dim=64, q_lora_rank=1024, o_lora_rank=1024, o_groups=8, attn_sliding_window=128, attn_sink=true, index_topk=512`)
  - `primus/configs/models/megatron/deepseek_v4_pro.yaml`
  (`hidden_size=7168, num_attention_heads=128, kv_channels=512, qk_pos_emb_head_dim=64, q_lora_rank=1536, o_lora_rank=1024, o_groups=16, attn_sliding_window=128, attn_sink=true, index_topk=1024`)

## Documents

- `[01-roadmap.md](./01-roadmap.md)` — phase overview, dependency graph,
exit criteria, and how plan-4 sits relative to plan-3.
- `[02-phase-details.md](./02-phase-details.md)` — phase-by-phase task
list, kernel design notes, edge cases, and risks.
- `[03-test-strategy.md](./03-test-strategy.md)` — gates per phase (the
forward + backward equivalence tests at V4-Flash and V4-Pro shapes are
the release gate).

## Scope


| In scope                                                                                                                                                                                            | Out of scope                                                                                                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `v4_attention` Triton kernel (forward + backward) covering `compress_ratio ∈ {0, 128}` — flash-attn-style with optional per-head learned sink, optional SWA, optional `[Sq, Sk]` additive bias mask | FP8 / FP4 / mxfp4 quantised forward (separate perf plan)                                                                               |
| `v4_csa_attention` Triton kernel (forward + backward) for `compress_ratio == 4` — fused local SWA branch + per-query top-K sparse branch with joint softmax and shared sink                         | RoPE fusion into the kernel (V4 applies RoPE before the kernel call; the kernel sees post-RoPE Q / K)                                  |
| `use_v4_triton_attention` + `use_v4_triton_csa_attention` config / CLI switches (default `False`); precedence vs. `use_turbo_attention` documented                                                                | Switching the default to `True` (a release-tagging step that follows after plan-4's smoke evidence is in)                              |
| Forward + backward unit tests at V4-Flash + V4-Pro shapes; tolerance budget per dtype documented                                                                                                    | Numerical alignment with `PrimusTurboAttention` (the Triton kernels are aligned against the eager-Python reference, not against Turbo) |
| `run_deepseek_v4.sh` smoke at TP=1 PP=1 EP=8 with `use_v4_triton_attention=True` and `use_v4_triton_csa_attention=True` (and Turbo attention off)                                                                 | Convergence run + long-context + EP scaling smokes (separate plan)                                                                     |


## Phase Map (added under Phase 23 in `../progress/status.md`)


| #       | Theme                                                                                                           | Source request                                                                                           |
| ------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **P24** | Test harness + V4 attention shape fixtures + eager-Python reference exposed for unit tests                      | "There must be unit tests; the Triton version and the original small-op version must produce aligned results — both forward and backward; test shapes follow V4-Flash and V4-Pro attention shapes." |
| **P25** | `v4_attention` Triton kernel (forward + backward) for `compress_ratio ∈ {0, 128}` + `use_v4_triton_attention` switch   | "My understanding: develop two attention versions — one normal attention used for compress=0 and 128 — implemented in Triton."                         |
| **P26** | `v4_csa_attention` Triton kernel (forward + backward) for `compress_ratio == 4` + `use_v4_triton_csa_attention` switch | "Plus a CSA version — use the two switches `use_v4_triton_attention` and `use_v4_triton_csa_attention` to control whether the new Triton versions are used."                                  |
| **P27** | V4 attention dispatch wiring + `run_deepseek_v4.sh` smoke (TP=1 PP=1 EP=8)                                      | "The PyTorch small-op version is currently in tree — keep it. You can add new Triton versions and use `use_v4_triton_attention` / `use_v4_triton_csa_attention` switches to control which is used."              |


