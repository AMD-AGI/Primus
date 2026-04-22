# Primus Weekly Engineering Report — 2026-W17

## 1. Time Window

- Start: Monday 2026-04-20 00:00:00 Asia/Shanghai (GMT+8)
- End: Wednesday 2026-04-22 18:36 Asia/Shanghai (GMT+8) (report generation time)
- Branch observed: `origin/main`

## 2. Executive Summary

- **1 PR merged to `main`** in the weekly window (Mon 2026-04-20 00:00 GMT+8 → now).
- Category breakdown: **Bug Fix: 1**; all other categories: 0.
- **No Primus-Turbo version bump this week.** Current `PRIMUS_TURBO_COMMIT` pins in both `ci.yaml` (`333b68d`) and `benchmark.yaml` (`a4488f6c`) are unchanged since March; month-to-date Turbo drift on `main` is zero.
- **Megatron-LM upstream drift: `plan sync` recommended.** Pin is `d3528a21` (2026-03-06); upstream `main` HEAD is `bbc6b4d1` with **344 commits ahead**, including MoE router/dispatch improvements, NVFP4 DDP weights, MXFP8/FP8 DPA recipe, Mamba→Hybrid rename + DeepSeek Sparse Attention port, TransformerEngine `v2.14` bump, and several checkpoint/FSDP fixes.
- **torchtitan upstream drift: `urgent sync` recommended.** Pin is `5fb7cc2e` (2025-10-15); upstream `main` HEAD is `71291346` with **502 commits ahead**, spanning GraphTrainer/precompile, MoE token-dispatcher rewrite, Fused QKV GQAttention, FlexAttention + CP, FSDP2 `fully_shard`, and DeepSeek-V3 CooR precompile.
- **Primus-Turbo month-to-date drift: `monitor` (no action needed).** Both CI and benchmark pins are identical to their 2026-03-30 values.
- This week's only PR (#674) continues the ongoing hardening of the Megatron recompute patch: it realigns `recompute_layer_patches` with the latest upstream `TransformerBlock._checkpointed_forward` and adds pipeline-offset and FP8-no-grad unit tests to catch future upstream drift automatically.
- No CI/infra, dependency, docs, refactor, perf, or Turbo/version-update PRs merged in this window.

## 3. Weekly PR Update Table

| PR | Merged Time (GMT+8) | Category | Key Update |
| --- | --- | --- | --- |
| [#674](https://github.com/AMD-AGI/Primus/pull/674) `fix(megatron): adapt recompute_layer_patches to the upstream Megatron and add UT` (author: `lhzhang333`) | 2026-04-22 16:30 | Bug Fix | Rewrites inner `custom`/`checkpoint_handler` closures to be byte-identical to Megatron's latest `TransformerBlock._checkpointed_forward`; keeps the delegation fast-path when `recompute_layer_ids` is unset; seeds the upstream-source SHA256 fingerprint guard; adds UTs for pipeline-stage offset mapping and the FP8-no-grad skip rule; removes the stale `PrimusTransformerBlock` subclass. |

## 4. Megatron-LM Drift Overview

- Upstream: `https://github.com/NVIDIA/Megatron-LM.git` (`main`)
- Pinned in Primus `main` (`third_party/Megatron-LM`): `d3528a21301db2d12e92912b3ec025dc8a2ed4d6` — *fix(moe): fix TE general_gemm API change (#3582)*, 2026-03-06
- Upstream `main` HEAD: `bbc6b4d14bd8f787b803b3382147dac4cecb20ec` — *chore: rotate oncall schedule*
- Commit gap: **upstream is 344 commits ahead of Primus pin**
- Month-to-date movement on Primus side: pin advanced from `3bec9aa9` (2026-02-26) → `d3528a21` (2026-03-06) inside PR #654 merged 2026-04-10 (282-commit upstream catch-up).
- Recommendation: **plan sync** — several releases' worth of MoE, precision, and FSDP changes have accumulated; schedule a controlled bump rather than urgent.

### Notable upstream areas that have moved since the pin

- **MoE routing/dispatch**: new router score function (#3673), shared-expert overlap improvements including FlexDispatcher support (#2207), fix for unnecessary permute padding in non-quantized dispatch (#4038).
- **Low precision**: NVFP4 native weights for DDP (#4005); Enable FP8 DPA for MXFP8 recipe (#4066); TransformerEngine bumped to `release_v2.14` (#4331).
- **Mamba / Hybrid models**: Rename `MambaModel`/`MambaStack` → `HybridModel`/`HybridStack` (#4099); QK layernorm support for DPA in `MambaModel` (#4067); port DeepSeek Sparse Attention to `MambaModel` (#3553); fine-grained activation offloading (#4173).
- **Checkpointing**: add `--async-ckpt-use-cpu-shm` (#4355); remove cross-rank sync during checkpoint load & deprecate `state_dict_loader.load_state_dict` (#2864); fix potential coredump on save (#1871).
- **FSDP / Distributed Optimizer**: Megatron-FSDP set to 0.5.0; fix `expt_device_mesh` build for MoE-only (#3831); fix `decoupled_grad`/DistOpt mechanics for MFSDP (#4133); layerwise-optimizer fixes (#4272, #4138).
- **Context parallelism & DiT**: Gated delta net CP support (#2642); add `conditions_embeddings` to `TransformerBlock`/`TransformerLayer` for DiT (#4134, later partially reverted by #4270).

### Megatron-LM upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| MoE | Router new score function; shared-expert overlap for FlexDispatcher; MoE DPA fixes | NVIDIA/Megatron-LM #3673, #2207, #4038 | Could unlock additional MoE recipes (DeepSeek / Mixtral variants) already referenced in Primus `examples/megatron/configs/MI300X/deepseek_v*`. |
| Low precision | NVFP4 native DDP weights; FP8 DPA for MXFP8; TE bumped to `release_v2.14` | NVIDIA/Megatron-LM #4005, #4066, #4331 | Requires matching TE/AITER versions in Primus Dockerfile; likely coordinated with next Turbo+aiter bump. |
| Mamba/Hybrid | `MambaModel`→`HybridModel` rename; DeepSeek Sparse Attention port; activation offloading | NVIDIA/Megatron-LM #4099, #3553, #4173 | Breaking import rename — Primus patch system (`primus/backends/megatron`) must audit any `MambaModel`/`MambaStack` references before next bump. |
| Checkpoint I/O | Async checkpoint CPU-SHM; removed cross-rank sync load; coredump fix | NVIDIA/Megatron-LM #4355, #2864, #1871 | Expected improvement for Primus pretrain at scale; validate with Primus async-ckpt configs. |
| FSDP / DistOpt | Megatron-FSDP 0.5.0; MFSDP `decoupled_grad`/DistOpt fixes; layerwise-optimizer fixes | NVIDIA/Megatron-LM #3831, #4133, #4272, #4138 | Primus FSDP launch paths (`primus/modules/trainer/megatron/*`) should be re-benchmarked post-sync. |
| Context parallel / DiT | Gated delta-net CP; DiT `conditions_embeddings` argument | NVIDIA/Megatron-LM #2642, #4134 / #4270 | No direct Primus consumer yet; monitor before surfacing in configs. |

## 5. torchtitan Drift Overview

- Upstream: `https://github.com/pytorch/torchtitan.git` (`main`)
- Pinned in Primus `main` (`third_party/torchtitan`): `5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021` — *Deepseek-V3 toml file minor fix (#1894)*, 2025-10-15
- Upstream `main` HEAD: `7129134633eb4d3a161e604b8c648439a7eee785` — *ci: pin torchvision alongside torch in vlm 8-GPU workflow (#3047)*
- Commit gap: **upstream is 502 commits ahead of Primus pin**
- Month-to-date movement on Primus side: none (submodule SHA unchanged in April).
- Recommendation: **urgent sync** — the pin is six months stale; upstream has undergone major refactors (GraphTrainer precompile, MoE token dispatcher, FlexAttention CP, FSDP2) that block adopting any new Primus torchtitan-backend features.

### Notable upstream areas that have moved since the pin

- **GraphTrainer / precompile**: CooR precompile for DeepSeek V3 (#2916) and `aot_fx_trace` compile mode (#2975); precompile integration tests in CI (#3043); regional-inductor precompile (#2883); `enable_cudagraph` config flag (#3049); FSDP bucketing pass disabled (#3044).
- **MoE**: token dispatcher introduced replacing token reorderer (#2842); EP setup moved from trainer to config registry (#2960); revert `torch.bmm` → scatter-add (#2775); remove unnecessary MoE padding (#2774).
- **Attention / FlexAttention**: Fused QKV GQAttention (#2878); combine `q_norm`+`k_norm` into `qk_norm` (#2872); FlexAttention bitwise-deterministic tests (#2903, #2989); 2-tier compilation with FlexAttention (#2929); refactor inner attention module (#2761); CP + block_causal + FlexAttention position fix (#2780).
- **FSDP2 & compile**: replace `amp` + `replicate` with `fully_shard` (#2900); lazy import of FSDP mesh helpers for older PyTorch (#2888); SimpleFSDP wrapper shared across same-type modules (#2754); migrate to `.compile()` API (#2688).

### torchtitan upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| GraphTrainer / precompile | CooR precompile for DeepSeek V3; precompile for `aot_fx_trace`; regional-inductor precompile; `enable_cudagraph` flag | pytorch/torchtitan #2916, #2975, #2883, #3049 | Major perf/UX upgrade for torchtitan-backed training in Primus; currently unavailable behind stale pin. |
| MoE | New token dispatcher replacing token reorderer; EP setup moved to config registry | pytorch/torchtitan #2842, #2960 | API surface change for torchtitan MoE configs in `primus/backends/torchtitan/**`; patch notes will need an update after sync. |
| Attention | Fused QKV GQAttention; `qk_norm` consolidation; FlexAttention bitwise-deterministic tests; 2-tier compilation with FlexAttention | pytorch/torchtitan #2878, #2872, #2903, #2929 | Potential perf uplift for Primus torchtitan attention path; determinism tests useful for CI. |
| FSDP2 / compile | `fully_shard` replaces amp+replicate; `.compile()` migration; shared SimpleFSDP wrapper | pytorch/torchtitan #2900, #2688, #2754 | Breaking public API adjustments; Primus torchtitan launcher and patches must be re-validated. |
| CI / dependencies | `torchvision` pin alongside `torch` in VLM 8-GPU workflow; tj-actions version bumps | pytorch/torchtitan #3047, #3048 | Good hygiene reference for Primus torchtitan CI. |

## 6. Primus-Turbo Monthly Drift Overview

- Drift type: **in-repo**, not upstream — compares Turbo version/SHA referenced on Primus `main` now vs the latest commit at or before `month_start_ts = 2026-04-01 00:00 Asia/Shanghai` (`2026-03-31 16:00 UTC`).
- Turbo is **not a submodule** in Primus. Canonical version source:
  - `.github/workflows/ci.yaml` → `PRIMUS_TURBO_COMMIT` (also wired through `.github/workflows/docker/Dockerfile`)
  - `.github/workflows/benchmark.yaml` → `PRIMUS_TURBO_COMMIT`
- Reference commit at month start on `main`: `d2af5327` (*chore: update turbo & add disable_turbo_grouped_mlp_low_precision (#633)*, 2026-03-30)
- Current state on `origin/main`:
  - `ci.yaml` `PRIMUS_TURBO_COMMIT`: `333b68d7c81b722b21b4aad10cd250c45f15027c` — *fix sm_scale none bug (#263)*
  - `benchmark.yaml` `PRIMUS_TURBO_COMMIT`: `a4488f6cdb15cfff4383c61af7922bb50803f0ea` — *feat: update triton impl for mi300 & mi355 (#252)*
- Month-start state on `main`:
  - `ci.yaml`: `333b68d7c81b722b21b4aad10cd250c45f15027c` (same)
  - `benchmark.yaml`: `a4488f6cdb15cfff4383c61af7922bb50803f0ea` (same)
- **No Primus-Turbo drift in this comparison window.**
- Recommendation: **monitor**. However, note the *pre-existing* skew between the two YAML pins (CI pin `333b68d` is 5 commits ahead of benchmark pin `a4488f6c` in Primus-Turbo history) — tracked separately and unchanged this month.

### Notable areas changed since month start

- **No changes this window** — both `ci.yaml` and `benchmark.yaml` Turbo pins on `main` are byte-identical to their 2026-03-30 values.

### Primus-Turbo monthly drift table

| Component | Current Version/SHA | Month-start Version/SHA | Delta Summary | Key Changes | Evidence |
| --- | --- | --- | --- | --- | --- |
| `PRIMUS_TURBO_COMMIT` (CI build) | `333b68d7c81b722b21b4aad10cd250c45f15027c` (*fix sm_scale none bug (#263)*) | `333b68d7c81b722b21b4aad10cd250c45f15027c` | No drift (0 commits) | No changes this window. | [`.github/workflows/ci.yaml` L17](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml#L17) |
| `PRIMUS_TURBO_AITER_COMMIT` (CI build) | `e83f9903c07001a0ec29e85d223f6e6cdbe00859` | `e83f9903c07001a0ec29e85d223f6e6cdbe00859` | No drift | No changes this window. | [`.github/workflows/ci.yaml` L18](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml#L18) |
| `PRIMUS_TURBO_COMMIT` (benchmark) | `a4488f6cdb15cfff4383c61af7922bb50803f0ea` (*feat: update triton impl for mi300 & mi355 (#252)*) | `a4488f6cdb15cfff4383c61af7922bb50803f0ea` | No drift | No changes this window. | [`.github/workflows/benchmark.yaml` L9](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/benchmark.yaml#L9) |

## 7. Source Links

- Primus main branch: https://github.com/AMD-AGI/Primus/tree/main
- Primus weekly PR listing (window): https://github.com/AMD-AGI/Primus/pulls?q=is%3Apr+is%3Amerged+base%3Amain+merged%3A%3E%3D2026-04-19T16%3A00%3A00Z
- PR #674: https://github.com/AMD-AGI/Primus/pull/674
- Megatron-LM pin: https://github.com/NVIDIA/Megatron-LM/commit/d3528a21301db2d12e92912b3ec025dc8a2ed4d6
- Megatron-LM upstream HEAD (at report time): https://github.com/NVIDIA/Megatron-LM/commit/bbc6b4d14bd8f787b803b3382147dac4cecb20ec
- Megatron-LM compare: https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...main
- torchtitan pin: https://github.com/pytorch/torchtitan/commit/5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021
- torchtitan upstream HEAD (at report time): https://github.com/pytorch/torchtitan/commit/7129134633eb4d3a161e604b8c648439a7eee785
- torchtitan compare: https://github.com/pytorch/torchtitan/compare/5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021...main
- Primus-Turbo CI pin: https://github.com/AMD-AGI/Primus-Turbo/commit/333b68d7c81b722b21b4aad10cd250c45f15027c
- Primus-Turbo benchmark pin: https://github.com/AMD-AGI/Primus-Turbo/commit/a4488f6cdb15cfff4383c61af7922bb50803f0ea
- Month-start reference commit on `main`: https://github.com/AMD-AGI/Primus/commit/d2af5327

---

*Generated automatically by the Primus weekly report automation. Factual statements are derived from `git log origin/main`, the pinned submodule SHAs in `third_party/`, and the `PRIMUS_TURBO_COMMIT` values in `.github/workflows/{ci,benchmark}.yaml` as observed at 2026-04-22 18:36 GMT+8. Upstream-HEAD SHAs and commit counts are snapshots at report generation time.*
