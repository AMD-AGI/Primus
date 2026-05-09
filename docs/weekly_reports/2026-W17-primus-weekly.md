# Primus Weekly Engineering Report — 2026-W17

## 1. Time Window

- Start: Monday 2026-04-20 00:00:00 Asia/Shanghai (GMT+8)
- End: Friday 2026-04-24 16:56 Asia/Shanghai (GMT+8) (report generation time)
- Branch observed: `origin/main`

## 2. Executive Summary

- **9 PRs merged to `main`** in the weekly window (Mon 2026-04-20 00:00 GMT+8 → Fri 2026-04-24 16:56 GMT+8).
- Category breakdown: **Bug Fix: 3**, **Performance Optimization: 2**, **CI/Infra: 2**, **Docs: 2**; Turbo/Dependency Version Update, Refactor, Other: 0.
- **No Primus-Turbo version bump this week.** Current `PRIMUS_TURBO_COMMIT` pins in both `ci.yaml` (`333b68d7`) and `benchmark.yaml` (`a4488f6c`) are unchanged since March; month-to-date Turbo drift on `main` is zero.
- **Megatron-LM upstream drift: `plan sync` recommended.** Pin is `d3528a21` (2026-03-06); upstream `main` HEAD is `a1165fab` with **365 commits ahead**, including MoE routing/dispatch improvements, MTP layers in token-per-expert logging, `MambaModel` → `HybridModel` rename, NVRx async compatibility, DDP parameter-layout refactor, TE `release_v2.14`, and several inference and checkpoint fixes.
- **torchtitan upstream drift: `urgent sync` recommended.** Pin is `5fb7cc2e` (2025-10-15); upstream `main` HEAD is `d40df991` with **514 commits ahead**, spanning GraphTrainer precompile (SAC + FSDP bucketing, CooR for DeepSeek-V3, CUDA-graph annotation pass), MoE token-dispatcher rewrite, Fused QKV GQAttention, FlexAttention + CP, FSDP2 `fully_shard`, and VLM workflow dependency pinning.
- **Primus-Turbo month-to-date drift: `monitor` (no action needed).** Both CI and benchmark pins are identical to their 2026-03-30 values.
- This week was dominated by **Megatron backend hardening** against upstream API drift: #672 realigns the patched MoE overlap `TransformerLayerSchedulePlan` after the upstream `post_attn` removal; #671 makes the muon optimizer wrapper runtime-signature aware against `get_megatron_optimizer()` drift; #674 rewrites `recompute_layer_patches` to be byte-identical to upstream `TransformerBlock._checkpointed_forward` and seeds a SHA256 fingerprint guard.
- Perf-category PRs: **#673** (`pp_warmup` optimization — parallel fwd+bwd warm-up on every PP rank, with a bit-identical loss-parity UT); **#684** introduces the opt-in `PRIMUS_EXIT_FAST` env to skip Python interpreter teardown after a successful Megatron trainer `cleanup()`, shaving ~22s of wall-time tail per process on MI355X DSV3 EP8 and ~2m off the MI300X Megatron-LM E2E UT suite.
- Tooling/infra: **#687** lands the shared backend-gap dashboard publishing toolchain (`tools/backend_gap_report/`) plus the initial torchtitan baseline report; this same shared site is being extended this week to also surface the weekly-report series. **#678** switches the JAX CI runner to a TAS node. Docs: **#683** published the initial W17 run, and **#690** refreshed it mid-week.

## 3. Weekly PR Update Table

| PR | Merged Time (GMT+8) | Category | Key Update |
| --- | --- | --- | --- |
| [#690](https://github.com/AMD-AGI/Primus/pull/690) `[Primus weekly report] 2026-W17` (author: `cursor[bot]`) | 2026-04-24 14:16 | Docs | Mid-week refresh of the automated W17 weekly report: extends the window through Fri 2026-04-24 09:10 GMT+8, brings the total to 6 PRs at the time of that run, and refreshes upstream drift snapshots for Megatron-LM and torchtitan. Superseded by this report. |
| [#684](https://github.com/AMD-AGI/Primus/pull/684) `feat(megatron): add PRIMUS_EXIT_FAST to exit training much faster` (author: `lhzhang333`) | 2026-04-24 11:17 | Performance Optimization | Adds an opt-in `PRIMUS_EXIT_FAST=1` env to `MegatronBaseTrainer.cleanup()`: after a successful cleanup and only when `on_error=False`, Primus calls `os._exit(0)` to skip normal Python interpreter/torchrun teardown. Measured ~22s saved on MI355X DSV3 4L EP8 post-train tail and ~2m saved on the MI300X Megatron-LM E2E UT suite; documented as experimental (not recommended for production graceful-shutdown paths). |
| [#687](https://github.com/AMD-AGI/Primus/pull/687) `feat(backend-gap): add dashboard publishing toolchain and torchtitan baseline report` (author: `WangLingxun`) | 2026-04-24 10:37 | CI/Infra | Ships the shared backend-gap dashboard publishing toolchain under `tools/backend_gap_report/` (static site shell + `build_dashboard_index.py` + `build_site_bundle.py`), seeds project-level skill docs, and publishes the initial TorchTitan-vs-upstream-main report set (`docs/backend-gap/reports/torchtitan/upstream-main/{report,summary}.md` + `dashboard-data/reports/torchtitan-upstream-main-2026-04-21.json`). This is the shared dashboard that this week's weekly-report flow extends to surface `Weekly Reports` as a first-class section. |
| [#672](https://github.com/AMD-AGI/Primus/pull/672) `fix(megatron): drop stale post_attn usage from patched MoE overlap schedule` (author: `WangLingxun`) | 2026-04-23 15:13 | Bug Fix | Aligns Primus `TransformerModelChunkSchedulePlan` with the current upstream `TransformerLayerSchedulePlan` after Megatron removed the `post_attn` node, eliminating a runtime `AttributeError` under `patch_moe_overlap=True`; preserves `ep_overlap_early_attn_memory_release` ordering and adds a regression test for the patched MoE-overlap schedule-plan path. |
| [#671](https://github.com/AMD-AGI/Primus/pull/671) `Fix/megatron muon optimizer signature` (author: `WangLingxun`) | 2026-04-23 15:07 | Bug Fix | Hardens the muon optimizer wrapper against `get_megatron_optimizer()` runtime signature drift; uses `inspect.signature` to support both keyword and mixed positional/keyword call patterns; resolves muon-specific args only when a muon optimizer is detected and passes non-muon optimizers through unchanged; adds regression tests for keyword vs positional `config_overrides`, muon vs non-muon, and correct parameter binding under different signatures. |
| [#678](https://github.com/AMD-AGI/Primus/pull/678) `[CICD]switch jax runner to tas node` (author: `llying-001`) | 2026-04-23 10:32 | CI/Infra | Switches the JAX unit-test runner to a TAS node in `.github/workflows/ci.yaml`. |
| [#683](https://github.com/AMD-AGI/Primus/pull/683) `[Primus weekly report] 2026-W17` (author: `cursor[bot]`) | 2026-04-23 07:11 | Docs | Adds the initial automated W17 weekly engineering report (`docs/weekly_reports/2026-W17-primus-weekly.md`) covering the Mon 2026-04-20 → Wed 2026-04-22 window. Superseded by #690 and by the current run. |
| [#673](https://github.com/AMD-AGI/Primus/pull/673) `opt(megatron): optimize pp_warmup and add corresponding UT` (author: `lhzhang333`) | 2026-04-22 20:32 | Performance Optimization | Rewrites `run_pp_warmup` to fabricate per-rank synthetic activations/output grads using `get_tensor_shapes`, so every PP rank runs one warm-up fwd+bwd **in parallel** (bypassing p2p) and exercises all CUDA/TE/FP8/NCCL lazy init paths concurrently; adds rigorous state-isolation (RNG + grad buffers) plus an end-to-end UT validating bit-for-bit loss parity and iter-1 speedup on `PP_SIZE>1`. |
| [#674](https://github.com/AMD-AGI/Primus/pull/674) `fix(megatron): adapt recompute_layer_patches to the upstream Megatron and add UT` (author: `lhzhang333`) | 2026-04-22 16:30 | Bug Fix | Rewrites inner `custom`/`checkpoint_handler` closures to be byte-identical to Megatron's latest `TransformerBlock._checkpointed_forward`; keeps the delegation fast-path when `recompute_layer_ids` is unset; seeds the upstream-source SHA256 fingerprint guard; adds UTs for pipeline-stage offset mapping and the FP8-no-grad skip rule; removes the stale `PrimusTransformerBlock` subclass. |

## 4. Megatron-LM Drift Overview

- Upstream: `https://github.com/NVIDIA/Megatron-LM.git` (`main`)
- Pinned in Primus `main` (`third_party/Megatron-LM`): `d3528a21301db2d12e92912b3ec025dc8a2ed4d6` — *fix(moe): fix TE general_gemm API change (#3582)*, 2026-03-06
- Upstream `main` HEAD: `a1165fabcad97eae3778f386839c233dfabf3f8b` — *Inference: Fix broken functional tests on gitlab (#4454)* (2026-04-24)
- Commit gap: **upstream is 365 commits ahead of Primus pin**
- Month-to-date movement on Primus side: pin advanced from `3bec9aa9` (2026-02-26) → `d3528a21` (2026-03-06) inside PR #654 merged 2026-04-10 (282-commit upstream catch-up). No further submodule SHA change in April.
- Recommendation: **plan sync** — several releases' worth of MoE, precision, Mamba→Hybrid rename, and FSDP/DistOpt changes have accumulated; schedule a controlled bump rather than urgent. This week's Primus-side fixes (#671, #672, #674) already pre-harden Primus against concrete upstream API drift points that must be validated during the sync.

### Notable upstream areas that have moved since the pin

- **MoE routing / dispatch**: new router score function (#3673), shared-expert overlap improvements including FlexDispatcher support (#2207), fix for unnecessary permute padding in non-quantized dispatch (#4038); MTP layers now counted in token-per-expert logging (#4412).
- **Mamba / Hybrid models**: `MambaModel`/`MambaStack` → `HybridModel`/`HybridStack` rename including the outside-of-`megatron/core` rename (#4159, follow-up to #4099); Mamba inference optimization (#4414); QK layernorm for DPA in `MambaModel` (#4067); port DeepSeek Sparse Attention to `MambaModel` (#3553); fine-grained activation offloading (#4173).
- **Low precision / TE**: NVFP4 native DDP weights (#4005); Enable FP8 DPA for MXFP8 recipe (#4066); TransformerEngine bumped to `release_v2.14` (#4331).
- **Checkpointing**: add `--async-ckpt-use-cpu-shm` (#4355); remove cross-rank sync during checkpoint load & deprecate `state_dict_loader.load_state_dict` (#2864); fix potential coredump on save (#1871); RL onload optimizer after logprobs (#4235).
- **FSDP / Distributed Optimizer / DDP**: Megatron-FSDP 0.5.0; `expt_device_mesh` fix for MoE-only (#3831); MFSDP `decoupled_grad`/DistOpt mechanics (#4133); layerwise-optimizer fixes (#4272, #4138); DDP refactor extracting parameter-layout computation into an optimizer classmethod (#3812); MFSDP log mcore detection only after imports succeed (#4400).
- **Inference / RL / misc**: NVRx async compatibility + defer resiliency import (#4420); misc inference fixes (#4397, #4454); RL token-throughput & packing metrics (#3877); nvtx_decorator checks `_nvtx_enabled` at call time (#4184); NullTokenizer for pretraining to reduce I/O (#4057); rampup batch-size scheduler replaced with custom step schedule (#4411, reverted earlier work #3779 via #4404).

### Megatron-LM upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| MoE | Router new score function;<br>shared-expert overlap for FlexDispatcher;<br>MoE DPA / permute-padding fixes;<br>MTP layers in token-per-expert logging | NVIDIA/Megatron-LM #3673, #2207, #4038, #4412 | Could unlock additional MoE recipes (DeepSeek / Mixtral variants) already referenced in Primus `examples/megatron/configs/MI300X/deepseek_v*`. |
| Mamba / Hybrid | `MambaModel` → `HybridModel` rename (outside-core rename landed);<br>Mamba inference opt;<br>DeepSeek Sparse Attention port;<br>fine-grained activation offloading | NVIDIA/Megatron-LM #4099, #4159, #4414, #3553, #4173 | **Breaking import rename** — Primus patch system (`primus/backends/megatron`) must audit any `MambaModel`/`MambaStack` references before the next bump. |
| Low precision / TE | NVFP4 native DDP weights;<br>FP8 DPA for MXFP8;<br>TE bumped to `release_v2.14` | NVIDIA/Megatron-LM #4005, #4066, #4331 | Requires matching TE/AITER versions in Primus Dockerfile; likely coordinated with next Turbo+aiter bump. |
| Checkpoint I/O | Async checkpoint CPU-SHM;<br>removed cross-rank sync load;<br>save coredump fix;<br>RL optimizer onload ordering | NVIDIA/Megatron-LM #4355, #2864, #1871, #4235 | Expected improvement for Primus pretrain at scale; validate with Primus async-ckpt configs. |
| FSDP / DistOpt / DDP | Megatron-FSDP 0.5.0;<br>MFSDP `decoupled_grad`/DistOpt fixes;<br>layerwise-optimizer fixes;<br>DDP parameter-layout refactor | NVIDIA/Megatron-LM #3831, #4133, #4272, #4138, #3812, #4400 | Primus FSDP/DDP launch paths (`primus/modules/trainer/megatron/*`) should be re-benchmarked post-sync; this directly touches paths exercised by this week's #673 `pp_warmup` optimization. |
| Inference / resiliency | NVRx async compatibility;<br>misc inference fixes;<br>nvtx decorator call-time gating | NVIDIA/Megatron-LM #4420, #4397, #4454, #4184 | Aligns with Primus inference-adjacent tooling; low-risk but worth validating Primus resiliency paths during the controlled bump. |
| Schedule plan API | `post_attn` node already removed from `TransformerLayerSchedulePlan` (consumed in Primus by #672) | Primus/#672, upstream schedule-plan change | Primus-side fix has already shipped; treat as confirmed drift that future bumps must keep in sync. |

## 5. torchtitan Drift Overview

- Upstream: `https://github.com/pytorch/torchtitan.git` (`main`)
- Pinned in Primus `main` (`third_party/torchtitan`): `5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021` — *Deepseek-V3 toml file minor fix (#1894)*, 2025-10-15
- Upstream `main` HEAD: `d40df991ac535108e428b0746a08b74a3cf6afc7` — *[GraphTrainer] Skip bucketing pass in precompile and re-enable tests (#3079)* (2026-04-24)
- Commit gap: **upstream is 514 commits ahead of Primus pin**
- Month-to-date movement on Primus side: none (submodule SHA unchanged in April).
- Recommendation: **urgent sync** — the pin is six months stale; upstream has undergone major refactors (GraphTrainer precompile, MoE token dispatcher, FlexAttention CP, FSDP2) that block adopting any new Primus torchtitan-backend features. This week, the shared backend-gap dashboard (#687) publishes the first tracked torchtitan-vs-upstream baseline report, which can be used to drive the sync plan.

### Notable upstream areas that have moved since the pin

- **GraphTrainer / precompile**: CooR precompile for DeepSeek V3 (#2916) and `aot_fx_trace` compile mode (#2975); precompile integration tests in CI (#3043); regional-inductor precompile (#2883); `enable_cudagraph` config flag (#3049); FSDP bucketing pass toggled (#3044), re-enabled with SAC improvements (#3060), and most recently skipped again in precompile with tests re-enabled (#3079); CUDA-graph kernel annotation pass (#2926); CPU offload pass for activation memory savings (#3064, reland); SAC pass refactor using `module_fqn` for layer boundaries (#3050).
- **MoE**: token dispatcher introduced replacing token reorderer (#2842); EP setup moved from trainer to config registry (#2960); revert `torch.bmm` → scatter-add (#2775); remove unnecessary MoE padding (#2774).
- **Attention / FlexAttention**: Fused QKV GQAttention (#2878); combine `q_norm`+`k_norm` into `qk_norm` (#2872); FlexAttention bitwise-deterministic tests (#2903, #2989); 2-tier compilation with FlexAttention (#2929); refactor inner attention module (#2761); CP + block_causal + FlexAttention position fix (#2780).
- **FSDP2 & compile**: replace `amp` + `replicate` with `fully_shard` (#2900); lazy import of FSDP mesh helpers for older PyTorch (#2888); SimpleFSDP wrapper shared across same-type modules (#2754); migrate to `.compile()` API (#2688); full DTensor for Qwen3 and Llama4 at TP region (#2149).
- **RL / trainer refactor**: RL trainer and generator refactors (#2985, #3001); rename inference example + consolidate vllm logical elements (#3045); drop `get_model_state_dict` in `push_model_state_dict` (#3066).
- **Datasets / CI / ROCm**: shuffle `HuggingFaceTextDataset` on re-loop and replay on resume (#3023); VLM 8-GPU workflow pins `torchvision` alongside `torch` (#3047); tj-actions version bumps (#3048); MI350 label used for all ROCm workflows (#2740); JIT/AOT tests gated off upstream partitioner regression (#3061).

### torchtitan upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| GraphTrainer / precompile | CooR precompile for DeepSeek V3;<br>precompile for `aot_fx_trace` + SAC/FSDP bucketing improvements;<br>regional-inductor precompile;<br>`enable_cudagraph` flag;<br>CUDA-graph kernel annotation pass;<br>CPU-offload activation pass | pytorch/torchtitan #2916, #2975, #3060, #3079, #2883, #3049, #2926, #3064 | Major perf/UX upgrade for torchtitan-backed training in Primus; currently unavailable behind stale pin. |
| MoE | New token dispatcher replacing token reorderer;<br>EP setup moved to config registry | pytorch/torchtitan #2842, #2960 | API surface change for torchtitan MoE configs in `primus/backends/torchtitan/**`; patch notes will need an update after sync. |
| Attention | Fused QKV GQAttention;<br>`qk_norm` consolidation;<br>FlexAttention bitwise-deterministic tests;<br>2-tier compilation with FlexAttention;<br>CP + block_causal + FlexAttention position fix | pytorch/torchtitan #2878, #2872, #2903, #2929, #2780 | Potential perf uplift for Primus torchtitan attention path; determinism tests useful for CI. |
| FSDP2 / compile / TP | `fully_shard` replaces amp+replicate;<br>`.compile()` migration;<br>shared SimpleFSDP wrapper;<br>full DTensor for Qwen3 / Llama4 TP | pytorch/torchtitan #2900, #2688, #2754, #2149 | Breaking public-API adjustments; Primus torchtitan launcher and patches must be re-validated. |
| RL trainer | RL trainer + generator refactors;<br>drop `get_model_state_dict` in `push_model_state_dict`;<br>inference-example rename + vllm consolidation | pytorch/torchtitan #2985, #3001, #3066, #3045 | Relevant to any Primus post-training / RL integrations on the torchtitan backend. |
| Datasets / CI / ROCm | Shuffle HF text dataset on re-loop;<br>`torchvision` pin in VLM workflow;<br>MI350 label across workflows;<br>tj-actions version bumps | pytorch/torchtitan #3023, #3047, #2740, #3048 | Good hygiene reference for Primus torchtitan CI and MI-series Docker dependency pinning. |

## 6. Primus-Turbo Monthly Drift Overview

- Drift type: **in-repo**, not upstream — compares Turbo version/SHA referenced on Primus `main` now vs the latest commit at or before `month_start_ts = 2026-04-01 00:00 Asia/Shanghai` (`2026-03-31 16:00 UTC`).
- Turbo is **not a submodule** in Primus. Canonical version source:
  - `.github/workflows/ci.yaml` → `PRIMUS_TURBO_COMMIT` (also wired through `.github/workflows/docker/Dockerfile`)
  - `.github/workflows/benchmark.yaml` → `PRIMUS_TURBO_COMMIT`
- Reference commit at month start on `main`: `76651575` (*[WIP][Megatron-LM] feat: reduce extra qkv transpose in attn (#625)*, 2026-03-31 14:19 GMT+8). The underlying Turbo pins at that commit are the same as the current values.
- Current state on `origin/main`:
  - `ci.yaml` `PRIMUS_TURBO_COMMIT`: `333b68d7c81b722b21b4aad10cd250c45f15027c` — *fix sm_scale none bug (#263)*
  - `ci.yaml` `PRIMUS_TURBO_AITER_COMMIT`: `e83f9903c07001a0ec29e85d223f6e6cdbe00859`
  - `benchmark.yaml` `PRIMUS_TURBO_COMMIT`: `a4488f6cdb15cfff4383c61af7922bb50803f0ea` — *feat: update triton impl for mi300 & mi355 (#252)*
- Month-start state on `main`: all three values identical to current.
- **No Primus-Turbo drift in this comparison window.**
- Recommendation: **monitor**. Note the *pre-existing* skew between the two YAML pins (CI pin `333b68d` is 5 commits ahead of benchmark pin `a4488f6c` in Primus-Turbo history) — tracked separately and unchanged this month.

### Notable areas changed since month start

- **No changes this window** — both `ci.yaml` and `benchmark.yaml` Turbo pins on `main` are byte-identical to their 2026-03-30 values.
- **No Turbo drift in this comparison window.**

### Primus-Turbo monthly drift table

| Component | Current Version/SHA | Month-start Version/SHA | Delta Summary | Key Changes | Evidence |
| --- | --- | --- | --- | --- | --- |
| `PRIMUS_TURBO_COMMIT` (CI build) | `333b68d7c81b722b21b4aad10cd250c45f15027c` (*fix sm_scale none bug (#263)*) | `333b68d7c81b722b21b4aad10cd250c45f15027c` | No drift (0 commits) | No changes this window. | [`.github/workflows/ci.yaml` L17](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml#L17) |
| `PRIMUS_TURBO_AITER_COMMIT` (CI build) | `e83f9903c07001a0ec29e85d223f6e6cdbe00859` | `e83f9903c07001a0ec29e85d223f6e6cdbe00859` | No drift | No changes this window. | [`.github/workflows/ci.yaml` L18](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml#L18) |
| `PRIMUS_TURBO_COMMIT` (benchmark) | `a4488f6cdb15cfff4383c61af7922bb50803f0ea` (*feat: update triton impl for mi300 & mi355 (#252)*) | `a4488f6cdb15cfff4383c61af7922bb50803f0ea` | No drift | No changes this window. | [`.github/workflows/benchmark.yaml` L9](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/benchmark.yaml#L9) |

## 7. Source Links

- Primus main branch: https://github.com/AMD-AGI/Primus/tree/main
- Primus weekly PR listing (window): https://github.com/AMD-AGI/Primus/pulls?q=is%3Apr+is%3Amerged+base%3Amain+merged%3A%3E%3D2026-04-19T16%3A00%3A00Z
- PR #671: https://github.com/AMD-AGI/Primus/pull/671
- PR #672: https://github.com/AMD-AGI/Primus/pull/672
- PR #673: https://github.com/AMD-AGI/Primus/pull/673
- PR #674: https://github.com/AMD-AGI/Primus/pull/674
- PR #678: https://github.com/AMD-AGI/Primus/pull/678
- PR #683: https://github.com/AMD-AGI/Primus/pull/683
- PR #684: https://github.com/AMD-AGI/Primus/pull/684
- PR #687: https://github.com/AMD-AGI/Primus/pull/687
- PR #690: https://github.com/AMD-AGI/Primus/pull/690
- Megatron-LM pin: https://github.com/NVIDIA/Megatron-LM/commit/d3528a21301db2d12e92912b3ec025dc8a2ed4d6
- Megatron-LM upstream HEAD (at report time): https://github.com/NVIDIA/Megatron-LM/commit/a1165fabcad97eae3778f386839c233dfabf3f8b
- Megatron-LM compare: https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...main
- torchtitan pin: https://github.com/pytorch/torchtitan/commit/5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021
- torchtitan upstream HEAD (at report time): https://github.com/pytorch/torchtitan/commit/d40df991ac535108e428b0746a08b74a3cf6afc7
- torchtitan compare: https://github.com/pytorch/torchtitan/compare/5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021...main
- Primus-Turbo CI pin: https://github.com/AMD-AGI/Primus-Turbo/commit/333b68d7c81b722b21b4aad10cd250c45f15027c
- Primus-Turbo benchmark pin: https://github.com/AMD-AGI/Primus-Turbo/commit/a4488f6cdb15cfff4383c61af7922bb50803f0ea
- Month-start reference commit on `main`: https://github.com/AMD-AGI/Primus/commit/76651575

---

*Generated automatically by the Primus weekly report automation. Factual statements are derived from `git log origin/main`, the pinned submodule SHAs in `third_party/`, and the `PRIMUS_TURBO_COMMIT` values in `.github/workflows/{ci,benchmark}.yaml` as observed at 2026-04-24 16:56 GMT+8. Upstream-HEAD SHAs and commit counts are snapshots at report generation time.*
