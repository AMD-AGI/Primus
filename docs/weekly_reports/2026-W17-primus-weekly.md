# Primus Weekly Engineering Report — 2026-W17

## 1. Time Window

- Week ID: `2026-W17` (ISO week)
- Start: 2026-04-20 00:00 (Asia/Shanghai, GMT+8)
- End (report generation time): 2026-04-22 17:59 (Asia/Shanghai, GMT+8)
- Base branch analyzed: `main` of [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus)

## 2. Executive Summary

- Total merged PRs to `main` in window: **1**.
- Category breakdown: Bug Fix ×1; Performance Optimization ×0; Turbo/Dependency Version Update ×0; CI/Infra ×0; Refactor ×0; Docs ×0; Other ×0.
- No Turbo or dependency version updates landed this week.
- The only merged change refines the Primus patch over Megatron `TransformerBlock._checkpointed_forward`, adds a source-fingerprint guard, and adds unit tests for pipeline-stage offset handling and the FP8/FP4 no-grad skip rule. See PR [#674](https://github.com/AMD-AGI/Primus/pull/674).
- Megatron-LM submodule is **drifting behind upstream**:
  - Pinned SHA in Primus `main`: `d3528a2` (2026-03-06).
  - Upstream `main` HEAD: `bbc6b4d` (2026-04-22).
  - Upstream is **344 commits ahead** of the pinned submodule.
- Notable upstream areas that have moved since the pin: MoE (shared-expert overlap, new router score, gated delta-net CP), optimizers (cuda-graphed Adam, layerwise distopt fixes, more emerging optimizers, NVFP4 native weights), checkpointing (async DCP/FSDP save, coredump fix, removal of cross-rank sync, save-after-transient-NaN fix), inference (FA4 inference, MTP inference fixes, Mamba EP inference UT), RL (staleness histograms, skip-train fix, reward-stop-token fix, optimizer onload after logprobs), Megatron-FSDP hardening, Hybrid/Mamba renames (MambaModel → HybridModel) and DeepSeek Sparse Attention port to Mamba.
- The renaming of `MambaModel`/`MambaStack` to `HybridModel`/`HybridStack` upstream is a breaking rename; any Primus code referencing those classes will need adaptation when syncing.
- The drift is large but the only Primus merge this week already adapts an upstream-coupled patch with a fingerprint check, i.e. the project has tooling to detect upstream breaks. Recommendation: **plan sync** (scheduled, non-urgent) — the gap is 344 commits and includes breaking renames + inference/optimizer additions, but there is no in-tree blocker forcing an immediate bump this week.

## 3. Weekly PR Update Table

| PR | Merged Time (GMT+8) | Category | Key Update |
| --- | --- | --- | --- |
| [#674](https://github.com/AMD-AGI/Primus/pull/674) | 2026-04-22 16:30 | Bug Fix | `fix(megatron): adapt recompute_layer_patches to the upstream Megatron and add UT` — rewrites the `custom` / `checkpoint_handler` closures to be byte-for-byte identical to upstream Megatron's `TransformerBlock._checkpointed_forward`, keeps the `recompute_layer_ids is None` delegation fast-path, removes the stale `PrimusTransformerBlock` subclass, seeds `_EXPECTED_MEGATRON_CHECKPOINTED_FORWARD_SHA256`, and adds UTs for pipeline-stage offset mapping and the `(fp8 or fp4) and not requires_grad` skip rule. Author: `lhzhang333`. |

## 4. Megatron-LM Drift Overview

- Submodule path: `third_party/Megatron-LM`
- Upstream: `https://github.com/NVIDIA/Megatron-LM.git` (`main`)
- Pinned SHA in Primus `main`: `d3528a21301db2d12e92912b3ec025dc8a2ed4d6` (committed 2026-03-06 UTC)
- Upstream `main` HEAD at report time: `bbc6b4d14bd8f787b803b3382147dac4cecb20ec` (committed 2026-04-22 UTC)
- Commits upstream-ahead of pinned SHA: **344**
- Drift age: ~47 days.
- Drift status: **behind** (significant).
- Recommendation: `plan sync` — schedule a submodule bump in a dedicated PR because upstream contains both breaking renames (`MambaModel` → `HybridModel`) and new capabilities (FA4 inference, cuda-graphed Adam, async DCP/FSDP save) that Primus users will eventually want, but no merged Primus PR this week strictly requires the newer commit.

## 5. Upstream Feature Delta Table

The table below samples notable upstream changes (Megatron-LM) that have landed since the pinned submodule SHA. Full list of 344 commits is reproducible via `git log d3528a2..origin/main` in the upstream clone.

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| MoE | Shared-expert overlap support, including FlexDispatcher | [#2207](https://github.com/NVIDIA/Megatron-LM/pull/2207) (`ebfa13852`) | Potential additional overlap of shared-expert compute with dispatcher comm; relevant to Primus MoE training recipes. |
| MoE | New score function for the MoE router | [#3673](https://github.com/NVIDIA/Megatron-LM/pull/3673) (`eb80b7491`) | New routing knob for experiments in Primus MoE configs. |
| MoE / CP | Gated delta-net context parallel (CP) | [#2642](https://github.com/NVIDIA/Megatron-LM/pull/2642) (`20ba03fec`) | Enables CP for gated delta-net; relevant if Primus exposes delta-net CP variants. |
| MoE | Fix unnecessary permute padding for non-quantized MoE dispatch | [#4038](https://github.com/NVIDIA/Megatron-LM/pull/4038) (`567d4d468`) | Memory/perf improvement on the MoE dispatch path. |
| Quant / FP | Enable FP8 DPA for MXFP8 recipe | [#4066](https://github.com/NVIDIA/Megatron-LM/pull/4066) (`22e0bb5fd`) | Affects MXFP8 code paths if Primus tracks MXFP8 recipes. |
| Quant / FP | NVFP4 native weights for DDP | [#4005](https://github.com/NVIDIA/Megatron-LM/pull/4005) (`e1db4a03d`) | Adds NVFP4 weight storage under DDP; useful for low-precision Primus exploration. |
| Optimizer | CUDA graph for Adam | [#3429](https://github.com/NVIDIA/Megatron-LM/pull/3429) (`3d87bfc1b`) | Optimizer-side speedup path; Primus can opt-in after sync. |
| Optimizer | Emerging-optimizers integration refactor + additional optimizers | [#4113](https://github.com/NVIDIA/Megatron-LM/pull/4113) (`5b512b45d`), [#3907 / #4119](https://github.com/NVIDIA/Megatron-LM/pull/4119) (`c9797adbf`) | Aligns with the `third_party/Emerging-Optimizers` submodule Primus already declares. |
| Optimizer | Fix layerwise distributed optimizer with `expt_dp_size=1` | [#4138](https://github.com/NVIDIA/Megatron-LM/pull/4138) (`51bcf1470`) | Correctness fix for layerwise distopt combos. |
| Inference | FA4 Inference | [#4186](https://github.com/NVIDIA/Megatron-LM/pull/4186) (`76ac7c24b`) | New inference kernel; not a training-path change but worth tracking for eval/serve. |
| Inference | Miscellaneous MTP inference fixes | [#4191](https://github.com/NVIDIA/Megatron-LM/pull/4191) (`980211ae6`) | Inference correctness fixes (MTP). |
| Inference | Port DeepSeek Sparse Attention to `MambaModel` | [#3553](https://github.com/NVIDIA/Megatron-LM/pull/3553) (`a00e9443c`) | New attention variant on the hybrid/Mamba stack. |
| Checkpointing | Remove cross-rank sync during checkpoint load, deprecate legacy `torch.distributed.checkpoint.state_dict_loader.load_state_dict` path | [#2864](https://github.com/NVIDIA/Megatron-LM/pull/2864) (`0602523f7`) | Faster / less-fragile load path; verify Primus checkpoint load sites on sync. |
| Checkpointing | Async DCP and FSDP save support | [#4027](https://github.com/NVIDIA/Megatron-LM/pull/4027) (`69f3b3400`) | Async save for DCP/FSDP checkpoints. |
| Checkpointing | `--async-ckpt-use-cpu-shm` argument | [#4355](https://github.com/NVIDIA/Megatron-LM/pull/4355) (`997896883`) | New CLI knob; Primus launch scripts may need to be aware. |
| Checkpointing | Fix potential coredump when saving a checkpoint | [#1871](https://github.com/NVIDIA/Megatron-LM/pull/1871) (`ded22f428`) | Stability fix on save path. |
| Checkpointing | Do not save after transient NaN / Inf (`RerunStateMachine` crash fix) | [#3981](https://github.com/NVIDIA/Megatron-LM/pull/3981) (`0b5e3ae5f`) | Avoids poisoned checkpoints. |
| RL | Onload optimizer after logprobs computation | [#4235](https://github.com/NVIDIA/Megatron-LM/pull/4235) (`e7789676f`) | RL memory scheduling improvement. |
| RL | Fix RL to work with `--skip-train` | [#4249](https://github.com/NVIDIA/Megatron-LM/pull/4249) (`98a51eb2c`) | Correctness for RL eval-only runs. |
| RL | Fix RL reward due to stop token | [#4096](https://github.com/NVIDIA/Megatron-LM/pull/4096) (`e4d3a4c4f`) | Reward correctness; relevant to any Primus RL recipe downstream. |
| RL | Add tables and histogram for RL staleness | [#4097](https://github.com/NVIDIA/Megatron-LM/pull/4097) (`23663a870`) | New RL observability surface. |
| Hybrid / Mamba | Rename `MambaModel`/`MambaStack` to `HybridModel`/`HybridStack` | [#4099](https://github.com/NVIDIA/Megatron-LM/pull/4099) (`15e07a2dd`) | **Breaking rename**. Any Primus import of `MambaModel`/`MambaStack` will break on sync. |
| Hybrid / Mamba | QK layernorm support for dot-product attention in `MambaModel` | [#4067](https://github.com/NVIDIA/Megatron-LM/pull/4067) (`e15ec3c04`) | New toggle for hybrid models. |
| Hybrid / Mamba | Unit test for Mamba EP inference (eager fallback with mixed CUDA graphs) | [#4085](https://github.com/NVIDIA/Megatron-LM/pull/4085) (`8cf6b355a`) | Expands inference coverage; not a direct dependency change. |
| Megatron-FSDP | Build `expt_device_mesh` only for MoE models | [#3831](https://github.com/NVIDIA/Megatron-LM/pull/3831) (`25129bf3d`) | Init-time correctness for non-MoE FSDP paths. |
| Megatron-FSDP | Log mcore detection only after imports succeed | [#4400](https://github.com/NVIDIA/Megatron-LM/pull/4400) (`e5ec9ab91`) | Cleaner FSDP import diagnostics. |
| Megatron-FSDP | Fix incorrectly set `decoupled_grad` and DistOpt mechanics | [#4133](https://github.com/NVIDIA/Megatron-LM/pull/4133) (`ab43d43f0`) | Correctness fix for MFSDP + distopt combo. |
| Megatron-FSDP | Refactor uneven DTensor to full tensor + UT | [#3190](https://github.com/NVIDIA/Megatron-LM/pull/3190) (`159e34720`) | Internal MFSDP refactor; watch for API changes on sync. |
| Batch schedule | Replace rampup batch size scheduler with custom step batch size schedules (landed, reverted, re-landed) | [#3779](https://github.com/NVIDIA/Megatron-LM/pull/3779) (`c9e03d0c3`), [#4411](https://github.com/NVIDIA/Megatron-LM/pull/4411) (`532ad926b`), revert `a52112dae` | New batch-size schedule API; if Primus relies on rampup-batch CLI/behavior, expect adaptation work. |
| CI / Infra (upstream) | CI retry loops, flaky-test marking, launcher refactors, docs version picker fixes | e.g. [#4387](https://github.com/NVIDIA/Megatron-LM/pull/4387) (`9c210f7ac`), [#4298](https://github.com/NVIDIA/Megatron-LM/pull/4298) (`6636eb073`), [#4367](https://github.com/NVIDIA/Megatron-LM/pull/4367) (`4ece77d90`) | Infra-only; no Primus impact beyond aligning local checkout of the submodule. |
| Dependencies | Bump TransformerEngine to `release_v2.14`, bump DeepEP to `34152ae`, set `megatron-fsdp` to `0.5.0` | [#4331](https://github.com/NVIDIA/Megatron-LM/pull/4331) (`ceac26946`), [#4228](https://github.com/NVIDIA/Megatron-LM/pull/4228) (`123645bb7`), `97f9ab6f2` | When Primus syncs the submodule, its training containers should expose matching TE / DeepEP / megatron-fsdp versions. |

Note: the table is a curated high-signal subset; it is not exhaustive over all 344 upstream commits.

## 6. Source Links

- Primus repository: https://github.com/AMD-AGI/Primus
- Primus `main` HEAD at report time: commit `4990e51` ("fix(megatron): adapt recompute_layer_patches to the upstream Megatron and add UT (#674)").
- Merged PR search used for this report (week window):
  `gh pr list --repo AMD-AGI/Primus --state merged --search "merged:>=2026-04-19T16:00:00Z base:main"`
- Merged PR this week: https://github.com/AMD-AGI/Primus/pull/674
- Megatron-LM upstream: https://github.com/NVIDIA/Megatron-LM
- Pinned Megatron-LM commit: https://github.com/NVIDIA/Megatron-LM/commit/d3528a21301db2d12e92912b3ec025dc8a2ed4d6
- Upstream Megatron-LM HEAD at report time: https://github.com/NVIDIA/Megatron-LM/commit/bbc6b4d14bd8f787b803b3382147dac4cecb20ec
- Upstream compare (pinned → HEAD): https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...bbc6b4d14bd8f787b803b3382147dac4cecb20ec
