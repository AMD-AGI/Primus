# Primus Weekly Engineering Report — 2026-W17

## 1. Time window

- **ISO week:** 2026-W17
- **Repository:** [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus)
- **Base branch:** `main`
- **Window start:** 2026-04-20 00:00 (Asia/Shanghai, GMT+8)
- **Window end:** 2026-04-22 18:26 (Asia/Shanghai, GMT+8, report generation time)

All PR merge times in this report are normalized to GMT+8.

## 2. Executive summary

- **Total merged PRs on `main` this window:** 1
- **Category breakdown:** Bug Fix × 1. No Performance, CI/Infra, Refactor, Docs, or Other PRs merged in the window so far.
- **Turbo / dependency / version bumps this week:** None detected (no `requirements*.txt`, pyproject, or Turbo version bumps among merged PRs this window).
- **Megatron-LM submodule drift status:** significant — Primus `main` still pins the submodule to a commit from **2026-03-06** while upstream `main` has advanced **344 commits** (latest upstream commit 2026-04-22).
- **Notable upstream capabilities already shipped but not yet in Primus:** Mamba → Hybrid model rename, NVFP4 native weights for DDP, FA4 inference, DeepSeek sparse attention port to MambaModel, MoE shared expert overlap improvements, context-parallel gated delta net, rampup-batch-size redesign, TransformerEngine v2.14 bump, DeepEP 34152ae bump, async checkpoint CPU-SHM support.
- **Legacy code removals in upstream to watch:** legacy BERT, T5, biencoder/realm, and vision code paths have been removed in upstream; any Primus patch still referencing those modules will need adjustment.
- **Recommendation:** **plan sync** — upstream drift (≈ 7 weeks, 344 commits, including API-level refactors such as Mamba→Hybrid rename and MoE router changes) is now large enough to risk non-trivial conflicts against Primus patches in `primus/backends/megatron/`. A dedicated submodule-bump PR is recommended before drift grows further.
- **This week's only merged change** (PR #674) hardens the `recompute_layer_patches` Megatron wrapper and adds the SHA256 fingerprint guard, which will make the next Megatron sync safer to execute.

## 3. Weekly PR update table

| PR | Merged Time (GMT+8) | Category | Key Update |
|----|---------------------|----------|------------|
| [#674](https://github.com/AMD-AGI/Primus/pull/674) | 2026-04-22 16:30 | Bug Fix | `fix(megatron): adapt recompute_layer_patches to the upstream Megatron and add UT` — by `lhzhang333`. Rewrites the `TransformerBlock._checkpointed_forward` wrapper so its `custom` / `checkpoint_handler` closures are byte-for-byte identical to upstream Megatron, keeps the `recompute_layer_ids is None` fast-path as a pure delegate, seeds `_EXPECTED_MEGATRON_CHECKPOINTED_FORWARD_SHA256` so a fingerprint guard fires on future upstream edits, adds tests for pipeline-stage offset mapping and the fp8/fp4 no-grad skip rule, and removes the stale `PrimusTransformerBlock` subclass. |

If more PRs merge into `main` after this report is generated but still inside window 2026-W17, they will be captured in the next weekly run.

## 4. Megatron-LM drift overview

- **Submodule path:** `third_party/Megatron-LM`
- **Upstream:** [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (`main`)
- **Pinned SHA in Primus `main`:** [`d3528a21`](https://github.com/NVIDIA/Megatron-LM/commit/d3528a21301db2d12e92912b3ec025dc8a2ed4d6) — `fix(moe): fix TE general_gemm API change (#3582)` (2026-03-06 14:21 UTC)
- **Upstream `main` HEAD at report time:** [`bbc6b4d1`](https://github.com/NVIDIA/Megatron-LM/commit/bbc6b4d14bd8f787b803b3382147dac4cecb20ec) — `chore: rotate oncall schedule` (2026-04-22 09:34 UTC)
- **Commits upstream is ahead:** **344**
- **Approximate calendar gap:** about 7 weeks

**Notable themes in the upstream delta (factual, based on commit subjects):**

- **MoE feature additions:** shared-expert overlap for FlexDispatcher, new router score function, activation / tokens-per-expert logging, shared-expert stream reduction, gated delta net context parallelism.
- **Attention / model surface changes:** FA4 inference path, DeepSeek Sparse Attention ported to `MambaModel`, QK layernorm for dot-product attention in `MambaModel`, MLA V-padding fix when Q/V head dims differ for THD, removal of unused `packed_attention_mask` parameter.
- **Distributed / optimizer plumbing:** NVFP4 native weights for DDP, layer-wise param allgather buffer reuse, `param_index_map` unpacked numel offsets, layerwise distributed optimizer argument cleanup.
- **Checkpoint / reliability:** `--async-ckpt-use-cpu-shm` flag, coredump fix on checkpoint save, checkpoint-save timing refactor.
- **RL training:** onload optimizer after logprobs compute, RL staleness tables / histograms, fix for `--skip-train`, stop-token reward fix.
- **Large renames / refactors:** `MambaModel` / `MambaStack` renamed to `HybridModel` / `HybridStack`; legacy BERT, T5, biencoder/realm, and vision code removed; `CLAUDE.md` → `AGENTS.md` rename and skills reorganization.
- **Dependency / build bumps:** TransformerEngine bumped to `release_v2.14`, DeepEP bumped to `34152ae`, `megatron-fsdp` set to `0.5.0`.
- **CI / infra:** configurable launcher (ft_launcher / torchrun) for functional tests, GB200 1-node variants, apt-get retry, uv retry, action UX improvements, strict review mode.

## 5. Upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
|------|------------------------|----------------------|----------------------------|
| Model rename | `MambaModel` / `MambaStack` renamed to `HybridModel` / `HybridStack` | [#4099](https://github.com/NVIDIA/Megatron-LM/pull/4099) / `15e07a2d` | Any Primus code importing `MambaModel` / `MambaStack` (configs, patches, examples) will need to be updated on the next submodule sync. |
| Legacy removal | Legacy BERT, T5, biencoder/realm, vision modules removed | [#4204](https://github.com/NVIDIA/Megatron-LM/pull/4204) / [#4203](https://github.com/NVIDIA/Megatron-LM/pull/4203) / [#4205](https://github.com/NVIDIA/Megatron-LM/pull/4205) / [#4202](https://github.com/NVIDIA/Megatron-LM/pull/4202) | If Primus still exposes those legacy model code paths, they will break on next bump. Need audit of `primus/backends/megatron/` and example YAMLs. |
| MoE | Shared-expert overlap for FlexDispatcher; new router score function; shared-expert stream reduction | [#2207](https://github.com/NVIDIA/Megatron-LM/pull/2207), [#3673](https://github.com/NVIDIA/Megatron-LM/pull/3673), [#3752](https://github.com/NVIDIA/Megatron-LM/pull/3752) | Potential throughput uplift for Primus MoE training (DeepSeek / Mixtral-class configs). Verify that Primus MoE patches do not conflict with the new router signature. |
| Attention | FA4 inference; DeepSeek Sparse Attention on `MambaModel`; MLA V-padding fix | [#4186](https://github.com/NVIDIA/Megatron-LM/pull/4186), [#3553](https://github.com/NVIDIA/Megatron-LM/pull/3553), [#3003](https://github.com/NVIDIA/Megatron-LM/pull/3003) | Unlocks newer attention kernels; Primus attention patches should be re-validated. |
| DDP / Optimizer | NVFP4 native weights for DDP; reuse of grad buffer for layer-wise param allgather | [#4005](https://github.com/NVIDIA/Megatron-LM/pull/4005), [#3751](https://github.com/NVIDIA/Megatron-LM/pull/3751) | Potential memory / throughput win; check interaction with Primus FP8/FP4 patches. |
| Checkpoint | `--async-ckpt-use-cpu-shm` flag; coredump fix on checkpoint save | [#4355](https://github.com/NVIDIA/Megatron-LM/pull/4355), [#1871](https://github.com/NVIDIA/Megatron-LM/pull/1871) | Directly relevant to Primus large-scale pretrain stability and resume speed. |
| Batch scheduling | Replace rampup batch size scheduler with custom step batch size schedules | [#3779](https://github.com/NVIDIA/Megatron-LM/pull/3779) (+ [#4411](https://github.com/NVIDIA/Megatron-LM/pull/4411) re-land) | Training config / CLI surface change; Primus pretrain configs that use rampup BS need review. |
| Dependencies | TransformerEngine bumped to `release_v2.14`; DeepEP bumped to `34152ae`; `megatron-fsdp` 0.5.0 | [#4331](https://github.com/NVIDIA/Megatron-LM/pull/4331), [#4228](https://github.com/NVIDIA/Megatron-LM/pull/4228), `97f9ab6f` | Primus Docker / requirements files will need matching bumps at sync time. |
| CI / Infra | Configurable launcher (ft_launcher / torchrun) for functional tests; GB200 1-node variants; apt-get / uv retry | [#4298](https://github.com/NVIDIA/Megatron-LM/pull/4298), [#4334](https://github.com/NVIDIA/Megatron-LM/pull/4334), [#4209](https://github.com/NVIDIA/Megatron-LM/pull/4209), [#4387](https://github.com/NVIDIA/Megatron-LM/pull/4387) | Lower-risk; borrow patterns if helpful in Primus CI. |

> Assumption: upstream commit subjects are treated as authoritative for categorization. No line-by-line diff audit against Primus patches was performed in this weekly run; a full audit should be part of the sync PR.

## 6. Source links

- Primus merged-PR query (this week): https://github.com/AMD-AGI/Primus/pulls?q=is%3Apr+is%3Amerged+base%3Amain+merged%3A%3E%3D2026-04-19T16%3A00%3A00Z
- Primus PR #674: https://github.com/AMD-AGI/Primus/pull/674
- Primus submodule pin (tree view): https://github.com/AMD-AGI/Primus/tree/main/third_party/Megatron-LM
- Megatron-LM pinned commit (`d3528a21`): https://github.com/NVIDIA/Megatron-LM/commit/d3528a21301db2d12e92912b3ec025dc8a2ed4d6
- Megatron-LM upstream HEAD (`bbc6b4d1`): https://github.com/NVIDIA/Megatron-LM/commit/bbc6b4d14bd8f787b803b3382147dac4cecb20ec
- Megatron-LM diff (pinned → upstream main): https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...main
