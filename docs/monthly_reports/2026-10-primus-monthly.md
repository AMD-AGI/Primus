# Primus Monthly Engineering Report — 2026-10

## Time window

- **Timezone:** Asia/Shanghai (GMT+8)
- **Window start:** 2026-09-01T17:05:00+08:00
- **Window end:** 2026-10-01T17:02:00+08:00
- **Span:** ~30 days
- **Coverage note:** The window is continuity-based. It starts at the `end` of
  the previous published report (the `2026-09` monthly report, which ended
  2026-09-01T17:05:00+08:00) and runs to now. All prior weekly and monthly
  report windows merge into a single contiguous coverage group (every gap
  between consecutive published reports is ≤ 7 days), so there is **no
  unresolved (> 7 day) history gap**. Because `window_start` (2026-09-01) falls
  before the first day of the current calendar month (2026-10-01), this run
  covers September activity that a plain calendar-month window (Oct 1 → now)
  would have silently dropped; `gap_detected = true` for that reason only. The
  span (~30 days) is within the healthy monthly range (< 45 days), so no
  over-sized-window sanity flag is raised.

## Executive summary

- This report covers **2026-09-01 → 2026-10-01** (~30 days). The window extends
  before the October calendar-month boundary because it chains off the previous
  report's end (2026-09-01); it therefore captures essentially all of
  September's merged-PR and pin activity. `gap_detected = true` on the "earlier
  than first-of-month" condition, but there is **no unresolved history gap** —
  the prior weekly/monthly reports form one contiguous coverage group.
- **73 PRs** merged to `main` in the window. The largest categories are
  **CI/Infra (20)**, **Other/feature-enablement (16)**, and **Bug Fix (12)**,
  reflecting a month dominated by test/coverage hardening (the PRPUNDIT test
  series), v26.6/v26.7 release plumbing, new backends/models (SpecForge offline
  + online, MiniMax-M3, DLRM-v4), and FP4/MXFP4 + FlyDSL performance work.
- **Backend pins (backend-gap-tracked): unchanged in-window.**
  `third_party/Megatron-LM` stays at `d3528a21` and `third_party/torchtitan`
  stays at `73a0e697` (upstream `v0.2.2` tag) across the whole window.
  Therefore **no backend-gap report regeneration** is performed this run
  (`backend_gap_updated = false`).
- **Other in-window pin changes (not backend-gap-tracked targets):**
  `third_party/maxtext` advanced `b47d74bf` → `b3c53763` (PR #1172, align JAX
  backends with v26.7), and the **Primus-Turbo CI pin** advanced `6d5ff979` →
  `ae1ac068` (+47 commits, bumped in-window via PR #1195 for MegaMoE FlyDSL
  ownership). maxtext has no backend-gap report set (only Megatron-LM and
  torchtitan do), and Primus-Turbo is covered in the quarterly-drift section
  below.
- **Upstream drift is large and still growing:** Megatron-LM is **1662 commits**
  behind upstream `main` (was 1371 at the 2026-09 report); torchtitan is **1194
  commits** behind upstream `main` since the `v0.2.2` tag (was 918).
  Recommendation for both remains **plan sync**. Primus-Turbo tracked forward
  normally → **monitor**.
- **Primus-Turbo quarter boundary:** Q4 2026 began **2026-10-01**, i.e. only
  hours before this report. The current CI, benchmark, AITER, TRITON, and UCCL
  pins all equal their quarter-start values, so there is **no quarterly drift
  yet** for Q4. The September CI-pin movement is reported as monthly-window
  context, not quarterly drift.

## Monthly PR update table

| PR | Merged Time (GMT+8) | Category | Key Update |
| --- | --- | --- | --- |
| [#1062](https://github.com/AMD-AGI/Primus/pull/1062) | 2026-09-01 17:15 | Other | DeepSeek-V4 packed-sequence (THD) SFT — full model at 4k and 128k on 3 nodes |
| [#1065](https://github.com/AMD-AGI/Primus/pull/1065) | 2026-09-01 17:26 | Refactor | Reuse Primus config merge and isolate the Megatron CLI (data layer) |
| [#1059](https://github.com/AMD-AGI/Primus/pull/1059) | 2026-09-02 03:02 | Other | DLRM-v4 (TorchRec/HSTU) projection workload + first-principles MI350X calibration |
| [#1075](https://github.com/AMD-AGI/Primus/pull/1075) | 2026-09-02 07:56 | CI/Infra | Add JAX v26.6 dockerfile (OOB release) |
| [#1078](https://github.com/AMD-AGI/Primus/pull/1078) | 2026-09-02 13:54 | Turbo/Dependency Version Update | Bump transformers 5.5.0 → 5.10.1 |
| [#1077](https://github.com/AMD-AGI/Primus/pull/1077) | 2026-09-02 13:54 | Turbo/Dependency Version Update | Bump transformers 4.57.6 → 5.10.1 (megatron_bridge pretrain hook) |
| [#1076](https://github.com/AMD-AGI/Primus/pull/1076) | 2026-09-02 13:55 | Turbo/Dependency Version Update | Bump transformers 4.50.0 → 5.10.1 (diffusion pretrain hook) |
| [#1080](https://github.com/AMD-AGI/Primus/pull/1080) | 2026-09-03 07:14 | Performance Optimization | Tune GDN/KDA 1B MI355X pretrain configs for high-occupancy throughput |
| [#1098](https://github.com/AMD-AGI/Primus/pull/1098) | 2026-09-07 07:10 | CI/Infra | Cover Compressor overlap and non-overlap windowing (PRPUNDIT-25) |
| [#1106](https://github.com/AMD-AGI/Primus/pull/1106) | 2026-09-07 07:10 | CI/Infra | Cover fused clamped SwiGLU Triton autograd (PRPUNDIT-23) |
| [#1105](https://github.com/AMD-AGI/Primus/pull/1105) | 2026-09-07 07:12 | CI/Infra | Pin AttentionResidualMixer against an independent mix (PRPUNDIT-22) |
| [#1103](https://github.com/AMD-AGI/Primus/pull/1103) | 2026-09-07 07:14 | CI/Infra | Pin Llama3 TurboAttention GQA layout (PRPUNDIT-20) |
| [#1099](https://github.com/AMD-AGI/Primus/pull/1099) | 2026-09-07 07:15 | CI/Infra | Pin Wan CausalConv3d streaming cache (PRPUNDIT-24) |
| [#1101](https://github.com/AMD-AGI/Primus/pull/1101) | 2026-09-07 07:15 | CI/Infra | Pin DeepSeek-V3 TurboAttention BSHD layout (PRPUNDIT-18) |
| [#1102](https://github.com/AMD-AGI/Primus/pull/1102) | 2026-09-07 07:16 | CI/Infra | Cover GPT-OSS turbo sink attention forward (PRPUNDIT-19) |
| [#1100](https://github.com/AMD-AGI/Primus/pull/1100) | 2026-09-07 07:17 | CI/Infra | Pin AdaLNContinuous scale/shift order and gradients (PRPUNDIT-17) |
| [#1094](https://github.com/AMD-AGI/Primus/pull/1094) | 2026-09-07 07:18 | Docs | Add AINIC bundle version guide (networking docs) |
| [#1079](https://github.com/AMD-AGI/Primus/pull/1079) | 2026-09-07 07:19 | Bug Fix | Clone the Flux validation loss out of Megatron's rescale path |
| [#1112](https://github.com/AMD-AGI/Primus/pull/1112) | 2026-09-07 17:48 | Bug Fix | Parse scientific-notation CLI overrides as floats (e.g. `--lr 1e-5`) |
| [#1104](https://github.com/AMD-AGI/Primus/pull/1104) | 2026-09-07 17:49 | CI/Infra | Pin Qwen3 TurboAttention layout and qk-norm (PRPUNDIT-21) |
| [#1096](https://github.com/AMD-AGI/Primus/pull/1096) | 2026-09-07 19:56 | CI/Infra | Add missing test for AvgDown3D.forward (testintel) |
| [#1114](https://github.com/AMD-AGI/Primus/pull/1114) | 2026-09-08 10:36 | Docs | Bump current release tags v26.5 → v26.6 |
| [#1113](https://github.com/AMD-AGI/Primus/pull/1113) | 2026-09-08 12:46 | Other | Add Llama-3.1-8B BF16 LoRA-SFT recipe for MI355X (Megatron examples) |
| [#1111](https://github.com/AMD-AGI/Primus/pull/1111) | 2026-09-08 12:49 | Bug Fix | Raise FileNotFoundError (not `assert False`) in setup_backend_path (survives `python -O`) |
| [#1097](https://github.com/AMD-AGI/Primus/pull/1097) | 2026-09-08 12:59 | CI/Infra | Add missing test for Decoder3d.forward (testintel) |
| [#1108](https://github.com/AMD-AGI/Primus/pull/1108) | 2026-09-08 15:04 | Bug Fix | Fix TE requantize_data UnboundLocalError on the captured `inp` |
| [#1070](https://github.com/AMD-AGI/Primus/pull/1070) | 2026-09-08 20:18 | Other | GPT-OSS MXFP4 deosc BF16 snapshots and simplified 20B recipe |
| [#1054](https://github.com/AMD-AGI/Primus/pull/1054) | 2026-09-08 20:19 | Bug Fix | Fix XLA_FLAGS precedence inversion; add XLA_FLAGS_APPEND; document flag layering (JAX) |
| [#1118](https://github.com/AMD-AGI/Primus/pull/1118) | 2026-09-09 08:20 | Docs | Corrections to the AINIC bundle networking guide |
| [#1119](https://github.com/AMD-AGI/Primus/pull/1119) | 2026-09-09 08:22 | Other | Add Qwen3-30B-A3B and Qwen2.5-72B MXFP4 pretrain recipes for MI355X |
| [#1120](https://github.com/AMD-AGI/Primus/pull/1120) | 2026-09-09 10:21 | Docs | Update bare-metal install and docs to v26.6 |
| [#1095](https://github.com/AMD-AGI/Primus/pull/1095) | 2026-09-09 11:34 | Other | JAX MaxText v26.6 primus-cli integration |
| [#1124](https://github.com/AMD-AGI/Primus/pull/1124) | 2026-09-09 13:41 | CI/Infra | spur-cluster-status skill: report QoS node caps vs live usage |
| [#1125](https://github.com/AMD-AGI/Primus/pull/1125) | 2026-09-09 13:54 | CI/Infra | spur-cluster-status skill: QoS node caps vs live usage (follow-up) |
| [#1121](https://github.com/AMD-AGI/Primus/pull/1121) | 2026-09-09 14:58 | Docs | Organize technical blogs and remove the deprecated docs tree |
| [#1126](https://github.com/AMD-AGI/Primus/pull/1126) | 2026-09-09 15:03 | Docs | Drop the Instella-MoE blog from the article lists |
| [#999](https://github.com/AMD-AGI/Primus/pull/999) | 2026-09-11 08:25 | Refactor | Retire the examples launchers in favour of primus-cli |
| [#962](https://github.com/AMD-AGI/Primus/pull/962) | 2026-09-11 08:25 | Bug Fix | Work around Triton buffer-store miscompile and ROCm compiled-training NaNs |
| [#1127](https://github.com/AMD-AGI/Primus/pull/1127) | 2026-09-11 11:05 | Performance Optimization | Run the GPT-OSS dense QKVO GEMMs on Turbo FlyDSL |
| [#1133](https://github.com/AMD-AGI/Primus/pull/1133) | 2026-09-12 08:24 | Docs | Update docs for the AINIC predownloaded bundle |
| [#1134](https://github.com/AMD-AGI/Primus/pull/1134) | 2026-09-14 14:15 | Bug Fix | Add support for amd-smi |
| [#1137](https://github.com/AMD-AGI/Primus/pull/1137) | 2026-09-14 14:17 | Docs | Fix the system-validation link |
| [#1139](https://github.com/AMD-AGI/Primus/pull/1139) | 2026-09-15 09:07 | Other | Add SpecForge backend for offline DFlash capture and training |
| [#1128](https://github.com/AMD-AGI/Primus/pull/1128) | 2026-09-15 10:07 | Performance Optimization | SDMA support for Megatron with the distributed optimizer |
| [#1157](https://github.com/AMD-AGI/Primus/pull/1157) | 2026-09-15 14:50 | CI/Infra | Reformat (lint) the SDMA distributed-optimizer files |
| [#1140](https://github.com/AMD-AGI/Primus/pull/1140) | 2026-09-15 14:50 | Other | Update mamba-370m config file for MI325X |
| [#1155](https://github.com/AMD-AGI/Primus/pull/1155) | 2026-09-15 14:54 | Other | Add gfx1250 native LoRA SFT recipes (Megatron examples) |
| [#1163](https://github.com/AMD-AGI/Primus/pull/1163) | 2026-09-16 13:48 | CI/Infra | Skip SpecForge unit tests until the overlay image is in CI |
| [#1143](https://github.com/AMD-AGI/Primus/pull/1143) | 2026-09-16 16:46 | CI/Infra | Add JAX v26.7 dockerfile (OOB release) |
| [#1142](https://github.com/AMD-AGI/Primus/pull/1142) | 2026-09-16 16:50 | CI/Infra | Add Primus v26.7 dockerfile (OOB release) |
| [#1158](https://github.com/AMD-AGI/Primus/pull/1158) | 2026-09-16 17:14 | Bug Fix | Stop multiplying DeepSeek-V4 TFLOPs by hc_mult |
| [#1162](https://github.com/AMD-AGI/Primus/pull/1162) | 2026-09-17 10:04 | Other | Expose per-head max attention logit on the FLA fused backend for qk_clip |
| [#1165](https://github.com/AMD-AGI/Primus/pull/1165) | 2026-09-17 11:20 | Performance Optimization | Add fused grouped GEMM support |
| [#1172](https://github.com/AMD-AGI/Primus/pull/1172) | 2026-09-18 06:16 | CI/Infra | JAX docker release — align the JAX backends with v26.7 (bumps third_party/maxtext) |
| [#1171](https://github.com/AMD-AGI/Primus/pull/1171) | 2026-09-18 06:16 | Docs | Update documentation for v26.7 |
| [#1170](https://github.com/AMD-AGI/Primus/pull/1170) | 2026-09-21 12:57 | Other | Add the release-docs agent |
| [#1173](https://github.com/AMD-AGI/Primus/pull/1173) | 2026-09-21 13:01 | Other | Add perf batch runner and result extractor (tools) |
| [#1178](https://github.com/AMD-AGI/Primus/pull/1178) | 2026-09-21 13:52 | Bug Fix | Add the MI350X GPU env file for gfx950 WarpSpeed |
| [#1168](https://github.com/AMD-AGI/Primus/pull/1168) | 2026-09-21 13:57 | Performance Optimization | Enable RCCL-SDMA parameter gather by default (GPT-OSS) |
| [#1182](https://github.com/AMD-AGI/Primus/pull/1182) | 2026-09-21 19:13 | CI/Infra | Gate GPU unit tests on the `ci:gpu` PR label |
| [#1174](https://github.com/AMD-AGI/Primus/pull/1174) | 2026-09-22 09:30 | Performance Optimization | Enable fused MXFP4 weight-gradient accumulation |
| [#1179](https://github.com/AMD-AGI/Primus/pull/1179) | 2026-09-22 09:31 | Other | Bake the MXFP4 TE env into YAML `env:` blocks (Megatron examples) |
| [#1167](https://github.com/AMD-AGI/Primus/pull/1167) | 2026-09-22 09:33 | Refactor | Rename the Hylo stack back to Zebra-Llama; fix hybrid eval/convert tooling |
| [#1183](https://github.com/AMD-AGI/Primus/pull/1183) | 2026-09-22 16:52 | Bug Fix | Initialize the GPT-OSS router bias (torchtitan) |
| [#1164](https://github.com/AMD-AGI/Primus/pull/1164) | 2026-09-22 23:33 | Other | Add qk_clip (MuonClip) patch for hybrid models and the distributed optimizer |
| [#1156](https://github.com/AMD-AGI/Primus/pull/1156) | 2026-09-23 20:09 | Performance Optimization | Turbo Llama-3.1-8B FP4 dense MLP, residual RMSNorm, and skip-y |
| [#1181](https://github.com/AMD-AGI/Primus/pull/1181) | 2026-09-24 07:59 | Bug Fix | Apply the FLA Triton autotune patch on the primus-cli pretrain path |
| [#1184](https://github.com/AMD-AGI/Primus/pull/1184) | 2026-09-24 14:14 | Performance Optimization | Match MXFP4 de-osc to the forward grid and cut its step cost (Megatron) |
| [#1195](https://github.com/AMD-AGI/Primus/pull/1195) | 2026-09-28 13:11 | Turbo/Dependency Version Update | Bump Primus-Turbo to pick up MegaMoE FlyDSL ownership (#525) |
| [#1196](https://github.com/AMD-AGI/Primus/pull/1196) | 2026-09-28 14:51 | Bug Fix | Honour turbo_mega_moe_precision for MegaMoE experts (DeepSeek-V4) |
| [#1191](https://github.com/AMD-AGI/Primus/pull/1191) | 2026-09-29 00:40 | Other | Add SpecForge backend for online training (Mooncake + SGLang) |
| [#1186](https://github.com/AMD-AGI/Primus/pull/1186) | 2026-09-29 06:53 | Performance Optimization | Skip overwritten grad clears and isolate the gradient reduce-scatter (Megatron) |
| [#1197](https://github.com/AMD-AGI/Primus/pull/1197) | 2026-09-29 10:52 | Other | Add MiniMax-M3 (text tower) model support |

**Category breakdown (73 PRs):** CI/Infra 20 · Other 16 · Bug Fix 12 ·
Performance Optimization 9 · Docs 9 · Turbo/Dependency Version Update 4 ·
Refactor 3.

## Megatron-LM drift overview

- **Drift target:** `third_party/Megatron-LM`
- **Upstream:** `https://github.com/NVIDIA/Megatron-LM.git` (`main`)
- **Pinned SHA in Primus `main`:** `d3528a21301db2d12e92912b3ec025dc8a2ed4d6` (2026-03-06)
- **Pin change in-window:** None — the pin is unchanged across the entire window.
- **Upstream `main` HEAD SHA:** `1b638343ed4b3a3a630103a666e4266ac064cf0b`
- **Upstream ahead of pin by:** **1662 commits** (behind_by = 0; the pin is an
  ancestor of upstream `main`).
- **Source-declared Megatron Core version:** `0.16.0rc0` at the pin
  (`megatron/core/package_info.py`); upstream `main` now declares `0.20.0`
  (was `0.18.0` at the 2026-09 report).
- **Recommendation:** `plan sync` — the gap is very large and continues to grow
  (was ~1371 commits at the 2026-09 report), but no in-window pin change forces
  an urgent action.

### Megatron-LM upstream feature delta table

Notable upstream areas that have moved since the pin (integration-relevant):

| Area | Notable upstream additions/fixes since the pin |
| --- | --- |
| Megatron-FSDP & distributed runtime | - **MFSDP v2**: continued MFSDP v2 hardening — CUDA-graph test optimizer cleanup and default profiler activities (upstream #7729, #7726)<br>- **Mixed precision**: added/fixed MXFP8, uneven-DTensor, and frozen-parameter paths<br>- **Checkpointing**: added DCP and FSDP async-save support; dtype-only fix in `load_tensors_metadata` (#7620)<br>- **Overlap**: refined all-gather / reduce-scatter overlap and precision-aware optimizer behavior |
| Distributed optimizer & checkpoint | - **GTP dedup**: deduplicate GTP-replicated params with the optimizer's own GTP group (#7359)<br>- **State-dict keys**: restrict distributed-optimizer state-dict key types to str/int (#7080)<br>- **GlobalLayout**: added explicit rank segments and per-placement builders (#7670)<br>- **Checkpoint conversion**: fixed torch-dist → FSDP DTensor for hybrid Gated-DeltaProduct checkpoints (#7674) |
| MoE, router & expert parallelism | - **Overlap**: improved shared-expert overlap and FlexDispatcher support<br>- **Router**: added a new router score function<br>- **Precision**: added NVFP4 native weights for DDP<br>- **Backprop**: added A2A-combine backprop overlap with wgrad GEMM |
| Hybrid / Mamba, attention & inference | - **Hybrid models**: added `megatron/core/models/hybrid/` and renamed Mamba stack concepts toward Hybrid naming<br>- **Attention**: added YARN and DeepSeek Sparse Attention paths; selective replay for wide-residual streams (#6805)<br>- **Inference**: added CUDA-graph MTP inference, prefix caching, and removed the multimodal media byte limit (#7702)<br>- **Transformer Engine**: build updated to TE release 2.20 tip (#7683) |
| Packaging & version metadata | - **Core version**: upstream advanced `megatron/core/package_info.py` from `0.16.0rc0` (pin) to `0.20.0`<br>- **Deps**: continued evolution of `pyproject.toml` / `megatron/core/requirements.txt` dependency groups<br>- **CI/docs**: broad workflow/docs surface churn across the gap |

> Assumption: the notable-area descriptions above carry forward the
> fact-checked backend-gap report for the same unchanged pin (`d3528a21`),
> refreshed with upstream commit titles observed on `main` today; only the
> ahead-count, upstream HEAD, and core version are updated to current values.

## TorchTitan drift overview

- **Drift target:** `third_party/torchtitan`
- **Upstream:** `https://github.com/pytorch/torchtitan.git` (`main`)
- **Pinned SHA in Primus `main`:** `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce`
  (the tagged **v0.2.2** release, 2026-02-20)
- **Pin change in-window:** None — the pin is unchanged across the entire window.
- **Upstream `main` HEAD SHA:** `1aaee42bf476c74d746ee842fc23cf09612b3aa9`
- **Upstream ahead of pin by:** **1194 commits** (behind_by = 0; the pin is an
  ancestor of upstream `main`).
- **Version semantics:** `assets/version.txt` = `0.2.2` at the pin; upstream
  `main` is still `0.2.2` (dev toward the next tag).
- **Recommendation:** `plan sync` — the pin sits on a maintained tagged release,
  but upstream `main` continues to advance (was ~918 commits at the 2026-09
  report).

### TorchTitan upstream feature delta table

Notable upstream areas that have moved since the `v0.2.2` pin:

| Area | Notable upstream additions/fixes since the pin |
| --- | --- |
| New / evolving models | - **Kimi K2 / K3**: added `kimi_k2_7` earlier and continued `kimi_k3` work (pipeline-cache page, InvariantRowParallelLinear vision projections, #4950)<br>- **Qwen 3.5**: tie embeddings for the 0.8B/2B/4B flavors (#4916) and OffsetRMSNorm compile support<br>- **DeepSeek-V4**: packed-document support (#4943), compressed-KV placement (#4932), and Attention-Gym CSA/SWA/HCA paths (#4855, #4856, #4858)<br>- **GPT-OSS**: added `torchtitan/models/gpt_oss/` with `spmd_types` enablement |
| Graph trainer & RL stack | - **graph_trainer**: removed legacy AOT export helpers, graph-based EP chunking, and PP stage metadata inference for SPMD (#4979, #4952, #4944)<br>- **RL**: token-in/token-out generator sampling, sticky-session inter-generator router, and RL unit tests in CI (#4976, #4986, #4889)<br>- **local SPMD**: `local_spmd` regions via `spmd.local_map` (#4960)<br>- **TorchFT**: made the TorchFT quorum mode configurable (#4754) |
| Compile & CUDA-graph performance | - **Local compile**: OffsetRMSNorm and gated-RMSNorm/loss local compilation (#4928, #4894)<br>- **CUDA graph**: capture optimizer updates separately, reuse graph-owned grads, capture grad accumulation in one graph (#4660, #4659, #4658)<br>- **Compile cleanup**: removed TransformerBlock compilation (#4895)<br>- **DeepEP**: free dispatch handles replayed by FullAC early-stopped recompute (#4918) |
| Config, deps & integration coupling | - **Config loading**: replaced the Tyro CLI with Python config loading (#4908); reorganized training recipes (#4913)<br>- **Dependencies**: removed the torchcomms dependency (#4941)<br>- **Naming**: renamed `ParallelDims` to `ParallelismContext` (#4905)<br>- **Primus coupling**: `primus/backends/torchtitan/` supplies the adapter/trainer/patches and wraps upstream `torchtitan.train.Trainer`; Primus initializes the GPT-OSS router bias (#1183) and continues config-drift guards |

## Primus-Turbo quarterly drift overview

- **Drift type:** current version vs quarter-start version on Primus `main`.
- **Quarter start (Q4 2026):** 2026-10-01T00:00:00+08:00 (anchor commit
  `bc8fd627`, the latest `main` commit at/ before the quarter boundary). The
  quarter began only hours before this report.
- **CI pin (`.github/workflows/ci.yaml` `PRIMUS_TURBO_COMMIT`):**
  quarter-start `ae1ac068` → current `ae1ac068` (**unchanged — 0 commits**).
- **Benchmark pin (`.github/workflows/benchmark.yaml` `PRIMUS_TURBO_COMMIT`):**
  quarter-start `a04a233c` → current `a04a233c` (**unchanged — 0 commits**).
- **Companion pins (all unchanged since quarter start):**
  - AITER: `0f3c58e6` (AITER **v0.1.14.post1** tag commit)
  - TRITON: `09500db9`
  - UCCL: `5afb4117`
- **No Q4 quarterly drift yet** — the quarter is ~0 days old at report time, so
  the current pins equal their quarter-start values.
- **Monthly-window context (not Q4 drift):** during September the CI pin
  advanced `6d5ff979` → `ae1ac068` (**+47 commits**, bumped in-window via PR
  #1195 for MegaMoE FlyDSL ownership of its kernels, upstream #525). This is
  reported here only to document in-window Primus-Turbo movement; it is folded
  into the previous (Q3) quarter and the current month, not into Q4 drift.
- **Potential impact to Primus:** the September CI-pin movement is mostly
  additive FlyDSL MXFP4/MXFP8 GEMM, grouped-GEMM, and attention kernel work
  plus correctness fixes (grouped-GEMM racing conditions, MegaMoE kernel
  ownership); low-risk to consume.
- **Recommendation:** `monitor` — Primus is tracking Primus-Turbo forward on a
  normal cadence; there is no quarterly drift to act on yet for Q4.

### Primus-Turbo quarterly drift table

No `third_party/Primus-Turbo` quarterly drift in this comparison window: Q4 2026
began on 2026-10-01, only hours before this report, and every tracked pin equals
its quarter-start value. The following areas moved in the **September monthly
window** (CI pin `6d5ff979` → `ae1ac068`, +47 commits) and are listed for
context; they belong to the Q3 quarter, not Q4:

| Area | Notable changes in the September monthly window (context only) |
| --- | --- |
| FlyDSL GEMM / grouped-GEMM | - **MXFP4**: dense GEMM optimizations and `grouped_gemm_fp4` with refactored kernel structure (#514, #483, #487)<br>- **MXFP8**: dense GEMM / dual-cast quant speedups on gfx950 and grouped-MLP MXFP8 support (#505, #503)<br>- **BF16**: added BF16 FlyDSL grouped-GEMM kernels and gfx1250 BF16 grouped GEMM (#486, #508)<br>- **Correctness**: fixed grouped-GEMM LDS racing (K_ITERS==2), work-steal numerics, and wgrad dispatch/accumulation (#522, #527, #521) |
| MoE / MegaMoE kernels | - **Ownership**: let MegaMoE own its FlyDSL kernels instead of sharing them (#525, pulled into Primus via #1195)<br>- **Pad-aware MLP**: pad-aware fused grouped MLP (padN+padK) for gpt-oss-20b MoE (#488)<br>- **Grouped GEMM GLU**: more activation types and tuned FP8 grouped-GEMM opts for padded MoE (#496, #490)<br>- **Grad accum**: optional overwrite of fused expert wgrad output (#534) |
| Attention & RoPE | - **Gluon**: added a Gluon forward attention backend (#469)<br>- **gfx950**: faster dual-wave forward attention and backward at head-dim 64/128; online forward softmax (#480, #535)<br>- **gfx1250**: added a Triton dense attention backend (#481)<br>- **QKV RoPE**: standalone QKV RoPE fwd+bwd for Llama-3.1-8B (#519) |
| Quant, fusions & build | - **MXFP4 UoS**: FlyDSL MXFP4 UoS modes plus `mlp_fp4` fusion and `rmsnorm_residual_fp4` (#494, #509)<br>- **Per-tensor fusion**: per-tensor quant fusion in the MLP part and quant-only optimization stack (#493, #475)<br>- **Activations**: clamped-limit support for GELU in grouped-GEMM GLU (#506)<br>- **CI/compat**: carry the opaque metaclass torch 2.11 requires; clean stale submodule checkouts (#501, #513) |

## Source links

- Merged-PR query (GMT+8 window 2026-09-01 17:05 → 2026-10-01 17:02):
  `gh pr list --repo AMD-AGI/Primus --state merged --base main --search "merged:2026-09-01T09:05:00Z..2026-10-01T09:02:00Z"`
- Megatron-LM upstream: <https://github.com/NVIDIA/Megatron-LM/tree/main>
- Megatron-LM compare (pin → upstream main): <https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...main>
- TorchTitan upstream: <https://github.com/pytorch/torchtitan/tree/main>
- TorchTitan compare (pin → upstream main): <https://github.com/pytorch/torchtitan/compare/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce...main>
- Primus-Turbo: <https://github.com/AMD-AGI/Primus-Turbo>
- Primus-Turbo compare (September CI-pin movement): <https://github.com/AMD-AGI/Primus-Turbo/compare/6d5ff979eb019fbbcd91790ac812024cca05a882...ae1ac068e8ac08b98c7467f27b3cbd53ebe292af>
- Primus `main` submodule pins: <https://github.com/AMD-AGI/Primus/tree/main/third_party>
- Primus CI turbo pins: <https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml>
