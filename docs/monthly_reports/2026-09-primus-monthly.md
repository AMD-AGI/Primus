# Primus Monthly Engineering Report — 2026-09

## Time window

- **Timezone:** Asia/Shanghai (GMT+8)
- **Window start:** 2026-08-03T17:03:00+08:00
- **Window end:** 2026-09-01T17:05:00+08:00
- **Span:** ~29 days
- **Coverage note:** The window is continuity-based. It starts at the `end` of
  the previous published report (the `2026-08` monthly report, which ended
  2026-08-03T17:03:00+08:00) and runs to now. Because `window_start` (2026-08-03)
  falls before the first day of the current calendar month (2026-09-01), this
  run covers August activity that a plain calendar-month window (Sep 1 → now)
  would have silently dropped. `gap_detected = true` for this reason. No
  unresolved (> 7 day) coverage gap exists between prior published reports — the
  published history is contiguous — and the span (~29 days) is within the healthy
  monthly range (< 45 days), so no over-sized-window sanity flag is raised.

## Executive summary

- This report covers **2026-08-03 → 2026-09-01** (~29 days). The window extends
  before the September calendar-month boundary because it chains off the previous
  report's end (2026-08-03); it therefore captures essentially all of August's
  merged-PR and pin activity. `gap_detected = true` on the "earlier than
  first-of-month" condition, but there is **no unresolved history gap** — the
  prior weekly/monthly reports form one contiguous coverage group.
- **79 PRs** merged to `main` in the window. The largest categories are
  **Other/feature-enablement (23)**, **Bug Fix (18)**, and **CI/Infra (11)**,
  reflecting a month dominated by model/backend enablement (Kimi K3, NeMo
  AutoModel/Wan 2.2, DeepSeek-V4 CP + gfx942, FLUX FP8, gfx1250 multi-GPU) and
  hardening (Turbo, torchtitan, KDA, pipeline, CLI, packaging).
- **Backend pins (backend-gap-tracked): unchanged in-window.**
  `third_party/Megatron-LM` stays at `d3528a21` and `third_party/torchtitan`
  stays at `73a0e697` (upstream `v0.2.2` tag) across the whole window.
  Therefore **no backend-gap report regeneration** is performed this run
  (`backend_gap_updated = false`).
- **Other in-window pin changes (not backend-gap-tracked targets):**
  `third_party/maxtext` advanced `a7c6c7e5` → `b47d74bf` (PR #1021, quantized
  MoE fix), and the **Primus-Turbo CI pin** advanced `9b5d3092`-generation →
  `6d5ff979` via several bumps (#933, #1028, #1048). maxtext has no backend-gap
  report set (only Megatron-LM and torchtitan do), and Primus-Turbo is covered
  in the quarterly-drift section below.
- **Upstream drift is large and unchanged in trend:** Megatron-LM is **1371
  commits** behind upstream `main`; torchtitan is **918 commits** behind upstream
  `main` since the `v0.2.2` tag. Recommendation for both remains **plan sync**.
  Primus-Turbo tracked forward normally this quarter → **monitor**.

## Monthly PR update table

| PR | Merged Time (GMT+8) | Category | Key Update |
| --- | --- | --- | --- |
| [#946](https://github.com/AMD-AGI/Primus/pull/946) | 2026-08-04 09:53 | Other | Remove internal cluster node names from the repo |
| [#926](https://github.com/AMD-AGI/Primus/pull/926) | 2026-08-05 14:30 | Bug Fix | Restore Megatron 'compute per GPU' TFLOP label; robust log parsers |
| [#918](https://github.com/AMD-AGI/Primus/pull/918) | 2026-08-05 14:31 | Docs | Add bare-metal JAX/MaxText install guide and scripts |
| [#949](https://github.com/AMD-AGI/Primus/pull/949) | 2026-08-05 17:13 | Bug Fix | Correct DeepSeek-V4-Flash attention/MoE math; reproducible 4N recipe |
| [#953](https://github.com/AMD-AGI/Primus/pull/953) | 2026-08-06 17:11 | Docs | Refresh README feature/model/news sections for 2026 |
| [#933](https://github.com/AMD-AGI/Primus/pull/933) | 2026-08-07 15:02 | Turbo/Dependency Version Update | Bump PRIMUS_TURBO_COMMIT to 69deeab7 to fix loss NaN |
| [#923](https://github.com/AMD-AGI/Primus/pull/923) | 2026-08-07 18:31 | Other | Add ragged FP8 MoE path + pinned runtime image for GPT-OSS 20B MLPerf |
| [#939](https://github.com/AMD-AGI/Primus/pull/939) | 2026-08-10 15:35 | Bug Fix | Gate Megatron device_id process-group init (AIMA-227) |
| [#835](https://github.com/AMD-AGI/Primus/pull/835) | 2026-08-10 16:05 | Other | Add NeMo AutoModel (Wan 2.2 diffusion) as first-class backend |
| [#961](https://github.com/AMD-AGI/Primus/pull/961) | 2026-08-10 16:29 | CI/Infra | Tier trainer E2E suites; add example-config smoke test |
| [#952](https://github.com/AMD-AGI/Primus/pull/952) | 2026-08-10 16:31 | Bug Fix | Migrate torchtitan configs to v0.2.2 debug section; catch next rename |
| [#966](https://github.com/AMD-AGI/Primus/pull/966) | 2026-08-11 09:32 | Other | Default RCCL_AINIC_ROCE=1 to enable built-in ANP |
| [#951](https://github.com/AMD-AGI/Primus/pull/951) | 2026-08-11 10:37 | CI/Infra | Update bare-metal install to v26.5 |
| [#967](https://github.com/AMD-AGI/Primus/pull/967) | 2026-08-11 12:36 | Other | MLLOG support for Llama2-70B LoRA SFT on MI355X |
| [#969](https://github.com/AMD-AGI/Primus/pull/969) | 2026-08-11 14:21 | Refactor | Remove last references to deleted primus.modules package |
| [#959](https://github.com/AMD-AGI/Primus/pull/959) | 2026-08-11 16:55 | Other | Allow ragged Turbo grouped GEMM across precision modes |
| [#935](https://github.com/AMD-AGI/Primus/pull/935) | 2026-08-11 18:56 | Performance Optimization | Fix DSv3-16B v0.2.2 perf regression on MI355X |
| [#956](https://github.com/AMD-AGI/Primus/pull/956) | 2026-08-12 07:59 | Turbo/Dependency Version Update | Include flash-linear-attention==0.5.2 for hybrid models |
| [#974](https://github.com/AMD-AGI/Primus/pull/974) | 2026-08-12 08:01 | Refactor | Migrate Llama2 MLPerf SFT logging to primus_mllog |
| [#965](https://github.com/AMD-AGI/Primus/pull/965) | 2026-08-12 10:22 | Bug Fix | Support MTP layers under recompute_layer_ids (Megatron) |
| [#947](https://github.com/AMD-AGI/Primus/pull/947) | 2026-08-12 10:39 | Other | Add Kimi K3 text-backbone pretraining support |
| [#957](https://github.com/AMD-AGI/Primus/pull/957) | 2026-08-12 14:02 | Docs | Doc update for v26.5 |
| [#975](https://github.com/AMD-AGI/Primus/pull/975) | 2026-08-12 17:35 | Other | Revert #947 primus_turbo.py grouped-linear changes |
| [#973](https://github.com/AMD-AGI/Primus/pull/973) | 2026-08-13 08:36 | Refactor | Integrate primus mllog package; remove mlperf logging from tree |
| [#968](https://github.com/AMD-AGI/Primus/pull/968) | 2026-08-13 08:39 | Other | Registry-based GEMM backend selection + projection enhancements |
| [#977](https://github.com/AMD-AGI/Primus/pull/977) | 2026-08-13 11:07 | Other | FLUX FP8 baseline implementation for MLPerf |
| [#909](https://github.com/AMD-AGI/Primus/pull/909) | 2026-08-13 16:52 | Other | Enable native-TE MXFP4 training for Flux on MI355X |
| [#984](https://github.com/AMD-AGI/Primus/pull/984) | 2026-08-14 07:07 | Bug Fix | Stop Mamba ROCm patch from aborting non-Mamba runs |
| [#982](https://github.com/AMD-AGI/Primus/pull/982) | 2026-08-14 13:35 | Other | Stabilize MXFP4 convergence with weight de-oscillation (Megatron) |
| [#964](https://github.com/AMD-AGI/Primus/pull/964) | 2026-08-14 18:44 | CI/Infra | Give each E2E test its own TorchInductor cache |
| [#989](https://github.com/AMD-AGI/Primus/pull/989) | 2026-08-17 10:31 | CI/Infra | Bump Primus base image to v26.5 |
| [#991](https://github.com/AMD-AGI/Primus/pull/991) | 2026-08-17 17:14 | Refactor | Drop redundant early Megatron sys.path insert |
| [#983](https://github.com/AMD-AGI/Primus/pull/983) | 2026-08-19 08:20 | Other | MXFP8 expert precision for fused MegaMoE layer (Megatron) |
| [#995](https://github.com/AMD-AGI/Primus/pull/995) | 2026-08-19 11:21 | Other | Add Turbo grad-accum fusion support (Megatron) |
| [#996](https://github.com/AMD-AGI/Primus/pull/996) | 2026-08-19 13:28 | Bug Fix | Support evolving TE fused-attention API (MLPerf) |
| [#997](https://github.com/AMD-AGI/Primus/pull/997) | 2026-08-20 09:58 | Other | Add Primus-Turbo backend option; remove Turbo global TP guard |
| [#954](https://github.com/AMD-AGI/Primus/pull/954) | 2026-08-21 07:31 | Other | Add native Hugging Face to Megatron converters |
| [#998](https://github.com/AMD-AGI/Primus/pull/998) | 2026-08-21 09:24 | Other | Enable FlyDSL Turbo attention for GPT-OSS |
| [#1004](https://github.com/AMD-AGI/Primus/pull/1004) | 2026-08-21 09:46 | Other | Enable DLRM-v4 (TorchRec/HSTU) via workload registry |
| [#1005](https://github.com/AMD-AGI/Primus/pull/1005) | 2026-08-21 12:46 | Performance Optimization | Tune hybrid model batch size to maximize GPU memory (Megatron) |
| [#1000](https://github.com/AMD-AGI/Primus/pull/1000) | 2026-08-21 16:28 | CI/Infra | Shallow-clone build-time git deps to cut docker build time |
| [#976](https://github.com/AMD-AGI/Primus/pull/976) | 2026-08-21 16:29 | Docs | Describe training launches in terms of primus-cli |
| [#942](https://github.com/AMD-AGI/Primus/pull/942) | 2026-08-21 18:04 | CI/Infra | Restore torch E2E coverage in unit-vs-E2E summary |
| [#925](https://github.com/AMD-AGI/Primus/pull/925) | 2026-08-22 07:38 | Refactor | MaxText & MaxDiffusion refactor |
| [#1012](https://github.com/AMD-AGI/Primus/pull/1012) | 2026-08-22 07:39 | Other | Add MLPerf Llama 3.1 8B Docker launcher; update MI355X configs |
| [#1014](https://github.com/AMD-AGI/Primus/pull/1014) | 2026-08-22 15:51 | CI/Infra | Add Primus v26.5.1 dockerfile (OOB release) |
| [#1019](https://github.com/AMD-AGI/Primus/pull/1019) | 2026-08-24 09:43 | Performance Optimization | Drop pure_nnx fp8 workaround; tune gemma4_26B-fp8 batch (maxtext) |
| [#1020](https://github.com/AMD-AGI/Primus/pull/1020) | 2026-08-24 19:58 | Performance Optimization | v26.6 MI300X batch-size tuning (maxtext) |
| [#1021](https://github.com/AMD-AGI/Primus/pull/1021) | 2026-08-25 09:12 | Turbo/Dependency Version Update | Update third_party/maxtext submodule to include quantized MoE fix |
| [#1022](https://github.com/AMD-AGI/Primus/pull/1022) | 2026-08-25 11:22 | Other | Disable use_turbo_grouped_mm on MI325X (torchtitan) |
| [#1029](https://github.com/AMD-AGI/Primus/pull/1029) | 2026-08-25 11:46 | Other | Revert disable of use_turbo_grouped_mm on MI325X |
| [#1030](https://github.com/AMD-AGI/Primus/pull/1030) | 2026-08-25 13:22 | Bug Fix | Unblock Qwen3-32B posttrain on transformers 5.x (megatron_bridge) |
| [#1031](https://github.com/AMD-AGI/Primus/pull/1031) | 2026-08-26 10:38 | Bug Fix | Fix Primus-Turbo fp4 autocast (Megatron) |
| [#1032](https://github.com/AMD-AGI/Primus/pull/1032) | 2026-08-26 10:49 | Bug Fix | Clamp pipeline warmup for short batches |
| [#1028](https://github.com/AMD-AGI/Primus/pull/1028) | 2026-08-26 10:52 | Turbo/Dependency Version Update | Bump Primus-Turbo for flydsl compile cache fixes |
| [#1011](https://github.com/AMD-AGI/Primus/pull/1011) | 2026-08-27 09:24 | Performance Optimization | Qwen3-30B MI355X tuned defaults (recompute/batch) |
| [#1045](https://github.com/AMD-AGI/Primus/pull/1045) | 2026-08-27 09:25 | Performance Optimization | Update MI355X Llama 3.1 8B & GPT-OSS 20B pretrain YAML defaults |
| [#1007](https://github.com/AMD-AGI/Primus/pull/1007) | 2026-08-27 09:26 | Refactor | Rename hybrid configs (zebra_ for hybrids, FLA names) |
| [#1015](https://github.com/AMD-AGI/Primus/pull/1015) | 2026-08-27 15:25 | Turbo/Dependency Version Update | Bump transformers from 4.57.3 to 5.5.0 |
| [#899](https://github.com/AMD-AGI/Primus/pull/899) | 2026-08-27 15:27 | CI/Infra | Add unit/integration tests for ODC (PR #864) |
| [#1046](https://github.com/AMD-AGI/Primus/pull/1046) | 2026-08-27 17:20 | Bug Fix | Accumulate Turbo weight gradient on every microbatch |
| [#1049](https://github.com/AMD-AGI/Primus/pull/1049) | 2026-08-28 11:37 | Performance Optimization | Avoid full logits copy in fused cross-entropy (Megatron) |
| [#1050](https://github.com/AMD-AGI/Primus/pull/1050) | 2026-08-28 11:38 | Bug Fix | Pad fused KDA in_proj past hipBLASLt bf16 dead zone |
| [#1051](https://github.com/AMD-AGI/Primus/pull/1051) | 2026-08-28 11:38 | Performance Optimization | Add opt-in fused SwiGLU+fc2 via FLA swiglu_linear |
| [#1048](https://github.com/AMD-AGI/Primus/pull/1048) | 2026-08-28 14:16 | Turbo/Dependency Version Update | Bump Primus-Turbo to pick up DeepEP cheap-fence fix |
| [#1058](https://github.com/AMD-AGI/Primus/pull/1058) | 2026-08-31 08:14 | CI/Infra | Cover sequence-parallel TP linear forward (PRPUNDIT-16) |
| [#1057](https://github.com/AMD-AGI/Primus/pull/1057) | 2026-08-31 08:15 | Bug Fix | Keep fused_softcap finite on saturation-range logits (PRPUNDIT-5) |
| [#1056](https://github.com/AMD-AGI/Primus/pull/1056) | 2026-08-31 08:15 | CI/Infra | Pin FLAFlashAttention.forward layout contract (PRPUNDIT-4) |
| [#1044](https://github.com/AMD-AGI/Primus/pull/1044) | 2026-08-31 08:17 | Bug Fix | Bound topk index in V4 DSA forward kernel |
| [#1041](https://github.com/AMD-AGI/Primus/pull/1041) | 2026-08-31 13:35 | Bug Fix | Rename Turbo grouped GEMM config (torchtitan) |
| [#1061](https://github.com/AMD-AGI/Primus/pull/1061) | 2026-08-31 15:20 | Bug Fix | Keep VCS MLPerf pins out of default wheel deps |
| [#1047](https://github.com/AMD-AGI/Primus/pull/1047) | 2026-08-31 20:33 | Other | DeepSeek-V4 CP adaptation + gfx942 (MI308X) support; 128k CP SFT |
| [#1052](https://github.com/AMD-AGI/Primus/pull/1052) | 2026-09-01 07:41 | Docs | Update JAX MaxText training recipe for 26.6 |
| [#1063](https://github.com/AMD-AGI/Primus/pull/1063) | 2026-09-01 08:13 | Bug Fix | Do not fall back to an IP for *_SOCKET_IFNAME |
| [#1043](https://github.com/AMD-AGI/Primus/pull/1043) | 2026-09-01 08:38 | Other | gfx1250 multi-GPU enablement |
| [#1060](https://github.com/AMD-AGI/Primus/pull/1060) | 2026-09-01 08:46 | CI/Infra | Add Primus v26.6 dockerfile (OOB release) |
| [#992](https://github.com/AMD-AGI/Primus/pull/992) | 2026-09-01 12:43 | Performance Optimization | Fuse and enable DeepSeek-V4 indexer distillation loss |
| [#1068](https://github.com/AMD-AGI/Primus/pull/1068) | 2026-09-01 15:28 | Refactor | Remove unused train_launcher CLI entry |
| [#1002](https://github.com/AMD-AGI/Primus/pull/1002) | 2026-09-01 15:51 | Bug Fix | Honour explicit CLI flags whose value equals parser default |

**Category breakdown (79 PRs):** Other 23 · Bug Fix 18 · CI/Infra 11 ·
Performance Optimization 9 · Refactor 7 · Turbo/Dependency Version Update 6 ·
Docs 5.

## Megatron-LM drift overview

- **Drift target:** `third_party/Megatron-LM`
- **Upstream:** `https://github.com/NVIDIA/Megatron-LM.git` (`main`)
- **Pinned SHA in Primus `main`:** `d3528a21301db2d12e92912b3ec025dc8a2ed4d6` (2026-03-06)
- **Pin change in-window:** None — the pin is unchanged across the entire window.
- **Upstream `main` HEAD SHA:** `114888079215f3acee3ebbd314b5279c587cf3a1`
- **Upstream ahead of pin by:** **1371 commits** (behind_by = 0; pin is an
  ancestor of upstream `main`).
- **Source-declared Megatron Core version at pin:** `0.16.0rc0`
  (`megatron/core/package_info.py`); upstream `main` declares `0.18.0`.
- **Recommendation:** `plan sync` — the gap is very large and continues to grow
  (was ~1050 commits at the 2026-08 report), but no in-window pin change forces
  an urgent action.

### Megatron-LM upstream feature delta table

Notable upstream areas that have moved since the pin (integration-relevant):

| Area | Notable upstream additions/fixes since the pin |
| --- | --- |
| Megatron-FSDP & distributed runtime | - **FSDP docs/refactor**: unified/refactored Megatron-FSDP documentation<br>- **FusedAdam**: fixed `use_decoupled_grad` handling in Megatron-FSDP<br>- **Mixed precision**: added/fixed MXFP8, uneven-DTensor, and frozen-parameter paths<br>- **Checkpointing**: added DCP and FSDP async-save support<br>- **Overlap**: refined all-gather / reduce-scatter overlap and precision-aware optimizer behavior |
| MoE, router & expert parallelism | - **Overlap**: improved shared-expert overlap and FlexDispatcher support<br>- **Router**: added a new router score function<br>- **Precision**: added NVFP4 native weights for DDP<br>- **Dispatch**: fixed non-quantized MoE dispatch padding<br>- **Backprop**: added A2A-combine backprop overlap with wgrad GEMM |
| Hybrid / Mamba & inference | - **Hybrid models**: added `megatron/core/models/hybrid/`<br>- **Naming**: renamed Mamba model/stack concepts toward Hybrid naming<br>- **Attention**: added YARN and DeepSeek Sparse Attention paths for Hybrid/Mamba<br>- **Inference**: added CUDA-graph support for MTP inference and prefix caching |
| Packaging & version metadata | - **Core version**: upstream advanced `megatron/core/package_info.py` from `0.16.0rc0` (pin) to `0.18.0`<br>- **Deps**: continued evolution of `pyproject.toml` / `megatron/core/requirements.txt` dependency groups<br>- **CI/docs**: broad workflow/docs surface churn across the gap |

> Assumption: the notable-area descriptions above are carried from the
> fact-checked backend-gap report for the same unchanged pin (`d3528a21`); only
> the ahead-count and upstream HEAD are refreshed to today's values.

## TorchTitan drift overview

- **Drift target:** `third_party/torchtitan`
- **Upstream:** `https://github.com/pytorch/torchtitan.git` (`main`)
- **Pinned SHA in Primus `main`:** `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce`
  (the tagged **v0.2.2** release, 2026-02-20)
- **Pin change in-window:** None — the pin is unchanged across the entire window.
- **Upstream `main` HEAD SHA:** `496b11d43860bb8d27b54568c76db6310ae7f55e`
- **Upstream ahead of pin by:** **918 commits** (behind_by = 0; pin is an
  ancestor of upstream `main`).
- **Version semantics:** `assets/version.txt` = `0.2.2` at the pin; upstream
  `main` is still `0.2.2` (dev toward the next tag).
- **Recommendation:** `plan sync` — the pin sits on a maintained tagged release,
  but upstream `main` continues to advance (was ~741 commits at the 2026-08
  report).

### TorchTitan upstream feature delta table

Notable upstream areas that have moved since the `v0.2.2` pin:

| Area | Notable upstream additions/fixes since the pin |
| --- | --- |
| New model directories | - **Kimi K2**: added `torchtitan/models/kimi_k2_7/` (upstream #3532)<br>- **Qwen 3.5**: added `torchtitan/models/qwen3_5/`<br>- **GPT-OSS**: added `torchtitan/models/gpt_oss/` with `spmd_types` enablement<br>- **DeepSeek-V3**: added mxfp8 debug config and compile fixes |
| Structure & shared abstractions | - **Common layer**: added shared `torchtitan/models/common/` abstractions<br>- **Flux**: moved `flux` into `torchtitan/models/flux/`<br>- **Experiments**: continued growth of the `experiments/` directory<br>- **Configs**: introduced the v0.2.2 debug-config section (Primus adapted via #952/#1041) |
| Dependencies & CI | - **Tokenizers**: added tokenizer-download support for newer `transformers`/`hub`<br>- **Test matrix**: expanded test dependencies and CI matrix<br>- **Nightly channels**: advanced `torch`/`torchao` and ROCm nightly channels past the tag<br>- **Release anchor**: pin anchored to `torch-2.12.0.dev20260220` / `torchao-0.17.0.dev20260220` |
| Integration coupling in Primus | - **Outer layer**: `primus/backends/torchtitan/` supplies the adapter/trainer/patches (~57 files)<br>- **Trainer**: Primus torchtitan trainer wraps upstream `torchtitan.train.Trainer`<br>- **Config drift guards**: Primus catches renamed configs (#952, #1041)<br>- **Grouped-mm toggles**: MI325X `use_turbo_grouped_mm` disabled then reverted (#1022 → #1029) |

## Primus-Turbo quarterly drift overview

- **Drift type:** current version vs quarter-start version on Primus `main`.
- **Quarter start (Q3 2026):** 2026-07-01T00:00:00+08:00
  (anchor commit `cfaba778`).
- **CI pin (`.github/workflows/ci.yaml` `PRIMUS_TURBO_COMMIT`):**
  quarter-start `3c39ef25` → current `6d5ff979` (**+107 commits**).
- **Benchmark pin (`.github/workflows/benchmark.yaml` `PRIMUS_TURBO_COMMIT`):**
  quarter-start `3c39ef25` → current `a04a233c` (**+35 commits**).
- **Companion pins:**
  - AITER: `b5e03ed1` → `0f3c58e6` (AITER **v0.1.14.post1** tag commit)
  - TRITON: `88b227e2` → `09500db9`
  - UCCL: `5afb4117` (unchanged)
- **In-window CI-pin bumps:** #933 (loss-NaN fix), #1028 (flydsl compile cache),
  #1048 (DeepEP cheap-fence fix).
- **Potential impact to Primus:** mostly additive FP8/MXFP4/MXFP8 GEMM and MoE
  kernel work plus correctness fixes; low-risk to consume. The DeepEP cheap-fence
  fix (#482 upstream, pulled via #1048) restores a system-scope write-back and is
  a correctness-relevant reason to stay current.
- **Recommendation:** `monitor` — Primus is tracking Primus-Turbo forward on a
  normal cadence within the quarter.

### Primus-Turbo quarterly drift table

Notable areas changed since quarter start (2026-07-01):

| Area | Notable changes since quarter start |
| --- | --- |
| FlyDSL GEMM/grouped-GEMM backends | - **MXFP8**: added FlyDSL backend for MXFP8 GEMM and native grouped MXFP8 GEMM on gfx950 (#390, #410)<br>- **MXFP4**: added FlyDSL backend for MXFP4 GEMM/grouped-GEMM (#424)<br>- **Autotune**: decoupled FlyDSL grouped fp8 autotune from M and added per-op autotune (#419, #447)<br>- **Correctness**: fp4 recipe fixes and wide-store fp16 correctness (#414, #440) |
| MoE / MegaMoE kernels | - **MegaMoE**: fused Mega-Kernel MoE on FlyDSL (#412) and fused MXFP8 MegaMoE dispatch+FC1/FC2+combine (#456)<br>- **Grad accum**: native fused grad-accumulation for gpt-oss-20b MoE (#470)<br>- **Permute**: barrier/coherence fixes in permute preprocessing (#435, #449)<br>- **gpt-oss tuning**: MXFP4 GEMM/quant tuning for gpt-oss-20b MoE on gfx950 (#460, #468) |
| Attention & sparse paths | - **HipKittens**: added gfx950 attention backend for head-dim 64/128 causal + SWA (#462)<br>- **Sparse attention**: added Triton and FlyDSL sparse-attention implementations (#420)<br>- **Sparse-MLA**: dispatch each pass through its own KernelBackend (#464)<br>- **Layouts**: sbhd & thd attention support for meta and gpt-oss (#458) |
| Hardware enablement & fixes | - **gfx1250**: build support parts 1 & 2 (#374, #423)<br>- **Grad-accum fuse**: tensorwise/mxfp8/mxfp4 support (#454, #459)<br>- **ODC backend**: migrated rocSHMEM communication backend into Primus-Turbo (#409)<br>- **DeepEP**: stop defaulting to the cheap fence to keep system-scope write-back (#482) |

## Source links

- Merged-PR query (GMT+8 window 2026-08-03 17:03 → 2026-09-01 17:05):
  `gh pr list --repo AMD-AGI/Primus --state merged --base main --search "merged:2026-08-03T09:03:00Z..2026-09-01T09:01:00Z"`
- Megatron-LM upstream: <https://github.com/NVIDIA/Megatron-LM/tree/main>
- Megatron-LM compare (pin → upstream main): <https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...main>
- TorchTitan upstream: <https://github.com/pytorch/torchtitan/tree/main>
- TorchTitan compare (pin → upstream main): <https://github.com/pytorch/torchtitan/compare/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce...main>
- Primus-Turbo: <https://github.com/AMD-AGI/Primus-Turbo>
- Primus-Turbo compare (Q3 start → current CI pin): <https://github.com/AMD-AGI/Primus-Turbo/compare/3c39ef259aa6d724c77c481e926466e7a167e938...6d5ff979eb019fbbcd91790ac812024cca05a882>
- Primus `main` submodule pins: <https://github.com/AMD-AGI/Primus/tree/main/third_party>
- Primus CI turbo pins: <https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml>
