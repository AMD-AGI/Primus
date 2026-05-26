# Primus Weekly Engineering Report — 2026-W18

## 1. Time Window

- Start: Monday 2026-04-27 00:00:00 Asia/Shanghai (GMT+8)
- End: Friday 2026-05-01 09:10 Asia/Shanghai (GMT+8) (report generation time)
- Branch observed: `origin/main`

## 2. Executive Summary

- **2 PRs merged to `main`** in the weekly window (Mon 2026-04-27 00:00 GMT+8 → Fri 2026-05-01 09:10 GMT+8).
- Category breakdown: **Bug Fix: 1**, **Other (feature): 1**; Performance Optimization, Turbo/Dependency Version Update, CI/Infra, Refactor, Docs: 0.
- **No backend pin or Turbo pin changes this week.** `third_party/Megatron-LM` is still pinned at `d3528a21` and `third_party/torchtitan` at `5fb7cc2e`. `PRIMUS_TURBO_COMMIT` is still `333b68d7` in `ci.yaml` and `a4488f6c` in `benchmark.yaml`. The week's only `.github/workflows/ci.yaml` change (in #693) commented out the ROCm Docker Hub login lines and does not move any version pin.
- **Megatron-LM upstream drift: `plan sync` (unchanged).** Pin is `d3528a21` (2026-03-06); upstream `main` HEAD is `3460bba1` (2026-05-01) — **413 commits ahead** (+48 since W17 snapshot `a1165fab`). Notable new upstream activity in this window: AllgatherV inference dispatcher (#4258), permute fusion in hybrid EP (#4089), DeepSeek/MoE prep (per-block MoE routing storage for prefix caching #4301, MTP CUDA graphs #4260), Megatron-FSDP doc unification (#4418), inference graph standardization (#4485) and inference RL graph fix (#4323), heterogeneous TP/DP MIMO `ColocatedBridgeCommunicator` (#4368), checkpoint integrity verification (#4305), `--global-batch-size` removal in step-batch-size release tests (#4545), and a build move of `mamba-ssm`/`causal-conv1d` to optional `[ssm]` extras (#4517).
- **torchtitan upstream drift: `urgent sync` (unchanged).** Pin is `5fb7cc2e` (2025-10-15); upstream `main` HEAD is `70340f4e` (2026-04-30) — **566 commits ahead** (+52 since W17 snapshot `d40df991`). Major new upstream activity in this window: GraphTrainer unified activation-memory framework (#3118), full inductor compilation pass (#3141), async-TP / micro-pipeline TP graph pass (#3129), HybridEP integration with GraphTrainer (#3007, #3177), MoE-series consolidation (`All2AllTokenDispatcher` for EP=1 and EP>1 #3125, ETP deprecation #3167), `ChunkedCELoss` (#2937), Full DTensor config-based sharding for Llama3/Qwen3/Llama4/DSV3/GPT-OSS (#2963, #2969), `MeshDimName → MeshAxisName` rename (#3113), and a CI-side `replace-imports-with-any` pattern to keep CI green when optional packages are missing (#3180).
- **Primus-Turbo month-to-date drift (April → May 1 GMT+8): `monitor` (no action needed).** Both CI and benchmark `PRIMUS_TURBO_COMMIT` pins remain byte-identical to their values at month start (2026-04-01 00:00 GMT+8) and to their previously reported W17 values.
- This week was the **second pass on the upstream Primus-Turbo `TEGroupedMLP` integration on the Megatron backend** (#693): the `te_spec_provider` patch no longer disables PrimusTurbo when `use_turbo_grouped_mlp=True` + `moe_grouped_gemm=True` + TE>=1.9.0; `validate_args_on_rocm` now hard-rejects `use_turbo_grouped_mlp=True` combined with `moe_use_legacy_grouped_gemm=True`; the legacy GroupedMLP/`grouped_gemm_util` code is preserved under `primus/backends/megatron/core/transformer/moe/deprecated_2caa681a1/` for backward compatibility with older configs that still set `moe_use_legacy_grouped_gemm=True`; and `PrimusTurboDeepEPTokenDispatcher` no longer gates `deepep_use_cuda_num_tokens_per_expert` on the legacy GroupedGEMM flag. Several MI300X/MI355X model YAMLs (`deepseek_v2_lite`, `deepseek_v3`, `gpt_oss_20B`, `qwen3_30B_A3B`, `qwen3_235B_A22B`) had their stale `moe_use_legacy_grouped_gemm: true` lines dropped to align with the new validation.
- **New model family: LFM/LFM2** (#651) — Megatron backend gains LiquidAI LFM2 support: `primus/configs/models/megatron/lfm_base.yaml` + `lfm2_8B_A1B.yaml`, a Primus implementation of the LFM2 short-convolution layer (`Lfm2ShortConv`) wired through a new `primus/backends/megatron/patches/gpt_decoder_layer_specs_patches.py`, and three example MI355X pretrain YAMLs (`lfm2_8B_A1B-{BF16,FP8,FP8-te-precision}-pretrain.yaml`). Also includes a small docker-build CI fix (commenting out `rocmshared` Docker Hub logins in the build/push workflow).
- Tooling/infra/docs: no merged tooling, infra-only, refactor, or docs PRs in this window. The shared backend-gap dashboard from W17's #687 is left unmodified.

## 3. Weekly PR Update Table

| PR | Merged Time (GMT+8) | Category | Key Update |
| --- | --- | --- | --- |
| [#693](https://github.com/AMD-AGI/Primus/pull/693) `fix: keep legacy groupedgemm on megatron backend` (author: `zhenhuang12`) | 2026-04-28 15:27 | Bug Fix | Second-pass alignment of PrimusTurbo with the upstream `TEGroupedMLP` path on the Megatron backend. Removes the patch-level guard that disabled PrimusTurbo when `use_turbo_grouped_mlp + moe_grouped_gemm + TE>=1.9.0`, and adds a hard `validate_args_on_rocm` check that forbids `use_turbo_grouped_mlp=True` combined with `moe_use_legacy_grouped_gemm=True`. `PrimusTurboDeepEPTokenDispatcher` no longer requires `moe_use_legacy_grouped_gemm=True` to enable `deepep_use_cuda_num_tokens_per_expert`. The legacy GroupedMLP path is preserved under `primus/backends/megatron/core/transformer/moe/deprecated_2caa681a1/` for backward-compat with older configs that still set the legacy flag. Sync-Free MoE stage 2/3 now requires `use_turbo_grouped_mlp=True` (instead of the old legacy flag). Strips the now-stale `moe_use_legacy_grouped_gemm: true` lines from the MI300X/MI355X DeepSeek-V2/V3, GPT-OSS-20B, and Qwen3 30B/235B example YAMLs. |
| [#651](https://github.com/AMD-AGI/Primus/pull/651) `LFM model support` (author: `wenxie-amd`) | 2026-04-28 11:44 | Other | Adds initial Megatron-backend support for LiquidAI's LFM2 model family. Introduces `primus/configs/models/megatron/lfm_base.yaml` + `lfm2_8B_A1B.yaml`, a Primus implementation of the LFM2 short-convolution "attention" layer (`Lfm2ShortConv`) plus an LFM-aware GPT layer-spec builder, and a new Megatron patch (`primus/backends/megatron/patches/gpt_decoder_layer_specs_patches.py`) that routes `get_gpt_decoder_layer_specs` to the Primus implementation when LFM-specific layer types are configured. Ships three MI355X example pretrain configs: `lfm2_8B_A1B-BF16-pretrain.yaml`, `lfm2_8B_A1B-FP8-pretrain.yaml`, `lfm2_8B_A1B-FP8-te-precision.yaml`. Also includes a docker-build CI fix in `.github/workflows/ci.yaml` (comments out four `docker login -u rocmshared ... ROCM_DOCKER_HUB_TOKEN` lines in the image build/push job). |

## 4. Megatron-LM Drift Overview

- Upstream: `https://github.com/NVIDIA/Megatron-LM.git` (`main`)
- Pinned in Primus `main` (`third_party/Megatron-LM`): `d3528a21301db2d12e92912b3ec025dc8a2ed4d6` — *fix(moe): fix TE general_gemm API change (#3582)*, 2026-03-06
- Upstream `main` HEAD: `3460bba1d6f945ec04f47fdb1dcee6da3259fcd8` — *Update copy-pr-bot.yaml [skip ci]* (2026-05-01)
- Last upstream functional change in this window: `83e7466c` — *Fixes for modelopt examples and SFTTokenizer for transformers v5 (#4450)* (2026-04-30)
- Commit gap: **upstream is 413 commits ahead of Primus pin** (+48 since the W17 snapshot `a1165fab`).
- Month-to-date movement on Primus side: pin unchanged in April; last submodule SHA bump on `main` was `3bec9aa9` → `d3528a21` inside PR #654 (merged 2026-04-10).
- Recommendation: **plan sync** (unchanged from W17). The accumulated upstream change set continues to grow (now MoE permute fusion in hybrid EP, AllgatherV inference dispatcher, MTP CUDA graphs, RL inference graph fixes, MFSDP doc unification) without changing Primus's existing sync risk profile.

### Notable upstream areas that have moved since the pin

- **MoE / EP**: permute fusion in hybrid EP (#4089) on top of W17's router, FlexDispatcher, and MTP-token-per-expert work; per-block MoE routing storage for prefix caching (#4301); previously reported router score function (#3673), shared-expert overlap incl. FlexDispatcher (#2207), permute-padding fix (#4038), MTP token-per-expert logging (#4412).
- **Inference / CUDA graphs**: AllgatherV inference dispatcher and old-dispatcher simplification (#4258); CUDA graphs for MTP inference (#4260); avoid nsys profile crash with CUDA graphs (#4541); standardize misc graph interface (#4485); fix inference graph override in RL flow (#4323); local-CG bugfixes for latent MoE loss-curve gaps (#4433); embedding/output layer in `full_iteration_inference` graph for hybrid models (#4440).
- **Mamba / Hybrid models**: avoid redundant HBM reloads in `causal_conv1d_update` shift loop (#4460); build move of `mamba-ssm`/`causal-conv1d` to optional `[ssm]` extra (#4517); on top of W17's `MambaModel`/`MambaStack` → `HybridModel`/`HybridStack` rename (#4099, #4159), Mamba inference opt (#4414), QK layernorm in `MambaModel` DPA (#4067), DeepSeek Sparse Attention port (#3553), fine-grained activation offloading (#4173), YARN support for hybrid_model (#4244).
- **Megatron-FSDP / DistOpt / DDP**: documentation unified and refactored (#4418); fix `FusedAdam.use_decoupled_grad` mis-set for Megatron-FSDP (#4427); add `reduce_scatter_with_fp32_accumulation` knob (#4410); on top of W17's MFSDP 0.5.0, MFSDP `decoupled_grad`/DistOpt fixes (#4133), layerwise-optimizer fixes (#4272, #4138), DDP parameter-layout refactor (#3812), MFSDP log gating (#4400).
- **Checkpoint / safety**: checkpoint integrity verification (#4305); SafeUnpickler class for safe pickle usage (#4319); SHA-256 prefix-cache hash replacing polynomial rolling hash (#4158); `weights_only=False` removal (#4434); `validate_access_integrity` knob on dist-ckpt load (#4422); fix checkpoint loading with rerun state machine (#4448); on top of W17's async-ckpt CPU-SHM (#4355) and cross-rank-sync removal (#2864).
- **Heterogeneous training / RL / misc**: `ColocatedBridgeCommunicator` for heterogeneous TP/DP MIMO training (#4368); ModelOpt examples + SFTTokenizer fixes for transformers v5 (#4450); ModelOpt list-format `quant_cfg` fix (#4187); YAML quant recipe in PTQ + first/last layer modifier removal (#4503); `--global-batch-size` removal from step-batch-size schedule release tests (#4545); training-migration container/serialization classes (#4227, #4309); upstream skill-doc updates (#4502, #4542).

### Megatron-LM upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| MoE / EP | Permute fusion in hybrid EP;<br>per-block MoE routing storage for prefix caching;<br>(carries) router score function, FlexDispatcher overlap, MTP token-per-expert logging | NVIDIA/Megatron-LM #4089, #4301, #3673, #2207, #4412 | Primus DeepSeek/Mixtral configs (`examples/megatron/configs/MI300X/deepseek_v*`, `qwen3_*`) may pick up additional EP perf knobs; aligns with this week's Primus-side groupedgemm cleanup in #693. |
| Inference / CUDA graphs | New AllgatherV inference dispatcher;<br>CUDA graphs for MTP inference;<br>standardized misc graph interface;<br>RL inference-graph override fix;<br>local-CG bugfixes for latent MoE loss curves | NVIDIA/Megatron-LM #4258, #4260, #4485, #4323, #4433, #4541 | Inference/post-train paths for Primus DSV3/MoE configs benefit; expand validation when sync lands. |
| Mamba / Hybrid | `causal_conv1d_update` HBM-reload reduction;<br>`mamba-ssm`/`causal-conv1d` move to optional `[ssm]` extra;<br>(carries) outside-core `MambaModel` → `HybridModel` rename | NVIDIA/Megatron-LM #4460, #4517, #4159, #4244 | The `[ssm]` extra split affects Primus Dockerfile install layering; Primus must audit `MambaModel`/`MambaStack` references in `primus/backends/megatron` before the next bump. |
| FSDP / DistOpt / DDP | MFSDP doc unification;<br>FusedAdam `use_decoupled_grad` Megatron-FSDP fix;<br>`reduce_scatter_with_fp32_accumulation` knob;<br>(carries) MFSDP 0.5.0 + layout refactor | NVIDIA/Megatron-LM #4418, #4427, #4410, #3812, #4400 | Primus FSDP/DDP launch paths (`primus/modules/trainer/megatron/*`) should re-validate post-sync; this week's Primus-side validation (#693) makes the Turbo grouped-MLP path the new default for MoE, which interacts with these MFSDP changes. |
| Checkpoint / safety | Checkpoint integrity verification;<br>SafeUnpickler;<br>SHA-256 prefix-cache hash;<br>`weights_only=False` removal;<br>`validate_access_integrity` knob | NVIDIA/Megatron-LM #4305, #4319, #4158, #4434, #4422, #4448 | Tighter checkpoint hardening; Primus pretrain at scale should benefit; coordinate when bumping the pin. |
| Heterogeneous / RL / misc | `ColocatedBridgeCommunicator` (NMFW-17);<br>ModelOpt fixes for transformers v5;<br>YAML quant recipe in PTQ;<br>step-batch-size release-test fix | NVIDIA/Megatron-LM #4368, #4450, #4187, #4503, #4545 | New heterogeneous-training entry point may inform future Primus multi-pod configs; ModelOpt fixes are required when Primus moves to transformers v5. |
| Schedule plan API (carry-over) | `post_attn` node already removed from `TransformerLayerSchedulePlan` (consumed in Primus by W17 #672) | Primus/#672, upstream schedule-plan change | Primus-side fix already shipped; treat as confirmed drift that future bumps must keep in sync. |

## 5. torchtitan Drift Overview

- Upstream: `https://github.com/pytorch/torchtitan.git` (`main`)
- Pinned in Primus `main` (`third_party/torchtitan`): `5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021` — *Deepseek-V3 toml file minor fix (#1894)*, 2025-10-15
- Upstream `main` HEAD: `70340f4ec31d9e1dbd448506cc4423934c3cd62f` — *[CI] Use replace-imports-with-any to avoid missing packages causing CI failures (#3180)* (2026-04-30)
- Commit gap: **upstream is 566 commits ahead of Primus pin** (+52 since the W17 snapshot `d40df991`).
- Month-to-date movement on Primus side: none (submodule SHA unchanged in April).
- Recommendation: **urgent sync** (unchanged from W17). The pin is now ~6.5 months stale; another major refactor wave landed upstream this week (GraphTrainer activation-memory framework, full inductor compilation pass, async-TP graph pass, MoE token-dispatcher consolidation, ETP deprecation, ChunkedCELoss, Full DTensor config-based sharding) that further widens the gap from the existing baseline report.

### Notable upstream areas that have moved since the pin

- **GraphTrainer / precompile**: unified framework for activation memory management (#3118); full inductor compilation pass (#3141); joint graph bucketing + prefetching that composes with SAC (#3056); async-TP / micro-pipeline TP graph pass (#3129); start deprecating jit/aot compile modes in graph_trainer (#2788); `apply_graph_ac` removed (#3147); `log_activation_memory_policy` for per-tensor inspection (#3062); on top of W17's CooR precompile for DSV3 (#2916), `aot_fx_trace` (#2975), regional-inductor precompile (#2883), `enable_cudagraph` (#3049), CUDA-graph annotation pass (#2926), CPU-offload activation pass (#3064), and SAC pass refactor (#3050).
- **HybridEP / MoE**: HybridEP enabled with GraphTrainer (#3007); HybridEP reads `comm_backend` from model config (#3177); `[MoE][3/n]` consolidate EP=1 and EP>1 to all use `All2AllTokenDispatcher` (#3125); `[MoE][4/n]` deprecate Expert Tensor Parallel (ETP) (#3167); on top of W17's token-dispatcher introduction (#2842) and EP-config-registry move (#2960).
- **Module / DTensor / TP**: Full DTensor config-based sharding infrastructure with Llama3 adoption (#2963) and follow-up for Qwen3, Llama4, DSV3, GPT-OSS (#2969); `MeshDimName → MeshAxisName` rename (#3113); remove `LocalMapInnerAttention`, use static `LocalMapSpec` (#2986); on top of W17's `fully_shard` migration (#2900), `.compile()` migration (#2688), and SimpleFSDP wrapper sharing (#2754).
- **Attention / loss**: `ChunkedCELoss` (#2937) plus disable chunked CE in graph trainer (#3115); revert "Remove MATH from ScaledDotProductAttention default backends" (#3135) after the original change (#3080); fused QKV support in Qwen3-VL state-dict adapter (#3102); on top of W17's Fused QKV GQAttention (#2878), `qk_norm` consolidation (#2872), FlexAttention bitwise-deterministic tests (#2903, #2989), 2-tier compilation with FlexAttention (#2929), CP+block_causal+FlexAttention position fix (#2780).
- **RL / inference / experiments**: RL env-rollout-based controller refactor (#3073); patched-Qwen3 parallel plan removed and merged into core torchtitan Qwen3 parallel plan (#3070); reset_prefix_cache (#3095); deprecate VLM experiment (#3151); remove stale autoparallel/deepseek_v3 experiment (#2271); on top of W17's RL trainer/generator refactors (#2985, #3001) and inference-example/vllm consolidation (#3045).
- **Datasets / CI / ROCm**: CI uses `replace-imports-with-any` to avoid missing-package failures (#3180); shuffle ChatDataset before splitting across nodes (#3131); reproducible training resume across epoch boundaries for map and streaming datasets (#3008); MoE loss comparison guard added to CI (#3081); ROCm experimental workflows toggled off again (#3140) after a brief revert (#3097); MI350 loss numbers updated (#3078); SAC test compatibility with PyTorch indexed storage (#3098); on top of W17's HF text-dataset reshuffle (#3023), VLM `torchvision` pin (#3047), MI350 label rollout (#2740), and tj-actions bumps (#3048).

### torchtitan upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| GraphTrainer / precompile | Unified activation-memory framework;<br>full inductor compilation pass;<br>joint graph bucketing + prefetching that composes with SAC;<br>async-TP / micro-pipeline TP graph pass;<br>jit/aot compile-mode deprecation;<br>activation-memory policy logging | pytorch/torchtitan #3118, #3141, #3056, #3129, #2788, #3062 | Major continuing perf/UX upgrade; remains unavailable behind the stale pin. Will require coordinated patches in `primus/backends/torchtitan/**` when Primus syncs. |
| HybridEP / MoE | HybridEP integration with GraphTrainer;<br>HybridEP `comm_backend` from model config;<br>`All2AllTokenDispatcher` consolidation across EP=1 and EP>1;<br>ETP deprecation | pytorch/torchtitan #3007, #3177, #3125, #3167 | Direct impact on Primus torchtitan MoE configs and any planned EP topology in `primus/modules/trainer/torchtitan/*`; ETP removal is a breaking config-level change. |
| Module / DTensor / TP | Full DTensor config-based sharding for Llama3/Qwen3/Llama4/DSV3/GPT-OSS;<br>`MeshDimName → MeshAxisName` rename;<br>`LocalMapInnerAttention` removed in favor of static `LocalMapSpec` | pytorch/torchtitan #2963, #2969, #3113, #2986 | Public-API rename + refactor; Primus torchtitan launcher and patches must be re-validated. |
| Attention / loss | `ChunkedCELoss`;<br>graph-trainer chunked-CE gating;<br>SDP-default-backend revert (re-include MATH);<br>fused QKV in Qwen3-VL adapter | pytorch/torchtitan #2937, #3115, #3135, #3102 | Loss-side knob useful for memory-bound training; SDP backend revert reduces breakage risk for Primus tests when sync lands. |
| RL / experiments | RL env-rollout-based controller refactor;<br>upstream Qwen3 parallel plan absorbs patched RL plan;<br>VLM and stale autoparallel/dsv3 experiments deprecated/removed | pytorch/torchtitan #3073, #3070, #3151, #2271 | Relevant for Primus post-training/RL on torchtitan; may unblock removing internal RL patches after sync. |
| Datasets / CI / ROCm | `replace-imports-with-any` CI pattern;<br>reproducible epoch-boundary resume for map/streaming datasets;<br>ChatDataset cross-node shuffle;<br>MoE loss comparison CI guard;<br>MI350 loss-number refresh | pytorch/torchtitan #3180, #3008, #3131, #3081, #3078 | Useful CI/MI350 hygiene reference for Primus torchtitan CI; the loss-comparison guard pattern can be mirrored in Primus torchtitan UTs. |

## 6. Primus-Turbo Monthly Drift Overview

- Drift type: **in-repo**, not upstream — compares Turbo version/SHA referenced on Primus `main` now vs the latest commit at or before `month_start_ts = 2026-04-01 00:00 Asia/Shanghai` (`2026-03-31 16:00 UTC`).
- Turbo is **not a submodule** in Primus. Canonical version source:
  - `.github/workflows/ci.yaml` → `PRIMUS_TURBO_COMMIT` (also wired through `.github/workflows/docker/Dockerfile`)
  - `.github/workflows/benchmark.yaml` → `PRIMUS_TURBO_COMMIT`
- Reference commit at month start on `main`: `76651575` (*[WIP][Megatron-LM] feat: reduce extra qkv transpose in attn (#625)*, 2026-03-31 14:19 GMT+8). The underlying Turbo pins at that commit are byte-identical to today's values.
- Current state on `origin/main`:
  - `ci.yaml` `PRIMUS_TURBO_COMMIT`: `333b68d7c81b722b21b4aad10cd250c45f15027c` — *fix sm_scale none bug (#263)*
  - `ci.yaml` `PRIMUS_TURBO_AITER_COMMIT`: `e83f9903c07001a0ec29e85d223f6e6cdbe00859`
  - `benchmark.yaml` `PRIMUS_TURBO_COMMIT`: `a4488f6cdb15cfff4383c61af7922bb50803f0ea` — *feat: update triton impl for mi300 & mi355 (#252)*
- Month-start state on `main`: all three values identical to current.
- **No Primus-Turbo drift in this comparison window.**
- Recommendation: **monitor**. The pre-existing skew between the two YAML pins (CI pin `333b68d` is 5 commits ahead of benchmark pin `a4488f6c` in Primus-Turbo history) is unchanged this month. This week's Primus-side change in #693 modifies how Primus *uses* Turbo's grouped-MLP path on the Megatron backend but does not bump any Turbo pin.

### Notable areas changed since month start

- **No changes this window** — both `ci.yaml` and `benchmark.yaml` Turbo pins on `main` are byte-identical to their 2026-03-30 values.
- **Primus-side Turbo integration moved**: PR #693 (this window) drops the legacy GroupedGEMM gating in `_is_primus_turbo_enabled` and `PrimusTurboDeepEPTokenDispatcher` so that Turbo's `TEGroupedMLP` path is now the default, but no `PRIMUS_TURBO_COMMIT` was bumped.
- **AITER pin unchanged**: `PRIMUS_TURBO_AITER_COMMIT` is identical to the month-start value.
- **Benchmark pin unchanged**: `benchmark.yaml` `PRIMUS_TURBO_COMMIT` is identical to the month-start value.
- **No Turbo drift in this comparison window.**

### Primus-Turbo monthly drift table

| Component | Current Version/SHA | Month-start Version/SHA | Delta Summary | Key Changes | Evidence |
| --- | --- | --- | --- | --- | --- |
| `PRIMUS_TURBO_COMMIT` (CI build) | `333b68d7c81b722b21b4aad10cd250c45f15027c` (*fix sm_scale none bug (#263)*) | `333b68d7c81b722b21b4aad10cd250c45f15027c` | No drift (0 commits) | No changes this window. Primus-side use of Turbo grouped-MLP changed in #693 without bumping the pin. | [`.github/workflows/ci.yaml` L17](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml#L17) |
| `PRIMUS_TURBO_AITER_COMMIT` (CI build) | `e83f9903c07001a0ec29e85d223f6e6cdbe00859` | `e83f9903c07001a0ec29e85d223f6e6cdbe00859` | No drift | No changes this window. | [`.github/workflows/ci.yaml` L18](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml#L18) |
| `PRIMUS_TURBO_COMMIT` (benchmark) | `a4488f6cdb15cfff4383c61af7922bb50803f0ea` (*feat: update triton impl for mi300 & mi355 (#252)*) | `a4488f6cdb15cfff4383c61af7922bb50803f0ea` | No drift | No changes this window. | [`.github/workflows/benchmark.yaml` L9](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/benchmark.yaml#L9) |

## 7. Source Links

- Primus main branch: https://github.com/AMD-AGI/Primus/tree/main
- Primus weekly PR listing (window): https://github.com/AMD-AGI/Primus/pulls?q=is%3Apr+is%3Amerged+base%3Amain+merged%3A%3E%3D2026-04-26T16%3A00%3A00Z
- PR #651: https://github.com/AMD-AGI/Primus/pull/651
- PR #693: https://github.com/AMD-AGI/Primus/pull/693
- Megatron-LM pin: https://github.com/NVIDIA/Megatron-LM/commit/d3528a21301db2d12e92912b3ec025dc8a2ed4d6
- Megatron-LM upstream HEAD (at report time): https://github.com/NVIDIA/Megatron-LM/commit/3460bba1d6f945ec04f47fdb1dcee6da3259fcd8
- Megatron-LM compare: https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...main
- torchtitan pin: https://github.com/pytorch/torchtitan/commit/5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021
- torchtitan upstream HEAD (at report time): https://github.com/pytorch/torchtitan/commit/70340f4ec31d9e1dbd448506cc4423934c3cd62f
- torchtitan compare: https://github.com/pytorch/torchtitan/compare/5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021...main
- Primus-Turbo CI pin: https://github.com/AMD-AGI/Primus-Turbo/commit/333b68d7c81b722b21b4aad10cd250c45f15027c
- Primus-Turbo benchmark pin: https://github.com/AMD-AGI/Primus-Turbo/commit/a4488f6cdb15cfff4383c61af7922bb50803f0ea
- Month-start reference commit on `main`: https://github.com/AMD-AGI/Primus/commit/76651575
- Last week's report (W17): https://github.com/AMD-AGI/Primus/blob/main/docs/weekly_reports/2026-W17-primus-weekly.md

---

*Generated automatically by the Primus weekly report automation. Factual statements are derived from `git log origin/main`, the pinned submodule SHAs in `third_party/`, and the `PRIMUS_TURBO_COMMIT` values in `.github/workflows/{ci,benchmark}.yaml` as observed at 2026-05-01 09:10 GMT+8. Upstream-HEAD SHAs and commit counts are snapshots at report generation time.*
