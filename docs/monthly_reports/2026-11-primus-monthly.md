# Primus Monthly Engineering Report — 2026-11

## Time window

- Timezone: Asia/Shanghai (GMT+8)
- Window start: `2026-10-01T17:02:00+08:00`
- Window end: `2026-11-01T17:02:00+08:00`
- Span: ~31 days (continuity-based; chains off the previous published report end `2026-10-01T17:02:00+08:00`)
- `gap_detected`: `true` — the window start (`2026-10-01`) is earlier than the first day of the current calendar month (`2026-11-01`), so this run backfills the final days of October that a plain calendar-month window would have dropped. No unresolved (> 7 day) coverage gap exists between published reports; all weekly and monthly reports chain contiguously.

This report covers `2026-10-01 → 2026-11-01` so that every PR merged and every backend pin change since the last report (`2026-10`) is captured. The window does not exceed 45 days, so no extended-backfill sanity flag is raised.

## Executive summary

- **80 PRs merged** on `main` in the window, led by **Other/feature** work (24), **bug fixes** (22), and **performance optimization** (16).
- **Megatron-LM pin CHANGED in-window**: PR [#1321](https://github.com/AMD-AGI/Primus/pull/1321) bumped `third_party/Megatron-LM` from `d3528a21` (Megatron Core `0.16.0rc0`) to the **`core_v0.19.2` release tag** (`4b4acac9`, committed 2026-09-18). This is a backend version/pin change, so the Megatron backend-gap report set was regenerated against upstream `NVIDIA/Megatron-LM` `main`.
- After the bump, the Megatron pin is **1011 commits behind** upstream `main` (`037b883c`, Core `0.21.0`) — a much smaller gap than the pre-bump ~1662 commits, but still a meaningful planned-sync target. Recommendation: **plan sync**.
- **TorchTitan pin UNCHANGED** (`73a0e697`, upstream `v0.2.2` tag). Upstream `main` is now **1537 commits ahead** (up from 1194 at the last report). Recommendation: **plan sync**. No TorchTitan backend-gap regeneration was needed because the pin did not move.
- **Primus-Turbo moved materially inside Q4** (quarter started `2026-10-01` GMT+8): the CI-tracked `PRIMUS_TURBO_COMMIT` advanced from `ae1ac068` to `d04ffd9b` (**28 commits**, spanning tagged releases `v0.6.0` and `v0.6.1`), and dependency pins bumped (AITER `v0.1.14.post1 → v0.1.24.post1`, Triton `→ release/3.9.x (3.9.0)`). Recommendation: **monitor**.
- Other in-window submodule movement with no backend-gap report set (noted, not regenerated): `maxtext` (PRs [#1228](https://github.com/AMD-AGI/Primus/pull/1228), [#1289](https://github.com/AMD-AGI/Primus/pull/1289)) and `Automodel` (PR [#1314](https://github.com/AMD-AGI/Primus/pull/1314)); `HunyuanWorld-Mirror-rocm` was added as a new submodule (PR [#1242](https://github.com/AMD-AGI/Primus/pull/1242)).

## Monthly PR update table

Category breakdown: Other 24 · Bug Fix 22 · Performance Optimization 16 · CI/Infra 8 · Turbo/Dependency Version Update 6 · Docs 3 · Refactor 1 (total 80).

| PR | Merged Time (GMT+8) | Category | Key Update |
| --- | --- | --- | --- |
| [#1209](https://github.com/AMD-AGI/Primus/pull/1209) | 2026-10-02 04:43 | Bug Fix | [Fix] Tune batch size in Zebra Llama-Mamba 3B BF16 pretraining config to avoid MI300 OOM |
| [#1215](https://github.com/AMD-AGI/Primus/pull/1215) | 2026-10-02 08:33 | Bug Fix | fix(megatron): disable Triton recompiles for zebra Llama KDA 1B |
| [#1203](https://github.com/AMD-AGI/Primus/pull/1203) | 2026-10-02 22:46 | Other | feat(projection): Adding a MaxText/JAX backend to Primus Projection |
| [#1220](https://github.com/AMD-AGI/Primus/pull/1220) | 2026-10-03 17:38 | Other | Revert "feat(projection): Adding a MaxText/JAX backend to Primus Projection" |
| [#1202](https://github.com/AMD-AGI/Primus/pull/1202) | 2026-10-03 21:38 | Other | feat(projection): Adding a TorchTitan backend to Primus Projection |
| [#1221](https://github.com/AMD-AGI/Primus/pull/1221) | 2026-10-05 21:42 | Other | chore(mlperf): update Training ruleset to 6.1 |
| [#1223](https://github.com/AMD-AGI/Primus/pull/1223) | 2026-10-06 09:10 | Bug Fix | fix(mlperf): consume merged precision logging events |
| [#1219](https://github.com/AMD-AGI/Primus/pull/1219) | 2026-10-06 09:21 | Performance Optimization | perf(megatron): fuse GPT-OSS clipping and Adam state traffic |
| [#1207](https://github.com/AMD-AGI/Primus/pull/1207) | 2026-10-06 13:14 | Performance Optimization | feat(megatron): expose CE gradient ReduceScatter opt-in |
| [#1208](https://github.com/AMD-AGI/Primus/pull/1208) | 2026-10-06 13:15 | Performance Optimization | feat(gptoss): integrate FlyDSL QK RoPE and RMSNorm-MXFP4 QKV fusions |
| [#1222](https://github.com/AMD-AGI/Primus/pull/1222) | 2026-10-06 13:15 | Performance Optimization | perf(megatron): integrate fused GPT-OSS deoscillation |
| [#1225](https://github.com/AMD-AGI/Primus/pull/1225) | 2026-10-07 08:05 | Performance Optimization | perf(megatron): tune MI325X env settings and extend MI325X perf configs |
| [#1206](https://github.com/AMD-AGI/Primus/pull/1206) | 2026-10-07 08:07 | Other | SpecForge launch modes and homogeneous multi-node capture/offline |
| [#1226](https://github.com/AMD-AGI/Primus/pull/1226) | 2026-10-08 02:58 | CI/Infra | [OOB Release] add JAX v26.8 dockerfile |
| [#1228](https://github.com/AMD-AGI/Primus/pull/1228) | 2026-10-08 03:00 | Turbo/Dependency Version Update | feat(maxtext): align MaxText backend with v26.8 |
| [#1229](https://github.com/AMD-AGI/Primus/pull/1229) | 2026-10-08 03:01 | CI/Infra | feat(tools): add an end-to-end convergence test for Megatron and MaxText |
| [#1230](https://github.com/AMD-AGI/Primus/pull/1230) | 2026-10-08 07:52 | Other | feat(glm5_next): add GLM-5.3-Flash (KDA + DSA hybrid with mHC) support |
| [#1218](https://github.com/AMD-AGI/Primus/pull/1218) | 2026-10-08 08:10 | Performance Optimization | perf(mi355x): tune pretrain configurations |
| [#1224](https://github.com/AMD-AGI/Primus/pull/1224) | 2026-10-08 11:28 | Other | feat(projection): DLRM/HSTU explicit backward GEMMs and arch-scaled attention epilogue |
| [#1232](https://github.com/AMD-AGI/Primus/pull/1232) | 2026-10-09 08:17 | Other | feat(projection): MaxText/JAX backend for Primus Projection |
| [#1236](https://github.com/AMD-AGI/Primus/pull/1236) | 2026-10-09 12:43 | Turbo/Dependency Version Update | ci: bump Primus-Turbo to e2d9f1d7 and stop forcing the DeepEP cheap fence in MoE recipes |
| [#1237](https://github.com/AMD-AGI/Primus/pull/1237) | 2026-10-09 18:22 | Bug Fix | fix(runner): keep a preset HIP_VISIBLE_DEVICES instead of overwriting it |
| [#1235](https://github.com/AMD-AGI/Primus/pull/1235) | 2026-10-10 02:04 | Docs | docs(release): update documentation for v26.8 |
| [#1234](https://github.com/AMD-AGI/Primus/pull/1234) | 2026-10-10 12:16 | Bug Fix | fix(torchtitan): gate the primus_turbo attention converter on use_turbo_attention |
| [#1243](https://github.com/AMD-AGI/Primus/pull/1243) | 2026-10-10 12:57 | Other | feat(mega_moe): accept MiniMax-M3's swigluoai (quick_geglu) activation |
| [#1240](https://github.com/AMD-AGI/Primus/pull/1240) | 2026-10-10 16:37 | Performance Optimization | perf(megatron): integrate opt-in BF16 router backward GEMMs |
| [#1241](https://github.com/AMD-AGI/Primus/pull/1241) | 2026-10-11 09:42 | Performance Optimization | perf(moe): defer forward and backward gathers to Turbo MXFP4 quantization |
| [#1040](https://github.com/AMD-AGI/Primus/pull/1040) | 2026-10-12 08:23 | Bug Fix | fix(deepseek_v4): contract mHC comb on the column-normalised axis |
| [#1130](https://github.com/AMD-AGI/Primus/pull/1130) | 2026-10-12 13:28 | Refactor | refactor(examples): group per-backend material under its backend directory |
| [#1247](https://github.com/AMD-AGI/Primus/pull/1247) | 2026-10-12 16:06 | Performance Optimization | perf(qwen3): Qwen3-30B-A3B FP8 on MI355X to 40k tokens/s/GPU with Primus-Turbo tensorwise |
| [#1242](https://github.com/AMD-AGI/Primus/pull/1242) | 2026-10-13 07:18 | Other | feat(worldmirror): add HunyuanWorld-Mirror stage 1 and stage 2 training |
| [#1248](https://github.com/AMD-AGI/Primus/pull/1248) | 2026-10-14 16:16 | Other | feat(mi455x): add FP8 hybrid LoRA SFT proxy recipes |
| [#1249](https://github.com/AMD-AGI/Primus/pull/1249) | 2026-10-14 17:05 | Bug Fix | fix(megatron): use the real model vocab under mock data |
| [#1252](https://github.com/AMD-AGI/Primus/pull/1252) | 2026-10-15 09:44 | Performance Optimization | perf(mi355x): tune GPT-OSS 20B and DeepSeek V2 Lite pretrain configs |
| [#1245](https://github.com/AMD-AGI/Primus/pull/1245) | 2026-10-15 10:07 | Performance Optimization | Add opt-in Turbo fused cross entropy for TP1 training |
| [#1199](https://github.com/AMD-AGI/Primus/pull/1199) | 2026-10-15 13:53 | Other | [Megatron-LM] Add ZAYA1-8B Megatron pretraining implementation |
| [#1260](https://github.com/AMD-AGI/Primus/pull/1260) | 2026-10-16 09:45 | CI/Infra | feat(tools/perf): run control, provenance, failure reasons, diffusion/DLRM backend and Excel report |
| [#1257](https://github.com/AMD-AGI/Primus/pull/1257) | 2026-10-16 10:20 | Bug Fix | fix(megatron): recommend and verify expandable segments for Turbo grouped GEMM (AIMA-242) |
| [#1251](https://github.com/AMD-AGI/Primus/pull/1251) | 2026-10-16 10:46 | CI/Infra | feat(auto_benchmark): add release suites, backends and record the setup with each result |
| [#1265](https://github.com/AMD-AGI/Primus/pull/1265) | 2026-10-19 08:19 | Bug Fix | fix(glm5_next): run packed KDA and DSA under context parallel |
| [#1254](https://github.com/AMD-AGI/Primus/pull/1254) | 2026-10-19 08:20 | Bug Fix | fix(glm5_next): accept text-only HF checkpoint names in the GLM-5.3 loader |
| [#1261](https://github.com/AMD-AGI/Primus/pull/1261) | 2026-10-19 08:25 | CI/Infra | feat(convergence_test): add maxtext llama2_7B fp8 recipe for CDNA4 |
| [#1233](https://github.com/AMD-AGI/Primus/pull/1233) | 2026-10-19 08:30 | Performance Optimization | feat(torchtitan): add Triton fused AdamW with a device-sized grid, on by default |
| [#1255](https://github.com/AMD-AGI/Primus/pull/1255) | 2026-10-19 15:29 | Docs | docs(specforge): publish SpecForge guides on the ROCm docs site |
| [#1269](https://github.com/AMD-AGI/Primus/pull/1269) | 2026-10-20 10:10 | Other | feat(zaya1): add FP8 recipes that train cold on mock data |
| [#1266](https://github.com/AMD-AGI/Primus/pull/1266) | 2026-10-20 10:58 | CI/Infra | feat(tools/perf): record the platform with every batch; fix TorchTitan and Megatron-Bridge TFLOP/s |
| [#1272](https://github.com/AMD-AGI/Primus/pull/1272) | 2026-10-21 08:05 | Other | mi455x_v1: add MI455X (gfx1250) DeepSeek-V3 proxy and GPT-OSS examples |
| [#1271](https://github.com/AMD-AGI/Primus/pull/1271) | 2026-10-21 08:23 | Performance Optimization | feat(mega_moe): adopt the Mega MoE buffers sized to the dispatched rows (Primus-Turbo #566, #568) |
| [#1274](https://github.com/AMD-AGI/Primus/pull/1274) | 2026-10-21 14:08 | CI/Infra | ci/docker, torchtitan: v26.8 CI follow-ups (origami -Bsymbolic, SAC profiler ops, perf_host flake) |
| [#1275](https://github.com/AMD-AGI/Primus/pull/1275) | 2026-10-21 15:25 | Turbo/Dependency Version Update | fix(flydsl): port Primus FlyDSL kernels to the 0.3.x API |
| [#1285](https://github.com/AMD-AGI/Primus/pull/1285) | 2026-10-21 17:18 | Other | feat(mega_moe): run GPT-OSS on MegaMoE (Turbo bump, expert biases) |
| [#1273](https://github.com/AMD-AGI/Primus/pull/1273) | 2026-10-21 22:17 | Bug Fix | fix(megatron): stop sys.modules scans from building unused apex ops |
| [#1281](https://github.com/AMD-AGI/Primus/pull/1281) | 2026-10-22 09:13 | Bug Fix | fix(megatron): record_stream prefetched batches on the consumer stream |
| [#1283](https://github.com/AMD-AGI/Primus/pull/1283) | 2026-10-22 09:50 | Turbo/Dependency Version Update | ci: bump the MaxText CI image to maxtext-v26.8 |
| [#1286](https://github.com/AMD-AGI/Primus/pull/1286) | 2026-10-23 22:16 | Performance Optimization | feat(megatron): reuse the logits buffer in TE fused cross entropy (overwrite_input) |
| [#1303](https://github.com/AMD-AGI/Primus/pull/1303) | 2026-10-26 08:31 | Other | chore(config): update MI300X Llama-3.1-70B FP8 batch size |
| [#1294](https://github.com/AMD-AGI/Primus/pull/1294) | 2026-10-26 08:36 | Other | update the mi325x yaml config files for v26.9 image |
| [#1289](https://github.com/AMD-AGI/Primus/pull/1289) | 2026-10-26 08:38 | Turbo/Dependency Version Update | maxtext: bump to release/v26.9, silence TE return_max_logit warnings, default autotune level 3 |
| [#1262](https://github.com/AMD-AGI/Primus/pull/1262) | 2026-10-26 08:42 | Bug Fix | fix(hybrid): point the Zebra converter wrappers at a KDA checkpoint |
| [#1276](https://github.com/AMD-AGI/Primus/pull/1276) | 2026-10-27 08:08 | Other | feat(zaya1): size MI355X 8B batches to GPU memory |
| [#1176](https://github.com/AMD-AGI/Primus/pull/1176) | 2026-10-27 08:39 | Bug Fix | fix(megatron): QK-Norm on MLA decoupled-RoPE to stop hybrid divergence |
| [#1216](https://github.com/AMD-AGI/Primus/pull/1216) | 2026-10-27 08:45 | Bug Fix | fix(torchtitan): bound DeepSeek compile at routed experts |
| [#1290](https://github.com/AMD-AGI/Primus/pull/1290) | 2026-10-27 08:52 | Bug Fix | fix(hybrid): read the FLA switches without requiring Megatron-LM args |
| [#1291](https://github.com/AMD-AGI/Primus/pull/1291) | 2026-10-27 08:56 | Bug Fix | fix(megatron-bridge): load SQuAD by its namespaced id in the Zebra SFT configs |
| [#1278](https://github.com/AMD-AGI/Primus/pull/1278) | 2026-10-27 08:58 | CI/Infra | add zayz1 model into the benchmarking test list |
| [#1315](https://github.com/AMD-AGI/Primus/pull/1315) | 2026-10-27 08:59 | Bug Fix | fix(specforge): install tensorboard with its dependencies |
| [#1301](https://github.com/AMD-AGI/Primus/pull/1301) | 2026-10-27 09:02 | Bug Fix | fix(runner): stop MoE hangs on hosts with amdgpu DRM older than 3.64 |
| [#1312](https://github.com/AMD-AGI/Primus/pull/1312) | 2026-10-27 13:06 | Other | mi455x_v2: add MI455X Llama-3.1-8B example (FlyDSL attention) and hipBLASLt GEMM tuning |
| [#1314](https://github.com/AMD-AGI/Primus/pull/1314) | 2026-10-27 13:11 | Other | feat(automodel): rename backend, AutoModel bump, patch-registered hooks, DDP AC repair, profiler |
| [#1322](https://github.com/AMD-AGI/Primus/pull/1322) | 2026-10-27 20:36 | Other | feat(automodel): FLUX and Wan training paths, synthetic data, and schnell guidance |
| [#1256](https://github.com/AMD-AGI/Primus/pull/1256) | 2026-10-28 08:22 | Other | feat(inkling): add Inkling (Small and Large) text backbone to the Megatron backend |
| [#1331](https://github.com/AMD-AGI/Primus/pull/1331) | 2026-10-28 09:00 | Docs | docs(release-notes): remove mention of MI308X |
| [#1329](https://github.com/AMD-AGI/Primus/pull/1329) | 2026-10-28 09:02 | Performance Optimization | perf(megatron): tune MI300X GPT-OSS-20B and Llama 70B pretrain configs |
| [#1323](https://github.com/AMD-AGI/Primus/pull/1323) | 2026-10-28 09:20 | Other | feat(automodel): Primus-Turbo attention overrides and FP8/MXFP4 linear swaps |
| [#1190](https://github.com/AMD-AGI/Primus/pull/1190) | 2026-10-28 12:53 | Performance Optimization | feat(megatron): add Triton fused AdamW on the shared kernels, on by default |
| [#1324](https://github.com/AMD-AGI/Primus/pull/1324) | 2026-10-28 15:20 | Other | feat(automodel): Ideogram-4 training with var-len attention, an encoded cache, ZeRO-1 and context parallelism |
| [#1321](https://github.com/AMD-AGI/Primus/pull/1321) | 2026-10-29 14:20 | Turbo/Dependency Version Update | chore(megatron): bump Megatron-LM to core_v0.19.2 |
| [#1340](https://github.com/AMD-AGI/Primus/pull/1340) | 2026-10-30 19:08 | Bug Fix | fix(mega_moe): run MegaMoE on Megatron-LM core_v0.19.2 with PrimusTopKRouter |
| [#1297](https://github.com/AMD-AGI/Primus/pull/1297) | 2026-10-30 19:09 | Bug Fix | fix(megatron_bridge): prefer rope_parameters over default_theta when resolving rope_theta |
| [#1298](https://github.com/AMD-AGI/Primus/pull/1298) | 2026-10-30 19:09 | Bug Fix | fix(megatron_bridge): use gpt_step for text-only models in posttrain |

## Megatron-LM drift overview

The `third_party/Megatron-LM` pin **changed in this window**: PR [#1321](https://github.com/AMD-AGI/Primus/pull/1321) (merged 2026-10-29) bumped the submodule and `primus/_thirdparty.lock` from `d3528a21` (Megatron Core source version `0.16.0rc0`) to the **`core_v0.19.2` release tag** `4b4acac9` (committed 2026-09-18). Because this is a backend version/pin change, the Megatron backend-gap report set was regenerated (see `docs/backend-gap/reports/megatron/upstream-main/`).

| Item | Primus pin (post-bump) | Upstream `NVIDIA/Megatron-LM` `main` |
| --- | --- | --- |
| Pinned commit | `4b4acac9` (`core_v0.19.2` tag) | `037b883c` |
| Megatron Core source version | `0.19.2` (`megatron/core/package_info.py`) | `0.21.0` |
| Commit date | 2026-09-18 | 2026-11-01 |
| Commit gap | Behind by `1011` commits (ahead-by-0; `28` release-only commits not on `main`) | - |
| Diff size | `2722 files changed, 531548 insertions, 293033 deletions` | - |

The bump cut the pre-bump drift (~1662 commits against `main` at the last report) down to 1011 commits, but a meaningful gap remains. Because `4b4acac9` is a release tag rather than a `main` ancestor, it carries 28 release-only commits not present on `main`; the authoritative behind-count is 1011.

Notable upstream areas that have moved since the pin:

- **Megatron-FSDP (MFSDP v2)**: DP-outer gradient-reduction stream overlap, `SchedulePolicy` grouping, MXFP8-recompute unshard planes, and recorded execution order for prefetch.
- **MoE / expert parallelism**: Triton fused batch-invariant unpermute, THD/sequence packing for NCCL EP, sigmoid-MoE bit-exact inference routing, and all-to-all router-prob release after combine.
- **SSM / Hybrid (GDP, KDA)**: FP32 `A_log`/`dt_bias` under BF16, in-place GDP `in_proj` GTP shards, CuTeDSL chunkwise-CP boundary compilation, and KDA (Kimi Delta Attention) hybrid layers.
- **Inference**: FlashInfer MXFP8 expert-weight refitting, per-token NVFP4 squared-ReLU/SwiGLU MoE via CuTe DSL, disaggregated end-to-end inference, and prefix-cache prompt scoring.
- **Optimizer / training**: `OptimizerConfig` migration, configurable RMSNorm-gamma weight decay, mHC precision controls with full-activation recompute, and LayerShardedMuon streaming.

## Megatron-LM upstream feature delta table

| Area | Representative upstream changes since `core_v0.19.2` |
| --- | --- |
| Megatron-FSDP (MFSDP v2) | DP-outer gradient reduction on its own stream (#6973)<br>MXFP8-recompute unshard planes in backward (#8432)<br>Record execution order for prefetch (#8478) |
| MoE / expert parallelism | Fused Triton batch-invariant MoE unpermute (#8559)<br>THD and sequence packing for NCCL EP (#8458)<br>Bit-exact sigmoid MoE inference routing (#8296) |
| SSM / Hybrid (GDP / KDA) | FP32 GDP `A_log`/`dt_bias` under BF16 (#8528)<br>CuTeDSL GDP chunkwise-CP boundary compile (#8420)<br>KDA hybrid layers with GLM-style two-stage gates (#7054) |
| Inference | FlashInfer MXFP8 expert-weight refit (#8560)<br>Per-token NVFP4 squared-ReLU / SwiGLU MoE via CuTe DSL (#8262)<br>Disaggregated inference end-to-end (#6675) |
| Optimizer / training | Migrate `OptimizerConfig` (#8213)<br>Configurable weight decay for RMSNorm gamma (#8246)<br>mHC precision controls + full activation recompute (#7521) |

## TorchTitan drift overview

The `third_party/torchtitan` pin **did not change** in this window. It remains `73a0e697`, which corresponds to the upstream `v0.2.2` release tag. No TorchTitan backend-gap regeneration was performed because the backend version/pin was unchanged.

| Item | Primus pin | Upstream `pytorch/torchtitan` `main` |
| --- | --- | --- |
| Pinned commit | `73a0e697` (`v0.2.2` tag) | `d86c77d6` |
| Declared version (`assets/version.txt`) | `0.2.2` | `0.2.2` |
| Commit date | - | 2026-11-01 |
| Commit gap | Behind by `1537` commits (pin is an ancestor of `main`; behind-by-0) | - |

The gap widened from 1194 commits at the last report to 1537, driven entirely by upstream movement; the Primus pin is unchanged. Upstream `assets/version.txt` still declares `0.2.2`, so the version string is stable even as `main` advances.

## TorchTitan upstream feature delta table

No TorchTitan pin movement occurred in this comparison window, so there is **no TorchTitan drift in this comparison window** attributable to a Primus-side change. The Primus-pinned `v0.2.2` baseline is unchanged; the growth to 1537 commits behind reflects upstream-only activity. For the standing feature-delta analysis against upstream `main`, see the TorchTitan backend-gap report (`docs/backend-gap/reports/torchtitan/upstream-main/`), which was last regenerated when the pin moved (2026-08-03).

| Area | Status |
| --- | --- |
| Primus TorchTitan pin | Unchanged (`73a0e697`, `v0.2.2`) in this window |
| Upstream `main` | Advanced to `d86c77d6` (1537 commits ahead of the pin) |
| Backend-gap regeneration | Not triggered (pin unchanged) |

## Primus-Turbo quarterly drift overview

- Current calendar quarter: **Q4 2026** (`quarter_start_ts = 2026-10-01T00:00:00+08:00`).
- Quarter-start anchor commit on Primus `main`: `bc8fd627` (2026-09-29, the last `main` commit at/before the quarter start).
- The CI-tracked Primus-Turbo pin (`PRIMUS_TURBO_COMMIT` in `.github/workflows/ci.yaml`) advanced from `ae1ac068` (quarter start) to `d04ffd9b` (current) — a **28-commit** span (ahead-by-28, behind-by-0, 143 files). The span crosses the tagged releases `v0.6.0` and `v0.6.1` (the quarter-start pin predates `v0.6.0`; the current pin sits just past `v0.6.1`).
- Dependency pins bumped alongside the Turbo pin this quarter: **AITER** `v0.1.14.post1` (`0f3c58e6`) → `v0.1.24.post1` (`04896172`), and **Triton** `09500db9` → `fc8d31a5` (`release/3.9.x`, 3.9.0). **UCCL** is unchanged (`5afb4117`).
- Potential impact to Primus: the new Turbo revision lands FlyDSL/MXFP4 grouped-GEMM, MoE permute-backend, and Mega MoE memory-sizing work that several in-window Primus PRs depend on (e.g. [#1271](https://github.com/AMD-AGI/Primus/pull/1271), [#1285](https://github.com/AMD-AGI/Primus/pull/1285), [#1275](https://github.com/AMD-AGI/Primus/pull/1275)); the AITER and Triton version bumps change the kernel/toolchain baseline the images are built against. Recommendation: **monitor**.

Notable areas changed since quarter start:

- **FlyDSL kernels**: MXFP4 grouped GEMM optimized on gfx950, GPT-OSS QK-RoPE and RMSNorm-MXFP4 fusions, and FlyDSL upgraded to `0.3.4.1`.
- **Mega MoE**: buffers/memory sized to the dispatched rows (BF16 and MXFP8), MiniMax-M3 SwiGLU-OAI activation, and GPT-OSS expert-bias support.
- **MoE dispatch / permute**: HIP and Triton `moe_permute` backends (Triton default), HIP permute in `DeepEPTokenDispatcher`, and tensorwise-FP8 permute for the grouped MLP.
- **Attention**: FlyDSL varlen-backward build fixes, SWA head_dim-128 forward correctness, FP8-attention layout handling, and gfx1250/MI455X FlyDSL MLA attention.
- **Dependency baseline**: AITER bumped to `v0.1.24.post1` and Triton to `release/3.9.x` (3.9.0).

## Primus-Turbo quarterly drift table

| Item | Quarter start (2026-10-01) | Current (2026-11-01) |
| --- | --- | --- |
| `PRIMUS_TURBO_COMMIT` (CI) | `ae1ac068` | `d04ffd9b` |
| Tagged release coverage | pre-`v0.6.0` | just past `v0.6.1` |
| Commit delta | - | +28 commits (143 files) |
| AITER pin | `0f3c58e6` (`v0.1.14.post1`) | `04896172` (`v0.1.24.post1`) |
| Triton pin | `09500db9` | `fc8d31a5` (`release/3.9.x`, 3.9.0) |
| UCCL pin | `5afb4117` | `5afb4117` (unchanged) |

## Source links

- Primus repository: [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus)
- Merged-PR query (window): `gh pr list --repo AMD-AGI/Primus --state merged --base main --search "merged:2026-10-01T09:02:00Z..2026-11-01T09:02:00Z"`
- Megatron-LM pin bump: PR [#1321](https://github.com/AMD-AGI/Primus/pull/1321); MegaMoE core_v0.19.2 compat follow-up PR [#1340](https://github.com/AMD-AGI/Primus/pull/1340)
- Upstream Megatron-LM: [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (`main` HEAD `037b883c`, Core `0.21.0`); pin tag `core_v0.19.2` = `4b4acac9`
- Upstream TorchTitan: [pytorch/torchtitan](https://github.com/pytorch/torchtitan) (`main` HEAD `d86c77d6`); pin `73a0e697` = `v0.2.2`
- Primus-Turbo: [AMD-AGI/Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) (`PRIMUS_TURBO_COMMIT` quarter drift `ae1ac068` → `d04ffd9b`)
- Megatron backend-gap report: `docs/backend-gap/reports/megatron/upstream-main/report.md`
- Canonical pin source: `primus/_thirdparty.lock`; Turbo/AITER/Triton pins: `.github/workflows/ci.yaml`
