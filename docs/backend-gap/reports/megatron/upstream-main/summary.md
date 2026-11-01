# Primus Megatron Upstream Gap One-Page Summary

> Date: 2026-11-01

## High-Level Comparison

| Item | Current Megatron-LM in Primus | Upstream `NVIDIA/Megatron-LM` `main` | Impact |
| --- | --- | --- | --- |
| Source-declared Megatron Core version | `0.19.2` from `megatron/core/package_info.py` | `0.21.0` | Spans two upstream minor-version steps |
| Pinned commit | `4b4acac9` (`core_v0.19.2` tag) | `037b883c` | Pin was just bumped (PR #1321); gap much smaller than before |
| Commit gap | Behind by `1011` commits | - | Not a small patch-level gap, but materially reduced |
| Diff size | `2722 files changed, 531548 insertions, 293033 deletions` | - | Broad upstream churn |
| Integration model | `third_party/Megatron-LM` + Primus trainer / adapter / patches | Upstream mainline | Upgrade is more than a submodule bump |
| Integration footprint | `primus/backends/megatron/` has about `569` files; report-covered Primus Megatron directories have about `1243` files | No Primus integration layer | Upgrade blast radius is large |
| Dependency metadata | `pyproject.toml` and `megatron/core/requirements.txt` unchanged vs upstream | Same dependency metadata | Main concern is API / capability drift, not dependency metadata drift |

## Dependency / Package Facts

| Item | Current Megatron-LM in Primus (`4b4acac9`) | Upstream `main` (`037b883c`) |
| --- | --- | --- |
| `pyproject.toml` project dependencies | `torch>=2.6.0`, `numpy`, `packaging>=24.2` | Identical |
| `megatron/core/requirements.txt` | `torch`, `packaging` | Identical |
| Source-declared version source | `megatron/core/package_info.py`: `0.19.2` | `0.21.0` with git SHA suffix logic |
| Canonical Primus pin record | `primus/_thirdparty.lock`: `4b4acac9…` | - |

## Representative Upstream Changes

| Area | Representative changes since `core_v0.19.2` |
| --- | --- |
| Megatron-FSDP (MFSDP v2) | DP-outer gradient-reduction stream overlap, MXFP8-recompute unshard planes, recorded execution order, `fully_shard` on `ShortcutMoEBlock` |
| MoE / Router | Fused Triton batch-invariant unpermute, THD/sequence packing for NCCL EP, bit-exact sigmoid-MoE routing, all-to-all router-prob release |
| SSM / Hybrid (GDP / KDA) | FP32 GDP `A_log`/`dt_bias`, in-place GTP shards, CuTeDSL chunkwise-CP compile, KDA hybrid layers with GLM two-stage gates |
| Inference | FlashInfer MXFP8 expert-weight refit, per-token NVFP4 CuTe-DSL MoE, disaggregated end-to-end inference, prefix-cache scoring |
| Optimizer / training | `OptimizerConfig` migration, RMSNorm-gamma weight decay, mHC precision + full activation recompute, LayerShardedMuon streaming |
| Tests | Broad functional and unit-test changes across MoE, inference, FSDP/MFSDP, SSM, resharding, and training |

## Primus Outer Integration Layer

The Primus Megatron integration layer is broad and directly coupled to upstream internals:

- `primus/backends/megatron/` (trainer, adapters, extensions, patches)
- `primus/configs/models/megatron/`
- `examples/megatron/`
- `tests/unit_tests/backends/megatron/`

Direct upstream dependency paths include:

- `megatron.training.*` (arguments, initialize, pretrain, training, run_config, global_vars)
- `megatron.core.pipeline_parallel.*`
- `megatron.core.extensions.transformer_engine`
- `megatron.core.models.gpt.*` / `megatron.core.models.backends`
- `megatron.core.transformer.moe.*` / `megatron.core.transformer.multi_token_prediction`
- `megatron.core.parallel_state`, `megatron.core.process_groups_config`, `megatron.core.enums`

## Recommendation

**Plan sync.** The 2026-10 bump to `core_v0.19.2` cut the gap from ~1662 commits to 1011 and realigned Primus with a tagged release, but a meaningful gap to `main` (Core `0.21.0`) remains. The largest risks on a further sync are API compatibility in MFSDP v2, MoE/router, SSM/Hybrid (GDP/KDA), inference, and optimizer paths, which overlap with existing Primus patch surfaces (e.g. the main-only DDP `_post_param_sync` fast path is skipped on `0.19.2`).

## Evidence Sources

- `third_party/Megatron-LM/megatron/core/package_info.py`
- `third_party/Megatron-LM/pyproject.toml`
- `third_party/Megatron-LM/megatron/core/requirements.txt`
- `primus/_thirdparty.lock`
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- Primus pin bump: PR [#1321](https://github.com/AMD-AGI/Primus/pull/1321)
- `primus/backends/megatron/*`
- `tests/unit_tests/backends/megatron/*`
