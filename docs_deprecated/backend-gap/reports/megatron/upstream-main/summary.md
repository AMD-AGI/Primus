# Primus Megatron Upstream Gap One-Page Summary

> Date: 2026-04-30

## High-Level Comparison

| Item | Current Megatron-LM in Primus | Upstream `NVIDIA/Megatron-LM` `main` | Impact |
| --- | --- | --- | --- |
| Source-declared Megatron Core version | `0.16.0rc0` from `megatron/core/package_info.py` | `0.18.0` | Spans multiple upstream version steps |
| Pinned commit | `d3528a21` | `0d98cb83` | Current submodule is materially behind |
| Commit gap | Behind by `403` commits | - | Not a small patch-level gap |
| Diff size | `1300 files changed, 615916 insertions, 326835 deletions` | - | Very broad upstream churn |
| Integration model | `third_party/Megatron-LM` + Primus trainer / adapter / patches | Upstream mainline | Upgrade is more than a submodule bump |
| Integration footprint | `primus/backends/megatron/` has about `136` files; report-covered Primus Megatron directories have about `365` files | No Primus integration layer | Upgrade blast radius is large |
| Dependency metadata | `pyproject.toml` and `megatron/core/requirements.txt` unchanged in compared range | Same dependency metadata | Main concern is API / capability drift, not dependency metadata drift |

## Dependency / Package Facts

| Item | Current Megatron-LM in Primus | Upstream `main` |
| --- | --- | --- |
| `pyproject.toml` project dependencies | `torch>=2.6.0`, `numpy`, `packaging>=24.2` | No direct diff |
| `megatron/core/requirements.txt` | `torch`, `packaging` | No direct diff |
| Source-declared version source | `megatron/core/package_info.py`: `0.16.0rc0` | `0.18.0` with git SHA suffix logic |

## Representative Upstream Changes

| Area | Representative changes |
| --- | --- |
| Megatron-FSDP | Documentation refactor; mixed precision, MXFP8, uneven DTensor, frozen parameter, async save, and optimizer fixes |
| MoE / Router | Shared expert overlap, FlexDispatcher support, new router score function, non-quantized dispatch padding fix, A2A combine backprop overlap |
| Hybrid / Mamba | Added `megatron/core/models/hybrid`; renamed Mamba concepts toward Hybrid; added YARN and DeepSeek Sparse Attention paths |
| Inference | CUDA graph support for MTP inference, prefix caching, per-block MoE routing storage, text generation controller reordering |
| MiMo / RL | MiMo communication and optimizer updates; RL agents, inference flows, token throughput and packing metrics |
| Resharding | Updated planner, execution, transforms, and copy services |
| Tests | Broad functional and unit-test changes across MoE, inference, FSDP, SSM, resharding, and training |

## Primus Outer Integration Layer

The Primus Megatron integration layer is broad and directly coupled to upstream internals:

- `primus/backends/megatron/`
- `primus/modules/trainer/megatron/`
- `primus/configs/models/megatron/`
- `examples/megatron/`
- `tests/unit_tests/backends/megatron/`

Direct upstream dependency paths include:

- `megatron.training.*`
- `megatron.core.distributed.*`
- `megatron.core.optimizer.*`
- `megatron.core.transformer.moe.*`
- `megatron.core.extensions.transformer_engine`
- `megatron.core.models.gpt.*`
- `megatron.core.pipeline_parallel.*`

## Recommendation

Treat this as a planned sync, not a trivial dependency bump. The largest risks are API compatibility in trainer initialization, distributed/FSDP paths, MoE/router behavior, inference/Hybrid changes, and Primus patch compatibility.

## Evidence Sources

- `third_party/Megatron-LM/megatron/core/package_info.py`
- `third_party/Megatron-LM/pyproject.toml`
- `third_party/Megatron-LM/megatron/core/requirements.txt`
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- `primus/backends/megatron/*`
- `primus/modules/trainer/megatron/*`
- `tests/unit_tests/backends/megatron/*`
