# Primus Megatron vs Upstream `main` Comparison Report

> Date: 2026-04-30
> Scope: Current Megatron-LM bundled in `Primus` vs upstream `NVIDIA/Megatron-LM` `origin/main`

## High-Level Comparison

| Item | Current Megatron-LM in Primus | Upstream `NVIDIA/Megatron-LM` `main` |
| --- | --- | --- |
| Source-declared Megatron Core version | `0.16.0rc0` from `megatron/core/package_info.py` | `0.18.0` from `origin/main:megatron/core/package_info.py` |
| Pinned commit | `d3528a21` | `0d98cb83` |
| Commit date | 2026-03-06 | 2026-04-29 |
| Commit gap | Behind by `403` commits | - |
| Git relation | `merge-base(HEAD, origin/main) = HEAD` | - |
| Diff size | `1300 files changed, 615916 insertions, 326835 deletions` | - |
| Integration model | `third_party/Megatron-LM` + Primus outer trainer / adapter / patches | Upstream mainline |
| Integration footprint | `primus/backends/megatron/` has about `136` files; report-covered Primus Megatron directories have about `365` files | No Primus integration layer |
| Private submodule commits | None | - |

## Dependency and Package Metadata Comparison

| Item | Current Megatron-LM in Primus | Upstream `main` |
| --- | --- | --- |
| `pyproject.toml` project dependencies | `torch>=2.6.0`, `numpy`, `packaging>=24.2` | No diff in the compared range |
| `megatron/core/requirements.txt` | `torch`, `packaging` | No diff in the compared range |
| Source-declared version source | `megatron/core/package_info.py`: `0.16.0rc0` | `0.18.0` with git SHA suffix logic |
| Dev dependency groups | Present in `pyproject.toml` | No direct diff in the compared range |
| Workflow / docs surface | Existing CI and docs | About `42` changed entries across selected workflow/docs paths |

## Directory and Capability Differences

### Megatron-FSDP and Distributed Runtime

Upstream `main` has continued Megatron-FSDP and distributed-runtime work:

- Unified and refactored Megatron-FSDP documentation.
- Fixed FusedAdam `use_decoupled_grad` handling in Megatron-FSDP.
- Added and fixed mixed-precision / MXFP8 / uneven DTensor / frozen parameter paths.
- Added support around DCP and FSDP async save.
- Refined parameter layout, all-gather / reduce-scatter overlap, and precision-aware optimizer behavior.

### MoE, Router, and Expert Parallelism

The upstream gap includes active MoE and expert-parallel changes:

- Improved shared expert overlap and FlexDispatcher support.
- Added a new router score function.
- Added NVFP4 native weights for DDP.
- Fixed non-quantized MoE dispatch padding.
- Added overlap for A2A combine backprop with wgrad GEMM.
- Added broader MoE inference and prefix-cache related updates.

### Hybrid / Mamba and Inference

Upstream `main` has expanded the Hybrid/Mamba and inference surface:

- Added `megatron/core/models/hybrid/`.
- Renamed Mamba model / stack concepts toward Hybrid naming.
- Added YARN support and DeepSeek Sparse Attention paths for Hybrid/Mamba work.
- Added CUDA graph support for MTP inference and prefix caching.
- Added per-block MoE routing storage for prefix caching.
- Reorganized order of operations in inference context and text generation controller.

### MiMo, RL, and Resharding

Upstream has broader non-core-training surface area:

- Updated `examples/mimo/` and `megatron/core/models/mimo/`.
- Added `ColocatedBridgeCommunicator` and `MimoOptimizer` related paths.
- Updated RL agents, token throughput / packing metrics, and RL inference flows.
- Added and evolved `megatron/core/resharding/` APIs and copy services.

## Change Hotspots

| Area | Representative changes |
| --- | --- |
| `megatron/core/distributed/fsdp/` | Megatron-FSDP fixes, docs, DTensor conversion, mixed precision, MXFP8, async save |
| `megatron/core/transformer/moe/` | Shared expert overlap, router score function, MoE dispatch fixes |
| `megatron/core/inference/` | CUDA graph / prefix caching / hybrid inference / text generation controller updates |
| `megatron/core/models/hybrid/` | Added Hybrid model/block/layer allocation/spec files |
| `megatron/core/models/mimo/` | MiMo communication, config, optimizer, and submodule updates |
| `megatron/core/resharding/` | Added / evolved resharding planner, execution, transforms, and copy services |
| `examples/rl/` and `megatron/rl/` | RL agents, inference API, packing metrics, and reward fixes |
| `tests/` | Broad functional and unit-test churn across MoE, inference, FSDP, SSM, resharding, and training |

## Primus Outer Integration Layer

### Related Directories

The Primus outer integration layer is mainly distributed across:

- `primus/backends/megatron/`
- `primus/modules/trainer/megatron/`
- `primus/configs/models/megatron/`
- `examples/megatron/`
- `tests/unit_tests/backends/megatron/`

### Directly Referenced Upstream Paths

| Primus code location | Direct upstream dependency path |
| --- | --- |
| `primus/modules/trainer/megatron/trainer.py` | `megatron.core.distributed`, `megatron.core.optimizer`, `megatron.training.*`, dataset builders, checkpointing, initialization, parallel state |
| `primus/modules/trainer/megatron/utils.py` | `megatron.core.parallel_state`, `megatron.training.global_vars`, pipeline-parallel schedules, transformer block/layer helpers |
| `primus/backends/megatron/megatron_base_trainer.py` | `megatron.training.arguments`, `megatron.training.initialize` |
| `primus/backends/megatron/core/extensions/transformer_engine_spec_provider.py` | `megatron.core.extensions.transformer_engine`, `megatron.core.transformer.moe.experts`, `megatron.core.models.backends` |
| `primus/backends/megatron/core/extensions/primus_turbo.py` | `megatron.core.tensor_parallel`, `megatron.core.transformer.moe.token_dispatcher`, `megatron.core.transformer.transformer_config`, `megatron.training.global_vars` |
| `primus/backends/megatron/patches/turbo/te_spec_provider_patches.py` | `megatron.core.extensions`, `megatron.core.models.gpt.gpt_layer_specs`, `megatron.core.models.gpt.moe_module_specs`, `megatron.core.transformer.multi_token_prediction` |
| `primus/backends/megatron/patches/muon_optimizer_patches.py` | `megatron.training.training.get_megatron_optimizer` namespace |

## Evidence Sources

- `third_party/Megatron-LM/megatron/core/package_info.py`
- `third_party/Megatron-LM/pyproject.toml`
- `third_party/Megatron-LM/megatron/core/requirements.txt`
- `third_party/Megatron-LM/README.md`
- `third_party/Megatron-LM/.github/workflows/*`
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- `primus/backends/megatron/*`
- `primus/modules/trainer/megatron/*`
- `primus/configs/models/megatron/*`
- `examples/megatron/*`
- `tests/unit_tests/backends/megatron/*`
