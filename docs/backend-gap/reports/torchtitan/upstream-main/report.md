# Primus TorchTitan vs Upstream `main` Comparison Report

> Date: 2026-04-21
> Scope: Current TorchTitan bundled in `Primus` vs upstream `pytorch/torchtitan` `origin/main`

## High-Level Comparison

| Item | Current TorchTitan in Primus | Upstream `pytorch/torchtitan` `main` |
| --- | --- | --- |
| Submodule version | `0.1.0` | `0.2.2` |
| Pinned commit | `5fb7cc2e` | `fc54b897` |
| Commit date | 2025-10-15 | 2026-04-20 |
| Commit gap | Behind by `493` commits | - |
| Git relation | `merge-base(HEAD, origin/main) = HEAD` | - |
| Diff size | `447 files changed, 56071 insertions, 17716 deletions` | - |
| Integration model | `third_party/torchtitan` + Primus outer layer (`adapter / trainer / patches`) | Upstream mainline |
| Integration footprint | `primus/backends/torchtitan/` has about `90` files; total across report-covered directories is about `147` files | No Primus integration layer |
| Private submodule commits | None | - |
| Extra private requirements | `requirements-torchtitan.txt` has comments only, with no effective dependency entries | - |

## Torch / TorchAO / Dependency Comparison

### Install Channels and Version Semantics

| Item | Current TorchTitan in Primus | Upstream `main` |
| --- | --- | --- |
| README nightly channel | `nightly/cu126` | `nightly/cu130` |
| Workflow install channel | `nightly/cu126` | `nightly/cu130`; ROCm uses `nightly/rocm7.1` |
| Workflow `torch-version` parameter | No explicit fixed version | Empty string in `set-matrix.yaml` |
| `v0.1.0` release anchor | `torch-2.8.0.dev20250617+cu126` / `torchao-0.12.0.dev20250617+cu126` | - |
| `v0.2.2` release anchor | - | `torch-2.12.0.dev20260220+cu126` / `torchao-0.17.0.dev20260220+cu126` |

### Python Dependency Differences

| Dependency | Current TorchTitan in Primus | Upstream `main` |
| --- | --- | --- |
| `torchdata` | `>=0.8.0` | Explicitly installed as `0.12.0.dev20260327` in workflow |
| `datasets` | `>=2.21.0` | `>=3.6.0`, constrained to `<4.8.0` in CI |
| `tokenizers` | No fixed lower bound | `>=0.15.0` |
| `safetensors` | Not present | Added |
| `wandb` | Dev-only dependency | Moved into runtime dependencies |
| `einops` | Not present | Added |
| `pillow` | Not present | Added |
| `av` | Not present | Added for VLM-related dependencies |
| `torchvision` | Not present | Added in VLM / CPU tests |
| `expecttest` | Pinned at `0.1.6` | `>=0.2.0` |
| `pyrefly` | Not present | `==0.45.1` |
| `numpy` | Not present | Added to dev / CI dependencies |
| `tyro` | No fixed lower bound | Raised to `>=1.0.5` in CI dependencies |
| `tomli` | Present in runtime dependencies | Removed from runtime dependencies |

## Directory and Capability Differences

### Model Directories

The current TorchTitan in Primus still follows an earlier model layout. Upstream `main` has added new model families and shared abstractions:

- Added `torchtitan/models/common/` with shared modules: `attention / decoder / embedding / feed_forward / linear / moe / param_init / rmsnorm / rope / token_dispatcher`
- Added `torchtitan/models/gpt_oss/`
- Added `torchtitan/models/qwen3_vl/`
- Added `torchtitan/models/flux/`; `flux` was moved from `experiments/flux`
- `llama3 / llama4 / qwen3 / deepseek_v3` were reorganized into a more unified shape around `config_registry.py`, `parallelize.py`, and `state_dict_adapter.py`

### `experiments/` Directory

The current TorchTitan in Primus keeps an earlier experiments layout. Upstream `main` added or moved the following:

- Added `torchtitan/experiments/autoparallel/`
- Added `torchtitan/experiments/graph_trainer/`
- Added `torchtitan/experiments/rl/`
- Added `torchtitan/experiments/transformers_modeling_backend/`
- Added `torchtitan/experiments/ft/`; content from `components/ft` moved here
- Main content from `torchtitan/experiments/flux/` moved into `torchtitan/models/flux/`
- Multiple files in `torchtitan/experiments/simple_fsdp/` were removed or moved
- Multiple files in `torchtitan/experiments/torchcomms/` were removed

### `distributed/` and `components/`

The current TorchTitan in Primus keeps older distributed/components layouts. Upstream `main` added or continuously evolved these paths:

- Added `torchtitan/distributed/compile.py`
- Added `torchtitan/distributed/context_parallel.py`
- Added `torchtitan/distributed/deepep/`
- Added `torchtitan/distributed/fsdp.py`
- `torchtitan/distributed/tensor_parallel.py` continuously updated
- `torchtitan/distributed/pipeline_parallel.py` continuously updated
- `torchtitan/distributed/expert_parallel.py` continuously updated
- Added `torchtitan/components/quantization/module_utils.py`
- `torchtitan/components/quantization/float8.py` continuously updated
- `torchtitan/components/quantization/mx.py` continuously updated
- `torchtitan/components/metrics.py` continuously updated
- `torchtitan/components/optimizer.py` continuously updated
- `torchtitan/components/tokenizer.py` continuously updated

## Change Hotspots

| Area | Representative changes |
| --- | --- |
| `tests/unit_tests/` | Broader unit-test coverage |
| `.github/workflows/` | Added `integration_test_4gpu_rl.yaml`, `integration_test_8gpu_autoparallel.yaml`, `integration_test_8gpu_graph_trainer.yaml`, `integration_test_8gpu_rl_h100.yaml`, `integration_test_8gpu_transformers_modeling_backend.yaml`, `set-matrix.yaml` |
| `torchtitan/experiments/graph_trainer/` | Added `compile.py / cudagraph.py / precompile.py / passes.py / storage.py`; covers `llama3 / deepseek_v3 / qwen3`; includes `tests/test_profiler.py` |
| `torchtitan/models/common/` | Added `attention / decoder / embedding / feed_forward / linear / moe / param_init / rmsnorm / rope / token_dispatcher` |
| `docs/` | Added `docs/mxfp8.md`, `docs/bf16_optimizer_states.md`, and updated docs such as `release / debugging / metrics / datasets / checkpoint` |
| `torchtitan/models/flux/` | `flux` moved from `experiments/flux` into `models/flux` |
| `torchtitan/distributed/` | Added `compile.py`, `context_parallel.py`, `deepep/deepep.py`, `deepep/hybridep.py`, `fsdp.py` |
| `torchtitan/experiments/rl/` | Added `actors/`, `models/vllm_wrapper.py`, `simple_grpo_sum_digits.py` |
| `torchtitan/components/` | Added `module_utils`; continuous updates in `float8 / mx / metrics / optimizer / tokenizer / validate` |
| `torchtitan/models/gpt_oss/` | Added `gpt_oss` |
| `torchtitan/models/qwen3_vl/` | Added `qwen3_vl` |
| `torchtitan/experiments/autoparallel/` | Added `llama3`, `deepseek_v3`, `local_map_deepseek_v3` |
| `torchtitan/experiments/transformers_modeling_backend/` | Added transformers-based modeling backend experiments |

## Primus Outer Integration Layer

### Related Directories

The Primus outer integration layer is mainly distributed across:

- `primus/backends/torchtitan/`
- `primus/modules/trainer/torchtitan/`
- `primus/configs/modules/torchtitan/`
- `examples/torchtitan/`
- `runner/helpers/hooks/train/pretrain/torchtitan/`
- `tests/unit_tests/backends/torchtitan/`
- `tests/trainer/test_torchtitan_trainer.py`

### Directly Referenced Upstream Paths

| Primus code location | Direct upstream dependency path |
| --- | --- |
| `primus/modules/trainer/torchtitan/pre_trainer.py` | `torchtitan.config.job_config.JobConfig`, `torchtitan.train.Trainer` |
| `primus/backends/torchtitan/patches/turbo/attention_patches.py` | `torchtitan.models.llama3.model.model`, `torchtitan.models.llama4.model.model`, `torchtitan.models.deepseek_v3.model.model` |
| `primus/backends/torchtitan/patches/turbo/fp8_linear_patches.py` | `torchtitan.components.quantization.float8` |
| `primus/backends/torchtitan/patches/turbo/mx_linear_patches.py` | `torchtitan.components.quantization.mx` |
| `primus/backends/torchtitan/patches/turbo/moe_grouped_mm_patches.py` | `torchtitan.models.moe.moe` |

## Evidence Sources

- `third_party/torchtitan/pyproject.toml`
- `third_party/torchtitan/README.md`
- `third_party/torchtitan/docs/release.md`
- `third_party/torchtitan/.github/workflows/*`
- [pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- [pytorch/torchtitan/docs/release.md](https://github.com/pytorch/torchtitan/blob/main/docs/release.md)
- `primus/backends/torchtitan/*`
- `primus/modules/trainer/torchtitan/*`
- `docs/backends/torchtitan/patch-notes.md`
