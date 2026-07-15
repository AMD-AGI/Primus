# Primus TorchTitan Upstream Gap One-Page Summary

> Date: 2026-04-21

## High-Level Comparison

| Item | Current TorchTitan in Primus | Upstream `pytorch/torchtitan` `main` | Impact |
| --- | --- | --- | --- |
| Submodule version | `0.1.0` | `0.2.2` | Spans multiple upstream release iterations |
| Pinned commit | `5fb7cc2e` | `fc54b897` | Current submodule is significantly behind |
| Commit gap | Behind by `493` commits | - | Not a small patch-level gap anymore |
| Integration model | `third_party/torchtitan` + Primus outer layer (`adapter / trainer / patches`) | Upstream mainline | Upgrade is more than a submodule bump |
| Integration footprint | `primus/backends/torchtitan/` has about `90` files; total across summary-covered directories is about `147` files | No Primus integration layer | Upgrade blast radius is large |
| Private submodule commits | None | - | Git history remains traceable and clean |
| Extra private requirements | `requirements-torchtitan.txt` has comments only, with no effective dependency entries | - | Primus mainly reuses upstream dependencies |
| Torch / TorchAO version semantics | `nightly/cu126`; `v0.1.0` anchors `torch-2.8.0.dev20250617+cu126` / `torchao-0.12.0.dev20250617+cu126` | GitHub current `main` tracks the latest nightly at the time (`CUDA cu130`, `ROCm rocm7.1`) | The key comparison is `cu126` generation vs `cu130/rocm7.1` generation |
| `torchdata` | `>=0.8.0` | `0.12.0.dev20260327` (workflow) | Data stack baseline increased |
| `datasets` | `>=2.21.0` | `>=3.6.0, <4.8.0` | Compatibility and streaming behavior changed |
| `tokenizers` | No fixed lower bound | `>=0.15.0` | Tokenizer baseline increased |
| Runtime added deps | No runtime `safetensors / wandb / einops / pillow` | Added | Runtime environment is heavier |
| Multimodal added deps | No `av / torchvision` | Added | Multimodal environment requirements increased |
| New upstream capabilities | Existing training main path | `graph_trainer / autoparallel / rl / qwen3_vl / gpt_oss` | Upstream capability surface is much broader |

## Torch / TorchAO / Dependencies

| Item | Current TorchTitan in Primus | Upstream `main` |
| --- | --- | --- |
| README nightly channel | `nightly/cu126` | `nightly/cu130` |
| Workflow install channel | `nightly/cu126` | `nightly/cu130`; ROCm uses `nightly/rocm7.1` |
| Workflow `torch-version` parameter | No explicit fixed version | Empty string in `set-matrix.yaml` |
| `v0.1.0` release anchor | `torch-2.8.0.dev20250617+cu126` / `torchao-0.12.0.dev20250617+cu126` | - |
| `torchdata` | `>=0.8.0` | Explicitly installed as `0.12.0.dev20260327` in workflow |
| `datasets` | `>=2.21.0` | `>=3.6.0, <4.8.0` |
| `tokenizers` | No fixed lower bound | `>=0.15.0` |
| `safetensors` | No | Added |
| `wandb` | Dev-only dependency | Moved into runtime dependencies |
| `einops` | No | Added |
| `pillow` | No | Added |
| `av` | No | Added |
| `torchvision` | No | Added |

## Representative Upstream Changes

| Area | Representative changes |
| --- | --- |
| `models/` | Added `gpt_oss`, `qwen3_vl`; moved `flux` into `models/flux`; added shared `models/common` layer |
| `distributed/` | Added `compile.py`, `context_parallel.py`, `deepep/`, `fsdp.py` |
| `experiments/` | Added `autoparallel`, `graph_trainer`, `rl`, `transformers_modeling_backend`, `ft` |
| `components/` | Added `module_utils`; continuous updates in `float8 / mx / metrics / optimizer / tokenizer / validate` |
| `.github/workflows/` | Added `integration_test_4gpu_rl.yaml`, `integration_test_8gpu_autoparallel.yaml`, `integration_test_8gpu_graph_trainer.yaml`, `integration_test_8gpu_transformers_modeling_backend.yaml`, `set-matrix.yaml` |
| `docs/` | Added `docs/mxfp8.md`, `docs/bf16_optimizer_states.md`, and updated docs such as `release / debugging / metrics / datasets / checkpoint` |

## Primus Outer Integration Layer

The Primus outer integration layer is mainly distributed across:

- `primus/backends/torchtitan/`
- `primus/modules/trainer/torchtitan/`
- `primus/configs/modules/torchtitan/`
- `examples/torchtitan/`
- `runner/helpers/hooks/train/pretrain/torchtitan/`
- `tests/unit_tests/backends/torchtitan/`
- `tests/trainer/test_torchtitan_trainer.py`

Directly referenced upstream paths include:

- `torchtitan.config.job_config.JobConfig`
- `torchtitan.train.Trainer`
- `torchtitan.models.llama3.model.model`
- `torchtitan.models.llama4.model.model`
- `torchtitan.models.deepseek_v3.model.model`
- `torchtitan.components.quantization.float8`
- `torchtitan.components.quantization.mx`
- `torchtitan.models.moe.moe`

## Evidence Sources

- `third_party/torchtitan/README.md`
- `third_party/torchtitan/docs/release.md`
- `third_party/torchtitan/.github/workflows/*`
- [pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- [pytorch/torchtitan/docs/release.md](https://github.com/pytorch/torchtitan/blob/main/docs/release.md)
