# Primus TorchTitan vs Upstream `main` Comparison Report

> Date: 2026-08-03
> Scope: Current TorchTitan bundled in `Primus` vs upstream `pytorch/torchtitan` `origin/main`

## High-Level Comparison

| Item | Current TorchTitan in Primus | Upstream `pytorch/torchtitan` `main` |
| --- | --- | --- |
| Submodule version (`assets/version.txt`) | `0.2.2` | `0.2.2` (dev toward next release) |
| Pinned commit | `73a0e697` (**v0.2.2** tag) | `681fd4b5` |
| Commit date | 2026-02-20 | 2026-08-01 |
| Commit gap | Behind by `741` commits | - |
| Git relation | `merge-base(HEAD, origin/main) = HEAD` (pin is ancestor of upstream main) | - |
| Diff size | `698 files changed, 123580 insertions, 35965 deletions` | - |
| Integration model | `third_party/torchtitan` + Primus outer layer (`adapter / trainer / patches`) | Upstream mainline |
| Integration footprint | `primus/backends/torchtitan/` has about `57` files; total torchtitan-related Python across the Primus tree is about `70` files | No Primus integration layer |
| Private submodule commits | None (pin is the upstream `v0.2.2` tag commit) | - |

> **Change since the previous report (2026-04-21).** Primus advanced the
> torchtitan pin from `5fb7cc2e` (v0.1.0, 2025-10-15) to `73a0e697` (the tagged
> **v0.2.2** release, 2026-02-20) in Primus PR #871 (merged 2026-07-17). The pin
> now sits on a maintained tagged release rather than an unreleased mainline
> commit, but upstream `main` has since advanced `741` commits past v0.2.2.

## Torch / TorchAO / Dependency Comparison

### Install Channels and Version Semantics

| Item | Current TorchTitan in Primus (`v0.2.2`) | Upstream `main` |
| --- | --- | --- |
| `assets/version.txt` | `0.2.2` | `0.2.2` (not yet bumped for the next tag) |
| Release anchor | `v0.2.2` tag (`torch-2.12.0.dev20260220+cu126` / `torchao-0.17.0.dev20260220+cu126`) | Tracks the latest nightly at HEAD time |
| Nightly channel trajectory | `cu126` generation at the v0.2.2 tag | Upstream `main` continues to advance nightly `torch` / `torchao` and ROCm channels |

> Note: because Primus is now pinned to the upstream `v0.2.2` **tag**, the
> torch / torchao dependency semantics at the pin match upstream's own v0.2.2
> release anchors. The remaining gap is the `741` commits upstream `main` has
> merged since that tag.

### Python Dependency Notes

- Primus does not add effective private dependency entries for torchtitan beyond
  the upstream package; the outer integration layer reuses upstream torchtitan
  dependencies.
- Upstream `main` has continued to evolve its dependency and CI matrix since
  v0.2.2 (tokenizer download support for newer `transformers` / `hub`, expanded
  test dependencies). These changes are part of the `741`-commit gap.

## Directory and Capability Differences (upstream `main` since the `v0.2.2` pin)

### Model Directories

- Added `torchtitan/models/kimi_k2_7/` (Kimi K2, upstream #3532)
- Added `torchtitan/models/qwen3_5/` (Qwen 3.5)
- Added `torchtitan/models/gpt_oss/` and GPT-OSS enablement under `spmd_types`
- Added shared `torchtitan/models/common/` abstractions and moved `flux` into `torchtitan/models/flux/`
- `deepseek_v3` gained mxfp8 debug config and compile fixes

### `experiments/` Directory

- Added `torchtitan/experiments/graph_trainer/` (GraphPP runner, EP overlap/chunking passes, FSDP collective splitting, dI/dW backward splitting)
- Added `torchtitan/experiments/rl/` (GRPO/DAPO examples, cudagraph knobs, `DPRequestRouter`, entropy/perf metrics)
- Added `torchtitan/experiments/transformers_modeling_backend/` (MoE support #2679, SFT #3243)
- Added `torchtitan/experiments/torchft/` (fault-tolerant training)

### `distributed/` and `components/`

- Upgraded DeepEP to **v2 APIs** enabling cudagraphable mode (#3808); added `deepep/hybridep.py`
- Added `torchtitan/distributed/minimal_async_ep/` async-EP kernels (int32 overflow fix)
- Added `torchtitan/distributed/compile.py`, `fsdp.py`, `full_dtensor.py`, `spmd_types.py`
- Removed legacy `torchtitan/distributed/expert_parallel.py` and `dual_pipe_v.py`
- Continuous updates to `tensor_parallel.py`, `pipeline_parallel.py`, `context_parallel.py`, `activation_checkpoint.py`

## Change Hotspots

| Area | Representative changes since the `v0.2.2` pin |
| --- | --- |
| `torchtitan/models/` | Added `kimi_k2_7`, `qwen3_5`; GPT-OSS enablement; shared `models/common` layer |
| `torchtitan/experiments/graph_trainer/` | GraphPP runner integration, EP overlap/chunking passes, FSDP collective splitting |
| `torchtitan/experiments/rl/` | GRPO/DAPO examples, cudagraph capture knobs, `DPRequestRouter`, batch-invariant FSDP |
| `torchtitan/distributed/` | DeepEP v2 cudagraphable APIs, `minimal_async_ep` kernels, `compile.py`, `fsdp.py`, `full_dtensor.py`, `spmd_types.py` |
| `torchtitan/components/quantization/` | mxfp8 MoE `ep=1` enablement (#3935); continued float8 / mx updates |
| Checkpoint | Remote fsspec checkpoint paths (e.g. `gs://`) via filesystem helpers (#3887) |
| `.github/workflows/` | AMD 8-GPU-feature CI fixes, AutoParallel device-agnostic tests |

## Primus Outer Integration Layer

### Related Directories

The Primus outer integration layer is mainly distributed across:

- `primus/backends/torchtitan/` (adapter, argument builder, models, patches, primus-turbo extensions)
- `primus/backends/torchtitan/torchtitan_pretrain_trainer.py`
- `primus/configs/modules/torchtitan/`
- `examples/torchtitan/`
- `tests/trainer/test_torchtitan_trainer.py`
- `tests/unit_tests/modules/trainer/torchtitan/`

> The prior report's `primus/modules/trainer/torchtitan/` path no longer exists:
> Primus PR #851 (merged 2026-07-08) removed `primus/modules` and migrated
> still-used code into `primus/core` and `primus/backends`. The torchtitan
> trainer now lives at `primus/backends/torchtitan/torchtitan_pretrain_trainer.py`.

### Directly Referenced Upstream Paths (verified on Primus `main`)

| Primus code location | Direct upstream dependency path |
| --- | --- |
| `primus/backends/torchtitan/torchtitan_pretrain_trainer.py` | `torchtitan.train.Trainer`, `torchtitan.tools.logging` |
| `primus/backends/torchtitan/patches/turbo/attention_patches.py` | `torchtitan.models.llama3.model.model`, `torchtitan.models.llama4.model.model`, `torchtitan.models.deepseek_v3.model.model`, `torchtitan.models.qwen3.model.model` |
| `primus/backends/torchtitan/patches/turbo/fp8_linear_patches.py` | `torchtitan.components.quantization.float8`, `torchtitan.protocols.model_converter` |
| `primus/backends/torchtitan/patches/turbo/mx_linear_patches.py` | `torchtitan.components.quantization.mx`, `torchtitan.protocols.model_converter` |
| `primus/backends/torchtitan/patches/turbo/moe_grouped_mm_patches.py` | `torchtitan.models.moe.moe` |

Additional coupling added since the prior report includes GPT-OSS sink-attention
and DeepSeek-V3 classic-attention turbo patches, async-TP patches, and
`primus_turbo_extensions` model converters — all under
`primus/backends/torchtitan/patches/` and `primus/backends/torchtitan/primus_turbo_extensions/`.

## Evidence Sources

- `third_party/torchtitan` submodule pin in Primus `main` (`73a0e697`, v0.2.2 tag)
- `third_party/torchtitan/assets/version.txt` (`0.2.2` at both pin and upstream main)
- [pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- [pytorch/torchtitan `v0.2.2` tag](https://github.com/pytorch/torchtitan/releases/tag/v0.2.2)
- [compare `73a0e697...main`](https://github.com/pytorch/torchtitan/compare/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce...main)
- `primus/backends/torchtitan/*` (adapter, trainer, patches, primus-turbo extensions)
- Primus PR [#871](https://github.com/AMD-AGI/Primus/pull/871) (torchtitan v0.2.2 bump), PR [#851](https://github.com/AMD-AGI/Primus/pull/851) (primus/modules removal)
