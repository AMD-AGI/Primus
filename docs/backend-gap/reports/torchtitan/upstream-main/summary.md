# Primus TorchTitan Upstream Gap One-Page Summary

> Date: 2026-08-03

## High-Level Comparison

| Item | Current TorchTitan in Primus | Upstream `pytorch/torchtitan` `main` | Impact |
| --- | --- | --- | --- |
| Submodule version | `0.2.2` (v0.2.2 tag) | `0.2.2` (dev toward next release) | Pin is now on a maintained tagged release |
| Pinned commit | `73a0e697` | `681fd4b5` | Pin advanced from `5fb7cc2e`/v0.1.0 in Primus PR #871 (2026-07-17) |
| Commit gap | Behind by `741` commits | - | Still a large gap, but from a tagged anchor |
| Diff size | `698 files, +123580 / -35965` | - | Upgrade to newer main is a substantial diff |
| Integration model | `third_party/torchtitan` + Primus outer layer (`adapter / trainer / patches`) | Upstream mainline | Upgrade is more than a submodule bump |
| Integration footprint | `primus/backends/torchtitan/` ~`57` files; ~`70` torchtitan-related Python files across tree | No Primus integration layer | Upgrade blast radius is moderate-to-large |
| New upstream capabilities | Existing v0.2.2 training path | `kimi_k2_7 / qwen3_5 / gpt_oss`, `graph_trainer / rl / torchft`, DeepEP v2 | Upstream capability surface is broader |

## What Changed Since the Previous Report (2026-04-21)

- Primus bumped the torchtitan pin from `5fb7cc2e` (v0.1.0, 2025-10-15) to
  `73a0e697` (the tagged **v0.2.2** release, 2026-02-20) in PR #871.
- The pin now sits on a maintained tagged release instead of an unreleased
  mainline commit; torch/torchao anchors at the pin match upstream v0.2.2.
- The remaining gap is upstream `main` advancing `741` commits past v0.2.2.

## Representative Upstream Changes (since the `v0.2.2` pin)

| Area | Representative changes |
| --- | --- |
| `models/` | Added `kimi_k2_7` (#3532), `qwen3_5`; GPT-OSS enablement; shared `models/common` layer |
| `distributed/` | DeepEP v2 cudagraphable APIs (#3808), `minimal_async_ep`, `compile.py`, `fsdp.py`, `full_dtensor.py`, `spmd_types.py` |
| `experiments/` | `graph_trainer` (GraphPP, EP overlap), `rl` (GRPO/DAPO), `transformers_modeling_backend` (MoE #2679, SFT #3243), `torchft` |
| `components/quantization/` | mxfp8 MoE `ep=1` (#3935); continued float8 / mx updates |
| Checkpoint / CI | Remote fsspec checkpoint paths `gs://` (#3887); AMD 8-GPU CI + AutoParallel device-agnostic tests |

## Primus Outer Integration Layer

The Primus outer integration layer is mainly distributed across:

- `primus/backends/torchtitan/` (adapter, argument builder, models, patches, primus-turbo extensions)
- `primus/backends/torchtitan/torchtitan_pretrain_trainer.py`
- `primus/configs/modules/torchtitan/`
- `examples/torchtitan/`
- `tests/trainer/test_torchtitan_trainer.py`

Directly referenced upstream paths (verified on Primus `main`):

- `torchtitan.train.Trainer`, `torchtitan.tools.logging`
- `torchtitan.models.llama3.model.model`, `torchtitan.models.llama4.model.model`, `torchtitan.models.deepseek_v3.model.model`, `torchtitan.models.qwen3.model.model`
- `torchtitan.components.quantization.float8`, `torchtitan.components.quantization.mx`
- `torchtitan.protocols.model_converter`
- `torchtitan.models.moe.moe`

> Note: `primus/modules/trainer/torchtitan/` no longer exists; PR #851 migrated
> the trainer into `primus/backends/torchtitan/torchtitan_pretrain_trainer.py`.

## Evidence Sources

- `third_party/torchtitan` pin (`73a0e697`, v0.2.2) and `assets/version.txt`
- [pytorch/torchtitan](https://github.com/pytorch/torchtitan) · [v0.2.2 tag](https://github.com/pytorch/torchtitan/releases/tag/v0.2.2)
- [compare `73a0e697...main`](https://github.com/pytorch/torchtitan/compare/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce...main)
- Primus PR [#871](https://github.com/AMD-AGI/Primus/pull/871), PR [#851](https://github.com/AMD-AGI/Primus/pull/851)
