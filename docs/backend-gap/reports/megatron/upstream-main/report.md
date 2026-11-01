# Primus Megatron vs Upstream `main` Comparison Report

> Date: 2026-11-01
> Scope: Current Megatron-LM bundled in `Primus` vs upstream `NVIDIA/Megatron-LM` `origin/main`

## High-Level Comparison

| Item | Current Megatron-LM in Primus | Upstream `NVIDIA/Megatron-LM` `main` |
| --- | --- | --- |
| Source-declared Megatron Core version | `0.19.2` from `megatron/core/package_info.py` | `0.21.0` from `origin/main:megatron/core/package_info.py` |
| Pinned commit | `4b4acac9` (`core_v0.19.2` release tag) | `037b883c` |
| Commit date | 2026-09-18 | 2026-11-01 |
| Commit gap | Behind by `1011` commits | - |
| Git relation | Pin is a release tag, not a `main` ancestor: `ahead_by = 1011`, `behind_by = 28` (merge-base `818b2580`) | - |
| Diff size | `2722 files changed, 531548 insertions, 293033 deletions` | - |
| Integration model | `third_party/Megatron-LM` + Primus outer trainer / adapter / patches | Upstream mainline |
| Integration footprint | `primus/backends/megatron/` has about `569` files; report-covered Primus Megatron directories have about `1243` files | No Primus integration layer |
| Private submodule commits | None (upstream NVIDIA tag) | - |

The Primus pin was bumped during the 2026-10 window (PR [#1321](https://github.com/AMD-AGI/Primus/pull/1321), merged 2026-10-29) from `d3528a21` (Core `0.16.0rc0`) to the `core_v0.19.2` release tag `4b4acac9`. Because `4b4acac9` is a release tag rather than a `main` ancestor, it carries 28 release-only commits that are not on `main`; the authoritative behind-count against `main` is `1011` commits.

## Dependency and Package Metadata Comparison

| Item | Current Megatron-LM in Primus (`4b4acac9`) | Upstream `main` (`037b883c`) |
| --- | --- | --- |
| `pyproject.toml` project dependencies | `torch>=2.6.0`, `numpy`, `packaging>=24.2` | `torch>=2.6.0`, `numpy`, `packaging>=24.2` (unchanged) |
| `megatron/core/requirements.txt` | `torch`, `packaging` | `torch`, `packaging` (unchanged) |
| Source-declared version source | `megatron/core/package_info.py`: `0.19.2` | `0.21.0` with git SHA suffix logic |
| Canonical Primus pin record | `primus/_thirdparty.lock`: `4b4acac9a1d28ea6829c8d4f566d75698a21249d` | - |

Package dependency metadata is unchanged between the pin and upstream `main`: the project `dependencies` list and `megatron/core/requirements.txt` are identical. The primary drift is upstream API and capability movement (Core `0.19.2` → `0.21.0`), not dependency-metadata drift.

## Directory and Capability Differences

### Megatron-FSDP (MFSDP v2) and Distributed Runtime

Upstream `main` has continued heavy MFSDP v2 work since `core_v0.19.2`:

- Overlap the DP-outer gradient reduction on its own stream (#6973).
- Add `SchedulePolicy.unshard_all_planes_in_backward` for MXFP8 recomputation in MFSDP v2 (#8432).
- Record MFSDP execution order for prefetch (#8478) and group execution settings in `SchedulePolicy` (#8585).
- Wrap each `ShortcutMoEBlock` with `fully_shard` (#8312) and harden DDP gradient-scaling asserts (#8579).

### MoE, Router, and Expert Parallelism

The upstream gap includes active MoE and expert-parallel changes:

- Fused Triton kernel for the batch-invariant MoE unpermute (#8559).
- THD and sequence packing support for NCCL EP (#8458).
- Bit-exact sigmoid-MoE inference routing optimization (#8296).
- Release router probs in the all-to-all dispatcher after combine (#8297).

### SSM / Hybrid (GDP, KDA)

Upstream `main` has expanded the SSM/Hybrid surface:

- Keep GDP `A_log` and `dt_bias` in FP32 under BF16 training (#8528).
- Save GDP `in_proj` GTP shards in place instead of gathering them (#8530).
- Compile the CuTeDSL GDP chunkwise-CP boundary variants at the first forward (#8420).
- Add KDA (Kimi Delta Attention) hybrid layers with GLM-style two-stage gates (#7054) and first-class `HybridModel` config lists in Core (#7555).

### Inference and Resharding

Upstream has broad inference and resharding movement:

- Keep one copy of FlashInfer MXFP8 expert weights by refitting into the routed layout (#8560).
- Per-token NVFP4 squared-ReLU and SwiGLU MoE via FlashInfer CuTe DSL (#8262).
- Disaggregated inference end-to-end (#6675) and prefix-cache prompt scoring (#8426).
- Spread resharding refit sources across replicas sharing one DP group (#8502).

### Optimizer and Training

- Migrate `OptimizerConfig` (#8213).
- Add configurable weight decay for RMSNorm gamma (#8246).
- Add mHC precision controls, mean output contract, and full activation recompute with mHC (#7521).
- Stream local Newton-Schulz batches in `LayerShardedMuon` (#8379).

## Change Hotspots

| Area | Representative changes |
| --- | --- |
| `megatron/core/distributed/` | MFSDP v2 stream overlap, MXFP8-recompute unshard planes, recorded execution order, DDP assert hardening |
| `megatron/core/transformer/` | Triton fused unpermute, NCCL EP THD/packing, sigmoid-MoE routing, schedule-node activation ownership |
| `megatron/core/ssm/` | FP32 GDP `A_log`/`dt_bias`, in-place GTP shards, CuTeDSL chunkwise-CP compile, KDA hybrid layers |
| `megatron/core/inference/` | FlashInfer MXFP8 refit, NVFP4 CuTe-DSL MoE, disaggregated inference, prefix-cache scoring |
| `megatron/core/resharding/` | Refit source spreading, NCCL M2N backend docs, refit batching |
| `megatron/core/optimizer/` | `OptimizerConfig` migration, RMSNorm-gamma weight decay, mHC precision, LayerShardedMuon streaming |
| `tests/` | Broad functional and unit-test churn across MoE, inference, FSDP/MFSDP, SSM, resharding, and training |

## Primus Outer Integration Layer

### Related Directories

The Primus outer integration layer is mainly distributed across:

- `primus/backends/megatron/` (about 569 files)
- `primus/configs/models/megatron/` (about 104 files)
- `examples/megatron/` (about 407 files)
- `tests/unit_tests/backends/megatron/` (about 163 files)

Note: the legacy `primus/modules/trainer/megatron/` tree no longer exists; the Megatron trainer now lives under `primus/backends/megatron/` (`megatron_base_trainer.py`, `megatron_pretrain_trainer.py`).

### Directly Referenced Upstream Paths

| Primus code location | Direct upstream dependency path |
| --- | --- |
| `primus/backends/megatron/megatron_base_trainer.py` | `megatron.training.arguments`, `megatron.training.initialize`, `megatron.training.global_vars`, `megatron.training.mlflow_setup` |
| `primus/backends/megatron/megatron_pretrain_trainer.py` | `megatron.training.pretrain`, `megatron.training.training`, `megatron.training.run_config`, `megatron.core.pipeline_parallel`, `megatron.core.num_microbatches_calculator`, `megatron.core.process_groups_config` |
| `primus/backends/megatron/core/extensions/transformer_engine_spec_provider.py` | `megatron.core.extensions.transformer_engine`, `megatron.core.models.backends.BackendSpecProvider`, `megatron.core.transformer.moe.experts`, `megatron.core.fusions.fused_layer_norm` |
| `primus/backends/megatron/core/extensions/primus_turbo.py` | `megatron.core.enums` (`Fp4Recipe`, `Fp8Recipe`), `megatron.core.extensions.transformer_engine`, `megatron.core.parallel_state`, `megatron.core.process_groups_config`, `megatron.core.transformer.enums` |
| `primus/backends/megatron/patches/turbo/te_spec_provider_patches.py` | `megatron.core.extensions`, `megatron.core.models.gpt.gpt_layer_specs`, `megatron.core.models.gpt.moe_module_specs`, `megatron.core.transformer.multi_token_prediction` |
| `primus/backends/megatron/patches/ddp_post_param_sync_patches.py` | Patches Megatron-LM DDP `_post_param_sync` (main-only fast path; skipped on `core_v0.19.2`) |

## Evidence Sources

- `third_party/Megatron-LM/megatron/core/package_info.py` (at `4b4acac9` and `origin/main`)
- `third_party/Megatron-LM/pyproject.toml`
- `third_party/Megatron-LM/megatron/core/requirements.txt`
- `primus/_thirdparty.lock`
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (`core_v0.19.2` tag `4b4acac9`; `main` HEAD `037b883c`)
- Primus pin bump: PR [#1321](https://github.com/AMD-AGI/Primus/pull/1321); MegaMoE compat follow-up PR [#1340](https://github.com/AMD-AGI/Primus/pull/1340)
- `primus/backends/megatron/*`
- `primus/configs/models/megatron/*`
- `examples/megatron/*`
- `tests/unit_tests/backends/megatron/*`
