# Environment Variables Reference

This document catalogs environment variables you may encounter when running Primus on AMD GPUs: distributed launchers, Primus runners and CLI, YAML substitution, libraries (NCCL/RCCL, ROCm, PyTorch, JAX), and optional integrations (Hugging Face, WandB, MLflow).

**Legend**

- **Required**: Must be set for the stated workflow; otherwise the job fails or mis-ranks.
- **Optional**: Has a safe default or is only needed for specific features.
- **Set by**: Typical source (launcher, `runner/helpers/envs/*.sh`, user shell, container host).
- **Used in**: Representative Primus paths; many variables are also read by NVIDIA NCCL, AMD RCCL, PyTorch, or JAX without Primus wrapping them.

---

## 1. PyTorch distributed

Set by `torchrun`, Slurm launchers, or `runner/primus-cli-direct.sh` / `runner/primus-cli-slurm-entry.sh`. Consumed by PyTorch distributed, RCCL, and Primus helpers.

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `MASTER_ADDR` | `localhost` (direct / `base_env.sh`) | User, Slurm entry (`primus-cli-slurm-entry.sh`), or validation fallback (`runner/lib/validation.sh`) | `primus/pretrain.py`, `primus/modules/base_module.py`, `primus/core/utils/env.py`, `primus/tools/preflight/network/network_probe.py`, PyTorch rendezvous | Rendezvous hostname or IP for process group initialization. **Required** for multi-node if not using Slurm auto-detection. |
| `MASTER_PORT` | `1234` (direct), `29500` in some Python defaults | Config / CLI / user | Same as `MASTER_ADDR`; `validation.sh` enforces 1024–65535 | TCP port for the store backing `torch.distributed`. |
| `RANK` | `0` if unset in helpers | `torchrun` | `primus/tools/utils.py`, `primus/tools/preflight/global_vars.py`, projection and profiler code | Global rank index. |
| `WORLD_SIZE` | `1` | `torchrun` | Preflight, projection, `primus/modules/base_module.py` | Total number of processes. |
| `LOCAL_RANK` | `0` | `torchrun` | `primus/modules/base_module.py`, GPU selection in benchmarks and trainers | GPU index on this node. |
| `LOCAL_WORLD_SIZE` | `1` (Python) / `8` in benchmarks default | `torchrun` | `primus/tools/preflight/*.py`, `strided_allgather_bench.py` | Processes (GPUs) per node. |
| `NODE_RANK` | `0` | `primus-cli-direct` / `primus-cli-slurm-entry.sh` | `primus/pretrain.py`, logging in `runner/lib/common.sh` | Zero-based node index in multi-node jobs. |
| `NNODES` | `1` | Direct config (`runner/.primus.yaml`), `primus-cli-slurm-entry.sh` | `primus/pretrain.py`, `primus/core/projection/training_config.py` | Number of nodes in the job. |
| `GPUS_PER_NODE` | `8` | `runner/.primus.yaml` direct section, `primus-cli-slurm-entry.sh`, `validation.sh` | `primus/core/projection/module_profilers/*.py`, training config helpers | GPUs per node for world-size math and binding. |

---

## 2. Primus core

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `PRIMUS_TRAIN_RUNTIME` | (empty → `"core"`) | User | `primus/cli/subcommands/train.py` | Selects pretrain runtime: `"core"` (default) or `"legacy"`. Invalid values are ignored with a stderr warning. |
| `PRIMUS_PATCHES` | `""` / `"all"` | User | `primus/core/patches/patch_runner.py` | `"all"` or empty enables all patches; `"none"` disables; comma list enables subset. |
| `PRIMUS_LOG_LEVEL` | `INFO` | User; debug paths in `runner/primus-cli-*.sh` set `DEBUG` | `runner/lib/common.sh` | Log verbosity: `DEBUG`, `INFO`, `WARN`, `ERROR`. |
| `PRIMUS_LOG_TIMESTAMP` | `1` | User | `runner/lib/common.sh` | `1` prefixes logs with timestamps; `0` disables. |
| `PRIMUS_LOG_COLOR` | `1` (auto-off if not a TTY) | User; tests may set `0` | `runner/lib/common.sh` | ANSI colors in runner logs. |
| `PRIMUS_DEBUG` | `0` | User | `runner/helpers/envs/primus-env.sh` | `1` enables `set -x` in the env loader for shell tracing. |
| `PRIMUS_SKIP_VALIDATION` | `0` | User / tests | `runner/helpers/envs/primus-env.sh` | `1` skips `validate_distributed_params` (not recommended). |
| `PRIMUS_EXPECT_IB` | (unset) | User | `primus/tools/preflight/network/network_standard.py` | When `1`, preflight treats InfiniBand as expected for validation. |
| `PRIMUS_CLUSTER` | `amd-aig-poolside` (CLI default) | User | `primus/tools/benchmark/rccl_bench_args.py` | Cluster label for RCCL benchmark tooling. |
| `PRIMUS_GPU_ARCH` | (auto / `"mi300x"` in simulators) | User / CLI | `primus/core/projection/simulation_backends/origami_backend.py`, `sdpa_simulator.py`, `projection.py` CLI | GPU architecture string for performance projection. |
| `PRIMUS_GPU_CLOCK_MHZ` | (unset) | User | Same as `PRIMUS_GPU_ARCH` | Optional clock override for projection. |
| `PRIMUS_GPU_DEVICE` | `0` | User | `origami_backend.py` | GPU index for hardware detection in projection. |
| `PRIMUS_GEMM_BACKEND` | (unset) | User | `primus/core/projection/simulation_backends/factory.py` | Selects GEMM simulation backend by name. |
| `PRIMUS_PREFLIGHT_MIN_FREE_MEM_GB` | `1` | User | `primus/tools/preflight/gpu/utils.py` | Minimum free GPU memory (GB) for preflight checks. |
| `PRIMUS_PREFLIGHT_MIN_TFLOPS` | `10.0` | User | `primus/tools/preflight/gpu/utils.py` | Minimum TFLOPS threshold for preflight GEMM checks. |
| `PRIMUS_TURBO_AUTO_TUNE` | (unset) | User / tests | `tests/trainer/test_megatron_trainer.py` (integration) | Enables Turbo auto-tuning in supported Turbo/Megatron test flows; not referenced in core `primus/` Python outside tests. **Optional**. |
| `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND` | `TURBO` | User; hooks may set `DEEP_EP` | `primus/modules/trainer/megatron/utils.py`, `examples/run_pretrain.sh`, `runner/helpers/hooks/05_using_uep.sh` | MoE dispatch/combine backend selector. |

---

## 3. Primus YAML substitution

Parsed by `primus/core/config/yaml_loader.py` for patterns `${VAR}` (required) and `${VAR:default}` (optional). Typical experiment YAMLs under `examples/` use these for sweep-friendly overrides.

| Variable | Typical default in YAML | Where set | Where used | Description |
|----------|-------------------------|-----------|------------|-------------|
| `PRIMUS_TEAM` | `"amd"` | User | Resolved before module merge in experiment YAML | Work group / team segment in paths. |
| `PRIMUS_USER` | `"root"` | User | Experiment YAML | User name segment. |
| `PRIMUS_EXP_NAME` | per-example | User | Experiment YAML | Experiment folder name. |
| `PRIMUS_WORKSPACE` | `"./output"` | User | Experiment YAML | Root workspace for artifacts. |
| `PRIMUS_TP` | `1` | User | Megatron example YAMLs | `tensor_model_parallel_size` override. |
| `PRIMUS_PP` | `1` | User | Megatron example YAMLs | `pipeline_model_parallel_size` override. |
| `PRIMUS_EP` | `1` | User | Megatron example YAMLs | `expert_model_parallel_size` override. |
| `PRIMUS_SEQ_LENGTH` | per-model | User | Megatron example YAMLs | Sequence length override. |
| `PRIMUS_MAX_POSITION_EMBEDDINGS` | `4096` or `131072` | User | `examples/megatron/**/*.yaml`, tests | Position embedding cap override. |
| `PRIMUS_GLOBAL_BATCH_SIZE` | per-model | User | Megatron example YAMLs | Global batch override. |
| `PRIMUS_NUM_LAYERS` | per-model | User | Tests and MoE examples | Transformer layer count override. |
| `PRIMUS_MOE_LAYER_FREQ` | MoE patterns | User | MoE examples / tests | MoE layer frequency pattern. |
| `PRIMUS_TOKENIZED_DATA_PATH` | `null` | User | Megatron examples | Path to tokenized training data. |
| `PRIMUS_MODEL` | per-stack | User | Megatron examples | Model preset stem (e.g. `llama3_8B`). |
| `PRIMUS_VPP` | `null` | User | `tests/trainer/test_megatron_trainer.yaml` | Virtual pipeline stages override. |

---

## 4. NCCL / RCCL

Primus seeds many of these in `runner/helpers/envs/base_env.sh`. RCCL honors NCCL-compatible variables on AMD GPUs. See [NCCL environment](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) and [RCCL environment](https://rocm.docs.amd.com/projects/rccl/en/develop/api-reference/env-variables.html).

| Variable | Default (Primus base) | Where set | Where used | Description |
|----------|------------------------|-----------|------------|-------------|
| `NCCL_DEBUG` | unset | User / `base_env.sh` empty default | Preflight reports, RCCL runtime | Log verbosity: `NONE`, `WARN`, `INFO`, `TRACE`, etc. **Optional** unless debugging comms. |
| `NCCL_SOCKET_IFNAME` | derived from `IP_INTERFACE` | `base_env.sh` | `primus/tools/preflight/network/*.py`, GPU topology helpers | Socket NIC for host networking. |
| `GLOO_SOCKET_IFNAME` | same as NCCL if unset | `base_env.sh` | Preflight | Gloo TCP backend interface. |
| `NCCL_IB_HCA` | auto via `get_nccl_ib_hca.sh` if empty | `base_env.sh`, container passthrough | Preflight, multi-node tuning | InfiniBand HCAs to use. |
| `NCCL_IB_GID_INDEX` | `3` | `base_env.sh` | RCCL | GID index for IB/RoCE; many sites use `1` for RoCE v2 (override as needed). |
| `NCCL_IB_TC` | (unset) | User | RCCL | InfiniBand traffic class. |
| `NCCL_IB_FIFO_TC` | (unset) | User | RCCL | InfiniBand FIFO traffic class. |
| `NCCL_IB_ROCE_VERSION_NUM` | (unset) | User | RCCL | RoCE version selection. |
| `NCCL_PXN_DISABLE` | `1` | `base_env.sh` | RCCL | Disable PXN (PCIe cross-NIC); set `0` to enable. |
| `NCCL_P2P_NET_CHUNKSIZE` | `524288` | `base_env.sh` | RCCL | P2P network chunk size tuning. |
| `NCCL_PROTO` | (unset) | User | RCCL | Protocol selection (e.g. `Simple`, `LL`, `LL128`). |
| `NCCL_CROSS_NIC` | `0` | `base_env.sh` | RCCL | Cross-NIC communication policy. |
| `NCCL_IB_RETRY_CNT` | (unset) | User | RCCL | IB retry count. |
| `NCCL_IB_TIMEOUT` | (unset) | User | RCCL | IB timeout. |
| `NCCL_NET_GDR_LEVEL` | (unset) | User | Preflight summaries | GPUDirect RDMA level. |
| `NCCL_IB_DISABLE` | `0` | User / env | Preflight | Disable IB; use sockets only. |
| `NCCL_DMABUF_ENABLE` | (unset) | User | RCCL | DMA-BUF registration path. |
| `NCCL_IGNORE_CPU_AFFINITY` | (unset) | User | RCCL | Ignore CPU affinity hints. |
| `NCCL_IB_QPS_PER_CONNECTION` | (unset) | User | RCCL | IB QPs per connection. |
| `NCCL_MAX_P2P_CHANNELS` | (unset) | User | RCCL | Cap P2P channels. |
| `NCCL_GDR_FLUSH_DISABLE` | (unset) | User | RCCL | Disable GDR flush. |
| `NCCL_IB_USE_INLINE` | (unset) | User | RCCL | Inline IB sends. |
| `NCCL_NET_PLUGIN` | (unset) | User | RCCL | Alternate network plugin (e.g. `librccl-anp.so`). |
| `RCCL_MSCCL_ENABLE` | `0` | `base_env.sh` | RCCL | Enable MSCCL algorithms. |
| `RCCL_MSCCLPP_THRESHOLD` | `1GiB` default | `base_env.sh` | RCCL | MSCCL++ message-size threshold. |
| `RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING` | `0` in hooks | `runner/helpers/hooks/03_enable_ainic.sh`, `enable_ainic.sh` | RCCL | Stricter GDR flush memory ordering; relevant for some NIC/GPU combos. |
| `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK` | `0` | `base_env.sh` | PyTorch + RCCL | Tensor allocator hook for NCCL registration. |
| `TORCH_NCCL_HIGH_PRIORITY` | `1` | `base_env.sh` | PyTorch | High-priority NCCL streams. |

---

## 5. ROCm / HSA / HIP

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `HSA_ENABLE_SDMA` | `1` | `base_env.sh` | ROCm runtime | Enable SDMA engines for copies. |
| `HSA_NO_SCRATCH_RECLAIM` | `0` | `base_env.sh`, container passthrough | ROCm runtime; documented for MoE stability | `1` keeps scratch allocated (often used for MoE stability). See [ROCR environment](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-7.1.1/environment_variables.html). |
| `HIP_VISIBLE_DEVICES` | `0..GPUS_PER_NODE-1` | `base_env.sh` | ROCm device visibility | Restricts which GPU indices ROCm exposes. |
| `ROCBLAS_DEFAULT_ATOMICS_MODE` | (unset) | User | `primus/modules/trainer/megatron/utils.py` | Read for deterministic / accuracy-sensitive GEMM behavior. |

---

## 6. CUDA / PyTorch

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | `base_env.sh`; Megatron trainer may adjust | `primus/modules/trainer/megatron/trainer.py`, Megatron patches | Limits concurrent CUDA connections; often `1` for TP/PP overlap. |
| `TORCH_COMPILE_DISABLE` | `0` | User | `primus/modules/trainer/megatron/utils.py` | Disable `torch.compile` when `1`. |

---

## 7. Transformer Engine

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `NVTE_ROCM_ENABLE_MXFP8` | `1` | `base_env.sh` | Transformer Engine on ROCm | Enable MXFP8 paths. |
| `NVTE_CK_USES_BWD_V3` | `0` | `base_env.sh`, container passthrough | TE / CK | Use CK backward v3 kernels. |
| `NVTE_CK_IS_V3_ATOMIC_FP32` | (unset; examples print `0`) | User / `examples/run_pretrain.sh`, container passthrough | TE / CK | Atomic FP32 mode for CK v3 backward. |
| `PATCH_TE_FLASH_ATTN` | `0` | `base_env.sh`, container passthrough | `runner/helpers/hooks/01_patch_te_flash_attn_max_version.sh` | Trigger TE flash-attn patch hook when `1`. |

---

## 8. Caches and authentication

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `HF_HOME` | `${DATA_PATH}/huggingface` | `base_env.sh`, `primus/core/utils/env_setup.py`, `primus/pretrain.py` | Hugging Face libraries | Cache for models and datasets. |
| `HF_TOKEN` | (unset) | User, container passthrough | Hugging Face Hub | Auth for gated models. **Required** for private/gated assets. |
| `TORCH_HOME` | under workspace | `primus/core/utils/env_setup.py` | PyTorch Hub | Torch Hub cache root. |
| `TRANSFORMERS_CACHE` | aligned with HF layout | `primus/core/utils/env_setup.py` | `transformers` | Model cache for Transformers. |
| `WANDB_API_KEY` | (unset) | User, container passthrough | WandB client, Megatron trainer checks | API key for logging. **Required** for WandB when enabled. |
| `WANDB_PROJECT` | (unset) | User / TorchTitan patch | `primus/backends/torchtitan/patches/wandb_patches.py` | Project name. |
| `WANDB_RUN_NAME` | (unset) | User / patches | Same | Run display name. |
| `WANDB_TEAM` | (unset) | User | TorchTitan metrics (entity) | WandB team/entity. |
| `DATABRICKS_HOST` | (unset) | User | `primus/modules/trainer/megatron/trainer.py` | Required for Databricks-hosted MLflow when MLflow logging is enabled. |
| `DATABRICKS_TOKEN` | (unset) | User | Databricks APIs | Auth token paired with host. |
| `MLFLOW_TRACKING_URI` | (unset) | User | `mlflow` (via Megatron integrations) | MLflow tracking server URI. **Optional** unless using MLflow. |
| `MLFLOW_REGISTRY_URI` | (unset) | User | MLflow | Model registry endpoint. |
| `NLTK_DATA` | (unset) | User | `runner/helpers/hooks/train/pretrain/megatron/preprocess_data.py`, Megatron-LM tools | Punkt and other tokenizer data for preprocessing. |
| `TOKENIZED_DATA_PATH` | per-hook default | User | `runner/helpers/hooks/train/pretrain/megatron/prepare.py` | Pre-tokenized dataset location for Megatron data prep hooks. |

---

## 9. HipBLASLt tuning

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `PRIMUS_HIPBLASLT_TUNING_STAGE` | `0` | User | `examples/run_pretrain.sh` | Stages `0` off, `1` dump shapes, `2` offline tune, `3` apply tuned kernels. |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | (unset) | User / tuning scripts | `examples/run_pretrain.sh` | Path to tuned-kernel override file for stage `3`. |
| `TE_HIPBLASLT_TUNING_RUN_COUNT` | varies | User | `examples/run_pretrain.sh` | Number of benchmark runs per shape during TE HipBLASLt tuning. |
| `TE_HIPBLASLT_TUNING_ALGO_COUNT` | varies | User | `examples/run_pretrain.sh` | Transformer Engine HipBLASLt search breadth. |
| `TE_HIPBLASLT_TUNING_ALGO_FILE` | (unset) | User | TE + HipBLASLt | Algorithm file for TE tuning flows. |
| `TE_HIPBLASLT_TUNING` | (unset) | User | `examples/run_pretrain.sh` | When set, interacts with deterministic mode and tuning stages (disable conflicting modes per script comments). |
| `HIPBLASLT_LOG_LEVEL` | (unset) | User | HipBLASLt | Library log level. |
| `HIPBLASLT_LOG_MASK` | (unset) | User | HipBLASLt | Bitmask for log categories. |

---

## 10. Build and rebuild

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `REBUILD_PRIMUS_TURBO` | `0` | User, container passthrough | `runner/helpers/hooks/00_rebuild_primus_turbo.sh` | `1` rebuilds Primus-Turbo on startup. |
| `REBUILD_BNXT` | `0` | User, container passthrough | `runner/helpers/hooks/02_rebuild_bnxt.sh` | `1` rebuilds BNXT driver artifacts when packaged. |
| `USING_AINIC` | (unset) | User | `runner/helpers/hooks/03_enable_ainic.sh` | `1` enables AINIC-oriented networking hooks. |
| `MAX_JOBS` | (unset) | User / tooling | `tools/daily/safe_wrapper.py` | Parallel compile jobs for pip builds. |
| `BACKEND_PATH` | (unset) | User | `primus/pretrain.py`, `primus/core/backend/backend_adapter.py` | Override checkout path for third-party backends (Megatron, TorchTitan, MaxText). |

---

## 11. Container passthrough

`runner/.primus.yaml` lists names forwarded from the host into training containers (`container.options.env`). Primus does not assign values here; it only whitelists keys for `--env` forwarding.

Forwarded keys:

`MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK`, `GPUS_PER_NODE`, `DOCKER_IMAGE`, `HF_TOKEN`, `WANDB_API_KEY`, `ENABLE_NUMA_BINDING`, `REBUILD_PRIMUS_TURBO`, `USING_AINIC`, `PATCH_TE_FLASH_ATTN`, `REBUILD_BNXT`, `HSA_NO_SCRATCH_RECLAIM`, `NVTE_CK_USES_BWD_V3`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `NCCL_IB_GID_INDEX`, `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32`, `NVTE_CK_IS_V3_ATOMIC_FP32`

---

## 12. Slurm

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `SLURM_NNODES` / `SLURM_JOB_NUM_NODES` | job-dependent | Slurm | `primus-cli-slurm-entry.sh` (`NNODES` export), preflight probes | Node count for the allocation. |
| `SLURM_NODEID` | job-dependent | Slurm | Mapped to `NODE_RANK` in `primus-cli-slurm-entry.sh` | Node index. |
| `SLURM_PROCID` | job-dependent | Slurm | Fallback for `NODE_RANK` when `SLURM_NODEID` is unset | Process id within the Slurm step (entry script). |
| `SLURM_JOB_ID` | job-dependent | Slurm | `primus/tools/preflight/host/host_probe.py` | Job identifier string. |

---

## 13. Debug and pipeline

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `DUMP_PP_DIR` | `output/pp_data` | User | `primus/modules/trainer/megatron/trainer.py` | Directory for pipeline-parallel debug dumps. |
| `DEBUG_SIMULATOR` | `0` | User | `primus/core/projection/performance_projection/simulator.py` | `1` enables verbose projection simulator logging. |
| `RECORD_OFFLOAD_MEMORY_INFO` | `0` | User | `primus/core/pipeline_parallel/handler/offload_handler.py` | Record offload memory stats when `1`. |
| `RECORD_OFFLOAD_MEMORY_INFO_DIR` | `output` | User | `primus/core/pipeline_parallel/scheduler/scheduler.py` | Output directory for offload memory logs. |
| `USE_PINNED_OFFLOAD` | `0` | User | `offload_handler.py` | Use pinned host memory for offload buffers when `1`. |

---

## 14. JAX / XLA (MaxText)

Primus MaxText hooks print recommended values in `runner/helpers/hooks/train/pretrain/maxtext/prepare.py`; MaxText and JAX read them directly.

| Variable | Default | Where set | Where used | Description |
|----------|---------|-----------|------------|-------------|
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | e.g. `.97` in prepare hook | User / hook output | JAX / XLA allocator | Fraction of GPU memory pre-allocated for JAX. |
| `DUMP_HLO_DIR` | `${PRIMUS_PATH}/output/xla_dump_hlo` (example) | User | XLA via `XLA_FLAGS` composition | Directory for HLO dumps when enabled. |
| `DUMP_HLO` | `0` | User | Prepare hook → XLA flags | Gate HLO dumping (`1` enables in hook samples). |

**Note:** MaxText also propagates many knobs through `XLA_FLAGS` and `LIBTPU_INIT_ARGS` upstream; see MaxText sources for the full matrix.
