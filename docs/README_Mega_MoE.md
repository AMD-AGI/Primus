# MegaMoE

MegaMoE is a **FlyDSL**-based fused MoE layer that replaces Megatron's native `MoELayer`. It fuses
the expert-parallel all-to-all communication into the grouped GEMMs via two fused kernels:

- **dispatch grouped GEMM** (`dispatch_grouped_gemm`): fuses token dispatch (all-to-all) into the
  L1 grouped GEMM.
- **grouped GEMM combine** (`grouped_gemm_combine`): fuses the L2 grouped GEMM into combine
  (all-to-all) + weighted reduce.

Together with a **fused router** (score function + group-limited top-k + aux score) and the
intermediate SwiGLU, the full expert path is `dispatch_grouped_gemm → SwiGLU → grouped_gemm_combine`;
the load-balancing aux loss is computed internally and returned. Runtime target is **EP-only
(TP=1) + bf16**.

## Prerequisites

- **Runtime**: ROCm ≥ 7.0, Python ≥ 3.10, PyTorch ≥ 2.6.0 (ROCm build); gfx942 / gfx950. The DeepEP
  baseline additionally needs the optional **rocSHMEM**. Image `rocm/primus:v26.3` is recommended.
- **Primus-Turbo with MegaMoE**: MegaMoE requires the **`feat/flydsl_mega_kernel`** branch of
  Primus-Turbo (`git@github.com:AMD-AGI/Primus-Turbo.git`, being merged into `main`). The default
  image does not ship this kernel, so Primus-Turbo must be rebuilt from that branch. See upstream
  [main README](https://github.com/AMD-AGI/Primus-Turbo/blob/feat/flydsl_mega_kernel/README.md) and
  [MegaMoE doc](https://github.com/AMD-AGI/Primus-Turbo/blob/feat/flydsl_mega_kernel/docs/README_Mega_MoE.md)
  for build details. Install it via the rebuild hook:

**Primus rebuild hook (build from source).** The system hook
`runner/helpers/hooks/00_rebuild_primus_turbo.sh` clones + builds + installs the given branch before
the training command; each node builds in a node-local dir so multi-node runs avoid shared-fs
conflicts. Use it to track the branch's latest code:

```bash
export REBUILD_PRIMUS_TURBO=1                       # trigger the hook
export PRIMUS_TURBO_REF=feat/flydsl_mega_kernel     # MegaMoE branch
export GPU_ARCHS="gfx950"                           # build only target arch (multiple: semicolon-separated)
# Optional: custom build dir (default /tmp/primus_turbo_<hostname>)
# export PRIMUS_TURBO_BUILD_DIR=/tmp/primus_turbo_build
```

## Configuration

Enable the fused MegaMoE layer in the training config:

```yaml
enable_primus_turbo: true
use_turbo_mega_moe: true   # full fused MoE layer replacement (EP-only / TP=1 / bf16)
```

The patch is applied only when **all** of these hold: `enable_primus_turbo=True`,
`use_turbo_mega_moe=True`, `tensor_model_parallel_size==1`, `params_dtype==bf16`, and an EP process
group exists.

The following model settings are **required** — MegaMoE asserts on anything else:

```yaml
tensor_model_parallel_size: 1           # EP-only, TP=1
add_bias_linear: false                  # no bias in linear layers
# gated SwiGLU + SiLU activation
```

Unsupported (each raises an error): sequence-level / global aux loss, z-loss, sinkhorn, and input
jitter (only the standard `aux_loss` is supported); aux-loss-free expert bias
(`enable_expert_bias=True` raises `NotImplementedError`).

The examples below use the rebuild hook (`REBUILD_PRIMUS_TURBO=1
PRIMUS_TURBO_REF=feat/flydsl_mega_kernel`) to build Primus-Turbo from source.

### Example 1 — single-node EP8, 4 layers (`run_pretrain_cli.sh`)

1 node × 8 GPUs, `TP=1 / PP=1 / EP=8`, DeepSeek-V3 BF16, `GBS = MBS*GPUS*GA = 2*8*64 = 1024`.
Minimal fused-MegaMoE run from `Primus/`:

```bash
#!/bin/bash
set -e

# Model config
export EXP=examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml
# Build Primus-Turbo from source before training (hook)
export REBUILD_PRIMUS_TURBO=1
export PRIMUS_TURBO_REF=feat/flydsl_mega_kernel
export GPU_ARCHS=gfx950

# Parallelism (EP-only) + fused MegaMoE
bash examples/run_pretrain_cli.sh \
  --num_layers 4 \
  --micro_batch_size 2 \
  --global_batch_size 1024 \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --expert_model_parallel_size 8 \
  --moe_layer_freq 1 \
  --moe_shared_expert_intermediate_size None \
  --pipeline_model_parallel_layout null \
  --recompute_granularity null \
  --recompute_num_layers 0 \
  --recompute_layer_ids null \
  --enable_primus_turbo True \
  --use_turbo_mega_moe True \
  --mock_data True
```


### Example 2 — 8-node EP8/PP8

8 nodes × 8 GPUs = 64 GPU, `TP=1 / PP=8 / EP=8` → `DP = 64/(TP*PP) = 8`,
`GBS = MBS*DP*GA = 2*8*64 = 1024`.

```bash
set -e

# Cluster geometry (8 nodes x 8 GPUs = 64 GPUs)
export NNODES=8

export USING_AINIC="${USING_AINIC:-1}"
export CLEAN_DOCKER_CONTAINER=1

# Parallelism: EP-only compute + PP staging. DP = NNODES*GPUS_PER_NODE/(TP*PP) = 8
export PRIMUS_TP=1
export PRIMUS_PP=8
export PRIMUS_EP=8
export USING_AINIC=1

# Toggle the fused MegaMoE layer + model config
export EXP=examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml

bash examples/run_slurm_pretrain_cli.sh \
  --train_iters 15 \
  --micro_batch_size 2 \
  --global_batch_size 1024 \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 8 \
  --pipeline_model_parallel_layout "Ett|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttL" \
  --expert_model_parallel_size 8 \
  --moe_shared_expert_intermediate_size None \
  --mtp_num_layers 0 \
  --enable_primus_turbo True \
  --use_turbo_mega_moe True \
  --mock_data True
```
