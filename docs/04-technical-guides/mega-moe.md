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

- **Runtime**: ROCm ≥ 7.0, Python ≥ 3.10, PyTorch ≥ 2.6.0 (ROCm build); gfx950. The DeepEP
  baseline additionally needs the optional **rocSHMEM**. Image `rocm/primus:v26.3` is recommended.
- **Primus-Turbo with MegaMoE**: MegaMoE requires Primus-Turbo
  (`https://github.com/AMD-AGI/Primus-Turbo.git`) at commit
  **`9b5d3092efcbc087657b233d8e9ae662cee6ec6b` or newer** on `main`. The default image does not ship
  this kernel, so Primus-Turbo must be rebuilt from source. See upstream
  [main README](https://github.com/AMD-AGI/Primus-Turbo/blob/main/README.md) and
  [MegaMoE doc](https://github.com/AMD-AGI/Primus-Turbo/blob/main/docs/README_Mega_MoE.md)
  for build details. Install it via the rebuild hook:

**Primus rebuild hook (build from source).** The system hook
`runner/helpers/hooks/00_rebuild_primus_turbo.sh` clones + builds + installs the given ref before
the training command; each node builds in a node-local dir so multi-node runs avoid shared-fs
conflicts. Point it at a commit for reproducibility, or at `main` to track the latest code:

```bash
export REBUILD_PRIMUS_TURBO=1                       # trigger the hook
export PRIMUS_TURBO_REF=9b5d3092efcbc087657b233d8e9ae662cee6ec6b   # min required commit (or main)
export GPU_ARCHS="gfx950"                           # build only target arch (multiple: semicolon-separated)
# Optional: custom build dir (default /tmp/primus_turbo_<hostname>)
# export PRIMUS_TURBO_BUILD_DIR=/tmp/primus_turbo_build
```

## Design: two stages + weight modules (for DDP overlap)

The expert path could be exposed as a *single* fused op taking both `w1` and `w2`
(`fused_mega_moe(x, topk_idx, topk_weights, w1, w2, group)` still exists in Primus-Turbo). Primus
instead drives it as **two stages**, each owning one weight, wrapped in **two tiny weight modules**
(`primus/backends/megatron/core/extensions/mega_moe.py`):

```
MegaMoEExperts
├── fc1_weight : MegaMoEWeightModule   # w1 [g, 2I, H] gate+up ; forward() -> w1
└── fc2_weight : MegaMoEWeightModule   # w2 [g, H, I]  down    ; forward() -> w2


FORWARD (in order)                          BACKWARD (in order)
──────────────────────────────────────      ──────────────────────────────────────
w1 = fc1_weight()                           stage2.backward  ->  dW2
  hook: all-gather(w1), wait
                                            hook on fc2_weight
stage1: dispatch + GEMM1        <──┐          -> reduce-scatter(dW2)  ──┐
                                   │ overlap                            │ overlap
w2 = fc2_weight()                  │        stage1.backward  ->  dW1  ──┘
  hook: all-gather(w2)  ───────────┘
                                            hook on fc1_weight
stage2: SwiGLU + GEMM2 + combine              -> reduce-scatter(dW1)
  (waits for w2 here)                         (overlaps the next layer)
```

`MegaMoEWeightModule` holds a single `torch.nn.Parameter` and its `forward()` just returns it. It
computes nothing — its only job is to be a **hook site** for the two collectives Megatron's
distributed optimizer wants to overlap, both of which are driven at module / parameter granularity:

- **Parameter all-gather** (`overlap_param_gather`) rides the *forward pre-hook*, which is per
  module. With a single call site taking `(w1, w2)` both gathers must land before any compute; split,
  `w2`'s gather is issued at `fc2_weight` and overlaps stage1.
- **Gradient reduce-scatter** (`overlap_grad_reduce`) rides the *grad hook*, which fires when a
  parameter's `.grad` appears. One fused autograd node emits `dW1` and `dW2` together at the end of
  the layer backward; split, `dW2` lands early and its reduce-scatter hides under stage1's backward.

The split is purely at the Python/autograd level — the kernels are unchanged.

## Configuration

Enable the fused MegaMoE layer in the training config:

```yaml
enable_primus_turbo: true
use_turbo_mega_moe: true   # MegaMoE layer replacement (EP-only / TP=1 / bf16)
```

The patch is applied only when **all** of these hold: `enable_primus_turbo=True`,
`use_turbo_mega_moe=True`, `tensor_model_parallel_size==1`, `params_dtype==bf16`, and an EP process
group exists.

### Expert precision

```yaml
turbo_mega_moe_precision: mxfp8   # bf16 (default) | mxfp8; read only when use_turbo_mega_moe is on
```

`mxfp8` runs the two expert stages in MXFP8 (dispatch + fc1, SwiGLU, fc2 + combine, and the dW1/dW2
wgrads). This is deliberately **not** wired to Megatron's `--fp8`, which selects a TE fp8 recipe for
the dense layers and has no path to this fused op — keeping them separate lets the MoE be A/B'd on
its own, and avoids a TE recipe change silently altering MoE behaviour it does not describe.

Parameters stay bf16, so initialization, checkpointing and the optimizer see nothing new. The op
maintains the mxfp8 weight quant in an internal cache keyed on `w._version`, and the
`megatron.turbo.mega_moe_weight_generation` patch drops that cache once per optimizer step — the
key alone is not enough, because the precision-aware optimizer updates the weights without ever
bumping `_version`.

Not supported on the fp8 path: activation recompute (`moe_layer_recompute`,
`recompute_granularity=full`) and CUDA-graph capture. Both replay the forward while the op still
holds a live symmetric buffer and a cross-rank spin-wait handshake; the layer asserts on the
recompute settings up front.

The following model settings are **required** — MegaMoE asserts on anything else:

```yaml
tensor_model_parallel_size: 1           # EP-only, TP=1
add_bias_linear: false                  # no bias in linear layers
# gated SwiGLU + SiLU activation
```

Unsupported (each raises an error): sequence-level / global aux loss, z-loss, sinkhorn, and input
jitter (only the standard `aux_loss` is supported); aux-loss-free expert bias
(`enable_expert_bias=True` raises `NotImplementedError`).

### Router force load balancing (`moe_router_force_load_balancing_type`)

The benchmark config sets `moe_router_force_load_balancing: true`, which discards the real router
decision and forces every expert to receive a similar number of tokens — this removes run-to-run
expert-imbalance noise so throughput numbers are comparable. `moe_router_force_load_balancing_type`
selects *how* the balancing is done (it only has an effect when force load balancing is on):

| value | behavior |
| --- | --- |
| `even` (Primus default) | Deterministic round-robin `(token_idx * topk + k) % num_experts`. Per-expert token counts are exactly equal **and identical every step**, so the grouped-GEMM shapes (`M_total`, per-expert `M`) never change. |
| `uniform` | Megatron-LM's original behavior: the logits are replaced with random values before routing, so token counts are balanced only *statistically* and fluctuate step to step, like real routing. |

**Use `uniform` for benchmarking.** `even` produces perfectly constant, aligned per-expert shapes
that the non-fused (DeepEP / grouped-GEMM) baseline benefits from disproportionately — no autotune
or shape-recompile churn, no padding waste — while MegaMoE is designed to absorb ragged, varying
token counts. Measuring under `even` therefore understates MegaMoE's gain; `uniform` keeps the
balancing (for reproducibility) but preserves the step-to-step shape variation of real training.

The examples below use the rebuild hook (`REBUILD_PRIMUS_TURBO=1
PRIMUS_TURBO_REF=9b5d3092efcbc087657b233d8e9ae662cee6ec6b`) to build Primus-Turbo from source.

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
export PRIMUS_TURBO_REF=9b5d3092efcbc087657b233d8e9ae662cee6ec6b
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
  --moe_router_force_load_balancing_type uniform \
  --enable_primus_turbo True \
  --use_turbo_mega_moe True \
  --mock_data True
```


### Example 2 — single-node EP8 with full CUDA graph

Same geometry as Example 1, plus TE full-scope CUDA graph capture
(`--external_cuda_graph True` maps to `cuda_graph_impl="transformer_engine"`; scope `full` is
normalized to `[]` by Megatron, i.e. capture the whole layer). MegaMoE is fully sync-free — no
device-to-host sync or CPU-side wait in the expert path — so it captures cleanly and is
`torch.compile`-friendly.

```bash
#!/bin/bash
set -e

export EXP=examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml
export REBUILD_PRIMUS_TURBO=1
export PRIMUS_TURBO_REF=9b5d3092efcbc087657b233d8e9ae662cee6ec6b
export GPU_ARCHS=gfx950

bash examples/run_pretrain_cli.sh \
  --num_layers 4 \
  --micro_batch_size 2 \
  --global_batch_size 1024 \
  --train_iters 15 \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --expert_model_parallel_size 8 \
  --moe_layer_freq 1 \
  --moe_shared_expert_intermediate_size None \
  --pipeline_model_parallel_layout null \
  --recompute_granularity null \
  --recompute_num_layers 0 \
  --recompute_layer_ids null \
  --moe_router_force_load_balancing_type uniform \
  --external_cuda_graph True \
  --cuda_graph_scope full \
  --cuda_graph_warmup_steps 3 \
  --enable_primus_turbo True \
  --use_turbo_mega_moe True \
  --mock_data True
```

### Example 3 — 8-node EP8/PP8

8 nodes × 8 GPUs = 64 GPU, `TP=1 / PP=8 / EP=8` → `DP = 64/(TP*PP) = 8`,
`GBS = MBS*DP*GA = 2*8*64 = 1024`.

```bash
set -e

# Cluster geometry (8 nodes x 8 GPUs = 64 GPUs)
export NNODES=8

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
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --recompute_layer_ids null \
  --moe_router_force_load_balancing_type uniform \
  --enable_primus_turbo True \
  --use_turbo_mega_moe True \
  --mock_data True
```
