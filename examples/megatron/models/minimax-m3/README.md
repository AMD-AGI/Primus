# MiniMax-M3 proxy on MI455X

MiniMax-M3 BF16 pretraining on MI455X (gfx1250), as a shallow proxy of the real 60-layer model: the
first three layers stay dense, and every later layer is one MSA (MiniMax Sparse Attention) layer plus
one MoE layer (128 experts, top-4, hidden 6144, 64 query / 4 KV heads of dim 128). The launcher's
default is 4 layers on one GPU; the baseline is 8 layers on four GPUs (see Baseline). Sequence length
4096, MBS 1, mock data, full recompute of the first layer. The config is
`examples/megatron/configs/MI455X/minimax_m3-BF16-pretrain.yaml` and the launcher is
`run_minimax_m3_mi455x.sh`: with `PRIMUS_LAUNCHER=container` (default) it starts a new container of
`DOCKER_IMAGE` from the host and re-runs itself inside it; with `PRIMUS_LAUNCHER=direct` it runs inside
the current container.

MSA is not Primus-Turbo attention. `msa_backend=flydsl` (the default) runs the kernels under
`primus/backends/megatron/core/transformer/minimax_m3/flydsl/`. On gfx950 those are the MFMA kernels.
On gfx1250 they are the wave32 WMMA kernels (`*_w32.py`); a tree without that port fails at compile
with `Cannot select: permlane32.swap`. `PRIMUS_MSA_BACKEND=eager` is the plain-PyTorch reference and
does not need those kernels. The experts use Primus-Turbo's grouped GEMM and RMSNorm uses
Primus-Turbo. DeepEP stays off by default (see Known issues).

The default image is `amdprimus/amdprimus:tas-20261109`,
`amdprimus/amdprimus@sha256:5e287acb708c936ae3a29771c92ceeb2f17c03d686e9a6cb254a0d974a5346cb`. It is
`amdprimus/amdprimus:tas-20261019` (see the DeepSeek-V3 example's README),
`amdprimus/amdprimus@sha256:d653324d124e218f43c92eaac3cfc5d05059bd45fb4abf5bdfb78b987af4af87`: torch
2.14.0 on ROCm 10.2.0a20261011, Triton 3.8.0, Transformer Engine 2.19.0.dev0 (bfe94fbe5), FlyDSL
0.3.4.1 and AITER 0.1.24.post1, with its Primus-Turbo 0.6.1.dev2 replaced by Primus-Turbo
`dev/mi455x_v4` 126a485b (0.6.1.dev31, Primus-Turbo PR #632), built as the base image builds it (gfx942,
gfx950 and gfx1250 in one build, OpenMPI off the paths so rocSHMEM is not linked). That Primus-Turbo has
the gfx1250 FlyDSL attention this example uses (see FlyDSL attention in the dense layers), the grouped
GEMM fix and tuning of tas-20261109 image below, and FlyDSL DeepEP. The default before it was
`amdprimus/amdprimus:tas-20261019-turbo-4c74631`,
`amdprimus/amdprimus@sha256:b471601ddc4736e5bfc1d2d69e9fc5b7f39d9a5496e50b6f0f4138453b2a82fc`, the same
base with Primus-Turbo `main` 4c74631 (0.6.1.dev16) built the same way.

Every run on this page before Primus-Turbo 4c74631 image used tas-20261019 and its own Primus-Turbo,
except the FlyDSL attention runs, which used a Primus-Turbo checkout through `TURBO_DIR`; the runs from
that section up to tas-20261109 image used the 4c74631 image. Every run
outside hipBLASLt tuning before that section used hipBLASLt's own kernels, which
`HIPBLASLT_TUNING_OVERRIDE_FILE=` restores, and every run before Expert grouped GEMM on FlyDSL used
Primus-Turbo's Triton grouped GEMM, which `TURBO_GROUPED_GEMM_BACKEND=default` restores.

## Run

```bash
# On the host, 1 GPU (the default: 4 layers, EP1, GBS 1). Pick a free device on a shared node.
HIP_VISIBLE_DEVICES=1 bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh

# The baseline: 4 GPUs, 8 layers, EP4, GBS 32
GPUS_PER_NODE=4 PRIMUS_TOTAL_LAYERS=8 GBS=32 bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh

# Inside a container
PRIMUS_LAUNCHER=direct bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh

# MiniMax-M3.1-Flash (MSA in every layer, see MiniMax-M3.1-Flash), the baseline shape
EXP=examples/megatron/configs/MI455X/minimax_m3_1_flash-BF16-pretrain.yaml GPUS_PER_NODE=4 PRIMUS_TOTAL_LAYERS=8 GBS=32 \
    bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh
```

Two runs at once on one node need different `MASTER_PORT`s.

**FlyDSL attention for the dense layers.** The launcher runs the three dense layers on Primus-Turbo's
FlyDSL attention when the Primus-Turbo in use has its gfx1250 kernels for head dim 128
(`primus_turbo/flydsl/attention/gfx1250/`, Primus-Turbo PR #580). The default image's Primus-Turbo
has them. That of `DOCKER_IMAGE=amdprimus/amdprimus:tas-20261019` predates them; to use them with that
image, build a Primus-Turbo checkout in place for it once (gfx1250 only, OpenMPI off the paths so
rocSHMEM is not linked) and pass it as `TURBO_DIR`; the launcher puts it first on `PYTHONPATH`:

```bash
git clone https://github.com/AMD-AGI/Primus-Turbo.git /path/to/Primus-Turbo
git -C /path/to/Primus-Turbo checkout ff4898e4  # dev/mi455x_v3, the commit measured below
git -C /path/to/Primus-Turbo submodule update --init --recursive
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v /path/to/Primus-Turbo:/path/to/Primus-Turbo \
    -w /path/to/Primus-Turbo amdprimus/amdprimus:tas-20261019 bash -c \
    'PATH=$(tr : "\n" <<< "$PATH" | grep -v openmpi | paste -sd:) GPU_ARCHS=gfx1250 python3 setup.py build_ext --inplace'
HIP_VISIBLE_DEVICES=1 DOCKER_IMAGE=amdprimus/amdprimus:tas-20261019 TURBO_DIR=/path/to/Primus-Turbo \
    bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh
```

ff4898e4 pins FlyDSL 0.3.4.1, the image's version.

**Settings** are environment variables:

- Parallelism: `GPUS_PER_NODE` (default 1), `HIP_VISIBLE_DEVICES`, `PRIMUS_TP` (must be 1), `PRIMUS_PP`
  (default 1), `PRIMUS_EP` (default `GPUS_PER_NODE`). Expert parallel must divide both 128 and the
  data-parallel size. `GBS` defaults to `MBS` times that data-parallel size.
- Model depth: `PRIMUS_TOTAL_LAYERS` (default 4, minimum 4; 8 is the MI355X proxy and the baseline
  here, 60 is the full model). The launcher keeps the first three layers dense and sets
  `moe_layer_freq`, `sparse_attention_freq` and `sparse_disable_index_value` to the same length.
- Training: `TRAIN_ITERS` (default 10; at least 3, because the yaml warms the learning rate up over
  2), `MBS` (default 1), `GBS`, `PRIMUS_SEQ_LENGTH` (default 4096), `PRIMUS_MAX_POSITION_EMBEDDINGS`
  (default `PRIMUS_SEQ_LENGTH`; the preset would otherwise build the RoPE cache at 1,048,576),
  `PRIMUS_RECOMPUTE_NUM_LAYERS` (default 0, no recomputation; it was 1). The yaml's
  `recompute_method: block` recomputes the first that many layers: dense layer 1 on M3, an MSA layer
  on M3.1-Flash, whose recomputation cost 2.7 ms a step on one GPU, where memory is not the limit.
- Gradients with one micro-batch on one rank (`DP == 1` and `GBS == MBS`):
  `PRIMUS_BF16_GRAD_SINGLE_MICROBATCH` (default 1) passes `--grad_reduce_in_bf16 True` and sets
  `PRIMUS_FUSED_ADAM_BF16_GRAD=1` and `PRIMUS_FUSED_ADAM_DIRECT_GRAD=1`: the HIP Adam reads each
  parameter's BF16 `.grad` directly, and the DDP buffer is neither zeroed nor accumulated into. With
  nothing to accumulate or reduce, the FP32 buffer only ever held `0 + grad`, so this is bit-identical;
  on M3.1-Flash 1 GPU it removed ~13 ms of gradient passes. Other shapes keep FP32 gradients.
- MSA: `PRIMUS_MSA_BACKEND`, `flydsl` (default) or `eager`.
- Experts: `USE_TURBO_GROUPED_GEMM` (default `True`; `False` is TE's grouped GEMM, one hipBLASLt GEMM
  per expert) and `TURBO_GROUPED_GEMM_BACKEND` (default `bf16:flydsl`; `default` is Triton here, and
  `--gradient_accumulation_fusion True` needs it). See Expert grouped GEMM on FlyDSL.
- `TURBO_DIR` (default unset: the image's Primus-Turbo): a Primus-Turbo checkout built in place for
  the image, see above. A relative path is taken from the current directory.
- Dense layers: `USE_TURBO_ATTENTION` runs Primus-Turbo attention in the three dense layers, with
  `PRIMUS_TURBO_ATTN_BACKEND`, `flydsl` or `triton`. They default to `True` and `flydsl` when the
  Primus-Turbo in use (`TURBO_DIR`, else the image's) has `primus_turbo/flydsl/attention/gfx1250/`,
  as the default image's does, and to `False` and `triton` otherwise. Asking for `flydsl` without
  those kernels stops the launcher before training. Primus-Turbo's own default backend fails with
  `invalid argument for fmha_bwd` here. MSA layers are unaffected.
- Optimizer: `PRIMUS_FUSED_ADAM_CLIP` (default `1`) runs Primus's HIP Adam, which applies the
  gradient-clip coefficient inside the kernel. `PRIMUS_FUSED_ADAM_BF16_WRITEBACK` and
  `PRIMUS_STATIC_BF16_LOSS_SCALE_CPU` (default: the value of `PRIMUS_FUSED_ADAM_CLIP`) also have it
  write the BF16 parameters and keep the loss scale on the host. Megatron then logs no grad norm. With
  `PRIMUS_FUSED_ADAM_CLIP=0` Adam is Primus's Triton AdamW (`use_triton_fused_adam`, on by default),
  and `--use_triton_fused_adam False` keeps TE's FusedAdam. `--use_precision_aware_optimizer True`
  needs `PRIMUS_FUSED_ADAM_CLIP=0`: the BF16 writeback needs separate FP32 master parameters, and the
  run stops at the first step without them.
- `HIPBLASLT_TUNING_OVERRIDE_FILE` (default: the file in `hipblaslt_tuning/`; empty: hipBLASLt's own
  kernel choice): see hipBLASLt tuning below. A file of your own may be given relative to the current
  directory; the launcher checks that it can be read and mounts it into the container.
- `MOCK_DATA` (default `True`). `PROFILE=True` profiles iterations 6–7 on rank 0 (trace under
  `output/<team>/<user>/<exp>/tensorboard/`, about 41 MB for the baseline); add
  `--profile_ranks '[0,1,2,3]'` for every rank. `PATCH_TRITON_EXPERT_SCHEDULING` (default `1`): see
  Triton expert scheduling below.
- Config and output: `EXP` (the experiment yaml; MiniMax-M3.1-Flash's is
  `examples/megatron/configs/MI455X/minimax_m3_1_flash-BF16-pretrain.yaml`), and `PRIMUS_WORKSPACE`
  (default `output`), `PRIMUS_TEAM`, `PRIMUS_USER`, `PRIMUS_EXP_NAME`, which make up the output
  directory `<workspace>/<team>/<user>/<exp>`.

The defaults are in the script. Arguments to the script are appended to the Primus command line, for
example `--use_turbo_autotune True`.

**Output.** The console log is written to `output/<team>/<user>/<exp>/train.log`. From the host,
training runs as root in the new container, so the files it creates are owned by root. Inside a
container it runs as the container's user, which has to be root for the Triton patch. Iterations 1–2
compile the FlyDSL and Triton kernels (iteration 1 takes about 30 s), so the ms/iter on this page is
Primus's average over iterations 3–10 (`log_avg_skip_iterations=2`). Memory is the logged
`hip mem usage`, which counts the whole device, other processes included. With the default
`PRIMUS_FUSED_ADAM_CLIP=1`, check that the log has `[Patch:megatron.optimizer.te_fused_adam_clip]
enabled`: when its HIP extension does not build, Primus logs the error and trains without it.

**Triton expert scheduling.** Triton 3.8 turns on AMDGPU expert scheduling for gfx1250, which makes
Triton kernels fault or return NaN at random. The launcher sets `TRITON_HIP_USE_EXPERT_SCHEDULING=0`,
which Triton applies whenever it compiles a kernel, and with `PATCH_TRITON_EXPERT_SCHEDULING=1` (the
default) also makes `is_expert_scheduling_enabled()` in the container's Triton return `False`, which
invalidates Inductor's cache of kernels compiled without the variable. The patch needs root and stays
in a long-lived container; set `PATCH_TRITON_EXPERT_SCHEDULING=0` to skip it. See the Llama-3.1-8B
README for the same note.

## Baseline

The four-GPU stand-in for MiniMax-M3 until a full node is available: EP4, 8 layers (three dense, five
MSA/MoE), MBS 1, GBS 32, sequence 4096, mock data, and the rest of the launcher's defaults, including
uniform routing (`moe_router_force_load_balancing: true`). EP4 uses Megatron's `alltoall` dispatcher.

- Eight layers fit with room to spare: 280 GB per GPU at EP4, and each further MSA/MoE layer adds
  about 33 GB.
- With four layers the 200k-token output layer and the three dense layers carry about 85% of the
  linear-layer FLOPs. With eight, the five MSA/MoE layers carry about half; in the full model, about
  91%.
- GBS 32 gives each rank 8 micro-batches per step, as in the MI355X 8-layer config (EP8, GBS 64), so
  the optimizer step is spread over 8 micro-batches instead of following every one.

What it leaves out: at EP4 each expert gets 512 tokens per micro-batch (1024 at EP8, more beyond); no
pipeline parallel; one node; uniform routing and mock data; sequence 4096 only.

The results in this section predate the kernel changes in the later sections. For the launcher's
current defaults, see Baseline with the current defaults.

Results on node A, one of the two shared 4x MI455X nodes these runs used, with this branch on
`dev/mi455x_v3` c19456f1, one run at a time on the four otherwise idle GPUs. Every run finished 10
iterations with 0 NaN.

| Layers | MBS / GBS | ms/iter | Tokens/s/GPU | TFLOP/s/GPU | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- | --- | --- | --- |
| 8 (baseline) | 1 / 32 | 1536.6 | 21,325 | 565.5 | 280.2 GB | 8.19595 |
| 8 | 1 / 4 | 247.6 | 16,545 | 438.8 | 280.2 GB | 9.11894 |
| 4 | 1 / 32 | 736.3 | 44,504 | 734.0 | 147.5 GB | 9.24521 |

- The baseline repeated at 1538.2 ms with the same loss.
- From GBS 4 to GBS 32, tokens/s per GPU rose 29% and memory did not change. The two 8-layer runs
  give 184 ms per micro-batch plus 64 ms per step that does not grow with the micro-batch count: the
  optimizer step and the gradient reduction.
- Each MSA/MoE layer adds about 200 ms per step at GBS 32, 25 ms per micro-batch.
- TFLOP/s per GPU falls from 734 with four layers to 565 with eight, because the MSA/MoE layers run
  less efficiently than the output layer and the dense layers. The full model will be lower still.
  Compare throughput only between runs of the same depth.

Where the baseline step goes: GPU kernel time of profiled iteration 7 on rank 0, in ms. Profiling
makes that step 1694 ms instead of 1537.

| Kernels | ms | Share |
| --- | --- | --- |
| hipBLASLt GEMMs: dense layers, attention projections, shared experts, output layer | 385.5 | 25.7% |
| FlyDSL MSA and indexer | 359.9 | 24.0% |
| elementwise and reductions | 223.3 | 14.9% |
| Primus-Turbo grouped GEMM (Triton), all expert GEMMs | 204.9 | 13.7% |
| TE unfused softmax in the three dense attention layers | 106.5 | 7.1% |
| RCCL collectives | 71.0 | 4.7% |
| copies and fills | 48.6 | 3.2% |
| norm, RoPE | 28.9 | 1.9% |
| optimizer: HIP Adam and the clipping l2norm | 28.2 | 1.9% |
| other: MoE permute and router, cross entropy, the rest | 43.0 | 2.9% |
| total kernel time / GPU span | 1499.7 / 1693.7 | |

- 105 ms of the elementwise time is 1024 fp32 `add_` kernels from Megatron's DDP gradient hook,
  which accumulate each micro-batch's gradients into the fp32 `main_grad`
  (`gradient_accumulation_fusion` is off).
- The GPU is idle for 194 ms of the profiled step; the profiler accounts for part of that.
- The trace also has 24 host-to-device copies of 32 KB, the input batch, on stream 0. Their 518 ms
  is almost all waiting behind the compute on stream 1, so they are not in the table.
- The four ranks differ by under 3% in MSA and expert GEMM time; routing is uniform.

Switches on the baseline, one run each:

| Switch | ms/iter | Tokens/s/GPU | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- | --- |
| none (baseline) | 1536.6 | 21,325 | 280.2 GB | 8.19595 |
| `MBS=2` (4 micro-batches of 2 per rank) | 1384.4 | 23,669 | 301.6 GB | 8.14196 |
| `--overlap_grad_reduce True` | 1551.6 | 21,119 | 280.4 GB | 8.19586 |
| `--overlap_grad_reduce True --overlap_param_gather True` | 1555.3 | 21,068 | 280.4 GB | 8.19586 |
| `--gradient_accumulation_fusion True` | 1762.4 | 18,593 | 287.5 GB | 8.19598 |

- `MBS=2` gives 11% more tokens/s for 21 GB more memory. Each expert then gets 1024 tokens per
  micro-batch, as at EP8 with MBS 1.
- The overlaps stay about 1% slower with 8 micro-batches per step. The gradient reduction happens
  once per step, and all collectives together take 71 ms of the profiled step.
- `gradient_accumulation_fusion` is 15% slower, as on one GPU (14–16%); not profiled at EP4.

## Results on `dev/mi455x_v3` c19456f1

c19456f1 has Megatron-LM core_v0.19.2 (4b4acac9) and Primus's Triton AdamW. These runs use the
defaults (HIP fused Adam, NaN check off) on node B, the other shared 4x MI455X node, one
run at a time on an otherwise idle node. Every run finished 10 iterations with 0 NaN, and the fused
Adam patch logged that it was enabled.

| GPUs | Layers | MBS / GBS | ms/iter | Tokens/s/GPU | TFLOP/s/GPU | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1 / 1 | 142.4 | 28,757 | 474.4 | 237.7 GB | 10.98864 |
| 2 (EP2) | 4 | 1 / 2 | 130.1 | 31,493 | 519.7 | 177.9 GB | 10.31194 |
| 4 (EP4) | 4 | 1 / 4 | 121.6 | 33,684 | 555.8 | 147.3 GB | 9.66907 |
| 4 (EP4) | 8 | 1 / 4 | 254.9 | 16,068 | 426.1 | 280.0 GB | 9.11894 |

Repeats were within 0.4%: 142.4–142.9 ms on one GPU over three runs (two of them beside other runs on
the other GPUs), 130.1–130.2 ms on EP2, 121.6 ms twice at EP4 and 254.9–255.1 ms with 8 layers. On
the same node with the same defaults, the old base (a624bb95 plus this example, Megatron-LM d3528a21)
ran at 142.3, 130.5, 121.9 and 255.5 ms with the same losses to the last logged digit, and used 4–9 GB
more memory per GPU. The node matters more than the base: on node A the defaults
ran at 133.8 ms on one GPU and 126.8 ms on EP2 (old base), and 132.3 ms on one GPU (this branch merged
into `dev/mi455x_v3` db5fe1fd). Compare runs on one node.

Where the EP4 step goes: GPU kernel time of profiled iteration 7 on rank 0 (`PROFILE=True`), in ms.

| Kernels | EP4, 4 layers | EP4, 8 layers |
| --- | --- | --- |
| hipBLASLt GEMMs: dense layers, attention projections, shared experts, output layer | 39.7 | 50.7 |
| RCCL collectives | 17.0 | 25.4 |
| elementwise and reductions | 16.8 | 41.4 |
| TE unfused softmax in the three dense attention layers | 13.4 | 13.5 |
| FlyDSL MSA and indexer | 9.6 | 45.5 |
| optimizer: HIP Adam and the clipping l2norm | 8.7 | 33.0 |
| Primus-Turbo grouped GEMM (Triton), all expert GEMMs | 5.5 | 26.4 |
| copies and fills | 3.4 | 9.4 |
| norm, RoPE | 2.3 | 3.7 |
| other: MoE permute and router, cross entropy, the rest | 2.2 | 4.5 |
| total kernel time / GPU span | 118.6 / 123.1 | 253.4 / 276.3 |

The expert GEMMs are Primus-Turbo's Triton kernels (`_grouped_bf16_persistent_gemm_kernel` and
`_grouped_variable_k_gemm_kernel`, under `PrimusTurboColumnParallelGroupedLinear` and
`PrimusTurboRowParallelGroupedLinear`), not TE's grouped GEMM. In the trace every FlyDSL kernel is
named `k_fn_0`; the launching Python frame tells MSA forward, MSA backward and the indexer apart.

The three Adam paths, on the same node. The runs in each column ran side by side: one GPU each on
separate GPUs, and EP2 on GPUs 0–1 and 2–3.

| Adam | Settings | 1 GPU | EP2 | lm loss, iteration 10 (1 GPU / EP2) |
| --- | --- | --- | --- | --- |
| Primus HIP fused Adam (default) | `PRIMUS_FUSED_ADAM_CLIP=1` | 142.9 ms | 130.1 ms | 10.98864 / 10.31194 |
| Primus Triton AdamW | `PRIMUS_FUSED_ADAM_CLIP=0` | 157.5 ms | 138.5 ms | 10.98862 / 10.31211 |
| TE FusedAdam | `PRIMUS_FUSED_ADAM_CLIP=0 --use_triton_fused_adam False` | 157.9 ms | not run | 10.98837 / – |

Triton AdamW is 10% slower than the HIP Adam on one GPU and 6.5% slower on EP2, and on one GPU no faster
than TE's FusedAdam, so the default stays `PRIMUS_FUSED_ADAM_CLIP=1`. Triton AdamW and TE FusedAdam
log the grad norm (10.378 and 10.379 at iteration 10 on one GPU); the HIP Adam does not.

The unit tests pass on gfx1250: `tests/unit_tests/backends/megatron/test_minimax_m3_*.py`,
`test_fused_adam_clip_patches.py` and `test_l2norm_large_tensor_patches.py`, 131 items, 43 of them
the FlyDSL tests (forward and backward against eager, the indexer, and the `MinimaxSparseAttention`
module).

## FlyDSL attention in the dense layers

One GPU of node A (4 layers, EP1, GBS 8, so 8 micro-batches per step as on the
baseline), this branch at 6ee53b2c, with the image's Primus-Turbo or Primus-Turbo `dev/mi455x_v3`
ff4898e4 through `TURBO_DIR`. The runs alternated, one at a time; every one finished 10 iterations
with 0 NaN.

| Dense attention | Primus-Turbo | ms/iter | TFLOP/s/GPU | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- | --- | --- |
| TE (unfused softmax) | image | 824.0 | 655.9 | 238.0 GB | 10.01469 |
| TE (unfused softmax) | ff4898e4 | 823.6, 822.8 | 656.2, 656.8 | 238.0 GB | 10.01467, 10.01467 |
| Primus-Turbo FlyDSL | ff4898e4 | 643.2, 644.6 | 840.3, 838.3 | 230.3 GB | 10.01465, 10.01455 |

- FlyDSL attention takes the step from 823 to 644 ms, 22% faster, and 7.7 GB less memory: no fp32
  score matrix and no separate softmax.
- Swapping the Primus-Turbo checkout alone changes nothing (824.0 against 823.6 ms).
- Its fused backward accumulates dQ with fp32 atomics, so runs are not bitwise identical; the losses
  agree to five digits over 10 iterations.
- During the first TE run on ff4898e4, short-lived processes of another user were on the same GPU
  before iteration 1 finished; the timed iterations 3–10 match the repeat.
- On the EP4 baseline it saves 191 ms per step, 18.7%, and 5.3 GB (Baseline with the current
  defaults): 106 ms of TE softmax and 90 ms of score and value GEMMs per profiled step become 21 ms.

## MSA backward: two tokens per dK/dV WMMA

On the baseline, MSA backward is the largest MSA cost: 41–42 ms of kernels per micro-batch on rank 0
(five layers, about 8.2 ms each, the indexer included) against 11.2–11.5 ms for the forward. The gfx1250
dK/dV kernel ran its dS and P contractions as a K=32 WMMA with only the low K=16 filled, one token at
a time, and read Q and dO from LDS one element per load. It now gives each lane half one of two
tokens, so they fill K=32, and keeps Q and dO transposed in LDS, where a lane's 16 rows of one column
are two 16-byte loads.

At the proxy layer shape (S=4096, B=1, 64 query / 4 KV heads, top-16 blocks of 128), CUDA events, 5
iterations after 2 warmups, one GPU, twice each:

| Op | before | after |
| --- | --- | --- |
| attention forward | 1.25 ms | 1.25 ms |
| attention backward | 7.35, 7.34 ms | 3.01, 3.01 ms |
| indexer forward | 0.34 ms | 0.32–0.34 ms |
| indexer backward | 0.81–0.88 ms | 0.79–0.86 ms |

End to end on one GPU with FlyDSL dense attention (as in the previous section: 4 layers, one of them
MSA, GBS 8), alternating:

| MSA backward | ms/iter | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- |
| before | 643.6, 643.5 | 230.3 GB | 10.01477, 10.01462 |
| two tokens per WMMA | 608.4, 608.6 | 230.3 GB | 10.01461, 10.01473 |

- 35 ms per step, 4.3 ms per MSA layer and micro-batch, as the operator timing predicts. On the
  baseline that would be about 175 ms per step (five MSA layers, 8 micro-batches). There, with the
  next two sections, MSA and indexer kernel time fell from 368 to 115 ms per profiled step (Baseline
  with the current defaults).
- The backward is now 2.4 times the forward. The losses differ in the fourth decimal between repeats
  of the same tree, as with FlyDSL dense attention alone (its dQ atomics); the MSA backward itself
  has no atomics.

## MSA forward and dQ: two key tiles per WMMA

The forward's PV and the backward dQ kernel's dS·K had the same pattern as dK/dV before: 16 keys a
step in the low half of a K=32 WMMA, the transposed operand read from LDS one element per load. They
now take 32 keys a step as two 16-key tiles, one per lane half, and keep V (forward) and K (dQ)
transposed in LDS.

Operator timing as in the previous section, against 6adec738, alternating, twice each:

| Op | before | after |
| --- | --- | --- |
| attention forward | 1.25, 1.25 ms | 0.72, 0.71 ms |
| attention backward | 3.01, 3.01 ms | 1.65, 1.65 ms |
| indexer forward | 0.32, 0.33 ms | 0.31, 0.34 ms |
| indexer backward | 0.83, 0.78 ms | 0.77, 0.71 ms |

End to end, one GPU, 4 layers, GBS 8, FlyDSL dense attention, alternating:

| MSA kernels | ms/iter | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- |
| 6adec738 | 608.9, 608.6 | 230.3 GB | 10.01466, 10.01473 |
| two key tiles per WMMA | 593.1, 593.8 | 230.3 GB | 10.01468, 10.01467 |

- 15 ms per step, 1.9 ms per MSA layer and micro-batch, as the operator timing predicts. On the
  baseline (five MSA layers, 8 micro-batches) the backward part alone would be about 54 ms per step;
  Baseline with the current defaults has the MSA changes together. The baseline's MSA forward is
  bound by kernel launches, so its share may not show.
- The indexer kernels are unchanged.

## hipBLASLt tuning

The launcher sets `HIPBLASLT_TUNING_OVERRIDE_FILE` to
`hipblaslt_tuning/minimax_m3-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt` (empty: hipBLASLt's
own choice). It has one row for each of the 45 hipBLASLt GEMM shapes of a one-GPU run at MBS 1 and
sequence 4096, with and without `gradient_accumulation_fusion`: the dense layers, the attention and
indexer projections, the shared experts, the router and the output layer. The experts' grouped GEMM is
Primus-Turbo's and not in it. With TP 1 every rank of an EP run has the same shapes. hipBLASLt uses a row
only while its solution index still names the kernel it was tuned with (the file name gives the
hipBLASLt commit and ROCm version), so other builds and shapes fall back to its default. With Primus's
own tuning flow (`PRIMUS_HIPBLASLT_TUNING=1`, see `examples/README.md`) the launcher sets no default.

The file was made by logging the GEMMs of a short one-GPU run with `HIPBLASLT_LOG_MASK=32`, once
without and once with `--gradient_accumulation_fusion True`, and searching them with
`examples/megatron/guides/offline_tune/offline_tune_gemm.py` and a `hipblaslt-bench` built as in the
GPT-OSS example.

One GPU, 4 layers, GBS 8, FlyDSL dense attention, with the dK/dV change but not the previous section's;
the runs alternated in the order of the rows, then the first two rows again:

| hipBLASLt kernels | `gradient_accumulation_fusion` | ms/iter | TFLOP/s/GPU | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- | --- | --- |
| hipBLASLt's own | off | 608.4, 609.3 | 888.2, 887.0 | 230.3 GB | 10.01459, 10.01463 |
| this file | off | 581.7, 580.3 | 929.1, 931.2 | 230.3 GB | 10.01475, 10.01476 |
| hipBLASLt's own | on | 823.8 | 656.0 | 235.2 GB | 10.01474 |
| this file | on | 803.1 | 672.9 | 235.2 GB | 10.01461 |

- 28 ms per step, 4.6%.
- `gradient_accumulation_fusion` stays off: its fp32-output weight-gradient GEMMs are slow even with
  the best kernel hipBLASLt has for them (for 6144 x 24576 x 4096 into fp32 with beta 1, about 590
  TFLOP/s against about 2300 for the bf16 output).
- On the EP4 baseline it saves 29 ms per step, 2.8% (Baseline with the current defaults).

## MSA forward: fewer kernel launches

In the baseline's profile the MSA forward is bound by kernel launches rather than by the GPU. Three
changes cut the PyTorch ops it launches, each with bitwise the same results:

- A layer's four RoPE calls (query, key, and the indexer's query and key) share one cos/sin table
  (`minimax_m3/rope.py`); Megatron's `_apply_rotary_pos_emb_bshd` recomputes it on every call.
- The indexer's forced-block ids depend on the shapes only and are cached.
- The sparse indexer loss fills its masks in place, on tensors no backward saves, and its `where`
  takes a scalar.

One MSA layer's training forward at the proxy shape on one GPU, timed without a sync from an idle GPU
(median of 30), twice each: 2.33, 2.30 ms against 2.07, 2.10 ms of CPU time, and 126 against 97 aten
ops that launch a kernel. The baseline runs 40 MSA layer forwards a step, so up to about 10 ms; the
one-GPU run's 8 are too few to show end to end.

## Expert grouped GEMM on FlyDSL

With the previous sections on one GPU, the expert GEMMs were 101.6 of 559.0 ms of kernel time in a
profiled step. Primus-Turbo's Triton kernels ran the weight gradients at 229–271 TFLOP/s and the
forward and data gradients at 650–700, against 2000–2900 for the hipBLASLt GEMMs. The launcher now
defaults to `TURBO_GROUPED_GEMM_BACKEND=bf16:flydsl`, Primus-Turbo's FlyDSL grouped GEMM.

Its weight-gradient backend declines more than 64 experts (`GroupedGEMMVariableKFlyDSLBackend.MAX_G`),
and Primus-Turbo raises when the backend it was told to use declines, so one GPU's 128 local experts
stopped at the first backward. Primus-Turbo's comment there gives 64 as a measured limit on gfx950 and
says the gfx1250 kernel was measured correct up to 160. The `megatron.turbo.flydsl_grouped_gemm_max_groups`
patch (`primus/backends/megatron/patches/turbo/flydsl_grouped_gemm_patches.py`) raises the cap to the
local expert count on gfx1250, up to 160, when the backend is FlyDSL. From EP2 on there are at most 64
local experts and it changes nothing.

One MoE layer's fc1 (6144 to 6144) and fc2 (3072 to 6144), forward and backward, through
`primus_turbo.pytorch.ops.grouped_gemm` with balanced routing; CUDA events, one GPU, twice each:

| Backend | 128 experts x 128 rows (one GPU) | 32 experts x 512 rows (EP4) |
| --- | --- | --- |
| Triton (`default`) | 11.77, 11.79 ms | 6.55, 6.55 ms |
| hipBLASLt | 15.44, 15.45 ms | not run |
| FlyDSL | 5.46, 5.44 ms | 4.55, 4.58 ms |

The FlyDSL weight gradients run at 612–626 TFLOP/s at the one-GPU shape and the forward and data
gradients at 1330–1660. Against an fp32 reference all three backends have the same relative error,
1.7e-3, which is the bf16 rounding of the output.

End to end, one GPU, 4 layers, GBS 8, the launcher's defaults on 105e9f33 plus this change:

| Primus-Turbo | Expert grouped GEMM | ms/iter | TFLOP/s/GPU | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- | --- | --- |
| `TURBO_DIR` (FlyDSL dense attention) | Triton | 565.3 | 956.0 | 230.3 GB | 10.01476 |
| `TURBO_DIR` (FlyDSL dense attention) | FlyDSL | 513.1, 515.3, 513.5 | 1053.3, 1048.8, 1052.5 | 230.3 GB | 10.01468, 10.01467, 10.01470 |
| the image's (TE dense attention) | Triton | 743.5 | 726.9 | 238.0 GB | 10.01456 |
| the image's (TE dense attention) | FlyDSL | 692.0 | 780.9 | 238.0 GB | 10.01456 |

- 52 ms per step, 9%: 6.5 ms per MoE layer and micro-batch, as the operator timing predicts.
- With TE's dense attention the run has no atomics, and its loss is the same to the last logged digit
  with either kernel.
- FlyDSL compiles its kernels in iterations 1 and 2, which take about 6 and 3 s longer.
- `--gradient_accumulation_fusion True` needs `TURBO_GROUPED_GEMM_BACKEND=default`: FlyDSL's
  weight-gradient GEMM has no beta=1 epilogue, and Primus-Turbo would raise at the first backward. The
  launcher stops before training instead.
- On the EP4 baseline it saves 75 ms per step, 7.3% (Baseline with the current defaults), about the
  80 ms the operator timing gives (2.0 ms per MoE layer and micro-batch, five layers, 8
  micro-batches). Expert GEMM kernel time there fell from 215 to 116 ms per profiled step.

## Primus-Turbo 4c74631 image

Until tas-20261109 image below, the launcher defaulted to
`amdprimus/amdprimus:tas-20261019-turbo-4c74631`, which needs no `TURBO_DIR` for FlyDSL dense attention.
One GPU of node B (4 layers, EP1, GBS 8), the
launcher's defaults on c7034e33 apart from the image, against dc23c84c, the tree before FlyDSL
attention in the dense layers. The runs alternated one at a time on one GPU, Triton experts last; the
c7034e33 runs with tas-20261019's own Primus-Turbo ran earlier the same day. Every run finished 10
iterations with 0 NaN.

| Tree | Image | Primus-Turbo | Dense attention | Experts | ms/iter | TFLOP/s/GPU | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dc23c84c | tas-20261019 | the image's | TE | Triton | 854.3, 854.8 | 632.6, 632.2 | 237.8 GB | 10.01467, 10.01467 |
| c7034e33 | tas-20261019 | the image's | TE | FlyDSL | 719.0, 718.5 | 751.6, 752.2 | 237.9 GB | 10.01456, 10.01456 |
| c7034e33 | tas-20261019 | ff4898e4 (`TURBO_DIR`) | FlyDSL | FlyDSL | 532.7, 532.2 | 1014.6, 1015.4 | 230.2 GB | 10.01469, 10.01467 |
| c7034e33 | tas-20261019-turbo-4c74631 | the image's | FlyDSL | FlyDSL | 527.2, 527.5 | 1025.1, 1024.6 | 230.2 GB | 10.01470, 10.01487 |
| c7034e33 | tas-20261019-turbo-4c74631 | the image's | FlyDSL | Triton | 590.2 | 915.7 | 230.2 GB | 10.01473 |

- The 4c74631 image runs 1.0% faster than ff4898e4 through `TURBO_DIR` (527.4 against 532.5 ms) with
  the same memory; the losses agree to five digits, as between repeats with FlyDSL dense attention.
- Against dc23c84c, c7034e33 takes 37.7% less time per step, as on node A
  (824.0 against 513–515 ms). This node is 3.6–3.7% slower on both trees; compare runs on one node.
- FlyDSL experts are 10.6% faster than Triton's on 4c74631 (527.4 against 590.2 ms).
- The unit tests in Results on `dev/mi455x_v3` c19456f1, with `test_flydsl_grouped_gemm_patches.py`,
  pass on both images: 149 items.

## Baseline with the current defaults

The baseline (EP4, 8 layers, GBS 32) on the four GPUs of node B: the launcher's
defaults on d6c953a9 (the 4c74631 image, FlyDSL dense attention and experts, the hipBLASLt tuning
file, and the MSA kernel changes above) against those of dc23c84c (tas-20261019, TE dense attention,
Triton experts, hipBLASLt's own kernels). One run at a time, each started only when no other process
was on the GPUs: the two trees alternated, then the switches on d6c953a9, one run each. Every run
finished 10 iterations with 0 NaN.

| Tree | Switch | ms/iter | Tokens/s/GPU | TFLOP/s/GPU | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- | --- | --- | --- |
| dc23c84c | none | 1574.8, 1576.0 | 20,807, 20,792 | 551.8, 551.4 | 280.1 GB | 8.19595, 8.19595 |
| d6c953a9 | none | 1026.5, 1021.6 | 31,923, 32,074 | 846.5, 850.5 | 274.8 GB | 8.19573, 8.19577 |
| d6c953a9 | `USE_TURBO_ATTENTION=False` (TE dense attention) | 1215.1 | 26,968 | 715.1 | 280.1 GB | 8.19595 |
| d6c953a9 | `TURBO_GROUPED_GEMM_BACKEND=default` (Triton experts) | 1099.1 | 29,814 | 790.6 | 274.8 GB | 8.19566 |
| d6c953a9 | `HIPBLASLT_TUNING_OVERRIDE_FILE=` (hipBLASLt's own kernels) | 1053.2 | 31,114 | 825.0 | 274.8 GB | 8.19565 |

- d6c953a9 takes 35.0% less time per step than dc23c84c (1024.1 against 1575.4 ms), 1.54 times the
  tokens/s, and 5.3 GB less memory per GPU.
- Against d6c953a9, TE dense attention costs 191.1 ms per step (18.7%) and the 5.3 GB, Triton experts
  75.1 ms (7.3%; the operator timing predicted about 80), and hipBLASLt's own kernels 29.2 ms (2.8%).
  The other 256 ms of the 551 ms is the MSA kernel changes (two tokens per dK/dV WMMA, two key tiles
  per WMMA, fewer launches), for which the one-GPU runs predicted up to about 240 ms.
- With TE's dense attention, d6c953a9's loss is within 1e-6 of dc23c84c's (8.195952 against
  8.195951). With FlyDSL's, whose dQ uses fp32 atomics, the losses differ in the fifth decimal
  between repeats.
- dc23c84c is 2.5% slower here than on node A (1575.4 against 1537.4 ms,
  Baseline); compare runs on one node.
- During the first run of each tree, short-lived processes of another user were on the GPUs; those
  runs are within 0.5% of the second ones.

Where the step goes: GPU kernel time of profiled iteration 7 (`PROFILE=True --profile_ranks
'[0,1,2,3]'`), in ms, the mean of the four ranks. Profiling makes the step 1207 ms instead of 1024 on
d6c953a9 and 1739 instead of 1575 on dc23c84c.

| Kernels | dc23c84c | d6c953a9 |
| --- | --- | --- |
| hipBLASLt GEMMs: dense layers, attention and indexer projections, shared experts, router, output layer | 324.4 | 279.2 |
| elementwise and reductions | 241.8 | 233.2 |
| expert grouped GEMM: Primus-Turbo Triton (dc23c84c), FlyDSL (d6c953a9) | 214.7 | 116.0 |
| FlyDSL MSA and indexer | 368.0 | 115.2 |
| RCCL collectives | 90.3 | 102.1 |
| copies and fills | 49.1 | 45.1 |
| optimizer: HIP Adam and the clipping l2norm | 33.4 | 33.5 |
| norm, RoPE | 29.5 | 29.3 |
| stream memory operations (`__amd_rocclr_batchMemOp`) | 24.5 | 21.5 |
| dense attention: TE's unfused softmax (dc23c84c), Primus-Turbo FlyDSL (d6c953a9) | 106.1 | 20.9 |
| dense attention: TE's score and value GEMMs (`bmm`, `baddbmm`) | 90.0 | – |
| MoE permute and router | 18.1 | 18.5 |
| cross entropy, the rest | 7.2 | 7.5 |
| total kernel time / GPU span | 1597.1 / 1738.7 | 1022.1 / 1206.6 |

- MSA and the indexer take 14.4 ms per micro-batch instead of 46.0, and the three dense attention
  layers 2.6 ms instead of 24.5.
- The hipBLASLt GEMMs outside dense attention take 45 ms less. Of the changes, only the tuning file
  touches them, and it saves 29 ms end to end.
- The rest of the kernel time changed little, so it is now a larger share: 108 ms of the elementwise
  time is still the 1025 fp32 `add_` kernels of Megatron's DDP gradient hook (see Baseline), 10.6% of
  the kernel time.
- The ranks differ by up to 5% in the expert GEMMs and up to 3% in MSA and the hipBLASLt GEMMs;
  routing is uniform. They differ more in RCCL time (75–115 ms on either tree), stream memory
  operations and idle time, which include waiting for the other ranks; the least RCCL time on any
  rank is 74.8 ms on both trees.
- In d6c953a9's trace the Primus-Turbo FlyDSL kernels have names: `k_flash_attn_fwd`,
  `k_dkdv_fused_sp_0`, `k_delta_bshd_0`, `k_dq_cvt_0` and `k_redsp_0` for dense attention, and
  `kernel_grouped_nt_0` and `kernel_grouped_variable_k_0` for the experts. Primus's MSA kernels are
  still all `k_fn_0`.

## tas-20261109 image

The launcher defaults to `amdprimus/amdprimus:tas-20261109`, whose Primus-Turbo adds the gfx1250 grouped
GEMM tuning for 512 rows per expert (Primus-Turbo PR #627), M3's expert shapes at EP4. The baseline (EP4, 8
layers, GBS 32) and M3.1-Flash at the same shape, on the four GPUs of node C (another 4x MI455X node), with
the launcher's defaults apart from the image, inside a container (`PRIMUS_LAUNCHER=direct`), one run at a
time. The tree is this branch with Primus `main` merged in, which brings `main` PR #1370: it makes the
`msa_indexer_loss` reporting work again, whose all-reduce over the data-parallel group without context
parallel creates one more RCCL communicator, about 8.6 GB more memory per GPU than in the sections above at
the same speed and loss. Every run finished 10 iterations with 0 NaN.

| Model | Image | MoE dispatch | ms/iter | Tokens/s/GPU | TFLOP/s/GPU | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M3 | tas-20261019-turbo-4c74631 | all-to-all | 1013.9, 1003.6, 1018.1 | 32,319, 32,649, 32,186 | 857.0, 865.7, 853.5 | 283.25 GB | 8.19562, 8.19571, 8.19569 |
| M3 | tas-20261109 | all-to-all | 988.8, 986.9 | 33,139, 33,204 | 878.8, 880.4 | 283.26 GB | 8.19580, 8.19570 |
| M3 | tas-20261109 | DeepEP | 946.1 | 34,634 | 918.4 | 273.27 GB | 8.19581 |
| M3.1-Flash | tas-20261019-turbo-4c74631 | all-to-all | 1095.0 | 29,924 | 790.3 | 284.91 GB | 8.89782 |
| M3.1-Flash | tas-20261109 | all-to-all | 1069.7, 1074.6 | 30,634, 30,494 | 809.0, 805.4 | 284.92 GB | 8.89782, 8.89789 |
| M3.1-Flash | tas-20261109 | DeepEP | 1027.7 | 31,885 | 842.1 | 273.40 GB | 8.89797 |

- tas-20261109 takes 2.4% less time per step than the 4c74631 image on M3 (means: 987.9 against 1011.9
  ms) and 2.1% on M3.1-Flash (1072.2 against 1095.0 ms); Primus-Turbo PR #627 measured 2.0% for M3. The
  losses differ as between repeats.
- DeepEP is `--use_turbo_deepep True --turbo_deepep_num_cu 64` (Primus-Turbo's FlyDSL DeepEP, `flydsl_v2`,
  on 64 CUs), which the launcher does not set. Against all-to-all on the same image it takes 4.2% less
  time per step on M3 (946.1 against 987.9 ms) and 4.1% on M3.1-Flash (1027.7 against 1072.2 ms), with
  10.0 and 11.5 GB less memory per GPU.

## MiniMax-M3.1-Flash

[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)'s text config is M3's with
three changes: MSA in every layer (`sparse_attention_freq` and `sparse_disable_index_value` all 1; the
first three layers keep their dense MLPs), `rope_theta` 1e7 instead of 5e6, and no MTP fields. The
model preset `primus/configs/models/megatron/minimax_m3.1_flash.yaml` extends `minimax_m3.yaml`, and
`examples/megatron/configs/MI455X/minimax_m3_1_flash-BF16-pretrain.yaml` extends the M3 experiment,
so the launcher runs it with `EXP` set (see Run). With no dense attention layer, `USE_TURBO_ATTENTION`
has nothing to act on, and the recomputed first layer is an MSA layer.

The baseline shape (EP4, 8 layers, GBS 32) on node B, the launcher's defaults,
the two models alternating one at a time on the four otherwise idle GPUs. Every run finished 10
iterations with 0 NaN. These ran before the preset took its current file name in a79087ef; after the
rename, one GPU gives the same loss and grad norm at every iteration.

| Model | ms/iter | Tokens/s/GPU | TFLOP/s/GPU | Memory/GPU | lm loss, iteration 10 |
| --- | --- | --- | --- | --- | --- |
| M3 | 1024.2, 1025.1 | 31,995, 31,966 | 848.4, 847.6 | 274.8 GB | 8.19576, 8.19572 |
| M3.1-Flash | 1110.4, 1101.6 | 29,512, 29,747 | 779.4, 785.6 | 276.5 GB | 8.89782, 8.89782 |

- M3.1-Flash takes 81 ms (7.9%) more per step and 1.7 GB more memory: each of its three extra MSA
  layers costs 3.4 ms per micro-batch more than FlyDSL dense attention. On one GPU (4 layers, GBS 8)
  it ran at 602.9 ms against M3's 527.4 (Primus-Turbo 4c74631 image), 3.1 ms per extra MSA layer and
  micro-batch.
- At sequence 4096 an MSA query attends 1504.5 keys against 2048 for dense causal attention, so MSA
  saves little work there, and its kernels and indexer cost more than FlyDSL's dense attention. At
  32k it would be 1924.5 keys against 16384; not measured.
- M3.1-Flash's loss is higher at iteration 10 (8.898 against 8.196): its first layers also attend
  only the blocks an untrained indexer selects. With mock data and 10 iterations this says nothing
  about the released model. Without the dQ atomics of FlyDSL dense attention, its two runs have the
  same loss to the last digit.
- M3 ran within 0.1% of Baseline with the current defaults (1024.7 against 1024.1 ms).

Where the step goes: GPU kernel time of profiled iteration 7, in ms, the mean of the four ranks; M3's
is from Baseline with the current defaults (d6c953a9, the same code). Profiling makes M3.1-Flash's
step 1301 ms instead of 1106.

| Kernels | M3 | M3.1-Flash |
| --- | --- | --- |
| FlyDSL MSA and indexer | 115.2 | 190.9 |
| Primus-Turbo FlyDSL dense attention | 20.9 | – |
| elementwise and reductions | 233.2 | 249.0 |
| copies and fills | 45.1 | 56.8 |
| hipBLASLt GEMMs | 279.2 | 281.8 |
| expert grouped GEMM (FlyDSL) | 116.0 | 116.5 |
| RCCL collectives and stream memory operations | 123.6 | 147.8 |
| the rest: optimizer, norm and RoPE, MoE permute and router, cross entropy | 88.8 | 89.5 |
| total kernel time / GPU span | 1022.1 / 1206.6 | 1132.2 / 1301.2 |

- MSA kernels take 2.9–3.0 ms per layer and micro-batch on either model, the recomputed first layer
  included, so the three extra MSA layers add 75.7 ms of kernels for the 20.9 ms of dense attention
  they replace. Elementwise kernels and copies add another 27.5 ms.
- By module on rank 0, attention takes 278.5 ms against M3's 196.1 (82 ms more, of which 55 ms is
  the attention kernels): 34–35 ms per MSA layer per step against 9 per dense attention layer.
- RCCL and stream memory time differ by rank as on M3 (RCCL 91–138 ms); they include waiting for the
  other ranks.

## How the defaults were chosen

On the older bases dad0a081 and a624bb95 (Megatron-LM d3528a21), one GPU unless noted, with TE's
FusedAdam and the NaN check on unless noted, on node A. Every run finished with
0 NaN.

**Optimizer.** Megatron's step reads and writes the FP32 gradients and parameters (10.9 B on the
1-GPU proxy) several times: a norm over every gradient for the NaN check, the 1/DP gradient scale,
the clipping l2norm, the clip scale, Adam, and the BF16 parameter copy. At EP4 with 8 layers, these
passes, the gradient accumulation and zeroing took about 77 of the 281 ms of kernel time.
Cumulative, on a624bb95:

| Change | ms/iter |
| --- | --- |
| none: TE FusedAdam, NaN check on | 156.0 |
| `check_for_nan_in_loss_and_grad: false` | 150.6 |
| `PRIMUS_FUSED_ADAM_CLIP=1` | 140.2 |
| `PRIMUS_FUSED_ADAM_BF16_WRITEBACK=1`, `PRIMUS_STATIC_BF16_LOSS_SCALE_CPU=1` (the defaults) | 133.8 |

EP2 went from 138.2 to 126.8 ms. Iteration-10 losses moved from 10.98863 to 10.98864 on one GPU and
from 10.31213 to 10.31194 on EP2, because the clip coefficient is computed in FP32 on the device
rather than in Python. Profiled, Adam is one 27.6 ms kernel for both parameter groups and the
clipping l2norm 2.7 ms; the scale passes, the NaN-check norm and the parameter copy are gone.

**Training settings** (TE's grouped GEMM for the first two):

- `manual_gc_interval`. Megatron's `manual_gc` with interval 1 calls `gc.collect()` every step. With
  about 583,000 tracked Python objects after startup that takes about 280 ms, more than the step, and
  frees nothing. Interval 1 to 100: 463.9 to 183.6 ms on one GPU, 545.1 to 150.7 ms on EP2. The yaml
  uses 100. The MI355X M3 yamls use 1; their effect there was not measured.
- `overlap_grad_reduce` and `overlap_param_gather` on to off: 183.6 to 167.4 ms on one GPU, 150.7 to
  142.6 ms on EP2. The yaml turns them off. On the EP4 baseline they are about 1% slower (Baseline).
- `gradient_accumulation_fusion: true`: 183.6 to 209.8 ms with TE's grouped GEMM (overlap on), 158.2
  to 184.3 ms with Primus-Turbo's. The fusion is not what costs; the fp32 output is. With it, each
  dense weight gradient is a bf16 x bf16 GEMM written straight into the fp32 `main_grad` with beta 1.
  For that hipBLASLt on gfx1250 picks a `BSS` MT256x128x64 kernel: 35.4 ms a step, in place of about
  9.6 ms of the bf16-output kernels. The Triton expert weight-gradient GEMM goes from 7.0 to 16.5 ms
  with its fp32 accumulate epilogue. The fp32 `+=` it saves is 8.6 ms (profiled iteration 7, 163.9
  to 191.9 ms). Into a bf16 `main_grad` (`--main_grads_dtype bf16`, see Switches on one GPU)
  hipBLASLt keeps the same bf16 kernels (+0.6 ms for beta 1), the Triton GEMM adds 2.3 ms, the bf16
  `+=` saves 4.2 ms, and the step goes from 134.0 to 132.8 ms. The yaml leaves it off. Tuning
  hipBLASLt for the fp32-output shapes does not change that (hipBLASLt tuning).

**Expert grouped GEMM.** Primus-Turbo's grouped GEMM (Triton) against TE's (`USE_TURBO_GROUPED_GEMM=False`,
128 hipBLASLt launches per GEMM): 158.2 against 167.4 ms on one GPU, 139.0 against 142.6 ms on EP2,
and 337.4 against 368.8 ms on EP2 with 8 layers, losses within 0.001. It keeps a layer's experts in
one weight, which needs the `te_l2norm_large_tensor` patch (Known issues). At EP2,
`TURBO_GROUPED_GEMM_BACKEND=bf16:flydsl` ran at 136.3 ms, bit-identical to Triton over 10 iterations;
a standalone run at the EP2 shapes shows it does run the FlyDSL forward and weight-gradient kernels.

**MSA and indexer only**, at the proxy layer shape (S=4096, B=1, 64 query / 4 KV heads, top-16
blocks of 128, 4 index heads of dim 128). CUDA events, 5 iterations after 2 warmups, one GPU.
Eager backward is (forward+backward) minus a separately timed forward. Peak is allocator memory for
that call, not the training step's `hip mem usage`.

| Op | flydsl | eager |
| --- | --- | --- |
| attention forward | 1.26 ms, 0.36 GiB | 17.68 ms, 21.50 GiB |
| attention backward | 7.40 ms, 0.61 GiB | 20.30 ms, 24.89 GiB |
| indexer forward | 0.30 ms | 0.57 ms |
| indexer backward | 0.72 ms | 1.32 ms |

Attention forward+backward is 8.65 ms versus 37.98 ms. End to end, FlyDSL takes the step from 189.8 to
158.2 ms and saves 16 GB on the 4-layer proxy, where one of the four layers is MSA.

### Switches on one GPU

Each switch on top of the defaults of the time (TE FusedAdam, NaN check on), one GPU, 10 iterations.
The defaults ran at 157.7 ms again during this sweep, so differences under 1% are noise.

| Switch | ms/iter | Memory | lm loss, iteration 10 |
| --- | --- | --- | --- |
| none (defaults) | 158.2 | 241.5 GB | 10.989 |
| `--use_precision_aware_optimizer True --main_grads_dtype bf16 --exp_avg_dtype bf16 --exp_avg_sq_dtype bf16` | 134.0 | 159.5 GB | 10.989 |
| the same and `--gradient_accumulation_fusion True` | 132.8 | 161.9 GB | 10.989 |
| `--use_precision_aware_optimizer True --exp_avg_dtype bf16 --exp_avg_sq_dtype bf16` | 146.0 | 179.8 GB | 10.989 |
| `PRIMUS_RECOMPUTE_NUM_LAYERS=0` | 151.2 | 241.5 GB | 10.989 |
| `USE_TURBO_ATTENTION=True` | 151.3 | 233.8 GB | 10.989 |
| `--use_turbo_autotune True` | 155.8 | 241.5 GB | 10.989 |
| `--use_turbo_norm_te_linear True` | 157.1 | 241.5 GB | 10.989 |
| `--use_turbo_cross_entropy True` | 157.2 | 240.7 GB | 10.989 |
| `--use_turbo_gemm True` | 157.4 | 241.5 GB | 10.989 |
| `--moe_shared_expert_overlap True` | 159.0 | 241.8 GB | 11.002 |
| `--fp8 hybrid` (delayed scaling) | 164.8 | 244.6 GB | 11.389 |
| `USE_TURBO_GROUPED_GEMM=False` | 167.4 | 245.5 GB | 10.989 |
| `--fp8 hybrid --fp8_recipe tensorwise` | 168.4 | 248.3 GB | 11.024 |
| `--gradient_accumulation_fusion True` | 184.3 | 243.0 GB | 10.989 |
| `PRIMUS_MSA_BACKEND=eager` | 189.8 | 257.9 GB | 10.988 |

- The precision-aware optimizer keeps exact FP32 master weights, stored as the BF16 parameter plus a
  16-bit remainder (Megatron's `store_param_remainders`). That and BF16 Adam moments save 62 GB and
  12 ms; a BF16 `main_grad` saves another 20 GB and 12 ms. Profiled with all three, the time comes
  out of TE's Adam (27.8 to 18.7 ms), the gradient `+=` (10.4 to 5.0 ms), the parameter copy after
  the step (5.0 to 0.5 ms) and zeroing the gradient buffer. The losses agree over 10 iterations,
  which says nothing about convergence, and a BF16 `main_grad` also accumulates micro-batches and
  reduces across data-parallel ranks in BF16. Not the default.
- `PRIMUS_RECOMPUTE_NUM_LAYERS=0` does not move the peak on the 4-layer proxy: the recomputed layer
  is dense layer 1, whose activations at sequence 4096 are small next to the weights and optimizer.
- `USE_TURBO_ATTENTION=True` replaces TE's unfused softmax in the three dense layers.
- `--use_turbo_autotune` takes iteration 1 from 28 to 35 s.
- `--moe_shared_expert_overlap` has no dispatch to overlap with on one GPU.
- FP8: with the default `delayed` recipe only TE's dense linears run FP8, and Primus-Turbo logs
  `Primus-Turbo FP8 delayed not work since Primus-Turbo not support delayed scaling`, so the experts
  stay BF16. With `tensorwise` that warning is gone and Primus-Turbo's FP8 path covers the experts too.
  Both are slower than BF16 here and start with a higher loss; neither was profiled.
- `MBS=2` ran at 230.0 ms for twice the tokens: 35,623 tokens/s against 25,887, at 248.5 GB.
- Combined, on a624bb95 (defaults 156.0 ms there): `USE_TURBO_ATTENTION=True`,
  `PRIMUS_RECOMPUTE_NUM_LAYERS=0` and `--use_turbo_autotune True` ran at 145.2 ms, 233.8 GB, loss
  10.989. Adding the precision-aware optimizer with BF16 `main_grad` and moments and
  `--gradient_accumulation_fusion True` gave 122.0 ms and 172.3 GB, loss 10.988.

Accepted, but they do nothing on this model (loss identical to the defaults):

- `--use_turbo_fused_qk_rmsnorm_rope True` (158.6 ms): the fused op takes full, non-interleaved RoPE
  only, and M3's `rotary_percent: 0.5` falls back to Megatron's path.
- `--use_turbo_mega_moe True` (158.5 ms): applies only at EP 2 to 8 (Known issues).
- `turbo_sync_free_moe_stage: 1`: sets `moe_use_fused_router_with_aux_score` and `moe_permute_fusion`,
  which the yaml already sets.

Do not run:

- `--turbo_fused_grouped_gemm True` and `--use_turbo_fused_act_with_probs True` assert
  `glu_linear_offset == 0`; M3's `swigluoai` uses 1.0. The second also rules out sync-free stage 3.
- `--gradient_accumulation_fusion True` with `TURBO_GROUPED_GEMM_BACKEND=bf16:flydsl` (the default):
  the launcher stops; see Expert grouped GEMM on FlyDSL.
- `--fp8 e4m3 --fp8_recipe blockwise` and `--fp8_recipe mxfp8` stop at TE's arch check (`Device arch
  gfx94x or gfx95x is required for FP8 block scaling execution.`, `Device arch gfx95x or newer is
  required for MXFP8 execution.`). TE 2.19.0.dev0 leaves gfx1250 as a TODO until its GEMM is available.
- `--use_turbo_deepep True` and sync-free stages 2 and 3: Known issues.
- Primus-Turbo attention with its default backend: `invalid argument for fmha_bwd`, so the launcher
  always sets `PRIMUS_TURBO_ATTN_BACKEND`.
- `moe_use_legacy_grouped_gemm`, `PRIMUS_TP` above 1, context parallel, a dataloader attention mask,
  and a BF16 router: Known issues.
- `PRIMUS_TOTAL_LAYERS=8` on one GPU, where it does not fit. At EP2 it has 22.4 B parameters per GPU,
  about 380 GB of weights, gradients and Adam state before activations, and fills the device (337.4
  ms); recomputing all 8 layers was 14% slower and did not make room.

## Not yet measured

- Sequence 16384 and 32768. MSA's cost stays on the selected blocks; dense attention does not.
- Pipeline parallel. The launcher allows `PRIMUS_PP`, but no PP run has been done. Tensor parallel
  and context parallel are not implemented for MSA.
- EP8 and more than one node.
- MegaMoE (see Known issues), and DeepEP beyond one run per model at EP4 on 64 CUs (tas-20261109 image).
- A real dataset and real routing. These runs are mock data with uniform routing.
- The full 60-layer model. From the proxy's ~18 bytes/parameter (BF16 weights, FP32 gradients and
  Adam state), a training step at MBS 1, seq 4096 wants on the order of 40 MI455X GPUs (EP8 x PP5).
- The gfx950 MFMA kernels. This node cannot execute them.

## Known issues

- **TE l2norm and tensors of 2^31 elements.** TE's `multi_tensor_l2norm` reads out of bounds when it
  is given more than one tensor and one of them has 2^31 or more elements. Megatron's grad norm
  passes all gradients in one call, and the Primus-Turbo grouped GEMM keeps a layer's experts in one
  weight: fc1 is 128 x 6144 x 6144 = 4.8e9 elements on one GPU and 2.4e9 at EP2. Without a fix the
  first optimizer step faults (`HSA_STATUS_ERROR_MEMORY_FAULT` in `custom_multi_tensor_l2norm_kernel`),
  or, when the stray address happens to be mapped, training runs with a wrong grad norm (11.365
  instead of 11.406 at iteration 1). The `megatron.optimizer.te_l2norm_large_tensor` patch
  (`primus/backends/megatron/patches/te_patches/l2norm_large_tensor_patches.py`) splits such tensors
  into views of 2^30 elements for the norm. TE's Adam and scale kernels handle these sizes. From EP4
  on, M3's stacked expert weights stay under 2^31.
- **DeepEP.** The C++ DeepEP of tas-20261019's Primus-Turbo is written for MI355X (wave64) and has not
  been ported to gfx1250's wave32: with that image, `--use_turbo_deepep True` on EP2 stops in the first
  step with `DeepEP error: CPU recv timeout`; with TE's grouped GEMM it hung there instead. The default
  image's Primus-Turbo runs its FlyDSL DeepEP (`flydsl_v2`) on gfx1250 instead, with the correctness
  fixes of Primus-Turbo PRs #602 and #623; at EP4 on 64 CUs it runs M3 and M3.1-Flash (see tas-20261109
  image). The launcher keeps DeepEP off. Sync-free MoE
  stages 2 and 3 (`turbo_sync_free_moe_stage`) turn DeepEP on and have not been run. Stage 1's two
  settings, `moe_use_fused_router_with_aux_score` and `moe_permute_fusion`, are already in the yaml.
- **FlyDSL expert GEMM with 64 rows or fewer per expert.** In the previous default image, Primus-Turbo
  4c74631's gfx1250 FlyDSL grouped GEMM forward raises `ValueError: head_pipe=0 leaves too few deferred
  matrix ops for the head reads` when the experts average 64 rows or fewer (32 and 64 rows tried: 128
  experts with K and N from 1024 to 6144, and 8 experts at 1024); 128 rows and more run. It picks its kernel by the
  average, which is MBS x sequence x EP / 32 rows here (4 of 128 experts per token), whatever the
  routing: 128 on one GPU and 512 at EP4 at sequence 4096. One GPU at `PRIMUS_SEQ_LENGTH=2048` or
  less stops in the first forward; use `TURBO_GROUPED_GEMM_BACKEND=default` there. tas-20261019's
  Primus-Turbo runs 64-row experts, and so does the default image's: Primus-Turbo PR #611 fixed it
  (one GPU at `PRIMUS_SEQ_LENGTH=2048` ran 77.5 ms/iter with Primus-Turbo `dev/mi455x_v4` 24e989fd, which
  has #611, through `TURBO_DIR`).
- **MegaMoE.** `--use_turbo_mega_moe True` takes effect only at EP 2 to 8. With tas-20261019's
  Primus-Turbo, its FlyDSL `dispatch_prologue_kernel` does not build for gfx1250 at EP2 (`lld
  invocation failed`, then `All autotune configs failed`) in the first forward. The Primus-Turbo of
  the 4c74631 image and of the default image has the gfx1250 BF16 MegaMoE kernels (Primus-Turbo #582);
  they have not been run with M3.
- **gfx1250 MSA kernels.** `msa_backend=flydsl` on gfx1250 needs the wave32 WMMA port. It does not
  use `v_permlane32_swap`, MFMA, or `ds_read_tr16`. gfx950 still uses the original kernels; only the
  dispatch changed. The kernels address buffers with 32-bit offsets, so each tensor must stay under
  2 GiB.
- **Eager MSA memory.** Eager materialises the `[batch, heads, seq, seq]` fp32 score matrix and the
  indexer's `[batch, 4, seq, seq]` scores. At seq 4096 that is about 21 GiB for attention forward
  alone. FlyDSL does not.
- **No tensor parallel, context parallel, or dataloader mask.** `PRIMUS_TP` must be 1. FlyDSL rejects
  a dataloader attention mask (`create_attention_mask_in_dataloader: false`). Packed sequences are
  not supported.
- **Router dtype.** With 128 experts, BF16 router logits NaN on the first step
  (`found NaN in local grad norm for bucket #0`). `moe_router_dtype: fp32` is set in the yaml. With
  `overlap_grad_reduce: false` all gradients are in one bucket, so that message does not name a
  parameter.
- **Layer pattern.** `num_layers`, `moe_layer_freq`, `sparse_attention_freq` and
  `sparse_disable_index_value` must be the same length. The launcher writes all four. The first three
  layers are dense attention and a dense FFN (intermediate 12288); later layers are MSA plus MoE
  (expert intermediate 3072, one shared expert). Do not set `moe_use_legacy_grouped_gemm`: the
  activation is `quick_geglu` (`swigluoai`), which the legacy grouped MLP refuses.
- **RoPE cache.** Leave `max_position_embeddings` at the run's sequence length.
