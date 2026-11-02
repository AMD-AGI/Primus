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
Primus-Turbo. DeepEP stays off (see Known issues).

The default image is `amdprimus/amdprimus:tas-20261019` (see the DeepSeek-V3 example's README),
`amdprimus/amdprimus@sha256:d653324d124e218f43c92eaac3cfc5d05059bd45fb4abf5bdfb78b987af4af87`: torch
2.14.0 on ROCm 10.2.0a20261011, Triton 3.8.0, Transformer Engine 2.19.0.dev0 (bfe94fbe5),
Primus-Turbo 0.6.1.dev2, FlyDSL 0.3.4.1 and AITER 0.1.24.post1. Every run below used that image and
its own Primus-Turbo, except the FlyDSL attention runs, which used a Primus-Turbo checkout through
`TURBO_DIR` (see FlyDSL attention in the dense layers).

## Run

```bash
# On the host, 1 GPU (the default: 4 layers, EP1, GBS 1). Pick a free device on a shared node.
HIP_VISIBLE_DEVICES=1 bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh

# The baseline: 4 GPUs, 8 layers, EP4, GBS 32
GPUS_PER_NODE=4 PRIMUS_TOTAL_LAYERS=8 GBS=32 bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh

# Inside a container
PRIMUS_LAUNCHER=direct bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh
```

Two runs at once on one node need different `MASTER_PORT`s.

**FlyDSL attention for the dense layers.** The image's Primus-Turbo has no gfx1250 FlyDSL attention
for head dim 128; Primus-Turbo `dev/mi455x_v3` has it (PR #580). Build such a checkout in place for
the image once (gfx1250 only, OpenMPI off the paths so rocSHMEM is not linked), then pass it as
`TURBO_DIR`. The launcher puts it first on `PYTHONPATH` and runs the three dense layers on its FlyDSL
attention:

```bash
git clone https://github.com/AMD-AGI/Primus-Turbo.git /path/to/Primus-Turbo
git -C /path/to/Primus-Turbo checkout ff4898e4  # dev/mi455x_v3, the commit measured below
git -C /path/to/Primus-Turbo submodule update --init --recursive
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v /path/to/Primus-Turbo:/path/to/Primus-Turbo \
    -w /path/to/Primus-Turbo amdprimus/amdprimus:tas-20261019 bash -c \
    'PATH=$(tr : "\n" <<< "$PATH" | grep -v openmpi | paste -sd:) GPU_ARCHS=gfx1250 python3 setup.py build_ext --inplace'
HIP_VISIBLE_DEVICES=1 TURBO_DIR=/path/to/Primus-Turbo bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh
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
  `PRIMUS_RECOMPUTE_NUM_LAYERS` (default 1). The yaml's `recompute_method: block` recomputes the
  first that many layers, so the default recomputes dense layer 1, not an MSA/MoE layer.
- MSA: `PRIMUS_MSA_BACKEND`, `flydsl` (default) or `eager`.
- Experts: `USE_TURBO_GROUPED_GEMM` (default `True`; `False` is TE's grouped GEMM, one hipBLASLt GEMM
  per expert) and `TURBO_GROUPED_GEMM_BACKEND` (default `default`, which is Triton here; `bf16:flydsl`
  takes at most 64 local experts, so EP 2 or more).
- `TURBO_DIR` (default unset: the image's Primus-Turbo): a Primus-Turbo checkout built in place for
  the image, see above. A relative path is taken from the current directory.
- Dense layers: `USE_TURBO_ATTENTION` runs Primus-Turbo attention in the three dense layers, with
  `PRIMUS_TURBO_ATTN_BACKEND`, `flydsl` or `triton`. With `TURBO_DIR` set they default to `True` and
  `flydsl`; without it, to `False` and `triton`. `flydsl` needs
  `primus_turbo/flydsl/attention/gfx1250/`, and the launcher stops before training when the
  Primus-Turbo it would use lacks it. Primus-Turbo's own default backend fails with
  `invalid argument for fmha_bwd` here. MSA layers are unaffected.
- Optimizer: `PRIMUS_FUSED_ADAM_CLIP` (default `1`) runs Primus's HIP Adam, which applies the
  gradient-clip coefficient inside the kernel. `PRIMUS_FUSED_ADAM_BF16_WRITEBACK` and
  `PRIMUS_STATIC_BF16_LOSS_SCALE_CPU` (default: the value of `PRIMUS_FUSED_ADAM_CLIP`) also have it
  write the BF16 parameters and keep the loss scale on the host. Megatron then logs no grad norm. With
  `PRIMUS_FUSED_ADAM_CLIP=0` Adam is Primus's Triton AdamW (`use_triton_fused_adam`, on by default),
  and `--use_triton_fused_adam False` keeps TE's FusedAdam. `--use_precision_aware_optimizer True`
  needs `PRIMUS_FUSED_ADAM_CLIP=0`: the BF16 writeback needs separate FP32 master parameters, and the
  run stops at the first step without them.
- `HIPBLASLT_TUNING_OVERRIDE_FILE` (default unset: hipBLASLt's own kernel choice). This example has no
  tuned file yet. A file may be given relative to the current directory; the launcher mounts it into
  the container. See hipBLASLt tuning in the DeepSeek-V3 README for how such a file is made.
- `MOCK_DATA` (default `True`). `PROFILE=True` profiles iterations 6–7 on rank 0 (trace under
  `output/<team>/<user>/<exp>/tensorboard/`, about 41 MB for the baseline); add
  `--profile_ranks '[0,1,2,3]'` for every rank. `PATCH_TRITON_EXPERT_SCHEDULING` (default `1`): see
  Triton expert scheduling below.
- Config and output: `EXP` (the experiment yaml), and `PRIMUS_WORKSPACE` (default `output`), `PRIMUS_TEAM`,
  `PRIMUS_USER`, `PRIMUS_EXP_NAME`, which make up the output directory `<workspace>/<team>/<user>/<exp>`.

The defaults are in the script. Arguments to the script are appended to the Primus command line, for
example `--gradient_accumulation_fusion True`.

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

Results on `ctheliosr-rck-g02-e15-02` with this branch on `dev/mi455x_v3` c19456f1, one run at a time
on the four otherwise idle GPUs. Every run finished 10 iterations with 0 NaN.

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
defaults (HIP fused Adam, NaN check off) on the shared 4x MI455X node `ctheliosr-rck-g02-e15-01`, one
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
more memory per GPU. The node matters more than the base: on `ctheliosr-rck-g02-e15-02` the defaults
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

One GPU of `ctheliosr-rck-g02-e15-02` (4 layers, EP1, GBS 8, so 8 micro-batches per step as on the
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
- Not yet run on the EP4 baseline, where TE's dense attention is 106.5 ms of softmax plus about 78 ms
  of score and value GEMMs per profiled step.

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
  baseline that would be about 175 ms per step (five MSA layers, 8 micro-batches); not yet measured.
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
  not yet measured. The baseline's MSA forward is bound by kernel launches, so its share may not show.
- The indexer kernels are unchanged.

## How the defaults were chosen

On the older bases dad0a081 and a624bb95 (Megatron-LM d3528a21), one GPU unless noted, with TE's
FusedAdam and the NaN check on unless noted, on `ctheliosr-rck-g02-e15-02`. Every run finished with
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
  hipBLASLt for the fp32-output shapes was not tried.

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
- `--use_turbo_mega_moe True` (158.5 ms): applies only at EP 2 to 8, where it does not build (Known
  issues).
- `turbo_sync_free_moe_stage: 1`: sets `moe_use_fused_router_with_aux_score` and `moe_permute_fusion`,
  which the yaml already sets.

Do not run:

- `--turbo_fused_grouped_gemm True` and `--use_turbo_fused_act_with_probs True` assert
  `glu_linear_offset == 0`; M3's `swigluoai` uses 1.0. The second also rules out sync-free stage 3.
- `TURBO_GROUPED_GEMM_BACKEND=bf16:flydsl` on one GPU: `User specified backend FLYDSL cannot handle
  the given inputs`, because 128 local experts are over its 64. It runs at EP2.
- `--fp8 e4m3 --fp8_recipe blockwise` and `--fp8_recipe mxfp8` stop at TE's arch check (`Device arch
  gfx94x or gfx95x is required for FP8 block scaling execution.`, `Device arch gfx95x or newer is
  required for MXFP8 execution.`). TE 2.19.0.dev0 leaves gfx1250 as a TODO until its GEMM is available.
- `--use_turbo_deepep True`, sync-free stages 2 and 3, and MegaMoE: Known issues.
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
- DeepEP and MegaMoE, which do not run on gfx1250 yet (see Known issues).
- A real dataset and real routing. These runs are mock data with uniform routing.
- The full 60-layer model. From the proxy's ~18 bytes/parameter (BF16 weights, FP32 gradients and
  Adam state), a training step at MBS 1, seq 4096 wants on the order of 40 MI455X GPUs (EP8 x PP5).
- The gfx950 MFMA kernels. This node cannot execute them.
- hipBLASLt GEMM tuning of the kind in the GPT-OSS example, including the fp32-output weight-gradient
  GEMMs behind the `gradient_accumulation_fusion` slowdown.

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
- **DeepEP.** The DeepEP in this image is written for MI355X (wave64) and has not been ported to
  gfx1250's wave32. `--use_turbo_deepep True` on EP2 stops in the first step with `DeepEP error: CPU
  recv timeout`; with TE's grouped GEMM it hung there instead. Sync-free MoE stages 2 and 3
  (`turbo_sync_free_moe_stage`) turn DeepEP on, so they do not run either. Stage 1's two settings,
  `moe_use_fused_router_with_aux_score` and `moe_permute_fusion`, are already in the yaml.
- **MegaMoE.** `--use_turbo_mega_moe True` takes effect only at EP 2 to 8. At EP2 its FlyDSL
  `dispatch_prologue_kernel` does not build for gfx1250 (`lld invocation failed`, then `All autotune
  configs failed`) in the first forward.
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
