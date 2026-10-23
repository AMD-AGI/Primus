# Llama-3.1-8B on MI455X

Llama-3.1-8B pretraining on MI455X (gfx1250): the full model (32 layers, hidden 4096, 32 query / 8 KV
heads of dim 128), seq 8192, BF16, mock data, no activation recompute. Attention runs on Primus-Turbo's
FlyDSL kernels and RMSNorm on Primus-Turbo. The config is
`examples/megatron/configs/MI455X/llama3.1_8B-BF16-pretrain.yaml` and the launcher is
`run_llama3.1_8b_mi455x.sh`: with `PRIMUS_LAUNCHER=container` (default) it starts a new container of
`DOCKER_IMAGE` from the host and re-runs itself inside it; with `PRIMUS_LAUNCHER=direct` it runs inside
the current container.

The default image is `amdprimus/amdprimus:tas-20261019` (see the DeepSeek-V3 example's README). Its
Primus-Turbo has no gfx1250 FlyDSL attention for head dim 128 (only the MLA one), so this example needs a
Primus-Turbo checkout that has it (`primus_turbo/flydsl/attention/gfx1250/`, on `dev/mi455x_v2` since
Primus-Turbo PR #574), passed as `TURBO_DIR`. The numbers below were measured with Primus-Turbo
`dev/mi455x_v2` 85e85e70. With the backend left at `flydsl`, the launcher stops before training when the
Primus-Turbo it would use lacks that directory.

## Run

Build the Primus-Turbo checkout in place for the image once (gfx1250 only, OpenMPI off the paths so
rocSHMEM is not linked), then pass it as `TURBO_DIR`; the launcher puts it first on `PYTHONPATH`:

```bash
git clone https://github.com/AMD-AGI/Primus-Turbo.git /path/to/Primus-Turbo
git -C /path/to/Primus-Turbo checkout 85e85e70  # dev/mi455x_v2 with PR #574, the commit measured below
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v /path/to/Primus-Turbo:/path/to/Primus-Turbo \
    -w /path/to/Primus-Turbo amdprimus/amdprimus:tas-20261019 bash -c \
    'PATH=$(tr : "\n" <<< "$PATH" | grep -v openmpi | paste -sd:) GPU_ARCHS=gfx1250 python3 setup.py build_ext --inplace'

# On the host, 4 GPUs (the default: GPUS_PER_NODE=4, DP4, GBS 16)
TURBO_DIR=/path/to/Primus-Turbo bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh

# On the host, 1 GPU (GBS 4)
GPUS_PER_NODE=1 TURBO_DIR=/path/to/Primus-Turbo bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh

# Inside a container (mount the Primus-Turbo checkout too); GPUS_PER_NODE works the same way
PRIMUS_LAUNCHER=direct TURBO_DIR=/path/to/Primus-Turbo \
    bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh
```

**Settings** are environment variables:

- Parallelism: `GPUS_PER_NODE` (default 4, data parallel), `HIP_VISIBLE_DEVICES`, `PRIMUS_TP`, `PRIMUS_PP`.
- Training: `TRAIN_ITERS`, `MBS` (default 4), `GBS` (default `MBS` x `GPUS_PER_NODE`),
  `PRIMUS_RECOMPUTE_LAYER_IDS` (layers to recompute, e.g. `0,3`; empty, the default, disables
  recompute), `PRIMUS_SEQ_LENGTH` (default 8192) and `PRIMUS_MAX_POSITION_EMBEDDINGS` (default
  `PRIMUS_SEQ_LENGTH`; Megatron needs it to be at least the sequence length).
- Feature switches: `USE_TURBO_ATTENTION`, `USE_TURBO_RMS_NORM`, `MOCK_DATA`.
- `PRIMUS_TURBO_ATTN_BACKEND`: `flydsl` (default) or `triton`, see Attention backends below.
- `PROFILE=True` profiles iterations 6–7 (trace under `output/<team>/<user>/<exp>/tensorboard/`).
- `PATCH_TRITON_EXPERT_SCHEDULING` (default `1`): see Triton expert scheduling below.
- `HIPBLASLT_TUNING_OVERRIDE_FILE` (default: the file in `hipblaslt_tuning/`; empty: hipBLASLt's own
  kernel choice): see hipBLASLt tuning below.
- Config and output: `EXP` (the experiment yaml), and `PRIMUS_WORKSPACE` (default `output`), `PRIMUS_TEAM`,
  `PRIMUS_USER`, `PRIMUS_EXP_NAME`, which make up the output directory `<workspace>/<team>/<user>/<exp>`.

The defaults are in the script. Arguments to the script are appended to the Primus command line.

If the Primus-Turbo checkout pins a FlyDSL version other than the image's, install it with
`pip install --no-deps --target build/pydeps flydsl==<version>` in the checkout; `build/pydeps` is on the
`PYTHONPATH` as well.

**Output.** The console log is written to `output/<team>/<user>/<exp>/train.log`. From the host,
training runs as root in the new container, so the files it creates are owned by root. Inside a
container it runs as the container's user, which has to be root for the Triton patch.

**Triton expert scheduling.** Triton 3.8 turns on AMDGPU expert scheduling for gfx1250, which makes
Triton kernels (here the Inductor-compiled SwiGLU, and Primus-Turbo's Triton attention) fault or return
NaN at random. The launcher sets
`TRITON_HIP_USE_EXPERT_SCHEDULING=0`, which Triton applies whenever it compiles a kernel, and with
`PATCH_TRITON_EXPERT_SCHEDULING=1` (the default) also makes `is_expert_scheduling_enabled()` in the
container's Triton return `False`, which invalidates Inductor's cache of kernels compiled without the
variable. The patch needs root and stays in a long-lived container; set
`PATCH_TRITON_EXPERT_SCHEDULING=0` to skip it.

**hipBLASLt tuning.** The launcher sets `HIPBLASLT_TUNING_OVERRIDE_FILE` to
`hipblaslt_tuning/llama3.1_8b-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt` (empty: hipBLASLt's own
choice), which gives 11 of the 15 hipBLASLt GEMM shapes (4 x 8192 tokens per micro-batch) a faster kernel of
hipBLASLt's library, e.g. FC1 forward 1.32x and the output layer forward 1.36x. Other hipBLASLt builds and
shapes (another `MBS` x sequence length, or `PRIMUS_TP`) fall back to hipBLASLt's default. The file was made
like the DeepSeek-V3 example's; see hipBLASLt tuning in its README.

## Reference

Measured with `amdprimus/amdprimus:tas-20261019` and Primus-Turbo 85e85e70, MBS 4 (one micro-batch of
4 x 8192 tokens per GPU and iteration), started from the host. ms/iter is Primus's average over iterations
3–10, the mean of 3 runs that alternated between the rows; speedup is against the first row. Peak memory is
the largest `hip mem usage` in the log, which Primus prints for the last rank in GiB (labelled GB) as the
device's total minus free memory. Each row adds one step:

- Baseline: Primus-Turbo's Triton attention (`PRIMUS_TURBO_ATTN_BACKEND=triton`), hipBLASLt's own GEMM
  kernels (`HIPBLASLT_TUNING_OVERRIDE_FILE=`).
- FlyDSL: Primus-Turbo's FlyDSL attention.
- hipBLASLt tuning: the tuning file above. This row is the launcher's defaults.

**1 GPU** (`GPUS_PER_NODE=1`, GBS 4)

| Variant | ms/iter | Speedup | Tokens/s/GPU | TFLOP/s/GPU | Peak memory/GPU | lm loss at iteration 10 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1938 | 1.00x | 16,900 | 870 | 345.02 GiB | 10.1909 |
| + FlyDSL | 1116 | 1.74x | 29,400 | 1511 | 344.77 GiB | 10.1911–10.1919 |
| + hipBLASLt tuning (default) | 1024 | 1.89x | 32,000 | 1647 | 344.72 GiB | 10.1881–10.1894 |

**4 GPUs** (the launcher's defaults: DP4, GBS 16)

| Variant | ms/iter | Speedup | Tokens/s/GPU | TFLOP/s/GPU | Peak memory/GPU | lm loss at iteration 10 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1943 | 1.00x | 16,900 | 868 | 276.18 GiB | 8.9728 |
| + FlyDSL | 1138 | 1.71x | 28,800 | 1483 | 276.18 GiB | 8.9695–8.9717 |
| + hipBLASLt tuning (default) | 1047 | 1.86x | 31,300 | 1611 | 276.12 GiB | 8.9704–8.9784 |

TE's unfused attention (`USE_TURBO_ATTENTION=False`) is not in the tables: without recompute it runs out of
memory in the first iteration, and with `--recompute_granularity selective` it was about 4.5x slower than
FlyDSL (both without the tuning file; see Attention backends below). With the defaults, an iteration on
4 GPUs takes 2.2% longer than on 1 GPU (the data-parallel gradient reduce-scatter and parameter all-gather
are not fully hidden), and the distributed optimizer's sharding lowers the peak memory per GPU by about
69 GiB. The loss differs between 1 and 4 GPUs because the global batch does (4 vs 16 sequences). Without the
tuning file, `PRIMUS_SEQ_LENGTH=16384` with `MBS=2` (the same tokens per iteration) ran 1231 ms/iter on one
GPU.

From the host every run starts a new container and takes about 60 s from launch to exit on 1 or 4 GPUs
(the Triton patch invalidates Inductor's cache, and the FlyDSL kernels are compiled once); a repeat run on
one GPU in a long-lived container takes about 40 s.

**Profile** (1 GPU, `PROFILE=True`, iterations 6–7, without the tuning file): about 1042 ms of compute
kernels per iteration, and about 70 ms with no kernel running. hipBLASLt GEMMs 66%, attention 16% (forward
4%, backward 12%), SwiGLU 4%, optimizer 3.5%, RoPE, RMSNorm and loss 3%, copies and other elementwise
kernels 7%.

Attention alone at this shape (b 4, seq 8192, 32/8 heads, causal; mean of 100 calls after 20 warmups):

| Attention | Forward ms | Backward ms |
| --- | --- | --- |
| Primus-Turbo FlyDSL | 1.19 | 4.01 |
| Primus-Turbo Triton | 4.99 | 25.58 |
| TE unfused | 44.36 | 41.66 |

## Known issues

- **Attention backends.** The FlyDSL kernels (the default, `PRIMUS_TURBO_ATTN_BACKEND=flydsl`) take BF16
  attention with head dim 128, 1, 2, 4, 8 or 16 query heads per KV head, query lengths that are
  multiples of 64, KV lengths that are multiples of 32, no dropout, bias or sliding window, and q and k
  tensors of at most 1 GiB each (`MBS` <= 16 at seq 8192 with 32 heads per GPU); with the backend set to
  `flydsl`, any other call raises `ValueError`. The fused backward measured here needs causal attention,
  query lengths that are multiples of 64 and KV lengths that are multiples of 128; other calls use the
  deterministic two-kernel backward. With a
  Primus-Turbo that lacks `primus_turbo/flydsl/attention/gfx1250/` (e.g. the image's own), every call
  raises that `ValueError`. Primus-Turbo's Triton attention (`PRIMUS_TURBO_ATTN_BACKEND=triton`) needs
  Triton's expert scheduling off (see Triton expert scheduling above): with it on, the Triton attention
  forward made illegal memory accesses or gave NaN at this shape. `USE_TURBO_ATTENTION=False` runs TE's
  `DotProductAttention`, not PyTorch SDPA. On gfx1250 TE has neither of its fused attention backends for
  this input, and the image has no `flash_attn` package, so TE uses `UnfusedDotProductAttention`. That
  keeps the 8192 x 8192 attention probabilities of every layer for the backward pass, and at `MBS` 4 the
  first iteration runs out of memory; `--recompute_granularity selective` (recompute the core attention
  in the backward pass) makes it fit.
- **Determinism.** The fused FlyDSL backward adds dQ with atomics, so dQ (and from there the loss) can
  differ between runs in the last bits (lm loss 10.1913 and 10.1895 at iteration 10); dK and dV are
  reproducible. Primus passes Megatron's `deterministic_mode` to Primus-Turbo, which then runs the
  deterministic two-kernel backward instead. Megatron's deterministic mode is not set up by this example
  and was not measured: it needs `--deterministic_mode True --cross_entropy_loss_fusion False`, `NCCL_ALGO`
  set to one of Megatron's all-reduce algorithms (e.g. `Ring`), and, checked by Primus on ROCm,
  `PRIMUS_DETERMINISTIC=1`, `TORCH_COMPILE_DISABLE=1` (which also turns off the compiled SwiGLU),
  `ROCBLAS_DEFAULT_ATOMICS_MODE=0` and `PRIMUS_TURBO_AUTO_TUNE=0`.
