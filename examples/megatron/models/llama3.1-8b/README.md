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
Primus-Turbo checkout that has it (`primus_turbo/flydsl/attention/gfx1250/`, on the Primus-Turbo branch
`dev/lhz/llama31_8b_opt`), passed as `TURBO_DIR`.

## Run

Build the Primus-Turbo checkout in place for the image once (gfx1250 only, OpenMPI off the paths so
rocSHMEM is not linked), then pass it as `TURBO_DIR`; the launcher puts it first on `PYTHONPATH`:

```bash
git clone -b dev/lhz/llama31_8b_opt https://github.com/AMD-AGI/Primus-Turbo.git /path/to/Primus-Turbo
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v /path/to/Primus-Turbo:/path/to/Primus-Turbo \
    -w /path/to/Primus-Turbo amdprimus/amdprimus:tas-20261019 bash -c \
    'PATH=$(tr : "\n" <<< "$PATH" | grep -v openmpi | paste -sd:) GPU_ARCHS=gfx1250 python3 setup.py build_ext --inplace'

# On the host
TURBO_DIR=/path/to/Primus-Turbo bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh

# Inside a container (mount the Primus-Turbo checkout too)
PRIMUS_LAUNCHER=direct TURBO_DIR=/path/to/Primus-Turbo \
    bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh

# One GPU
GPUS_PER_NODE=1 TURBO_DIR=/path/to/Primus-Turbo bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh
```

**Settings** are environment variables:

- Parallelism: `GPUS_PER_NODE` (default 4, data parallel), `HIP_VISIBLE_DEVICES`, `PRIMUS_TP`, `PRIMUS_PP`.
- Training: `TRAIN_ITERS`, `MBS` (default 4), `GBS` (default `MBS` x `GPUS_PER_NODE`),
  `PRIMUS_RECOMPUTE_LAYER_IDS` (layers to recompute, e.g. `0,3`; empty, the default, disables
  recompute), `PRIMUS_SEQ_LENGTH` (default 8192).
- Feature switches: `USE_TURBO_ATTENTION`, `USE_TURBO_RMS_NORM`, `MOCK_DATA`.
- `PRIMUS_TURBO_ATTN_BACKEND`: `flydsl` (default), `hipkittens` or `triton`, see Attention backends below.
  With `hipkittens`, `PRIMUS_TURBO_HK_FWD` (`v0`, the default, `pp` or `flydsl`) and `PRIMUS_TURBO_HK_BWD`
  (`v0`, the default, or `flydsl`) pick the HipKittens kernels per direction, and `HK_HSACO_DIR` loads
  unpinned code objects from a directory (or several joined by `:`) instead of the checkout's pinned
  `bin/` (development only). `PRIMUS_TURBO_LOG_LEVEL=INFO` logs which kernels each direction runs.
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

**hipBLASLt tuning.** The example runs 15 GEMM shapes on hipBLASLt: forward, dgrad and wgrad of the
QKV, output projection, FC1, FC2 and output-layer linears, for 4 x 8192 tokens per micro-batch. For 11 of them
hipBLASLt's default kernel was slower than another kernel in its library, e.g. FC1 forward/dgrad 1.32x/1.28x,
FC2 wgrad 1.22x, QKV forward 1.22x and the output layer forward 1.36x. The launcher sets
`HIPBLASLT_TUNING_OVERRIDE_FILE` to
`hipblaslt_tuning/llama3.1_8b-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt`, which names those
kernels; hipBLASLt then picks them instead of its default. A row is used only while its solution index
still names the kernel it was tuned with, so with another hipBLASLt build, or other shapes (another `MBS`,
sequence length or `PRIMUS_TP`), hipBLASLt falls back to its own choice for that row. The file name gives the
hipBLASLt commit and the ROCm version it was tuned on (the default image's). It was made with
`hipblaslt-bench`, which the image does not ship (build it from the same rocm-libraries commit):

1. Run the example with `HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=<dir>/hipblaslt_%i.log`; every
   hipBLASLt call is logged as a `hipblaslt-bench` command line.
2. Run each distinct command with `HIPBLASLT_TUNING_FILE=<file>`,
   `HIPBLASLT_TUNING_USER_MAX_WORKSPACE=33554432` (Transformer Engine's hipBLASLt workspace on gfx1250),
   `--algo_method heuristic --requested_solution -1 --rotating 512 --flush`; hipblaslt-bench times every
   kernel of the library for the problem and appends the fastest to the file.
3. Time each problem with `--requested_solution 1` with and without
   `HIPBLASLT_TUNING_OVERRIDE_FILE=<file>`, alternately, and keep the rows that are at least 3% faster.
   Two of the 13 rows the search proposed failed this check (QKV dgrad 2.6% faster, QKV wgrad 1.6% slower).

Timed alone with `hipblaslt-bench`, the 15 GEMMs take 701 ms per iteration with hipBLASLt's choice and
605 ms with the file.

## Reference

Measured on one MI455X inside a container (`GPUS_PER_NODE=1`, so `GBS=4`: one micro-batch of 4 x 8192
tokens per iteration) with `amdprimus/amdprimus:tas-20261019` and Primus-Turbo
`dev/lhz/llama31_8b_opt` 62a3fcca. ms/iter is Primus's average over iterations 3–10 (the two FlyDSL rows:
the mean of 3 and 4 runs, alternated). Peak memory is the largest `hip mem usage` in the log. The 4-GPU
default was not measured.

| Variant | ms/iter | Tokens/s/GPU | TFLOP/s/GPU | Peak memory/GPU | lm loss at iteration 10 |
| --- | --- | --- | --- | --- | --- |
| Primus-Turbo FlyDSL attention (default) | 985 | 33,300 | 1713 | 344.69 GB | 10.1890–10.1907 |
| FlyDSL, `HIPBLASLT_TUNING_OVERRIDE_FILE=` (hipBLASLt's default kernels) | 1079 | 30,400 | 1563 | 344.74 GB | 10.1877–10.1901 |
| Primus-Turbo Triton attention (`PRIMUS_TURBO_ATTN_BACKEND=triton`) | 1846 | 17,800 | 914 | 344.94 GB | 10.1908 |
| TE unfused attention (`USE_TURBO_ATTENTION=False`), `--recompute_granularity selective` | 4922 | 6,700 | 343 | 354.41 GB | 10.1887 |

The hipBLASLt tuning file makes the default 8.7% faster. FlyDSL attention is 1.87x faster than Triton and
5.0x faster than TE. Without recompute, the TE row runs out of memory in the first iteration (see Attention
backends below).

The first run in a new container takes about 110 s from launch to exit (the Triton patch invalidates
Inductor's cache, and the FlyDSL kernels are compiled once); a repeat run takes about 46 s.

**Profile** (`PROFILE=True`, iterations 6–7): about 944 ms of compute kernels per iteration, and about
68 ms with no kernel running. hipBLASLt GEMMs 62% (586 ms; 689 ms with hipBLASLt's default kernels),
attention 18% (forward 4%, backward 13%), SwiGLU 5%, optimizer 4%, RoPE, RMSNorm and loss 3.5%, copies and
other elementwise kernels 8%.

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
  `flydsl`, any other call raises `ValueError`. The fused backward measured here needs causal attention
  and lengths that are multiples of 128; other shapes use the deterministic two-kernel backward. With a
  Primus-Turbo that lacks `primus_turbo/flydsl/attention/gfx1250/` (e.g. the image's own), every call
  raises that `ValueError`. `PRIMUS_TURBO_ATTN_BACKEND=hipkittens` runs the HipKittens kernels of the
  Primus-Turbo checkout (`primus_turbo/hipkittens/gfx1250_attn`, branch `dev/lhz/llama31_8b_hk`) where
  they apply (causal, 4 query heads per KV head for the backward, lengths that are multiples of 64 / 128,
  code objects built by `setup.py` and pinned by its `manifest.json`); every other call, a
  `deterministic_mode` backward, tensors that do not start on a 16-byte boundary and any direction whose
  code objects are missing run on the FlyDSL kernels, with one warning per reason, so it takes exactly
  the head-dim-128 calls `flydsl` takes. Other head dims (e.g. DeepSeek-V3's MLA) raise the `ValueError`
  above with `hipkittens` selected: keep `flydsl` for those models. Primus-Turbo's
  Triton attention (`PRIMUS_TURBO_ATTN_BACKEND=triton`) needs Triton's expert scheduling off (see Triton
  expert scheduling above): with it on, the Triton attention forward made illegal memory accesses or gave
  NaN at this shape. `USE_TURBO_ATTENTION=False` runs TE's
  `DotProductAttention`, not PyTorch SDPA. On gfx1250 TE has neither of its fused attention backends for
  this input, and the image has no `flash_attn` package, so TE uses `UnfusedDotProductAttention`. That
  keeps the 8192 x 8192 attention probabilities of every layer for the backward pass, and at `MBS` 4 the
  first iteration runs out of memory; `--recompute_granularity selective` (recompute the core attention
  in the backward pass) makes it fit.
- **Determinism.** The fused FlyDSL backward adds dQ with atomics, so dQ (and from there the loss) can
  differ between runs in the last bits (lm loss 10.1913 and 10.1895 at iteration 10); dK and dV are
  reproducible. Primus passes Megatron's `deterministic_mode` to Primus-Turbo, which then runs the
  deterministic two-kernel backward instead (slower; not measured here).
