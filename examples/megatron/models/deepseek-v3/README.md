# DeepSeek-V3 proxy on MI455X

DeepSeek-V3 MoE proxy pretraining on one node of 4x MI455X (gfx1250): full DeepSeek-V3 layer shapes
(MLA, hidden 7168), 4 MoE layers of 128 experts, EP4, seq 4096, BF16, mock data, no activation
recompute. Attention and the expert grouped GEMMs run on Primus-Turbo's FlyDSL kernels. The config is
`examples/megatron/configs/MI455X/deepseek_v3-BF16-pretrain.yaml` and the launcher is
`run_deepseek_v3_proxy_mi455x.sh`: with `PRIMUS_LAUNCHER=container` (default) it starts a new container
of `DOCKER_IMAGE` from the host and re-runs itself inside it; with `PRIMUS_LAUNCHER=direct` it runs inside
the current container.

The default image, `amdprimus/amdprimus:tas-20261019`, is `amdprimus/amdprimus:tas-20261015` (ROCm
10.2.0a20261011, PyTorch 2.14.0, FlyDSL 0.3.4.1, aiter 0.1.24.post1) with hipBLASLt built from
ROCm/rocm-libraries `develop` e1e7fdcab and Primus-Turbo `main` 21588d59 (#567: FlyDSL MLA attention and
the tuned gfx1250 BF16 grouped GEMM).

## Run

```bash
# On the host
bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh

# Inside a container
docker run -d --name dsv3 --privileged --network host --ipc host --device /dev/kfd --device /dev/dri \
    --group-add video -v "$PWD:$PWD" -w "$PWD" amdprimus/amdprimus:tas-20261019 sleep infinity
docker exec -it dsv3 bash
PRIMUS_LAUNCHER=direct bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh
```

**Settings** are environment variables:

- Parallelism and size: `GPUS_PER_NODE`, `HIP_VISIBLE_DEVICES`, `PRIMUS_TP`, `PRIMUS_PP`, `PRIMUS_EP`,
  `PRIMUS_TOTAL_LAYERS`, `PRIMUS_NUM_EXPERTS`, `PRIMUS_MOE_TOPK`.
- Training: `TRAIN_ITERS`, `MBS`, `GBS`, `PRIMUS_RECOMPUTE_LAYER_IDS` (layers to recompute, e.g. `0,3`;
  empty, the default, disables recompute).
- Feature switches: `USE_TURBO_ATTENTION`, `USE_TURBO_GROUPED_GEMM`, `USE_TURBO_DEEPEP`, `MOCK_DATA`.
- Primus-Turbo backends: `PRIMUS_TURBO_ATTN_BACKEND` (`flydsl` or `triton`) and
  `TURBO_GROUPED_GEMM_BACKEND` (`bf16:flydsl`, or `default` for Primus-Turbo's own choice: Triton, or
  auto-tuning if `PRIMUS_TURBO_AUTO_TUNE=1` is exported). Primus sets `PRIMUS_TURBO_GROUPED_GEMM_BACKEND`
  from `TURBO_GROUPED_GEMM_BACKEND` unless that is `default`, so the script refuses an exported
  `PRIMUS_TURBO_GROUPED_GEMM_BACKEND`.
- `PROFILE=True` profiles iterations 6–7 (trace under `output/<team>/<user>/<exp>/tensorboard/`).
- `PATCH_TRITON_EXPERT_SCHEDULING` (default `1`): see Triton expert scheduling below.
- `HIPBLASLT_TUNING_OVERRIDE_FILE` (default: the file in `hipblaslt_tuning/`; empty: hipBLASLt's own
  kernel choice): see hipBLASLt tuning below.
- Config and output: `EXP` (the experiment yaml), and `PRIMUS_WORKSPACE` (default `output`), `PRIMUS_TEAM`,
  `PRIMUS_USER`, `PRIMUS_EXP_NAME`, which make up the output directory `<workspace>/<team>/<user>/<exp>`.

The defaults are in the script. Arguments to the script are appended to the Primus command line. The
1-GPU variant has the per-GPU shape of the 4-GPU run (4 layers, 32 experts, 8 micro-batches of 2):

```bash
GPUS_PER_NODE=1 PRIMUS_EP=1 PRIMUS_NUM_EXPERTS=32 GBS=16 \
    bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh
```

**Another Primus-Turbo checkout.** Build it in place for the image once (gfx1250 only, OpenMPI off the
paths so rocSHMEM is not linked), then pass it as `TURBO_DIR`; the launcher puts it first on `PYTHONPATH`:

```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v /path/to/Primus-Turbo:/path/to/Primus-Turbo \
    -w /path/to/Primus-Turbo amdprimus/amdprimus:tas-20261019 bash -c \
    'PATH=$(tr : "\n" <<< "$PATH" | grep -v openmpi | paste -sd:) GPU_ARCHS=gfx1250 python3 setup.py build_ext --inplace'
TURBO_DIR=/path/to/Primus-Turbo bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh
```

If that checkout pins a FlyDSL version other than the image's, install it with
`pip install --no-deps --target build/pydeps flydsl==<version>` in the checkout; `build/pydeps` is on the
`PYTHONPATH` as well.

**Output.** The console log is written to `output/<team>/<user>/<exp>/train.log`. From the host,
training runs as root in the new container, so the files it creates are owned by root. Inside a
container it runs as the container's user, which has to be root for the Triton patch.

**Triton expert scheduling.** Triton 3.8 turns on AMDGPU expert scheduling for gfx1250, which makes
Triton kernels fault or return NaN at random. The launcher sets `TRITON_HIP_USE_EXPERT_SCHEDULING=0`,
which Triton applies whenever it compiles a kernel. Inductor, however, also loads kernels from its own
cache without compiling them, so a container that ran without the variable can keep expert-mode kernels.
With `PATCH_TRITON_EXPERT_SCHEDULING=1` (the default) the launcher therefore also makes
`is_expert_scheduling_enabled()` in the container's Triton return `False`, like the GPT-OSS example; that
also invalidates Inductor's cache. The patch needs root and stays in a long-lived container; set
`PATCH_TRITON_EXPERT_SCHEDULING=0` to skip it.

**hipBLASLt tuning.** The launcher sets `HIPBLASLT_TUNING_OVERRIDE_FILE` to
`hipblaslt_tuning/deepseek_v3_proxy-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt` (empty:
hipBLASLt's own choice), which gives 17 of the 27 hipBLASLt GEMM shapes of the 4-GPU run a faster kernel of
hipBLASLt's library. hipBLASLt uses a row only while its solution index still names the kernel it was tuned
with (the file name gives the hipBLASLt commit and ROCm version), so other builds and shapes fall back to its
default. To remake the file with `hipblaslt-bench` (not in the image; build it from the same rocm-libraries
commit): log the example's GEMMs with `HIPBLASLT_LOG_MASK=32`, search each with
`HIPBLASLT_TUNING_FILE=<file> HIPBLASLT_TUNING_USER_MAX_WORKSPACE=33554432` and `--algo_method heuristic
--requested_solution -1 --rotating 512 --flush`, and keep the rows that are at least 3% faster when replayed
through `HIPBLASLT_TUNING_OVERRIDE_FILE`.

## Reference

Measured with `amdprimus/amdprimus:tas-20261019` and the image's Primus-Turbo, started from the host.
ms/iter is Primus's average over iterations 3–10. The FlyDSL and hipBLASLt tuning rows are the mean of 2 runs
that alternated between the two rows; the baseline rows are the mean of 3 runs from an earlier set, and the
speedup is against them. Peak memory is the largest `hip mem usage` in the log, which Primus prints for the
last rank in GiB (labelled GB) as the device's total minus free memory. Each row adds one step:

- Baseline: TE's unfused attention and Primus-Turbo's Triton grouped GEMM
  (`USE_TURBO_ATTENTION=False TURBO_GROUPED_GEMM_BACKEND=default`), hipBLASLt's own GEMM kernels
  (`HIPBLASLT_TUNING_OVERRIDE_FILE=`).
- FlyDSL: Primus-Turbo's FlyDSL attention and grouped GEMM.
- hipBLASLt tuning: the tuning file above. This row is the launcher's defaults.

**1 GPU** (the 1-GPU variant above: 4 layers, 32 experts, GBS 16)

| Variant | ms/iter | Speedup | Tokens/s/GPU | TFLOP/s/GPU | Peak memory/GPU | lm loss at iteration 10 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1967 | 1.00x | 33,300 | 719 | 174.15 GiB | 5.49455 |
| + FlyDSL | 1010 | 1.95x | 64,900 | 1400 | 130.52 GiB | 5.49483 |
| + hipBLASLt tuning (default) | 960 | 2.05x | 68,200 | 1473 | 130.53 GiB | 5.49473 |

**4 GPUs** (the launcher's defaults: EP4, 128 experts, GBS 64)

| Variant | ms/iter | Speedup | Tokens/s/GPU | TFLOP/s/GPU | Peak memory/GPU | lm loss at iteration 10 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 2139 | 1.00x | 30,600 | 661 | 215.21 GiB | 5.56290–5.56295 |
| + FlyDSL | 1200 | 1.78x | 54,600 | 1179 | 170.71 GiB | 5.56274 |
| + hipBLASLt tuning (default) | 1132 | 1.89x | 57,900 | 1249 | 170.71 GiB | 5.56307 |

**Grad norms per iteration** of the default. Its two runs gave the same values on 1 and on 4 GPUs; an earlier
set of 4-GPU runs differed from them by up to 0.004.

| GPUs | Grad norms, iterations 1–10 |
| --- | --- |
| 1 | 5.292, 5.326, 5.587, 7.070, 12.976, 14.047, 13.659, 15.001, 13.475, 14.192 |
| 4 | 3.810, 3.778, 3.815, 5.138, 8.845, 10.095, 7.851, 7.169, 6.821, 6.650 |

**Profile of the 4-GPU run, rank 0** (`PROFILE=True`, iterations 6–7).

- About 899 ms of compute kernels per iteration: MoE 43% (expert grouped GEMMs 29%), output layer and
  loss 16%, attention projections 15%, core attention 15%, the rest 11% (4% for adding gradients into
  `main_grad`, as `gradient_accumulation_fusion` is off). The hipBLASLt GEMMs take 320 ms (36%); with
  hipBLASLt's own kernel choice they take 375 ms, 961 ms of compute in all.
- The MoE all-to-all kernels take about 144 ms per iteration, and the 4-GPU run takes 172 ms longer than the
  1-GPU variant with the same per-GPU work (1132 vs 960 ms/iter), so compute hides little of the all-to-all.

## Known issues

- **Attention backends.** MLA (q/k 192, v 128) runs on Primus-Turbo FlyDSL (the default,
  `PRIMUS_TURBO_ATTN_BACKEND=flydsl`), Primus-Turbo Triton (`PRIMUS_TURBO_ATTN_BACKEND=triton`) or TE
  `UnfusedDotProductAttention` (`USE_TURBO_ATTENTION=False`). The FlyDSL kernels take BF16 causal MHA
  with sequence lengths that are multiples of 64, fixed-length batches (no varlen) and a q tensor of at
  most 1 GiB (`MBS` <= 5 at seq 4096 with 128 heads per GPU); with the backend set to `flydsl`, any other
  call raises `ValueError`. A Primus-Turbo without the FlyDSL MLA kernels
  (`primus_turbo/flydsl/attention/gfx1250_mla/`, on `main` since 21588d59) raises that `ValueError` at the
  first attention call; with such an image (e.g. `tas-20261011`) or `TURBO_DIR` checkout, set
  `PRIMUS_TURBO_ATTN_BACKEND=triton` or `USE_TURBO_ATTENTION=False`. Context parallelism (CP > 1) goes
  through Primus-Turbo's `flash_attn_usp_func`, which ignores `PRIMUS_TURBO_ATTN_BACKEND` and always uses
  aiter/CK, which has no gfx1250 kernels for these head dims, so CP > 1 does not run with Primus-Turbo
  attention on gfx1250.
- **FlyDSL grouped GEMM.** It needs `gradient_accumulation_fusion: false` and at most 64 local experts;
  with the backend set to `bf16:flydsl`, other calls raise `ValueError`. Primus-Turbo documents a rare,
  nondeterministic wrong result from these kernels, with no error or NaN: about 1 call in 130,000 in
  steady state on the largest shape, and much more often over a process's first few hundred calls.
  `TURBO_GROUPED_GEMM_BACKEND=default` uses the Triton grouped GEMM.
- **Expandable segments.** Primus warns that `use_turbo_grouped_gemm=True` runs without
  `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`. Here that setting (in the yaml's `env:`) lowers the
  peak VRAM by only 1–3 GiB per GPU, makes the 4-GPU run 0.6% slower and the 4-GPU baseline 4% slower, and
  leaves the loss and grad norms unchanged, so the example does not set it.
