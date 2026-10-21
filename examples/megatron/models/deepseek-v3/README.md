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

## Reference

Measured on one node of 4x MI455X with `amdprimus/amdprimus:tas-20261019` and the image's Primus-Turbo,
the 4-GPU rows from the host and the 1-GPU row (the variant above) on one GPU inside a container. ms/iter
is Primus's average over iterations 3–10, the mean of 2–3 runs. Peak VRAM is the largest
`mem_info_vram_used` sampled every second.

| Variant | ms/iter | Tokens/s/GPU | TFLOP/s/GPU | Peak VRAM/GPU | lm loss at iteration 10 |
| --- | --- | --- | --- | --- | --- |
| 4-GPU | 1195 | 54,800 | 1184 | 152.2–154.5 GiB | 5.56274 |
| 1-GPU, 4 layers | 1005 | 65,200 | 1407 | 118.7 GiB | 5.49483 |
| 4-GPU, TE unfused attention and Triton grouped GEMM | 2153 | 30,400 | 657 | 196.8–198.9 GiB | 5.56295 |

The last row (`USE_TURBO_ATTENTION=False TURBO_GROUPED_GEMM_BACKEND=default`: TE's unfused attention and
Primus-Turbo's Triton grouped GEMM) is 1.80x slower than the FlyDSL defaults.

**Grad norms per iteration.** Repeated runs of a variant gave the same values.

| Variant | Grad norms, iterations 1–10 |
| --- | --- |
| 4-GPU | 3.810, 3.777, 3.815, 5.138, 8.841, 10.091, 7.848, 7.168, 6.820, 6.649 |
| 1-GPU, 4 layers | 5.292, 5.326, 5.587, 7.070, 12.976, 14.045, 13.664, 15.002, 13.477, 14.193 |

**Profile of the 4-GPU run, rank 0** (`PROFILE=True`, iterations 6–7).

- About 1018 ms of compute kernels per iteration: MoE 41% (expert grouped GEMMs 27%), output layer
  and loss 19%, attention projections 16%, core attention 14%, the rest 10%.
- About 144 ms per iteration of MoE all-to-all is not overlapped with compute.

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
  peak VRAM by only 1–3 GiB per GPU, makes the 4-GPU run 0.6% slower and the TE/Triton row 4% slower, and
  leaves the loss and grad norms unchanged, so the example does not set it.
