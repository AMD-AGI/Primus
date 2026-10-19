# DeepSeek-V3 proxy on MI455X

DeepSeek-V3 MoE proxy pretraining on one node of 4x MI455X (gfx1250): full DeepSeek-V3 layer shapes
(MLA, hidden 7168), 4 MoE layers of 128 experts, EP4, seq 4096, BF16, mock data, no activation
recompute. Attention and the expert grouped GEMMs run on Primus-Turbo's FlyDSL kernels. The config is
`examples/megatron/configs/MI455X/deepseek_v3-BF16-pretrain.yaml` and the launcher is
`run_deepseek_v3_proxy_mi455x.sh`: with `PRIMUS_LAUNCHER=container` (default) it runs from the host in a
new container of `DOCKER_IMAGE`; with `PRIMUS_LAUNCHER=direct` it runs inside the current container.

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

The defaults are in the script. Arguments to the script are appended to the Primus command line. Two
1-GPU variants: the first has the per-GPU shape of the 4-GPU run (4 layers, 32 experts, 8 micro-batches
of 2); the second is a quick 1-layer check whose time is dominated by the output layer.

```bash
GPUS_PER_NODE=1 PRIMUS_EP=1 PRIMUS_NUM_EXPERTS=32 GBS=16 \
    bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh
GPUS_PER_NODE=1 PRIMUS_EP=1 PRIMUS_TOTAL_LAYERS=1 PRIMUS_NUM_EXPERTS=32 GBS=16 \
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

**Output.** The console log is written to `output/<team>/<user>/<exp>/train.log`. Training runs as
root in both modes (the Triton patch needs it), so the files it creates are owned by root. After a run
inside a container has created `output/`, a run from the host cannot create its log directory there until
the owner is changed back (e.g. `sudo chown -R "$(id -u):$(id -g)" output`).

**Triton patch.** Every run applies `runner/helpers/patches/12_fix_triton_expert_scheduling.sh`. It
turns off Triton's gfx1250 expert scheduling, which otherwise makes Triton kernels fault or return NaN
at random.

## Reference

Measured on one node of 4x MI455X with `amdprimus/amdprimus:tas-20261019` and the image's Primus-Turbo,
the 4-GPU rows from the host and the 1-GPU rows (the two variants above) on one GPU inside a container.
ms/iter is Primus's average over iterations 3–10, the mean of 2–3 runs; runs that stopped on NaN/Inf
(see Known issues) count with their iterations before the stop. Peak VRAM is the largest
`mem_info_vram_used` sampled every second.

| Variant | ms/iter | Tokens/s/GPU | TFLOP/s/GPU | Peak VRAM/GPU | lm loss at iteration 10 |
| --- | --- | --- | --- | --- | --- |
| 4-GPU | 1227 | 53,400 | 1157 | 152.5–154.7 GiB | 5.57972 |
| 1-GPU, 4 layers | 1014 | 64,600 | 1400 | 119.0 GiB | 5.4998–5.4999 |
| 1-GPU, 1 layer | 443 | 147,900 | 1427 | 62.7 GiB | 9.45946 |
| 4-GPU, TE unfused attention and Triton grouped GEMM | 2179 | 30,100 | 652 | 196.9–199.1 GiB | 5.57987 |

The last row, the previous defaults (`USE_TURBO_ATTENTION=False TURBO_GROUPED_GEMM_BACKEND=default`), is
1.78x slower than the FlyDSL defaults.

**Grad norms per iteration.** Complete runs agree to within 0.003.

| Variant | Grad norms, iterations 1–10 |
| --- | --- |
| 4-GPU | 3.790, 3.758, 3.795, 4.995, 9.031, 10.179, 7.921, 7.203, 6.828, 6.681 |
| 1-GPU, 4 layers | 5.256, 5.309, 5.541, 7.045, 13.028, 14.245, 13.707, 15.155, 13.394, 13.587 |
| 1-GPU, 1 layer | 3.193, 3.232, 3.314, 3.263, 3.210, 3.713, 3.362, 3.779, 3.329, 3.289 |

**Profile of the 4-GPU run, rank 0** (`PROFILE=True`, iterations 6–7).

- About 1026 ms of compute kernels per iteration: MoE 41% (expert grouped GEMMs 27%), output layer
  and loss 19%, attention projections 16%, core attention 14%, the rest 10%.
- About 158 ms per iteration of MoE all-to-all is not overlapped with compute.

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
- **Wrong gradients and NaN/Inf stops.** On the node used for the numbers above, some runs leave the
  reference grad norms at a random iteration (a wrong value, NaN or Inf), with every attention and grouped
  GEMM backend, including the FlyDSL default; the same node also returns wrong results from plain
  hipBLASLt GEMMs. Megatron aborts only on NaN/Inf, so compare grad norms against the tables above.
- **Vocabulary on Primus `main`.** The numbers above use the mock-data vocabulary of 131072 tokens. Primus
  `main` (#1249) sets DeepSeek-V3's own vocabulary, 129280, which changes the output layer slightly and
  the lm loss and grad norms.
