# DeepSeek-V3 proxy on MI455X

DeepSeek-V3 MoE proxy pretraining on one node of 4x MI455X (gfx1250): full DeepSeek-V3 layer shapes
(MLA, hidden 7168), 4 MoE layers of 128 experts, EP4, seq 4096, BF16, mock data, no activation
recompute. Attention and the expert grouped GEMMs run on Primus-Turbo's FlyDSL kernels. The config is
`examples/megatron/configs/MI455X/deepseek_v3-BF16-pretrain.yaml`.

| File | Purpose |
| --- | --- |
| `run_deepseek_v3_proxy_mi455x.sh` | Launcher. With `PRIMUS_LAUNCHER=container` (default) it runs from the host in a new container of `DOCKER_IMAGE`; with `PRIMUS_LAUNCHER=direct` it runs inside the current container. |
| `prepare_mi455x.sh` | Optional host-side setup. `TURBO_DIR` builds a Primus-Turbo checkout for the image, and `CONTAINER` starts a container for direct runs. |

The default image, `amdprimus/amdprimus:tas-20261015`, is `amdprimus/amdprimus:tas-20261011` with
Primus-Turbo `dev/mi455x_v1` (e1a967a0: FlyDSL MLA attention and the tuned gfx1250 BF16 grouped GEMM),
flydsl 0.3.4.1 and aiter v0.1.24.post1.

## Run

```bash
# On the host, with the image's Primus-Turbo
bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh

# With a Primus-Turbo checkout instead (built once)
TURBO_DIR=/path/to/Primus-Turbo bash examples/megatron/models/deepseek-v3/prepare_mi455x.sh
TURBO_DIR=/path/to/Primus-Turbo bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh

# Inside a container
CONTAINER=dsv3 bash examples/megatron/models/deepseek-v3/prepare_mi455x.sh
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
  auto-tuning if `PRIMUS_TURBO_AUTO_TUNE=1` is exported). The script refuses Primus-Turbo's
  `PRIMUS_TURBO_GROUPED_GEMM_BACKEND`, which Primus would overwrite from `TURBO_GROUPED_GEMM_BACKEND`.
- `PROFILE=True` profiles iterations 6–7 (trace under `output/<team>/<user>/<exp>/tensorboard/`).
- Primus opt-ins, all on by default; set to `0` for the previous behaviour:
  - `PRIMUS_MOE_A2A_SKIP_SELF`: the MoE all-to-all copies the rank's own chunk locally and gives RCCL a list
    all-to-all without it, so the 3 peers fit one RCCL kernel instead of two (bit-identical).
  - `PRIMUS_NO_HOST_SYNC`: with TP=1, `get_batch` and TE's cross-entropy backward skip their blocking
    host-to-device copies (bit-identical).
  - `PRIMUS_TURBO_CROSS_ENTROPY`: the LM loss uses Primus-Turbo's cross entropy (TP=1; FP32 gradient
    arithmetic as current TE).
  - `PRIMUS_ROUTER_TRITON_GEMM`: Triton GEMMs for the FP32 router; the backward keeps FP32 accuracy.
- `EXTRA_ENV`: extra environment for the training processes, e.g. `EXTRA_ENV="GPU_MAX_HW_QUEUES=4"`.

The defaults are in the script. Arguments to the script are appended to the Primus command line. The
1-GPU variant:

```bash
GPUS_PER_NODE=1 PRIMUS_EP=1 PRIMUS_TOTAL_LAYERS=1 PRIMUS_NUM_EXPERTS=32 GBS=16 \
    bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh
```

**Output.** The console log is written to `output/<team>/<user>/<exp>/train.log`. Training runs as
root in both modes (the Triton patch needs it), so the files it creates are owned by root.

**Triton patch.** Every run applies `runner/helpers/patches/12_fix_triton_expert_scheduling.sh`. It
turns off Triton's gfx1250 expert scheduling, which otherwise makes Triton kernels fault or return NaN
at random.

## Reference

Measured on 4x MI455X with `amdprimus/amdprimus:tas-20261011` and the previous defaults, TE
`UnfusedDotProductAttention` and the Triton grouped GEMM, running inside a container. Primus-Turbo
a1800137 (then the head of `dev/mi455x_v1`) and the image's Primus-Turbo gave identical results. To rerun
the no-recompute row (add `PRIMUS_RECOMPUTE_LAYER_IDS=0,3` or the 1-GPU settings above for the others):

```bash
DOCKER_IMAGE=amdprimus/amdprimus:tas-20261011 USE_TURBO_ATTENTION=False TURBO_GROUPED_GEMM_BACKEND=default \
    bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh
```

| Variant | ms/iter | TFLOP/s/GPU | Peak VRAM/GPU | lm loss at iteration 10 |
| --- | --- | --- | --- | --- |
| No recompute | 2202 | 646 | 197–199 GiB | 5.57984 |
| `PRIMUS_RECOMPUTE_LAYER_IDS=0,3` | 2543 | 558 | 181–191 GiB | 5.57984 |
| 1-GPU (one MI455X, another node) | 671 | 942 | - | 9.4595 |

**Grad norms per iteration.**

| Variant | Grad norms, iterations 1–10 |
| --- | --- |
| No recompute | 3.790, 3.758, 3.795, 4.995, 9.034, 10.183, 7.921, 7.203, 6.827, 6.680 |
| 1-GPU | 3.193, 3.232, 3.314, 3.263, 3.210, 3.713, 3.362, 3.779, 3.329, 3.289 |

**Profile of the no-recompute run, rank 0.**

- Shares of GPU compute time: TE unfused attention 45% (its softmax alone 24%), MoE grouped GEMMs
  22%, output layer 10%.
- About 150 ms per iteration of MoE all-to-all is not overlapped with compute.
- `PRIMUS_RECOMPUTE_LAYER_IDS=0,3` adds about 320 ms of compute and 37 ms of all-to-all per iteration.

## Optimizations on top of FlyDSL attention and grouped GEMM

Measured on 4x MI455X with `amdprimus/amdprimus:tas-20261015` (one host, 2026-10-16; ms/iter over
iterations 3-10 without profiled iterations). Each row adds one change to the row above.

| Configuration | ms/iter | TFLOP/s/GPU | tokens/s/GPU |
| --- | --- | --- | --- |
| FlyDSL attention + FlyDSL grouped GEMM, all opt-ins `0` | 1218 | 1165 | 53.8k |
| + `PRIMUS_MOE_A2A_SKIP_SELF` | 1154 | 1230 | 56.8k |
| + `PRIMUS_NO_HOST_SYNC`, no dataloader attention mask | 1133 | 1253 | 57.8k |
| + `PRIMUS_TURBO_CROSS_ENTROPY` | 1119 | 1269 | 58.6k |
| + `PRIMUS_ROUTER_TRITON_GEMM` (the defaults) | 1089-1096 | 1295-1303 | 59.8-60.2k |

Per iteration on rank 0, the profile moves as follows. Exposed MoE all-to-all drops from 157 to 92 ms, at a cost of
10 ms for the local self copy. GPU idle drops from 79 to 41 ms. Cross entropy drops from 19 to 7 ms. The router GEMMs and
their casts drop from 34.5 to 8.7 ms (router module 52 to 25 ms). The profiled step drops from 1256 to 1118 ms.

Tried and not kept on this host:
- `moe_shared_expert_overlap: true`: hangs at iteration 3.
- `--overlap_moe_expert_parallel_comm True` (combined 1F1B): 1117 ms/iter, but NaN/Inf within 2-8 iterations in all
  3 runs.
- `use_turbo_fused_act_with_probs: true`: NaN on all ranks in both runs.
- `moe_router_bwd_bf16: true`: -7 ms, BF16 rounding of the router gradient.
- RCCL channel/chunk/buffer environment variables: no gain.

## Known issues

- **Attention backends.** MLA (q/k 192, v 128) runs on Primus-Turbo FlyDSL (the default,
  `PRIMUS_TURBO_ATTN_BACKEND=flydsl`), Primus-Turbo Triton (`PRIMUS_TURBO_ATTN_BACKEND=triton`) or TE
  `UnfusedDotProductAttention` (`USE_TURBO_ATTENTION=False`). The FlyDSL kernels take BF16 causal MHA
  with sequence lengths that are multiples of 64, fixed-length batches (no varlen) and a q tensor of at
  most 1 GiB (`MBS` <= 5 at seq 4096 with 128 heads per GPU); with the backend set to `flydsl`, any other
  call raises `ValueError`. A Primus-Turbo without the FlyDSL MLA package
  `primus_turbo/flydsl/attention/gfx1250_mla/` (added to `dev/mi455x_v1` by e1a967a0, not on `main`
  yet), such as the `tas-20261011` image, a container created by the previous `prepare_mi455x.sh` or a
  `TURBO_DIR` checkout of `main`, raises that `ValueError` at the first attention call: set
  `PRIMUS_TURBO_ATTN_BACKEND=triton` or `USE_TURBO_ATTENTION=False` there. Context parallelism (CP > 1)
  goes through Primus-Turbo's `flash_attn_usp_func`, which ignores `PRIMUS_TURBO_ATTN_BACKEND` and always
  uses aiter/CK, which has no gfx1250 kernels for these head dims, so CP > 1 does not run with
  Primus-Turbo attention on gfx1250.
- **FlyDSL grouped GEMM.** It needs `gradient_accumulation_fusion: false` and at most 64 local experts;
  with the backend set to `bf16:flydsl`, other calls raise `ValueError`. Primus-Turbo documents a rare,
  nondeterministic wrong result from these kernels, with no error or NaN: about 1 call in 130,000 in
  steady state on the largest shape, and much more often over a process's first few hundred calls.
  Compare grad norms against a reference run; `TURBO_GROUPED_GEMM_BACKEND=default` uses the Triton
  grouped GEMM.
- **Silent gradient corruption on one GPU.** 9 of 29 ten-iteration 1-GPU runs left the reference grad
  norms at a random iteration (a wrong value, NaN or Inf). All of these runs used the same GPU, the
  Triton grouped GEMM, and TE `UnfusedDotProductAttention` or Primus-Turbo Triton attention. Megatron
  aborts only on NaN/Inf, so compare grad norms against a reference run. None of six 4-GPU runs on
  another node (TE `UnfusedDotProductAttention`, Triton grouped GEMM) deviated.
