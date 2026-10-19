# GPT-OSS on MI455X (gfx1250)

BF16 pretrain of GPT-OSS 20B and a GPT-OSS 120B proxy on one MI455X node (4 GPUs, mock data).
`run_gpt_oss_mi455x.sh` reproduces the v0 baseline below.

## Run

From the host, the script starts `DOCKER_IMAGE` and re-runs itself inside it:

```bash
MODEL=20B  bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
MODEL=120B bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
```

Inside an already running container, add `PRIMUS_LAUNCHER=direct`. The log is written to
`output/<PRIMUS_TEAM>/<PRIMUS_USER>/<PRIMUS_EXP_NAME>/log_node0.txt`.

| variable | default | meaning |
|---|---|---|
| `MODEL` | `20B` | `20B` (24 layers) or `120B` (8-of-36-layer proxy, full recompute) |
| `DOCKER_IMAGE` | `amdprimus/amdprimus:tas-20261011` | image used by the container launcher |
| `PRIMUS_LAUNCHER` | `container` | `container` (from the host) or `direct` (inside the container) |
| `PRIMUS_EP` | `4` | expert parallel size; TP and PP are 1 |
| `MBS`, `GBS` | 20B: `8`, `64`; 120B: `1`, `8` | micro / global batch size |
| `TRAIN_ITERS` | `10` | |
| `NUM_LAYERS` | `8` | 120B proxy only |
| `CE_IMPL` | 20B: `te`; 120B: `native` | `--cross_entropy_fusion_impl` |
| `EXTRA_ARGS` | | extra Megatron overrides, e.g. `--sink_sliding_window 128` |
| `PATCH_TRITON_EXPERT_SCHEDULING` | `1` | patch Triton's gfx1250 expert-scheduling default off |

## v0 baseline (2026-10-13)

| model | layers | parallel | MBS / GBS | seq | ms/iter | TFLOP/s/GPU | tokens/s/GPU | lm loss @10 | peak memory |
|---|---|---|---|---|---|---|---|---|---|
| GPT-OSS 20B | 24 | EP4 DP4 | 8 / 64 | 4096 | 2114 | 819 | 30,996 | 11.34809 | 393 GB (91%) |
| GPT-OSS 120B proxy | 8 of 36 | EP4 DP4, full recompute | 1 / 8 | 4096 | 447 | 191 | 18,347 | 11.39837 | 201 GB (46%) |

- Hardware: 4x MI455X, 432 GB each, one node.
- Image: `amdprimus/amdprimus:tas-20261011`
  (`sha256:e49a25905a2e05fd3009478e6601021f7c775da503811eb3634b36837e6bfad7`). It ships
  torch 2.14.0+rocm10.2.0a20261011, Triton 3.8.0 and Transformer Engine 2.19.0.dev0.
- Primus: `dev/mi455x_v1` at `ab873556` plus this script.
- Primus-Turbo: the image's `1d47bb72` (FlyDSL 0.2.4) and `dev/mi455x_v1` `a1800137`
  (FlyDSL 0.3.4.1) give the same numbers. GPT-OSS uses Turbo's Triton attention and RMSNorm
  (`use_turbo_grouped_gemm: false`). The two commits differ only under `primus_turbo/flydsl`.
- Numbers are Megatron's averages over iterations 3-10 of one run. Repeats:
  - 20B, 5 runs: 2107.5-2114.6 ms. lm loss and grad norm (2.031) are bit-identical across runs.
  - 120B proxy, 4 runs: 436.5-446.5 ms. Iteration 3 is sometimes slow (554 ms); iterations
    4-10 run at about 430 ms.
- Every run finished 10/10 iterations with 0 NaN and `check_for_nan_in_loss_and_grad` on.

Shape: hidden 2880, 64 heads / 8 KV groups, 32 experts / top-4 for 20B (128 experts for 120B),
vocab 131072 (padded to 131200). Routing uses the example yaml's
`moe_router_force_load_balancing: true`.

## Known issues on this image

- **Triton expert scheduling.** Triton 3.8 enables `amdgpu-expert-scheduling-mode` for gfx1250,
  which produces random memory faults or NaN in Turbo Triton kernels.
  `TRITON_HIP_USE_EXPERT_SCHEDULING=0` alone still leaves some Inductor kernels in expert mode,
  so the script also patches `is_expert_scheduling_enabled()` in Triton's AMD backend.
- **GPU detection.** Primus reports the GPU as `AMD Radeon Graphics` and skips
  `runner/helpers/envs/MI455X.sh` unless `PRIMUS_GPU_MODEL=MI455X` is set.
- **20B native fused cross entropy fails at step 1** with `CUDA error: out of memory`. Inductor
  compiles Megatron's native CE, and its autotuner backs up the 8x4096x131200 fp32 logits (about
  17 GB per rank) into pinned host memory. Four ranks exceed the amdgpu pinned-memory limit. The
  script uses TE cross entropy for 20B. At MBS 4, where native CE fits, both implementations run
  clean: lm loss 11.34770 (native) vs 11.34772 (TE), and TE is about 1% faster.
- **120B profiler.** The 120B example yaml enables the torch profiler, which kills the process
  in `gilstate_tss_set` after training. The script disables it.
- **Turbo in the image is not `dev/mi455x_v1`.** The image installs Turbo `1d47bb72` as a copy in
  `site-packages`, not as an editable install of `/workspace/Primus-Turbo`. That build imports
  `flydsl.expr.buffer_ops`, which FlyDSL 0.3.4.1 removed. Upgrading FlyDSL alone therefore breaks
  Turbo, and the `dev/mi455x_v1` Python package has to replace the installed copy.
- **20B NaN from the gfx1250 runbook did not reproduce.** The runbook reports Inf at iteration 1
  for this 20B shape with Turbo attention. It did not occur in 8 runs here, all with the yaml
  default `overlap_param_gather: true`. The runbook describes the bug as intermittent, so
  keep the NaN check on.

## Differences from the real GPT-OSS model

v0 keeps the example yaml's model settings, so it stays comparable with the gfx1250 runbook:

- `primus/configs/models/megatron/gpt_oss_{20B,120B}.yaml` add a shared expert
  (`moe_shared_expert_intermediate_size: 2880`). GPT-OSS has none.
- `sink_sliding_window: 0`, so every layer is full causal attention with sinks. GPT-OSS uses a
  128-token window on even layers (`EXTRA_ARGS="--sink_sliding_window 128"`). The Turbo Triton
  kernel implements the window as a mask: it still visits every causal K block in the forward and
  both backward kernels. With the window on, 20B runs 10/10 with 0 NaN and lm loss 11.10439, but
  takes 2140 ms/iter instead of 2114: the window changes numerics and saves no compute.
- The vocabulary is 131072; GPT-OSS uses 201088.
