# GPT-OSS on MI455X (gfx1250)

BF16 pretrain of GPT-OSS 20B and a GPT-OSS 120B proxy on one MI455X node (4 GPUs, mock data).
`run_gpt_oss_mi455x.sh` runs `examples/megatron/configs/MI455X/gpt_oss_{20B,120B}-BF16-pretrain.yaml`
and reproduces the baseline below. The MI455X yamls are separate from the MI355X ones, so MI355X
tuning does not change this baseline.

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
| `DOCKER_IMAGE` | `amdprimus/amdprimus:tas-20261019` | image used by the container launcher |
| `PRIMUS_LAUNCHER` | `container` | `container` (from the host) or `direct` (inside the container) |
| `PRIMUS_EP` | `4` | expert parallel size; TP and PP are 1 |
| `MBS`, `GBS` | 20B: `8`, `64`; 120B: `1`, `8` | micro / global batch size |
| `TRAIN_ITERS` | `10` | |
| `NUM_LAYERS` | `8` | 120B proxy only |
| `CE_IMPL` | 20B: `te`; 120B: `native` | `--cross_entropy_fusion_impl` |
| `EXTRA_ARGS` | | extra Megatron overrides, e.g. `--sink_sliding_window 128` |
| `PATCH_TRITON_EXPERT_SCHEDULING` | `1` | patch Triton's gfx1250 expert-scheduling default off |

## Baseline (2026-10-20)

| model | layers | parallel | MBS / GBS | seq | ms/iter | TFLOP/s/GPU | tokens/s/GPU | lm loss @10 | peak memory |
|---|---|---|---|---|---|---|---|---|---|
| GPT-OSS 20B | 24 | EP4 DP4 | 8 / 64 | 4096 | 2208 | 820 | 29,680 | 11.50698 | 346 GB (80%) |
| GPT-OSS 120B proxy | 8 of 36 | EP4 DP4, full recompute | 1 / 8 | 4096 | 447 | 212 | 18,340 | 11.8436 | 207 GB (48%) |

- Hardware: 4x MI455X, 432 GB each, one node.
- Image: `amdprimus/amdprimus:tas-20261019`
  (`sha256:d653324d124e218f43c92eaac3cfc5d05059bd45fb4abf5bdfb78b987af4af87`). It ships
  torch 2.14.0+rocm10.2.0a20261011, Triton 3.8.0, Transformer Engine 2.19.0.dev0 and
  Primus-Turbo `main` 21588d59. GPT-OSS uses Turbo's Triton attention and RMSNorm
  (`use_turbo_grouped_gemm: false`). The previous image, `tas-20261011` (Turbo 1d47bb72), gives the
  same lm loss and grad norms and is within 1% in time.
- Primus: `dev/mi455x_v1` merged with `main` 5e8a4abb. The model uses GPT-OSS's vocabulary (201088,
  `main` #1249). The 20B yaml starts from the MI355X tuning in `main` #1252 (128-token sliding window
  on even layers, fused QK RMSNorm and RoPE, TE cross entropy, no all-to-all overlap, expandable
  segments) with Turbo DeepEP off (see Known issues).
- Numbers are Megatron's averages over iterations 3-10, the mean of 2 runs. Peak memory is the largest
  logged `hip mem usage`. Repeats:
  - 20B: 2204.8 and 2212.0 ms; on `tas-20261011` 2216.3 and 2219.8 ms.
  - 120B proxy: 445.4 and 447.9 ms; on `tas-20261011` 442.8 and 448.6 ms.
  - Runs of a model gave the same lm loss and grad norms.
- Every run finished 10/10 iterations with 0 NaN and `check_for_nan_in_loss_and_grad` on.
- 20B with `EXTRA_ARGS="--overlap_moe_expert_parallel_comm True"`: 2177 ms/iter and 385 GB (89%),
  same lm loss.

Shape: hidden 2880, 64 heads / 8 KV groups, 32 experts / top-4 for 20B (128 experts for 120B),
vocab 201088. Routing uses the yamls' `moe_router_force_load_balancing: true`.

## Known issues

- **Triton expert scheduling.** Triton 3.8 enables `amdgpu-expert-scheduling-mode` for gfx1250,
  which produces random memory faults or NaN in Turbo Triton kernels.
  `TRITON_HIP_USE_EXPERT_SCHEDULING=0` alone still leaves the Inductor kernels that Inductor loads
  from the cache of an earlier run in expert mode, so the script also patches
  `is_expert_scheduling_enabled()` in Triton's AMD backend.
- **Turbo DeepEP.** With `use_turbo_deepep: true` (the MI355X 20B yaml) the first dispatch on gfx1250
  times out (`DeepEP error: CPU recv timeout`) and the ranks keep the GPUs busy until they are
  killed. The MI455X yamls turn DeepEP off.
- **20B native fused cross entropy fails at step 1** with `CUDA error: out of memory`. Inductor
  compiles Megatron's native CE, and its autotuner backs up the 8x4096x201088 fp32 logits (about
  26 GB per rank) into pinned host memory. Four ranks exceed the amdgpu pinned-memory limit. The
  20B yaml uses TE cross entropy.
- **120B profiler.** The torch profiler kills the process in `gilstate_tss_set` after training, so
  the 120B yaml turns it off.
- **20B NaN from the gfx1250 runbook did not reproduce.** The runbook reports Inf at iteration 1
  for this 20B shape with Turbo attention. It did not occur in the runs here, all with the yaml
  default `overlap_param_gather: true`. The runbook describes the bug as intermittent, so
  keep the NaN check on.

## Differences from the real GPT-OSS model

The MI455X yamls keep these settings, so the runs stay comparable with the gfx1250 runbook:

- `primus/configs/models/megatron/gpt_oss_{20B,120B}.yaml` add a shared expert
  (`moe_shared_expert_intermediate_size: 2880`). GPT-OSS has none.
- The 120B yaml sets `sink_sliding_window: 0`, so every layer is full causal attention with sinks.
  GPT-OSS, and the 20B yaml, use a 128-token window on even layers. The Turbo Triton kernel
  implements the window as a mask: it still visits every causal K block in the forward and both
  backward kernels, so the window changes numerics and saves no compute.
