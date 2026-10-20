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
  - With the `MI455X` yamls, one run each: 20B 2202.3 ms, 120B proxy 449.0 ms, same lm loss and
    peak memory.
- Every run finished 10/10 iterations with 0 NaN and `check_for_nan_in_loss_and_grad` on.
- 20B with `EXTRA_ARGS="--overlap_moe_expert_parallel_comm True"`: 2177 ms/iter and 385 GB (89%),
  same lm loss.
- 20B at `MBS=4 GBS=64`: 2365 ms/iter, 766 TFLOP/s/GPU (7% below MBS 8), 259 GB (60%).
- 20B with `EXTRA_ARGS="--use_turbo_grouped_gemm True --turbo_grouped_gemm_backend bf16:flydsl"`:
  2115.8 and 2119.9 ms/iter (4% faster), 855 TFLOP/s/GPU, 30,940 tokens/s/GPU, same peak memory,
  lm loss 11.50695.

Shape: hidden 2880, 64 heads / 8 KV groups, 32 experts / top-4 for 20B (128 experts for 120B),
vocab 201088. Routing uses the yamls' `moe_router_force_load_balancing: true`.

### 120B tuning

The 120B yaml keeps the gfx1250 runbook's proxy settings. These overrides run the same proxy at
4.2x the tokens/s:

```bash
MODEL=120B MBS=8 GBS=64 CE_IMPL=te EXTRA_ARGS="--sink_sliding_window 128 \
  --recompute_granularity null --recompute_method null --recompute_num_layers null \
  --use_turbo_grouped_gemm True --turbo_grouped_gemm_backend bf16:flydsl \
  --overlap_moe_expert_parallel_comm True --use_turbo_fused_qk_rmsnorm_rope True \
  --use_turbo_norm_te_linear True --use_turbo_rms_norm True --moe_permute_fusion True \
  --moe_use_fused_router_with_aux_score True --moe_router_fusion True --moe_router_dtype fp32 \
  --moe_shared_expert_overlap False --ddp_num_buckets 24 --ddp_average_in_collective True" \
  bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
```

This runs at 850 ms/iter, 889 TFLOP/s/GPU, 77,070 tokens/s/GPU and 302 GB (70%). The table builds it
up one change per row, one run each. Rows after the first also set `expandable_segments:True`, which
saves about 7 GB of peak memory at the same speed:

| change | MBS / GBS | ms/iter | TFLOP/s/GPU | tokens/s/GPU | peak memory |
|---|---|---|---|---|---|
| the 120B yaml | 1 / 8 | 449 | 211 | 18,245 | 207 GB (48%) |
| 20B yaml settings (sliding window, fused QK RMSNorm and RoPE, TE CE, Turbo RMSNorm, fused router and permute) | 1 / 8 | 396 | 239 | 20,696 | 197 GB (46%) |
| no recompute | 1 / 8 | 323 | 293 | 25,382 | 202 GB (47%) |
| larger batch | 8 / 64 | 947 | 798 | 69,204 | 276 GB (64%) |
| Turbo FlyDSL grouped GEMM | 8 / 64 | 869 | 869 | 75,410 | 277 GB (64%) |
| all-to-all overlap | 8 / 64 | 853 | 885 | 76,782 | 294 GB (68%) |

- With 128 experts, top-4 and EP4, each expert receives about 512 x MBS tokens per micro batch, too
  few to fill the GPU at MBS 1; throughput keeps rising with MBS.
- Without grouped GEMM: MBS 2 / 4 / 16 run at 41,785 / 57,717 / 73,939 tokens/s/GPU.
  MBS 16 / GBS 128 with grouped GEMM: 1683 ms, 898 TFLOP/s/GPU, 77,890 tokens/s/GPU, 361 GB (84%).
- The 20B yaml settings move lm loss at iteration 10 from 11.8436 to 11.5905 at MBS 1. Recompute,
  grouped GEMM and all-to-all overlap leave it unchanged to 6 digits.

## Known issues

- **Triton expert scheduling.** Triton 3.8 enables `amdgpu-expert-scheduling-mode` for gfx1250,
  which produces random memory faults or NaN in Turbo Triton kernels.
  `TRITON_HIP_USE_EXPERT_SCHEDULING=0` alone still leaves the Inductor kernels that Inductor loads
  from the cache of an earlier run in expert mode, so the script also patches
  `is_expert_scheduling_enabled()` in Triton's AMD backend.
- **Turbo DeepEP.** With `use_turbo_deepep: true` (the MI355X 20B yaml) the first dispatch on gfx1250
  times out (`DeepEP error: CPU recv timeout`) and the ranks keep the GPUs busy until they are
  killed. The MI455X yamls turn DeepEP off. Turbo DeepEP assumes 64-lane wavefronts
  (`kWarpSize = 64` in `csrc/include/primus_turbo/deep_ep/configs.h`); gfx1250 runs 32-lane waves.
  On gfx1250, `__ballot` returns only 32 bits, so `__all_sync(kFullWarpMask, ...)` is never true
  and `barrier_block`, which `notify_dispatch` calls first, never returns: on 4 GPUs Turbo's barrier
  spins for more than 20 s, and with only that mask changed to 32 bits it returns in under 0.1 s.
  `warp_reduce_sum` also returns twice the sum of the first 32 of the 64 logical lanes, so the mask
  change alone does not fix DeepEP. A 4-rank standalone run (GPT-OSS 120B MoE shape) gets a correct
  `get_dispatch_layout`, then `dispatch` raises `CPU recv timeout` after
  `PRIMUS_TURBO_DEEPEP_CPU_TIMEOUT`. Turbo's own test suite skips DeepEP on gfx1250.
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
