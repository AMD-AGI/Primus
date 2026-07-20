# Performance Experiment Synthesis — Pure GDN 1B / 100B-tokens

Captures the full 2026-05-26 perf-tuning session that took the run from
**+19.8% vs FLA to +3.7% vs FLA** (closing 85% of the gap).

## Final result

| Metric | Old prod (ZeRO-1) | New prod (FSDP+overlap) | Δ |
|---|---|---|---|
| Steady-state iter time | 2768 ms | **2395 ms** | **−373 ms (−13.5%)** |
| TFLOP/s/GPU | 736 | **834** | +13.3% |
| Tokens/s/GPU | 47,950 | **54,650** | +14.0% |
| **100B run wall time** | **73.2 h** | **63.4 h** | **−9.8 h** |
| vs FLA reference (2310 ms) | +19.8% | **+3.7%** | gap −16 pp |
| Persistent allocated/rank | 8.8 GB | **4.8 GB** | −46% (ZeRO-2 sharding works) |
| Loss @ iter N | bit-identical to FLA (within 1-2% of FLA's trajectory) |

## Experiments run (50 iters each)

| # | Setup | Result | Steady-state | Insight |
|---|---|---|---|---|
| EXP0 | ZeRO-1 baseline (then-current prod) | OK | 2768 ms (+19.8%) | reference |
| EXP1 | EXP0 + PyTorch profiler iters 30-33 | OK | 2763 ms | profiler overhead ≈0; trace captured for ground-truth GPU breakdown |
| EXP3 | EXP0 + `stderr_sink_level: WARNING` | OK (logs suppressed) | wall-time ~−60 ms vs EXP0 | **falsifies "logging is the gap"** — overhead is only ~2% |
| EXP4 | EXP0 + `overlap_grad_reduce: true` (ZeRO-1) | **SIGSEGV rank 5** | (failed) | confirms ZeRO-1 bucket overlap is broken for heterogeneous Mamba params on this Megatron+ROCm build |
| EXP5 | FSDP ZeRO-2 + `empty_cache_interval: 128` | OK | 2684 ms (+16.2%) | first FSDP win — 84 ms saved, memory drops 4 GB |
| EXP6 | FSDP ZeRO-3 (`optim_grads_params`) | `RuntimeError: Triton can't deref sharded params` | (failed) | **ZeRO-3 incompatible with FLA Triton GDN kernels** — params must stay un-sharded on the device they're called from |
| **EXP7** | **FSDP ZeRO-2 + `overlap_grad_reduce: true` + `overlap_param_gather: true`** | **OK** | **2415 ms (+4.5%)** | **WINNER — 269 ms saved; FSDP overlap is a DIFFERENT code path than ZeRO-1's (which segfaulted)** |
| EXP8 | EXP0 + `PRIMUS_FUSED_CE_CHUNKS=16` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | **OOM iter 1** | (failed) | CUBLAS workspace bump pushed past 192 GB |
| EXP9 | EXP7 + EXP8 | **OOM iter 1** | (failed) | same root cause as EXP8 |

## Ground-truth GPU breakdown (from EXP1 profiler trace)

One normal iter (2757 ms wall), GPU-only events:

| Category | GPU ms | % of wall | Comment |
|---|---|---|---|
| Linear GEMMs (hipBLASlt) | 1678 | 60.9% | identical to FLA |
| NCCL collectives | 212 | 7.7% | reduced to ~60 ms after EXP7 overlap |
| GDN+SwiGLU+RMSNorm Triton | 345 | 12.5% | identical to FLA |
| Pointwise + Reductions | 142 | 5.2% | Megatron grad-bucket scatter |
| GDN short-conv | 45 | 1.6% | causal_conv1d |
| Fused-CE loss (32 chunks) | 22 | 0.8% | already optimized |
| Other GPU | 67 | 2.4% | TE multi-tensor-apply, sorts |
| **GPU subtotal** | **2503** | **90.8%** | |
| CPU bubble (kernel launch loop) | **254** | **9.2%** | Megatron Python overhead — irreducible without CUDA Graphs |

## Where the 447 ms initial gap to FLA went

| Component | Δ vs FLA | Closed by | Remaining |
|---|---|---|---|
| NCCL all-reduce | +152 ms | **EXP7 overlap** | ~30 ms |
| Allocator thrash (peak VRAM) | +90 ms | **EXP5 empty_cache_interval=128** | ~10 ms |
| Pointwise + Reductions | +62 ms | mostly EXP7 (overlapped) | ~30 ms |
| Other GPU misc | +42 ms | partial | ~30 ms |
| Fused-CE chunks (32 vs 8) | +14 ms | **not closed** (OOM risk) | ~14 ms |
| **CPU bubble (Python launch loop)** | **+184 ms** | **not closed** (would need CUDA Graphs) | ~70 ms |
| **TOTAL** | **+447 ms** | **−362 ms** | **+85 ms (3.7%)** |

## Winning configuration

`examples/megatron/configs/MI300X/zebra_llama_1B_gdn_pure_100B-pretrain.yaml`:

```yaml
use_megatron_fsdp: true
data_parallel_sharding_strategy: optim_grads   # ZeRO-2 (NOT _params, that's ZeRO-3, breaks Triton)
overlap_grad_reduce: true                      # FSDP path's overlap (DIFFERENT from ZeRO-1's broken one)
overlap_param_gather: true                     # works alongside overlap_grad_reduce
empty_cache_interval: 128                      # rare flush — between flushes FSDP shines
ckpt_format: fsdp_dtensor                      # required by FSDP
```

`launch_gdn_pure_1B_100B.sh` exports (auto-applied):
```bash
export NCCL_BUFFSIZE=2097152                   # 2 MiB
export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_PEER=1               # all three required to fit FSDP workspace in <16 MiB/comm at 99% VRAM
```

`primus/backends/megatron/patches/env_patches.py:set_cuda_device_max_connections`
automatically sets `CUDA_DEVICE_MAX_CONNECTIONS=8` whenever FSDP is enabled
(required for stream parallelism — overlap won't work with =1).

## What we tried that did NOT help

- **CUBLAS_WORKSPACE_CONFIG=:4096:8** (EXP8) — would help GEMMs ~30 ms but adds 256 MB peak VRAM → OOM
- **HIPBLASLT_WORKSPACE_SIZE bump** (EXP8) — same OOM
- **Fused-CE chunks 32→16** (EXP8) — adds ~1 GB peak → OOM under EXP7's higher overlap-residency footprint
- **WARNING-level logging** (EXP3) — only saves ~60 ms, not the >100 ms I'd hypothesized
- **ZeRO-3 / `optim_grads_params`** (EXP6) — Triton GDN kernels can't deref sharded params
- **`overlap_grad_reduce=true` with ZeRO-1** (EXP4) — known Megatron bug for heterogeneous Mamba params

## Files

| Path | Purpose |
|---|---|
| `examples/megatron/configs/MI300X/zebra_llama_1B_gdn_pure_100B-pretrain.yaml` | production config |
| `examples/megatron/configs/MI300X/zebra_llama_1B_gdn_pure_exp7-fsdp-overlap.yaml` | the winning 50-iter diag (= prod config + reduced train_iters) |
| `launch_gdn_pure_1B_100B.sh` | production launcher (sets NCCL clamp, allocator knobs) |
| `primus/backends/megatron/patches/empty_cache_interval_patches.py` | the `empty_cache_interval` patch |
| `experiments/run_perf_exp.sh` | single-experiment runner — pass YAML + env-var overrides |
| `experiments/profile_breakdown.py` | PyTorch trace per-category analyzer (use `--gpu-only` for clean breakdown) |
| `experiments/results/archive/` | summaries from EXP0,1,3,5,7 + full EXP7 log |

## How to reproduce / extend

```bash
# Reproduce EXP7's 2415 ms result (50-iter diag, ~4 min):
bash experiments/run_perf_exp.sh exp7_diag \
    examples/megatron/configs/MI300X/zebra_llama_1B_gdn_pure_exp7-fsdp-overlap.yaml

# Try a new lever (e.g. enable a flag via env var override):
bash experiments/run_perf_exp.sh my_test \
    examples/megatron/configs/MI300X/zebra_llama_1B_gdn_pure_exp7-fsdp-overlap.yaml \
    MY_NEW_ENV_VAR=1

# Profile a run: edit the YAML to set profile: true, run, then:
python3 experiments/profile_breakdown.py \
    output/.../tensorboard/*.pt.trace.json \
    --gpu-only
```
