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
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | `hipblaslt_tuning/gpt_oss-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt` | hipBLASLt GEMM tuning file (see GEMM tuning); empty for hipBLASLt's default kernel selection |

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
- Measured before GEMM tuning (below), which the script now loads by default.
  `HIPBLASLT_TUNING_OVERRIDE_FILE=` reproduces the baseline. With the tuning file, 20B runs at
  2107 ms/iter, 860 TFLOP/s/GPU and 31,109 tokens/s/GPU.
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

This runs at 850 ms/iter, 889 TFLOP/s/GPU, 77,070 tokens/s/GPU and 302 GB (70%), measured before
GEMM tuning (below). With the tuning file, it runs at 818 ms/iter, 924 TFLOP/s/GPU and 80,150
tokens/s/GPU. The table builds it
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

## GEMM tuning

The script sets `HIPBLASLT_TUNING_OVERRIDE_FILE` to
`hipblaslt_tuning/gpt_oss-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt`, a hipBLASLt
tuning file for GPT-OSS 20B and 120B at MBS 8 and sequence 4096. The two models differ only in
layer count and number of experts (32 / 128), so they share the attention, shared expert and output
layer GEMMs, with 32768 tokens per micro batch. The file adds the router GEMMs of both models and
the expert GEMMs with 8192 (20B, EP2), 16384 (20B, EP4) and 4096 (120B, EP4) tokens per expert.
For the wgrad GEMMs the tuned kernels use other macro tiles (2880 x 5760 x 32768: 176x192 by
default, 192x256 tuned); the output layer and QKV forward keep the 256x256x128 tile but switch to
a kernel with 32x4 instead of 64x2 waves per workgroup. With `--use_turbo_grouped_gemm True` the
expert GEMMs do not go through hipBLASLt, and only the other rows apply. At MBS 1 (the 120B yaml)
only the shared expert GEMMs have rows: with 4096 tokens they have the shapes of the 120B expert
GEMMs at MBS 8. By the per-GEMM times they save about 0.6 ms of the 447 ms iteration.

On 4 GPUs (EP4 DP4, MBS 8 / GBS 64, 2 runs each with default and tuned alternating, iterations
3-10, 2026-10-23):

| model | GEMMs | ms/iter | TFLOP/s/GPU | tokens/s/GPU | lm loss @10 | peak memory |
|---|---|---|---|---|---|---|
| 20B | hipBLASLt default | 2200.2 (2203.5, 2196.8) | 823 | 29,788 | 11.50698 | 346 GB (80%) |
| | tuning file | 2106.8 (2110.7, 2102.9) | 860 | 31,109 | 11.50688 | 346 GB (80%) |
| 120B proxy, 8 of 36 layers, "120B tuning" settings | hipBLASLt default | 849.6 (850.2, 848.9) | 889 | 77,141 | 11.48167 | 302 GB (70%) |
| | tuning file | 817.7 (817.5, 817.9) | 924 | 80,150 | 11.48168, 11.48167 | 302 GB (70%) |

- 20B: 4.2% faster (93 ms per iteration). 120B proxy: 3.8% faster (32 ms).
- The tuned kernels round differently, so lm loss moves in the fifth or sixth significant digit.
- Every run finished 10/10 iterations with 0 NaN.

On 2 GPUs (EP2 DP2, MBS 8 / GBS 32, 2 runs each, iterations 3-10, 2026-10-23):

| model | GEMMs | ms/iter | TFLOP/s/GPU | tokens/s/GPU | lm loss @10 | peak memory |
|---|---|---|---|---|---|---|
| 20B, 12 of 24 layers | hipBLASLt default | 1198.1 (1198.5, 1197.6) | 851 | 54,702 | 11.47197 | 264 GB (61%) |
| | tuning file | 1142.3 (1143.8, 1140.8) | 892 | 57,372 | 11.47202 | 264 GB (61%) |
| 120B, 4 of 36 layers | hipBLASLt default | 540.6 (540.6, 540.6) | 910 | 121,231 | 11.24907 | 257 GB (59%) |
| | tuning file | 508.3 (508.4, 508.2) | 967 | 128,937 | 11.24908 | 257 GB (59%) |

- 20B runs with the 20B yaml: 4.7% faster.
- 120B runs with the settings from "120B tuning": 6.0% faster. These use Turbo grouped GEMM, so
  the expert GEMMs do not go through hipBLASLt.
- Short runs with `HIPBLASLT_LOG_MASK=32` (20B at MBS 8; 120B at MBS 1 and 8, and at MBS 16
  without grouped GEMM) showed every GEMM shape with a row using the file's solution index, and the
  others hipBLASLt's default.
- These runs and the per-GEMM times below come from a different node than the baseline. There the
  default selection ran within 0.4% of the baseline (2200 against 2208 ms, 850 against 850 ms), but
  clock and power behavior can differ between MI455X nodes, so compare the relative change across
  nodes, not the times.

**Per GEMM.** `hipblaslt_tuning/gpt_oss-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.md`
times all 39 GEMM shapes with `hipblaslt-bench`, default selection against the file, at the
steady-state clock. Weighted by how often each GEMM runs, the GEMM time per GPU and iteration in ms:

| configuration | GEMMs, default | GEMMs, tuned | measured ms/iter |
|---|---|---|---|
| 20B, 2 GPUs, 12 layers | 601.5 | 552.0 (-49.5) | 1198.1 -> 1142.3 (-55.8) |
| 20B, 4 GPUs | 1051.4 | 973.3 (-78.1) | 2200.2 -> 2106.8 (-93.4) |
| 120B, 2 GPUs, 4 layers | 183.5 | 159.9 (-23.5) | 540.6 -> 508.3 (-32.3) |
| 120B proxy, 4 GPUs, "120B tuning" settings | 242.7 | 216.1 (-26.7) | 849.6 -> 817.7 (-31.9) |

- The measured savings are 13-37% larger than the GEMM time the file saves.
- Largest gains: output layer fprop 1.48x, QKV fprop 1.38x, router fprop 3.58x (20B) and 1.25x
  (120B), and the shared expert and expert fc1 / fc2 wgrad 1.06-1.35x.
- 19 of the 37 rows select the default kernel.
- Tuning does not help the fp32 router wgrad: about 1.05 ms per call, at 6 TFLOP/s for 20B and
  23 TFLOP/s for 120B; 50 ms per iteration for 20B on 4 GPUs.

**Reproducing the 4-GPU comparison.** For 20B, from the host:

```bash
MODEL=20B PRIMUS_EXP_NAME=gpt_oss_20B_hipblaslt_tuned \
  bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
MODEL=20B PRIMUS_EXP_NAME=gpt_oss_20B_hipblaslt_default HIPBLASLT_TUNING_OVERRIDE_FILE= \
  bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
for e in 20B_hipblaslt_tuned 20B_hipblaslt_default; do
  grep -oP "elapsed time per iteration \(ms\): \K[0-9.]+" \
    output/amd/$(whoami)/gpt_oss_$e/log_node0.txt |
    sed -n '3,10p' | awk -v e=$e '{s += $1} END {printf "%s: %.1f ms/iter\n", e, s / NR}'
done
```

For the 120B proxy, run the "120B tuning" command the same two ways, with
`PRIMUS_EXP_NAME=gpt_oss_120B_hipblaslt_tuned` and `gpt_oss_120B_hipblaslt_default`, and loop over
those names.

**hipBLASLt versions.** A solution index is a position in one hipBLASLt build's kernel library, so
the files are named `<model>-<arch>-hipblaslt_<git>-rocm_<version>.txt` after the build they were
tuned on. Each row also records the kernel name, and this hipBLASLt (1.5.0) drops a row at
load when its index names another kernel, so a file from another build falls back to default
selection rather than running the wrong kernel. Re-tune when `DOCKER_IMAGE` changes.

**How the file was made.** Primus' three-stage flow (`examples/README.md`, "HipblasLT Auto Tuning"),
on 2 GPUs, inside the container (the container launcher does not pass the tuning variables on).
Stage 1 dumps the shapes; the layer count does not change them. EP2 at MBS 16 gives each expert the
16384 tokens of EP4 at MBS 8; only its expert shapes were kept:

```bash
COMMON="PRIMUS_LAUNCHER=direct PRIMUS_HIPBLASLT_TUNING=1 GPUS_PER_NODE=2 HIP_VISIBLE_DEVICES=0,1
  PRIMUS_EP=2 GBS=32 TRAIN_ITERS=3"
env $COMMON PRIMUS_HIPBLASLT_TUNING_STAGE=1 MBS=8 EXTRA_ARGS="--num_layers 4" \
  bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
env $COMMON PRIMUS_HIPBLASLT_TUNING_STAGE=1 MBS=16 EXTRA_ARGS="--num_layers 2" \
  HIPBLASLT_LOG_FILE=$PWD/output/tune_hipblaslt/mbs16.txt \
  bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
grep -E -- "-n 16384 |-k 16384 " output/tune_hipblaslt/mbs16.txt | sort -u \
  > output/tune_hipblaslt/gpt_oss_20B-BF16-pretrain/gemm_shape/ep4_experts.txt
python examples/megatron/guides/offline_tune/offline_tune_gemm.py \
  --dump-shape-path-or-file output/tune_hipblaslt/gpt_oss_20B-BF16-pretrain/gemm_shape \
  --tune-result-path $PWD/tune_gemm_results.txt --num-devices 2
```

For 120B, stage 1 ran at MBS 8 with the "120B tuning" settings for the router GEMMs, and at MBS 16
without grouped GEMM for the expert GEMMs (4096 tokens per expert, as at EP4 and MBS 8). Stage 2
ran only on the 9 shapes that 20B does not have:

```bash
A120="--sink_sliding_window 128 --recompute_granularity null --recompute_method null
  --recompute_num_layers null --use_turbo_fused_qk_rmsnorm_rope True --use_turbo_norm_te_linear True
  --use_turbo_rms_norm True --moe_permute_fusion True --moe_use_fused_router_with_aux_score True
  --moe_router_fusion True --moe_router_dtype fp32 --moe_shared_expert_overlap False
  --ddp_num_buckets 24 --ddp_average_in_collective True"
env $COMMON PRIMUS_HIPBLASLT_TUNING_STAGE=1 MODEL=120B NUM_LAYERS=2 CE_IMPL=te MBS=8 \
  EXTRA_ARGS="$A120 --use_turbo_grouped_gemm True --turbo_grouped_gemm_backend bf16:flydsl
  --overlap_moe_expert_parallel_comm True" HIPBLASLT_LOG_FILE=$PWD/output/tune_hipblaslt/120B_mbs8.txt \
  bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
env $COMMON PRIMUS_HIPBLASLT_TUNING_STAGE=1 MODEL=120B NUM_LAYERS=2 CE_IMPL=te MBS=16 \
  EXTRA_ARGS="$A120" HIPBLASLT_LOG_FILE=$PWD/output/tune_hipblaslt/120B_mbs16.txt \
  bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
mkdir -p output/tune_hipblaslt/gpt_oss_120B/gemm_shape
{ grep -E -- "-m 128 |-n 128 |-k 128 " output/tune_hipblaslt/120B_mbs8.txt
  grep -E -- "-n 4096 |-k 4096 " output/tune_hipblaslt/120B_mbs16.txt | grep -v 65536; } | sort -u \
  > output/tune_hipblaslt/gpt_oss_120B/gemm_shape/new_shapes.txt
python examples/megatron/guides/offline_tune/offline_tune_gemm.py \
  --dump-shape-path-or-file output/tune_hipblaslt/gpt_oss_120B/gemm_shape \
  --tune-result-path $PWD/tune_gemm_results_120B.txt --num-devices 2
```

- Stage 2 needs `hipblaslt-bench`, which the image does not have (see below).
- Stage 2 took 23 minutes for 20B, where the three output layer GEMMs take about 10 minutes each,
  and 19 s for the 9 120B shapes. The script's optional report CSV fails at the end with this
  `hipblaslt-bench` (its command regex predates `--alphai`), but the tuning file is complete.
- The file holds the rows of both results (a row is a header line and a value line after the
  `Git Version` line), sorted, with three corrections. `offline_tune_gemm.py` times each candidate
  over only 20 iterations:
  - Its winner for 2880 x 16384 x 2880 (NN) was 1.5% slower than the default kernel in an A/B;
    re-tuned with 100 warm-up and 200 timed iterations, the default kernel won, and that row was
    replaced.
  - For the expert fc2 fprop at 8192 and 4096 tokens per expert, steady-state A/Bs favored the
    default kernel, so the file has no row for them (see the per-GEMM file).

### Building hipblaslt-bench

The image ships ROCm as TheRock Python wheels, without `/opt/rocm` and without `hipblaslt-bench`,
which `offline_tune_gemm.py` runs as `/opt/rocm/bin/hipblaslt-bench`.
`hipblaslt_tuning/hipblaslt_bench/CMakeLists.txt` builds only the bench, from the hipBLASLt sources
of the image's TheRock build and against the image's installed hipBLASLt, so the bench reports the
solution indices that training uses. Inside the container, from the repository root (about
1.5 minutes: 1 to fetch, the rest to build):

```bash
apt-get update && apt-get install -y liblapack-dev libblas-dev
D=$(rocm-sdk path --root)
SHA=$(python3 -c 'import json, sys; print(next(s["pin_sha"] for s in json.load(open(sys.argv[1]))["submodules"] if s["submodule_name"] == "rocm-libraries"))' \
  "$D/share/therock/therock_manifest.json")
SRC=$PWD/output/rocm-libraries
B=$PWD/output/hipblaslt-bench
git init -q "$SRC"
git -C "$SRC" sparse-checkout set projects/hipblaslt
git -C "$SRC" fetch --depth 1 --filter=blob:none https://github.com/ROCm/rocm-libraries.git "$SHA"
git -C "$SRC" checkout -q FETCH_HEAD
cmake -G Ninja -S examples/megatron/models/gpt-oss/hipblaslt_tuning/hipblaslt_bench -B "$B" \
  -DHIPBLASLT_SRC="$SRC/projects/hipblaslt" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1250 \
  -DCMAKE_PREFIX_PATH="$D;$D/lib/cmake" -DCMAKE_CXX_COMPILER="$D/lib/llvm/bin/amdclang++" \
  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_SKIP_BUILD_RPATH=ON
cmake --build "$B" --target hipblaslt-bench
mkdir -p /opt/rocm/bin && ln -sf "$B/clients/hipblaslt-bench" /opt/rocm/bin/hipblaslt-bench
```

- For `tas-20261019` the manifest pins `rocm-libraries` at 3d8f334b (hipBLASLt 1.5.0,
  `e1e7fdcab9`) with no TheRock patches. If a later manifest lists patches for it, apply them
  before building.
- `ldd "$B/clients/hipblaslt-bench"` should resolve `libhipblaslt.so.1` to
  `_rocm_sdk_devel/lib`, which is identical to the `_rocm_sdk_libraries` copy that torch loads.
- `liblapack-dev` and `libblas-dev` are only needed to build; the image has the runtime libraries.
  The build stays in `output/` of the mounted repository, so a new container only needs the last
  line again.

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
