# ROCm 10.2 port — DLRMv4 on MI355X

The DLRMv4 trainer on the UTD TheRock ROCm 10.2 image, using the image's own software, compared with the
`rocm/primus:v26.3` stack.

- **Branch:** `dlrmv4_opt-mi355-rocm10.2`, cut from `dlrmv4_opt-mi355` @ `d14a7cb`.
- **Rule:** the branch does not depend on `rocm/primus:v26.3`. v26.3 is only the baseline we measure against, running
  the unmodified `d14a7cb` tree.
- **Status (2026-11-03):** ROCm 10.2 is faster than v26.3 on the same nodes at 1, 2 and 4 nodes.
  1 node: +0.75 to +1.12% throughput (4 nodes tested), −0.9 to −1.4% time to target (7 pairs, 3 seeds).
  2 / 4 nodes (MLPerf RCP rules, 3 seeds each): −0.83% / −0.94% mean time to target, faster in 6 of 6 pairs.
  Convergence matches (§2).
- **Second system (2026-11-04), cluster B, Docker, 1 node:** reproduced. +1.16% throughput; −0.16 / −0.66 / −0.77% time
  to target (3 seeds, faster in 3 of 3). Needs T28 (pinned pre-grow memory check) on shared nodes. Multi-node pending.
- **cluster B re-check (2026-11-06), another node (b079), Docker, 1 node, 10.2 only:** convergence reproduced (same evals
  to target, AUC within 0.0006 per seed), MLPerf 6.1.0 compliance passes; time to target +1.5 / +1.0 / +0.7% (seed
  20261103 at eval 7) on a node with −0.6% 200-step throughput. Needs T29 (`NUMPY_MADVISE_HUGEPAGE=0`): without it two
  of three back-to-back runs stalled (+5.7%, +14.9%).

## 1. Software stack

| Component | **ROCm 10.2 (target)** | v26.3 (baseline, measurement only) |
|---|---|---|
| Docker image | `unifiedtrainingdockers.azurecr.io/utd/nightly:primus_rocm10.2_20261016`<br>`@sha256:da52c92f4040be08705ccbf356a807bc0252e384b516640318df280b987e9de6` | `rocm/primus:v26.3`<br>`@sha256:1a02b74a94d82131f3f119a94ccbed45bb5b4cd77a0616cb4e6a29e64a048482` |
| Image recipe | `scripts/cluster_a/dlrmv4_utd_rocm10.2.def` (Apptainer; keeps the image's GPU stack) | `scripts/cluster_a/dlrmv4.def` (reinstalls torch / FBGEMM / torchrec) |
| Trainer code | this branch | `dlrmv4_opt-mi355` @ `d14a7cb`, unmodified |
| ROCm | TheRock **10.2.0a20261015**, Python wheels (`_rocm_sdk_*`); `/opt/rocm` = symlink added by the recipe | 7.2.1 (`/opt/rocm`) |
| HIP | **7.17.26412** | 7.2.53211 |
| Compiler (hipcc) | AMD clang **24.0.0git** (`bdc35841`) | AMD clang 22.0.0git (roc-7.2.1) |
| Python | 3.12.3 | 3.12.3 |
| torch | **2.16.0a0+rocm10.2.0a20261015** (git `ef46daf1`) | 2.12.0+rocm7.2 (git `7661cd9c`) |
| c10d `nccl` backend | **nccl2** (torchcomms, torch 2.16 default) | legacy `ProcessGroupNCCL` |
| Triton | **3.9.0** | 3.6.0 |
| FlyDSL | **0.2.4** | 0.1.1.dev409 |
| FBGEMM GPU | **fbgemm-gpu-nightly-rocm 2026.10.16** (`80bd3c07`) | `10b77573` (built by the recipe) |
| torchrec | **1.9.2** | 1.7.0a0+bf55480 |
| hipBLASLt / rocBLAS | **1.5.0** (`100500-3dc098eb`) / **5.8.0** | 1.2.2 (`100202-dabb6df2b9`) / 5.2.0 |
| RCCL | **2.32.3** (TheRock; no MSCCL) | 2.27.7 |
| rocPRIM | **4.8** (`400800`) | 4.2 (`400200`) |
| Profiler backend (kineto) | rocprofiler-sdk 1.4.2 | roctracer |
| Added by the recipe | polars-u64-idx 1.33.1, gin-config 0.5.0, mlcommons logging 6.0.0-rc6, `/opt/rocm` symlink | same 3 packages |
| Image environment | `FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL=1` (recipe). Runs also set `GPU_USE_DEVICE_QUEUE=0 DEBUG_CLR_AQL_DEV_QUEUE=0` (neutral, I7) | as shipped |

Test systems:

| | cluster A | cluster B |
|---|---|---|
| Nodes | 8× MI355X (`gfx950`), bare metal | 8× MI355X (`gfx950`), VM, 2 NUMA nodes, 2.7 TB RAM |
| Driver | amdgpu 7.1.3 | amdgpu 6.16.13, kernel 6.8.12 |
| Container | Apptainer (`.def`) | Docker (`recommendation/Dockerfile`, built on the node) |
| Network | 8× 400 Gb/s RoCE (`ionic`), `scripts/cluster_a/cluster_a_fabric.env` | 8× 400 Gb/s RoCE (`ionic`), cluster B fabric env in `scripts/cluster_b/ab_r10_multinode.sbatch` |
| Scales tested | 1, 2, 4 nodes | 1 node |

## 2. Results

Same node for every comparison; ROCm 10.2 = tree v6 (commit `0752a3f`); v26.3 = unmodified `d14a7cb`.

**Throughput, 200-step benchmark** (median `global_sps` over steps 20–200, 2 runs per arm)

| Node | a002 | a004 | a007 | a006 |
|---|---|---|---|---|
| 10.2 vs v26.3 | +0.90% | +1.07% | +1.12% | +0.75% |

Steady step time (steps 80–200) is 0.6–1.1% lower on 10.2.

**Time to target** (full MLPerf runs to AUC 0.75, seed 731542219)

| Node | a002 | a004 | a006 | a002 |
|---|---|---|---|---|
| v26.3 / 10.2 (s) | 911.3 / 903.3 | 909.5 / 901.2 | 912.4 / 903.4 | 910.9 / 902.6 |
| Difference | −0.88% | −0.91% | −0.98% | −0.91% |

**Convergence, 3 seeds** (one node per seed, both stacks)

| Seed | Evals to target, v26.3 / 10.2 | Final AUC, v26.3 / 10.2 | Time to target, v26.3 / 10.2 | Difference |
|---|---|---|---|---|
| 731542219 | 5 / 5 | 0.7573 / 0.7571 | 908.5 / 899.2 s | −1.03% |
| 1234567 | 6 / 6 | 0.7564 / 0.7558 | 957.7 / 948.9 s | −0.92% |
| 20261103 | 8 / 7 | 0.7542 / 0.7501 | 1042.2 / 984.4 s | −1.42% at eval 7 |

All 14 full runs end `run_stop success`. For seed 20261103 the AUC matches within ±0.001 at every eval; eval 7
reached 0.74999 on v26.3 and 0.75010 on 10.2, so 10.2 stopped one eval earlier. That one-eval difference is
threshold noise, so the time is compared at eval 7 (−1.42%).

**cluster B, 1 node** (Docker; `scripts/cluster_b/ab_r10_1node.sbatch`; same node per comparison; each job first runs a
short untimed prime of both arms, then N before A)

| 200-step throughput (3 runs per arm, order A N N A A N) | v26.3 | 10.2 | Difference |
|---|---|---|---|
| Median `global_sps`, steps 20–200 | 83,070 · 83,803 · 83,806 | 84,553 · 84,466 · 84,577 | **+1.16%** (every 10.2 run faster) |

| Seed | Evals to target, v26.3 / 10.2 | Final AUC, v26.3 / 10.2 | Time to target, v26.3 / 10.2 | Difference |
|---|---|---|---|---|
| 731542219 | 5 / 5 | 0.7570 / 0.7572 | 915.4 / 909.4 s | −0.66% |
| 1234567 | 6 / 6 | 0.7564 / 0.7555 | 959.7 / 958.2 s | −0.16% |
| 20261103 | 7 / 7 | 0.7510 / 0.7502 | 1010.3 / 1002.5 s | −0.77% |

All runs end `run_stop success`; MLPerf compliance passes on 10.2 (mlcommons logging 6.0.0-rc6). MLPerf init:
~200 s on 10.2 vs ~280 s on v26.3. Mean −0.53% (cluster A, same seeds: −1.1%).

Notes for this system:
- **Order and warm-up matter.** The first run on a freshly set-up node pays a one-time cost, and v26.3's first eval is
  slow unless a full 10.2 run came before it. Hence the untimed primes and the fixed order N, A.
- **Memory-pressured nodes.** With a near-full page cache, T27's up-front pin made 10.2 stall mid-training (up to
  24 windows with 41× steps). T28 removes it: 0 stall windows after the check reclaims the cache.
- **RDMA:** both images use their own rdma-core on these nodes (no host library overlay).

**Multi-node convergence, 3 seeds per scale** (MLPerf RCP rules, `SHAPE=rcp`; both stacks on the same nodes, order
alternated)

| Nodes (GBS, LR, `SKIP_EVAL_EPOCH_PCT`) | Seed | Evals to target | Final AUC, v26.3 / 10.2 | Time to target, v26.3 / 10.2 | Difference |
|---|---|---|---|---|---|
| 2 (16384, 2e-6, 0.0303) | 731542219 | 6 / 6 | 0.7552 / 0.7520 | 581.8 / 575.7 s | −1.06% |
| 2 | 1234567 | 8 / 8 | 0.7579 / 0.7541 | 627.1 / 623.5 s | −0.57% |
| 2 | 20261103 | 9 / 9 | 0.7530 / 0.7522 | 653.1 / 647.5 s | −0.85% |
| 4 (32768, 4e-6, 0.041) | 731542219 | 5 / 5 | 0.7507 / 0.7512 | 380.1 / 376.0 s | −1.09% |
| 4 | 1234567 | 9 / 9 | 0.7527 / 0.7517 | 430.5 / 425.1 s | −1.25% |
| 4 | 20261103 | 10 / 10 | 0.7513 / 0.7517 | 443.0 / 440.9 s | −0.48% |

Mean −0.83% (2 nodes) and −0.94% (4 nodes). All 12 runs end `run_stop success`, and each pair stops at the same eval
and sample count. On 10.2 the b1pp / fused sequence all-to-all paths arm on every rank, with no JIT after `run_start`.
RCCL alone (2 nodes): 10.2's 2.32.3 matches 2.27.7 at ≥ 256 MB all-reduce (371–378 GB/s bus bandwidth), is 23% lower at
64 MB and 5% lower on a 64 MB all_to_all. None of that shows in training.

**Parity checks on 10.2**
- **Arming:** every fast path armed on 8/8 ranks, the same as v26.3: MX6 forward, L11 DO8, X2LB, FP8 Indexed-X,
  both diets, the Triton spec replay and the HIP preload. No rank logs DISABLED, FAILED CLOSED or declined.
- **No lazy work after `run_start`, training or eval:** 0 Triton compiles and 0 hipBLASLt library loads. The only
  exception is 3 FBGEMM bundles at step 1 (~89 ms), which predate 10.2.
- **Race check:** eval after every window, 32/32 evals clean.
- **MLPerf init:** 180–197 s vs 314–319 s. Init is not part of time to target.

**How the gap closed** (200-step, same node vs v26.3)

| Step | Gap |
|---|---|
| 10.2 runtime with v26.3's Triton / FlyDSL (diagnostic "compat" image) | −2.12% |
| + T14 (clang 24 spill fix) | −0.20% |
| Native image, first full port | −3.2% (MX6 forward off on Triton 3.9) |
| + T21 and the replay / preload tables | −0.4% |
| + T10, T20, T23, T24 (tree v5) | +0.6 to +0.9% |
| + T25–T27 (tree v6) | +0.75 to +1.12%; time to target −0.9% |

## 3. Changes from the baseline

Status: all done in the branch unless marked otherwise. Gains are same-node measurements on 10.2.

### 3.1 Image and launchers

| # | Change | Why |
|---|---|---|
| I1 | UTD image, digest-pinned; keep its torch / Triton / FlyDSL / FBGEMM / torchrec | target stack |
| I2 | Add polars-u64-idx, gin-config, mlcommons logging (`--no-deps`, pip-freeze guard) | missing in the image |
| I3 | `/opt/rocm` → `_rocm_sdk_devel` symlink | trainer HIP builds call `/opt/rocm/bin/hipcc` |
| I4 | `FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL=1` | parity with v26.3 (measured neutral) |
| I6 | Do not set `TRITON_LIBHIP_PATH` | Triton needs a name ending `.so`; TheRock ships `.so.7` |
| I7 | Image HIP queue knobs | measured neutral (+0.08%) |
| L1 | `scripts/cluster_a/run_dlrmv4.sh`: same-node A/B (`ORDER`, `SIF_X`, `ENV_X`, `ROCM_BIND_X`, `REC_X`), fresh `/tmp` and home per run, no core dumps, process-group cleanup, crash watchdog | fair, safe A/Bs |
| L6 | `run_dlrmv4.sh` multi-node (`submit -N <n>`): per-node staging, one `srun` step per run with all CPUs per task, `NNODES` / `NODE_RANK` / `MASTER_ADDR`; `SHAPE=rcp` = MLPerf RCP rules for 1 / 2 / 4 nodes; `scripts/cluster_a/cluster_a_fabric.env` (site RoCE settings, no library override) | 2- and 4-node A/Bs. An `srun` task gets 1 CPU by default: 8 ranks on one core, RCCL at 4–12 GB/s, and 10.2's RCCL fails `sched_setaffinity` |
| L2 | `check_image.sh`, `probe_image.py`, `path_diff.py`, `coload_shim.c`, `tspec_record.sbatch`, `gen_blas_replay_table.py` | diagnostics and table recording |
| I8 | `Dockerfile`, `requirements.txt`, `docs/training_recipe.md`, README: the 10.2 image as shipped plus the same additions as the `.def` | built and run on cluster B (Docker), ~8 min per node; stack matches the `.def` |
| L7 | `scripts/cluster_b/ab_r10_1node.sbatch`, `ab_r10_multinode.sbatch`: same-node Docker A/B calling `run_and_time.sh` (the cluster A runner's shapes), fresh caches per run, untimed primes, GPU-idle wait before exit, containers removed first on cancel | cluster B A/Bs (§2) |
| L4 | `scripts/launch_slurm.sh`: 10.2 image and version pins, no reinstall of torch / FBGEMM / torchrec, no v26.3 overlay, per-stack baked image, forwards `TRITON_SPEC_REPLAY_XVER` | written; not yet run end to end |

### 3.2 Toolchain ports

| # | Change | Without it |
|---|---|---|
| T1 | rocPRIM 4.8 `TargetWaveSize` in the onesweep sort | sort extension does not build |
| T2 | Census parsers accept clang 24's resource-report format | X2H / L11 LNMD gates fail closed |
| T3 | Pass torchrec 1.9.2's new keyword flags through the bf16 embedding wrapper | `TypeError` at step 1 |
| T7 | Triton 3.9 removed block pointers: pointer + mask in layer norm, preproc tail, attention forward, LN×U×dropout. Every load / store has its own int64 mask, so bits stay exact above 2 GB | fatal at step 1 |
| T8 | FlyDSL 0.2.4: API rename, rewriter rules, `known_block_size`; `_fast_call` launches (host time below v26.3); debug-location depth for compile time | FlyDSL LN / attention backward unavailable |
| T13 | Triton 3.9 parity settings (`TRITON_DEFAULT_FP_FUSION=1`, `TRITON_HIP_USE_ASYNC_COPY=0`) via bootstrap + gin | lost FMA contraction; slower pipelined kernels |
| T14 | L11 DO8 backward without spills under clang 24 (T14b: numerics consistent with Triton 3.9) | DO8 chain off: **−1.97%** |

### 3.3 Correctness on 10.2

| # | Issue | Fix |
|---|---|---|
| T17 | After each eval, `attach()` dropped an unfinished all_to_all. nccl2 keeps only its input alive, so RCCL wrote into reused memory: 27 of 27 eval-heavy runs crashed or hung | `attach()` completes torchrec's input dist; superseded collectives are stream-waited before release (no cost) |
| T18 | Trainer traces had 0 GPU events (rocprofiler-sdk env leaked into the ranks) | `train_ranker.py` unsets it for traced runs |
| T19 | torch 2.16's fp8 cast saturates where 2.12 gives NaN, failing the FP8 Indexed-X check | reference uses explicit e4m3 codes |
| T21 | Triton 3.9 sums `store_t` layer-norm stats in a different order, so the MX6 producer check fails | `store_t` reproduces 3.6's order bit for bit; 64-token blocks (MX6 worth ~2.5%) |
| T22 | Delta-backward diet was pinned to Triton 3.6 | ported to 3.9 with a runtime bitwise self-check |
| T25 | Triton 3.9's bundled LLVM emits a gfx950-invalid `v_pk_*_f32 op_sel` form. Live in dq8 (nondeterministic output); latent in two LN launches and FlyDSL `bwd_dkdv` | affected launches compile without packed FP32; FlyDSL operand order changed; 0 invalid forms in production binaries |
| T16 | `l11_outproj_lnmd_test` compares against hipBLASLt 10.2's dgrad | open: test issue, not a kernel issue |

### 3.4 Performance on 10.2

| # | Cause on 10.2 | Fix | Gain |
|---|---|---|---|
| T10 | hipBLASLt 1.5 picks a slower FP8 dgrad tile (`MT256x240x128`) | per-stack pin to solution 33058 (in-order K) through a hipBLASLt C-API executor | +0.25–0.5% |
| T15 | X2LB fused backward fails its bitwise check (K-rotated hipBLASLt dgrad) | in-order pin (T10) lets it arm | ~0 |
| T20 | Triton 3.9 codegen: EC kernels spill, rowloop / `_e8_fin` / maskgen slower | per-kernel code and geometry changes, gated to Triton 3.9 | +0.43% |
| T23 | nccl2 `getFuture()` blocks the backward stream on each DDP all-reduce; also breaks the per-GPU work balancing | DDP communication hook that defers the wait | +0.4% per step; most of the time-to-target gap |
| T24 | torch 2.16 / rocPRIM issue ~2.4× more memset launches (86 vs 36 per step) | fills folded into kernels; onesweep clears its state in-kernel | +0.2% |
| T4 | hipBLASLt 1.5 honours FP8 `scale_result`; the scale-mode probe would turn K512 / MX6 off | `input_scale` is the code default (probe opt-in) | keeps them armed |

### 3.5 One-time costs in time to target

| # | Cost on 10.2 | Fix |
|---|---|---|
| T11 | Triton spec replay not armed on 3.9: 67–73 compiles per rank in timed steps | replay supports 3.9; table re-recorded, incl. eval shapes → 0 |
| T12 | HIP preload table keyed to v26.3 builds (25 lazy loads) | per-stack table for TheRock's layout (0 failures) |
| T26 | hipBLASLt loads 10.2-only kernel libraries (up to 475 MB) at first use of new GEMM shapes: 7–9 s of stalls | per-stack replay table loads them before `run_start` |
| T27 | Pinned host pools grow during training and the first eval (~3 s) | pre-grown before `run_start` |
| T28 | T27 pins ~180 GiB per node; on a node with a near-full page cache the kernel then reclaims during training and ranks stall (cluster B) | before pinning, check FREE memory per NUMA node; if short, reclaim page cache (`host_mem_reclaim.py`, unprivileged, ~1 min, untimed) and re-check; if still short, stop before `run_start`. gin item (121) |
| T29 | numpy `madvise(MADV_HUGEPAGE)`s large arrays: ~1 M 2 MiB page faults per run in the data path. Under THP `madvise` / defrag `madvise` (cluster B) each fault compacts synchronously once memory is fragmented: all-rank stall windows (up to 41 of 71 windows > 2×), time to target up to +15%, worse with node uptime and back-to-back runs | `NUMPY_MADVISE_HUGEPAGE=0` (`run_and_time.sh`, Dockerfile): 0 THP faults, 0 compaction stalls, 0 stall windows in 3 back-to-back full runs; 200-step throughput unchanged (−0.06%) |

## 4. Open items

- Independent re-verification of tree v6; a full RCP seed set (3 seeds per scale so far).
- `launch_slurm.sh` run on the 10.2 image (L4). Docker is validated at 1 node on cluster B (I8, L7).
- Multi-node on cluster B (2 and 4 nodes, with T28): queued.
- T28 headroom (512 GiB per NUMA node) is set from 1-node runs; revisit if a stall shows up with the check passing.
- The Triton 3.9 spec rows were recorded at 8 ranks; at 16 / 32 ranks the replay fills the rank-count-dependent
  kernels from translated 3.6 rows (0 errors). Re-record at 16 and 32 ranks if a compile shows up after `run_start`.
- Test owners: T16; `preproc_l2_x2lb_test` expectations; `mx6_fwd_callsite` SPF grid; `preproc_dedup_cpu`
  (torch 2.16); `stage_ref_only_hardening`.
- T5: extension build directories are not keyed by stack (the runner uses a fresh `/tmp` per run instead).
- Small: 3 FBGEMM bundles load at step 1 (~89 ms); eval-heavy configs still compile 10–24 Triton kernels per rank.
- Decisions: the `outproj_mx6_seam_test` re-pin, and the zero-fill hunk left out of `triton_hstu_linear.py`.

## 5. Upstream issues

Not yet filed.

| # | Project | Issue | Evidence / stake |
|---|---|---|---|
| U2 | ROCm LLVM (clang 24) | `l11_outproj_lnmd_bwd_do8` spills 8 VGPRs (none on clang 22) | −1.97% until worked around (T14) |
| U5 | PyTorch c10d nccl2; torchrec | nccl2 frees a dropped `Work`'s output while RCCL still writes; torchrec `attach()` never waits its input dist | memory corruption; 27/27 crash or hang (T17) |
| U6 | rocprofiler-sdk; kineto | SDK exports `ROCPROFILER_REGISTER_LIBRARY` to child processes; kineto silently records no GPU events | 0 GPU events in traces (T18) |
| U8 | Triton 3.9 (bundled LLVM) | gfx950-invalid `v_pk_*_f32 op_sel`; fixed upstream in llvm/llvm-project#232016 (+#233970) | nondeterministic output; minimal repro ready (T25) |
| U9 | FlyDSL 0.2.4 | same encoding; no option to disable packed FP32; first-call compile 3.2× slower | latent (T25, T8) |
| U10 | PyTorch c10d nccl2 | `WorkNCCL::getFuture()` blocks the caller's stream, so DDP bucket all-reduce no longer overlaps backward | +0.4% / step and ~2% time to target (T23) |
| U11 | hipBLASLt 1.5; torch 2.16 TunableOp | heuristic picks a slower FP8 tile; TunableOp cannot run FP8 ScaledGemm | +0.25–0.5% (T10) |
| U12 | rocPRIM 4.8 | onesweep sort issues 17 memsets per 64-bit sort | 20% sort time (T24) |
| U13 | hipBLASLt 1.5 | lazy loading of large kernel libraries at first use | 7–9 s per run (T26) |
| U1 | ROCm LLVM | resource remarks lost the `remark:` form | handled by T2 |
| U3 | TheRock image | two `libamdhip64.so.7` copies, no `/opt/rocm`, `CU_NUM=304` baked in | handled by I3, I6 |

## 6. Reproduce

```bash
# image (login node; anonymous pull)
DEF=scripts/cluster_a/dlrmv4_utd_rocm10.2.def SIF=$SCRATCH/apptainer/dlrmv4_utd_rocm10.2.sif bash scripts/cluster_a/run_dlrmv4.sh build

# same-node A/B vs v26.3 (unmodified d14a7cb tree); SHAPE=full and SEED=<n> for time to target / convergence
SIF_A=$SCRATCH/apptainer/dlrmv4.sif REC_A=<d14a7cb tree> \
SIF_N=$SCRATCH/apptainer/dlrmv4_utd_rocm10.2.sif ROCM_BIND_N=0 \
ENV_N="FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL=1 GPU_USE_DEVICE_QUEUE=0 DEBUG_CLR_AQL_DEV_QUEUE=0" \
SHAPE=200step ORDER="A N N A" bash scripts/cluster_a/run_dlrmv4.sh submit -t 01:00:00
bash scripts/cluster_a/run_dlrmv4.sh compare $SCRATCH/logs/ab_200step_<job>

# multi-node convergence A/B: same variables, MLPerf RCP rules for the node count (2 or 4)
SHAPE=rcp SEED=<n> ORDER="A N" bash scripts/cluster_a/run_dlrmv4.sh submit -N 2 -t 01:45:00

# cluster B (Docker): same-node 1-node A/B; images are loaded/built on the node
sbatch -p <partition> -N1 --gpus-per-node=8 -t 02:00:00 scripts/cluster_b/ab_r10_1node.sbatch shape=200step "order=A N N A A N"
sbatch -p <partition> -N1 --gpus-per-node=8 -t 02:30:00 scripts/cluster_b/ab_r10_1node.sbatch shape=full seed=<n> "order=N A"
```
