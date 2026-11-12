# FLUX v26.9 single-node performance qualification

**Single-node throughput parity passed on 8 × MI355X.** The corrected v26.9
composition is about **2.27% faster** than the v26.7-derived AMD FLUX control
in both forward-order and reverse-order comparisons. This qualifies the specific
image + Primus source + recipe below, not full MLPerf convergence or multi-node
performance. The default launcher image remains the previous AMD runtime;
select the v26.9 image and profile explicitly.

## Reproduce the composition

From the Primus repository root, build a small source-update layer:

```bash
docker build --network=host \
  -f examples/mlperf/flux1/Dockerfile.v26.9 \
  -t primus-flux:v26.9-main-25fb2df6 .
```

Host networking was needed for GitHub DNS on the test cluster; omit it where
normal Docker build networking works. The Dockerfile pins the official v26.9
base digest and Turbo main commit. It copies tracked Turbo Python/FlyDSL files,
checks their hashes, and retains the image's compiled extensions. **No pip/apt
installation or native library compilation is performed.**

Run inside an allocation owning all eight GPUs, using a new output directory:

```bash
DOCKER_IMAGE=primus-flux:v26.9-main-25fb2df6 \
FLUX_CONFIG=config_1n_gbs256_a6w4_v26_9.sh \
FLUX_CLUSTER=crusoe DOCKER_PRIVILEGED=0 \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/unique-output \
PRIMUS_SKIP_PIP=1 PRIMUS_AUTO_INSTALL=0 PRIMUS_AUTO_DEPS_SYNC=0 \
MLPERF_CLEAR_CACHES=false MAX_STEPS=1024 LOG_FREQ=10 \
bash examples/mlperf/flux1/run_with_docker.sh
```

`DATA_ROOT` contains `cc12m_preprocessed/`, `coco_preprocessed/` and
`empty_encodings/`. `DOCKER_PRIVILEGED=0` uses the launcher's explicit device and
capability grants; the old privileged mode remains available. Cache clearing is
disabled for this performance test, not claimed as an official MLPerf cache-clear
run. Do not run training concurrently on the same GPUs.

The new profile keeps MBS32, GA1, GBS256, LR0.00025, warmup800, seed10007,
A6W4/MXFP4, single-block SDMA and K-major copy-out. Its runtime changes are:

```text
TORCH_DIST_USE_NCCL2=1
GPU_MAX_HW_QUEUES=4
CUDA_DEVICE_MAX_CONNECTIONS=2
FLUX_FSDP_RS_STREAM=normal
```

The image exports `TORCH_DIST_USE_NCCL2=0`. The host launcher forwards the profile's
value, but direct `docker exec` must explicitly override it with `-e
TORCH_DIST_USE_NCCL2=1`. Profile overrides are respected. Use
`FLUX_FSDP_RS_STREAM=default` to retain the existing stream for diagnostics.
Ordinary runtime JIT and the existing small gcc SDMA interposer remain required;
this is not a zero-compilation runtime.

## Dependency identity

| Component | v26.9 composition | v26.7-derived AMD FLUX control |
|---|---|---|
| Base image digest | `rocm/primus@sha256:921604bee3a7289e95f37db5a4d990dc530f64bdbcabd80efd952aef98647d27` | `rocm/mlperf@sha256:f4f4df148369ee52ccc79eb699913d321a6a532ad03dc03a6bc951593d503e16` |
| Torch | `2.16.0a0+rocm10.2.0a20261020` | `2.12.0+rocm10.0.0` |
| Triton | `3.9.0+gitfc8d31a5.rocm10.2.0a20261020` | `3.8.0+git4cff872c.rocm10.0.0` |
| FlyDSL | `0.3.4.1` | `0.2.4` |
| AITER | `0.1.27` | `0.1.14.post1` |
| Turbo Python/FlyDSL source | `25fb2df6b2ee8bd14e9a93522c54dbc60b1903ef` (upstream main, PR633) | `6a4209844cf660042ca133e1662a6fbc519f6c48` |
| Turbo native build | `de8fa6ca7f4986334070d5ee62beee07552f8ad8` | `6a4209844cf660042ca133e1662a6fbc519f6c48` |
| Turbo distribution metadata | `0.6.1.dev4` | `0.4.1.dev51` |

The control is **not bare `rocm/primus:v26.7`**. Both arms use the same Primus
source based on `b438b0d5c12cddf6835842f3d67debbdffa26c8b` plus this change,
but runtime-specific communication settings: the control retains queues2 /
connections1 / the default RS stream. All model, optimizer, precision, data and
evaluation settings remain fixed.

This is an explicit **FLUX-only mixed-provenance source overlay**, not a native
Turbo build from main. Main's newer DeepEP Config properties are not in the image
extension; arbitrary MoE/DeepEP/full-main compatibility is not qualified. Do not
identify it using `pip freeze` or `_build_info.py` alone, and do not run a package
upgrade or requirements sync over the prepared runtime.

The image stores `/opt/flux-base-packages.txt` and
`/opt/flux-turbo-source-manifest.json`, including all 332 source-file hashes and
the two preserved native-library hashes:

- `lib/libprimus_turbo_kernels.so`:
  `07451a8fc6a3712315a09bb20520f0b09491b0b12081510967731ea1c37d7289`
- `pytorch/_C.cpython-312-x86_64-linux-gnu.so`:
  `780d82576afc13af27edffcb7b3849c1c82ca152ff818ed2270f8e1937ee9ae1`

The built layer's manifest matches the tested candidate exactly. Build-time
source checks pass without GPUs; importing the complete Turbo PyTorch package
requires a visible GPU. The actual launcher was additionally tested for120steps
in the built layer, with no privileged mode or dependency install, and exited0.

## Root cause and minimal fix

The initial v26.9 migration ran but lost about16.2% throughput. The reference
Kailash source contained additional communication adaptations that were not in
the b438-based Primus candidate. Its scoped Turbo FLUX kernels were already in
main; no kernel rollback or native rebuild was needed.

Sequential120-step ablations, averaging step20–120 timing windows:

| v26.9 runtime | Forward ms | Backward ms | Total ms/step |
|---|---:|---:|---:|
| queues2/connections1, default RS, buffer cap1 | 161.72 | 239.41 | 415.39 |
| same, buffer cap32 | 162.89 | 237.12 | 414.28 |
| queues4/connections2, default RS, cap1 | 124.10 | 239.37 | 377.71 |
| queues4/connections2, normal RS, cap1 | 123.91 | 199.28 | 337.58 |
| same, cap2 | 124.69 | 198.87 | 337.88 |
| same, cap32 | 123.89 | 199.15 | 337.38 |
| v26.7 control | 132.42 | 198.57 | 345.18 |

This isolates a queue/stream overlap regression, not simply slower compute
kernels. The final code installs the normal-priority RS stream once, after FSDP
initializes its shared communication context and before backward. Buffer
expansion showed no useful gain, so its experimental code was removed. Both
measured images expose the cap API; it is incorrect to attribute the regression
solely to a new cap in v26.9.

The other compatibility fix accepts PyTorch's optional `all_gather_output_fn`
argument in K-major copy-out and forwards it on the ordinary fallback path.
The existing SDMA path remains enabled. No HIP-IPC AllGather implementation,
new fusion kernels, precision changes or gradient-accumulation changes were
needed. Kailash's GA2/GBS512 reference supplied the optimization clue, but was
not substituted for this GA1/GBS256 acceptance recipe.

## Paired qualification

Tested2026-11-12 on the same Crusoe MI355X node, sequentially in both orders.
Each fresh run completed1024updates, all eight ranks returned normally, and full
validation consumed29696samples. Train-only timing synchronizes all ranks after
updates800 and1000; throughput below uses the **slowest rank**, not selected log
lines or an average of instantaneous rates. Compiler caches were separate per run.

| Order / arm | Steady seconds | Samples/GPU/s | 8-GPU samples/s | Full validation loss |
|---|---:|---:|---:|---:|
| First: v26.9 fixed | 67.811652 | 94.37906 | 755.032 | 0.655733 |
| First: v26.7 control | 69.351530 | 92.28347 | 738.268 | 0.655140 |
| Reverse: v26.7 control | 69.363911 | 92.26700 | 738.136 | 0.654380 |
| Reverse: v26.9 fixed | 67.823240 | 94.36294 | 754.903 | 0.654610 |

Throughput improves **2.271% / 2.272%** in the two pairs. Validation-inclusive
cumulative throughput at1024 is **725.140 / 724.652** samples/s for v26.9,
versus **702.129 / 700.447** for the respective controls. Validation-loss deltas
are+0.000593 and+0.000230; these are short-run observations, not multi-seed
numerical or convergence qualification. No nonfinite losses or runtime failures
were found. Maximum allocated memory remains about121.50GB (control121.65GB);
maximum reserved memory is243.42GB (control156.65GB), so allocator reservation
is not identical across versions.

The two relevant CPU test files pass on both images: **128 passed, 1 GPU test
skipped**. Tests cover callback compatibility, profile invariants/overrides,
FSDP lazy-init/one-shot stream ordering, and launcher dependency/privilege options.

Capped runs intentionally record MLPerf `run_stop status="aborted"` because the
0.586 target was not reached. This is **not full convergence/TTQ qualification**,
and no multi-node performance or official submission claim is made.

Internal evidence bundle: `/shared_nfs/zirui/runs/flux-v26.9-1n-20261112`.
Key receipts: `audit/fixed-qualification-summary.jsonl`, the four `*-fixed-*`
run directories, `audit/rs-ablation-summary.jsonl`,
`audit/release-manifest-equivalence.log`, `release-launcher-smoke-01/`, and
`audit/final-cpu-v{269,267}.log`. Image publication and source commit/PR update
are separate release operations, not performed by this qualification.
