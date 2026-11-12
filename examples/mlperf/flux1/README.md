# FLUX.1-Schnell MLPerf training

The in-tree `diffusion` backend runs FLUX.1-Schnell with A6W4 forward,
MXFP4 backward and FSDP on MI355X. The default single-node profile follows the
[AMD recipe](https://github.com/AMD-AGI/mlperf-training/tree/4ae78b35cd5e4a51fbe5a0a4baa5e2b40ae0b597/flux1/primus):
GBS256, MBS32, GA1, LR `0.00025`, 800 warmup updates and no precision switch.
Low-precision paths are opt-in; ordinary BF16/FP8 training remains available.

## Runtime image

The launcher defaults to the verified AMD runtime by immutable digest:

```bash
docker pull rocm/mlperf@sha256:f4f4df148369ee52ccc79eb699913d321a6a532ad03dc03a6bc951593d503e16
```

Its build history matches the committed AMD Dockerfile: Primus-Turbo
`6a4209844cf660042ca133e1662a6fbc519f6c48`, RCCL
`d75507948fa94106ea2adf5c158af31e6fbda6ef` and checksum-verified ROCr.
The image carries a Training 6.1 tag; that label alone does not establish MLPerf
6.0 compliance or performance parity. Use the same digest on every node and
record any `DOCKER_IMAGE` override. Registry access is required to pull this
image. If it was transferred with `docker save`/`load`, verify the local image ID
`sha256:f0a1f5c0641ce55d37ab6ce35ef02f859ed6e3b6573b222dbc701d9715a6e482`
and pass it as `DOCKER_IMAGE`; loading an archive may not retain RepoDigest
metadata. The launcher mounts this Primus checkout at
`/workspace/Primus`, replacing only the packaged Primus source. Linux 6.8+
is required for RCCL-SDMA. `LAUNCH_MODE=primus` (default) enables the CLI's
cuMem, allocator and interposer prerequisites; `native` is available for direct
`torchrun` development but does not execute those hooks.

For **v26.9 single-node training**, use the source-only layer in
`Dockerfile.v26.9` and `config_1n_gbs256_a6w4_v26_9.sh`. Paired throughput
qualification passed without rebuilding the image's native dependencies. See
[V26_9_VALIDATION.md](V26_9_VALIDATION.md) for build/launch commands, the precise
Turbo source/native composition, results and scope. Full convergence and
multi-node v26.9 performance are not qualified by these short runs.

## Data

Allow approximately 2.5 TB for the preprocessed MLPerf datasets:

```bash
mkdir -p /path/to/data && cd /path/to/data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri
```

`DATA_ROOT` must contain `cc12m_preprocessed/`, `coco_preprocessed/` and
`empty_encodings/`. See the [MLPerf preprocessing instructions](https://github.com/mlcommons/training/tree/master/text_to_image#preprocessing).

## Profiles and launch

Run from the Primus repository root inside an allocation owning all GPUs.
Each node launches one process, which creates eight workers with `torchrun`.

| `FLUX_CONFIG` | Nodes | MBS | GA | GBS | Path |
|---|---:|---:|---:|---:|---|
| `config_1n_gbs256_a6w4.sh` (default) | 1 | 32 | 1 | 256 | AMD single-node A6W4 schedule |
| `config_1n_gbs256_a6w4_v26_9.sh` | 1 | 32 | 1 | 256 | Same schedule, v26.9 communication settings |
| `config_2n_gbs1024_a6w4.sh` | 2 | 32 | 2 | 1024 | Four-node schedule adapted to two nodes |
| `config_4n_gbs1024.sh` | 4 | 32 | 1 | 1024 | A6W4 with FP8-forward switch |
| `config_1n_gbs512.sh` | 1 | 64 | 1 | 512 | Legacy TorchAO/Inductor FP8 |
| `config_1n_gbs1024.sh` | 1 | 32 | 4 | 1024 | Legacy selective FlyDSL FP8 |
| `config_2n_gbs1024.sh` | 2 | 32 | 2 | 1024 | Legacy selective FlyDSL FP8/HSDP |

Single-node Crusoe launch:

```bash
FLUX_CLUSTER=crusoe \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/unique-output \
bash examples/mlperf/flux1/run_with_docker.sh
```

For DCCS, use `FLUX_CLUSTER=dccs`. Inside a two-node Slurm allocation:

```bash
FLUX_CLUSTER=dccs \
FLUX_CONFIG=config_2n_gbs1024_a6w4.sh \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/unique-output \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

The two-/four-node adaptations keep LR `0.00025`, 1600 warmup updates and
precompile the FP8-forward switch after update 7325. They are not an official
AMD two-node submission schedule. The single-node profile leaves the four-node
C++ wrapper and additional pack/FC1-GELU extensions off to match AMD's baseline;
those knobs and the existing four-node defaults remain available.

For a smoke test, prefix `MAX_STEPS=1 SAVE_STRATEGY=none MLPERF_CLEAR_CACHES=false`.
Use `FSDP_ALL_GATHER_BACKEND=default` as an SDMA-only control. Common overrides
include `SEED`, `LOG_FREQ`, `MASTER_PORT`, `SAVE_STRATEGY`,
`RESUME_FROM_CHECKPOINT`, `EVAL_DATALOADER_NUM_WORKERS` and
`EVAL_DATALOADER_PREFETCH_FACTOR`. Mixed-precision debug overrides on legacy
profiles require matching `MLLOG_LOWEST_NUMERICAL_PRECISION_IN_*` disclosures;
select an A6W4 profile for its configured MXFP4 metadata. `TORCHINDUCTOR_CACHE_SEED` accepts a matching
prewarmed cache directory/archive and copies it to container-local storage;
do not share a writable Inductor cache across ranks on NFS.

## Networking and measurement

| Setting | DCCS | Crusoe |
|---|---|---|
| Socket interface | `fenic` | `ens3` |
| `NCCL_NET_PLUGIN` | `none` | `librccl-anp.so` |
| `NCCL_RMA_DISABLE` | `0` | `1` |
| `NCCL_IB_HCA` | `ionic_0,...,ionic_7` | `ionic_1,...,ionic_7` |

Explicit environment overrides win. Validate every selected rail before changing
`NCCL_IB_HCA`; link-up alone does not prove cross-node GDR integrity. Crusoe's
unavailable plugin name selects built-in IB rather than the image's IB-CAST
path. Multi-node launch requires `/dev/infiniband` and the host libionic symlink;
set `LIBIONIC_ABI4_PATH` for a nonstandard installation.

Full MLPerf validation consumes all 29,696 samples every 262,144 training samples
and targets loss `0.586`. Compare complete `run_start` to `run_stop` with
`status=success`, including validation. Keep the node(s), source, image, seed,
schedule and evaluation fixed, and record all runtime changes. Short throughput
windows and interrupted runs are not TTQ evidence; one paired run is not an
official multi-seed MLPerf result. Resumed-session elapsed/throughput excludes
prior training and is not complete fresh-start TTQ.
