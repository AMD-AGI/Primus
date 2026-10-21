# FLUX.1-Schnell A6W4 MLPerf Training

This recipe runs FLUX.1-Schnell on MI355X GPUs with the in-tree `diffusion`
backend. The default four-node path uses A6W4 forward and MXFP4 backward,
then switches the forward to precompiled FP8 after update 7325. It targets the
MLPerf validation loss threshold of `0.586`. The low-precision paths remain
opt-in at runtime; importing Primus without their environment flags preserves
the native BF16/FP8 behavior.

## Docker image

The launch scripts reuse the proven image by immutable digest (no rebuild):

```text
zirui3/primus-v26.7-flux:v0.1
sha256:082afc328d867f79ab5c4a46bc29c909bd891e068a32506a10768e6a6cc7e7c8
```

```bash
docker pull zirui3/primus-v26.7-flux:v0.1@sha256:082afc328d867f79ab5c4a46bc29c909bd891e068a32506a10768e6a6cc7e7c8
```

The launcher bind-mounts this checkout at `/workspace/Primus`; the A6W4
integration and forward-switch hooks are normal in-tree Primus source. The
image already packages Primus-Turbo revision
`e031c5bb3746e8b667676bd20e2697d75fe9f2b2`. DCCS and Crusoe use this same
digest; `FLUX_CLUSTER` changes runtime networking only, never the image.

## Prepare data

Approximately 2.5 TB of storage is required. Download the preprocessed MLPerf
TorchTitan datasets:

```bash
mkdir -p /path/to/data
cd /path/to/data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri
```

The resulting data root must contain:

```text
/path/to/data/
├── cc12m_preprocessed/
├── coco_preprocessed/
└── empty_encodings/
```

See the [MLPerf FLUX preprocessing instructions](https://github.com/mlcommons/training/tree/master/text_to_image#preprocessing)
for details.

## Select a configuration

Each `config_*.sh` combines `config_common.sh` with its qualified
shape-specific settings. The four-node default selects the A6W4 recipe block;
the existing slim profiles retain their prior FP8 defaults:

| `FLUX_CONFIG` | Nodes | MBS | GA | GBS | FP8 GEMM | Compile mode |
|---|---:|---:|---:|---:|---|---|
| `config_1n_gbs512.sh` | 1 | 64 | 1 | 512 | TorchAO/Inductor | `max-autotune-no-cudagraphs` |
| `config_1n_gbs1024.sh` | 1 | 32 | 4 | 1024 | selective FlyDSL | default |
| `config_2n_gbs1024.sh` | 2 | 32 | 2 | 1024 | selective FlyDSL | default |
| `config_4n_gbs1024.sh` (default) | 4 | 32 | 1 | 1024 | A6W4/MXFP4, then FP8 forward | `max-autotune-no-cudagraphs` |

The default profile uses 32-rank pure FSDP (`DP_REPLICATE=1`), LR `0.00025`,
1600 warmup steps, SDMA only for `single_blocks`, the C++ Inductor wrapper,
Kyle's FC1/GELU dual and double-pack candidate, and forward-switch precompile
at step 7325. The MBS64 profile
retains checkpoint ratio `0.25`.

Inside an allocation, select a configuration and use the same launcher. The
launcher invokes the diffusion backend directly with `torchrun`; it does not
use the Primus CLI or runtime. Set `LAUNCH_MODE=primus` to run the same profile
through `primus-cli` for integration validation.

```bash
FLUX_CONFIG=config_1n_gbs512.sh \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/output \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

Use `config_1n_gbs1024.sh` for single-node MBS32 kernel development. The
4-node A6W4 MLPerf target is the default, so it needs no `FLUX_CONFIG` override:

```bash
FLUX_CLUSTER=crusoe \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/output \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

For a short smoke test, append `MAX_STEPS=1 SAVE_STRATEGY=none` before `bash`.

## Submit the default four-node target on Crusoe

```bash
REPO=/shared_nfs/zirui/code/Primus
DATA_ROOT=/shared_nfs/zirui/data
OUTPUT_ROOT=/shared_nfs/zirui/runs/flux-a6w4-4n

sbatch -A amd-spur -p amd-spur --qos=amd-spur-qos \
  -N4 --ntasks-per-node=1 --exclusive --gpus-per-node=8 -t 04:00:00 \
  --output="$OUTPUT_ROOT-%j.slurm.log" \
  --wrap="cd '$REPO' && FLUX_CLUSTER=crusoe DATA_ROOT='$DATA_ROOT' \
    OUTPUT_ROOT='$OUTPUT_ROOT'-\$SLURM_JOB_ID \
    bash examples/mlperf/flux1/run_with_docker_slurm.sh"
```

Add `--nodelist=node1,node2,node3,node4` only when specific idle nodes are
required.

Common overrides include `MAX_STEPS`, `SEED`, `MASTER_PORT`, `SAVE_STRATEGY`,
`SAVE_STEPS`, `RESUME_FROM_CHECKPOINT`, and `MLPERF_CLEAR_CACHES=false`.
Evaluation throughput can be tuned independently with `EVAL_BATCH_SIZE`,
`EVAL_DATALOADER_NUM_WORKERS`, and `EVAL_DATALOADER_PREFETCH_FACTOR`; their
defaults are the training local batch size, the training worker count (`4`),
and `4`. These knobs do not change the required full 29,696-sample evaluation
or its 262,144-training-sample cadence.
## DCCS and Crusoe networking

Set one site profile on every launch. Explicit environment overrides still win.

| Variable | `FLUX_CLUSTER=dccs` | `FLUX_CLUSTER=crusoe` |
|---|---|---|
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | `fenic` | `ens3` |
| `NCCL_NET_PLUGIN` | `none` | `librccl-anp.so` |
| `NCCL_RMA_DISABLE` | `0` | `1` |
| `NCCL_IB_HCA` | `ionic_0,...,ionic_7` | `ionic_1,...,ionic_7` |

Both profiles retain DMA-BUF, `NCCL_CROSS_NIC=2`, symmetric GIN disabled, and
one hierarchical CE context for the SDMA path. On the pinned image the Crusoe
plugin name is intentionally unavailable, which makes RCCL use regular built-in
IB; `NCCL_NET_PLUGIN=none` selects IB-CAST there and fails with Ionic local
access errors. Crusoe excludes rail 0 because it is degraded on a qualified
node pair; override `NCCL_IB_HCA` to restore it only after validating the
selected nodes.

DCCS launch inside an existing allocation:

```bash
FLUX_CLUSTER=dccs \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/output \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

The launcher uses the host's generic `/usr/lib/x86_64-linux-gnu/libionic.so`
symlink by default. Set `LIBIONIC_ABI4_PATH` only when libionic is installed
elsewhere, or set `NCCL_IB_DISABLE=1` when the selected nodes have no AINIC
routes.

## Files

```text
examples/mlperf/flux1/
├── Dockerfile
├── README.md
├── config_common.sh
├── config_1n_gbs512.sh
├── config_1n_gbs1024.sh
├── config_2n_gbs1024.sh
├── config_4n_gbs1024.sh
├── flux.1_schnell_t2i-native.yaml
├── flux.1_schnell_t2i-pretrain.yaml  # Primus CLI alternative
├── requirements.txt
├── run_with_docker.sh
└── run_with_docker_slurm.sh
```
