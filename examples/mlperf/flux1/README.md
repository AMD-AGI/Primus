# FLUX.1-Schnell FP8 MLPerf Training

This recipe runs FLUX.1-Schnell on MI355X GPUs with the in-tree `diffusion`
backend. The defaults enable tensorwise FP8 and target the MLPerf validation
loss threshold of `0.586`.

## Docker image

The launch scripts use `zirui3/primus-v26.3-flux:v0.4` by default. This image
contains the selective FlyDSL and natural-backward optimizations from the
[`Dockerfile`](Dockerfile) in this directory.

```bash
docker pull zirui3/primus-v26.3-flux:v0.4
```

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

Each `config_*.sh` sources `config_common.sh` and then exports its qualified
shape-specific settings:

| `FLUX_CONFIG` | Nodes | MBS | GA | GBS | FP8 GEMM | Compile mode |
|---|---:|---:|---:|---:|---|---|
| `config_1n_gbs512.sh` | 1 | 64 | 1 | 512 | TorchAO/Inductor | `max-autotune-no-cudagraphs` |
| `config_1n_gbs1024.sh` | 1 | 32 | 4 | 1024 | selective FlyDSL | default |
| `config_4n_gbs1024.sh` (default) | 4 | 32 | 1 | 1024 | selective FlyDSL | default |

The MBS64 profile retains checkpoint ratio `0.25`. Both MBS32 profiles use
ratio `0`, forward-input FP8 reuse, the qualified natural-layout wgrads, and
MBS32 Inductor fusion benchmarking.

Inside an allocation, select a configuration and use the same launcher:

```bash
FLUX_CONFIG=config_1n_gbs512.sh \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/output \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

Use `config_1n_gbs1024.sh` for single-node MBS32 kernel development. The
4-node MLPerf target is the default, so it needs no `FLUX_CONFIG` override:

```bash
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/output \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

For a short smoke test, append `MAX_STEPS=1 SAVE_STRATEGY=none` before `bash`.

## Submit the default four-node target

```bash
REPO=/shared_nfs/zirui/code/primus-compile
DATA_ROOT=/shared_nfs/zirui/data
OUTPUT_ROOT=/shared_nfs/zirui/runs/flux-fp8-4n

sbatch -A amd-spur -p amd-spur --qos=amd-spur-qos \
  -N4 --ntasks-per-node=1 --exclusive --gpus-per-node=8 -t 04:00:00 \
  --output="$OUTPUT_ROOT-%j.slurm.log" \
  --wrap="cd '$REPO' && DATA_ROOT='$DATA_ROOT' \
    OUTPUT_ROOT='$OUTPUT_ROOT'-\$SLURM_JOB_ID \
    bash examples/mlperf/flux1/run_with_docker_slurm.sh"
```

Add `--nodelist=node1,node2,node3,node4` only when specific idle nodes are
required. For a holder allocation entered externally with `spur run --jobid`,
follow [`luanch-multi-nodes.md`](luanch-multi-nodes.md); nested `srun` does not
fan out.

Common overrides include `MAX_STEPS`, `SEED`, `MASTER_PORT`, `SAVE_STRATEGY`,
`SAVE_STEPS`, `RESUME_FROM_CHECKPOINT`, and `MLPERF_CLEAR_CACHES=false`.

## Files

```text
examples/mlperf/flux1/
├── Dockerfile
├── README.md
├── config_common.sh
├── config_1n_gbs512.sh
├── config_1n_gbs1024.sh
├── config_4n_gbs1024.sh
├── flux.1_schnell_t2i-pretrain.yaml
├── luanch-multi-nodes.md
├── requirements.txt
├── run_with_docker.sh
└── run_with_docker_slurm.sh
```
