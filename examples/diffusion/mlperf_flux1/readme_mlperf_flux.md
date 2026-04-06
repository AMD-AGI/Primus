# MLPerf FLUX.1 (text-to-image) — Primus data prep and training

This folder automates the **MLPerf Training v5.1 FLUX.1** dataset flow for Primus Megatron-Bridge pretrain. Upstream reference: [MLCommons `text_to_image` README](https://github.com/mlcommons/training/blob/master/text_to_image/README.md).

## Contents

| File | Role |
|------|------|
| `setup_mlperf_flux1_automated.sh` | One-shot: pip deps → download → WDS export → `prepare_energon_dataset_flux.py` → `energon prepare` |
| `download_mlperf_flux1_datasets.sh` | MLCommons R2 download only (`minimal` / `preprocessed` / `all`) |
| `export_wds_to_flux_prepare_folder.py` | WebDataset `.tar` shards → flat images + `.txt` captions |
| `requirements-mlperf-flux1-setup.txt` | Python packages for setup (export, webdataset, etc.) |

`prepare_energon_dataset_flux.py` lives under `examples/diffusion/recipes/flux/` and is invoked by the automated setup.

## Quick start (recommended)

Run from **any directory** (the setup script changes into the Primus repo root automatically):

```bash
export MLPERF_FLUX1_ROOT=/data/mlperf_flux1   # optional; default in script is /data/mlperf_flux1
bash examples/diffusion/mlperf_flux1/setup_mlperf_flux1_automated.sh minimal
```

Then point training at the prepared Energon dataset and launch pretrain from the **Primus repo root**:

```bash
source "${MLPERF_FLUX1_ROOT}/mlperf_flux1_energon.env"
# or: export MLPERF_FLUX1_ENERGON="${MLPERF_FLUX1_ROOT}/cc12m_flux_wds"

./primus-cli direct -- train pretrain \
  --config examples/megatron_bridge/configs/MI300X/flux_12b_pretrain_mlperf_flux1.yaml
```

The `@Primus/examples/diffusion/mlperf_flux1/setup_mlperf_flux1_automated.sh` suffix attaches this setup script to the YAML in the Primus CLI config resolution path you use.

**Hugging Face:** if `black-forest-labs/FLUX.1-dev` requires authentication for prepare or train, run `huggingface-cli login` (or set `HF_TOKEN`) before the steps that load the model.

## Download phases

| Phase | What it pulls |
|-------|----------------|
| `minimal` | CC12M-on-disk training subset + COCO-2014 val subset + `val2014_30k.tsv` (usual starting point) |
| `preprocessed` | Large precomputed packs (~2.5 TB total); TorchTitan-style reference workflow |
| `all` | `minimal` then `preprocessed` |

Example download-only (from Primus root, or any cwd — script uses `MLPERF_FLUX1_ROOT` and `cd`s into it):

```bash
export MLPERF_FLUX1_ROOT=/data/mlperf_flux1
bash examples/diffusion/mlperf_flux1/download_mlperf_flux1_datasets.sh minimal
```

## Typical layout under `MLPERF_FLUX1_ROOT`

After MLCommons download (`minimal`):

- `cc12m_disk/` — training WebDataset shards  
- `coco/` — validation WebDataset shards  
- `val2014_30k.tsv` — eval protocol TSV  

After automated setup:

- `cc12m_for_prepare/` — flat images + captions (export step)  
- `cc12m_flux_wds/` — FLUX-style Energon WDS + metadata after `prepare_energon_dataset_flux.py` and `energon prepare`  
- `mlperf_flux1_energon.env` — `source` this to set `MLPERF_FLUX1_ROOT` and `MLPERF_FLUX1_ENERGON`

Training should use the **Energon** tree (`cc12m_flux_wds`), not `cc12m_disk` or `cc12m_for_prepare` directly.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `MLPERF_FLUX1_ROOT` | Dataset root (default `/data/mlperf_flux1` in automated setup; `./mlperf_flux1_data` if unset in download-only script) |
| `MLPERF_FLUX1_INPUT_GLOB` | Override glob for export, e.g. `'/data/mlperf_flux1/**/*.tar'` |
| `MLPERF_FLUX1_ENERGON` | Path passed to training config (usually `.../cc12m_flux_wds`) |
| `FORCE_REDOWNLOAD` | Set to `1` to re-run MLCommons downloads even if data already looks present |

## Skipping and forcing steps

**Automated setup** (`setup_mlperf_flux1_automated.sh`):

- `SKIP_PIP`, `SKIP_DOWNLOAD`, `SKIP_EXPORT_WDS`, `SKIP_PREPARE_FLUX`, `SKIP_ENERGON_PREPARE` — any non-empty value skips that stage entirely.  
- `FORCE_EXPORT_WDS`, `FORCE_PREPARE_FLUX`, `FORCE_ENERGON_PREPARE` — force re-run even when outputs look complete.  
- `MLPERF_FLUX1_EXPORT_MIN_PAIRS` — minimum `*.txt` files in `cc12m_for_prepare` to treat export as done (default `100`).

**Download script** (`download_mlperf_flux1_datasets.sh`):

- Skips `minimal` downloads when `val2014_30k.tsv` exists and both `cc12m_disk/` and `coco/` contain WebDataset `.tar` shards.  
- Skips `preprocessed` downloads when `cc12m_preprocessed/`, `coco_preprocessed/`, and `empty_encodings/` each exist and contain files.  
- `FORCE_REDOWNLOAD=1` disables those skips.

## Manual pieces (if not using full automation)

See the header comments in `download_mlperf_flux1_datasets.sh` for exact `python` / `energon prepare` one-liners, or run the individual stages with the same paths as in `setup_mlperf_flux1_automated.sh`.

## Config and parallelism

- YAML: `examples/megatron_bridge/configs/MI300X/flux_12b_pretrain_mlperf_flux1.yaml`  
- Defaults target **single node, 8 GPUs** (data parallel). If you OOM or hit FP8/Transformer Engine issues, adjust tensor parallelism or `precision_config` in that file (see comments there).

## Alternative: `torchrun`

If you do not use `primus-cli direct`, you can still run pretrain with `torchrun` and `--backend-path` to your Megatron-Bridge clone; see comments in the YAML and in `download_mlperf_flux1_datasets.sh`.
