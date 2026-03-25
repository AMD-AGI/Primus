# Data Preparation Guide

Primus routes training through **Megatron-LM**, **TorchTitan**, and **MaxText**. Each backend expects its own data format and preprocessing pipeline. This guide summarizes how to prepare data, how to use **mock** data for smoke tests, and which environment variables commonly apply.

Scripts referenced below live under the Primus repository root, for example:

- `examples/megatron/preprocess_data.py`
- `examples/megatron/prepare.py`
- `examples/megatron/prepare_bookcorpus_megatron_dataset.py`
- `examples/torchtitan/prepare.py`

---

## 1. Overview

| Backend | Format | Typical entry |
|---------|--------|----------------|
| Megatron | Indexed `.bin` + `.idx` datasets | `data_path`, `train_data_path`, tokenizer args |
| TorchTitan | Hugging Face datasets + local assets | `training.dataset`, `training.dataset_path`, `model.hf_assets_path` |
| MaxText | TFDS / Hugging Face / Grain / synthetic | `dataset_type`, paths per pipeline |

All backends support **synthetic or mock** data for configuration and scaling tests without large downloads.

---

## 2. Mock data (testing)

### Megatron

Set in the trainer module:

```yaml
mock_data: true
```

Default in `primus/configs/modules/megatron/trainer_base.yaml` is `false`. When `true`, training uses generated data matching configured dimensions so you can validate YAML, parallelism, and throughput without real corpora.

### TorchTitan

```yaml
training:
  mock_data: true
```

Default in `primus/configs/modules/torchtitan/pre_trainer.yaml` is `true` (useful for quick runs; set `false` and supply real datasets for production).

### MaxText

Use `dataset_type: synthetic` (or other synthetic paths in MaxText configs). See `third_party/maxtext/src/MaxText/configs/base.yml` and model YAMLs under `third_party/maxtext/src/MaxText/configs/`.

---

## 3. Megatron data pipeline

### Inputs

- Raw **JSON** or **JSONL** text (one JSON object per line for JSONL).
- Optional **sentence splitting** via NLTK when `--split-sentences` is used (requires NLTK data; see [Environment variables](#6-environment-variables-for-data)).

### Preprocessing: `examples/megatron/preprocess_data.py`

The script tokenizes input and writes **Megatron indexed datasets** (`.bin` + `.idx`). It uses `build_tokenizer` from Primus’s Megatron tokenizer integration and accepts tokenizer flags from `_add_tokenizer_args`.

**Important arguments** (from the script’s argparse):

| Argument | Description |
|----------|-------------|
| `--input` | Path to input JSON (required). |
| `--json-keys` | Keys to read (default `text`). |
| `--output-prefix` | Output path **without** suffix; produces `{prefix}_{key}_{document|sentence}.bin` and `.idx`. |
| `--workers` | Number of worker processes (required). |
| `--partitions` | Split input for parallel preprocessing (default `1`). |
| `--split-sentences` | Run NLTK sentence splitting before encode. |
| `--append-eod` | Append end-of-document token. |

**Example** (mirrors `examples/megatron/prepare.py` for BookCorpus-style flows):

```bash
python3 examples/megatron/preprocess_data.py \
  --input /path/to/train.json \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /path/to/tokenizer \
  --output-prefix /path/to/out/bookcorpus_train \
  --workers "$(nproc)" \
  --split-sentences \
  --partitions 2
```

### Configuring training runs

| Parameter | Notes |
|-----------|--------|
| `data_path` | Single path or **weighted blend**: `0.5 /path/a 0.5 /path/b` |
| `train_data_path`, `valid_data_path`, `test_data_path` | Separate splits when used |
| `split` | Train/valid/test ratio string, e.g. `"99,1,0"` (default in `trainer_base.yaml`) or `"98,2,0"` for train/valid/test |
| `dataloader_type` | Megatron dataloader type; default in `trainer_base.yaml` is `null` (set explicitly in experiments as needed) |

### BookCorpus example scripts

- **`examples/megatron/prepare_bookcorpus_megatron_dataset.py`** — downloads BookCorpus to JSON via Hugging Face `datasets`, optional `--out-dir`.
- **`examples/megatron/prepare.py`** — orchestrates download, train/valid split, and calls `preprocess_data.py` with tokenizer settings from Primus config; respects `TOKENIZED_TRAIN_DATA_PATH` / `TOKENIZED_EVAL_DATA_PATH` for output locations.

### Tokenizers

Tokenizer type and model path are set on the model preset (for example `tokenizer_type`, `tokenizer_model` in `primus/configs/models/megatron/language_model.yaml` comments list `Llama2Tokenizer`, `HuggingFaceTokenizer`, etc.).

---

## 4. TorchTitan data pipeline

TorchTitan uses **Hugging Face datasets** style identifiers and local paths.

| Key | Default (`pre_trainer.yaml`) | Description |
|-----|------------------------------|-------------|
| `training.dataset` | `c4` | Dataset identifier for TorchTitan loaders. |
| `training.dataset_path` | `null` | Local directory for dataset assets when needed. |

Tokenizer and model assets are resolved from **`model.hf_assets_path`** (or equivalent in your model preset). The preparation script **`examples/torchtitan/prepare.py`**:

- Resolves the TorchTitan checkout path.
- Runs `scripts/download_hf_assets.py` inside TorchTitan to fetch tokenizer assets for a given `repo_id`.
- Uses `HF_TOKEN` when the model or dataset is gated.

---

## 5. MaxText data pipeline

MaxText configuration is defined in upstream YAML (for example `third_party/maxtext/src/MaxText/configs/base.yml`).

| Parameter | Meaning |
|-----------|---------|
| `dataset_type` | One of `synthetic`, `hf`, `grain`, `tfds` (per `base.yml` comments). |
| `hf_path`, `hf_data_dir`, `hf_train_files` | Hugging Face pipeline inputs when `dataset_type: hf`. |
| `per_device_batch_size` | Batch sizing on each device. |
| `packing` | Sequence packing for efficiency (default `True` in `base.yml`). |

See MaxText’s data input documentation for Grain and TFDS specifics.

---

## 6. Environment variables for data

| Variable | Usage |
|----------|--------|
| `TOKENIZED_DATA_PATH` / `PRIMUS_TOKENIZED_DATA_PATH` | Tokenized dataset locations for Megatron hooks and examples (see `production_docs/03-configuration-reference/environment-variables.md`). |
| `TOKENIZED_TRAIN_DATA_PATH`, `TOKENIZED_EVAL_DATA_PATH` | Override output paths in `examples/megatron/prepare.py`. |
| `DATA_PATH` | General data root used in scripts and CI-style launches. |
| `HF_TOKEN` | **Required** for gated Hugging Face models and some datasets (TorchTitan `prepare.py`, Kubernetes examples in `examples/README.md`). |
| `HF_HOME` | Hugging Face cache directory (used in `examples/megatron/prepare.py`). |
| `NLTK_DATA` | NLTK tokenizer data directory for sentence splitting in `preprocess_data.py` when `NLTK_DATA` is set. |

---

## Summary

1. Use **mock** / **synthetic** data to validate configs and performance before investing in large preprocessing jobs.
2. For **Megatron**, convert JSON/JSONL to `.bin`/`.idx` with `preprocess_data.py` and point `data_path` or split paths at the outputs.
3. For **TorchTitan**, set `training.dataset` / `dataset_path` and run **`examples/torchtitan/prepare.py`** to fetch tokenizer assets.
4. For **MaxText**, configure `dataset_type` and `per_device_batch_size` per upstream `base.yml` and model YAMLs.
