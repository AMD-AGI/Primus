# Qwen3-32B Post-Training on Megatron-Bridge

## Quick Start (Recommended)

### Using Mock Dataset
**Recommended for testing and development** - No network or dataset issues:

```bash
# From Primus root
bash examples/megatron_bridge/run_qwen3_32b_sft_mock.sh

# Or directly
bash run.sh examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain_mock.yaml
```

## Known Issues with Real Datasets

### Issue 1: HuggingFace Dataset Namespace Migration (Jan 2026)

**Problem:**
- Old path `squad` → 404 Not Found
- HuggingFace enforced `owner/dataset` namespace format

**Solution:**
- Use `rajpurkar/squad` instead of `squad`
- Already updated in config templates

### Issue 2: SQuAD Dataset Metadata Compatibility

**Problem:**
```
ValueError: Feature type 'List' not found.
Available feature types: ['Value', 'ClassLabel', ..., 'LargeList', 'Sequence', ...]
```

**Root Cause:**
- `rajpurkar/squad` dataset metadata uses deprecated `List` feature type
- Modern `datasets` library (>=3.0) removed `List`, replaced with `Sequence`/`LargeList`
- Dataset maintainer hasn't updated metadata yet

**Solutions:**

#### Option 1: Use Mock Dataset (Best)  ✅
```yaml
dataset:
  _target_: megatron.bridge.data.datasets.mock_dataset.MockDatasetConfig
  seq_length: 2048
  vocab_size: 152064  # Qwen3 vocab size
  num_train_samples: 50000
  num_val_samples: 5000
  num_test_samples: 5000
```

#### Option 2: Use Alternative Dataset (Recommended) ✅

**Verified Compatible Datasets:**

##### A. Stanford Alpaca (Cleaned)
- **Dataset:** `yahma/alpaca-cleaned`
- **Size:** 51,760 instruction-following samples
- **Features:** instruction, input, output
- **Config:** `qwen3_32b_sft_alpaca.yaml`

```bash
bash run.sh examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_alpaca.yaml
```

##### B. Databricks Dolly 15K
- **Dataset:** `databricks/databricks-dolly-15k`
- **Size:** 15,011 instruction-response pairs
- **Features:** instruction, context, response, category
- **Config:** `qwen3_32b_sft_dolly.yaml`

```bash
bash run.sh examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_dolly.yaml
```

##### C. OpenAssistant (oasst1)
- **Dataset:** `OpenAssistant/oasst1`
- **Size:** 84,437 conversation messages
- **Features:** Multi-turn conversations with rankings
- **Note:** Requires custom processing function

##### D. Custom Dataset Configuration
```yaml
dataset:
  _target_: megatron.bridge.data.builders.hf_dataset.HFDatasetConfig
  dataset_name: "your-dataset-name"  # Any compatible HF dataset
  seq_length: 2048
  seed: 5678
  dataloader_type: "batch"
  num_workers: 2
  do_validation: true
  val_proportion: 0.1
```

#### Option 3: Use Local Dataset
If you have a local copy:
```yaml
dataset:
  _target_: megatron.bridge.data.builders.hf_dataset.HFDatasetConfig
  dataset_name: "/path/to/local/squad"
  seq_length: 2048
```

## Configuration Files

### Available Datasets

| Config File | Dataset | Size | Status | Use Case |
|-------------|---------|------|--------|----------|
| `qwen3_32b_sft_posttrain_mock.yaml` | Mock Data | Configurable | ✅ Ready | Testing, Development |
| `qwen3_32b_sft_alpaca.yaml` | Alpaca Cleaned | 51K | ✅ Verified | Instruction Following |
| `qwen3_32b_sft_dolly.yaml` | Dolly 15K | 15K | ✅ Verified | General Instructions |
| `qwen3_32b_sft_posttrain.yaml` | Squad (default) | N/A | ⚠️ Incompatible | Reference Only |

### Mock Dataset (Recommended for Testing)
- **File:** `examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain_mock.yaml`
- **Use case:** Testing, development, no network required
- **Features:** Fast, reliable, no external dependencies

### Real Datasets (Production Ready)

#### Alpaca Cleaned - Instruction Following
- **File:** `examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_alpaca.yaml`
- **Dataset:** `yahma/alpaca-cleaned`
- **Samples:** 51,760 instruction-response pairs
- **Use case:** General-purpose instruction tuning
- **Command:**
  ```bash
  bash run.sh examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_alpaca.yaml
  ```

#### Dolly 15K - Diverse Instructions
- **File:** `examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_dolly.yaml`
- **Dataset:** `databricks/databricks-dolly-15k`
- **Samples:** 15,011 high-quality instruction-response pairs
- **Categories:** Open Q&A, Closed Q&A, Creative Writing, etc.
- **Use case:** Diverse instruction capabilities
- **Command:**
  ```bash
  bash run.sh examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_dolly.yaml
  ```

### Legacy Config (Reference Only)
- **File:** `examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml`
- **Use case:** Template for custom dataset configuration
- **Note:** Default squad dataset has compatibility issues (see troubleshooting)

## Troubleshooting

### Error: 404 Not Found for dataset
- **Cause:** Dataset path changed (namespace migration)
- **Fix:** Update to `owner/dataset` format (e.g., `rajpurkar/squad`)

### Error: Feature type 'List' not found
- **Cause:** Dataset metadata incompatibility
- **Fix:** Use mock dataset or alternative dataset

### Network/Access Issues
- **Cause:** Firewall, proxy, or HuggingFace Hub unavailable
- **Fix:** Use mock dataset (no network required)

## Support

For issues or questions:
1. Check this README for known issues
2. Verify your configuration matches the templates
3. Try mock dataset first to isolate dataset-specific issues
