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

#### Option 2: Use Alternative Dataset
```yaml
dataset:
  _target_: megatron.bridge.data.builders.hf_dataset.HFDatasetConfig
  dataset_name: "rajpurkar/squad_v2"  # Try v2 if available
  seq_length: 2048
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

### Mock Dataset (Recommended)
- **File:** `examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain_mock.yaml`
- **Use case:** Testing, development, no network required
- **Features:** Fast, reliable, no external dependencies

### Real Dataset (When Available)
- **File:** `examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml`
- **Use case:** Production training with actual data
- **Note:** Requires compatible dataset and network access

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
