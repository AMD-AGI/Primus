# Offline Dataset Support - Implementation Summary

## Overview

Added comprehensive offline dataset support to the SFT trainer, enabling training with local JSONL and JSON files without requiring internet connectivity or HuggingFace Hub access.

## What Was Implemented

### 1. Core Functionality

**New Function: `load_jsonl_file()`**
- Loads JSONL (JSON Lines) files efficiently
- Handles line-by-line parsing
- Provides detailed error messages with line numbers
- Logs progress for monitoring

**Updated: `SFTDataset.__init__()`**
- Automatic file type detection (JSONL, JSON, or HuggingFace)
- Support for `.jsonl` file extension
- Support for `.json` file extension (JSON array)
- Fallback to HuggingFace Hub for non-local paths
- Converts local data to HuggingFace Dataset format for compatibility

### 2. Supported Data Formats

#### JSONL Format (Preferred for Large Datasets)
```jsonl
{"instruction": "Question 1", "response": "Answer 1"}
{"instruction": "Question 2", "response": "Answer 2"}
```

#### JSON Array Format
```json
[
    {"instruction": "Question 1", "response": "Answer 1"},
    {"instruction": "Question 2", "response": "Answer 2"}
]
```

### 3. Field Name Flexibility

The loader automatically handles multiple field name conventions:

| Purpose | Supported Names (priority order) |
|---------|----------------------------------|
| Instruction | `instruction` → `prompt` → `question` |
| Response | `response` → `output` → `answer` |
| Input Context | `input` |
| System Prompt | `system` |

### 4. Documentation

**Created/Updated:**
- `primus/backends/megatron/README_SFT.md` - Added offline dataset section
- `IMPLEMENTATION_SUMMARY.md` - Updated with offline support info
- `docs/OFFLINE_DATASET_GUIDE.md` - Comprehensive guide with examples
- `examples/megatron/configs/MI355X/llama3_8B-BF16-sft.yaml` - Added offline example

### 5. Tools & Utilities

**Created: `examples/megatron/convert_to_jsonl.py`**
- Convert HuggingFace datasets to JSONL
- Convert CSV files to JSONL
- Support for custom column names
- Progress monitoring

Usage:
```bash
# Convert HuggingFace dataset
python convert_to_jsonl.py --dataset tatsu-lab/alpaca --output data.jsonl

# Convert CSV file
python convert_to_jsonl.py --csv data.csv --output data.jsonl
```

### 6. Testing

**Created: `tests/unit_tests/backends/megatron/test_sft_dataset_offline.py`**
- Test JSONL file loading
- Test JSON array file loading
- Test error handling (missing files, invalid JSON)
- Test dataset integration
- Test file type detection

## Usage

### Before (Online Only)
```yaml
modules:
  trainer:
    overrides:
      sft_dataset_name: "tatsu-lab/alpaca"  # Requires internet
```

### After (Online OR Offline)
```yaml
modules:
  trainer:
    overrides:
      # Option 1: HuggingFace Hub (online)
      sft_dataset_name: "tatsu-lab/alpaca"
      
      # Option 2: Local JSONL file (offline)
      # sft_dataset_name: "/path/to/data.jsonl"
      
      # Option 3: Local JSON file (offline)
      # sft_dataset_name: "/path/to/data.json"
```

## Technical Details

### File Detection Logic
```python
is_local_file = (
    dataset_name.endswith('.jsonl') or 
    dataset_name.endswith('.json') or 
    os.path.isfile(dataset_name)
)
```

### Data Flow
```
Local File (.jsonl/.json)
    ↓
load_jsonl_file() or json.load()
    ↓
List[Dict]
    ↓
HFDataset.from_list()
    ↓
SFTDataset (standardized interface)
```

## Benefits

| Feature | Online (HF Hub) | Offline (JSONL) |
|---------|----------------|-----------------|
| Internet Required | ✅ Yes | ❌ No |
| Data Privacy | ⚠️ Uploaded | ✅ Local |
| Setup Complexity | 🟢 Low | 🟢 Low |
| Custom Data | 🟡 Upload | 🟢 Direct |
| Format | 🔒 Fixed | 🔓 Flexible |
| Reproducibility | 🟡 May Change | ✅ Fixed |

## Backward Compatibility

✅ **100% Backward Compatible**
- Existing HuggingFace dataset configurations work unchanged
- No breaking changes to API
- All existing tests pass
- Automatic detection of local vs remote

## Error Handling

Provides clear error messages for common issues:

1. **File not found:**
   ```
   FileNotFoundError: JSONL file not found: /path/to/file.jsonl
   ```

2. **Invalid JSON:**
   ```
   JSONDecodeError: Invalid JSON on line 5 in file.jsonl: ...
   ```

3. **Wrong format:**
   ```
   ValueError: JSON file must contain a list of objects
   ```

## Performance Characteristics

- **Memory**: Entire dataset loaded into memory (as with HuggingFace datasets)
- **Loading Speed**: Fast for files <1GB
- **Recommended**: Use JSONL for files >100MB (easier to stream in future)

## Future Enhancements

Potential improvements (not in current implementation):
- CSV format support (via pandas)
- Parquet format support
- Streaming for very large files
- Configurable field name mapping
- Automatic train/val/test splitting

## Testing & Verification

All tests passed:
- ✅ Load JSONL file with 5 samples
- ✅ Load JSON array file with 3 samples
- ✅ Detect local vs HuggingFace paths correctly
- ✅ Handle missing files gracefully
- ✅ Handle invalid JSON with clear errors
- ✅ Field name flexibility works
- ✅ Integration with SFTDataset works

## Files Changed

**Modified:**
1. `primus/backends/megatron/core/datasets/sft_dataset.py` - Core implementation
2. `primus/backends/megatron/README_SFT.md` - Documentation
3. `IMPLEMENTATION_SUMMARY.md` - Summary update
4. `examples/megatron/configs/MI355X/llama3_8B-BF16-sft.yaml` - Example config

**Created:**
1. `docs/OFFLINE_DATASET_GUIDE.md` - Comprehensive guide
2. `examples/megatron/convert_to_jsonl.py` - Conversion utility
3. `tests/unit_tests/backends/megatron/test_sft_dataset_offline.py` - Unit tests

## Code Quality

- ✅ Type hints added
- ✅ Comprehensive docstrings
- ✅ Error handling with meaningful messages
- ✅ Logging for debugging
- ✅ Unit tests with good coverage
- ✅ Follows existing code style
- ✅ No breaking changes

## Summary

This implementation successfully addresses the requirement for offline dataset support while maintaining full backward compatibility. Users can now:

1. Train with local JSONL/JSON files (no internet needed)
2. Keep sensitive data private (stays local)
3. Use custom datasets without uploading
4. Switch easily between online and offline modes
5. Convert existing datasets using provided tools

The implementation is production-ready, well-tested, and documented.
