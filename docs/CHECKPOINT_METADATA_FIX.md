# Checkpoint Structure Fix for SFT Training

## Problem

When converting HuggingFace checkpoints to Megatron format using Megatron-Bridge, two issues occur:

1. The `latest_checkpointed_iteration.txt` file contains `0`
2. The checkpoint is in `iter_0000000/` directory

This causes Megatron-LM to fail with:

```python
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/checkpoint/release'
```

## Root Cause

Megatron-LM's checkpoint system has strict conventions:

1. **Metadata file** (`latest_checkpointed_iteration.txt`):
   - Must contain `iteration > 0` OR the string `"release"`
   
2. **Directory structure**:
   - If iteration N: checkpoint must be in `iter_{N:07d}/` (e.g., `iter_0000100/`)
   - If release: checkpoint must be in `release/` directory

**The mismatch**: Megatron-Bridge creates:
- Metadata: `0` (invalid)
- Directory: `iter_0000000/`

When we change metadata to `"release"`, Megatron looks for `release/` directory but finds `iter_0000000/` instead.

## Solution

### Immediate Fix (Manual)

For existing converted checkpoints:

```bash
cd /path/to/checkpoint
# Step 1: Update metadata
echo "release" > latest_checkpointed_iteration.txt

# Step 2: Rename directory
mv iter_0000000 release
```

### Automated Fix (In Conversion Script)

The conversion hook now automatically fixes both metadata and directory:

```python
# Fix metadata and directory structure
metadata_file = megatron_path / "latest_checkpointed_iteration.txt"
iter_dir = megatron_path / "iter_0000000"
release_dir = megatron_path / "release"

if metadata_file.exists() and iter_dir.exists():
    with open(metadata_file, 'r') as f:
        content = f.read().strip()
    
    if content == "0":
        # Step 1: Update metadata
        with open(metadata_file, 'w') as f:
            f.write("release")
        
        # Step 2: Rename directory
        if not release_dir.exists():
            iter_dir.rename(release_dir)
```

## Files Modified

1. **`runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py`**
   - Added automatic metadata fix
   - Added automatic directory renaming
   - Ensures consistency between metadata and directory structure

## Testing

✅ Manual fix applied to existing checkpoint:

```bash
$ cat /wekafs/xiaoming/dev/Primus/data/megatron_checkpoints/Meta-Llama-3-8B/latest_checkpointed_iteration.txt
release

$ ls /wekafs/xiaoming/dev/Primus/data/megatron_checkpoints/Meta-Llama-3-8B/
release/  latest_checkpointed_iteration.txt  latest_train_state.pt
```

## Why Both Fixes Are Needed

**Metadata alone is not enough**:
```python
# Megatron checkpoint naming (checkpointing.py:150-153)
def get_checkpoint_name(checkpoints_path, iteration, release=False, ...):
    if release:
        directory = 'release'  # <-- Expects this directory
    else:
        directory = 'iter_{:07d}'.format(iteration)
```

When metadata says `"release"`, Megatron constructs path: `checkpoints_path/release/`

If the actual directory is `iter_0000000/`, it won't be found → `FileNotFoundError`

## Complete Directory Structure

**Before fix** (from Megatron-Bridge):
```
Meta-Llama-3-8B/
├── latest_checkpointed_iteration.txt  (contains "0")
├── latest_train_state.pt
└── iter_0000000/
    ├── mp_rank_00_model_states.pt
    └── ... (checkpoint files)
```

**After fix** (Megatron-LM compatible):
```
Meta-Llama-3-8B/
├── latest_checkpointed_iteration.txt  (contains "release")
├── latest_train_state.pt
└── release/
    ├── mp_rank_00_model_states.pt
    └── ... (checkpoint files)
```

## Related Code

**Megatron-LM checkpoint loading** (`third_party/Megatron-LM/megatron/training/checkpointing.py`):

```python
# Line 144-153: Directory naming convention
def get_checkpoint_name(checkpoints_path, iteration, release=False, ...):
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    # ... constructs full path using 'directory'

# Line 273-290: Metadata parsing
def read_metadata(tracker_filename):
    iteration, release = -1, False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
    assert iteration > 0 or release  # Must satisfy this
    return iteration, release

# Line 1000: Using release flag to construct path
checkpoint_name = get_checkpoint_name(load_dir, iteration, release, ...)
# If release=True, this returns: load_dir/release/
```

## Future Improvements

Consider upstreaming these fixes to Megatron-Bridge's `AutoBridge.import_ckpt()`:
1. Create `release/` directory instead of `iter_0000000/`
2. Write `"release"` to metadata file instead of `"0"`

This would eliminate the need for post-processing.

## References

- Megatron-LM checkpointing: `third_party/Megatron-LM/megatron/training/checkpointing.py`
- Conversion hook: `runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py`
- Directory naming: Lines 144-153 in checkpointing.py
- Metadata parsing: Lines 273-290 in checkpointing.py
