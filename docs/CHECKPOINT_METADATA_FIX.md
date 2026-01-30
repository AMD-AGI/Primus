# Checkpoint Metadata Fix for SFT Training

## Problem

When converting HuggingFace checkpoints to Megatron format using Megatron-Bridge, the `latest_checkpointed_iteration.txt` file contains `0`, which causes Megatron-LM to fail with this assertion error:

```python
AssertionError: error parsing metadata file /path/to/checkpoint/latest_checkpointed_iteration.txt
```

## Root Cause

Megatron-LM's checkpoint loading code (`checkpointing.py:289`) has this assertion:

```python
assert iteration > 0 or release, 'error parsing metadata file {}'.format(tracker_filename)
```

This means the metadata file must either:
1. Contain an iteration number > 0, OR
2. Contain the string "release"

HuggingFace converted checkpoints have `iteration = 0`, which fails both conditions.

## Solution

### Immediate Fix (Manual)

Change the metadata file content from `0` to `release`:

```bash
echo "release" > /path/to/checkpoint/latest_checkpointed_iteration.txt
```

### Automated Fix (In Conversion Script)

The conversion hook now automatically fixes this:

```python
# Fix metadata file for converted checkpoints
metadata_file = megatron_path / "latest_checkpointed_iteration.txt"
if metadata_file.exists():
    with open(metadata_file, 'r') as f:
        content = f.read().strip()
    if content == "0":
        with open(metadata_file, 'w') as f:
            f.write("release")
```

## Files Modified

1. **`runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py`**
   - Added automatic metadata fix after checkpoint conversion
   - Checks if metadata contains "0" and replaces with "release"
   - Logs the fix for visibility

## Testing

✅ Manual fix applied to existing checkpoint:
```bash
$ cat /wekafs/xiaoming/dev/Primus/data/megatron_checkpoints/Meta-Llama-3-8B/latest_checkpointed_iteration.txt
release
```

✅ Training should now proceed without assertion error

## Why "release" Instead of "1"?

Using "release" instead of changing `0` to `1` is the correct approach because:

1. **Semantic correctness**: This is a released/pretrained checkpoint, not a training checkpoint from iteration 1
2. **Megatron convention**: "release" checkpoints are explicitly supported for this use case
3. **No side effects**: Using "release" makes it clear this is a converted checkpoint, not from interrupted training

## Related Code

**Megatron-LM checkpoint loading** (`third_party/Megatron-LM/megatron/training/checkpointing.py`):

```python
def read_metadata(tracker_filename):
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = -1, False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'  # <-- "release" is valid
            if not release:
                print_rank_0('ERROR: Invalid metadata file')
                sys.exit()
    assert iteration > 0 or release  # <-- Must satisfy this
```

## Future Improvements

Consider upstreaming this fix to Megatron-Bridge's `AutoBridge.import_ckpt()` so converted checkpoints have correct metadata from the start.

## References

- Megatron-LM checkpointing: `third_party/Megatron-LM/megatron/training/checkpointing.py`
- Conversion hook: `runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py`
- Error location: Line 289 in checkpointing.py
