# Parallel State API Fix

## Problem

Training was crashing with:
```
RuntimeError: module 'megatron.core.tensor_parallel' has no attribute 'get_data_parallel_world_size'
```

## Root Cause

The SFT trainer was using incorrect Megatron-LM API:
- Used `tensor_parallel.get_data_parallel_world_size()` 
- Used `tensor_parallel.get_data_parallel_group()`
- These functions don't exist in `megatron.core.tensor_parallel`

## Solution

Changed to use the correct `parallel_state` module:

**Before (incorrect):**
```python
from megatron.core import tensor_parallel

if tensor_parallel.get_data_parallel_world_size() > 1:
    torch.distributed.all_reduce(loss, group=tensor_parallel.get_data_parallel_group())
```

**After (correct):**
```python
from megatron.core import parallel_state

if parallel_state.get_data_parallel_world_size() > 1:
    torch.distributed.all_reduce(loss, group=parallel_state.get_data_parallel_group())
```

## Megatron-LM API Reference

### Correct Module: `megatron.core.parallel_state`

Data parallel operations:
- `parallel_state.get_data_parallel_group()` - Returns data parallel process group
- `parallel_state.get_data_parallel_rank()` - Returns rank within data parallel group
- `parallel_state.get_data_parallel_world_size()` - Returns size of data parallel group

Tensor parallel operations:
- `parallel_state.get_tensor_model_parallel_group()`
- `parallel_state.get_tensor_model_parallel_rank()`
- `parallel_state.get_tensor_model_parallel_world_size()`

Pipeline parallel operations:
- `parallel_state.get_pipeline_model_parallel_group()`
- `parallel_state.get_pipeline_model_parallel_rank()`
- `parallel_state.get_pipeline_model_parallel_world_size()`

### tensor_parallel vs parallel_state

- `megatron.core.tensor_parallel` - Contains tensor parallelism operations (scatter, gather, all_reduce for tensors)
- `megatron.core.parallel_state` - Contains parallelism state management (groups, ranks, world sizes)

## Changes Made

1. **Line 159**: Changed import from `tensor_parallel` to `parallel_state`
2. **Line 241**: Updated `get_data_parallel_world_size()` call
3. **Line 242**: Updated `get_data_parallel_group()` call
4. **Line 206**: Removed unused `tensor_parallel` import from loss_func

## Why This Matters

The loss averaging in SFT training requires:
1. Computing loss on each data parallel rank
2. All-reducing the loss across data parallel ranks
3. This ensures consistent training across all replicas

Using the wrong API would cause:
- Runtime errors (attribute not found)
- Incorrect loss computation
- Training failures

## Verification

The fix can be verified by checking:
1. Import uses `parallel_state` not `tensor_parallel`
2. All data parallel operations use `parallel_state` module
3. Training proceeds without API-related errors
4. Loss is properly averaged across data parallel ranks

## References

This fix aligns with usage in other Megatron files:
- `primus/backends/megatron/training/utils.py` (line 20)
- `primus/backends/megatron/core/distributed/finalize_model_grad.py` (line 70)
- `primus/backends/megatron/core/optimizer/moun.py` (line 202)
