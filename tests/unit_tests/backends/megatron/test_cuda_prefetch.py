###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for ``CudaPrefetchIterator`` cross-stream memory safety.

Batches are allocated on the prefetch stream and consumed on the compute
stream. Unless each returned tensor is ``record_stream()``-ed on the consumer,
the caching allocator hands a freed batch's block straight back to the
prefetch stream while compute kernels that read it are still queued, and the
next HtoD copy overwrites it. On gpt-oss-120B FP8 (delayed recipe) that
corrupted the token ids read by the embedding backward and surfaced as
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION.

Run:
    python -m pytest tests/unit_tests/backends/megatron/test_cuda_prefetch.py -v
"""

import pytest
import torch

from primus.backends.megatron.data.cuda_prefetch import CudaPrefetchIterator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

_NUMEL = 1 << 20
# Long enough that the consumer kernel is still queued when the next prefetch
# allocates, short enough to keep the test fast.
_SLEEP_CYCLES = 1_000_000_000


def _batches(n):
    for i in range(n):
        yield {
            "tokens": torch.full((_NUMEL,), i, dtype=torch.int64).pin_memory(),
            "loss_mask": torch.ones(_NUMEL, dtype=torch.float32).pin_memory(),
            "meta": i,
        }


def test_returned_batch_is_on_gpu_and_ordered():
    it = CudaPrefetchIterator(_batches(3))
    for i in range(3):
        batch = next(it)
        assert batch["tokens"].is_cuda
        assert batch["loss_mask"].dtype == torch.bfloat16
        assert batch["meta"] == i
        assert int(batch["tokens"][0]) == i
    with pytest.raises(StopIteration):
        next(it)


def test_freed_batch_is_not_reused_while_consumer_kernels_are_pending():
    it = CudaPrefetchIterator(_batches(3))
    batch = next(it)
    torch.cuda.synchronize()

    tokens = batch["tokens"]
    freed_ptr = tokens.data_ptr()

    # Queue a consumer read behind a long-running kernel, then drop every
    # Python reference while that read is still pending on the compute stream.
    torch.cuda._sleep(_SLEEP_CYCLES)
    result = tokens.sum()
    del batch, tokens

    # Returns batch 1 and allocates batch 2 on the prefetch stream.
    next(it)
    prefetched = it._next_batch["tokens"]
    assert (
        prefetched.data_ptr() != freed_ptr
    ), "prefetch reused a block the compute stream has not finished reading"

    torch.cuda.synchronize()
    assert int(result) == 0
