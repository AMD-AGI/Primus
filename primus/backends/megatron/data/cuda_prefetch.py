###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
CUDA stream-based data prefetcher for overlapping HtoD transfers with compute.

Wraps an iterator (typically RerunDataIterator -> MegatronDataloaderWrapper ->
Energon SavableDataLoader) and transfers the next batch to GPU on a dedicated
secondary stream.

Each ``__next__`` call waits for the in-flight HtoD to complete, returns the
GPU batch, and immediately kicks off the *next* CPU fetch + HtoD dispatch so
the transfer overlaps with forward/backward compute.

Requirements:
    - Upstream DataLoader must use pin_memory=True for truly async non_blocking
      transfers. Energon's SavableDataLoader satisfies this.
    - Batches must be dict[str, Tensor | Any].
"""

import torch


class CudaPrefetchIterator:
    """Prefetch data batches to GPU on a secondary CUDA stream.

    On construction, eagerly fetches the first batch and dispatches its HtoD
    transfer (cold start).  Each subsequent ``__next__`` waits for the previous
    HtoD, grabs the GPU batch, kicks off the *next* prefetch, and returns.

    ``wait_stream()`` (not CUDA events) is used because prefetch depth is 1.
    The returned batch's tensors are ``record_stream()``-ed on the consuming
    stream: they are allocated on the prefetch stream, so without it a tensor the
    caller frees (on the CPU, while the GPU may still be reading it -- the CPU
    runs ahead) goes straight back to the prefetch stream's pool, and the next
    prefetch's copy can overwrite it before the consuming stream has read it.
    """

    def __init__(self, iterator, compute_dtype=torch.bfloat16):
        if isinstance(iterator, (list, tuple)):
            # Virtual pipeline parallel hands Megatron a list of per-chunk
            # iterators; this single-stream prefetcher only wraps one iterator.
            raise TypeError(
                "CudaPrefetchIterator expects a single data iterator, got "
                f"{type(iterator).__name__}. Skip prefetch for virtual pipeline "
                "parallel or wrap each per-chunk iterator separately."
            )
        self._iterator = iterator
        self._stream = torch.cuda.Stream()
        self._dtype = compute_dtype
        self._next_batch = None
        self._prefetch()

    def _prefetch(self):
        """Fetch next batch from upstream and dispatch HtoD on secondary stream."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self._stream):
            gpu_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        gpu_batch[k] = v.to(dtype=self._dtype, device="cuda", non_blocking=True)
                    else:
                        gpu_batch[k] = v.cuda(non_blocking=True)
                else:
                    gpu_batch[k] = v
            self._next_batch = gpu_batch

    def __next__(self):
        current = torch.cuda.current_stream()
        current.wait_stream(self._stream)
        batch = self._next_batch
        self._next_batch = None
        if batch is None:
            raise StopIteration
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                v.record_stream(current)
        self._prefetch()
        return batch

    def __iter__(self):
        return self
