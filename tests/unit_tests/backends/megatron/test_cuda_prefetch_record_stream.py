"""CudaPrefetchIterator: a batch the caller frees while the consuming stream still has to read it must not be
overwritten by the next prefetch (its tensors live on the prefetch stream's pool unless record_stream()-ed)."""
import pytest
import torch

from primus.backends.megatron.data.cuda_prefetch import CudaPrefetchIterator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_freed_batch_not_overwritten_before_read():
    n, nb = 16 << 20, 64
    host = [torch.full((n,), float(i), dtype=torch.float32).pin_memory() for i in range(4)]
    it = CudaPrefetchIterator(({"x": host[i % 4], "i": i} for i in range(nb)), compute_dtype=torch.float32)
    burn = torch.randn(2048, 2048, device="cuda")
    sums = []
    for _ in range(nb - 1):
        b = next(it)
        for _ in range(8):  # keep the consuming stream busy so the read is still queued when the batch is freed
            burn = burn @ burn * 1e-3
        sums.append((b["i"], b["x"].sum()))
        del b
    assert all(s.item() == float(i % 4) * n for i, s in sums)
