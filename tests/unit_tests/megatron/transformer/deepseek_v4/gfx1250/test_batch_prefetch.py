###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Next-batch prefetch for the full-iteration CUDA graph (``PRIMUS_FUSE2_PREFETCH``; the patch is installed
directly here, independent of the switch value).

A stand-in subclass of Megatron's ``FullCudaGraphWrapper`` (its graph call is replaced by a no-op) is patched and
driven like a training loop: the batches the steps read through the patched ``data_read`` are the batches the
iterator yields, in the same order, with the same bytes; when the iterator runs out in the middle of a prefetch
the call neither raises nor leaves a half-consumed prefetch behind, and the steps after it read through
Megatron's own ``data_read``.
"""

import types

import pytest
import torch

from .helpers import requires_gfx1250

pytestmark = requires_gfx1250


def _batches(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        {
            "tokens": torch.randint(0, 1000, (1, 16), generator=g),
            "labels": torch.randint(0, 1000, (1, 16), generator=g),
            "loss_mask": torch.rand(1, 16, generator=g),
        }
        for _ in range(n)
    ]


@pytest.fixture
def wrapper(monkeypatch):
    fcg = pytest.importorskip("megatron.core.full_cuda_graph")
    rsm = pytest.importorskip("megatron.core.rerun_state_machine")
    from primus.backends.megatron.patches import fuse2_prefetch_patches as P

    class _Graph(fcg.FullCudaGraphWrapper):
        def __call__(self, *args, **kwargs):  # stands for the graph replay
            return "replayed"

    monkeypatch.setattr(fcg, "FullCudaGraphWrapper", _Graph)
    monkeypatch.setattr(fcg.StaticBufferLoader, "static_buffers", {"training": [], "validation": []})
    monkeypatch.setattr(
        rsm, "get_rerun_state_machine", lambda: types.SimpleNamespace(mode=rsm.RerunMode.DISABLED)
    )
    P.install()
    return _Graph(lambda *a, **k: None)


def _train(w, it, model, n_micro, steps):
    """Per step: read the batches, then 'run' the step (which launches the next prefetch)."""
    got = []
    for _ in range(steps):
        data = w.data_read(it, model, True, n_micro)
        got.append([{k: v.clone() for k, v in next(data[0]).items()} for _ in range(n_micro)])
        assert w(data_iterator=it, model=model, num_microbatches=n_micro, forward_only=False) == "replayed"
        torch.cuda.synchronize()
    return got


@pytest.mark.parametrize("n_micro", [1, 2])
def test_prefetch_preserves_batch_order_and_bytes(wrapper, n_micro):
    steps = 3
    bs = _batches(steps * n_micro)
    got = _train(wrapper, iter(bs), [object()], n_micro, steps)
    flat = [b for step in got for b in step]
    assert len(flat) == len(bs)
    for want, have in zip(bs, flat):
        for k in want:
            assert torch.equal(want[k], have[k].cpu()), f"batch field {k} differs"


@pytest.mark.parametrize("n_micro,total", [(1, 3), (2, 5)])
def test_prefetch_at_end_of_data(wrapper, n_micro, total):
    bs = _batches(total)
    it = iter(bs)
    steps = total // n_micro
    got = _train(
        wrapper, it, [object()], n_micro, steps
    )  # the last call's prefetch hits the end of the iterator
    flat = [b for step in got for b in step]
    for want, have in zip(bs, flat):
        assert torch.equal(want["tokens"], have["tokens"].cpu())
    assert getattr(wrapper, "_f2_pf", None) is None
    assert getattr(wrapper, "_f2_off", False), "prefetch stays on after the data ran out"
    # a further step falls back to Megatron's data_read, which ends the epoch the usual way
    with pytest.raises(StopIteration):
        wrapper.data_read(it, [object()], True, n_micro)
