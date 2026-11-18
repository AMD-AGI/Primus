###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the shared buffer that carries the var-len packing to every
Ideogram-4 attention layer.

WHAT THESE ARE DEFENDING:
  Every attention module holds the SAME tensor object, so the adapter publishes
  the packing with one ``copy_`` per step. Losing that sharing is silent:
  ``to_empty()`` gives each module its own buffer, and a publish that still
  trusted the sharing would reach one layer while every other layer read
  uninitialized memory, with nothing raised. So these pin the sharing itself:

    * every consumer reads the same tensor after a publish;
    * a checkpoint wrapper, which forwards attribute lookups to the module it
      wraps, is not counted as a second consumer;
    * a publish after ``to_empty()`` restores one shared tensor, and says so;
    * a new batch size replaces the buffer, and says so;
    * a model without the var-len processor has no consumers, which is an error
      only when the caller requires one;
    * a module that gains the processor after the first publish is included;
    * ``clear_packing`` removes the buffer from every consumer.

The processor tests attach a packing to one module by hand, and the adapter tests
replace ``publish_packing`` with a recorder, so neither exercises any of this.
CPU only, with no diffusers, AutoModel or Primus-Turbo.
"""

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.models.ideogram4 import (  # noqa: E402
    attn_processor,
    cu_seqlens,
    packing_buffer,
)

CPU = torch.device("cpu")


class FakeAttention(torch.nn.Module):
    """What the transport sees of an attention module: the var-len processor
    it is found by, and a parameter for ``to_empty()`` to materialize."""

    def __init__(self):
        super().__init__()
        self.processor = attn_processor.Ideogram4VarlenAttnProcessor()
        self.proj = torch.nn.Linear(4, 4)


def wrapped_model(layers=3):
    """Attention modules inside checkpoint wrappers, as activation checkpointing
    leaves them in a real run."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    return torch.nn.Sequential(*(checkpoint_wrapper(FakeAttention()) for _ in range(layers)))


def consumers(model):
    """The attention modules, found without the code under test."""
    return [m for m in model.modules() if isinstance(m, FakeAttention)]


def packing_for(text_lengths):
    return cu_seqlens.build_cu_seqlens(text_lengths, max_text_tokens=8, num_image_tokens=4)


@pytest.fixture(autouse=True)
def fresh_log(monkeypatch):
    """Warnings are logged once per process; start each test with none logged."""
    monkeypatch.setattr(packing_buffer, "_logged", set())


def test_every_consumer_reads_the_same_tensor():
    """The point of the buffer: one ``copy_`` reaches every layer."""
    model = wrapped_model()
    published = packing_for([3, 5])
    packing_buffer.publish_packing(model, published, 11, device=CPU, required=True)

    modules = consumers(model)
    shared = getattr(modules[0], packing_buffer.PACKING_ATTR)
    assert all(getattr(m, packing_buffer.PACKING_ATTR) is shared for m in modules)
    assert shared.tolist() == published.tolist()
    assert all(getattr(m, packing_buffer.BOUND_ATTR) == 11 for m in modules)


def test_a_checkpoint_wrapper_is_counted_once():
    """The wrapper answers ``processor`` with its child's, so a plain getattr
    would count every layer twice."""
    model = wrapped_model()
    found = packing_buffer.attention_modules(model)
    assert len(found) == 3
    assert found == consumers(model)


def test_to_empty_breaks_sharing_and_publishing_restores_it(caplog):
    """Rule 2 in packing_buffer.py: meta-device materialization quietly ends the
    sharing, so the next publish has to notice and restore it."""
    model = wrapped_model()
    packing_buffer.publish_packing(model, packing_for([3, 5]), 11, device=CPU, required=True)

    model.to_empty(device="cpu")
    modules = consumers(model)
    held = [getattr(m, packing_buffer.PACKING_ATTR) for m in modules]
    assert len({id(t) for t in held}) == len(modules)

    fresh = packing_for([1, 7])
    with caplog.at_level("WARNING"):
        packing_buffer.publish_packing(model, fresh, 11, device=CPU, required=True)

    shared = getattr(modules[0], packing_buffer.PACKING_ATTR)
    assert all(getattr(m, packing_buffer.PACKING_ATTR) is shared for m in modules)
    assert shared.tolist() == fresh.tolist()
    assert any("no longer shared" in r.getMessage() for r in caplog.records)


def test_a_new_batch_size_replaces_the_buffer(caplog):
    model = wrapped_model()
    two_rows, three_rows = packing_for([3, 5]), packing_for([3, 5, 2])
    assert (two_rows.numel(), three_rows.numel()) == (5, 7)

    packing_buffer.publish_packing(model, two_rows, 11, device=CPU, required=True)
    with caplog.at_level("WARNING"):
        packing_buffer.publish_packing(model, three_rows, 11, device=CPU, required=True)

    modules = consumers(model)
    shared = getattr(modules[0], packing_buffer.PACKING_ATTR)
    assert shared.numel() == 7
    assert all(getattr(m, packing_buffer.PACKING_ATTR) is shared for m in modules)
    assert shared.tolist() == three_rows.tolist()
    assert any("length changed" in r.getMessage() for r in caplog.records)


def test_publishing_without_consumers():
    """Publishing into the void would leave every layer deriving its own packing
    from the mask, so a caller that requires a consumer gets an error instead."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    built = packing_for([3, 5])
    with pytest.raises(RuntimeError, match="no attention module can read it"):
        packing_buffer.publish_packing(model, built, 11, device=CPU, required=True)
    assert packing_buffer.publish_packing(model, built, 11, device=CPU, required=False) is None


def test_clear_packing_removes_the_buffer_from_every_consumer():
    """Rule 4 in packing_buffer.py: anything that runs the model outside the
    adapter clears the packing first, which has to reach every consumer."""
    model = wrapped_model()
    packing_buffer.publish_packing(model, packing_for([3, 5]), 11, device=CPU, required=True)

    assert packing_buffer.clear_packing(model) == 3
    for module in consumers(model):
        assert not hasattr(module, packing_buffer.PACKING_ATTR)
        assert not hasattr(module, packing_buffer.BOUND_ATTR)
        assert not hasattr(module, packing_buffer.TOKENS_ATTR)


def test_a_same_shape_republish_reuses_the_buffer():
    """The steady state: one copy_ into the tensor already installed."""
    model = wrapped_model()
    first = packing_buffer.publish_packing(model, packing_for([3, 5]), 11, device=CPU, required=True)
    second = packing_buffer.publish_packing(model, packing_for([1, 7]), 11, device=CPU, required=True)
    assert second is first
    assert second.tolist() == packing_for([1, 7]).tolist()


def test_the_token_total_is_recorded_for_the_stale_packing_check():
    """B*S as a host integer, so the processor can compare it with its batch."""
    model = wrapped_model()
    packing_buffer.publish_packing(model, packing_for([3, 5]), 11, device=CPU, required=True)
    assert all(getattr(m, packing_buffer.TOKENS_ATTR) == 2 * 12 for m in consumers(model))


def test_a_module_that_gains_the_processor_later_is_included(caplog):
    """The consumers are found on every publish, so a processor swapped in after
    the first one still receives the shared buffer."""
    model = torch.nn.Sequential(FakeAttention(), FakeAttention())
    model[1].processor = object()
    packing_buffer.publish_packing(model, packing_for([3, 5]), 11, device=CPU, required=True)
    assert not hasattr(model[1], packing_buffer.PACKING_ATTR)

    model[1].processor = attn_processor.Ideogram4VarlenAttnProcessor()
    with caplog.at_level("WARNING"):
        packing_buffer.publish_packing(model, packing_for([1, 7]), 11, device=CPU, required=True)

    shared = getattr(model[0], packing_buffer.PACKING_ATTR)
    assert getattr(model[1], packing_buffer.PACKING_ATTR) is shared
    assert shared.tolist() == packing_for([1, 7]).tolist()
    assert any("gained the var-len processor" in r.getMessage() for r in caplog.records)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_an_unindexed_device_does_not_reinstall_every_step():
    """'cuda' and 'cuda:<current>' are the same place."""
    model = wrapped_model()
    first = packing_buffer.publish_packing(model, packing_for([3, 5]), 11, device="cuda", required=True)
    second = packing_buffer.publish_packing(model, packing_for([1, 7]), 11, device="cuda", required=True)
    assert second is first


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
