###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 packing transport (``ideogram4_packing_buffer``).

WHY THIS TEST EXISTS:
  The transport's whole value is an invariant that nothing else checks: **every attention
  module holds the SAME buffer object**, so the adapter's single ``copy_`` per step is visible
  to all 34 layers. Every way that invariant can lapse fails silently at runtime -- one layer
  reads the published packing and the rest read something else, so var-len flash attends
  across segment boundaries and training corrupts with no error raised anywhere. The two
  lapses that can actually happen in production are a batch-size change (buffer of the wrong
  shape) and a meta-device materialization pass (``to_empty()`` hands every module its own
  tensor), so both are exercised here directly.

  These tests need only ``torch`` -- no diffusers, no nemo_automodel, no GPU. The fake
  attention modules carry a real ``Ideogram4VarlenAttnProcessor`` because membership in the
  transport is decided by processor type; constructing one is cheap (aiter is imported lazily
  inside the attention call, not at class definition).
"""

import logging

import pytest

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - CPU-less lint environments
    pytest.skip("Ideogram-4 packing buffer tests require torch", allow_module_level=True)

from primus.backends.nemo_automodel import ideogram4_packing_buffer as transport
from primus.backends.nemo_automodel import ideogram4_varlen_attn as varlen_mod
from primus.backends.nemo_automodel.ideogram4_packing_buffer import (
    BOUND_ATTR,
    PACKING_ATTR,
    attention_modules,
    clear_packing,
    publish_packing,
    resolve_packing,
)
from primus.backends.nemo_automodel.ideogram4_varlen_attn import (
    Ideogram4VarlenAttnProcessor,
)


@pytest.fixture(autouse=True)
def _reset_log_dedupe():
    """The transport logs each condition once per process; tests assert on those logs."""
    transport._logged.clear()
    yield
    transport._logged.clear()


class _FakeAttention(nn.Module):
    """Stands in for ``Ideogram4Attention``: what matters is the processor it carries."""

    def __init__(self, varlen: bool = True):
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)
        self.processor = Ideogram4VarlenAttnProcessor() if varlen else object()


class _FakeBlock(nn.Module):
    def __init__(self, varlen: bool = True):
        super().__init__()
        self.attention = _FakeAttention(varlen)


class _FakeModel(nn.Module):
    """``model.layers`` of blocks, mirroring ``Ideogram4Transformer2DModel``'s shape."""

    def __init__(self, layers: int = 4, varlen: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(_FakeBlock(varlen) for _ in range(layers))

    @property
    def attentions(self):
        return [block.attention for block in self.layers]


def _packing(batch: int = 3, seq: int = 8, offset: int = 2) -> torch.Tensor:
    """A ``2B+1`` int32 packing shaped like ``build_cu_seqlens``' output."""
    starts = []
    for b in range(batch):
        starts += [b * seq, b * seq + offset]
    starts.append(batch * seq)
    return torch.tensor(starts, dtype=torch.int32)


def _buffers(model: _FakeModel):
    return [getattr(attn, PACKING_ATTR) for attn in model.attentions]


class TestPublishSharesOneObject:
    """The invariant the design rests on."""

    def test_all_modules_share_one_buffer(self):
        model = _FakeModel(layers=6)
        shared = publish_packing(model, _packing())

        buffers = _buffers(model)
        assert all(buf is shared for buf in buffers), (
            "attention modules do not share ONE buffer object, so the adapter's single copy_ "
            "would reach only some layers and the rest would attend on stale data"
        )
        assert shared.dtype == torch.int32, "flash_attn_varlen_func requires int32 cu_seqlens"

    def test_one_copy_is_visible_from_every_module(self):
        model = _FakeModel(layers=4)
        publish_packing(model, _packing(offset=2))

        second = _packing(offset=5)
        publish_packing(model, second)

        for attn in model.attentions:
            assert torch.equal(getattr(attn, PACKING_ATTR), second)

    def test_republish_mutates_in_place_rather_than_rebinding(self):
        """Rebinding per step is what makes a buffer look like a fresh input to Dynamo."""
        model = _FakeModel()
        first = publish_packing(model, _packing(offset=1))
        again = publish_packing(model, _packing(offset=3))
        assert again is first, "the buffer object must be stable across steps, only its values move"

    def test_max_seqlen_reaches_every_module(self):
        model = _FakeModel(layers=3)
        publish_packing(model, _packing(), max_seqlen=64)
        assert [getattr(a, BOUND_ATTR) for a in model.attentions] == [64, 64, 64]

    def test_buffer_is_non_persistent(self):
        """Derived per step: it must never enter a checkpoint."""
        model = _FakeModel()
        publish_packing(model, _packing())
        leaked = [k for k in model.state_dict() if PACKING_ATTR in k]
        assert not leaked, f"non-persistent buffer leaked into the state dict: {leaked}"

    def test_publishes_to_the_requested_device(self):
        model = _FakeModel()
        shared = publish_packing(model, _packing(), device=torch.device("cpu"))
        assert shared.device == torch.device("cpu")


class TestInvariantRepair:
    """Both production ways the sharing can lapse, and the recovery from each."""

    def test_shape_change_reinstalls_and_stays_shared(self):
        model = _FakeModel(layers=5)
        publish_packing(model, _packing(batch=3))

        wider = _packing(batch=8)
        shared = publish_packing(model, wider)

        assert shared.numel() == wider.numel()
        assert all(buf is shared for buf in _buffers(model))
        assert torch.equal(shared, wider)

    def test_shape_change_warns_because_it_forces_a_recompile(self, caplog):
        model = _FakeModel()
        publish_packing(model, _packing(batch=3))
        with caplog.at_level(logging.WARNING):
            publish_packing(model, _packing(batch=4))
        assert any("length changed" in r.message for r in caplog.records), (
            "a cu_seqlens length change silently recompiles the graph every time the batch "
            "size moves; it has to be visible in the log"
        )

    def test_materialization_that_unshares_buffers_is_repaired(self):
        """``to_empty()`` gives every module its own tensor. Measured, not hypothetical."""
        model = _FakeModel(layers=4)
        publish_packing(model, _packing())

        # Exactly what Module._apply / to_empty does: a fresh tensor per module.
        for attn in model.attentions:
            attn._buffers[PACKING_ATTR] = torch.zeros_like(getattr(attn, PACKING_ATTR))
        assert len({id(b) for b in _buffers(model)}) == 4, "fixture did not actually unshare"

        wanted = _packing(offset=4)
        shared = publish_packing(model, wanted)

        assert all(buf is shared for buf in _buffers(model)), "sharing was not restored"
        for attn in model.attentions:
            assert torch.equal(getattr(attn, PACKING_ATTR), wanted), (
                "a module still holds its own buffer, so it would attend on whatever "
                "materialization left there -- silently"
            )

    def test_unsharing_is_reported(self, caplog):
        model = _FakeModel(layers=2)
        publish_packing(model, _packing())
        for attn in model.attentions:
            attn._buffers[PACKING_ATTR] = torch.zeros_like(getattr(attn, PACKING_ATTR))
        with caplog.at_level(logging.WARNING):
            publish_packing(model, _packing())
        assert any("no longer shared" in r.message for r in caplog.records)


class TestSelfGating:
    """No var-len processor means no state to maintain."""

    def test_model_without_varlen_processors_gets_nothing(self):
        model = _FakeModel(varlen=False)
        assert attention_modules(model) == []
        assert publish_packing(model, _packing()) is None
        for attn in model.attentions:
            assert not hasattr(attn, PACKING_ATTR)

    def test_required_publish_raises_when_nothing_can_read_it(self):
        """The adapter publishes with ``required=True``, and this is why.

        Having built a packing, a model that cannot read it is a misconfiguration: every layer
        falls back to deriving its own from the mask, and if that happens on a SUBSET of ranks,
        data parallelism averages gradients from two different attention paths. No log would
        show it -- Primus quiets non-zero ranks once training starts -- so the only place this
        can surface is an exception.
        """
        model = _FakeModel(varlen=False)
        with pytest.raises(RuntimeError, match="no attention module can read it"):
            publish_packing(model, _packing(), required=True)

    def test_consumers_are_discovered_and_cached(self):
        model = _FakeModel(layers=7)
        assert len(attention_modules(model)) == 7
        assert attention_modules(model) is attention_modules(model), "consumer list should be cached"


class TestResolvePacking:
    """What the processor calls on every attention invocation."""

    def test_reads_the_buffer_off_the_module(self):
        model = _FakeModel()
        published = publish_packing(model, _packing(), max_seqlen=32)
        cu, bound = resolve_packing(model.attentions[0])
        assert cu is published and bound == 32

    def test_explicit_argument_wins(self):
        """Keeps a future kwargs route (patching the block forward) able to override."""
        model = _FakeModel()
        publish_packing(model, _packing(offset=2), max_seqlen=32)
        override = _packing(offset=6)
        cu, bound = resolve_packing(model.attentions[0], override, 99)
        assert cu is override and bound == 99

    def test_no_buffer_means_no_packing(self):
        """The processor must then fall through to its mask-derived legacy path."""
        model = _FakeModel()
        cu, bound = resolve_packing(model.attentions[0])
        assert cu is None and bound is None

    def test_clear_packing_restores_the_legacy_path(self):
        model = _FakeModel(layers=3)
        publish_packing(model, _packing())
        assert clear_packing(model) == 3
        cu, _ = resolve_packing(model.attentions[0])
        assert cu is None


class TestStalePackingGuard:
    """A packing published for a different batch size must fail loudly, not attend wrongly.

    The check is on ``numel()`` alone -- static shape metadata -- so it costs a guard and no
    host sync inside the compiled region, and it runs before any aiter kernel is touched
    (which is why this test needs no GPU).
    """

    @staticmethod
    def _qkv(batch: int, seq: int = 8, heads: int = 2, dim: int = 4):
        shape = (batch, seq, heads, dim)
        return torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)

    def test_rejects_a_packing_from_a_different_batch_size(self):
        proc = Ideogram4VarlenAttnProcessor()
        q, k, v = self._qkv(batch=2)
        stale = _packing(batch=5)  # published when B was 5
        with pytest.raises(ValueError, match="2\\*B\\+1"):
            proc._attention(q, k, v, None, stale, 8)

    def test_passes_a_matching_packing_through_to_the_kernel(self, monkeypatch):
        """The guard must not stand in the way of the path it protects.

        Stubs the aiter call so this runs on CPU, and checks the processor hands the kernel the
        packing it was given -- plus the STATIC ``max_seqlen``, not a value derived from data.
        """
        seen = {}

        def _fake_varlen(q, k, v, cu_seqlens, max_seqlen, deterministic=False):
            seen["cu_seqlens"] = cu_seqlens
            seen["max_seqlen"] = max_seqlen
            return torch.zeros_like(q)

        monkeypatch.setattr(varlen_mod, "varlen_flash_attention", _fake_varlen)

        proc = Ideogram4VarlenAttnProcessor()
        q, k, v = self._qkv(batch=3, seq=8, heads=2, dim=4)
        packing = _packing(batch=3, seq=8)

        out = proc._attention(q, k, v, None, packing, 8)

        assert out.shape == (3, 8, 2, 4), "the packed (B*L,H,D) result must be unpacked again"
        assert torch.equal(seen["cu_seqlens"], packing)
        assert seen["max_seqlen"] == 8


class TestKernelNeverSeesTheSharedBuffer:
    """The kernel gets a private copy, because aiter's var-len op WRITES its cu_seqlens.

    aiter saves the tensor for its backward and then mutates it, so every call bumps the
    autograd version counter. Handing 34 layers the same buffer moved its version 34 times per
    forward while each layer's backward still expected the version it saved, and the step died
    in ``.backward()`` ("IntTensor[5] is at version 35; expected version 34" -- measured on the
    first on-model compile run, 2026-08-04). The fake kernel below mutates its argument the way
    aiter does, so this test fails if the clone is ever dropped.
    """

    @staticmethod
    def _kernel(seen, monkeypatch):
        def _fake_varlen(q, k, v, cu_seqlens, max_seqlen, deterministic=False):
            seen.append(cu_seqlens)
            cu_seqlens.add_(0)  # aiter's in-place write, which bumps the version counter
            return torch.zeros_like(q)

        monkeypatch.setattr(varlen_mod, "varlen_flash_attention", _fake_varlen)

    def test_published_buffer_is_not_the_tensor_handed_to_the_kernel(self, monkeypatch):
        seen: list = []
        self._kernel(seen, monkeypatch)

        model = _FakeModel(layers=3)
        published = publish_packing(model, _packing(batch=3, seq=8), max_seqlen=8)
        version_before = published._version

        shape = (3, 8, 2, 4)
        for attn in model.attentions:
            cu, bound = resolve_packing(attn)
            attn.processor._attention(
                torch.zeros(shape), torch.zeros(shape), torch.zeros(shape), None, cu, bound
            )

        assert len(seen) == 3
        for passed in seen:
            assert passed is not published, (
                "the kernel was handed the SHARED buffer; its in-place write bumps the version "
                "counter every layer and the backward then rejects the packing it saved"
            )
            assert torch.equal(passed, published), "the private copy must carry the same packing"
        assert published._version == version_before, (
            "the shared buffer's autograd version moved during the forward, which is exactly "
            "what makes each layer's saved packing look stale at backward time"
        )


class TestWrappersAreNotCountedTwice:
    """Only modules that OWN a var-len processor are consumers.

    Activation checkpointing wraps each attention module in a wrapper that forwards unknown
    attributes to its child, so a plain ``getattr(module, "processor")`` answers for the wrapper
    too and every block is counted twice (68 for 34 blocks, seen in the first on-model run).
    The copy_ still reaches everyone, but the one health number in the log stops meaning
    anything -- "34 modules" and "half the model" become indistinguishable.
    """

    class _DelegatingWrapper(nn.Module):
        def __init__(self, wrapped: nn.Module):
            super().__init__()
            self._checkpoint_wrapped_module = wrapped

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._checkpoint_wrapped_module, name)

    def test_only_the_wrapped_module_is_a_consumer(self):
        model = _FakeModel(layers=4)
        for block in model.layers:
            block.attention = self._DelegatingWrapper(block.attention)

        modules = attention_modules(model)

        assert len(modules) == 4, f"expected one consumer per block, got {len(modules)}"
        assert all(not isinstance(m, self._DelegatingWrapper) for m in modules)
        # And the transport still works through the wrapper: the processor is called with the
        # real module, which is the object that holds the buffer.
        shared = publish_packing(model, _packing())
        for block in model.layers:
            cu, _ = resolve_packing(block.attention._checkpoint_wrapped_module)
            assert cu is shared


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
