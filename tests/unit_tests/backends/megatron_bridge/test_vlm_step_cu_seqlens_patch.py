###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the vlm_step packed cu_seqlens patch."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PACKED_LEN = 2032
PADDED_LEN = 2048
SEQ_LENGTH = 4096


@pytest.fixture
def patch_mod():
    import primus.backends.megatron_bridge.patches.vlm_step_cu_seqlens_patches as m

    return m


def _packed_batch(seq_lens=(1000, 700, 332), pad_cu_seqlens_to=None):
    """Mimic the packed SFT collate output for micro_batch_size=1."""
    total = sum(seq_lens)
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + n)
    if pad_cu_seqlens_to is not None:
        cu.extend([total] * (pad_cu_seqlens_to - len(cu)))
    cu.append(-1)
    cu_seqlens = torch.IntTensor([cu])
    loss_mask = torch.ones(1, total, dtype=torch.long)
    return {
        "tokens": torch.arange(total).unsqueeze(0),
        "labels": torch.arange(1, total + 1).unsqueeze(0),
        "loss_mask": loss_mask,
        "position_ids": torch.cat([torch.arange(n) for n in seq_lens]).unsqueeze(0),
        "attention_mask": None,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_argmin": torch.argmin(cu_seqlens, dim=1, keepdim=True),
        "max_seqlen": torch.IntTensor([[max(seq_lens)]]),
        "visual_inputs": None,
    }


def _segment_sum(cu_seqlens, cu_seqlens_argmin):
    valid = cu_seqlens[0, : int(cu_seqlens_argmin.item())]
    return int((valid[1:] - valid[:-1]).sum()), valid


def _pad(x, n, value):
    return torch.nn.functional.pad(x, (0, n), value=value)


def _fake_upstream_get_batch(data_iterator, cfg, use_mtp=False, *, pg_collection):
    """Reproduce the no-PP padding of the vendored vlm_step.get_batch."""
    batch = next(data_iterator)
    n = PADDED_LEN - batch["tokens"].size(1)
    return (
        _pad(batch["tokens"], n, 0),
        _pad(batch["labels"], n, -100),
        _pad(batch["loss_mask"], n, 0),
        None,
        _pad(batch["position_ids"], n, 0),
        batch["cu_seqlens"],
        batch["cu_seqlens_argmin"],
        batch["max_seqlen"],
        None,
    )


class TestExtendCuSeqlens:
    def test_appends_padding_segment(self, patch_mod):
        b = _packed_batch()
        cu, argmin, max_seqlen = patch_mod.extend_cu_seqlens_to(
            b["cu_seqlens"], b["cu_seqlens_argmin"], b["max_seqlen"], PADDED_LEN
        )
        total, valid = _segment_sum(cu, argmin)
        assert total == PADDED_LEN
        assert valid.tolist() == [0, 1000, 1700, PACKED_LEN, PADDED_LEN]
        assert cu.shape == b["cu_seqlens"].shape
        assert int(max_seqlen.item()) == 1000
        assert int(b["cu_seqlens_argmin"].item()) == 4, "input must not be modified in place"

    def test_no_free_slot_grows_tensor(self, patch_mod):
        b = _packed_batch()
        cu = b["cu_seqlens"][:, :-1]
        cu, argmin, _ = patch_mod.extend_cu_seqlens_to(cu, torch.tensor([[4]]), None, PADDED_LEN)
        assert _segment_sum(cu, argmin)[0] == PADDED_LEN

    def test_pad_cu_seqlens_zero_length_segments(self, patch_mod):
        b = _packed_batch(pad_cu_seqlens_to=8)
        cu, argmin, _ = patch_mod.extend_cu_seqlens_to(
            b["cu_seqlens"], b["cu_seqlens_argmin"], b["max_seqlen"], PADDED_LEN
        )
        assert _segment_sum(cu, argmin)[0] == PADDED_LEN
        assert cu.shape == b["cu_seqlens"].shape

    def test_max_seqlen_covers_padding_segment(self, patch_mod):
        b = _packed_batch(seq_lens=(10, 10))
        _, _, max_seqlen = patch_mod.extend_cu_seqlens_to(
            b["cu_seqlens"], b["cu_seqlens_argmin"], b["max_seqlen"], 128
        )
        assert int(max_seqlen.item()) == 108

    def test_noop_without_padding(self, patch_mod):
        b = _packed_batch()
        out = patch_mod.extend_cu_seqlens_to(
            b["cu_seqlens"], b["cu_seqlens_argmin"], b["max_seqlen"], PACKED_LEN
        )
        assert out[0] is b["cu_seqlens"]

    def test_idempotent(self, patch_mod):
        b = _packed_batch()
        once = patch_mod.extend_cu_seqlens_to(
            b["cu_seqlens"], b["cu_seqlens_argmin"], b["max_seqlen"], PADDED_LEN
        )
        twice = patch_mod.extend_cu_seqlens_to(*once, PADDED_LEN)
        assert twice[0] is once[0]

    def test_without_argmin_or_sentinel(self, patch_mod):
        cu, argmin, _ = patch_mod.extend_cu_seqlens_to(
            torch.IntTensor([[0, 1000, PACKED_LEN]]), None, None, PADDED_LEN
        )
        assert argmin is None
        assert cu.tolist() == [[0, 1000, PACKED_LEN, PADDED_LEN]]


class TestWrappedGetBatch:
    def test_padded_tokens_match_cu_seqlens(self, patch_mod, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "megatron.core.utils", SimpleNamespace(get_pg_size=lambda group=None: 1)
        )
        get_batch = patch_mod.wrap_vlm_get_batch(_fake_upstream_get_batch)
        tokens, _, loss_mask, _, position_ids, cu, argmin, _, _ = get_batch(
            iter([_packed_batch()]), cfg=None, pg_collection=SimpleNamespace(cp=None)
        )
        assert tokens.size(1) == PADDED_LEN == position_ids.size(1)
        assert _segment_sum(cu, argmin)[0] == tokens.size(1)
        assert loss_mask[0, PACKED_LEN:].sum() == 0
        assert loss_mask[0, :PACKED_LEN].sum() == PACKED_LEN

    def test_skipped_under_context_parallel(self, patch_mod, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "megatron.core.utils", SimpleNamespace(get_pg_size=lambda group=None: 2)
        )
        warnings = []
        monkeypatch.setattr(patch_mod, "warning_rank_0", warnings.append)
        get_batch = patch_mod.wrap_vlm_get_batch(_fake_upstream_get_batch)
        b = _packed_batch()
        outputs = get_batch(iter([b]), cfg=None, pg_collection=SimpleNamespace(cp=object()))
        get_batch(iter([_packed_batch()]), cfg=None, pg_collection=SimpleNamespace(cp=object()))
        assert outputs[5] is b["cu_seqlens"]
        assert len(warnings) == 1


@pytest.fixture
def upstream_vlm_step(monkeypatch):
    bridge_src = Path(__file__).resolve().parents[4] / "third_party" / "Megatron-Bridge" / "src"
    monkeypatch.syspath_prepend(str(bridge_src))
    try:
        import megatron.bridge.training.vlm_step as vlm_step
    except Exception as e:  # Megatron-Bridge deps (TE, transformers, ...) unavailable
        pytest.skip(f"megatron.bridge.training.vlm_step not importable: {e}")
    return vlm_step


class TestUpstreamVlmStep:
    def test_condition_matches_vendored_vlm_step(self, patch_mod, upstream_vlm_step):
        assert patch_mod._vlm_step_pads_without_cu_seqlens(None)

    def test_vendored_get_batch_is_fixed(self, patch_mod, upstream_vlm_step, monkeypatch):
        monkeypatch.setattr(upstream_vlm_step, "get_batch_from_iterator", lambda it, *a, **kw: dict(next(it)))
        cfg = SimpleNamespace(dataset=SimpleNamespace(), model=SimpleNamespace(seq_length=SEQ_LENGTH))
        pg = SimpleNamespace(pp=None, cp=None)

        unpatched = upstream_vlm_step.get_batch(iter([_packed_batch()]), cfg, pg_collection=pg)
        assert unpatched[0].size(1) == PADDED_LEN
        assert _segment_sum(unpatched[5], unpatched[6])[0] == PACKED_LEN

        get_batch = patch_mod.wrap_vlm_get_batch(upstream_vlm_step.get_batch)
        tokens, labels, loss_mask, _, _, cu, argmin, _, _ = get_batch(
            iter([_packed_batch()]), cfg, pg_collection=pg
        )
        assert _segment_sum(cu, argmin)[0] == tokens.size(1) == PADDED_LEN
        assert loss_mask[0, PACKED_LEN:].sum() == 0
        assert (labels[0, PACKED_LEN:] == -100).all()
