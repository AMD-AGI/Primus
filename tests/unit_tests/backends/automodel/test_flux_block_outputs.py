###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for ``models/flux/block_outputs.py``: FLUX single-stream blocks return fresh tensors.

WHAT THESE ARE DEFENDING:
  ``block_outputs.py`` must return the same values as diffusers' block, as fresh tensors rather
  than views of one, and must pass through outputs another patch already made fresh (a clone of an
  opaque op's output would be a real copy).

A small block on the CPU in FP32: the patch changes which tensors exist, not the arithmetic. The
class-level patches are undone after each test.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
transformer_flux = pytest.importorskip("diffusers.models.transformers.transformer_flux")

DIM, HEADS, HEAD_DIM, S_TXT, S_IMG, BATCH = 64, 2, 32, 8, 16, 2


@pytest.fixture
def block_cls(monkeypatch):
    """FluxSingleTransformerBlock, with every attribute the patch sets restored afterwards."""
    from primus.backends.automodel.models.flux import block_outputs

    cls = transformer_flux.FluxSingleTransformerBlock
    monkeypatch.setattr(cls, "forward", cls.forward)
    for attr in (block_outputs._PATCHED_ATTR, block_outputs.FRESH_OUTPUTS_ATTR):
        monkeypatch.setattr(cls, attr, False, raising=False)
    return cls


def _inputs():
    gen = torch.Generator().manual_seed(0)
    hidden = torch.randn(BATCH, S_IMG, DIM, generator=gen)
    encoder = torch.randn(BATCH, S_TXT, DIM, generator=gen)
    temb = torch.randn(BATCH, DIM, generator=gen)
    ids = torch.zeros(S_TXT + S_IMG, 3)
    ids[S_TXT:, 1] = torch.arange(S_IMG) // 4
    ids[S_TXT:, 2] = torch.arange(S_IMG) % 4
    cos, sin = transformer_flux.FluxPosEmbed(theta=10000, axes_dim=[8, 12, 12])(ids)
    return hidden, encoder, temb, (cos.float(), sin.float())


def _block(cls):
    torch.manual_seed(0)
    return cls(dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM).eval()


class TestUnaliasedOutputs:
    def test_fresh_tensors_with_the_same_values(self, block_cls):
        from primus.backends.automodel.models.flux import block_outputs

        block = _block(block_cls)
        hidden, encoder, temb, rope = _inputs()
        with torch.no_grad():
            ref_txt, ref_img = block(hidden, encoder, temb, rope)
            assert ref_txt._base is not None  # diffusers returns two views of one tensor
            assert block_outputs.install()
            txt, img = block(hidden, encoder, temb, rope)
        assert txt._base is None and img._base is None
        assert torch.equal(txt, ref_txt) and torch.equal(img, ref_img)

    def test_passes_fresh_outputs_through(self, block_cls, monkeypatch):
        from primus.backends.automodel.models.flux import block_outputs

        fresh = (torch.ones(1, S_TXT, DIM), torch.ones(1, S_IMG, DIM))
        monkeypatch.setattr(block_cls, "forward", lambda self, *a, **k: fresh)
        monkeypatch.setattr(block_cls, block_outputs.FRESH_OUTPUTS_ATTR, True, raising=False)
        assert block_outputs.install()
        txt, img = _block(block_cls)(None, None, None)
        assert txt is fresh[0] and img is fresh[1]
