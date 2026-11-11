###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the FLUX block patches that need no kernel of their own: un-aliased single-stream
outputs and the image-only last block.

WHAT THESE ARE DEFENDING:
  ``block_outputs.py`` must return the same values as diffusers' block, as fresh tensors rather
  than views of one, and must pass through outputs another patch already made fresh (a clone of an
  opaque op's output would be a real copy).

  ``last_block.py``'s forward drops the text tokens' MLP and ``proj_out``: its image tokens must
  match diffusers' block, and its text output must be the incoming text tokens, detached, since
  nothing reads it.

A small block on the CPU in FP32: the patches change which tensors exist, not the arithmetic. The
class-level patches are undone after each test.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
transformer_flux = pytest.importorskip("diffusers.models.transformers.transformer_flux")

DIM, HEADS, HEAD_DIM, S_TXT, S_IMG, BATCH = 64, 2, 32, 8, 16, 2


@pytest.fixture
def block_cls(monkeypatch):
    """FluxSingleTransformerBlock, with every attribute the patches set restored afterwards."""
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


class TestImageOnlyLastBlock:
    def test_matches_diffusers_on_the_image_tokens(self, block_cls):
        from primus.backends.automodel.models.flux import last_block

        block = _block(block_cls)
        hidden, encoder, temb, rope = _inputs()
        with torch.no_grad():
            _, ref_img = block(hidden, encoder, temb, rope)
            txt, img = last_block._image_only_forward(block, hidden, encoder, temb, rope)
        torch.testing.assert_close(img, ref_img, rtol=1e-5, atol=1e-5)
        assert torch.equal(txt, encoder) and not txt.requires_grad

    def test_text_tokens_get_their_attention_gradient(self, block_cls):
        """The text tokens still feed the attention's keys and values, so they keep that gradient."""
        from primus.backends.automodel.models.flux import last_block

        block = _block(block_cls)
        hidden, encoder, temb, rope = _inputs()
        enc_ref, enc = encoder.clone().requires_grad_(), encoder.clone().requires_grad_()
        _, ref_img = block(hidden, enc_ref, temb, rope)
        ref_img.sum().backward()
        _, img = last_block._image_only_forward(block, hidden, enc, temb, rope)
        img.sum().backward()
        torch.testing.assert_close(enc.grad, enc_ref.grad, rtol=1e-5, atol=1e-5)
