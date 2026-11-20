###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for ``models/flux/block_outputs.py``: FLUX single-stream blocks return fresh tensors.

WHAT THESE ARE DEFENDING:
  ``block_outputs.py`` must return the same values and gradients as diffusers' block, as fresh
  tensors rather than views of one, also when the block is compiled, where AOTAutograd otherwise
  hands the views back. It must wrap the forward once, and leave the block alone when the class is
  not importable or its forward does not return the text and image pair. Its patch must apply only
  with its flag set, and not to another configured transformer.

A small block on the CPU in FP32: the patch changes which tensors exist, not the arithmetic. The
class-level patches are undone after each test.
"""

import inspect
import logging
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
transformer_flux = pytest.importorskip("diffusers.models.transformers.transformer_flux")

if (
    "encoder_hidden_states"
    not in inspect.signature(transformer_flux.FluxSingleTransformerBlock.forward).parameters
):
    pytest.skip(
        "this diffusers' single-stream block returns one tensor, not the text and image pair",
        allow_module_level=True,
    )

from primus.core.patches import PatchContext  # noqa: E402

DIM, HEADS, HEAD_DIM, S_TXT, S_IMG, BATCH = 64, 2, 32, 8, 16, 2
LOGGER = "primus.backends.automodel.models.flux.block_outputs"
MODULE_PATH = "diffusers.models.transformers.transformer_flux"
PATCH_NAME = "automodel.models.flux.single_block_fresh_outputs"
FLAG = "primus_perf.flux_single_block_fresh_outputs"


@pytest.fixture
def block_cls(monkeypatch):
    """FluxSingleTransformerBlock, with every attribute the patch sets restored afterwards."""
    from primus.backends.automodel.models.flux import block_outputs

    cls = transformer_flux.FluxSingleTransformerBlock
    monkeypatch.setattr(cls, "forward", cls.forward)
    monkeypatch.setattr(cls, block_outputs._PATCHED_ATTR, False, raising=False)
    return cls


def _inputs(requires_grad=False):
    gen = torch.Generator().manual_seed(0)
    hidden = torch.randn(BATCH, S_IMG, DIM, generator=gen, requires_grad=requires_grad)
    encoder = torch.randn(BATCH, S_TXT, DIM, generator=gen, requires_grad=requires_grad)
    temb = torch.randn(BATCH, DIM, generator=gen)
    ids = torch.zeros(S_TXT + S_IMG, 3)
    ids[S_TXT:, 1] = torch.arange(S_IMG) // 4
    ids[S_TXT:, 2] = torch.arange(S_IMG) % 4
    cos, sin = transformer_flux.FluxPosEmbed(theta=10000, axes_dim=[8, 12, 12])(ids)
    return hidden, encoder, temb, (cos.float(), sin.float())


def _block(cls):
    torch.manual_seed(0)
    return cls(dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM).eval()


def _run_with_grads(block):
    """The block's outputs and the gradients of its parameters and inputs, for fixed cotangents."""
    hidden, encoder, temb, rope = _inputs(requires_grad=True)
    txt, img = block(hidden, encoder, temb, rope)
    gen = torch.Generator().manual_seed(1)
    loss = (txt * torch.randn(txt.shape, generator=gen)).sum() + (
        img * torch.randn(img.shape, generator=gen)
    ).sum()
    loss.backward()
    grads = [hidden.grad, encoder.grad] + [p.grad for p in block.parameters()]
    return txt, img, grads


class TestFreshOutputs:
    def test_fresh_tensors_with_the_same_values(self, block_cls):
        from primus.backends.automodel.models.flux import block_outputs

        block = _block(block_cls)
        hidden, encoder, temb, rope = _inputs()
        with torch.no_grad():
            ref_txt, ref_img = block(hidden, encoder, temb, rope)
            assert ref_txt._base is not None  # diffusers returns two views of one tensor
            assert block_outputs.install() is True
            txt, img = block(hidden, encoder, temb, rope)
        assert txt._base is None and img._base is None
        assert torch.equal(txt, ref_txt) and torch.equal(img, ref_img)

    def test_the_same_gradients(self, block_cls):
        from primus.backends.automodel.models.flux import block_outputs

        _, _, ref_grads = _run_with_grads(_block(block_cls))
        assert block_outputs.install() is True
        _, _, grads = _run_with_grads(_block(block_cls))
        assert all(torch.equal(g, r) for g, r in zip(grads, ref_grads, strict=True))

    def test_a_compiled_block_returns_fresh_tensors(self, block_cls):
        from primus.backends.automodel.models.flux import block_outputs

        torch._dynamo.reset()
        ref = _block(block_cls)
        ref.compile(backend="aot_eager")
        ref_txt, ref_img, ref_grads = _run_with_grads(ref)
        assert ref_txt._base is not None  # AOTAutograd hands back views of the intermediate

        assert block_outputs.install() is True
        torch._dynamo.reset()
        block = _block(block_cls)
        block.compile(backend="aot_eager")
        txt, img, grads = _run_with_grads(block)
        torch._dynamo.reset()
        assert txt._base is None and img._base is None
        assert torch.equal(txt, ref_txt) and torch.equal(img, ref_img)
        assert all(torch.equal(g, r) for g, r in zip(grads, ref_grads, strict=True))

    def test_install_is_idempotent(self, block_cls):
        from primus.backends.automodel.models.flux import block_outputs

        assert block_outputs.install() is True
        wrapped = block_cls.forward
        assert block_outputs.install() is True
        assert block_cls.forward is wrapped


class TestDeclines:
    def test_a_missing_block_is_left_alone(self, block_cls, monkeypatch, caplog):
        from primus.backends.automodel.models.flux import block_outputs

        forward = block_cls.forward
        monkeypatch.setitem(sys.modules, MODULE_PATH, None)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert block_outputs.install() is False
        assert block_cls.forward is forward
        assert "FluxSingleTransformerBlock is not importable" in caplog.text

    def test_a_single_tensor_forward_is_left_alone(self, block_cls, monkeypatch, caplog):
        from primus.backends.automodel.models.flux import block_outputs

        def single_tensor_forward(self, hidden_states, temb, image_rotary_emb=None):
            return hidden_states

        monkeypatch.setattr(block_cls, "forward", single_tensor_forward)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert block_outputs.install() is False
        assert block_cls.forward is single_tensor_forward
        assert not getattr(block_cls, block_outputs._PATCHED_ATTR)
        assert "takes no encoder_hidden_states" in caplog.text


def _ctx(transformer_cls=None):
    params = SimpleNamespace(
        model=SimpleNamespace(pipeline_spec=SimpleNamespace(transformer_cls=transformer_cls))
    )
    return PatchContext(
        backend="automodel",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=params), "backend_args": params},
    )


class TestPatchCondition:
    def _condition(self):
        from primus.core.patches.patch_registry import PatchRegistry

        patch = PatchRegistry.get(PATCH_NAME)
        assert patch is not None
        return patch.condition

    def test_off_by_default(self):
        assert self._condition()(_ctx("FluxTransformer2DModel")) is False

    def test_applies_to_flux_with_the_flag(self, set_option):
        set_option(FLAG, True)
        assert self._condition()(_ctx("FluxTransformer2DModel")) is True
        assert (
            self._condition()(_ctx()) is True
        ), "a config that names no transformer, as with from_pretrained"

    def test_declines_another_transformer(self, set_option):
        set_option(FLAG, True)
        assert self._condition()(_ctx("WanTransformer3DModel")) is False

    def test_apply_wraps_the_forward(self, block_cls, set_option):
        from primus.backends.automodel.patches.flux_block_outputs_patches import apply

        forward = block_cls.forward
        set_option(FLAG, True)
        apply(_ctx("FluxTransformer2DModel"))
        assert block_cls.forward is not forward
        assert block_cls.forward.__wrapped__ is forward
