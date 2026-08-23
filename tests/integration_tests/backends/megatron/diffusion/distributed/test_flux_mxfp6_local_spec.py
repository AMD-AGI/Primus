# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Integration tests for Flux model with MXFP6 local spec.

Tests that a real (small) Flux model constructs, uses the MXFP6 linear types, and
produces valid output and gradients under PrimusTurboMXFP6LocalSpecProvider.

The MXFP4 sibling of this file uses batch_size=2, which MXFP6 cannot: MXFP6 needs
M, N and K all multiples of 256, and at IMG_SIZE_TINY the image stream is only 64
tokens, so batch_size=2 gives M=128. batch_size=4 lifts every stream over the bar --
image 4*64=256, text 4*128=512, and the concatenated single-block stream 4*192=768.
Unlike MXFP4 there is also no backend pinning fixture, because MXFP6 has a single
backend (AITER A6W6) and no preshuffle contract to satisfy.
"""

import pytest
import torch

from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
    MXFP6ColumnParallelLinear,
    MXFP6RowParallelLinear,
)
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from primus.backends.megatron.core.models.diffusion.flux.utils import (
    generate_image_position_ids,
    pack_latents,
)
from tests.unit_tests.backends.megatron.conftest import requires_mxfp6
from tests.unit_tests.backends.megatron.diffusion.constants import (
    CLIP_L_EMBEDDING_DIM,
    IMG_SIZE_TINY,
    T5_XXL_EMBEDDING_DIM,
    TEXT_SEQ_LEN_MEDIUM,
    VAE_LATENT_CHANNELS,
)
from tests.utils import PrimusUT

# Chosen so every MXFP6 GEMM sees M % 256 == 0; see the module docstring.
BATCH_SIZE = 4

SPEC_LINEAR_NAMES = {
    "linear_qkv",
    "added_linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
}


class TestFluxMXFP6LocalSpec(PrimusUT):
    """Integration tests for Flux with the MXFP6 local spec provider."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def _make_mxfp6_config(self, **overrides):
        defaults = dict(transformer_impl="local", fp6="mxfp6")
        defaults.update(overrides)
        return FluxConfig.flux_535m(**defaults)

    def _make_inputs(self, batch_size=BATCH_SIZE):
        height, width = IMG_SIZE_TINY, IMG_SIZE_TINY
        channels = VAE_LATENT_CHANNELS
        txt_seq_len = TEXT_SEQ_LEN_MEDIUM

        img = torch.randn(batch_size, channels, height, width, dtype=torch.bfloat16).cuda()
        txt = torch.randn(batch_size, txt_seq_len, T5_XXL_EMBEDDING_DIM, dtype=torch.bfloat16).cuda()
        y = torch.randn(batch_size, CLIP_L_EMBEDDING_DIM, dtype=torch.bfloat16).cuda()
        timesteps = torch.rand(batch_size, dtype=torch.bfloat16).cuda()

        packed_img = pack_latents(img).transpose(0, 1)
        txt_t = txt.transpose(0, 1)

        img_ids = generate_image_position_ids(batch_size, height, width, device="cuda")
        txt_ids = torch.zeros(batch_size, txt_seq_len, 3).cuda()

        return packed_img, txt_t, y, timesteps, img_ids, txt_ids

    @staticmethod
    def _activate_zero_init(model):
        """Break Flux's zero-init so the model is actually live.

        Flux deliberately zero-inits ``proj_out`` and the last linear of every
        ``adaLN_modulation`` (AdaLN-Zero), which makes a freshly built model output
        exactly zero and gives every transformer linear an exactly zero gradient. A
        forward/backward test on an unmodified model is therefore vacuous: it passes
        whether or not MXFP6 computes anything, which is why the assertions below run
        against an activated model and check for non-zero rather than non-None.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "adaLN_modulation" in name and param.dim() == 2:
                    torch.nn.init.normal_(param, std=0.02)
            torch.nn.init.normal_(model.proj_out.weight, std=0.02)

    def _mxfp6_linears(self, model):
        return [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, (MXFP6ColumnParallelLinear, MXFP6RowParallelLinear))
        ]

    def _assert_grads_nonzero(self, model, mode):
        linears = self._mxfp6_linears(model)
        assert linears, "No MXFP6 linears in the model"

        dead = [
            name for name, module in linears if module.weight.grad is None or not module.weight.grad.any()
        ]
        assert not dead, f"MXFP6 linears with missing or all-zero grad in {mode} mode: {dead}"

    @requires_mxfp6
    def test_flux_535m_mxfp6_constructs(self):
        model = Flux(self._make_mxfp6_config())
        assert isinstance(model, Flux)

    @requires_mxfp6
    def test_flux_535m_mxfp6_linear_types(self):
        """Every spec-provided linear must be an MXFP6 variant, not a bf16 fallback."""
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        model = Flux(self._make_mxfp6_config())

        found_any = False
        for name, module in model.named_modules():
            leaf_name = name.rsplit(".", 1)[-1] if "." in name else name
            if leaf_name not in SPEC_LINEAR_NAMES:
                continue
            found_any = True
            if isinstance(module, ColumnParallelLinear):
                assert isinstance(
                    module, MXFP6ColumnParallelLinear
                ), f"{name}: expected MXFP6ColumnParallelLinear, got {type(module).__name__}"
            if isinstance(module, RowParallelLinear):
                assert isinstance(
                    module, MXFP6RowParallelLinear
                ), f"{name}: expected MXFP6RowParallelLinear, got {type(module).__name__}"

        assert found_any, "No spec-provided linears found in model"

    @requires_mxfp6
    def test_flux_535m_is_inert_at_init(self):
        """Documents why the tests below activate the model first.

        If Flux ever stops zero-initialising its output head, this test fails and the
        ``_activate_zero_init`` workaround can be reconsidered.
        """
        model = Flux(self._make_mxfp6_config()).cuda().to(torch.bfloat16)
        model.train()

        output = model(*self._make_inputs())
        assert not output.any(), "Flux is no longer zero-init at the output head"

        output.sum().backward()
        assert all(
            module.weight.grad is None or not module.weight.grad.any()
            for _, module in self._mxfp6_linears(model)
        ), "gradients are non-zero at init, so the activation step may be unnecessary"

    @requires_mxfp6
    def test_flux_535m_mxfp6_forward_backward(self):
        model = Flux(self._make_mxfp6_config()).cuda().to(torch.bfloat16)
        model.train()
        self._activate_zero_init(model)

        output = model(*self._make_inputs())

        assert len(output.shape) == 3
        assert output.shape[1] == BATCH_SIZE
        assert output.any(), "Output is all zero even after activation"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        output.sum().backward()
        self._assert_grads_nonzero(model, "pure MXFP6")

    @requires_mxfp6
    def test_flux_535m_mxfp6_hybrid_forward_backward(self):
        """Hybrid backward (MXFP6 fwd / FP8 bwd) works through the full model."""
        config = self._make_mxfp6_config(mxfp6_backward_precision="fp8")
        model = Flux(config).cuda().to(torch.bfloat16)
        model.train()
        self._activate_zero_init(model)

        output = model(*self._make_inputs())

        assert output.any(), "Hybrid output is all zero even after activation"
        assert not torch.isnan(output).any(), "Hybrid output contains NaN"
        assert not torch.isinf(output).any(), "Hybrid output contains Inf"

        output.sum().backward()
        self._assert_grads_nonzero(model, "hybrid")

    @requires_mxfp6
    def test_flux_535m_mxfp6_differs_from_bf16_but_tracks_it(self):
        """MXFP6 must actually quantize, yet stay close to the bf16 model.

        Guards the two failure modes a construction-only test misses: silently
        falling back to bf16 (outputs identical) and a broken GEMM (outputs
        uncorrelated).
        """
        inputs = self._make_inputs()

        def build(**overrides):
            torch.manual_seed(1234)
            model = (
                Flux(FluxConfig.flux_535m(transformer_impl="local", **overrides)).cuda().to(torch.bfloat16)
            )
            torch.manual_seed(4321)
            self._activate_zero_init(model)
            return model

        with torch.no_grad():
            mxfp6_out = build(fp6="mxfp6")(*inputs).float()
            bf16_out = build()(*inputs).float()

        assert bf16_out.any(), "bf16 reference is all zero, so the comparison is vacuous"
        assert not torch.equal(
            mxfp6_out, bf16_out
        ), "MXFP6 output is bit-identical to bf16, so the MXFP6 path did not run"

        cos = torch.nn.functional.cosine_similarity(mxfp6_out.flatten(), bf16_out.flatten(), dim=0).item()
        assert cos > 0.99, f"MXFP6 output cosine similarity to bf16 is only {cos:.5f}"
