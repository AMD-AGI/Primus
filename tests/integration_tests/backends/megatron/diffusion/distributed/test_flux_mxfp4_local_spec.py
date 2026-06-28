# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Integration tests for Flux model with MXFP4 local spec.

Tests that a real (small) Flux model constructs, uses correct linear types,
and produces valid output+gradients when using PrimusTurboMXFP4LocalSpecProvider.
"""

import pytest
import torch

from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
    MXFP4ColumnParallelLinear,
    MXFP4RowParallelLinear,
)
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from primus.backends.megatron.core.models.diffusion.flux.utils import (
    generate_image_position_ids,
    pack_latents,
)
from tests.unit_tests.backends.megatron.conftest import requires_mxfp4
from tests.unit_tests.backends.megatron.diffusion.constants import (
    CLIP_L_EMBEDDING_DIM,
    IMG_SIZE_TINY,
    T5_XXL_EMBEDDING_DIM,
    TEXT_SEQ_LEN_MEDIUM,
    VAE_LATENT_CHANNELS,
)
from tests.utils import PrimusUT


class TestFluxMXFP4LocalSpec(PrimusUT):
    """Integration tests for Flux with MXFP4 local spec provider."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @pytest.fixture(autouse=True)
    def _pin_fp4_aiter(self, monkeypatch):
        """Pin the FP4 GEMM backend to AITER with autotune off.

        MXFP4 module ``__init__`` runs ``_assert_preshuffle_contract``, which
        requires the FP4 GEMM backend pinned to AITER with autotune off (the
        only config under which ``_enable_preshuffle()`` is True). Without this,
        Flux construction raises ``RuntimeError`` before the model is built.
        Mirrors ``_pin_fp4_aiter`` in the unit MXFP4 tests
        (test_primus_turbo_mxfp4_local.py); monkeypatch auto-restores on
        teardown. Also clears any baked-empty ``PRIMUS_TURBO_GEMM_BACKEND`` so
        the in-code pin is authoritative.

        Use a ``defaultdict(None)`` rather than a bare ``{FP4: AITER}`` so that
        the hybrid (FP4 fwd / FP8 bwd) test's FP8 lookup falls back to the
        default backend instead of raising ``KeyError`` -- ``get_gemm_backend``
        indexes ``_gemm_backend[precision]`` directly when it is non-None.
        """
        import collections
        import os

        from primus_turbo.pytorch.core.backend import (
            BackendType,
            GlobalBackendManager,
            PrecisionType,
        )

        if os.environ.get("PRIMUS_TURBO_GEMM_BACKEND", None) == "":
            monkeypatch.delenv("PRIMUS_TURBO_GEMM_BACKEND", raising=False)
        pinned = collections.defaultdict(lambda: None)
        if GlobalBackendManager._gemm_backend:
            pinned.update(GlobalBackendManager._gemm_backend)
        pinned[PrecisionType.FP4] = BackendType.AITER
        monkeypatch.setattr(GlobalBackendManager, "_gemm_backend", pinned)
        monkeypatch.setattr(GlobalBackendManager, "_auto_tune", False)

    def _make_mxfp4_config(self, **overrides):
        defaults = dict(
            transformer_impl="local",
            fp4="mxfp4",
            fp4_recipe="mxfp4",
        )
        defaults.update(overrides)
        return FluxConfig.flux_535m(**defaults)

    def _make_inputs(self, batch_size=2):
        height, width = IMG_SIZE_TINY, IMG_SIZE_TINY
        channels = VAE_LATENT_CHANNELS
        # FP4 GEMM requires M divisible by 16. TEXT_SEQ_LEN_SHORT=77 is not
        # aligned; use 128 so batch*seq = 256 (multiple of 16).
        txt_seq_len = TEXT_SEQ_LEN_MEDIUM

        img = torch.randn(batch_size, channels, height, width, dtype=torch.bfloat16).cuda()
        txt = torch.randn(batch_size, txt_seq_len, T5_XXL_EMBEDDING_DIM, dtype=torch.bfloat16).cuda()
        y = torch.randn(batch_size, CLIP_L_EMBEDDING_DIM, dtype=torch.bfloat16).cuda()
        timesteps = torch.rand(batch_size, dtype=torch.bfloat16).cuda()

        packed_img = pack_latents(img)
        packed_img = packed_img.transpose(0, 1)
        txt_t = txt.transpose(0, 1)

        img_ids = generate_image_position_ids(batch_size, height, width, device="cuda")
        txt_ids = torch.zeros(batch_size, txt_seq_len, 3).cuda()

        return packed_img, txt_t, y, timesteps, img_ids, txt_ids

    @requires_mxfp4
    def test_flux_535m_mxfp4_constructs(self):
        config = self._make_mxfp4_config()
        model = Flux(config)
        assert model is not None
        assert isinstance(model, Flux)

    @requires_mxfp4
    def test_flux_535m_mxfp4_linear_types(self):
        """Verify spec-provided linears are MXFP4 variants."""
        config = self._make_mxfp4_config()
        model = Flux(config)

        spec_linear_names = {
            "linear_qkv",
            "added_linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
        }

        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        found_any = False
        for name, module in model.named_modules():
            leaf_name = name.rsplit(".", 1)[-1] if "." in name else name
            if leaf_name not in spec_linear_names:
                continue
            found_any = True
            if isinstance(module, ColumnParallelLinear):
                assert isinstance(
                    module, MXFP4ColumnParallelLinear
                ), f"{name}: expected MXFP4ColumnParallelLinear, got {type(module).__name__}"
            if isinstance(module, RowParallelLinear):
                assert isinstance(
                    module, MXFP4RowParallelLinear
                ), f"{name}: expected MXFP4RowParallelLinear, got {type(module).__name__}"

        assert found_any, "No spec-provided linears found in model"

    @requires_mxfp4
    def test_flux_535m_mxfp4_forward_backward(self):
        """Forward+backward pass produces valid output and gradient flow."""
        config = self._make_mxfp4_config()
        model = Flux(config).cuda().to(torch.bfloat16)
        model.train()

        inputs = self._make_inputs(batch_size=2)

        output = model(*inputs)

        assert output is not None
        assert len(output.shape) == 3
        assert output.shape[1] == 2
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        output.sum().backward()

        has_mxfp4_grad = False
        for name, module in model.named_modules():
            if isinstance(module, (MXFP4ColumnParallelLinear, MXFP4RowParallelLinear)):
                if module.weight.grad is not None:
                    has_mxfp4_grad = True
                    break

        assert has_mxfp4_grad, "No MXFP4 linear has a weight gradient -- gradient did not flow"

    @requires_mxfp4
    def test_flux_535m_mxfp4_hybrid_forward_backward(self):
        """Hybrid backward (FP4 fwd / FP8 bwd) works through the full model."""
        config = self._make_mxfp4_config(mxfp4_backward_precision="fp8")
        model = Flux(config).cuda().to(torch.bfloat16)
        model.train()

        inputs = self._make_inputs(batch_size=2)

        output = model(*inputs)

        assert output is not None
        assert not torch.isnan(output).any(), "Hybrid output contains NaN"
        assert not torch.isinf(output).any(), "Hybrid output contains Inf"

        output.sum().backward()

        has_mxfp4_grad = False
        for name, module in model.named_modules():
            if isinstance(module, (MXFP4ColumnParallelLinear, MXFP4RowParallelLinear)):
                if module.weight.grad is not None:
                    has_mxfp4_grad = True
                    break

        assert has_mxfp4_grad, "No MXFP4 linear has a weight gradient in hybrid mode"
