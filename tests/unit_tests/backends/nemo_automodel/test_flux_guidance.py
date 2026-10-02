###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the FLUX.1 ``use_guidance_embeds`` repair.

AutoModel's FluxAdapter stores ``use_guidance_embeds`` and then always passes a
guidance tensor, so FLUX.1-schnell -- whose transformer has no guidance embedder --
dies at step 0 inside diffusers. These tests drive a tiny real
``FluxTransformer2DModel`` through the real adapter, because the failure is an
argument-count mismatch between the two and a stub of either would hide it.

CPU only; skipped where torch, diffusers or AutoModel are not importable.
"""

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
flux_adapters = pytest.importorskip("nemo_automodel.components.flow_matching.adapters.flux")
adapter_base = pytest.importorskip("nemo_automodel.components.flow_matching.adapters.base")

from primus.backends.nemo_automodel.models.flux import guidance  # noqa: E402

FluxAdapter = flux_adapters.FluxAdapter

# Latent channels C; the transformer sees 2x2-packed patches, so in_channels is 4*C.
LATENT_CHANNELS, LATENT_HW, BATCH, TEXT_LEN = 4, 8, 2, 6
JOINT_DIM, POOLED_DIM = 32, 16


def tiny_flux(guidance_embeds):
    torch.manual_seed(0)
    return diffusers.FluxTransformer2DModel(
        patch_size=1,
        in_channels=4 * LATENT_CHANNELS,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=JOINT_DIM,
        pooled_projection_dim=POOLED_DIM,
        guidance_embeds=guidance_embeds,
        axes_dims_rope=(4, 6, 6),
    )


def context():
    torch.manual_seed(1)
    latents = torch.randn(BATCH, LATENT_CHANNELS, LATENT_HW, LATENT_HW)
    return adapter_base.FlowMatchingContext(
        noisy_latents=latents,
        latents=latents,
        timesteps=torch.full((BATCH,), 500.0),
        sigma=torch.full((BATCH,), 0.5),
        task_type="t2v",
        data_type="image",
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch={
            "text_embeddings": torch.randn(BATCH, TEXT_LEN, JOINT_DIM),
            "pooled_prompt_embeds": torch.randn(BATCH, POOLED_DIM),
        },
    )


@pytest.fixture
def patched(monkeypatch):
    # Registered with monkeypatch first so the class is restored after each test.
    monkeypatch.setattr(FluxAdapter, "prepare_inputs", FluxAdapter.prepare_inputs)
    monkeypatch.setattr(FluxAdapter, "forward", FluxAdapter.forward)
    monkeypatch.setattr(FluxAdapter, guidance._PATCHED_ATTR, False, raising=False)
    assert guidance.install() is True


class TestPremise:
    def test_a_schnell_transformer_rejects_a_guidance_tensor(self):
        """The failure the repair exists for, reproduced without the adapter."""
        model = tiny_flux(guidance_embeds=False)
        inputs = FluxAdapter(guidance_scale=1.0).prepare_inputs(context())
        inputs.pop("_original_shape")
        with pytest.raises(TypeError, match="positional arguments"):
            model(**inputs, return_dict=False)


@pytest.mark.usefixtures("patched")
class TestRepair:
    def test_schnell_trains_a_step_with_guidance_embeds_off(self):
        model = tiny_flux(guidance_embeds=False)
        adapter = FluxAdapter(guidance_scale=1.0, use_guidance_embeds=False)
        inputs = adapter.prepare_inputs(context())
        assert inputs["guidance"] is None

        pred = adapter.forward(model, inputs)
        assert pred.shape == (BATCH, LATENT_CHANNELS, LATENT_HW, LATENT_HW)
        pred.float().pow(2).mean().backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_dev_is_unchanged(self):
        model = tiny_flux(guidance_embeds=True)
        adapter = FluxAdapter(guidance_scale=3.5, use_guidance_embeds=True)
        inputs = adapter.prepare_inputs(context())
        assert torch.equal(inputs["guidance"], torch.full((BATCH,), 3.5))
        assert adapter.forward(model, inputs).shape == (BATCH, LATENT_CHANNELS, LATENT_HW, LATENT_HW)

    @pytest.mark.parametrize(
        "model_has_embedder, use_guidance_embeds, fix",
        [(False, True, "false"), (True, False, "true")],
        ids=["schnell-with-guidance", "dev-without-guidance"],
    )
    def test_a_mismatch_names_the_config_key(self, model_has_embedder, use_guidance_embeds, fix):
        model = tiny_flux(guidance_embeds=model_has_embedder)
        adapter = FluxAdapter(guidance_scale=1.0, use_guidance_embeds=use_guidance_embeds)
        inputs = adapter.prepare_inputs(context())
        with pytest.raises(ValueError, match=rf"use_guidance_embeds: {fix}"):
            adapter.forward(model, inputs)

    def test_install_is_idempotent(self):
        prepare_inputs = FluxAdapter.prepare_inputs
        assert guidance.install() is True
        assert FluxAdapter.prepare_inputs is prepare_inputs


class TestRegistration:
    def test_the_patch_is_registered_ungated(self):
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        patch = next(
            (
                p
                for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")
                if p.id == "nemo_automodel.models.flux.guidance"
            ),
            None,
        )
        assert patch is not None
        assert patch.condition(None) is True
