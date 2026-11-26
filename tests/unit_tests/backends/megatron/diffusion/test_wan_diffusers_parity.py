# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Forward parity between the WAN backbone and diffusers' ``WanTransformer3DModel``.

The layer-spec tests pin the state-dict keys and the converter tests pin the
tensor mapping, but neither runs a forward, so a wrong norm, RoPE layout, AdaLN
split or head fusion would still load cleanly. This builds a two-block model
with random weights in diffusers, converts it with the production converter,
and compares one forward on every attention path.

diffusers runs in fp32 as the reference. The bf16 Primus output must land as
close to it as diffusers' own bf16 forward does, give or take a small margin,
so the bound tracks the kernels rather than a hand-picked tolerance.

GPU-only, and needs diffusers plus the Megatron / TE / Primus-Turbo stack of
the ROCm training container.
"""

import argparse

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

diffusers = pytest.importorskip("diffusers")

from megatron.training.global_vars import set_args  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from primus.backends.megatron.core.models.diffusion.wan.checkpoint_converter import (  # noqa: E402
    convert_hf_to_primus,
    wan_hf_config_dict,
)
from primus.backends.megatron.core.models.diffusion.wan.config import (  # noqa: E402
    WanConfig,
)
from primus.backends.megatron.core.models.diffusion.wan.model import (  # noqa: E402
    WanTransformer3D,
    _load_backbone_checkpoint,
)

# head_dim 128 keeps the attention on the aiter kernels training runs.
TINY = dict(
    hidden_size=256,
    num_attention_heads=2,
    num_dit_layers=2,
    ffn_hidden_size=512,
    text_embed_dim=64,
)
LATENTS = (2, 16, 2, 16, 16)  # [B, C, T, H, W]: 128 tokens per sample
TEXT = (2, 24, TINY["text_embed_dim"])

PATHS = [
    pytest.param("transformer_engine", False, id="te_thd"),
    pytest.param("local", False, id="local_bshd"),
    pytest.param("local", True, id="local_thd"),
]


def _relative_error(actual, expected):
    return ((actual.float() - expected).norm() / expected.norm()).item()


@pytest.fixture(autouse=True)
def _parallel(init_parallel_state):
    set_args(argparse.Namespace(enable_turbo_attention_float8=False))


@pytest.fixture(scope="module")
def reference():
    """A random diffusers model, one set of inputs, and its fp32 and bf16 outputs."""
    torch.manual_seed(0)
    config = WanConfig.wan2_1_t2v_1_3b(**TINY)
    hf_config = {k: v for k, v in wan_hf_config_dict(config).items() if not k.startswith("_")}
    model = diffusers.WanTransformer3DModel(**hf_config).cuda().eval()
    state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    latents = torch.randn(LATENTS, device="cuda")
    text = torch.randn(TEXT, device="cuda")
    timestep = torch.tensor([150.0, 850.0], device="cuda")

    with torch.no_grad():
        fp32 = model(latents, timestep, text, return_dict=False)[0]
        bf16 = model.to(torch.bfloat16)(latents.bfloat16(), timestep, text.bfloat16(), return_dict=False)[0]

    return dict(
        state_dict=state_dict,
        inputs=(latents, timestep, text),
        fp32=fp32,
        bf16_error=_relative_error(bf16, fp32),
    )


@pytest.mark.parametrize("transformer_impl,local_thd", PATHS)
def test_forward_matches_diffusers(reference, transformer_impl, local_thd, tmp_path):
    """Convert, save, and load the way a run does, then compare one forward."""
    config = WanConfig.wan2_1_t2v_1_3b(
        **TINY, transformer_impl=transformer_impl, local_thd_attention=local_thd
    )
    primus_state_dict, _ = convert_hf_to_primus(reference["state_dict"], config, strict=True)
    ckpt = tmp_path / "primus_wan.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in primus_state_dict.items()}, str(ckpt))

    model = WanTransformer3D(config)
    _load_backbone_checkpoint(model, str(ckpt), subfolder=config.backbone_subfolder)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()

    latents, timestep, text = reference["inputs"]
    with torch.no_grad():
        out = model(latents.bfloat16(), timestep, text.bfloat16())

    assert out.shape == reference["fp32"].shape
    error = _relative_error(out, reference["fp32"])
    bound = 2.0 * reference["bf16_error"] + 1e-3
    assert (
        error <= bound
    ), f"{transformer_impl} (thd={local_thd}): {error:.2e} vs bf16 diffusers {reference['bf16_error']:.2e}"
