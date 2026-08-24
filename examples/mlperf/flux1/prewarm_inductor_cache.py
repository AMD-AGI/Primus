#!/usr/bin/env python3
"""Prewarm the fixed-shape FLUX block cache on one MI355X GPU."""

import gc
import os

import torch

from primus.backends.diffusion.attention import set_attention_backend
from primus.backends.diffusion.models.flux.layers import EmbedND
from primus.backends.diffusion.models.registrations.flux import build_flux_model


def main() -> None:
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not cache_dir:
        raise ValueError("Set TORCHINDUCTOR_CACHE_DIR to an empty persistent directory")

    torch.manual_seed(0)
    set_attention_backend("flash_attn_aiter")
    model = build_flux_model(
        {
            "model_preset": "flux.1-schnell",
            "config": {
                "float8_recipe": "tensorwise",
                "float8_gemm_backend": "selective_flydsl",
            },
        }
    )
    double = model.dit.double_blocks[0]
    single = model.dit.single_blocks[0]
    del model
    gc.collect()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    compile_args = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
    }
    double.to(device=device, dtype=dtype).compile(**compile_args)
    single.to(device=device, dtype=dtype).compile(**compile_args)

    embedder = EmbedND(dim=128, theta=10000, axes_dim=[16, 56, 56]).to(device)
    pe = embedder(torch.zeros((32, 512, 3), device=device, dtype=dtype))
    vec = torch.randn((32, 3072), device=device, dtype=dtype)
    img = torch.randn((32, 256, 3072), device=device, dtype=dtype, requires_grad=True)
    txt = torch.randn((32, 256, 3072), device=device, dtype=dtype, requires_grad=True)

    img_out, txt_out = double(img, txt, vec, pe)
    (img_out.float().square().mean() + txt_out.float().square().mean()).backward()
    torch.cuda.synchronize()
    print("Prewarmed double block", flush=True)
    del img, txt, img_out, txt_out, double
    gc.collect()
    torch.cuda.empty_cache()

    x = torch.randn((32, 512, 3072), device=device, dtype=dtype, requires_grad=True)
    single(x, vec, pe).float().square().mean().backward()
    torch.cuda.synchronize()
    print("Prewarmed single block", flush=True)
    print(f"Cache ready: {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
