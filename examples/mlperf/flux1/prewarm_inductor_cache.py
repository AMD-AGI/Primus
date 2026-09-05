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
                "float8_gemm_backend": os.environ.get("FLUX_FP8_GEMM_BACKEND", "selective_flydsl"),
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
    x = torch.randn((32, 512, 3072), device=device, dtype=dtype, requires_grad=True)
    single(x, vec, pe).float().square().mean().backward()
    torch.cuda.synchronize()
    print("Prewarmed batch-32 training blocks", flush=True)
    del img, txt, img_out, txt_out, x, pe, vec
    double.zero_grad(set_to_none=True)
    single.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "32"))
    eval_batch_sizes = (eval_batch_size,) if eval_batch_size == 32 else (eval_batch_size, 32)
    double.eval()
    single.eval()
    with torch.no_grad():
        for batch_size in eval_batch_sizes:
            pe = embedder(torch.zeros((batch_size, 512, 3), device=device, dtype=dtype))
            vec = torch.randn((batch_size, 3072), device=device, dtype=dtype)
            img = torch.randn((batch_size, 256, 3072), device=device, dtype=dtype)
            txt = torch.randn((batch_size, 256, 3072), device=device, dtype=dtype)
            double(img, txt, vec, pe)
            x = torch.randn((batch_size, 512, 3072), device=device, dtype=dtype)
            single(x, vec, pe)
            torch.cuda.synchronize()
            print(f"Prewarmed batch-{batch_size} evaluation blocks", flush=True)
            del img, txt, x, pe, vec

    print(f"Cache ready: {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
