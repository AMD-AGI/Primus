###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Build the Ideogram-4 pre-encoded cache that ``cache.py`` reads.

Pairs each image with its caption, encodes both with :class:`Ideogram4Processor`
(Flux-2 VAE for the latents, Qwen3-VL for the text features) and writes the flat
layout the dataloader expects:

    <out_dir>/metadata.json      index + grid/feature dims
    <out_dir>/samples/<i>.pt     {image_latents, llm_features, text_length, ...}

Encoding is a one-off offline step: it needs one GPU and the gated VAE and
text-encoder weights, neither of which training itself requires.

Selection is **deterministic** by default — images are taken in sorted order and
samples that cannot be encoded are skipped, not substituted — so the same source
directory and arguments reproduce the same cache. ``shuffle`` opts into a
seeded shuffle instead.

Reached from the CLI as ``primus data automodel-cache --model ideogram4``.
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

DEFAULT_VAE_SOURCE = "ideogram-ai/ideogram-4-nf4-diffusers"
DEFAULT_TEXT_ENCODER_SOURCE = "Qwen/Qwen3-VL-8B-Instruct"

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_cache(
    *,
    image_dir: str,
    caption_dir: str,
    output_dir: str,
    num_samples: int = 1024,
    resolution: int = 256,
    max_text_tokens: int = 128,
    vae_source: str = DEFAULT_VAE_SOURCE,
    text_encoder_source: str = DEFAULT_TEXT_ENCODER_SOURCE,
    tokenizer_source: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "bf16",
    seed: int = 1234,
    shuffle: bool = False,
    log_every: int = 64,
) -> Dict[str, Any]:
    """Encode up to ``num_samples`` image/caption pairs into a flat cache.

    Captions are read from ``<caption_dir>/<image stem>.txt``. Returns the
    metadata dict that was written to ``metadata.json``.
    """
    from primus.backends.nemo_automodel.models.ideogram4.processor import (
        Ideogram4Processor,
    )

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")

    image_path_root = Path(image_dir)
    caption_path_root = Path(caption_dir)
    out_root = Path(output_dir)
    for label, path in (("image_dir", image_path_root), ("caption_dir", caption_path_root)):
        if not path.is_dir():
            raise NotADirectoryError(f"{label} does not exist: {path}")
    (out_root / "samples").mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_path_root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if shuffle:
        random.Random(seed).shuffle(images)
    if not images:
        raise FileNotFoundError(f"no images with extensions {IMAGE_EXTENSIONS} under {image_path_root}")
    logger.info("Found %d images in %s", len(images), image_path_root)

    processor = Ideogram4Processor(
        vae_source=vae_source,
        text_encoder_source=text_encoder_source,
        tokenizer_source=tokenizer_source,
        device=device,
        dtype=DTYPES[dtype],
    )
    processor.load_models()

    samples_meta: list[dict] = []
    grid_h = grid_w = feature_dim = in_channels = None
    skipped_caption = skipped_too_long = skipped_error = 0
    start = time.time()

    for image_file in images:
        if len(samples_meta) >= num_samples:
            break

        caption_file = caption_path_root / (image_file.stem + ".txt")
        if not caption_file.exists():
            skipped_caption += 1
            continue
        prompt = caption_file.read_text(encoding="utf-8").strip()
        if not prompt:
            skipped_caption += 1
            continue

        try:
            image_tensor = processor.preprocess_image(Image.open(image_file), resolution)
            latents = processor.encode_image(image_tensor)
            text_encoding = processor.encode_text(prompt, max_text_tokens)
            if text_encoding is None:
                # Caption does not fit in max_text_tokens; the cache loader assumes
                # every sample fits, so drop it rather than truncate the prompt.
                skipped_too_long += 1
                continue
            sample = processor.get_cache_data(
                latents, text_encoding, prompt=prompt, image_path=str(image_file)
            )
        except Exception as e:
            logger.warning("Encode failed for %s: %s", image_file.name, e)
            skipped_error += 1
            continue

        if grid_h is None:
            grid_h, grid_w = sample["grid_h"], sample["grid_w"]
            feature_dim = int(text_encoding["llm_features"].shape[-1])
            in_channels = int(latents.shape[0])

        index = len(samples_meta)
        relative_path = f"samples/{index}.pt"
        torch.save(sample, out_root / relative_path)
        samples_meta.append(
            {"cache_file": relative_path, "text_length": sample["text_length"], "prompt": prompt}
        )

        if log_every and len(samples_meta) % log_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            logger.info(
                "Encoded %d/%d (%.2f samples/s)", len(samples_meta), num_samples, len(samples_meta) / elapsed
            )

    if not samples_meta:
        raise RuntimeError(
            f"encoded 0 samples from {len(images)} images "
            f"(skipped: no/empty caption={skipped_caption}, caption too long={skipped_too_long}, "
            f"encode error={skipped_error})"
        )

    metadata = {
        "model_type": "ideogram4",
        "resolution": resolution,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "in_channels": in_channels,
        "llm_features_dim": feature_dim,
        "max_text_tokens": max_text_tokens,
        "vae_source": vae_source,
        "text_encoder_source": text_encoder_source,
        "num_samples": len(samples_meta),
        "samples": samples_meta,
    }
    with open(out_root / "metadata.json", "w") as f:
        json.dump(metadata, f)

    elapsed = max(time.time() - start, 1e-6)
    logger.info("=" * 70)
    logger.info("Wrote %d samples to %s", len(samples_meta), out_root)
    logger.info(
        "grid=%sx%s llm_features_dim=%s in_channels=%s", grid_h, grid_w, feature_dim, in_channels
    )
    if skipped_caption or skipped_too_long or skipped_error:
        logger.info(
            "Skipped: no/empty caption=%d, caption too long=%d, encode error=%d",
            skipped_caption,
            skipped_too_long,
            skipped_error,
        )
    if len(samples_meta) < num_samples:
        logger.warning(
            "Requested %d samples but only %d were encodable from %d images",
            num_samples,
            len(samples_meta),
            len(images),
        )
    logger.info("Elapsed %.1fs (%.2f samples/s)", elapsed, len(samples_meta) / elapsed)
    logger.info("=" * 70)

    return metadata
