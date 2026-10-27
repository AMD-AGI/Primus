###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Build the pre-encoded Ideogram-4 cache that ``cache.py`` reads.

Pairs each image with its caption, encodes both, and writes the flat layout the
dataloader expects:

    <output_dir>/metadata.json     the index, plus the grid and feature dimensions
    <output_dir>/samples/<i>.pt    one encoded sample each

This is a one-off offline step. It needs one device and the encoder weights,
neither of which training itself requires afterwards -- which is the entire point
of caching.

SELECTION IS DETERMINISTIC by default: images are taken in sorted order, and a
sample that cannot be encoded is SKIPPED rather than substituted. The same
source directory and the same arguments reproduce the same sample set, the same
metadata and the same text features. Latent values can still differ in the low
bits, because the autoencoder runs in bf16. ``shuffle`` opts into a seeded
shuffle instead, which is still reproducible but no longer order-independent.

A caption that does not fit the token budget is DROPPED, not truncated. The
dataloader derives its text width from the longest caption in the cache, so a
truncated caption would silently become the training signal for its image; a
missing one is visible in the skip counts this reports and records. The budget
counts the chat-templated tokens the text encoder sees, which is more than the
caption's own.

A REBUILD REPLACES the previous cache in the same directory: its index is removed
and its sample files deleted, but only once the encoders have loaded, so a run
that fails before encoding anything leaves the old cache intact.

Reached from the command line as ``primus-cli direct --single -- data automodel-cache --model ideogram4``.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# The autoencoder downsamples 8x and the model packs 2x2 latent patches.
RESOLUTION_MULTIPLE = 16

# One unreadable image is skipped; this many in a row means something systemic
# (a device error, a wrong autoencoder) that would fail every image after it.
MAX_CONSECUTIVE_FAILURES = 8


def _write_metadata(output_root: Path, metadata: Dict[str, Any]) -> None:
    """Write the index atomically: a reader sees the old file, the new one, or none."""
    partial = output_root / "metadata.json.partial"
    with open(partial, "w") as handle:
        json.dump(metadata, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output_root / "metadata.json")


def build_cache(
    *,
    image_dir: str,
    caption_dir: str,
    output_dir: str,
    num_samples: int = 1024,
    resolution: int = 256,
    max_text_tokens: int = 128,
    vae_source: Optional[str] = None,
    text_encoder_source: Optional[str] = None,
    tokenizer_source: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "bf16",
    seed: int = 1234,
    shuffle: bool = False,
    log_every: int = 64,
) -> Dict[str, Any]:
    """Encode up to ``num_samples`` image and caption pairs into a flat cache.

    Captions are read from ``<caption_dir>/<image stem>.txt``. Returns the metadata
    that was written, so a caller can report the grid and feature dimensions
    without re-reading the file.
    """
    from primus.backends.automodel.models.ideogram4.processor import Ideogram4Processor

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}")
    if max_text_tokens < 1:
        raise ValueError(
            f"max_text_tokens must be at least 1, got {max_text_tokens}; it is the "
            "chat-templated token budget a caption has to fit"
        )
    if resolution < RESOLUTION_MULTIPLE or resolution % RESOLUTION_MULTIPLE:
        raise ValueError(
            f"resolution must be a positive multiple of {RESOLUTION_MULTIPLE} (the "
            f"autoencoder's 8x downsampling times the 2x2 patch), got {resolution}"
        )

    image_root = Path(image_dir)
    caption_root = Path(caption_dir)
    output_root = Path(output_dir)
    for label, path in (("image_dir", image_root), ("caption_dir", caption_root)):
        if not path.is_dir():
            raise NotADirectoryError(f"{label} does not exist: {path}")

    images = sorted(p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"no images with extensions {IMAGE_EXTENSIONS} under {image_root}")
    if shuffle:
        random.Random(seed).shuffle(images)
    logger.info("Found %d images in %s", len(images), image_root)

    processor_kwargs: Dict[str, Any] = {"device": device, "dtype": DTYPES[dtype]}
    # An unset source means "whatever this model defaults to", so leave it out
    # rather than passing None over the processor's own default.
    for name, value in (
        ("vae_source", vae_source),
        ("text_encoder_source", text_encoder_source),
        ("tokenizer_source", tokenizer_source),
    ):
        if value is not None:
            processor_kwargs[name] = value

    processor = Ideogram4Processor(**processor_kwargs)
    processor.load_models()

    # The output directory is touched only now that the encoders have loaded, so a
    # run that fails at load time leaves a previous cache usable. From here on the
    # old cache is replaced: its index first, since the loader treats the index as
    # the mark of a finished cache, then its samples, so none outlives the rebuild.
    samples_root = output_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata.json").unlink(missing_ok=True)
    for stale in samples_root.glob("*.pt"):
        stale.unlink()

    from PIL import Image

    entries: List[Dict[str, Any]] = []
    grid_h = grid_w = feature_dim = in_channels = None
    skipped_caption = skipped_too_long = skipped_error = 0
    consecutive_failures = 0
    start = time.time()

    for image_file in images:
        if len(entries) >= num_samples:
            break

        caption_file = caption_root / (image_file.stem + ".txt")
        if not caption_file.exists():
            skipped_caption += 1
            continue

        try:
            prompt = caption_file.read_text(encoding="utf-8").strip()
            if not prompt:
                skipped_caption += 1
                continue
            # The budget is checked on the tokens alone, before either encoder runs.
            # Dropped rather than truncated: see the module docstring.
            if processor.count_text_tokens(prompt) > max_text_tokens:
                skipped_too_long += 1
                continue
            with Image.open(image_file) as image:
                image_tensor = processor.preprocess_image(image, resolution)
            latents = processor.encode_image(image_tensor)
            text_encoding = processor.encode_text(prompt, max_text_tokens)
            if text_encoding is None:
                skipped_too_long += 1
                continue
            sample = processor.get_cache_data(
                latents, text_encoding, prompt=prompt, image_path=str(image_file)
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            # One unreadable file should not lose the hours already spent on the
            # rest, so it is counted and reported rather than raised. A run of them
            # is systemic, and would fail every file after it.
            logger.warning("Could not encode %s: %s", image_file.name, exc)
            skipped_error += 1
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{consecutive_failures} images in a row failed to encode, the last "
                    f"({image_file.name}) with: {exc}. That is not a bad file but a "
                    "problem every image will hit, so the build stops here."
                ) from exc
            continue
        consecutive_failures = 0

        if grid_h is None:
            grid_h, grid_w = sample["grid_h"], sample["grid_w"]
            feature_dim = int(text_encoding["llm_features"].shape[-1])
            in_channels = int(latents.shape[0])

        relative_path = f"samples/{len(entries)}.pt"
        torch.save(sample, output_root / relative_path)
        entries.append(
            {
                "cache_file": relative_path,
                "text_length": sample["text_length"],
                "prompt": prompt,
            }
        )

        if log_every and len(entries) % log_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            logger.info(
                "Encoded %d/%d (%.2f per second)",
                len(entries),
                num_samples,
                len(entries) / elapsed,
            )

    if not entries:
        raise RuntimeError(
            f"encoded nothing from {len(images)} images. Skipped: "
            f"{skipped_caption} with no or empty caption, "
            f"{skipped_too_long} with a caption over {max_text_tokens} tokens, "
            f"{skipped_error} that failed to encode. Captions are expected at "
            f"{caption_root}/<image stem>.txt."
        )

    metadata = {
        "model_type": "ideogram4",
        "resolution": resolution,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "in_channels": in_channels,
        "llm_features_dim": feature_dim,
        "max_text_tokens": max_text_tokens,
        "vae_source": processor.vae_source,
        "text_encoder_source": processor.text_encoder_source,
        "num_samples": len(entries),
        "skipped": {
            "no_caption": skipped_caption,
            "too_long": skipped_too_long,
            "failed": skipped_error,
        },
        "samples": entries,
    }
    # Written last. The dataloader treats the presence of metadata.json as the
    # cache being complete, so writing it earlier would make an interrupted build
    # look finished.
    _write_metadata(output_root, metadata)

    elapsed = max(time.time() - start, 1e-6)
    logger.info("Wrote %d samples to %s", len(entries), output_root)
    logger.info(
        "Latents %sx%sx%s, text features %s-wide, longest caption %d tokens",
        in_channels,
        grid_h,
        grid_w,
        feature_dim,
        max(e["text_length"] for e in entries),
    )
    if skipped_caption or skipped_too_long or skipped_error:
        logger.info(
            "Skipped: %d with no or empty caption, %d with a caption over %d tokens, "
            "%d that failed to encode",
            skipped_caption,
            skipped_too_long,
            max_text_tokens,
            skipped_error,
        )
    if len(entries) < num_samples:
        logger.warning(
            "Asked for %d samples but only %d of the %d images were encodable. The "
            "cache is usable; it is just smaller than requested.",
            num_samples,
            len(entries),
            len(images),
        )
    logger.info("Took %.1fs (%.2f samples per second)", elapsed, len(entries) / elapsed)

    return metadata
