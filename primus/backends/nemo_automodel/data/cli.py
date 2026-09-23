###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Data-command provider hook for the NeMo AutoModel backend.

A sibling of the Megatron provider, not an extension of it. The Megatron
diffusion commands write Energon WebDataset shards; this writes the flat
per-sample cache the AutoModel diffusion recipe reads. The two share neither an
on-disk format nor a code path, so they are separate providers behind the same
``primus data`` parser rather than modes of one command.
"""

import argparse
import logging

logger = logging.getLogger(__name__)

__all__ = ["register_data_subcommands"]


def _prepare_automodel_cache(args):
    """Build an offline pre-encoded cache for an AutoModel diffusion model."""
    from primus.backends.nemo_automodel.data.registry import get_cache_builder

    build_cache = get_cache_builder(args.model)

    logger.info("=" * 80)
    logger.info(f"AutoModel diffusion: building the offline cache for '{args.model}'")
    logger.info("=" * 80)

    kwargs = dict(
        image_dir=args.image_dir,
        caption_dir=args.caption_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        resolution=args.resolution,
        max_text_tokens=args.max_text_tokens,
        tokenizer_source=args.tokenizer_source,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    # An unset encoder source means "whatever this model defaults to", so leave it
    # out rather than passing None over the builder's own default.
    for name in ("vae_source", "text_encoder_source"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value

    metadata = build_cache(**kwargs)

    logger.info("=" * 80)
    logger.info("Cache complete.")
    logger.info(f"  Output:  {args.output_dir}")
    logger.info(f"  Samples: {metadata['num_samples']}")
    logger.info(
        f"  Latents: {metadata['in_channels']}x{metadata['grid_h']}x{metadata['grid_w']}, "
        f"text features {metadata['llm_features_dim']}-wide"
    )
    logger.info("Point the training config's data.dataloader.cache_dir at that directory.")
    logger.info("=" * 80)


def register_data_subcommands(data_subparsers: argparse._SubParsersAction) -> None:
    """Register the AutoModel data commands under ``primus data``."""
    # The registry is imported for its model list only. Its entries are dotted
    # strings, so building --model's choices here does not import any builder --
    # which matters, because a builder pulls in an autoencoder and a text encoder,
    # and 'primus data --help' should not need either.
    from primus.backends.nemo_automodel.data.registry import available_models

    automodel_models = available_models()
    cache_parser = data_subparsers.add_parser(
        "automodel-cache",
        help="Prepare a pre-encoded cache for AutoModel diffusion training",
        description=(
            "Build the offline pre-encoded cache for an AutoModel diffusion model.\n\n"
            "The autoencoder and text encoder run once here, so training needs\n"
            "neither their weights nor their memory. The output is a flat cache:\n"
            "  <output-dir>/metadata.json    the index, plus grid and feature dims\n"
            "  <output-dir>/samples/<i>.pt   one encoded sample each\n\n"
            "This is a single-GPU offline step and needs access to the encoder\n"
            "weights, which are often gated; set HF_TOKEN, and point HF_HOME at\n"
            "storage that survives the job.\n\n"
            "Selection is deterministic: images are taken in sorted order, so the\n"
            "same source and arguments reproduce the same cache. Use --shuffle to\n"
            "opt out. A caption that does not fit --max-text-tokens is skipped\n"
            "rather than truncated, and the counts are reported at the end.\n\n"
            f"Supported models: {', '.join(automodel_models)}\n\n"
            "Example:\n"
            "  primus data automodel-cache --model ideogram4 \\\n"
            "    --image-dir   /path/to/images \\\n"
            "    --caption-dir /path/to/captions \\\n"
            "    --output-dir  /path/to/cache \\\n"
            "    --num-samples 1024 --resolution 256 --max-text-tokens 128\n\n"
            "Then point the training config's data.dataloader.cache_dir at --output-dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cache_parser.add_argument(
        "--model",
        required=True,
        choices=automodel_models,
        help="Model whose cache format to build",
    )

    cache_source = cache_parser.add_argument_group("Source Configuration")
    cache_source.add_argument("--image-dir", required=True, help="Directory of source images")
    cache_source.add_argument(
        "--caption-dir",
        required=True,
        help="Directory of captions, one <image stem>.txt per image",
    )
    cache_source.add_argument("--output-dir", required=True, help="Cache output directory")
    cache_source.add_argument(
        "--num-samples", type=int, default=1024, help="Samples to encode (default: 1024)"
    )
    cache_source.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the source order (seeded) before selecting; default is sorted order",
    )
    cache_source.add_argument("--seed", type=int, default=1234, help="Shuffle seed (default: 1234)")

    cache_encode = cache_parser.add_argument_group("Encoding Options")
    cache_encode.add_argument(
        "--resolution", type=int, default=256, help="Square image resolution (default: 256)"
    )
    cache_encode.add_argument(
        "--max-text-tokens",
        type=int,
        default=128,
        help="Caption token budget; longer captions are skipped, not truncated (default: 128)",
    )
    cache_encode.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Encoder compute dtype (default: bf16)",
    )
    cache_encode.add_argument("--device", default="cuda", help="Encoder device (default: cuda)")

    cache_encoders = cache_parser.add_argument_group("Encoder Sources")
    cache_encoders.add_argument(
        "--vae-source", default=None, help="Autoencoder repo or local path (default: the model's own)"
    )
    cache_encoders.add_argument(
        "--text-encoder-source",
        default=None,
        help="Text-encoder repo or local path (default: the model's own)",
    )
    cache_encoders.add_argument(
        "--tokenizer-source",
        default=None,
        help="Tokenizer repo or local path (default: alongside the text encoder)",
    )

    cache_parser.set_defaults(func=lambda args, unknown: _prepare_automodel_cache(args))
