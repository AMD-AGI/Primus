#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Write a random-weight MiniMax-H3 proxy checkpoint, for training without the released weights.

WHY THIS EXISTS. MiniMax-H3's weight license excludes the US, EU, UK and South Korea,
so Q4 post-training runs on randomly initialized weights and synthetic data. Per
step that costs the same as fine-tuning the released weights, so throughput, memory
and scaling carry over; convergence and quality do not. Nothing here downloads or
reads anything from MiniMax: the architecture comes from diffusers' Apache 2.0
``MiniMaxH3Transformer3DModel`` defaults.

WHAT IT WRITES, a directory the diffusers pipeline loader reads:
    model_index.json       names the transformer-only training pipeline
    transformer/           config.json, and the weights unless --config-only
    PROXY.json             how it was made: seed, diffusers version, size

LoRA needs the weights: AutoModel applies it only on the fine-tune path, which loads
a checkpoint. Full fine-tuning can use either, since the random-initialization path
builds the model from ``config.json`` alone (``--config-only`` writes just that, on
the meta device, allocating no parameter storage).

USAGE:
    # reduced depth, for a single-GPU smoke test (about 0.9B parameters per 4 blocks)
    python tools/automodel/make_minimax_h3_proxy_checkpoint.py --out /scratch/h3_proxy_4 --num-layers 4
    # full size, weights in BF16 (66 GB), for LoRA
    python tools/automodel/make_minimax_h3_proxy_checkpoint.py --out /scratch/h3_proxy --dtype bfloat16
    # full size, config only, for full fine-tuning from random initialization
    python tools/automodel/make_minimax_h3_proxy_checkpoint.py --out /scratch/h3_config --config-only

Then point ``model.pretrained_model_name_or_path`` at the directory. The pipeline class
the directory names is registered by the Primus MiniMax-H3 patch at run time.

Never commit or publish the output: it is a proxy, and nothing about it is MiniMax-H3's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# So the tool runs from a checkout without Primus installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_SEED = 1234


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, help="Directory to write.")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer blocks (default: the released model's 50). Width stays full, so shapes are real.",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="Optional JSON object of further config values, e.g. '{\"num_refiner_layers\": 1}'.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Initialization seed (default: %(default)s)."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Storage dtype of the written weights. LoRA needs the storage and compute dtypes to match.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Write only the config (meta device, no parameter storage). Enough for full fine-tuning, not for LoRA.",
    )
    return parser.parse_args(argv)


def build_config_overrides(args) -> dict:
    overrides = {}
    if args.overrides:
        overrides = json.loads(args.overrides)
        if not isinstance(overrides, dict):
            raise ValueError("--overrides must be a JSON object")
    if args.num_layers is not None:
        if args.num_layers < 1:
            raise ValueError("--num-layers must be at least 1")
        overrides["num_layers"] = args.num_layers
    return overrides


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        overrides = build_config_overrides(args)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[make_minimax_h3_proxy_checkpoint] {exc}", file=sys.stderr)
        return 2

    try:
        import diffusers
        import torch
        from diffusers import MiniMaxH3Transformer3DModel
    except ImportError as exc:
        print(
            f"[make_minimax_h3_proxy_checkpoint] this diffusers has no MiniMax-H3 ({exc}); it needs "
            "diffusers 0.40 or later (the Primus AutoModel prepare hook caps it below 0.41).",
            file=sys.stderr,
        )
        return 1

    from primus.backends.automodel.models.minimax_h3 import pipeline as pipeline_lib

    transformer_dir = os.path.join(args.out, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)

    if args.config_only:
        with torch.device("meta"):
            model = MiniMaxH3Transformer3DModel(**overrides)
        model.save_config(transformer_dir)
    else:
        torch.manual_seed(args.seed)
        dtype = getattr(torch, args.dtype)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = MiniMaxH3Transformer3DModel(**overrides)
        finally:
            torch.set_default_dtype(previous)
        model.save_pretrained(transformer_dir, safe_serialization=True, max_shard_size="10GB")

    num_parameters = sum(p.numel() for p in model.parameters())
    with open(os.path.join(args.out, "model_index.json"), "w") as f:
        json.dump(pipeline_lib.model_index(diffusers.__version__), f, indent=2)
        f.write("\n")
    with open(os.path.join(args.out, "PROXY.json"), "w") as f:
        json.dump(
            {
                "what": "random-weight MiniMax-H3 proxy; not MiniMax's weights",
                "weights": None if args.config_only else args.dtype,
                "seed": None if args.config_only else args.seed,
                "diffusers": diffusers.__version__,
                "num_parameters": num_parameters,
                "config_overrides": overrides,
            },
            f,
            indent=2,
        )
        f.write("\n")

    # Reloaded and rebuilt from the file, as the recipe will, so a malformed directory
    # fails here instead of partway into a distributed run.
    config = MiniMaxH3Transformer3DModel.load_config(args.out, subfolder="transformer")
    with torch.device("meta"):
        MiniMaxH3Transformer3DModel.from_config(config)
    print(
        f"[make_minimax_h3_proxy_checkpoint] wrote {os.path.abspath(args.out)}: "
        f"{num_parameters:,} parameters, num_layers={config.get('num_layers')}, "
        f"weights={'none (config only)' if args.config_only else args.dtype}"
    )
    print(
        "[make_minimax_h3_proxy_checkpoint] point model.pretrained_model_name_or_path at that directory. "
        "Do not commit or publish it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
