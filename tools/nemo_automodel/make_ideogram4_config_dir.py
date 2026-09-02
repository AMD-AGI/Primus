#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Write a weightless Ideogram-4 config directory, for pretraining from scratch.

WHY THIS EXISTS. The AutoModel diffusion recipe's pretrain path reads the
transformer's config from a local directory and builds a randomly initialized
model from it. It needs ``<dir>/transformer/config.json`` and NO weights.

Ideogram-4's published weights are gated, but its architecture is not a secret and
the pretrain path never wants the weights anyway -- so nothing here downloads or
authenticates. It constructs the model on the meta device, which materializes the
structure and the registered config without allocating storage for a single
parameter, and writes out just the config. That is the difference between a few
megabytes of metadata and tens of gigabytes of float32 allocation for something
that gets thrown away immediately.

USAGE:
    python tools/nemo_automodel/make_ideogram4_config_dir.py --out <dir>

Then point ``model.pretrained_model_name_or_path`` at that directory. The Primus
Ideogram-4 presets under ``primus/configs/`` expect exactly this layout.

The written directory is reloaded before this exits, the same way the recipe will
load it, so a malformed one fails here rather than several minutes into a
distributed job.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default="./ideogram4_config",
        help=(
            "Directory to write. A 'transformer/config.json' is created inside it. "
            "Relative by default, so running this with no arguments writes into the "
            "working directory."
        ),
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help=(
            "Optional JSON object of config values to override, for building a "
            "smaller model than the default. Useful for a single-GPU smoke test, "
            "e.g. '{\"num_layers\": 2}'."
        ),
    )
    return parser.parse_args(argv)


def build_config(overrides=None):
    """Return ``(config_dict, how)`` for the transformer, without allocating it.

    ``how`` records whether the meta device was used, because the fallback path
    allocates the real thing and a caller watching memory deserves to know which
    one ran.
    """
    import torch
    from diffusers import Ideogram4Transformer2DModel

    kwargs = dict(overrides or {})
    try:
        # The meta device gives the full structure and the config recorded at
        # construction without backing storage. The config is what is being saved,
        # and it does not depend on the parameters existing.
        with torch.device("meta"):
            model = Ideogram4Transformer2DModel(**kwargs)
        return model, "the meta device"
    except Exception as exc:
        # Falling back rather than failing: the result is identical, it just costs
        # a real allocation. Reported at a level a user will see, since that cost
        # is large enough to be surprising.
        print(
            f"[make_ideogram4_config_dir] building on the meta device failed ({exc}); "
            "falling back to a real allocation, which needs enough host memory for "
            "the whole model",
            file=sys.stderr,
        )
        return Ideogram4Transformer2DModel(**kwargs), "a real allocation"


def main(argv=None) -> int:
    args = parse_args(argv)

    overrides = None
    if args.overrides:
        try:
            overrides = json.loads(args.overrides)
        except json.JSONDecodeError as exc:
            print(f"--overrides is not valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(overrides, dict):
            print("--overrides must be a JSON object", file=sys.stderr)
            return 2

    from diffusers import Ideogram4Transformer2DModel

    subfolder = os.path.join(args.out, "transformer")
    os.makedirs(subfolder, exist_ok=True)

    model, how = build_config(overrides)
    model.save_config(subfolder)
    print(f"[make_ideogram4_config_dir] wrote {os.path.join(subfolder, 'config.json')} " f"(built via {how})")

    # Reloaded exactly as the recipe will, so a malformed directory fails here
    # instead of partway into a distributed run.
    config = Ideogram4Transformer2DModel.load_config(args.out, subfolder="transformer")
    print(
        "[make_ideogram4_config_dir] reload OK: "
        f"num_layers={config.get('num_layers')} "
        f"in_channels={config.get('in_channels')} "
        f"attention_head_dim={config.get('attention_head_dim')} "
        f"num_attention_heads={config.get('num_attention_heads')} "
        f"llm_features_dim={config.get('llm_features_dim')}"
    )
    print(
        f"[make_ideogram4_config_dir] point model.pretrained_model_name_or_path at "
        f"{os.path.abspath(args.out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
