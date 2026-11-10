#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Export a MiniMax-H3 LoRA trained with Primus/AutoModel to diffusers' LoRA format.

WHY THIS EXISTS. AutoModel saves a LoRA in PEFT's own format: ``adapter_model.safetensors``
with keys prefixed ``base_model.model.`` (PEFT's wrapper) and ``adapter_config.json`` beside
it. Diffusers' LoRA loaders, which inference stacks such as xDiT use, read neither as is:
they look for ``pytorch_lora_weights.safetensors``, keys under the component's name
(``transformer.``), and the rank and alpha in the file's metadata. Loading AutoModel's file
directly fails, on the file name and then on the key prefix (measured with diffusers 0.40
and 0.41.0.dev0).

WHAT IT DOES. Reads the adapter, strips PEFT's prefix, checks every key is a LoRA factor of
a module the adapter config targets, and writes the file with diffusers' own
``MiniMaxH3LoraLoaderMixin.save_lora_weights``, so the layout is diffusers' and not an
imitation of it. Rank, alpha and the targets go into the file's metadata.

USAGE:
    python tools/automodel/export_minimax_h3_lora.py \\
        --checkpoint <checkpoint_dir>/epoch_1_step_11/model --out /scratch/h3_lora_diffusers

Then, in diffusers:
    transformer.load_lora_adapter(
        "/scratch/h3_lora_diffusers", weight_name="pytorch_lora_weights.safetensors", prefix="transformer"
    )
(the model-level loader needs ``weight_name``: without it, it looks for the ``.bin`` name)
or ``pipe.load_lora_weights(...)`` on a MiniMax-H3 pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

PEFT_PREFIX = "base_model.model."
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
FACTOR_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=f"The checkpoint's model directory, holding {ADAPTER_WEIGHTS} and {ADAPTER_CONFIG}.",
    )
    parser.add_argument(
        "--out", required=True, help="Directory to write pytorch_lora_weights.safetensors to."
    )
    return parser.parse_args(argv)


def convert(state: Dict, config: Dict) -> Tuple[Dict, Dict]:
    """PEFT-format adapter tensors and config to diffusers' transformer LoRA layers and metadata.

    Raises ValueError for a key that is not a LoRA factor of a targeted module, so a checkpoint
    from another adapter type or model fails here, not silently at load time.
    """
    targets = set(config.get("target_modules") or ())
    layers = {}
    for key, tensor in state.items():
        name = key[len(PEFT_PREFIX) :] if key.startswith(PEFT_PREFIX) else key
        suffix = next((s for s in FACTOR_SUFFIXES if name.endswith(s)), None)
        if suffix is None:
            raise ValueError(
                f"{key!r} is not a LoRA factor (expected a key ending in {' or '.join(FACTOR_SUFFIXES)})"
            )
        module = name[: -len(suffix)]
        if targets and module not in targets:
            raise ValueError(f"{key!r} is a factor of {module!r}, which the adapter config does not target")
        layers[name] = tensor
    if not layers:
        raise ValueError("the adapter has no tensors")
    metadata = {
        "r": int(config["r"]),
        "lora_alpha": config.get("lora_alpha", config["r"]),
        "target_modules": sorted(targets) if targets else sorted({k.rsplit(".lora_", 1)[0] for k in layers}),
        "bias": config.get("bias", "none"),
    }
    return layers, metadata


def main(argv=None) -> int:
    args = parse_args(argv)
    from diffusers.loaders.lora_pipeline import MiniMaxH3LoraLoaderMixin
    from safetensors.torch import load_file

    src = Path(args.checkpoint)
    for name in (ADAPTER_WEIGHTS, ADAPTER_CONFIG):
        if not (src / name).is_file():
            print(
                f"error: {src / name} not found; --checkpoint is the checkpoint's model directory",
                file=sys.stderr,
            )
            return 2
    config = json.loads((src / ADAPTER_CONFIG).read_text())
    if config.get("peft_type", "LORA") != "LORA":
        print(f"error: peft_type {config.get('peft_type')!r} is not LORA", file=sys.stderr)
        return 2
    layers, metadata = convert(load_file(str(src / ADAPTER_WEIGHTS)), config)
    MiniMaxH3LoraLoaderMixin.save_lora_weights(
        args.out, transformer_lora_layers=layers, transformer_lora_adapter_metadata=metadata
    )
    print(
        f"wrote {len(layers)} LoRA tensors (rank {metadata['r']}, alpha {metadata['lora_alpha']}, "
        f"{len(metadata['target_modules'])} modules) to {Path(args.out) / 'pytorch_lora_weights.safetensors'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
