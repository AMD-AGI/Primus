#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Convert HuggingFace Flux checkpoint to Primus format.

This tool converts HuggingFace Diffusers Flux transformer checkpoints to
Primus/Megatron-Core compatible format. It handles:
- QKV weight fusion for grouped-query attention (GQA)
- Key mapping from HF to Primus naming conventions
- Single block proj_out splitting
- Multi-file safetensors loading

Usage:
    # Convert FLUX.1-dev checkpoint
    python tools/checkpoint_conversion/convert_flux_hf_to_primus.py \\
        --input black-forest-labs/FLUX.1-dev \\
        --output checkpoints/primus_flux_12b.safetensors \\
        --variant flux_12b

    # Convert with custom architecture
    python tools/checkpoint_conversion/convert_flux_hf_to_primus.py \\
        --input path/to/checkpoint \\
        --output checkpoints/primus_custom.safetensors \\
        --variant custom \\
        --num-joint-layers 10 \\
        --num-single-layers 20

Example:
    $ python tools/checkpoint_conversion/convert_flux_hf_to_primus.py \\
        --input black-forest-labs/FLUX.1-dev \\
        --output checkpoints/primus_flux_12b.safetensors \\
        --variant flux_12b

    Converting flux_12b checkpoint
      Input: black-forest-labs/FLUX.1-dev
      Output: checkpoints/primus_flux_12b.safetensors
      Architecture: 19 joint + 38 single layers
    Loading HuggingFace checkpoint from: black-forest-labs/FLUX.1-dev
    ...
    Conversion complete!
"""

import argparse
import sys
from pathlib import Path

# Add primus to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from primus.backends.megatron.core.models.diffusion.flux import (
    FluxConfig,
    convert_hf_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Flux checkpoint to Primus format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert FLUX.1-dev (12B parameters)
  %(prog)s --input black-forest-labs/FLUX.1-dev \\
           --output checkpoints/primus_flux_12b.safetensors \\
           --variant flux_12b

  # Convert local checkpoint directory
  %(prog)s --input /path/to/flux/checkpoint \\
           --output primus_flux.safetensors \\
           --variant flux_12b

  # Convert with custom architecture
  %(prog)s --input /path/to/checkpoint \\
           --output primus_custom.safetensors \\
           --variant custom \\
           --num-joint-layers 10 \\
           --num-single-layers 20
        """,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to HF checkpoint (file, directory, or HF model ID like 'black-forest-labs/FLUX.1-dev')",
    )
    parser.add_argument("--output", required=True, help="Output path for Primus checkpoint (.safetensors)")
    parser.add_argument(
        "--variant",
        choices=["flux_535m", "flux_12b", "custom"],
        default="flux_12b",
        help="Flux variant (determines architecture). Default: flux_12b",
    )
    parser.add_argument(
        "--num-joint-layers", type=int, help="Number of joint layers (only with --variant custom)"
    )
    parser.add_argument(
        "--num-single-layers", type=int, help="Number of single layers (only with --variant custom)"
    )

    args = parser.parse_args()

    # Validate custom variant arguments
    if args.variant == "custom":
        if not args.num_joint_layers or not args.num_single_layers:
            parser.error("--num-joint-layers and --num-single-layers are required with --variant custom")
    else:
        if args.num_joint_layers or args.num_single_layers:
            parser.error("--num-joint-layers and --num-single-layers can only be used with --variant custom")

    # Create config based on variant
    if args.variant == "flux_535m":
        config = FluxConfig.flux_535m()
    elif args.variant == "flux_12b":
        config = FluxConfig.flux_12b()
    else:  # custom
        config = FluxConfig(num_joint_layers=args.num_joint_layers, num_single_layers=args.num_single_layers)

    # Print conversion info
    print("=" * 80)
    print(f"Converting {args.variant} checkpoint")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Architecture: {config.num_joint_layers} joint + {config.num_single_layers} single layers")
    print("=" * 80)
    print()

    # Convert checkpoint
    try:
        convert_hf_checkpoint(
            checkpoint_path=args.input,
            flux_config=config,
            save_to=args.output,
        )

        print()
        print("=" * 80)
        print("✓ Conversion complete!")
        print(f"Primus checkpoint saved to: {args.output}")
        print("=" * 80)

        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Conversion failed: {e}")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
