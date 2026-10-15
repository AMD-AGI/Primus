#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Convert HuggingFace WAN checkpoint to Primus format.

This tool converts HuggingFace Diffusers WAN (2.1 / 2.2) video-diffusion
transformer checkpoints to Primus/Megatron-Core compatible format. It handles:
- QKV weight fusion into Megatron's per-head interleaved layout
  (self-attention ``linear_qkv``, cross-attention ``linear_kv``)
- Key mapping from HF to Primus naming conventions
- Feed-forward rename onto Megatron-Core's MLP (linear_fc1 / linear_fc2)
- Multi-file safetensors loading

WAN 2.2 A14B is a dual-expert model: run the tool once per expert subfolder
(``transformer`` and ``transformer_2``).

Usage:
    # Convert Wan2.1-T2V-1.3B checkpoint
    python tools/checkpoint_conversion/convert_wan_hf_to_primus.py \\
        --input Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer \\
        --output checkpoints/primus_wan21_t2v_1_3b.safetensors \\
        --variant wan2.1_t2v_1.3b

    # Convert with custom architecture
    python tools/checkpoint_conversion/convert_wan_hf_to_primus.py \\
        --input path/to/checkpoint \\
        --output checkpoints/primus_custom.safetensors \\
        --variant custom \\
        --hidden-size 1536 \\
        --num-attention-heads 12 \\
        --num-dit-layers 30 \\
        --ffn-hidden-size 8960

Example:
    $ python tools/checkpoint_conversion/convert_wan_hf_to_primus.py \\
        --input Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer \\
        --output checkpoints/primus_wan21_t2v_1_3b.safetensors \\
        --variant wan2.1_t2v_1.3b

    Converting wan2.1_t2v_1.3b checkpoint
      Input: Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer
      Output: checkpoints/primus_wan21_t2v_1_3b.safetensors
      Architecture: 30 layers, hidden 1536, 12 heads, ffn 8960
    Loading HuggingFace checkpoint from: ...
    ...
    Conversion complete!
"""

import argparse
import logging
import sys
from pathlib import Path

# Add primus to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from primus.backends.megatron.core.models.diffusion.wan import (
    WanConfig,
    convert_hf_checkpoint,
)

# CLI variant name -> WanConfig preset classmethod.
WAN_VARIANTS = {
    "wan2.1_t2v_1.3b": "wan2_1_t2v_1_3b",
    "wan2.1_t2v_14b": "wan2_1_t2v_14b",
    "wan2.2_ti2v_5b": "wan2_2_ti2v_5b",
    "wan2.2_t2v_a14b": "wan2_2_t2v_a14b",
}

CUSTOM_ARGS = ("hidden_size", "num_attention_heads", "num_dit_layers", "ffn_hidden_size")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace WAN checkpoint to Primus format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Wan2.1-T2V-1.3B
  %(prog)s --input Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer \\
           --output checkpoints/primus_wan21_t2v_1_3b.safetensors \\
           --variant wan2.1_t2v_1.3b

  # Convert local checkpoint directory
  %(prog)s --input /path/to/wan/checkpoint \\
           --output primus_wan.safetensors \\
           --variant wan2.1_t2v_14b

  # Convert both WAN 2.2 A14B experts (dual-expert model, one run each)
  %(prog)s --input Wan-AI/Wan2.2-T2V-A14B-Diffusers/transformer \\
           --output primus_wan22_a14b_high_noise.safetensors \\
           --variant wan2.2_t2v_a14b
  %(prog)s --input Wan-AI/Wan2.2-T2V-A14B-Diffusers/transformer_2 \\
           --output primus_wan22_a14b_low_noise.safetensors \\
           --variant wan2.2_t2v_a14b

  # Convert with custom architecture
  %(prog)s --input /path/to/checkpoint \\
           --output primus_custom.safetensors \\
           --variant custom \\
           --hidden-size 1536 \\
           --num-attention-heads 12 \\
           --num-dit-layers 30 \\
           --ffn-hidden-size 8960
        """,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to HF checkpoint (file, directory, or HF model ID like "
        "'Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer')",
    )
    parser.add_argument("--output", required=True, help="Output path for Primus checkpoint (.safetensors)")
    parser.add_argument(
        "--variant",
        choices=[*WAN_VARIANTS, "custom"],
        default="wan2.1_t2v_1.3b",
        help="WAN variant (determines architecture). Default: wan2.1_t2v_1.3b",
    )
    parser.add_argument("--hidden-size", type=int, help="Hidden size (only with --variant custom)")
    parser.add_argument(
        "--num-attention-heads", type=int, help="Number of attention heads (only with --variant custom)"
    )
    parser.add_argument("--num-dit-layers", type=int, help="Number of DiT layers (only with custom variant)")
    parser.add_argument("--ffn-hidden-size", type=int, help="FFN hidden size (only with --variant custom)")
    parser.add_argument("--prefix", default=None, help="Optional output key prefix (e.g. 'transformer.')")
    parser.add_argument("--strict", action="store_true", help="Fail if any key is dropped or missing")

    args = parser.parse_args()

    # Validate custom variant arguments
    custom_values = {name: getattr(args, name) for name in CUSTOM_ARGS}
    if args.variant == "custom":
        missing = [name for name, value in custom_values.items() if not value]
        if missing:
            flags = ", ".join("--" + name.replace("_", "-") for name in missing)
            parser.error(f"{flags} required with --variant custom")
        config = WanConfig(**custom_values)
    else:
        if any(custom_values.values()):
            flags = ", ".join("--" + name.replace("_", "-") for name in CUSTOM_ARGS)
            parser.error(f"{flags} can only be used with --variant custom")
        config = getattr(WanConfig, WAN_VARIANTS[args.variant])()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Print conversion info
    print("=" * 80)
    print(f"Converting {args.variant} checkpoint")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(
        f"  Architecture: {config.num_dit_layers} layers, hidden {config.hidden_size}, "
        f"{config.num_attention_heads} heads, ffn {config.ffn_hidden_size}"
    )
    print("=" * 80)
    print()

    # Convert checkpoint
    try:
        convert_hf_checkpoint(
            checkpoint_path=args.input,
            wan_config=config,
            save_to=args.output,
            prefix=args.prefix,
            strict=args.strict,
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
