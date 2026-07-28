#!/usr/bin/env python3
"""Create minimal metadata file for manually packed sequences."""

import argparse
import json
from pathlib import Path


def create_metadata(seq_length: int, output_path: str):
    """Create minimal metadata for packed sequences.

    Args:
        seq_length: The sequence length of your packed data
        output_path: Path to save the metadata JSON file
    """
    metadata = [
        {
            "max_samples_per_bin": 1,
            "dataset_max_seqlen": seq_length,
            "min_packed_seqlen": seq_length,
        }
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Created metadata file: {output_path}")
    print("  - max_samples_per_bin: 1")
    print(f"  - dataset_max_seqlen: {seq_length}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create packed metadata for MLPerf Llama2-70B")
    parser.add_argument("seq_length", type=int, help="Packed sequence length (e.g. 8192)")
    parser.add_argument("output_path", type=str, help="Path to packed_metadata.jsonl")
    args = parser.parse_args()

    create_metadata(args.seq_length, args.output_path)
