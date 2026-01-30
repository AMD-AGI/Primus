#!/usr/bin/env python3
"""Create minimal metadata file for manually packed sequences."""

import json
import sys

def create_metadata(seq_length: int, output_path: str):
    """Create minimal metadata for packed sequences.
    
    Args:
        seq_length: The sequence length of your packed data
        output_path: Path to save the metadata JSON file
    """
    # Minimal metadata that satisfies the requirements
    # Since each of your items is a single sequence, max_samples_per_bin is 1
    metadata = [
        {
            "max_samples_per_bin": 1,  # Each packed sequence contains 1 sample
            "dataset_max_seqlen": seq_length,  # Max sequence length in dataset
            "min_packed_seqlen": seq_length  # Min packed sequence length
        }
    ]
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created metadata file: {output_path}")
    print(f"  - max_samples_per_bin: 1")
    print(f"  - dataset_max_seqlen: {seq_length}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_metadata.py <seq_length> <output_path>")
        print("\nExample:")
        print("  python create_metadata.py 8192 /data/packed_metadata_path")
        sys.exit(1)
    
    seq_length = int(sys.argv[1])
    output_path = sys.argv[2]
    
    create_metadata(seq_length, output_path)
