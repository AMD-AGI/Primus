#!/usr/bin/env python3
"""Convert pre-tokenized data to proper packed sequence format.

This script adds the missing 'seq_start_id' field to existing tokenized .npy files
to make them compatible with Megatron-Bridge's packed sequence format.
"""

import numpy as np
import sys
from pathlib import Path


def convert_to_packed_format(input_path: str, output_path: str):
    """Convert tokenized data to packed format by adding seq_start_id.
    
    Args:
        input_path: Path to input .npy file with tokenized data
        output_path: Path to save converted packed format .npy file
    """
    print(f"Loading data from {input_path}...")
    data = np.load(input_path, allow_pickle=True)
    
    print(f"Input data shape: {data.shape}")
    print(f"First item keys: {list(data[0].keys())}")
    
    # Convert each item to packed format
    packed_data = []
    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing item {i}/{len(data)}...")
        
        # Each item is already a single sequence, so seq_start_id is just [0]
        packed_item = {
            'input_ids': item['input_ids'],
            'loss_mask': item['loss_mask'],
            'seq_start_id': [0]  # Single sequence starts at position 0
        }
        packed_data.append(packed_item)
    
    # Convert back to numpy array
    packed_data = np.array(packed_data, dtype=object)
    
    print(f"\nSaving packed data to {output_path}...")
    np.save(output_path, packed_data)
    
    print(f"✓ Conversion complete!")
    print(f"  - Input: {len(data)} items")
    print(f"  - Output: {len(packed_data)} items")
    print(f"  - First item now has keys: {list(packed_data[0].keys())}")
    
    # Verify the conversion
    test_item = packed_data[0]
    print(f"\nVerification:")
    print(f"  - input_ids shape: {test_item['input_ids'].shape}")
    print(f"  - loss_mask length: {len(test_item['loss_mask'])}")
    print(f"  - seq_start_id: {test_item['seq_start_id']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_packed_format.py <input.npy> <output.npy>")
        print("\nExample:")
        print("  python convert_to_packed_format.py /data/train.npy /data/train_packed.npy")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    convert_to_packed_format(input_path, output_path)
