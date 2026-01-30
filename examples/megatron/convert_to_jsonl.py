#!/usr/bin/env python3
"""
Example script to convert various dataset formats to JSONL for offline SFT training.

This script demonstrates how to:
1. Load a HuggingFace dataset
2. Convert it to JSONL format
3. Save for offline training

Usage:
    python convert_to_jsonl.py --dataset tatsu-lab/alpaca --output alpaca_train.jsonl
"""

import argparse
import json


def convert_hf_dataset_to_jsonl(dataset_name: str, output_file: str, split: str = "train"):
    """
    Convert a HuggingFace dataset to JSONL format.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "tatsu-lab/alpaca")
        output_file: Path to output JSONL file
        split: Dataset split to convert (default: "train")
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: HuggingFace datasets library required")
        print("Install with: pip install datasets")
        return
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} samples")
    
    # Convert to JSONL
    print(f"Writing to {output_file}...")
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            # Convert to dict if needed
            if not isinstance(item, dict):
                item = dict(item)
            
            # Write as JSON line
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
            
            if count % 1000 == 0:
                print(f"  Processed {count} samples...")
    
    print(f"✓ Successfully converted {count} samples to {output_file}")


def convert_csv_to_jsonl(csv_file: str, output_file: str, 
                         instruction_col: str = "instruction",
                         response_col: str = "response"):
    """
    Convert a CSV file to JSONL format.
    
    Args:
        csv_file: Path to input CSV file
        output_file: Path to output JSONL file
        instruction_col: Name of instruction column
        response_col: Name of response column
    """
    print(f"Loading CSV: {csv_file}")
    
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas library required for CSV conversion")
        print("Install with: pip install pandas")
        return
    
    # Read CSV
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")
    
    # Check columns exist
    if instruction_col not in df.columns or response_col not in df.columns:
        print(f"Error: Columns {instruction_col} and {response_col} must exist")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Convert to JSONL
    print(f"Writing to {output_file}...")
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            item = {
                "instruction": str(row[instruction_col]),
                "response": str(row[response_col])
            }
            
            # Add optional fields if they exist
            if "input" in df.columns and pd.notna(row["input"]):
                item["input"] = str(row["input"])
            if "system" in df.columns and pd.notna(row["system"]):
                item["system"] = str(row["system"])
            
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"✓ Successfully converted {count} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets to JSONL format for offline SFT training"
    )
    parser.add_argument(
        "--dataset",
        help="HuggingFace dataset name (e.g., tatsu-lab/alpaca)"
    )
    parser.add_argument(
        "--csv",
        help="Path to CSV file to convert"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--instruction-col",
        default="instruction",
        help="CSV column name for instructions (default: instruction)"
    )
    parser.add_argument(
        "--response-col",
        default="response",
        help="CSV column name for responses (default: response)"
    )
    
    args = parser.parse_args()
    
    if args.dataset:
        convert_hf_dataset_to_jsonl(args.dataset, args.output, args.split)
    elif args.csv:
        convert_csv_to_jsonl(
            args.csv, 
            args.output,
            args.instruction_col,
            args.response_col
        )
    else:
        parser.print_help()
        print("\nError: Must specify either --dataset or --csv")


if __name__ == "__main__":
    main()
