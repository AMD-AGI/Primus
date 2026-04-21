###############################################################################
# Prepare FineWeb-Edu Dataset for Megatron-LM
###############################################################################

import argparse
from pathlib import Path
from datasets import load_dataset

def prepare_fineweb_edu_dataset(out_dir: Path, sample_size: str = "10BT"):
    """
    Download FineWeb-Edu dataset and convert to JSON format.
    
    Args:
        out_dir: Output directory for JSON file
        sample_size: Size of dataset to download
            - "10BT" (10B tokens, ~13GB) - Recommended for testing
            - "100BT" (100B tokens, ~130GB)
            - "350BT" (350B tokens, ~450GB)
            - "sample-10BT", "sample-100BT", "sample-350BT" for samples
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # FineWeb-Edu dataset name format
    dataset_name = f"HuggingFaceFW/fineweb-edu"
    config_name = f"sample-{sample_size}" if not sample_size.startswith("sample-") else sample_size
    
    print(f"[Info] Loading fineweb-edu dataset ({sample_size}) from Hugging Face...")
    print(f"[Info] This may take a while depending on dataset size...")
    
    # Load dataset - you can specify num_proc for faster loading
    dataset = load_dataset(
        dataset_name,
        name=config_name,
        split="train",
        trust_remote_code=True,
        # streaming=True,  # Uncomment for very large datasets
    )
    
    output_file = out_dir / f"fineweb_edu_{sample_size}_megatron.json"
    print(f"[Info] Saving dataset to {output_file} ...")
    
    # Convert to JSON format that Megatron expects
    dataset.to_json(str(output_file))
    
    print(f"[Info] Dataset preparation completed: {output_file}")
    print(f"[Info] Total samples: {len(dataset)}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare FineWeb-Edu dataset for Megatron"
    )
    parser.add_argument(
        "--out-dir", 
        type=str, 
        default="./data/fineweb-edu", 
        help="Path to output directory"
    )
    parser.add_argument(
        "--sample-size",
        type=str,
        default="10BT",
        choices=["10BT", "100BT", "350BT", "sample-10BT", "sample-100BT", "sample-350BT"],
        help="Size of FineWeb-Edu dataset to download"
    )
    args = parser.parse_args()
    
    prepare_fineweb_edu_dataset(Path(args.out_dir), args.sample_size)