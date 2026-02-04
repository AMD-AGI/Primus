###############################################################################
# End-to-End FineWeb-Edu Preparation Script
###############################################################################

import argparse
import os
import subprocess
import time
from pathlib import Path

def prepare_fineweb_edu_dataset(
    primus_path: Path,
    data_path: Path,
    tokenizer_type: str,
    tokenizer_model: str,
    sample_size: str = "10BT",
    workers: int = None,
):
    """Prepare FineWeb-Edu dataset for Megatron training."""
    
    dataset_name = f"fineweb-edu-{sample_size}"
    dataset_path = data_path / dataset_name
    output_path = dataset_path / tokenizer_type
    
    # Set HuggingFace cache
    hf_home = Path(os.environ.get("HF_HOME", data_path / "huggingface"))
    os.environ["HF_HOME"] = str(hf_home)
    
    # Output tokenized files
    tokenized_prefix = output_path / f"fineweb_edu_{sample_size}"
    tokenized_bin = tokenized_prefix.with_name(
        f"{tokenized_prefix.name}_text_sentence.bin"
    )
    tokenized_idx = tokenized_prefix.with_name(
        f"{tokenized_prefix.name}_text_sentence.idx"
    )
    
    # Check if already processed
    if tokenized_bin.exists() and tokenized_idx.exists():
        print(f"[Info] Tokenized files exist, skipping preprocessing.")
        print(f"  - {tokenized_bin}")
        print(f"  - {tokenized_idx}")
        return tokenized_prefix.with_name(f"{tokenized_prefix.name}_text_sentence")
    
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_json = dataset_path / f"fineweb_edu_{sample_size}_megatron.json"
    
    # Step 1: Download dataset if not exists
    if dataset_json.exists():
        print(f"[Info] Found dataset file: {dataset_json}, skipping download.")
    else:
        print(f"[Info] Downloading FineWeb-Edu ({sample_size}) dataset...")
        subprocess.run(
            [
                "python3",
                str(primus_path / "examples/megatron/prepare_fineweb_edu_megatron_dataset.py"),
                "--out-dir",
                str(dataset_path),
                "--sample-size",
                sample_size,
            ],
            check=True,
        )
        print("[Info] Download completed.")
    
    # Step 2: Tokenize and create binary files
    print(f"[Info] Preprocessing dataset with tokenizer {tokenizer_type} / {tokenizer_model}")
    start = time.time()
    
    if workers is None:
        workers = os.cpu_count()
    
    subprocess.run(
        [
            "python3",
            str(primus_path / "examples/megatron/preprocess_data.py"),
            "--input",
            str(dataset_json),
            "--tokenizer-type",
            tokenizer_type,
            "--tokenizer-model",
            tokenizer_model,
            "--output-prefix",
            str(tokenized_prefix),
            "--workers",
            str(workers),
            "--split-sentences",
            "--append-eod",
            "--partitions",
            "4",  # Increase for larger datasets
            "--log-interval",
            "10000",
        ],
        check=True,
    )
    
    elapsed = int(time.time() - start)
    print(f"[Info] Preprocessing completed in {elapsed} seconds ({elapsed/60:.1f} minutes)")
    
    return tokenized_prefix.with_name(f"{tokenized_prefix.name}_text_sentence")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu dataset for Megatron-LM training"
    )
    parser.add_argument(
        "--primus-path",
        type=str,
        required=True,
        help="Root path to the Primus project"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to data directory"
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="HuggingFaceTokenizer",
        help="Tokenizer type (HuggingFaceTokenizer, GPT2BPETokenizer, etc.)"
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer model name or path"
    )
    parser.add_argument(
        "--sample-size",
        type=str,
        default="10BT",
        choices=["10BT", "100BT", "350BT", "sample-10BT", "sample-100BT", "sample-350BT"],
        help="FineWeb-Edu dataset size"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers for preprocessing (default: CPU count)"
    )
    
    args = parser.parse_args()
    
    primus_path = Path(args.primus_path).resolve()
    data_path = Path(args.data_path).resolve()
    
    print("=" * 70)
    print("FineWeb-Edu Dataset Preparation for Megatron-LM")
    print("=" * 70)
    print(f"Primus Path:    {primus_path}")
    print(f"Data Path:      {data_path}")
    print(f"Tokenizer:      {args.tokenizer_type} / {args.tokenizer_model}")
    print(f"Sample Size:    {args.sample_size}")
    print(f"Workers:        {args.workers or os.cpu_count()}")
    print("=" * 70)
    
    # Check HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        print("[Warning] HF_TOKEN not set. You may need it for gated datasets.")
    
    output_prefix = prepare_fineweb_edu_dataset(
        primus_path=primus_path,
        data_path=data_path,
        tokenizer_type=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model,
        sample_size=args.sample_size,
        workers=args.workers,
    )
    
    print("\n" + "=" * 70)
    print("SUCCESS! Dataset is ready for training.")
    print("=" * 70)
    print(f"\nAdd this to your training config:\n")
    print(f"  train_data_path: {output_prefix}")
    print(f"  mock_data: false")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()