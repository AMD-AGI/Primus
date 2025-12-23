###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
from pathlib import Path

import nltk
from datasets import load_dataset


def prepare_bookcorpus_dataset(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Downloading NLTK punkt tokenizer...")
    nltk.download("punkt")

    print(f"[Info] Loading bookcorpus dataset from Hugging Face...")
    dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)

    output_file = out_dir / "bookcorpus_megatron.json"
    print(f"[Info] Saving dataset to {output_file} ...")
    dataset.to_json(output_file)

    print("[Info] Dataset preparation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="tmp/data", help="Path to output JSON")
    args = parser.parse_args()
    prepare_bookcorpus_dataset(Path(args.out_dir))
