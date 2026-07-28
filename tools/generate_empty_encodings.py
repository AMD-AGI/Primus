#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Generate empty T5 and CLIP encodings for CFG dropout.

Runs T5-XXL and CLIP-L on the empty string ("") and saves the resulting
embeddings as .npy files. These are loaded by FluxPretrainTrainer at init
to replace text embeddings during classifier-free guidance dropout.

Using real model outputs (instead of torch.randn) is critical for training
convergence — see NVIDIA MLPerf Training v5.1 reference.

Usage:
    python tools/generate_empty_encodings.py \
        --output_dir /path/to/empty_encodings \
        --t5_model google/t5-v1_1-xxl \
        --clip_model openai/clip-vit-large-patch14 \
        --t5_max_length 256

Output files:
    t5_empty.npy   - shape (1, seq_len, 4096)
    clip_empty.npy - shape (1, 768)
"""

import argparse
import os

import numpy as np
import torch


def generate_t5_empty(model_name: str, max_length: int, device: str) -> np.ndarray:
    """Generate T5-XXL encoding for empty string."""
    from transformers import T5EncoderModel, T5Tokenizer

    print(f"Loading T5 tokenizer: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    print(f"Loading T5 model: {model_name}")
    model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device).eval()

    with torch.no_grad():
        inputs = tokenizer(
            "",
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)
        # outputs.last_hidden_state: (1, seq_len, 4096)
        embeddings = outputs.last_hidden_state.cpu().numpy()

    print(f"T5 empty encoding shape: {embeddings.shape}")
    del model
    torch.cuda.empty_cache()
    return embeddings


def generate_clip_empty(model_name: str, device: str) -> np.ndarray:
    """Generate CLIP-L pooled encoding for empty string."""
    from transformers import CLIPTextModel, CLIPTokenizer

    print(f"Loading CLIP tokenizer: {model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    print(f"Loading CLIP model: {model_name}")
    model = CLIPTextModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device).eval()

    with torch.no_grad():
        inputs = tokenizer(
            "",
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)
        # outputs.pooler_output: (1, 768)
        pooled = outputs.pooler_output.cpu().numpy()

    print(f"CLIP empty encoding shape: {pooled.shape}")
    del model
    torch.cuda.empty_cache()
    return pooled


def main():
    parser = argparse.ArgumentParser(description="Generate empty T5/CLIP encodings for CFG dropout")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save t5_empty.npy and clip_empty.npy",
    )
    parser.add_argument(
        "--t5_model",
        type=str,
        default="google/t5-v1_1-xxl",
        help="T5 model name or path (default: google/t5-v1_1-xxl)",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model name or path (default: openai/clip-vit-large-patch14)",
    )
    parser.add_argument(
        "--t5_max_length",
        type=int,
        default=256,
        help="Max sequence length for T5 encoding (default: 256 for schnell)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run models on (default: cuda if available)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    t5_path = os.path.join(args.output_dir, "t5_empty.npy")
    clip_path = os.path.join(args.output_dir, "clip_empty.npy")

    t5_embeddings = generate_t5_empty(args.t5_model, args.t5_max_length, args.device)
    np.save(t5_path, t5_embeddings)
    print(f"Saved T5 empty encodings to: {t5_path}")

    clip_embeddings = generate_clip_empty(args.clip_model, args.device)
    np.save(clip_path, clip_embeddings)
    print(f"Saved CLIP empty encodings to: {clip_path}")

    print("\nDone! Add to your YAML config:")
    print(f"  empty_encodings_path: {args.output_dir}")


if __name__ == "__main__":
    main()
