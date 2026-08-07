#!/usr/bin/env python3
"""Populate the two GPT-OSS attention JIT cache entries before torchrun."""

import argparse

import aiter_hd64_asm_override
import torch
from transformer_engine.pytorch.attention import DotProductAttention


def prewarm(sequence_length: int, micro_batch_size: int, window_left: int) -> None:
    attention = DotProductAttention(
        num_attention_heads=64,
        kv_channels=64,
        num_gqa_groups=8,
        attention_dropout=0.0,
        qkv_format="sbhd",
        attn_mask_type="causal",
        window_size=(window_left, 0),
    ).cuda()
    shapes = (
        (sequence_length, micro_batch_size, 64, 64),
        (sequence_length, micro_batch_size, 8, 64),
        (sequence_length, micro_batch_size, 8, 64),
    )
    query, key, value = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True) for shape in shapes
    )
    output = attention(query, key, value)
    if isinstance(output, tuple):
        output = output[0]
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    print(f"attention_prewarm=PASS window_left={window_left}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--window-left", type=int, action="append")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is not available")
    torch.cuda.set_device(0)
    torch.manual_seed(1234)

    windows = args.window_left or [128, -1]
    for window_left in windows:
        prewarm(args.sequence_length, args.micro_batch_size, window_left)
    if aiter_hd64_asm_override.get_dispatch_count() < len(windows):
        raise RuntimeError("Forward ASM override was not dispatched")


if __name__ == "__main__":
    main()
