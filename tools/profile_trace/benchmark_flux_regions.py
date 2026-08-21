#!/usr/bin/env python3
"""Benchmark FLUX compile and pointwise regions on a ROCm GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from primus.backends.diffusion.attention import set_attention_backend
from primus.backends.diffusion.models.flux.layers import LastLayer, SingleStreamBlock
from primus.backends.diffusion.models.flux.math import apply_rope, rope


def timed_backward(fn, inputs, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn(*inputs).float().sum().backward()
        for tensor in inputs:
            if tensor.grad is not None:
                tensor.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*inputs).float().sum().backward()
        for tensor in inputs:
            if tensor.grad is not None:
                tensor.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def benchmark_adaln(args) -> dict:
    shape = (args.batch, args.sequence, args.hidden)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    gate = torch.randn(args.batch, 1, args.hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scale = torch.randn_like(gate, requires_grad=True)
    shift = torch.randn_like(gate, requires_grad=True)

    def ln_modulate(value, scale_value, shift_value):
        return F.layer_norm(value, (args.hidden,), eps=1e-6) * (1 + scale_value) + shift_value

    def residual_ln_modulate(residual_value, value, gate_value, scale_value, shift_value):
        updated = residual_value + gate_value * value
        return F.layer_norm(updated, (args.hidden,), eps=1e-6) * (1 + scale_value) + shift_value

    eager_ms = timed_backward(
        residual_ln_modulate,
        (residual, x, gate, scale, shift),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    joint = torch.compile(residual_ln_modulate, fullgraph=True, mode=args.compile_mode)
    joint_ms = timed_backward(
        joint,
        (residual, x, gate, scale, shift),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    residual_gate = torch.compile(
        lambda residual_value, value, gate_value: residual_value + gate_value * value,
        fullgraph=True,
        mode=args.compile_mode,
    )
    ln_only = torch.compile(ln_modulate, fullgraph=True, mode=args.compile_mode)

    def split(residual_value, value, gate_value, scale_value, shift_value):
        return ln_only(residual_gate(residual_value, value, gate_value), scale_value, shift_value)

    return {
        "region": "adaln",
        "eager_ms": eager_ms,
        "split_compiled_ms": timed_backward(
            split,
            (residual, x, gate, scale, shift),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "joint_compiled_ms": joint_ms,
    }


def benchmark_qk_rope(args) -> dict:
    head_dim = args.hidden // args.heads
    shape = (args.batch, args.sequence, args.heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    ids = torch.arange(args.sequence, device="cuda", dtype=torch.float32).repeat(args.batch, 1)
    pe = rope(ids, head_dim, 10000).unsqueeze(2)

    def chain(query, key, embedding):
        query = F.rms_norm(query, (head_dim,))
        key = F.rms_norm(key, (head_dim,))
        query, key = apply_rope(query, key, embedding)
        return query + key

    eager_ms = timed_backward(chain, (q, k, pe), warmup=args.warmup, iterations=args.iterations)
    joint = torch.compile(chain, fullgraph=True, mode=args.compile_mode)
    joint_ms = timed_backward(
        joint,
        (q, k, pe),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    qk_norm = torch.compile(
        lambda query, key: (
            F.rms_norm(query, (head_dim,)),
            F.rms_norm(key, (head_dim,)),
        ),
        fullgraph=True,
        mode=args.compile_mode,
    )
    rope_only = torch.compile(apply_rope, fullgraph=True, mode=args.compile_mode)

    def split(query, key, embedding):
        query, key = qk_norm(query, key)
        query, key = rope_only(query, key, embedding)
        return query + key

    return {
        "region": "qk_rope",
        "eager_ms": eager_ms,
        "split_compiled_ms": timed_backward(
            split,
            (q, k, pe),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "joint_compiled_ms": joint_ms,
    }


def benchmark_output_head(args) -> dict:
    head = LastLayer(args.hidden, 1, 64).cuda().to(torch.bfloat16)
    head.init_weights()
    x = torch.randn(
        args.batch,
        args.sequence,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    vec = torch.randn(args.batch, args.hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    eager = timed_backward(head, (x, vec), warmup=args.warmup, iterations=args.iterations)
    head.compile(fullgraph=True, mode=args.compile_mode)
    compiled = timed_backward(head, (x, vec), warmup=args.warmup, iterations=args.iterations)
    return {"region": "output_head", "eager_ms": eager, "compiled_ms": compiled}


class SingleStack(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.blocks = nn.ModuleList(
            SingleStreamBlock(args.hidden, args.heads, mlp_ratio=4.0) for _ in range(args.depth)
        )

    def forward(self, x, vec, pe):
        for block in self.blocks:
            x = block(x, vec, pe)
        return x


def benchmark_single_stack(args) -> dict:
    set_attention_backend(args.attention_backend)
    stack = SingleStack(args).cuda().to(torch.bfloat16)
    if args.fp8:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        stack = convert_to_float8_training(
            stack,
            module_filter_fn=lambda module, fqn: type(module) is nn.Linear
            and fqn.rsplit(".", 1)[-1] in {"linear1", "linear2"},
            config=Float8LinearConfig(pad_inner_dim=False),
        )
    if args.strategy == "per_block":
        for block in stack.blocks:
            block.compile(fullgraph=True, mode=args.compile_mode)
    elif args.strategy == "stack":
        stack.compile(fullgraph=True, mode=args.compile_mode)

    x = torch.randn(
        args.batch,
        args.sequence,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    vec = torch.randn(args.batch, args.hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.arange(args.sequence, device="cuda", dtype=torch.float32).repeat(args.batch, 1)
    pe = rope(ids, args.hidden // args.heads, 10000).unsqueeze(2)
    latency = timed_backward(stack, (x, vec, pe), warmup=args.warmup, iterations=args.iterations)
    return {
        "region": "single_stack",
        "strategy": args.strategy,
        "depth": args.depth,
        "fp8": args.fp8,
        "latency_ms": latency,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region", choices=("adaln", "qk_rope", "output_head", "single_stack"))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--strategy", choices=("eager", "per_block", "stack"), default="per_block")
    parser.add_argument("--attention-backend", default="flash_attn_aiter")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--benchmark-fusion", action="store_true")
    parser.add_argument("--max-fusion-size", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if args.benchmark_fusion or args.max_fusion_size is not None:
        import torch._inductor.config

        torch._inductor.config.benchmark_fusion = args.benchmark_fusion
        if args.max_fusion_size is not None:
            torch._inductor.config.max_fusion_size = args.max_fusion_size
    torch.cuda.reset_peak_memory_stats()
    benchmark = {
        "adaln": benchmark_adaln,
        "qk_rope": benchmark_qk_rope,
        "output_head": benchmark_output_head,
        "single_stack": benchmark_single_stack,
    }[args.region]
    result = benchmark(args)
    result["benchmark_fusion"] = args.benchmark_fusion
    result["max_fusion_size"] = args.max_fusion_size
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
