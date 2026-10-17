#!/usr/bin/env python3
"""Kernel / nn.module -> logical-module + flop-class mapping for the V4
projection breakdown.

Two independent classifications are provided:

1. ``module_from_stack(stack)`` — primary: derive the logical module from the
   python call stack captured by ``with_stack=True`` (matches nn.module class
   names appearing in the stack frames). This is the accurate path (A13).

2. ``module_from_kernel(name)`` — fallback: derive the logical module purely from
   the GPU kernel name when no usable stack is attached.

``flop_class_from_kernel(name)`` returns the compute-bound FLOP class
(``gemm`` / ``grouped_gemm`` / ``attn``) or ``None`` (memory-bound, A14).

The rules are intentionally data-driven so they can be extended as kernels are
renamed. Order matters: the first matching rule wins.
"""

from __future__ import annotations

# --- logical module taxonomy (kept small per the design) --------------------
# attention sub-modules:
#   attn.qkv_proj, attn.rope, attn.indexer, attn.core, attn.o_proj, attn.norm
# moe sub-modules:
#   moe.router, moe.dispatch, moe.grouped_gemm, moe.act, moe.combine,
#   moe.shared_expert
# non-layer: embedding, output, loss

# (substring, logical_module) — matched against the python call stack text.
# nn.module class names / function names seen in V4 forward stacks.
STACK_MODULE_RULES: list[tuple[str, str]] = [
    ("Indexer", "attn.indexer"),
    ("Compressor", "attn.indexer"),
    ("apply_rotary", "attn.rope"),
    ("rope", "attn.rope"),
    ("linear_qkv", "attn.qkv_proj"),
    ("q_layernorm", "attn.norm"),
    ("k_layernorm", "attn.norm"),
    ("linear_proj", "attn.o_proj"),
    ("o_proj", "attn.o_proj"),
    ("core_attention", "attn.core"),
    ("DeepseekV4Attention", "attn.core"),
    ("MLASelfAttention", "attn.core"),
    ("SelfAttention", "attn.core"),
    ("input_layernorm", "attn.norm"),
    ("pre_mlp_layernorm", "moe.router"),
    ("TopKRouter", "moe.router"),
    ("router", "moe.router"),
    ("sinkhorn", "moe.router"),
    ("shared_expert", "moe.shared_expert"),
    ("token_dispatch", "moe.dispatch"),
    ("dispatch", "moe.dispatch"),
    ("combine", "moe.combine"),
    ("GroupedMLP", "moe.grouped_gemm"),
    ("SequentialMLP", "moe.grouped_gemm"),
    ("grouped", "moe.grouped_gemm"),
    ("experts", "moe.grouped_gemm"),
    ("activation", "moe.act"),
    ("swiglu", "moe.act"),
    ("MoELayer", "moe.grouped_gemm"),
    ("word_embeddings", "embedding"),
    ("embedding", "embedding"),
    ("output_layer", "output"),
    ("lm_head", "output"),
    ("loss", "loss"),
    ("cross_entropy", "loss"),
]

# (substring, logical_module) — matched against the GPU kernel name (fallback).
KERNEL_MODULE_RULES: list[tuple[str, str]] = [
    ("_v4_csa_attention", "attn.core"),
    ("_v4_attention", "attn.core"),
    ("_hc_compute", "attn.core"),
    ("hc_compute", "attn.core"),
    ("_indexer", "attn.indexer"),
    ("indexer", "attn.indexer"),
    ("_compressor", "attn.indexer"),
    ("compressor", "attn.indexer"),
    ("apply_rope", "attn.rope"),
    ("rotary", "attn.rope"),
    ("rope", "attn.rope"),
    ("_sinkhorn", "moe.router"),
    ("sinkhorn", "moe.router"),
    ("_v4_router", "moe.router"),
    ("deep_ep::", "moe.dispatch"),  # refined to dispatch/combine below
    ("dispatch", "moe.dispatch"),
    ("combine", "moe.combine"),
    ("GroupedGemm", "moe.grouped_gemm"),
    ("_stack_grouped_weight", "moe.grouped_gemm"),
    ("group_gemm", "moe.grouped_gemm"),
    ("swiglu", "moe.act"),
    ("embedding", "embedding"),
    ("cross_entropy", "loss"),
    ("nll_loss", "loss"),
]

# (substring, flop_class) — matched against the GPU kernel name.
# Order matters: grouped GEMM must be checked before generic GEMM.
FLOP_CLASS_RULES: list[tuple[str, str]] = [
    ("GroupedGemmKernel", "grouped_gemm"),
    ("grouped_gemm", "grouped_gemm"),
    ("group_gemm", "grouped_gemm"),
    ("_v4_csa_attention", "attn"),
    ("_v4_attention", "attn"),
    ("attention_fwd", "attn"),
    ("attention_bwd", "attn"),
    # generic dense GEMM kernels (hipBLASLt / rocBLAS / CK tile / Triton matmul)
    ("GemmKernel", "gemm"),
    ("Cijk_", "gemm"),
    ("gemm", "gemm"),
    ("matmul", "gemm"),
]


def module_from_stack(stack: str | None) -> str | None:
    """Return the logical module from a python call-stack string, or None."""
    if not stack:
        return None
    for needle, module in STACK_MODULE_RULES:
        if needle in stack:
            return module
    return None


def module_from_kernel(name: str) -> str:
    """Return the logical module from a kernel name (fallback)."""
    lowered = name
    for needle, module in KERNEL_MODULE_RULES:
        if needle in lowered:
            if module == "moe.dispatch" and "combine" in lowered:
                return "moe.combine"
            return module
    return "other"


def flop_class_from_kernel(name: str) -> str | None:
    """Return 'gemm' | 'grouped_gemm' | 'attn', or None for memory-bound."""
    for needle, klass in FLOP_CLASS_RULES:
        if needle in name:
            return klass
    return None


def is_compute_bound(name: str) -> bool:
    return flop_class_from_kernel(name) is not None
