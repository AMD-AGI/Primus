#!/usr/bin/env python3
"""Generate MOCK breakdown JSON for site development / demo.

Numbers are placeholders in the right *order of magnitude*, seeded from the
published P57 single-layer attention micro-bench (V4-Flash widths: B=1, H=64,
Sq=4096, D=512) and the P40 EP=8 MoE/kernel attribution. They are NOT measured
ground truth — replace with `parse_trace.py` output once real traces exist
(every file is marked provenance.mock = true).

Usage:
    python3 examples/deepseek-v4/projection/tools/gen_mock_data.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"

PRO_COMPRESS = [128, 128] + [4 if i % 2 == 0 else 128 for i in range(2, 60)] + [0]
FLASH_COMPRESS = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(2, 42)] + [0]

MODEL_CONFIGS = {
    "pro": {
        "num_layers": 61,
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "kv_channels": 512,
        "num_experts": 384,
        "moe_router_topk": 6,
        "moe_ffn_hidden_size": 3072,
        "moe_shared_expert_intermediate_size": 3072,
        "index_topk": 1024,
        "vocab_size": 129280,
        "compress_ratios": PRO_COMPRESS,
    },
    "flash": {
        "num_layers": 43,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "kv_channels": 512,
        "num_experts": 256,
        "moe_router_topk": 6,
        "moe_ffn_hidden_size": 2048,
        "moe_shared_expert_intermediate_size": 2048,
        "index_topk": 512,
        "vocab_size": 129280,
        "compress_ratios": FLASH_COMPRESS,
    },
}

# MI355X: BF16 matrix 2.5 PFLOPS, HBM3E 8 TB/s (AMD product page).
# MI455X (MI400): HBM4 19.6 TB/s; BF16 dense not officially published —
# estimated ~10 PFLOPS (half of the 20 PFLOPS FP8 spec).
HARDWARE = {
    "MI355X": {"peak_tflops_bf16": 2500.0, "hbm_bandwidth_gbps": 8000.0},
    "MI455X": {"peak_tflops_bf16": 10000.0, "hbm_bandwidth_gbps": 19600.0},
}


def row(module, time_us, flop_class=None, tflops=None):
    flops = (tflops * time_us * 1e6) if (flop_class and tflops) else None
    return {
        "module": module,
        "time_us": round(time_us, 1),
        "class": "compute_bound" if flop_class else "memory_bound",
        "flop_class": flop_class,
        "flops": flops,
        "tflops": tflops,
        "kernels": [],
    }


# Base (flash) attention core fwd/bwd by cr, from P57 micro-bench (ms -> us).
ATTN_CORE = {
    "0": {"fwd": 500.0, "bwd": 2080.0},
    "4": {"fwd": 1430.0, "bwd": 5110.0},
    "128": {"fwd": 570.0, "bwd": 2810.0},
}


def attention_breakdown(cr, s):
    """s = linear scale factor vs flash widths."""
    fwd = [
        row("attn.qkv_proj", 220 * s, "gemm", 480),
        row("attn.core", ATTN_CORE[cr]["fwd"] * s, "attn", 210),
        row("attn.rope", 35 * s),
        row("attn.o_proj", 160 * s, "gemm", 470),
        row("attn.norm", 25 * s),
    ]
    bwd = [
        row("attn.qkv_proj", 440 * s, "gemm", 480),
        row("attn.core", ATTN_CORE[cr]["bwd"] * s, "attn", 180),
        row("attn.rope", 45 * s),
        row("attn.o_proj", 320 * s, "gemm", 470),
        row("attn.norm", 30 * s),
    ]
    if cr == "4":  # CSA uses the Indexer/Compressor
        fwd.insert(2, row("attn.indexer", 300 * s, "gemm", 300))
        bwd.insert(2, row("attn.indexer", 600 * s, "gemm", 300))
    return {"forward": fwd, "backward": bwd}


def moe_breakdown(sg, sc):
    """sg = grouped-gemm scale, sc = comm/act scale vs flash."""
    fwd = [
        row("moe.router", 55 * sc),
        row("moe.dispatch", 820 * sc),
        row("moe.grouped_gemm", 1850 * sg, "grouped_gemm", 430),
        row("moe.act", 110 * sc),
        row("moe.shared_expert", 210 * sg, "gemm", 460),
        row("moe.combine", 990 * sc),
    ]
    bwd = [
        row("moe.router", 75 * sc),
        row("moe.dispatch", 900 * sc),
        row("moe.grouped_gemm", 3700 * sg, "grouped_gemm", 430),
        row("moe.act", 150 * sc),
        row("moe.shared_expert", 420 * sg, "gemm", 460),
        row("moe.combine", 1050 * sc),
    ]
    return {"forward": fwd, "backward": bwd}


def non_layer(s):
    return {
        "embedding": {"forward": [row("embedding", 120 * s)], "backward": [row("embedding", 60 * s)]},
        "output": {
            "forward": [row("output", 900 * s, "gemm", 300)],
            "backward": [row("output", 1800 * s, "gemm", 300)],
        },
        "loss": {"forward": [row("loss", 80)], "backward": [row("loss", 60)]},
    }


def cr_counts(compress):
    c = {"0": 0, "4": 0, "128": 0}
    for x in compress:
        c[str(x)] += 1
    return c


def build(model):
    cfg = MODEL_CONFIGS[model]
    # scale factors vs flash baseline
    if model == "pro":
        s_attn, s_grouped, s_comm, s_out = 1.75, 2.0, 1.3, 1.75
        opt_params, opt_time = 2_300_000_000, 1600.0
    else:
        s_attn, s_grouped, s_comm, s_out = 1.0, 1.0, 1.0, 1.0
        opt_params, opt_time = 1_000_000_000, 850.0

    layers = {
        cr: {"attention": attention_breakdown(cr, s_attn), "moe": moe_breakdown(s_grouped, s_comm)}
        for cr in ("0", "4", "128")
    }
    return {
        "schema_version": 1,
        "model": model,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "mock": True,
            "note": "MOCK data (P57/P40 order-of-magnitude); replace with parse_trace.py output",
        },
        "capture": {
            "gpu": "MI355X",
            "seq_length": 4096,
            "micro_batch_size": 1,
            "tokens_per_microbatch": 4096,
            "ep": 8,
            "ga_for_capture": 2,
            "optimizer": "adam",
            "distributed_optimizer": True,
            "recompute": "off",
            "measured_iter_time_ms": None,
        },
        "model_config": {**cfg, "cr_layer_counts": cr_counts(cfg["compress_ratios"])},
        "hardware": HARDWARE,
        "layers": layers,
        "non_layer": non_layer(s_out),
        "optimizer": {
            "type": "adam",
            "measured_params": opt_params,
            "time_us": opt_time,
            "bytes_per_param": 18,
            "class": "memory_bound",
        },
        "comm": {
            "ep_dispatch_us": None,
            "ep_combine_us": None,
            "note": "EP dispatch/combine included in moe rows; informational",
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in ("pro", "flash"):
        doc = build(model)
        path = OUT_DIR / f"{model}.json"
        path.write_text(json.dumps(doc, indent=2))
        print(f"[gen_mock_data] wrote {path}")


if __name__ == "__main__":
    main()
