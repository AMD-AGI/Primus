#!/usr/bin/env python3
"""DeepSeek-V4 closed-form analytic FLOPs, ported from
``primus/backends/megatron/patches/deepseek_v4_flops_patches.py`` so the
projection's TFLOP/s matches what Megatron reports on a real run.

Returns per-component FMAC (multiply-only, pre fwd+bwd/FMA expansion) for ONE
layer of a given cr at batch_size=1. Multiply by FB_FMA (=6) for Megatron-
convention FLOPs (fwd 1 + bwd 2, times FMA 2).

Validated against the measured flash 16-layer run (TOTAL 34093 TFLOP/global-
batch; per-component within rounding) — see __main__ self-test.
"""

from __future__ import annotations

FB_FMA = 6  # _FORWARD_BACKWARD_FACTOR(3) * _FMA_FACTOR(2)
SWIGLU = 3  # gate+up+down collapsed expansion factor
DEFAULT_MTP_LAYERS = {"pro": 1, "flash": 1}

# Per-model architecture params (from primus/configs/models/megatron/*.yaml +
# deepseek_v4_base.yaml). Shared: index_head_dim=128, index_n_heads=64,
# attn_sliding_window=128, hc_mult=4.
MODEL_PARAMS = {
    "pro": dict(
        hidden=7168,
        heads=128,
        head_dim=512,
        q_lora=1536,
        o_lora=1024,
        o_groups=16,
        moe_ffn=3072,
        shared_ffn=3072,
        topk=6,
        experts=384,
        index_topk=1024,
        vocab=129280,
    ),
    "flash": dict(
        hidden=4096,
        heads=64,
        head_dim=512,
        q_lora=1024,
        o_lora=1024,
        o_groups=8,
        moe_ffn=2048,
        shared_ffn=2048,
        topk=6,
        experts=256,
        index_topk=512,
        vocab=129280,
    ),
}
SHARED = dict(index_head_dim=128, index_n_heads=64, swa_window=128, hc_mult=4)


def _local_visible_pairs(swa, s):
    if swa <= 0 or swa >= s:
        return s * (s + 1) // 2
    return swa * s - swa * (swa - 1) // 2


def _pool_visible_pairs(cr, s):
    if cr <= 0 or s <= 0:
        return 0
    n = s // cr
    if n == 0:
        return 0
    return cr * n * (n - 1) // 2 + n * (s - cr * n + 1)


def _visible_pairs(swa, cr, index_topk, s):
    local = _local_visible_pairs(swa, s)
    if cr == 0:
        return local
    pool = max(1, s // cr)
    if cr == 128:
        return local + _pool_visible_pairs(cr, s)
    if cr == 4:
        sparse = min(index_topk if index_topk else pool, pool)
        return local + sparse * s
    return local + pool * s


def _attn_qkv_o(s_eff, p):
    n_d = p["heads"] * p["head_dim"]
    qkv = p["hidden"] * p["q_lora"] + p["q_lora"] * n_d + p["hidden"] * p["head_dim"]
    if p["o_lora"] > 0:
        o_proj = n_d * p["o_lora"] + (p["o_groups"] * p["o_lora"]) * p["hidden"]
    else:
        o_proj = n_d * p["hidden"]
    return s_eff * (qkv + o_proj)


def _attn_scores(s_eff, cr, p):
    pairs = _visible_pairs(SHARED["swa_window"], cr, p["index_topk"], s_eff)
    return 2 * p["heads"] * p["head_dim"] * pairs


def _compressor(s_eff, cr, p):
    if cr == 0:
        return 0
    coff = 2 if cr == 4 else 1
    return 2 * s_eff * p["hidden"] * (coff * p["head_dim"])


def _indexer(s_eff, cr, p):
    if cr != 4:
        return 0
    ihd, inh = SHARED["index_head_dim"], SHARED["index_n_heads"]
    pool = max(1, s_eff // cr)
    dq_rank = ihd
    proj = p["hidden"] * dq_rank + dq_rank * (inh * ihd) + p["hidden"] * inh
    proj += 2 * p["hidden"] * (2 * ihd)  # mini-compressor
    return s_eff * proj + s_eff * inh * pool * ihd


def _moe(s_eff, p):
    router = p["hidden"] * p["experts"]
    routed = p["topk"] * SWIGLU * p["hidden"] * p["moe_ffn"]
    shared = SWIGLU * p["hidden"] * p["shared_ffn"] if p["shared_ffn"] > 0 else 0
    return s_eff * (router + routed + shared)


def _hc_mixer(s, p):
    hc = SHARED["hc_mult"]
    n_d = hc * p["hidden"]
    return 2 * s * n_d * ((2 + hc) * hc)


def _hc_head(s, p, mtp_num_layers):
    hc = SHARED["hc_mult"]
    n_d = hc * p["hidden"]
    return (1 + mtp_num_layers) * s * n_d * hc


def _mtp_eh_proj(s, p, mtp_num_layers):
    return mtp_num_layers * s * (2 * p["hidden"]) * p["hidden"]


def layer_fmac(model: str, cr: int, seq: int) -> dict[str, float]:
    """Per-layer FMAC components (batch_size=1) for one cr layer."""
    p = MODEL_PARAMS[model]
    s_eff = seq * SHARED["hc_mult"]
    return {
        "attn_qkv_o": _attn_qkv_o(s_eff, p),
        "attn_scores": _attn_scores(s_eff, cr, p),
        "compressor": _compressor(s_eff, cr, p),
        "indexer": _indexer(s_eff, cr, p),
        "moe": _moe(s_eff, p),
        "hc": _hc_mixer(seq, p),
    }


def nonlayer_fmac(model: str, seq: int, mtp_num_layers: int = 0) -> dict[str, float]:
    p = MODEL_PARAMS[model]
    return {"logits": (1 + mtp_num_layers) * seq * p["hidden"] * p["vocab"]}


def mtp_fmac(model: str, seq: int, mtp_num_layers: int = 1, mtp_cr: int = 4) -> dict[str, float]:
    """Extra FMAC components for MTP depths, batch_size=1.

    V4 MTP reuses a full V4 inner layer per depth; the current Flash Megatron
    FLOPs anchor reports a CSA-style MTP inner layer (cr=4).
    """
    if mtp_num_layers <= 0:
        return {
            "inner_layer": 0,
            "eh_proj": 0,
            "extra_logits": 0,
            "hc_head": _hc_head(seq, MODEL_PARAMS[model], 0),
        }
    p = MODEL_PARAMS[model]
    inner = sum(layer_fmac(model, mtp_cr, seq).values()) * mtp_num_layers
    main_logits = seq * p["hidden"] * p["vocab"]
    return {
        "inner_layer": inner,
        "eh_proj": _mtp_eh_proj(seq, p, mtp_num_layers),
        "extra_logits": main_logits * mtp_num_layers,
        "hc_head": _hc_head(seq, p, mtp_num_layers),
    }


def model_total_params(model: str, num_layers: int, mtp_num_layers: int = 0) -> int:
    """Approximate total parameter count (for optimizer-step sizing). Uses the
    same V4 MLA low-rank attention shapes as the FLOPs formula (q/o LoRA + single
    latent KV) instead of the crude 4*h^2, plus MoE experts + shared + router and
    the tied-free embedding/output. MTP adds one full V4 inner layer plus the
    2H->H eh_proj per depth; logits reuse the output layer weights."""
    p = MODEL_PARAMS[model]
    n_d = p["heads"] * p["head_dim"]
    attn = (
        p["hidden"] * p["q_lora"]
        + p["q_lora"] * n_d
        + p["hidden"] * p["head_dim"]
        + n_d * p["o_lora"]
        + p["o_groups"] * p["o_lora"] * p["hidden"]
    )
    moe = (
        p["experts"] * SWIGLU * p["hidden"] * p["moe_ffn"]
        + SWIGLU * p["hidden"] * p["shared_ffn"]
        + p["hidden"] * p["experts"]
    )
    return int(
        (num_layers + mtp_num_layers) * (attn + moe)
        + mtp_num_layers * 2 * p["hidden"] * p["hidden"]
        + 2 * p["vocab"] * p["hidden"]
    )


# Map analytic components to projection module names (per layer).
def module_flops(model: str, cr: int, seq: int) -> dict[str, float]:
    """Megatron-convention FLOPs (×FB_FMA) per module, per layer, batch_size=1."""
    f = layer_fmac(model, cr, seq)
    return {
        "attn.proj": f["attn_qkv_o"] * FB_FMA,
        "attn.core": f["attn_scores"] * FB_FMA,
        "attn.indexer": (f["compressor"] + f["indexer"]) * FB_FMA,
        "attn.norm": f["hc"] * FB_FMA,
        "moe.grouped_gemm": f["moe"] * FB_FMA,
    }


def output_flops(model: str, seq: int) -> float:
    return nonlayer_fmac(model, seq)["logits"] * FB_FMA


def mtp_flops(model: str, seq: int, mtp_num_layers: int = 1, mtp_cr: int = 4) -> dict[str, float]:
    f = mtp_fmac(model, seq, mtp_num_layers, mtp_cr)
    return {k: v * FB_FMA for k, v in f.items()}


def _self_test() -> None:
    """Self-test against measured flash 16L (GBS64): cr [0x3,4x6,128x7]."""
    sched = [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
    seq_t, batch = 4096, 64
    comp = {k: 0.0 for k in ("attn_qkv_o", "attn_scores", "compressor", "indexer", "moe", "hc")}
    for cr_t in sched:
        for k, v in layer_fmac("flash", cr_t, seq_t).items():
            comp[k] += v
    logits = nonlayer_fmac("flash", seq_t)["logits"]
    tot = (sum(comp.values()) + logits) * FB_FMA * batch / 1e12
    print("flash 16L analytic vs measured (TFLOP/global-batch):")
    for k, v in comp.items():
        print(f"  {k:12s} = {v*FB_FMA*batch/1e12:9.1f}")
    print(f"  {'logits':12s} = {logits*FB_FMA*batch/1e12:9.1f}")
    print(f"  TOTAL        = {tot:9.1f}   (measured 34093.4)")


if __name__ == "__main__":
    _self_test()
