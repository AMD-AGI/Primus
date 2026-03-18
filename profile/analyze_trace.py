#!/usr/bin/env python3
"""Analyze PyTorch profiler trace JSON from DSV3 Megatron/Primus training runs.

Reports:
  Header: Config table
  A: Phase Summary (FWD/BWD/Other)
  B: Forward Layer Analysis (~17 semantic operators)
  C: Backward Layer Analysis (~17 semantic operators + FWD vs BWD comparison)
  Bottleneck Analysis
  HTML Timeline
"""

import json, sys, os, re
from collections import defaultdict

TRACE_FILE = sys.argv[1] if len(sys.argv) > 1 else (
    "/shared_aig/xiaoming/Primus-dev/"
    "primus-megatron-exp[dsv3-pretrain-mbs_2-gbs_512-PP_4-EP_8-VPP_1-"
    "turbodeepep_True-legacygg_True-profile_True-recompute_num_layers_0]"
    "-rank[0].1773739477052422662.pt.trace.json"
)
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/shared_aig/xiaoming/Primus-dev/profile"
EXP_NAME = "dsv3-pretrain-mbs_2-gbs_512-PP_4-EP_8-VPP_1-turbodeepep_True-legacygg_True-profile_True-recompute_num_layers_0"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Classification ──────────────────────────────────────────────────────────

def classify(name):
    if "nccl" in name.lower(): return "NCCL"
    if "fmha_fwd" in name: return "Attn-FWD"
    if "fmha_bwd" in name or "FmhaBwdOGradDotO" in name: return "Attn-BWD"
    if "deep_ep" in name: return "DeepEP"
    if "rmsnorm_fwd" in name: return "RMSNorm-FWD"
    if "rmsnorm_bwd" in name: return "RMSNorm-BWD"
    if "rotary_fwd" in name: return "RoPE-FWD"
    if "rotary_bwd" in name: return "RoPE-BWD"
    if "cast_transpose" in name: return "FP8-CastTranspose"
    if re.search(r"topk", name, re.I): return "MoE-TopK"
    if "unpermute" in name: return "MoE-Unpermute"
    if "permute" in name: return "MoE-Permute"
    if "Cijk" in name and "F8" in name: return "GEMM-FP8"
    if "Cijk" in name: return "GEMM-BF16"
    if "multi_tensor_apply" in name: return "Optimizer"
    if "triton" in name: return "Triton-Fused"
    if "fillBuffer" in name: return "MemSet"
    if "MEMORY_COPY" in name or "Memcpy" in name or "memcpy" in name: return "MemCopy"
    if "vectorized" in name or "elementwise" in name: return "Elementwise"
    return "Other"

CAT_STYLE = {
    "RMSNorm-FWD": ("#6366f1","norm"), "RMSNorm-BWD": ("#6366f1","norm"),
    "FP8-CastTranspose": ("#8b5cf6","cast"),
    "GEMM-FP8": ("#06b6d4","fp8"),
    "RoPE-FWD": ("#f59e0b","rope"), "RoPE-BWD": ("#f59e0b","rope"),
    "Attn-FWD": ("#ef4444","attn"), "Attn-BWD": ("#ef4444","attn"),
    "GEMM-BF16": ("#10b981","bf16"),
    "MoE-TopK": ("#f97316","moe"), "MoE-Permute": ("#f97316","moe"), "MoE-Unpermute": ("#f97316","moe"),
    "DeepEP": ("#ec4899","ep"), "Triton-Fused": ("#a3e635","tri"),
    "NCCL": ("#e74c3c","nccl"), "Optimizer": ("#7f8c8d","opt"),
    "Elementwise": ("#f1c40f","elem"), "MemSet": ("#64748b","mem"), "MemCopy": ("#64748b","mem"),
    "Other": ("#94a3b8","other"),
}
def cat_color(c): return CAT_STYLE.get(c, ("#94a3b8","other"))[0]
def cat_group(c): return CAT_STYLE.get(c, ("#94a3b8","other"))[1]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def group_adjacent(kernels):
    if not kernels: return []
    groups, cur_cat, cur = [], kernels[0]["cat"], [kernels[0]]
    for k in kernels[1:]:
        if k["cat"] == cur_cat:
            cur.append(k)
        else:
            ws = cur[0]["ts"]; we = max(x["ts"]+x["dur"] for x in cur)
            top = max(cur, key=lambda x: x["dur"])
            groups.append({"cat": cur_cat, "n": len(cur), "ksum": sum(x["dur"] for x in cur),
                           "ws": ws, "we": we, "wall": we-ws,
                           "topk": top["name"], "top_ext": top.get("ext_id")})
            cur_cat, cur = k["cat"], [k]
    ws = cur[0]["ts"]; we = max(x["ts"]+x["dur"] for x in cur)
    top = max(cur, key=lambda x: x["dur"])
    groups.append({"cat": cur_cat, "n": len(cur), "ksum": sum(x["dur"] for x in cur),
                   "ws": ws, "we": we, "wall": we-ws,
                   "topk": top["name"], "top_ext": top.get("ext_id")})
    return groups

def detect_fwd_layers(groups):
    ui = [i for i,g in enumerate(groups) if g["cat"]=="MoE-Unpermute"]
    if not ui: return []
    layers, prev = [], 0
    for u in ui:
        end = u
        for j in range(u+1, min(u+5, len(groups))):
            if groups[j]["cat"] in ("DeepEP","Elementwise","Triton-Fused"): end = j
            else: break
        layers.append((prev, end)); prev = end+1
    return layers

def detect_bwd_layers(groups):
    """BWD layer: DeepEP(dispatch) → MoE-Permute → Expert GEMMs → ... → Attn-BWD → ... → RMSNorm-BWD(Attn).
    Use MoE-Permute before each Attn-BWD as the anchor for layer start."""
    ai = [i for i,g in enumerate(groups) if g["cat"]=="Attn-BWD"]
    if not ai: return []
    layers = []
    for a in ai:
        # Layer end: last big RMSNorm-BWD after Attn-BWD
        rc, end = 0, a
        for j in range(a+1, len(groups)):
            if groups[j]["cat"]=="RMSNorm-BWD": rc+=1
            if rc>=4: end=j; break
            end=j
        # Layer start: find MoE-Permute before this Attn-BWD, then go to preceding DeepEP
        perm = None
        for j in range(a-1, -1, -1):
            if groups[j]["cat"]=="MoE-Permute": perm=j; break
        if perm is not None:
            # Go back one more to find DeepEP dispatch (or Elementwise/RMSNorm-BWD from prev layer)
            ss = perm
            for j in range(perm-1, -1, -1):
                if groups[j]["cat"]=="DeepEP": ss=j; break
                # Stop if we hit end of previous layer (RMSNorm-BWD with large ksum)
                if groups[j]["cat"]=="RMSNorm-BWD" and groups[j]["ksum"]>500: ss=j+1; break
                ss=j
        else:
            ss = max(0, a-30)
        layers.append((ss, end))
    return layers

def assign_phase(ts, fi, bi):
    for s,e in fi:
        if s<=ts<=e: return "Forward"
    for s,e in bi:
        if s<=ts<=e: return "Backward"
    return "Other"

# ─── Semantic Merge ──────────────────────────────────────────────────────────

def _find(layer, cat, start=0, n_min=None):
    for i in range(start, len(layer)):
        if layer[i]["cat"] == cat and (n_min is None or layer[i]["n"] >= n_min):
            return i
    return None

def _collect(layer, indices):
    gs = [layer[i] for i in indices]
    top = max(gs, key=lambda g: g["ksum"])
    return {"n": sum(g["n"] for g in gs), "ksum": sum(g["ksum"] for g in gs),
            "ws": min(g["ws"] for g in gs), "we": max(g["we"] for g in gs),
            "topk": top.get("topk", ""), "top_ext": top.get("top_ext")}

def _op(label, cat, n, ksum, ws, we, wall=None, topk="", top_ext=None):
    w = wall if wall is not None else ksum
    return {"label": label, "cat": cat, "n": n, "ksum": ksum, "ws": ws, "we": we, "wall": w,
            "topk": topk, "top_ext": top_ext}

def _single(label, cat, g):
    return _op(label, cat, g["n"], g["ksum"], g["ws"], g["we"], g["wall"],
               g.get("topk", ""), g.get("top_ext"))


def merge_fwd(groups, layer_range):
    s, e = layer_range
    L = groups[s:e+1]

    attn = _find(L, "Attn-FWD")
    if attn is None: return []
    ops = []

    # ── _forward_attention ──
    # RMSNorm(Attn): first RMSNorm-FWD only
    rn0 = _find(L, "RMSNorm-FWD", 0)
    if rn0 is not None and rn0 < attn:
        ops.append(_single("RMSNorm(Attn)", "RMSNorm-FWD", L[rn0]))

    # MLASelfAttention(QKV proj): merge ALL FP8-Cast + GEMM-FP8 + intermediate RMSNorms before Attn
    qkv_indices = [i for i in range((rn0 or 0)+1, attn)
                   if L[i]["cat"] in ("FP8-CastTranspose","GEMM-FP8","RMSNorm-FWD")]
    if qkv_indices:
        c = _collect(L, qkv_indices)
        ops.append(_op("MLASelfAttention(QKV proj)", "GEMM-FP8", c["n"], c["ksum"], c["ws"], c["we"], topk=c["topk"], top_ext=c["top_ext"]))

    # MLASelfAttention(RoPE)
    rp = [i for i in range(0, attn) if L[i]["cat"]=="RoPE-FWD"]
    if rp:
        c = _collect(L, rp)
        ops.append(_op("MLASelfAttention(RoPE)", "RoPE-FWD", c["n"], c["ksum"], c["ws"], c["we"], topk=c["topk"], top_ext=c["top_ext"]))

    # MLASelfAttention(FlashAttn)
    ops.append(_single("MLASelfAttention(FlashAttn)", "Attn-FWD", L[attn]))

    # MLASelfAttention(O proj): merge FP8-Cast + GEMM-FP8 after Attn until RMSNorm(MoE)
    moe_norm = _find(L, "RMSNorm-FWD", attn+1)
    o_indices = [i for i in range(attn+1, moe_norm or len(L))
                 if L[i]["cat"] in ("FP8-CastTranspose","GEMM-FP8")]
    if o_indices:
        c = _collect(L, o_indices)
        ops.append(_op("MLASelfAttention(O proj)", "GEMM-FP8", c["n"], c["ksum"], c["ws"], c["we"], topk=c["topk"], top_ext=c["top_ext"]))

    # ── _forward_mlp ──
    # RMSNorm(MoE)
    if moe_norm is not None:
        ops.append(_single("RMSNorm(MoE)", "RMSNorm-FWD", L[moe_norm]))

    # MoELayer(TopKRouter): merge Router GEMM + all TopK + interleaved small ops
    router = _find(L, "GEMM-BF16", (moe_norm or attn)+1)
    dispatch = _find(L, "DeepEP", (router or attn)+1)
    if router is not None and dispatch is not None and L[router]["n"] == 1:
        router_range = list(range(router, dispatch))
        if router_range:
            c = _collect(L, router_range)
            ops.append(_op("MoELayer(TopKRouter)", "MoE-TopK", c["n"], c["ksum"], c["ws"], c["we"], c["we"]-c["ws"], c["topk"], c["top_ext"]))

    # MoELayer(dispatch)
    if dispatch is not None:
        ops.append(_single("MoELayer(dispatch)", "DeepEP", L[dispatch]))

    # MoELayer(token_permute)
    permute = _find(L, "MoE-Permute", (dispatch or attn)+1)
    if permute is not None:
        ops.append(_single("MoELayer(token_permute)", "MoE-Permute", L[permute]))

    # MoELayer(GroupedMLP FFN1×N)
    ffn1 = _find(L, "GEMM-BF16", (permute or attn)+1, n_min=32)
    if ffn1 is not None:
        g = L[ffn1]
        ops.append(_op(f"MoELayer(GroupedMLP FFN1×{g['n']})", "GEMM-BF16", g["n"], g["ksum"], g["ws"], g["we"], g["wall"], g.get("topk",""), g.get("top_ext")))

    # MoELayer(GroupedMLP SwiGLU)
    swiglu = _find(L, "Triton-Fused", (ffn1 or attn)+1)
    if swiglu is not None:
        ops.append(_single("MoELayer(GroupedMLP SwiGLU)", "Triton-Fused", L[swiglu]))

    # MoELayer(GroupedMLP FFN2×N)
    ffn2 = _find(L, "GEMM-BF16", (swiglu or attn)+1, n_min=32)
    if ffn2 is not None:
        g = L[ffn2]
        ops.append(_op(f"MoELayer(GroupedMLP FFN2×{g['n']})", "GEMM-BF16", g["n"], g["ksum"], g["ws"], g["we"], g["wall"], g.get("topk",""), g.get("top_ext")))

    # MoELayer(token_unpermute)
    unpermute = _find(L, "MoE-Unpermute", (ffn2 or attn)+1)
    if unpermute is not None:
        ops.append(_single("MoELayer(token_unpermute)", "MoE-Unpermute", L[unpermute]))

    # MoELayer(combine)
    combine = _find(L, "DeepEP", (unpermute or attn)+1)
    if combine is not None:
        ops.append(_single("MoELayer(combine)", "DeepEP", L[combine]))

    return ops


def merge_bwd(groups, layer_range):
    s, e = layer_range
    L = groups[s:e+1]

    attn = _find(L, "Attn-BWD")
    if attn is None: return []
    ops = []

    # ── _forward_mlp BWD ──
    dispatch0 = _find(L, "DeepEP", 0)
    if dispatch0 is not None and dispatch0 < attn:
        ops.append(_single("MoELayer(dispatch)", "DeepEP", L[dispatch0]))

    permute = _find(L, "MoE-Permute", 0)
    if permute is not None and permute < attn:
        ops.append(_single("MoELayer(token_permute)", "MoE-Permute", L[permute]))

    dffn2 = _find(L, "GEMM-BF16", (permute or 0)+1, n_min=64)
    if dffn2 is not None and dffn2 < attn:
        g = L[dffn2]
        ops.append(_op(f"MoELayer(GroupedMLP dFFN2×{g['n']})", "GEMM-BF16", g["n"], g["ksum"], g["ws"], g["we"], g["wall"], g.get("topk",""), g.get("top_ext")))

    swiglu = _find(L, "Triton-Fused", (dffn2 or 0)+1)
    if swiglu is not None and dffn2 is not None and swiglu < attn:
        ops.append(_single("MoELayer(GroupedMLP SwiGLU BWD)", "Triton-Fused", L[swiglu]))

    dffn1 = _find(L, "GEMM-BF16", (swiglu or 0)+1, n_min=64)
    if dffn1 is not None and dffn1 < attn:
        g = L[dffn1]
        ops.append(_op(f"MoELayer(GroupedMLP dFFN1×{g['n']})", "GEMM-BF16", g["n"], g["ksum"], g["ws"], g["we"], g["wall"], g.get("topk",""), g.get("top_ext")))

    for i in range((dffn1 or 0)+1, attn):
        if L[i]["cat"]=="Elementwise" and L[i]["n"]>=4 and L[i]["ksum"]>1000:
            ops.append(_single("MoELayer(grad_acc)", "Elementwise", L[i])); break

    unpermute = _find(L, "MoE-Unpermute", (dffn1 or 0)+1)
    if unpermute is not None and unpermute < attn:
        ops.append(_single("MoELayer(token_unpermute)", "MoE-Unpermute", L[unpermute]))

    combine = _find(L, "DeepEP", (unpermute or 0)+1)
    if combine is not None and combine < attn:
        ops.append(_single("MoELayer(combine)", "DeepEP", L[combine]))

    for i in range((combine or 0)+1, attn):
        if L[i]["cat"]=="GEMM-BF16" and L[i]["n"]==1 and L[i]["ksum"]>100:
            ops.append(_single("MoELayer(Router wgrad)", "GEMM-BF16", L[i])); break

    for i in range(attn-1, -1, -1):
        if L[i]["cat"]=="RMSNorm-BWD" and L[i]["ksum"]>500:
            ops.append(_single("RMSNorm-BWD(MoE)", "RMSNorm-BWD", L[i])); break

    # ── _forward_attention BWD ──
    o_fp8 = [i for i in range(max(0, attn-10), attn) if L[i]["cat"]=="GEMM-FP8"]
    if o_fp8:
        c = _collect(L, o_fp8)
        ops.append(_op("MLASelfAttention(O wgrad)", "GEMM-FP8", c["n"], c["ksum"], c["ws"], c["we"], topk=c["topk"], top_ext=c["top_ext"]))

    ops.append(_single("MLASelfAttention(FlashAttn BWD)", "Attn-BWD", L[attn]))

    rope = _find(L, "RoPE-BWD", attn+1)
    if rope is not None:
        ops.append(_single("MLASelfAttention(RoPE BWD)", "RoPE-BWD", L[rope]))

    post_fp8 = [i for i in range(attn+1, len(L)) if L[i]["cat"]=="GEMM-FP8"]
    if post_fp8:
        c = _collect(L, post_fp8)
        ops.append(_op("MLASelfAttention(QKV wgrad)", "GEMM-FP8", c["n"], c["ksum"], c["ws"], c["we"], topk=c["topk"], top_ext=c["top_ext"]))

    for i in range(len(L)-1, attn, -1):
        if L[i]["cat"]=="RMSNorm-BWD" and L[i]["ksum"]>500:
            ops.append(_single("RMSNorm-BWD(Attn)", "RMSNorm-BWD", L[i])); break

    ops.sort(key=lambda x: x["ws"])
    return ops


# ─── Report Generation ───────────────────────────────────────────────────────

def gen_layer_report(title, ops):
    """Generate markdown for one layer analysis (Report B or C)."""
    if not ops:
        return f"# {title}\n\nCould not extract layer.\n"

    total_k = sum(o["ksum"] for o in ops)
    total_w = ops[-1]["we"] - ops[0]["ws"]

    lines = [f"# {title}\n"]

    # Compute gaps between ops
    gaps = {}
    for i in range(len(ops)-1):
        g = ops[i+1]["ws"] - ops[i]["we"]
        if g > 500:
            cause = ""
            a = ops[i]["label"]
            if "dispatch" in a: cause = "cross-node all-to-all"
            elif "combine" in a: cause = "EP combine return"
            elif "TopK" in a: cause = "CPU scheduling"
            else: cause = "scheduling gap"
            gaps[i] = (g/1000, cause)

    # Top-3 by kernel time for ★ markers
    sorted_by_k = sorted(range(len(ops)), key=lambda i: -ops[i]["ksum"])
    top3 = set(sorted_by_k[:3])

    def fmt_leaf(idx, o, prefix_pad):
        """Format one leaf line: #N  label  time(us, right-aligned)  [extras]"""
        t_us = o["ksum"]
        extras = []
        if idx in top3: extras.append("★")
        ovlp = o["ksum"]/o["wall"] if o["wall"]>0 else 0
        if ovlp > 1.1: extras.append(f"{ovlp:.2f}× overlap")
        if o["n"] > 1: extras.append(f"{o['n']} kernels")
        time_s = f"{t_us:>8.0f} us"
        extra_s = f"   {'  '.join(extras)}" if extras else ""
        return f"#{idx+1:<3} {o['label']:<38s} {time_s}{extra_s}"

    def gap_line(idx, prefix):
        if idx in gaps:
            g_ms, cause = gaps[idx]
            return f"{prefix}        [gap {g_ms*1000:.0f}us — {cause}]"
        return None

    # Build tree based on label prefixes
    is_fwd = "Forward" in title
    lines.append("```")

    if is_fwd:
        lines.append("TransformerLayer")
        lines.append("├── _forward_attention")
        # RMSNorm(Attn) = ops with "RMSNorm(Attn)"
        # MLA ops = ops with "MLASelfAttention"
        attn_norm = [i for i,o in enumerate(ops) if "RMSNorm(Attn)" in o["label"]]
        mla_ops = [i for i,o in enumerate(ops) if "MLASelfAttention" in o["label"]]
        moe_norm = [i for i,o in enumerate(ops) if "RMSNorm(MoE)" in o["label"]]
        moe_ops = [i for i,o in enumerate(ops) if "MoELayer" in o["label"]]

        for ai in attn_norm:
            lines.append(f"│   ├── {fmt_leaf(ai, ops[ai], '│   │')}")
            gl = gap_line(ai, "│   │")
            if gl: lines.append(gl)

        if mla_ops:
            lines.append("│   └── MLASelfAttention")
            for j, mi in enumerate(mla_ops):
                branch = "└──" if j == len(mla_ops)-1 else "├──"
                lines.append(f"│       {branch} {fmt_leaf(mi, ops[mi], '│       │')}")
                gl = gap_line(mi, "│       │")
                if gl: lines.append(gl)

        lines.append("└── _forward_mlp")
        for ni in moe_norm:
            lines.append(f"    ├── {fmt_leaf(ni, ops[ni], '    │')}")

        if moe_ops:
            lines.append("    └── MoELayer")
            # Sub-group: GroupedMLP ops
            gmlp_ops = [i for i in moe_ops if "GroupedMLP" in ops[i]["label"]]
            non_gmlp = [i for i in moe_ops if "GroupedMLP" not in ops[i]["label"]]

            all_moe_items = []
            gmlp_inserted = False
            for mi in moe_ops:
                if mi in gmlp_ops and not gmlp_inserted:
                    all_moe_items.append(("gmlp_group", gmlp_ops))
                    gmlp_inserted = True
                elif mi not in gmlp_ops:
                    all_moe_items.append(("leaf", mi))

            for j, item in enumerate(all_moe_items):
                is_last = j == len(all_moe_items)-1
                branch = "└──" if is_last else "├──"
                cont = "    " if is_last else "│   "

                if item[0] == "leaf":
                    mi = item[1]
                    lines.append(f"        {branch} {fmt_leaf(mi, ops[mi], '        '+cont)}")
                    gl = gap_line(mi, "        "+cont)
                    if gl: lines.append(gl)
                else:
                    lines.append(f"        {branch} GroupedMLP")
                    gops = item[1]
                    for k, gi in enumerate(gops):
                        gb = "└──" if k == len(gops)-1 else "├──"
                        lines.append(f"        {cont}   {gb} {fmt_leaf(gi, ops[gi], '        '+cont+'   │')}")
                        gl = gap_line(gi, "        "+cont+"   │")
                        if gl: lines.append(gl)
    else:
        lines.append("TransformerLayer BWD")
        moe_ops = [i for i,o in enumerate(ops) if "MoELayer" in o["label"] or "RMSNorm-BWD(MoE)" in o["label"]]
        attn_ops = [i for i,o in enumerate(ops) if "MLASelfAttention" in o["label"] or "RMSNorm-BWD(Attn)" in o["label"]]

        lines.append("├── _forward_mlp BWD")
        if moe_ops:
            gmlp_ops = [i for i in moe_ops if "GroupedMLP" in ops[i]["label"]]
            non_gmlp = [i for i in moe_ops if "GroupedMLP" not in ops[i]["label"]]

            all_items = []
            gmlp_inserted = False
            for mi in moe_ops:
                if mi in gmlp_ops and not gmlp_inserted:
                    all_items.append(("gmlp_group", gmlp_ops))
                    gmlp_inserted = True
                elif mi not in gmlp_ops:
                    all_items.append(("leaf", mi))

            lines.append("│   └── MoELayer")
            for j, item in enumerate(all_items):
                is_last = j == len(all_items)-1
                branch = "└──" if is_last else "├──"
                cont = "    " if is_last else "│   "
                if item[0] == "leaf":
                    mi = item[1]
                    lines.append(f"│       {branch} {fmt_leaf(mi, ops[mi], '│       '+cont)}")
                    gl = gap_line(mi, "│       "+cont)
                    if gl: lines.append(gl)
                else:
                    lines.append(f"│       {branch} GroupedMLP")
                    gops = item[1]
                    for k, gi in enumerate(gops):
                        gb = "└──" if k == len(gops)-1 else "├──"
                        lines.append(f"│       {cont}   {gb} {fmt_leaf(gi, ops[gi], '│       '+cont+'   │')}")

        lines.append("└── _forward_attention BWD")
        if attn_ops:
            mla_ops = [i for i in attn_ops if "MLASelfAttention" in ops[i]["label"]]
            norm_ops = [i for i in attn_ops if "RMSNorm-BWD(Attn)" in ops[i]["label"]]

            if mla_ops:
                lines.append("    ├── MLASelfAttention")
                for j, mi in enumerate(mla_ops):
                    branch = "└──" if j == len(mla_ops)-1 else "├──"
                    lines.append(f"    │   {branch} {fmt_leaf(mi, ops[mi], '    │   │')}")
                    gl = gap_line(mi, "    │   │")
                    if gl: lines.append(gl)
            for ni in norm_ops:
                lines.append(f"    └── {fmt_leaf(ni, ops[ni], '        ')}")

    lines.append("```\n")

    # Per-operator table
    lines.append("## Per-Operator Statistics (execution order)\n")
    lines.append("| # | Operator | Kernels | Wall (us) | Kernel (us) | % | Overlap | Top Kernel | Input Shape |")
    lines.append("|---|----------|---------|-----------|-------------|---|---------|------------|-------------|")
    for i, o in enumerate(ops):
        pct = o["ksum"]/total_k*100 if total_k>0 else 0
        ovlp = o["ksum"]/o["wall"] if o["wall"]>0 else 0
        ovlp_s = f"**{ovlp:.2f}×**" if ovlp > 1.1 else "-"
        bold = pct >= 5
        lb = f"**{o['label']}**" if bold else o["label"]
        wall_s = f"**{o['wall']:.0f}**" if bold else f"{o['wall']:.0f}"
        ksum_s = f"**{o['ksum']:.0f}**" if bold else f"{o['ksum']:.0f}"
        pct_s = f"**{pct:.1f}%**" if bold else f"{pct:.1f}%"
        topk = o.get("topk", "")
        topk_s = f"`{topk[:55]}`" if topk else ""
        shape_s = ext_shapes.get(o.get("top_ext"), "")
        lines.append(f"| {i+1} | {lb} | {o['n']} | {wall_s} | {ksum_s} | {pct_s} | {ovlp_s} | {topk_s} | {shape_s} |")
    lines.append(f"| | **Total** | **{sum(o['n'] for o in ops)}** | **{total_w:.0f}** | **{total_k:.0f}** | **100%** | | | |")
    lines.append("")

    # Overlap analysis
    expert_ops = [o for o in ops if o["cat"]=="GEMM-BF16" and o["n"]>=32]
    if expert_ops:
        lines.append("## Overlap Analysis\n")
        lines.append("Expert GEMM shows multi-stream kernel overlap:\n")
        for o in expert_ops:
            ovlp = o["ksum"]/o["wall"] if o["wall"]>0 else 0
            lines.append(f"- **{o['label']}**: kernel sum = {o['ksum']/1000:.2f}ms, "
                         f"wall = {o['wall']/1000:.2f}ms → **{ovlp:.2f}× overlap**")
        lines.append("")

    # Idle gaps
    gaps = []
    for i in range(len(ops)-1):
        gap = ops[i+1]["ws"] - ops[i]["we"]
        if gap > 100:
            cause = ""
            a, b = ops[i]["label"], ops[i+1]["label"]
            if "TopK" in a and "DeepEP" in b: cause = "CPU scheduling + DeepEP buffer prep"
            elif "DeepEP" in a and "Permute" in b: cause = "Cross-node all-to-all comm wait"
            elif "DeepEP" in a and "dispatch" in a: cause = "EP dispatch latency"
            elif "combine" in a: cause = "EP combine return latency"
            else: cause = "Kernel launch / scheduling gap"
            gaps.append((a, b, gap/1000, cause))

    if gaps:
        lines.append("## Idle Gaps\n")
        lines.append("| Between | Gap (ms) | Cause |")
        lines.append("|---------|----------|-------|")
        for a, b, g, c in gaps:
            lines.append(f"| {a} → {b} | {g:.2f} | {c} |")
        lines.append("")

    # Key takeaways
    lines.append("## Key Takeaways\n")
    sorted_ops = sorted(ops, key=lambda x: -x["ksum"])
    takeaway_n = 1
    for o in sorted_ops[:3]:
        pct = o["ksum"]/total_k*100
        lines.append(f"{takeaway_n}. **{o['label']} 占 {pct:.1f}%** ({o['ksum']/1000:.2f} ms)")
        if "BF16" in o["label"] and "FFN" in o["label"]:
            lines.append(f"   - 当前走 legacy grouped GEMM (BF16)，`moe_use_legacy_grouped_gemm=False` 可启用 FP8")
            lines.append(f"   - BF16→FP8 约 **2× GEMM throughput**")
        elif "Attn" in o["label"]:
            lines.append(f"   - 已用 Flash Attention，优化空间有限")
        elif "DeepEP" in o["label"]:
            lines.append(f"   - 含跨节点 all-to-all 通信延迟")
        takeaway_n += 1

    expert_total = sum(o["ksum"] for o in ops if o["cat"]=="GEMM-BF16" and o["n"]>=32)
    if expert_total > 0:
        ep = expert_total/total_k*100
        lines.append(f"\n{takeaway_n}. **Expert GEMM BF16 合计占 {ep:.1f}%** — 全层最大瓶颈，FP8 升级预估整层提速 30-40%")
        takeaway_n += 1

    fp8_total = sum(o["ksum"] for o in ops if o["cat"]=="GEMM-FP8")
    if fp8_total > 0:
        fp = fp8_total/total_k*100
        lines.append(f"\n{takeaway_n}. **FP8 路径仅用于 attention 投影**: {fp:.1f}% — 相比 Expert GEMM，FP8 覆盖率低")

    lines.append("")
    return "\n".join(lines)


# ─── Load & Process ──────────────────────────────────────────────────────────

print("=" * 70)
print("DSV3 Profile Trace Analysis")
print("=" * 70)

print(f"Loading: {os.path.basename(TRACE_FILE)}")
with open(TRACE_FILE) as f:
    data = json.load(f)
events = data["traceEvents"]
print(f"Events: {len(events)}")

step = next(e for e in events if e.get("name","").startswith("ProfilerStep#") and e.get("cat")=="user_annotation")
SS, SD, SE = step["ts"], step["dur"], step["ts"]+step["dur"]
print(f"Step: {step['name']} ({SD/1000:.2f} ms)")

fwd_intervals, bwd_intervals = [], []
for e in events:
    name, ts, dur = e.get("name",""), e.get("ts",0), e.get("dur",0) or 0
    if ts<SS or ts>=SE: continue
    if "schedules.py" in name and "forward_step" in name and "calc_loss" not in name:
        fwd_intervals.append((ts, ts+dur))
    elif "schedules.py" in name and "backward_step" in name:
        bwd_intervals.append((ts, ts+dur))
fwd_intervals.sort(); bwd_intervals.sort()
print(f"FWD: {len(fwd_intervals)}, BWD: {len(bwd_intervals)}")

# Build ext_id → shape map from cpu_op events
ext_shapes = {}
for e in events:
    if e.get("cat") != "cpu_op": continue
    args = e.get("args", {})
    ext_id = args.get("External id")
    if ext_id is None or "Input Dims" not in args: continue
    dims = args["Input Dims"]
    types = args.get("Input type", [])
    # Pick first tensor inputs with non-empty dims
    parts = []
    for d, t in zip(dims, types):
        if d and isinstance(d, list) and len(d) > 0:
            dtype = t.replace("c10::", "").lower() if t else ""
            shape = "×".join(str(x) for x in d)
            parts.append(f"{dtype}[{shape}]")
            if len(parts) >= 2: break
    if parts:
        ext_shapes[ext_id] = " ".join(parts)

all_kernels = sorted([
    {"name": e["name"], "ts": e["ts"], "dur": e.get("dur",0) or 0,
     "cat": classify(e["name"]), "ext_id": e.get("args",{}).get("External id")}
    for e in events if e.get("cat")=="kernel" and e.get("ph")=="X" and SS<=e["ts"]<SE
], key=lambda x: x["ts"])
print(f"Kernels: {len(all_kernels)}, Shape entries: {len(ext_shapes)}")


# ─── Report A ────────────────────────────────────────────────────────────────

phase_time = {"Forward": 0.0, "Backward": 0.0, "Other": 0.0}
for k in all_kernels:
    phase_time[assign_phase(k["ts"], fwd_intervals, bwd_intervals)] += k["dur"]
total_kernel = sum(phase_time.values())

rpt_a = ["# Report A: Phase Summary\n",
    "| Phase | Time (ms) | % |", "|-------|-----------|---|"]
for p in ["Forward","Backward","Other"]:
    t = phase_time[p]; rpt_a.append(f"| {p} | {t/1000:.2f} | {t/total_kernel*100:.1f}% |")
rpt_a.append(f"| **Total** | **{total_kernel/1000:.2f}** | **100%** |")
rpt_a.extend(["", f"- Forward microbatches: {len(fwd_intervals)}",
    f"- Backward microbatches: {len(bwd_intervals)}",
    f"- Total GPU kernel time: {total_kernel/1000:.2f} ms",
    f"- Step wall time (ProfilerStep): {SD/1000:.2f} ms",
    f"- GPU utilization (kernel / wall): {total_kernel/SD*100:.1f}%"])
print("Report A done")


# ─── Report B: Forward Layer Analysis ────────────────────────────────────────

print("Generating Report B (Forward Layer)...")
def get_layer_ops(intervals, detect_fn, merge_fn, mb_idx):
    if len(intervals) <= mb_idx: return []
    s, e = intervals[mb_idx]
    mk = sorted([k for k in all_kernels if s<=k["ts"]<=e], key=lambda x: x["ts"])
    groups = group_adjacent(mk)
    layers = detect_fn(groups)
    if len(layers) < 2: return []
    return merge_fn(groups, layers[1])

fwd_ops = get_layer_ops(fwd_intervals, detect_fwd_layers, merge_fwd, 2)
rpt_b = gen_layer_report("Report B: Forward Layer Analysis", fwd_ops)


# ─── Report C: Backward Layer Analysis ──────────────────────────────────────

print("Generating Report C (Backward Layer)...")
bwd_ops = get_layer_ops(bwd_intervals, detect_bwd_layers, merge_bwd, 1)
rpt_c = gen_layer_report("Report C: Backward Layer Analysis", bwd_ops)

# FWD vs BWD comparison
if fwd_ops and bwd_ops:
    fwd_by_cat = defaultdict(float)
    bwd_by_cat = defaultdict(float)
    for o in fwd_ops: fwd_by_cat[o["label"]] += o["ksum"]
    for o in bwd_ops: bwd_by_cat[o["label"]] += o["ksum"]

    fwd_t = sum(o["ksum"] for o in fwd_ops)
    bwd_t = sum(o["ksum"] for o in bwd_ops)

    rpt_c += "\n## FWD vs BWD Comparison\n\n"
    rpt_c += "| Metric | FWD | BWD | BWD/FWD |\n"
    rpt_c += "|--------|-----|-----|--------|\n"
    rpt_c += f"| Kernel time | {fwd_t/1000:.2f} ms | {bwd_t/1000:.2f} ms | {bwd_t/fwd_t:.2f}× |\n"

    fwd_expert = sum(o["ksum"] for o in fwd_ops if "GEMM-BF16" in o["cat"] and o["n"]>=32)
    bwd_expert = sum(o["ksum"] for o in bwd_ops if "GEMM-BF16" in o["cat"] and o["n"]>=64)
    if fwd_expert>0:
        rpt_c += f"| Expert GEMM | {fwd_expert/1000:.2f} ms | {bwd_expert/1000:.2f} ms | {bwd_expert/fwd_expert:.2f}× |\n"

    fwd_attn = sum(o["ksum"] for o in fwd_ops if o["cat"]=="Attn-FWD")
    bwd_attn = sum(o["ksum"] for o in bwd_ops if o["cat"]=="Attn-BWD")
    if fwd_attn>0:
        rpt_c += f"| Attention | {fwd_attn/1000:.2f} ms | {bwd_attn/1000:.2f} ms | {bwd_attn/fwd_attn:.2f}× |\n"

    fwd_ep = sum(o["ksum"] for o in fwd_ops if o["cat"]=="DeepEP")
    bwd_ep = sum(o["ksum"] for o in bwd_ops if o["cat"]=="DeepEP")
    if fwd_ep>0:
        rpt_c += f"| DeepEP | {fwd_ep/1000:.2f} ms | {bwd_ep/1000:.2f} ms | {bwd_ep/fwd_ep:.2f}× |\n"
    rpt_c += "\n"
print("Reports B/C done")


# ─── Bottleneck Analysis ─────────────────────────────────────────────────────

print("Generating Bottleneck Analysis...")
phase_cats = {"Forward": defaultdict(list), "Backward": defaultdict(list), "Other": defaultdict(list)}
for k in all_kernels:
    phase_cats[assign_phase(k["ts"], fwd_intervals, bwd_intervals)][k["cat"]].append(k)

all_cat_total = defaultdict(float)
for p in phase_cats:
    for c, kk in phase_cats[p].items():
        all_cat_total[c] += sum(k["dur"] for k in kk)

bf16_pct = all_cat_total["GEMM-BF16"]/total_kernel*100
nccl_other = sum(k["dur"] for k in phase_cats["Other"].get("NCCL",[]))
nccl_other_pct = nccl_other/total_kernel*100
deepep_pct = all_cat_total["DeepEP"]/total_kernel*100
elem_pct = all_cat_total["Elementwise"]/total_kernel*100

rpt_bot = ["# Bottleneck Analysis & Optimization Recommendations\n",
    "| # | Check | Value | Status | Recommendation |",
    "|---|-------|-------|--------|----------------|"]

st = "🔴 HIGH" if bf16_pct>40 else ("🟡 MEDIUM" if bf16_pct>20 else "🟢 OK")
rpt_bot.append(f"| 1 | GEMM-BF16 dominance | {bf16_pct:.1f}% | {st} | `moe_use_legacy_grouped_gemm=False` → FP8 |")
st = "🔴 HIGH" if nccl_other_pct>15 else ("🟡 MEDIUM" if nccl_other_pct>5 else "🟢 OK")
rpt_bot.append(f"| 2 | NCCL in Other phase | {nccl_other_pct:.1f}% | {st} | Enable `overlap_p2p_comm_warmup_flush` |")
st = "🟡 CHECK" if deepep_pct>5 else "🟢 OK"
rpt_bot.append(f"| 3 | DeepEP overhead | {deepep_pct:.1f}% | {st} | `turbo_sync_free_moe_stage >= 2` |")
st = "🔴 HIGH" if elem_pct>5 else ("🟡 MEDIUM" if elem_pct>2 else "🟢 OK")
rpt_bot.append(f"| 4 | Elementwise ops | {elem_pct:.1f}% | {st} | Check `bias_swiglu_fusion`, `moe_permute_fusion` |")

if fwd_ops:
    for o in fwd_ops:
        if o["cat"]=="GEMM-BF16" and o["n"]>=32:
            ovlp = o["ksum"]/o["wall"] if o["wall"]>0 else 0
            st = "🟢 OK" if ovlp>=1.5 else "🟡 CHECK"
            rpt_bot.append(f"| 5 | Expert GEMM overlap | {ovlp:.2f}× | {st} | {ovlp:.1f}× multi-stream parallel |")
            break

rpt_bot.extend(["", "\n## Full Category Time Budget\n",
    "| Category | Total (ms) | % of Step |", "|----------|-----------|-----------|"])
for cat, t in sorted(all_cat_total.items(), key=lambda x: -x[1]):
    rpt_bot.append(f"| {cat} | {t/1000:.2f} | {t/total_kernel*100:.1f}% |")
rpt_bot.append(f"| **Total** | **{total_kernel/1000:.2f}** | **100%** |")
print("Bottleneck done")


# ─── Write Markdown ──────────────────────────────────────────────────────────

def parse_exp_config(exp_name):
    key_map = {"PP":"PP","EP":"EP","mbs":"MBS","MBS":"MBS","gbs":"GBS","GBS":"GBS",
               "vpp":"VPP","VPP":"VPP","turbodeepep":"TurboDeepEP","legacygg":"LegacyGG",
               "recompute_num_layers":"Recompute layers","profile":"Profile"}
    config = []
    for part in exp_name.split("-"):
        if "_" in part:
            for kk in key_map:
                if part.lower().startswith(kk.lower()+"_"):
                    v = part[len(kk)+1:]
                    if v.lower() in ("true","false"): v = v.capitalize()
                    config.append((key_map[kk], v)); break
    return config

md_path = os.path.join(OUTPUT_DIR, "profile_analysis.md")
with open(md_path, "w") as f:
    f.write("# Profile Analysis\n\n| Item | Value |\n|------|-------|\n")
    f.write(f"| Trace | `{os.path.basename(TRACE_FILE)}` |\n")
    f.write(f"| Step | {step['name']} ({SD/1000:.2f} ms) |\n")
    for dk, dv in parse_exp_config(EXP_NAME): f.write(f"| {dk} | {dv} |\n")
    f.write("\n---\n\n")
    f.write("\n".join(rpt_a) + "\n\n---\n\n")
    f.write(rpt_b + "\n\n---\n\n")
    f.write(rpt_c + "\n\n---\n\n")
    f.write("\n".join(rpt_bot) + "\n")
print(f"\nMarkdown: {md_path}")


# ─── HTML Timeline ───────────────────────────────────────────────────────────

print("Generating HTML timeline...")

def ops_to_html_data(ops):
    if not ops: return []
    base = ops[0]["ws"]
    total_k = sum(o["ksum"] for o in ops)
    return [{"l":o["label"], "s":round((o["ws"]-base)/1000,4), "e":round((o["we"]-base)/1000,4),
             "w":round(o["wall"]/1000,4), "k":round(o["ksum"]/1000,4), "n":o["n"],
             "g":cat_group(o["cat"]), "c":cat_color(o["cat"]),
             "pct":round(o["ksum"]/total_k*100,1) if total_k>0 else 0,
             "ovlp":round(o["ksum"]/o["wall"],2) if o["wall"]>0 else 0} for o in ops]

fwd_data = ops_to_html_data(fwd_ops)
bwd_data = ops_to_html_data(bwd_ops)
fwd_tk = sum(d["k"] for d in fwd_data); bwd_tk = sum(d["k"] for d in bwd_data)
fwd_tw = fwd_data[-1]["e"]-fwd_data[0]["s"] if fwd_data else 0
bwd_tw = bwd_data[-1]["e"]-bwd_data[0]["s"] if bwd_data else 0

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DSV3 Layer Timeline</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0e17;color:#e0e6f0;font-family:'DM Sans',sans-serif;padding:32px;min-height:100vh}}
h1{{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:#7dd3fc;margin-bottom:4px}}
.sub{{font-size:13px;color:#64748b;margin-bottom:28px;font-family:'JetBrains Mono',monospace}}
.tabs{{display:flex;gap:0;margin-bottom:24px}}
.tab{{padding:10px 28px;cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;background:transparent;color:#64748b;border:none;border-bottom:2px solid transparent;transition:all .15s}}
.tab.active{{background:#1e293b;color:#7dd3fc;border-bottom:2px solid #7dd3fc;border-radius:6px 6px 0 0}}
.tab:hover:not(.active){{color:#94a3b8}}
.tab-content{{display:none}}.tab-content.active{{display:block}}
.row{{display:flex;align-items:center;margin-bottom:3px;height:34px}}
.lbl{{width:210px;min-width:210px;font-family:'JetBrains Mono',monospace;font-size:11px;text-align:right;padding-right:14px;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.bc{{flex:1;position:relative;height:26px}}
.bar{{position:absolute;height:26px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.6);min-width:3px;cursor:default;transition:filter .1s}}
.bar:hover{{filter:brightness(1.3);outline:2px solid #fff4}}
.gap{{position:absolute;height:26px;background:repeating-linear-gradient(90deg,transparent,transparent 4px,#ffffff06 4px,#ffffff06 8px);border-radius:3px}}
.axis{{display:flex;margin-top:10px;padding-left:210px;height:22px;position:relative}}
.tick{{position:absolute;font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;transform:translateX(-50%)}}
.grid{{position:absolute;width:1px;background:#1e293b;bottom:0;transform:translateX(-50%);pointer-events:none;z-index:-1}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;margin-top:24px;padding-left:210px}}
.li{{display:flex;align-items:center;gap:6px;font-size:12px;color:#94a3b8}}
.ld{{width:14px;height:14px;border-radius:3px}}
table{{margin-top:32px;border-collapse:collapse;width:100%;max-width:1000px;font-family:'JetBrains Mono',monospace;font-size:11px}}
th{{text-align:left;color:#7dd3fc;font-weight:600;padding:7px 10px;border-bottom:2px solid #1e293b}}
td{{padding:5px 10px;border-bottom:1px solid #1e293b;color:#cbd5e1}}
tr:hover td{{background:#1e293b44}}
.bi{{height:16px;border-radius:3px;min-width:1px}}
.note{{margin-top:20px;font-size:12px;color:#64748b;line-height:1.6;max-width:900px;padding-left:210px}}
.note b{{color:#94a3b8}}.hl{{color:#f59e0b;font-weight:600}}
.tooltip{{display:none;position:fixed;background:#0f172a;border:1px solid #334155;border-radius:6px;padding:8px 12px;font-family:'JetBrains Mono',monospace;font-size:11px;z-index:1000;pointer-events:none;color:#e0e6f0;box-shadow:0 4px 12px rgba(0,0,0,.5)}}
</style></head><body>
<h1>DSV3 Single Transformer Layer Timeline</h1>
<div class="sub">{EXP_NAME}<br>PP=4 EP=8 MBS=2 | LegacyGG=True TurboDeepEP=True</div>
<div class="tabs"><div class="tab active" onclick="switchTab('fwd',this)">Forward</div><div class="tab" onclick="switchTab('bwd',this)">Backward</div></div>
<div id="fwd" class="tab-content active"></div><div id="bwd" class="tab-content"></div>
<div id="tooltip" class="tooltip"></div>
<script>
const C={{norm:"#6366f1",cast:"#8b5cf6",fp8:"#06b6d4",rope:"#f59e0b",attn:"#ef4444",bf16:"#10b981",moe:"#f97316",ep:"#ec4899",tri:"#a3e635",nccl:"#e74c3c",opt:"#7f8c8d",elem:"#f1c40f",mem:"#64748b",other:"#94a3b8"}};
const legendMap={{"norm":"RMSNorm","cast":"FP8 Cast","fp8":"GEMM FP8","rope":"RoPE","attn":"Flash Attention","bf16":"GEMM BF16 (Expert)","moe":"MoE Control","ep":"DeepEP","tri":"Triton Fused","elem":"Elementwise","other":"Other"}};
const fwdData={json.dumps(fwd_data)};const bwdData={json.dumps(bwd_data)};
const fwdTotalK={fwd_tk:.3f},bwdTotalK={bwd_tk:.3f},fwdTotalW={fwd_tw:.3f},bwdTotalW={bwd_tw:.3f};
function switchTab(id,el){{document.querySelectorAll('.tab-content').forEach(e=>e.classList.remove('active'));document.querySelectorAll('.tab').forEach(e=>e.classList.remove('active'));document.getElementById(id).classList.add('active');el.classList.add('active');}}
function renderTimeline(cid,data,totalK,totalW){{
  const ct=document.getElementById(cid);if(!data.length){{ct.innerHTML='<p style="color:#64748b">No data</p>';return;}}
  const tMax=Math.max(...data.map(d=>d.e)),chartW=750,scale=chartW/tMax,rowH=data.length*37+40;
  let h='<div style="display:flex;gap:24px;padding-left:210px;font-family:JetBrains Mono,monospace;font-size:12px;margin-bottom:16px">';
  h+=`<span>Kernel: <span class="hl">${{totalK.toFixed(2)}} ms</span></span><span>Wall: <span class="hl">${{totalW.toFixed(2)}} ms</span></span><span>Operators: <span class="hl">${{data.length}}</span></span></div>`;
  h+='<div style="position:relative">';
  const tickStep=Math.max(0.5,Math.ceil(tMax/8*2)/2);
  for(let t=0;t<=tMax+0.01;t+=tickStep){{const x=210+t*scale;h+=`<div class="grid" style="left:${{x}}px;height:${{rowH}}px;top:0"></div>`;}}
  data.forEach((d,i)=>{{const left=d.s*scale,width=Math.max(3,(d.e-d.s)*scale),txt=width>50?d.k.toFixed(2):'';
    h+=`<div class="row"><div class="lbl">${{d.l}}</div><div class="bc">`;
    if(i>0){{const gap=d.s-data[i-1].e;if(gap>0.02){{const gL=data[i-1].e*scale,gW=Math.max(2,gap*scale);h+=`<div class="gap" style="left:${{gL}}px;width:${{gW}}px"></div>`;}}}}
    h+=`<div class="bar" style="left:${{left}}px;width:${{width}}px;background:${{d.c}}" onmousemove="showTip(event,'${{d.l.replace(/'/g,"\\\\'")}}',{{'w':${{d.w}},'k':${{d.k}},'n':${{d.n}},'o':${{d.ovlp}}}})" onmouseout="hideTip()">${{txt}}</div></div></div>`;
  }});
  h+=`<div class="axis">`;for(let t=0;t<=tMax+0.01;t+=tickStep){{h+=`<div class="tick" style="left:${{t*scale}}px">${{t.toFixed(1)}}ms</div>`;}}h+=`</div></div>`;
  const usedGroups=new Set(data.map(d=>d.g));h+=`<div class="legend">`;
  Object.entries(C).forEach(([g,c])=>{{if(usedGroups.has(g))h+=`<div class="li"><div class="ld" style="background:${{c}}"></div>${{legendMap[g]||g}}</div>`;}});h+=`</div>`;
  h+=`<table><tr><th>#</th><th>Operator</th><th>Cnt</th><th>Wall(ms)</th><th>Kernel(ms)</th><th>%</th><th>Overlap</th><th style="width:120px"></th></tr>`;
  data.forEach((d,i)=>{{const barW=Math.max(1,d.pct*1.2),isBold=d.pct>=5,s=isBold?'font-weight:700':'';
    h+=`<tr><td>${{i+1}}</td><td style="color:${{d.c}};${{s}}">${{d.l}}</td><td>${{d.n}}</td><td>${{d.w.toFixed(3)}}</td><td>${{d.k.toFixed(3)}}</td><td>${{d.pct}}%</td><td>${{d.ovlp>1.1?d.ovlp+'×':'-'}}</td><td><div class="bi" style="width:${{barW}}px;background:${{d.c}}"></div></td></tr>`;
  }});
  h+=`<tr style="font-weight:700"><td></td><td style="color:#7dd3fc">Total</td><td></td><td>${{totalW.toFixed(3)}}</td><td>${{totalK.toFixed(3)}}</td><td>100%</td><td></td><td></td></tr></table>`;
  const expertOps=data.filter(d=>d.g==='bf16'&&d.n>=32);h+=`<div class="note"><b>Notes:</b><br>`;
  if(expertOps.length){{const mo=Math.max(...expertOps.map(d=>d.ovlp));h+=`• Expert GEMM overlap: <span class="hl">${{mo.toFixed(2)}}×</span> (2-stream parallel)<br>`;}}
  h+=`• <span class="hl">Overlap > 1.0</span> = multi-stream parallelism</div>`;ct.innerHTML=h;
}}
function showTip(ev,name,d){{const t=document.getElementById('tooltip');t.style.display='block';t.style.left=(ev.clientX+15)+'px';t.style.top=(ev.clientY+15)+'px';t.innerHTML=`<b>${{name}}</b><br>Wall: ${{d.w.toFixed(3)}} ms<br>Kernel: ${{d.k.toFixed(3)}} ms<br>Count: ${{d.n}}<br>Overlap: ${{d.o}}×`;}}
function hideTip(){{document.getElementById('tooltip').style.display='none';}}
renderTimeline('fwd',fwdData,fwdTotalK,fwdTotalW);renderTimeline('bwd',bwdData,bwdTotalK,bwdTotalW);
</script></body></html>"""

html_path = os.path.join(OUTPUT_DIR, "layer_timeline.html")
with open(html_path, "w") as f: f.write(html)
print(f"HTML: {html_path}")
print(f"\n{'='*70}\nDone! Markdown: {md_path} | HTML: {html_path}\n{'='*70}")
