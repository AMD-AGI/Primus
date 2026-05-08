#!/usr/bin/env python3
"""Render the 4 architecture PNGs directly with Pillow.

We render via Pillow (not SVG) because the workspace's editor pipeline
sometimes corrupts multi-byte UTF-8 in arbitrary text files, and we need
CJK glyphs to render correctly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

FONT_REG = "/home/xiewen12/.local/share/fonts/NotoSansSC-Regular.otf"
FONT_BOLD = FONT_REG  # we only have Regular; use it for both

OUT_DIR = os.path.join(os.path.dirname(__file__), "diagrams")
os.makedirs(OUT_DIR, exist_ok=True)


def font(sz: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, sz)


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


def rounded_rect(d: ImageDraw.ImageDraw, b: Box, fill, outline, width=2, radius=10):
    d.rounded_rectangle([b.x, b.y, b.right, b.bottom], radius=radius, fill=fill, outline=outline, width=width)


def text_center(d: ImageDraw.ImageDraw, b: Box, text: str, f: ImageFont.FreeTypeFont, color="#222"):
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((b.cx - tw // 2, b.cy - th // 2 - bbox[1]), text, font=f, fill=color)


def text_left(d: ImageDraw.ImageDraw, x: int, y: int, text: str, f, color="#222"):
    d.text((x, y), text, font=f, fill=color)


def arrow(d, x1, y1, x2, y2, color="#444", width=2, head=8, dashed=False):
    if dashed:
        # Manual dashed line
        import math

        dx, dy = x2 - x1, y2 - y1
        dist = max(1, int(math.hypot(dx, dy)))
        ux, uy = dx / dist, dy / dist
        seg = 6
        gap = 4
        pos = 0
        while pos < dist:
            sx = x1 + ux * pos
            sy = y1 + uy * pos
            ex = x1 + ux * min(pos + seg, dist)
            ey = y1 + uy * min(pos + seg, dist)
            d.line([(sx, sy), (ex, ey)], fill=color, width=width)
            pos += seg + gap
    else:
        d.line([(x1, y1), (x2, y2)], fill=color, width=width)
    # Arrow head
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    p1 = (x2 - head * math.cos(ang - 0.5), y2 - head * math.sin(ang - 0.5))
    p2 = (x2 - head * math.cos(ang + 0.5), y2 - head * math.sin(ang + 0.5))
    d.polygon([(x2, y2), p1, p2], fill=color)


# ----------------------------------------------------------------------
# Diagram 1 — CSA (Compressed Sparse Attention)
# ----------------------------------------------------------------------


def render_csa():
    W, H = 1400, 920
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # Title
    d.text(
        (W // 2 - 360, 24),
        "CSA — Compressed Sparse Attention (compress_ratio = 4)",
        font=font(26, True),
        fill="#222",
    )
    d.text(
        (W // 2 - 470, 60),
        "每 4 个 raw token 池化成 1 个 compressed KV，再用 Indexer per-query 选 top-K（V4-Flash=512）",
        font=font(15),
        fill="#555",
    )

    # Input box
    inp = Box(560, 100, 280, 50)
    rounded_rect(d, inp, "#f5f5f5", "#333")
    text_center(d, inp, "hidden_states  [B, S, D]", font(16))

    # Q path
    q_panel = Box(60, 200, 360, 360)
    rounded_rect(d, q_panel, "#fff5e0", "#a06000", width=2, radius=12)
    text_center(d, Box(60, 210, 360, 30), "Q path", font(15, True), color="#603000")
    items = [
        ("wq_a:  D → q_lora_rank", 250),
        ("q_norm  RMSNorm", 305),
        ("wq_b:  q_lora_rank → H · D_h", 360),
        ("per-head rsqrt + partial RoPE(64)", 415),
    ]
    for txt, y in items:
        b = Box(80, y, 320, 38)
        rounded_rect(d, b, "white", "#a06000", width=1, radius=6)
        text_center(d, b, txt, font(15))
    text_center(d, Box(60, 470, 360, 25), "主路径 RoPE base = compress_rope_theta", font(13), color="#603000")
    text_center(d, Box(60, 495, 360, 25), "（因为本层 compress_ratio ≠ 0）", font(13), color="#603000")
    text_center(d, Box(60, 525, 360, 25), "→ Q [B, H=64, S, head_dim=512]", font(13), color="#603000")

    # KV path
    kv_panel = Box(460, 200, 420, 360)
    rounded_rect(d, kv_panel, "#e8f5ff", "#005080", width=2, radius=12)
    text_center(
        d, Box(460, 210, 420, 30), "KV path  (single KV head, K = V)", font(15, True), color="#003060"
    )
    items = [
        ("wkv:  D → head_dim   (single latent)", 250),
        ("kv_norm + partial RoPE(64)", 305),
    ]
    for txt, y in items:
        b = Box(480, y, 380, 38)
        rounded_rect(d, b, "white", "#005080", width=1, radius=6)
        text_center(d, b, txt, font(15))

    # SWA buffer
    sw = Box(480, 360, 180, 60)
    rounded_rect(d, sw, "white", "#005080", width=1, radius=6)
    text_center(d, Box(480, 365, 180, 25), "SWA buffer", font(14, True))
    text_center(d, Box(480, 388, 180, 25), "last 128 tokens", font(12), color="#005080")

    # Compressor
    cmp_box = Box(680, 360, 180, 60)
    rounded_rect(d, cmp_box, "white", "#005080", width=1, radius=6)
    text_center(d, Box(680, 365, 180, 25), "Compressor", font(14, True))
    text_center(d, Box(680, 388, 180, 25), "overlap pool (ratio 4)", font(12), color="#005080")

    # cat box
    cat = Box(480, 440, 380, 60)
    rounded_rect(d, cat, "#dff5dd", "#206020", width=2, radius=8)
    text_center(d, Box(480, 445, 380, 25), "cat(SW_KV[128], compressed_KV[S/4])", font(14, True))
    text_center(
        d, Box(480, 470, 380, 25), "→ full_kv  [B, 1, 128 + S/4, head_dim]", font(13), color="#205020"
    )

    # Indexer
    idx_panel = Box(920, 200, 420, 360)
    rounded_rect(d, idx_panel, "#f0e6ff", "#503080", width=2, radius=12)
    text_center(d, Box(920, 210, 420, 30), "Indexer  (only for ratio == 4)", font(15, True), color="#301060")
    items = [
        ("mini Compressor @ index_head_dim=128", 250),
        ("wq_b: q_lora_rank → 64·128  (index Q)", 305),
        ("scores = ReLU(Q · K^T) · weights_proj", 360),
    ]
    for txt, y in items:
        b = Box(940, y, 380, 38)
        rounded_rect(d, b, "white", "#503080", width=1, radius=6)
        text_center(d, b, txt, font(15))
    topk_box = Box(940, 415, 380, 50)
    rounded_rect(d, topk_box, "white", "#503080", width=1, radius=6)
    text_center(d, Box(940, 418, 380, 25), "topk over compressed pool", font(14, True))
    text_center(
        d, Box(940, 442, 380, 25), "indices [B, S, K] (-1 = causally invalid)", font(12), color="#503080"
    )

    # Sparse attention
    sa = Box(380, 620, 640, 140)
    rounded_rect(d, sa, "#e8ffe8", "#205020", width=2, radius=12)
    text_center(d, Box(380, 630, 640, 30), "sparse_attn  (dense + mask)", font(16, True), color="#103010")
    text_center(
        d,
        Box(380, 670, 640, 25),
        "probs = softmax([ Q · K^T · scaling + mask ; sinks[h] ])[..., :-1]",
        font(14),
    )
    text_center(d, Box(380, 695, 640, 25), "o = probs · V                (注意 attn_sink 拼接)", font(14))
    text_center(
        d,
        Box(380, 723, 640, 22),
        "mask = causal + sliding-window + per-query top-K from Indexer",
        font(12),
        color="#103010",
    )

    # Output
    out = Box(380, 790, 640, 70)
    rounded_rect(d, out, "white", "#444", width=2, radius=8)
    text_center(
        d,
        Box(380, 800, 640, 25),
        "inverse RoPE → wo_a (grouped low-rank) → wo_b → output [B, S, D]",
        font(15),
    )
    text_center(
        d,
        Box(380, 828, 640, 22),
        "attn_sink: 每个 head 一个可学习标量，作为 softmax 的 'opt-out' 列",
        font(12),
        color="#666",
    )

    # Arrows
    arrow(d, inp.cx, inp.bottom, q_panel.cx, q_panel.y)
    arrow(d, inp.cx, inp.bottom, kv_panel.cx, kv_panel.y)
    arrow(d, inp.cx, inp.bottom, idx_panel.cx, idx_panel.y)
    arrow(d, q_panel.cx, q_panel.bottom, sa.x + 40, sa.y)
    arrow(d, kv_panel.cx, kv_panel.bottom, sa.cx, sa.y)
    arrow(d, idx_panel.cx, idx_panel.bottom, sa.right - 40, sa.y, dashed=True)
    arrow(d, sa.cx, sa.bottom, out.cx, out.y)

    img.save(os.path.join(OUT_DIR, "csa.png"), "PNG", optimize=True)
    print(f"  csa.png  ({W}x{H})")


# ----------------------------------------------------------------------
# Diagram 2 — HCA (Heavily Compressed Attention)
# ----------------------------------------------------------------------


def render_hca():
    W, H = 1400, 920
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    d.text(
        (W // 2 - 380, 24),
        "HCA — Heavily Compressed Attention (compress_ratio = 128)",
        font=font(26, True),
        fill="#222",
    )
    d.text(
        (W // 2 - 540, 60),
        "每 128 个 raw token 池化成 1 个 compressed KV，所有 query 共享整个 pool（无 Indexer），仅按因果性截断",
        font=font(15),
        fill="#555",
    )

    inp = Box(560, 100, 280, 50)
    rounded_rect(d, inp, "#f5f5f5", "#333")
    text_center(d, inp, "hidden_states  [B, S, D]", font(16))

    # Q path
    q_panel = Box(60, 200, 380, 320)
    rounded_rect(d, q_panel, "#fff5e0", "#a06000", width=2, radius=12)
    text_center(d, Box(60, 210, 380, 30), "Q path  (same as CSA)", font(15, True), color="#603000")
    for txt, y in [
        ("wq_a → q_norm → wq_b", 250),
        ("per-head rsqrt + partial RoPE", 305),
    ]:
        b = Box(80, y, 340, 38)
        rounded_rect(d, b, "white", "#a06000", width=1, radius=6)
        text_center(d, b, txt, font(15))
    text_center(
        d, Box(60, 380, 380, 25), "RoPE base = compress_rope_theta = 160000", font(13), color="#603000"
    )
    text_center(d, Box(60, 405, 380, 25), "YaRN scaling factor = 16", font(13), color="#603000")
    text_center(d, Box(60, 440, 380, 25), "→ Q [B, H=64, S, head_dim=512]", font(13), color="#603000")

    # KV main
    kv_panel = Box(480, 200, 380, 320)
    rounded_rect(d, kv_panel, "#e8f5ff", "#005080", width=2, radius=12)
    text_center(d, Box(480, 210, 380, 30), "KV main path", font(15, True), color="#003060")
    for txt, y in [
        ("wkv:  D → 512  (single KV)", 250),
        ("kv_norm + partial RoPE", 305),
        ("SWA buffer  [B, 1, 128, 512]", 360),
    ]:
        b = Box(500, y, 340, 38)
        rounded_rect(d, b, "white", "#005080", width=1, radius=6)
        text_center(d, b, txt, font(15))
    text_center(d, Box(480, 425, 380, 25), "仅保留最近 128 个 token 的 KV", font(13), color="#003060")
    text_center(d, Box(480, 450, 380, 25), "提供短程精确注意力", font(13), color="#003060")

    # Compressor (HCA)
    cmp_panel = Box(900, 200, 460, 320)
    rounded_rect(d, cmp_panel, "#fde0e0", "#a02020", width=2, radius=12)
    text_center(
        d, Box(900, 210, 460, 30), "Compressor  (no overlap, ratio = 128)", font(15, True), color="#601010"
    )
    for txt, y in [
        ("wkv: D → head_dim;  wgate: D → head_dim", 250),
        ("reshape [B, S/128, 128, head_dim]", 305),
        ("softmax(score+ape, dim=ratio) · KV → pooled", 360),
    ]:
        b = Box(920, y, 420, 38)
        rounded_rect(d, b, "white", "#a02020", width=1, radius=6)
        text_center(d, b, txt, font(15))
    rope_box = Box(920, 415, 420, 50)
    rounded_rect(d, rope_box, "white", "#a02020", width=1, radius=6)
    text_center(d, Box(920, 418, 420, 25), "kv_norm + partial RoPE @ θ=160000", font(14, True))
    text_center(d, Box(920, 443, 420, 25), "→ pooled_KV [B, 1, S/128, 512]", font(12), color="#601010")

    # KV concat
    cat = Box(480, 560, 880, 56)
    rounded_rect(d, cat, "#dff5dd", "#205020", width=2, radius=8)
    text_center(
        d, Box(480, 568, 880, 25), "full_kv = cat( SW_KV[128] , pooled_KV[S/128] , dim=2 )", font(15, True)
    )
    text_center(
        d, Box(480, 593, 880, 22), "总 KV 长度 ≈ 128 + S/128，相比原始 S 的 1/128", font(12), color="#205020"
    )

    # mask explanation
    mask = Box(60, 560, 380, 156)
    rounded_rect(d, mask, "#fff8e6", "#a06000", width=2, radius=8)
    text_center(d, Box(60, 568, 380, 25), "压缩侧 mask 规则", font(15, True), color="#603000")
    text_center(d, Box(60, 600, 380, 22), "每个 query q 只能看到", font(13), color="#403000")
    text_center(d, Box(60, 625, 380, 22), "压缩位置 p 满足:", font(13), color="#403000")
    text_center(d, Box(60, 655, 380, 25), "p < (q + 1) // ratio", font(15, True), color="#603000")
    text_center(
        d, Box(60, 685, 380, 22), "（与 CSA 不同，HCA 所有 query 共享整个 pool）", font(12), color="#603000"
    )

    # Attention
    sa = Box(480, 640, 880, 130)
    rounded_rect(d, sa, "#e8ffe8", "#205020", width=2, radius=12)
    text_center(d, Box(480, 650, 880, 30), "eager_attention_with_sink", font(16, True), color="#103010")
    text_center(
        d,
        Box(480, 685, 880, 25),
        "scores = Q · full_kv^T · scaling + (causal + SWA + compress) mask",
        font(14),
    )
    text_center(d, Box(480, 710, 880, 25), "probs = softmax([ scores ; sinks[h] ])[..., :-1]", font(14))
    text_center(
        d,
        Box(480, 740, 880, 22),
        "同样的 attn_sink 机制；区别只在压缩侧 mask 的 '全可见' 模式",
        font(12),
        color="#103010",
    )

    # output
    out = Box(380, 800, 700, 70)
    rounded_rect(d, out, "white", "#444", width=2, radius=8)
    text_center(
        d,
        Box(380, 810, 700, 25),
        "inverse RoPE → wo_a (grouped low-rank) → wo_b → output [B, S, D]",
        font(15),
    )
    text_center(
        d,
        Box(380, 838, 700, 22),
        "CSA / HCA 共享同一套 output 投影；区别只在 KV 来源",
        font(12),
        color="#666",
    )

    # arrows
    arrow(d, inp.cx, inp.bottom, q_panel.cx, q_panel.y)
    arrow(d, inp.cx, inp.bottom, kv_panel.cx, kv_panel.y)
    arrow(d, inp.cx, inp.bottom, cmp_panel.cx, cmp_panel.y)
    arrow(d, kv_panel.cx, kv_panel.bottom, cat.cx - 100, cat.y)
    arrow(d, cmp_panel.cx, cmp_panel.bottom, cat.cx + 100, cat.y)
    arrow(d, cat.cx, cat.bottom, sa.cx, sa.y)
    arrow(d, sa.cx, sa.bottom, out.cx, out.y)

    img.save(os.path.join(OUT_DIR, "hca.png"), "PNG", optimize=True)
    print(f"  hca.png  ({W}x{H})")


# ----------------------------------------------------------------------
# Diagram 3 — mHC (Manifold-Constrained Hyper-Connections)
# ----------------------------------------------------------------------


def render_mhc():
    W, H = 1400, 1020
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    d.text(
        (W // 2 - 460, 22),
        "mHC — Manifold-Constrained Hyper-Connections (hc_mult = 4)",
        font=font(26, True),
        fill="#222",
    )
    d.text(
        (W // 2 - 600, 58),
        "每个 block 用 Sinkhorn-归一化的 (pre, post, comb) 三个权重，对 4 路并行 hidden 流做 manifold 约束的混合",
        font=font(15),
        fill="#555",
    )

    # Streams
    streams = Box(80, 110, 1240, 100)
    rounded_rect(d, streams, "#f0f4ff", "#1a4a8a", width=2, radius=12)
    text_center(
        d, Box(80, 120, 1240, 28), "x : [B, S, hc_mult=4, D]   (HC streams)", font(16, True), color="#0a2050"
    )
    for i in range(4):
        b = Box(220 + i * 200, 165, 160, 32)
        rounded_rect(d, b, "white", "#1a4a8a", width=1, radius=6)
        text_center(d, b, f"stream {i}  [B, S, D]", font(13))
    text_center(d, Box(1040, 170, 280, 25), "(由 embed unsqueeze + expand 而来)", font(12), color="#205050")

    # compute_weights
    cw = Box(80, 240, 1240, 200)
    rounded_rect(d, cw, "#fff0f0", "#a02020", width=2, radius=12)
    text_center(
        d,
        Box(80, 252, 1240, 28),
        "attn_hc.compute_weights( x )   /   ffn_hc.compute_weights( x )",
        font(16, True),
        color="#601010",
    )

    b1 = Box(110, 290, 460, 38)
    rounded_rect(d, b1, "white", "#a02020", width=1, radius=6)
    text_center(d, b1, "flat = x.flatten(2).float()  →  [B, S, 4·D]", font(15))
    b2 = Box(110, 340, 460, 38)
    rounded_rect(d, b2, "white", "#a02020", width=1, radius=6)
    text_center(d, b2, "mix = linear(flat, fn) · rsqrt(flat)  ([B,S,24])", font(15))

    b3 = Box(600, 290, 280, 38)
    rounded_rect(d, b3, "white", "#a02020", width=1, radius=6)
    text_center(d, b3, "pre  = sigmoid(...) + eps  [B,S,4]", font(14))
    b4 = Box(600, 340, 280, 38)
    rounded_rect(d, b4, "white", "#a02020", width=1, radius=6)
    text_center(d, b4, "post = 2 · sigmoid(...)    [B,S,4]", font(14))

    b5 = Box(900, 290, 380, 95)
    rounded_rect(d, b5, "white", "#a02020", width=1, radius=6)
    text_center(d, Box(900, 295, 380, 25), "comb_logit  → softmax(-1) + eps", font(14))
    text_center(d, Box(900, 320, 380, 25), "→ Sinkhorn-Knopp (20 iters)", font(14))
    text_center(d, Box(900, 348, 380, 25), "comb [B, S, 4, 4]", font(13))
    text_center(d, Box(900, 370, 380, 22), "≈ 双随机矩阵 (Birkhoff polytope)", font(12), color="#601010")

    # sub-block
    sb = Box(80, 470, 1240, 320)
    rounded_rect(d, sb, "#e8ffe8", "#205020", width=2, radius=12)
    text_center(d, Box(80, 482, 1240, 28), "Sub-block (Attention or FFN)", font(16, True), color="#103010")

    s1 = Box(120, 530, 280, 60)
    rounded_rect(d, s1, "white", "#205020", width=1, radius=8)
    text_center(d, Box(120, 535, 280, 25), "collapse", font(15, True))
    text_center(d, Box(120, 562, 280, 22), "collapsed = Σ_h pre[h] · x[h]", font(13))

    s2 = Box(420, 530, 200, 60)
    rounded_rect(d, s2, "white", "#205020", width=1, radius=8)
    text_center(d, Box(420, 535, 200, 25), "RMSNorm", font(15, True))
    text_center(d, Box(420, 562, 200, 22), "input_layernorm", font(11))

    s3 = Box(640, 530, 200, 60)
    rounded_rect(d, s3, "white", "#205020", width=1, radius=8)
    text_center(d, Box(640, 535, 200, 25), "f(·)", font(15, True))
    text_center(d, Box(640, 562, 200, 22), "Attention  /  MoE FFN", font(13))

    s4 = Box(860, 530, 320, 60)
    rounded_rect(d, s4, "white", "#205020", width=1, radius=8)
    text_center(d, Box(860, 535, 320, 25), "expand", font(15, True))
    text_center(d, Box(860, 562, 320, 22), "post · f(·) ⊕ comb @ x", font(13))

    formula_box = Box(120, 615, 1160, 150)
    rounded_rect(d, formula_box, "white", "#103010", width=2, radius=8)
    text_center(
        d, Box(120, 624, 1160, 28), "展开公式（new_x : [B, S, 4, D]）", font(16, True), color="#103010"
    )
    text_center(
        d,
        Box(120, 660, 1160, 28),
        "new_x[h]  =  post[h] · f( RMSNorm( Σ_k pre[k] · x[k] ) )  +  Σ_k comb[h, k] · x[k]",
        font(16, True),
        color="#103010",
    )
    text_center(
        d,
        Box(120, 700, 1160, 22),
        "第一项 = 本 sub-block 的输出按 post 加权回写到第 h 路",
        font(13),
        color="#205020",
    )
    text_center(
        d,
        Box(120, 723, 1160, 22),
        "第二项 = 把当前 4 路状态用近似双随机的 comb 重新混合，约束在 manifold 内",
        font(13),
        color="#205020",
    )

    # Output
    out = Box(80, 815, 1240, 56)
    rounded_rect(d, out, "#dff5dd", "#205020", width=2, radius=8)
    text_center(
        d, Box(80, 822, 1240, 25), "→ x : [B, S, 4, D]   送入下一 sub-block (attn 或 ffn)", font(15, True)
    )
    text_center(
        d,
        Box(80, 848, 1240, 22),
        "每个 Block 内 sub-block 序列： [ attn_hc, attention ] → [ ffn_hc, FFN ]，两次 collapse / expand",
        font(12),
        color="#205020",
    )

    head = Box(80, 895, 1240, 100)
    rounded_rect(d, head, "#f0e6ff", "#503080", width=2, radius=8)
    text_center(
        d,
        Box(80, 905, 1240, 25),
        "最后一层之后：HyperHead  (sigmoid 加权和，无 Sinkhorn)",
        font(15, True),
        color="#301060",
    )
    text_center(
        d,
        Box(80, 935, 1240, 25),
        "y = Σ_h sigmoid( linear(x.flatten(2), hc_fn) · scale + base + eps )[h] · x[h]    [B, S, D]",
        font(13),
        color="#301060",
    )
    text_center(
        d,
        Box(80, 965, 1240, 22),
        "把 hc_mult=4 路 collapse 回 1 路，再过 final RMSNorm 和 lm_head",
        font(12),
        color="#503080",
    )

    # arrows between layers
    arrow(d, streams.cx, streams.bottom, cw.cx, cw.y)
    arrow(d, cw.cx, cw.bottom, sb.cx, sb.y)
    arrow(d, s1.right, s1.cy, s2.x, s2.cy)
    arrow(d, s2.right, s2.cy, s3.x, s3.cy)
    arrow(d, s3.right, s3.cy, s4.x, s4.cy)

    img.save(os.path.join(OUT_DIR, "mhc.png"), "PNG", optimize=True)
    print(f"  mhc.png  ({W}x{H})")


# ----------------------------------------------------------------------
# Diagram 4 — Architecture overview
# ----------------------------------------------------------------------


def render_arch():
    W, H = 1600, 1100
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    d.text(
        (W // 2 - 360, 24),
        "DeepSeek-V4-Flash 整体模型架构 (43 layers + 1 MTP)",
        font=font(26, True),
        fill="#222",
    )
    d.text(
        (W // 2 - 580, 60),
        "HC 4-stream + Hybrid Attention (compress_ratios 决定每层走 dense / CSA / HCA) + MoE (前 3 层 hash routing)",
        font=font(14),
        fill="#555",
    )

    # embed
    e1 = Box(620, 110, 360, 44)
    rounded_rect(d, e1, "white", "#222", width=2, radius=8)
    text_center(d, e1, "ParallelEmbedding  (vocab 129280, D=4096)", font(15))

    e2 = Box(620, 170, 360, 44)
    rounded_rect(d, e2, "#f0f4ff", "#1a4a8a", width=2, radius=8)
    text_center(d, e2, "HC expand:  [B,S,D] → [B,S,4,D]", font(15, True))

    # Group panels for layer types
    panels = [
        (
            Box(60, 250, 360, 200),
            "Block 0–2 (num_hash_layers)",
            "#fffbea",
            "#a08000",
            [
                "attn_hc.compute_weights (pre/post/comb)",
                "SelfAttention (compress=0/4 in this band)",
                "ffn_hc.compute_weights",
                "MoE  (Hash Gate: tid2eid lookup)",
            ],
            "前 3 层固定 expert 分配，提高浅层稳定性",
            [None, None, None, ("hash", "#fff8e0", "#a08000")],
        ),
        (
            Box(440, 250, 320, 200),
            "CSA layers (compress_ratio = 4)",
            "#eaf6ff",
            "#0050a0",
            [
                "attn_hc + collapse",
                "SelfAttention + Compressor + Indexer",
                "ffn_hc + collapse",
                "MoE  (sqrtsoftplus + noaux_tc)",
            ],
            "每 4 token pool + per-query top-512 选取",
            [None, ("csa", "#ddf0ff", "#0050a0"), None, ("moe", "#fde0e0", "#a02020")],
        ),
        (
            Box(780, 250, 320, 200),
            "HCA layers (compress_ratio = 128)",
            "#eaf8e8",
            "#207020",
            [
                "attn_hc + collapse",
                "SelfAttention + Compressor (no Indexer)",
                "ffn_hc + collapse",
                "MoE  (sqrtsoftplus + noaux_tc)",
            ],
            "每 128 token pool 一个压缩 KV，因果可见全部",
            [None, ("hca", "#e0f6e0", "#207020"), None, ("moe", "#fde0e0", "#a02020")],
        ),
        (
            Box(1120, 250, 280, 200),
            "Block 42 (compress = 0)",
            "#f6f6f6",
            "#444",
            ["attn_hc + collapse", "SelfAttention (SWA only)", "ffn_hc + collapse", "MoE"],
            "最末层无长程压缩",
            [None, None, None, ("moe", "#fde0e0", "#a02020")],
        ),
    ]

    for panel_box, title, panel_fill, panel_stroke, items, footer, item_styles in panels:
        rounded_rect(d, panel_box, panel_fill, panel_stroke, width=2, radius=12)
        text_center(
            d, Box(panel_box.x, panel_box.y + 8, panel_box.w, 26), title, font(14, True), color=panel_stroke
        )
        y = panel_box.y + 42
        for i, (txt, sty) in enumerate(zip(items, item_styles)):
            b = Box(panel_box.x + 16, y, panel_box.w - 32, 28)
            if sty:
                rounded_rect(d, b, sty[1], sty[2], width=1, radius=4)
            else:
                rounded_rect(
                    d,
                    b,
                    "#f0f4ff" if "hc" in txt else "white",
                    "#1a4a8a" if "hc" in txt else "#444",
                    width=1,
                    radius=4,
                )
            text_center(d, b, txt, font(12))
            y += 32
        text_center(
            d, Box(panel_box.x, panel_box.bottom - 26, panel_box.w, 22), footer, font(11), color=panel_stroke
        )

    # compress_ratios pattern strip
    strip = Box(60, 480, 1480, 100)
    rounded_rect(d, strip, "white", "#444", width=2, radius=8)
    text_center(
        d,
        Box(60, 488, 1480, 25),
        "compress_ratios pattern  (V4-Flash, layer 0 → 42)",
        font(14, True),
        color="#222",
    )

    # ratios sequence
    ratios = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(40)] + [0]
    n = len(ratios)
    cell_w = 1440 / n
    for i, r in enumerate(ratios):
        cx = 80 + i * cell_w
        cb = Box(int(cx), 522, int(cell_w) - 1, 38)
        if r == 0:
            fill, stroke, label = "#f6f6f6", "#444", "0"
        elif r == 4:
            fill, stroke, label = "#ddf0ff", "#0050a0", "4"
        else:
            fill, stroke, label = "#e0f6e0", "#207020", "128"
        d.rectangle([cb.x, cb.y, cb.x + cb.w, cb.y + cb.h], fill=fill, outline=stroke)
        text_center(d, cb, label, font(10))
    text_center(d, Box(60, 565, 100, 18), "layer 0", font(11), color="#333")
    text_center(d, Box(1480, 565, 60, 18), "layer 42", font(11), color="#333")

    # Tail: HyperHead + RMSNorm + lm_head
    hh = Box(540, 610, 520, 50)
    rounded_rect(d, hh, "#f0f4ff", "#1a4a8a", width=2, radius=8)
    text_center(d, hh, "HyperHead  (4 streams → 1)", font(15, True))

    fn = Box(540, 670, 520, 44)
    rounded_rect(d, fn, "white", "#222", width=2, radius=8)
    text_center(d, fn, "final RMSNorm  →  lm_head (D → vocab)", font(15))

    lg = Box(540, 724, 520, 44)
    rounded_rect(d, lg, "white", "#222", width=2, radius=8)
    text_center(d, lg, "logits  [B, S, V=129280]", font(15, True))

    # MTP
    mtp = Box(60, 800, 1480, 130)
    rounded_rect(d, mtp, "#f0e0ff", "#503080", width=2, radius=12)
    text_center(
        d,
        Box(60, 810, 1480, 28),
        "MTP head x num_nextn_predict_layers (= 1)",
        font(15, True),
        color="#301060",
    )

    mtp_items = [
        ("embed", "shared with main model"),
        ("e_proj + h_proj + Block", "shifted-token + last hidden, HC enabled"),
        ("hc_head_fn", "单独 mtp HyperHead (separate from main)"),
        ("RMSNorm + lm_head", "shared with main model"),
        ("NextN logits", "multi-token prediction"),
    ]
    item_w = 1440 // len(mtp_items)
    for i, (h, sub) in enumerate(mtp_items):
        b = Box(80 + i * item_w, 850, item_w - 16, 60)
        rounded_rect(d, b, "white", "#503080", width=1, radius=6)
        text_center(d, Box(b.x, b.y + 8, b.w, 22), h, font(13, True))
        text_center(d, Box(b.x, b.y + 32, b.w, 22), sub, font(11))

    # Arrows
    arrow(d, e1.cx, e1.bottom, e2.cx, e2.y)
    arrow(d, e2.cx, e2.bottom, e2.cx, 250)
    arrow(d, hh.cx, 580, hh.cx, hh.y)  # from strip to HyperHead
    arrow(d, hh.cx, hh.bottom, fn.cx, fn.y)
    arrow(d, fn.cx, fn.bottom, lg.cx, lg.y)
    arrow(d, lg.cx, lg.bottom, mtp.cx, mtp.y)

    img.save(os.path.join(OUT_DIR, "architecture.png"), "PNG", optimize=True)
    print(f"  architecture.png  ({W}x{H})")


if __name__ == "__main__":
    print("Rendering diagrams to:", OUT_DIR)
    render_csa()
    render_hca()
    render_mhc()
    render_arch()
    print("done.")
