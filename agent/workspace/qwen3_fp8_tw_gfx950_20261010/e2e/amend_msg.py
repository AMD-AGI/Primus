"""Insert the end-to-end section into a commit message (before the Tests: paragraph).

Usage: git log -1 --format=%B | python amend_msg.py ROW_KEY > msg.txt
ROW_KEY marks the row of this change and selects the before/after pair.
"""

import sys

ROWS = [
    ("cfg", "submitted FP8 tensorwise config (Turbo e2d9f1d7)", 8894.0, 670.6, 29474, 277, "11.33393"),
    ("act", "+ use_turbo_fused_act_with_probs (Primus flag)", 7915.6, 753.5, 33117, 223, "11.33435"),
    ("base", "Turbo 1103b2df, base of these changes", 7925.4, 752.5, 33077, 222, "11.33427"),
    ("quant", "+ flat tensorwise FP8 quant kernel", 7870.9, 757.7, 33305, 222, "11.33415"),
    ("layout", "+ per-token get_dispatch_layout", 7774.2, 767.2, 33720, 222, "11.33431"),
    ("permute", "+ HIP permute in DeepEPTokenDispatcher", 7643.5, 780.3, 34296, 222, "11.33423"),
    ("cu160", "+ turbo_deepep_num_cu 160 (Primus flag)", 7353.4, 811.1, 35649, 223, "11.33456"),
    ("qknorm", "+ fused qk RMSNorm + RoPE, head_dim 128", 6987.1, 853.6, 37518, 224, "11.33373"),
]
PREV = {"quant": "base", "layout": "quant", "permute": "layout", "qknorm": "cu160"}

key = sys.argv[1]
by_key = {r[0]: r for r in ROWS}
cur, prev = by_key[key], by_key[PREV[key]]

lines = [
    "End-to-end: Qwen3-30B-A3B FP8 tensorwise pretrain on 8x MI355X (EP8, MBS 8,",
    "GBS 512, seq 4096, even routing), mean of iterations 11-20 of 20-iteration",
    "runs; each row adds one change on top of the row above:",
    f"  {'':49s} {'ms/iter':>7s} {'TFLOP/s':>7s} {'tok/s/GPU':>9s} {'mem':>6s} {'loss@20':>8s}",
]
for k, name, ms, tf, tok, mem, loss in ROWS:
    mark = "*" if k == key else " "
    lines.append(f"{mark} {name:49s} {ms:7.1f} {tf:7.1f} {tok:9,d} {mem:4d}GB {loss:>8s}")
d_ms = cur[2] - prev[2]
lines += [
    f"This change (*): {prev[2]:.1f} -> {cur[2]:.1f} ms/iter ({100 * d_ms / prev[2]:+.2f}%),",
    f"{prev[4]:,d} -> {cur[4]:,d} tokens/s/GPU ({100 * (cur[4] / prev[4] - 1):+.2f}%).",
    "Loss differences are within run-to-run noise (about 4e-4 at iteration 20).",
]
section = "\n".join(lines) + "\n"

msg = sys.stdin.read().rstrip("\n") + "\n"
idx = msg.find("\nTests:")
if idx < 0:
    idx = msg.find("\nRequires ")
if idx < 0:
    sys.stdout.write(msg + "\n" + section)
else:
    sys.stdout.write(msg[: idx + 1] + section + msg[idx:])
