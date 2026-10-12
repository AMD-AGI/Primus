#!/usr/bin/env python
"""Build the long-form SFT jsonl used by the gfx942 DeepSeek-V4 recipes.

Non-packed SFT pads every sample to `seq_length` and truncates above it, so rows are built
deliberately LONGER than the target: otherwise a short Alpaca row at 128k would be 99.9%
padding with loss_mask=0, i.e. a benchmark of padding rather than of attention.

Rows are alpaca-format {"instruction", "input", "output"}; the formatter masks the prompt
and supervises the response, so putting the bulk of the text in "output" yields a genuine
supervised span.

Usage:  python prepare_sft_data.py --tokenizer <dir> --out-dir <dir> [--lengths 4096 131072]
"""

import argparse
import json
import os
import sys


def load_corpus(tokenizer_dir):
    """Natural-language text. Falls back to local docs if the Hub is unreachable."""
    try:
        from datasets import load_dataset

        src = load_dataset("tatsu-lab/alpaca", split="train")
        blob = "".join(f"{r['instruction']} {r['input']} {r['output']}\n" for r in src)
        print(f"[data] alpaca: {len(src)} rows, {len(blob)} chars", flush=True)
        return blob
    except Exception as e:  # noqa: BLE001
        print(f"[data] alpaca unavailable ({type(e).__name__}); falling back to repo docs", flush=True)
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        parts = []
        for base, _, files in os.walk(os.path.join(root, "docs")):
            for fn in files:
                if fn.endswith((".md", ".py", ".txt")):
                    try:
                        parts.append(open(os.path.join(base, fn), encoding="utf-8", errors="ignore").read())
                    except OSError:
                        pass
        return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="local HF tokenizer directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768, 131072])
    ap.add_argument("--rows", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    blob = load_corpus(args.tokenizer)
    if len(blob) < 100_000:
        sys.exit(f"[data] corpus too small: {len(blob)} chars")

    # Calibrate chars-per-token once rather than tokenizing gigabytes.
    sample = blob[:400_000]
    cpt = len(sample) / len(tok(sample, add_special_tokens=False)["input_ids"])
    print(f"[data] chars/token = {cpt:.3f}", flush=True)

    cursor = 0

    def take(n_chars, cur):
        out, need = [], n_chars
        while need > 0:
            end = min(cur + need, len(blob))
            out.append(blob[cur:end])
            need -= end - cur
            cur = 0 if end >= len(blob) else end
        return "".join(out), cur

    for target in args.lengths:
        name = f"sft_{target}.jsonl"
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path):
            print(f"[data] {name} exists, skipping", flush=True)
            continue
        n_chars = int(target * 1.35 * cpt)  # headroom so truncation, not padding, sets the length
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(args.rows):
                text, cursor = take(n_chars, cursor)
                f.write(
                    json.dumps(
                        {"instruction": "Continue the following document.", "input": "", "output": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        first = json.loads(open(path, encoding="utf-8").readline())
        ntok = len(tok(first["output"], add_special_tokens=False)["input_ids"])
        ok = "OK" if ntok >= target else "SHORT!"
        print(
            f"[data] [{ok}] {name}: {args.rows} rows, "
            f"{os.path.getsize(path)/2**20:.1f} MB, row0={ntok} tokens (target {target})",
            flush=True,
        )


if __name__ == "__main__":
    main()
