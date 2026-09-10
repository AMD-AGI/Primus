#!/usr/bin/env python
"""Pack variable-length SFT samples into fixed-size THD bins.

The BSHD recipes in this directory deliberately build rows LONGER than the target so that
truncation, not padding, sets the length -- one sequence fills the whole micro-batch. Packed
(THD) training is the opposite problem: many SHORT sequences are concatenated into one flat
token stream and described by ``cu_seqlens``, so attention must not let a token in one
sequence see another.

This reads the same alpaca-format jsonl the BSHD path uses, tokenises each sample at its
NATURAL length, and greedily bin-packs samples into bins of exactly ``--bin-size`` tokens.
Sharing the source data with the BSHD recipe is the point: it makes the two paths
comparable, so a THD run that disagrees with a BSHD run on the same text is a bug rather
than a difference in the data.

Output is one json object per bin:

    {"input_ids": [...], "labels": [...], "cu_seqlens": [0, l0, l0+l1, ..., bin_size]}

``labels`` is -100 on prompt tokens and on the tail padding, so the loss mask is carried
explicitly rather than being re-derived downstream. The final bin is padded to bin_size with
a single trailing segment whose labels are all -100; ``cu_seqlens`` still ends at bin_size so
the segment count is honest and the attention mask blocks it from the real sequences.

Usage:
  python prepare_packed_data.py --tokenizer <dir> --out <file.jsonl> \
      [--src <alpaca.jsonl>] [--bin-size 131072] [--bins 16] [--max-seq 8192]
"""

import argparse
import json
import os
import sys


def load_samples(src, rows_needed):
    """(instruction, input, output) triples, from --src or from the Hub."""
    if src and os.path.exists(src):
        out = []
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append((r.get("instruction", ""), r.get("input", ""), r.get("output", "")))
        print(f"[data] {src}: {len(out)} rows", flush=True)
        return out
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"[data] alpaca: {len(ds)} rows", flush=True)
    return [(r["instruction"], r["input"], r["output"]) for r in ds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None, help="alpaca-format jsonl; omit to fetch from the Hub")
    ap.add_argument("--bin-size", type=int, default=131072, help="tokens per packed bin")
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument(
        "--max-seq",
        type=int,
        default=8192,
        help="drop samples longer than this so one sample cannot fill a whole bin",
    )
    ap.add_argument(
        "--align",
        type=int,
        default=1,
        help="LEGACY, leave at 1. Pads each sequence up to a multiple of this many "
        "tokens. This was once required for THD + context parallelism with the "
        "compressed (CSA/HCA) branches, because a pooling window whose rows "
        "landed on two CP ranks had nowhere to read the earlier ones from. A real "
        "left-boundary exchange now handles that, so alignment buys nothing and "
        "costs a great deal: at --align 128 on alpaca the runtime packer ends up with "
        "only 34.2%% of tokens supervised, against 56.1%% unaligned. Keep "
        "it only to reproduce the old aligned baseline; the tool reports the cost.",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    samples = load_samples(args.src, args.bins * 8)
    if not samples:
        sys.exit("[data] no samples")

    # Tokenise prompt and response separately so the loss mask is exact rather than a
    # guess at where the response starts.
    segs = []
    for instr, inp, outp in samples:
        prompt = f"{instr}\n{inp}\n" if inp else f"{instr}\n"
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(outp, add_special_tokens=False)["input_ids"]
        n = len(p_ids) + len(r_ids)
        if n == 0 or n > args.max_seq:
            continue
        segs.append((p_ids, r_ids))
    if not segs:
        sys.exit("[data] every sample was empty or longer than --max-seq")
    lens = sorted(len(p) + len(r) for p, r in segs)
    print(
        f"[data] usable {len(segs)} samples, len p50={lens[len(lens)//2]} "
        f"p99={lens[int(len(lens)*0.99)]} max={lens[-1]}",
        flush=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    if args.bin_size % args.align != 0:
        sys.exit(f"[data] --bin-size {args.bin_size} must be a multiple of --align {args.align}")
    written = 0
    si = 0
    n_pad_align = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        while written < args.bins:
            ids, labels, cu = [], [], [0]
            # Greedy first-fit: walk the sample list and take whatever still fits. Not
            # optimal packing, but the waste is one partial segment per bin and the order
            # stays deterministic, which matters for comparing runs.
            scanned = 0
            while scanned < len(segs) and len(ids) < args.bin_size:
                p_ids, r_ids = segs[si % len(segs)]
                si += 1
                scanned += 1
                n = len(p_ids) + len(r_ids)
                # Align the SEGMENT length, so every subsequent sequence also starts on a
                # multiple of `align` (cu_seqlens stays aligned by induction from 0).
                n_aligned = ((n + args.align - 1) // args.align) * args.align
                if len(ids) + n_aligned > args.bin_size:
                    continue
                ids.extend(p_ids)
                ids.extend(r_ids)
                labels.extend([-100] * len(p_ids))  # prompt is not supervised
                labels.extend(r_ids)  # response is
                if n_aligned > n:
                    ids.extend([pad_id] * (n_aligned - n))
                    labels.extend([-100] * (n_aligned - n))  # alignment padding, unsupervised
                    n_pad_align += n_aligned - n
                cu.append(len(ids))
            if len(ids) < args.bin_size:
                pad = args.bin_size - len(ids)
                ids.extend([pad_id] * pad)
                labels.extend([-100] * pad)
                cu.append(len(ids))  # padding is its own segment
            fh.write(json.dumps({"input_ids": ids, "labels": labels, "cu_seqlens": cu}) + "\n")
            written += 1

    first = json.loads(open(args.out, encoding="utf-8").readline())
    n_seg = len(first["cu_seqlens"]) - 1
    supervised = sum(1 for x in first["labels"] if x != -100)
    print(
        f"[data] {args.out}: {written} bins x {args.bin_size} tokens, "
        f"{os.path.getsize(args.out) / 2**20:.1f} MB",
        flush=True,
    )
    print(
        f"[data] bin0: {n_seg} segments, {supervised} supervised tokens "
        f"({100.0 * supervised / args.bin_size:.1f}%)",
        flush=True,
    )
    if args.align > 1:
        total = written * args.bin_size
        print(
            f"[data] alignment to {args.align}: {n_pad_align} padding tokens "
            f"({100.0 * n_pad_align / total:.1f}% of the corpus). Short samples cost the most "
            f"here -- alpaca's segment median is 84 tokens against align=128.",
            flush=True,
        )
    # cu_seqlens must be a multiple of `align` everywhere, or a compressed window will
    # straddle a CP shard boundary. Check rather than trust the arithmetic above.
    if args.align > 1:
        bad = [c for c in first["cu_seqlens"] if c % args.align != 0]
        if bad:
            sys.exit(f"[data] BUG: cu_seqlens entries not aligned to {args.align}: {bad[:5]}")
        print(f"[data] verified: every cu_seqlens boundary is a multiple of {args.align}", flush=True)


if __name__ == "__main__":
    main()
