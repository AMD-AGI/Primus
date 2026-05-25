"""One-shot diagnostic: compare Bridge packed parquet vs Native packed cache.

Goal
----
After the `labels = input_ids[1:]` shift fix landed in
`primus/backends/megatron/sft/packing.py`, Native iter-1 loss dropped to ~0.15
while Bridge sits at ~4.34 on the same Llama-2 70B + SQuAD recipe. The
hypothesis is that the two stacks disagree on what the ``loss_mask`` covers
(answer-only vs near-all-token). This script reads both caches directly and
dumps the structure side-by-side so the disagreement can be confirmed before
we touch any more training code.

Usage (inside the docker container that ran the experiments)::

    python -m primus.tools.diag.inspect_sft_data \
        --bridge-parquet /workspace/cache_persist/nemo/datasets/rajpurkar/squad/packed/NousResearch--Llama-2-70b-hf_pad_seq_to_mult1/training_4096.idx.parquet \
        --native-cache /workspace/hf_cache/datasets/primus_packed/sft_pack_7acbbddb02e3e3cd.pt \
        --tokenizer NousResearch/Llama-2-70b-hf \
        --num-samples 3

Either ``--bridge-parquet`` or ``--native-cache`` may be omitted; the script
runs whichever side(s) it can resolve.

What it prints
--------------
For each side (up to ``--num-samples`` packed sequences) it dumps:
  * shape, total tokens, mask=1 token count, supervised ratio
  * the *boundary* between mask=0 and mask=1 transitions (so we can see whether
    mask is set per-segment answer-only or near-all-token)
  * decoded text of the first 80 tokens
  * decoded text of the *supervised* span (mask=1 tokens), capped at 200 chars,
    to verify it actually corresponds to SQuAD answer text (not instruction).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence


def _try_import_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError:
        return None


def _try_import_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: F401
        return pq
    except ImportError:
        return None


def _load_tokenizer(name_or_path: str):
    """Lazy tokenizer load. HuggingFace fast tokenizer if available, else SP."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    except Exception as exc:
        print(f"[warn] AutoTokenizer load failed ({exc!r}); falling back to "
              "stub decoder that just prints token IDs.", file=sys.stderr)
        return None


def _decode(tok, ids: Sequence[int]) -> str:
    if tok is None:
        return f"<no-tokenizer> ids={list(ids)[:20]}..."
    try:
        return tok.decode(list(ids), skip_special_tokens=False)
    except Exception as exc:
        return f"<decode-failed: {exc!r}> ids={list(ids)[:20]}..."


def _mask_transitions(mask: Sequence[int]) -> List[tuple[int, int]]:
    """Return ``(start, end)`` half-open intervals where mask==1.

    Useful for spotting whether ``mask`` is one big contiguous block (all-token
    style) or many tiny blocks per sub-segment (answer-only style).
    """
    intervals: List[tuple[int, int]] = []
    cur_start = None
    for i, m in enumerate(mask):
        if m and cur_start is None:
            cur_start = i
        elif not m and cur_start is not None:
            intervals.append((cur_start, i))
            cur_start = None
    if cur_start is not None:
        intervals.append((cur_start, len(mask)))
    return intervals


def _summary(label: str, input_ids: Sequence[int], loss_mask: Sequence[int],
             labels: Sequence[int] | None, tok, num_token_preview: int = 80,
             num_supervised_preview: int = 6) -> None:
    n = len(input_ids)
    assert len(loss_mask) == n or len(loss_mask) == n - 1, (
        f"{label}: mask len {len(loss_mask)} != input len {n}"
    )
    supervised = sum(1 for m in loss_mask if m)
    intervals = _mask_transitions(loss_mask)

    print(f"\n=== {label} ===")
    print(f"  len(input_ids)            = {n}")
    print(f"  len(loss_mask)            = {len(loss_mask)}")
    if labels is not None:
        print(f"  len(labels)               = {len(labels)}")
    print(f"  supervised positions      = {supervised} "
          f"({supervised / max(1, len(loss_mask)) * 100:.2f}%)")
    print(f"  #mask=1 intervals         = {len(intervals)} (first 5: "
          f"{intervals[:5]})")

    # Decoded preview of the very start of the packed sequence.
    print(f"  decoded[0:{num_token_preview}]            = "
          f"{_decode(tok, input_ids[:num_token_preview])!r}")

    # First few supervised intervals -- this is the crucial diagnostic. If
    # answer-only, each interval should be short (~10-15 tokens) and decode to
    # SQuAD-style answer text. If all-token, intervals will be long and span
    # the whole context.
    for k, (s, e) in enumerate(intervals[:num_supervised_preview]):
        seg_ids = input_ids[s:e]
        if labels is not None and s < len(labels) and e <= len(labels):
            tgt_ids = labels[s:e]
            tgt_decode = _decode(tok, tgt_ids)
        else:
            tgt_decode = "<no-labels>"
        print(f"  supervised interval #{k}: [{s}, {e})  len={e - s}")
        print(f"     input ids   : {_decode(tok, seg_ids)!r}")
        print(f"     label target: {tgt_decode!r}")


def inspect_native_cache(path: Path, tok, num_samples: int) -> None:
    torch = _try_import_torch()
    if torch is None:
        print("[error] torch not installed; cannot read Native cache (.pt)",
              file=sys.stderr)
        return
    if not path.exists():
        print(f"[skip] Native cache not found: {path}", file=sys.stderr)
        return

    print(f"\n############### NATIVE cache: {path} ###############")
    packed = torch.load(str(path), weights_only=False)
    print(f"  total packed sequences = {len(packed)}")

    for i in range(min(num_samples, len(packed))):
        sample = packed[i]
        ids = sample["input_ids"].tolist() if hasattr(sample["input_ids"], "tolist") else list(sample["input_ids"])
        msk = sample["loss_mask"].tolist() if hasattr(sample["loss_mask"], "tolist") else list(sample["loss_mask"])
        labels = sample.get("labels")
        if labels is not None and hasattr(labels, "tolist"):
            labels = labels.tolist()
        elif labels is not None:
            labels = list(labels)

        cu = sample.get("cu_seqlens")
        if cu is not None and hasattr(cu, "tolist"):
            cu = cu.tolist()
        num_real = sample.get("num_real_segments")
        if hasattr(num_real, "item"):
            num_real = num_real.item()

        print(f"\n[Native sample {i}]  cu_seqlens(first 8)={cu[:8] if cu else None}  "
              f"num_real_segments={num_real}")
        _summary(f"Native[{i}]", ids, msk, labels, tok)


def inspect_bridge_parquet(path: Path, tok, num_samples: int) -> None:
    pq = _try_import_pyarrow()
    if pq is None:
        print("[error] pyarrow not installed; cannot read Bridge parquet",
              file=sys.stderr)
        return
    if not path.exists():
        print(f"[skip] Bridge parquet not found: {path}", file=sys.stderr)
        return

    print(f"\n############### BRIDGE parquet: {path} ###############")
    parquet_file = pq.ParquetFile(str(path))
    print(f"  total rows   = {parquet_file.metadata.num_rows}")
    print(f"  columns      = {parquet_file.schema.names}")

    # Read just the first row group (cheaper than full file).
    rg = parquet_file.read_row_group(0,
                                     columns=["input_ids", "seq_start_id", "loss_mask"])

    for i in range(min(num_samples, rg.num_rows)):
        ids = rg.column("input_ids")[i].as_py()
        msk = rg.column("loss_mask")[i].as_py()
        seq_start = rg.column("seq_start_id")[i].as_py()
        seq_boundaries = seq_start + [len(ids)]

        print(f"\n[Bridge sample {i}]  seq_boundaries(first 8)={seq_boundaries[:8]}  "
              f"#sub-segments={len(seq_boundaries) - 1}")
        _summary(f"Bridge[{i}]", ids, msk, labels=None, tok=tok)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bridge-parquet", type=Path, default=None,
                   help="Path to Bridge's training_*.idx.parquet")
    p.add_argument("--native-cache", type=Path, default=None,
                   help="Path to Native's sft_pack_<hash>.pt")
    p.add_argument("--tokenizer", type=str, default="NousResearch/Llama-2-70b-hf",
                   help="HF tokenizer to use for decoding (must match the run)")
    p.add_argument("--num-samples", type=int, default=3,
                   help="How many packed sequences to dump per side")
    args = p.parse_args(argv)

    tok = _load_tokenizer(args.tokenizer)

    if args.native_cache is not None:
        inspect_native_cache(args.native_cache, tok, args.num_samples)
    if args.bridge_parquet is not None:
        inspect_bridge_parquet(args.bridge_parquet, tok, args.num_samples)

    if args.native_cache is None and args.bridge_parquet is None:
        p.error("must pass at least one of --native-cache / --bridge-parquet")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
