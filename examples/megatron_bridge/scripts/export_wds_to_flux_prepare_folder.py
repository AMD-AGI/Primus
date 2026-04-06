#!/usr/bin/env python3
"""
Export MLPerf CC12M-on-disk WebDataset shards into a directory layout suitable for
examples/diffusion/recipes/flux/prepare_energon_dataset_flux.py (recursive images + sidecar .txt captions).

Typical shard sample keys vary; this script picks the first image-like key and caption from json/txt/json.gz.
"""

from __future__ import annotations

import argparse
import gzip
import json
from glob import glob
from pathlib import Path

import webdataset as wds


def _try_parse_caption_json(raw: bytes) -> str | None:
    try:
        obj = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    for ck in ("caption", "text", "TEXT", "alt", "description"):
        if ck in obj and obj[ck]:
            return str(obj[ck]).strip()
    return None


def _caption_from_sample(sample: dict) -> str:
    for k in ("json", "json.gz", "txt", "caption.txt", "text", "txt.gz"):
        if k not in sample:
            continue
        v = sample[k]
        if not isinstance(v, (bytes, bytearray)):
            continue
        raw = bytes(v)
        if k.endswith(".gz"):
            raw = gzip.decompress(raw)
        if k.startswith("json"):
            parsed = _try_parse_caption_json(raw)
            if parsed:
                return parsed
            continue
        return raw.decode("utf-8", errors="replace").strip()
    # Any other *.json / *.txt key
    for key, val in sample.items():
        if key.startswith("__"):
            continue
        if not isinstance(val, (bytes, bytearray)):
            continue
        lk = key.lower()
        if lk.endswith(".json") or lk.endswith(".json.gz"):
            raw = bytes(val)
            if lk.endswith(".gz"):
                raw = gzip.decompress(raw)
            parsed = _try_parse_caption_json(raw)
            if parsed:
                return parsed
        elif lk.endswith(".txt") and not lk.endswith(".jpg"):
            return bytes(val).decode("utf-8", errors="replace").strip()
    return ""


def _image_key(sample: dict) -> tuple[str, bytes] | None:
    for key, val in sample.items():
        if key.startswith("__"):
            continue
        if not isinstance(val, (bytes, bytearray)):
            continue
        lk = key.lower()
        if lk in ("jpg", "jpeg", "png", "webp") or any(
            lk.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")
        ):
            return key, bytes(val)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="WebDataset → flat folder for FLUX Energon prepare.")
    p.add_argument(
        "--input-glob",
        type=str,
        required=True,
        help='Glob for shards, e.g. "/data/cc12m_disk/**/*.tar" (quote to avoid shell expansion).',
    )
    p.add_argument("--output-dir", type=str, required=True, help="Directory to create with images + .txt captions.")
    p.add_argument("--max-samples", type=int, default=0, help="Stop after N samples (0 = no limit).")
    args = p.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob(args.input_glob, recursive=True))
    if not paths:
        raise SystemExit(f"No shards matched --input-glob {args.input_glob!r}")
    # Keep image keys as raw bytes; avoid .decode() which turns jpg into tensors/PIL.
    dataset = wds.WebDataset(paths)

    n = 0
    for sample in dataset:
        img = _image_key(sample)
        if img is None:
            continue
        _k, data = img
        ext = Path(_k).suffix.lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        stem = f"{n:08d}"
        img_path = out / f"{stem}{ext}"
        txt_path = out / f"{stem}.txt"
        img_path.write_bytes(data)
        cap = _caption_from_sample(sample)
        txt_path.write_text(cap, encoding="utf-8")
        n += 1
        if n % 1000 == 0:
            print(f"Exported {n} samples…")
        if args.max_samples and n >= args.max_samples:
            break

    if n == 0:
        raise SystemExit(
            "No samples exported. Check --input-glob (must match .tar shards). "
            "Example: --input-glob '/data/mlperf_flux1/cc12m_disk/*.tar'"
        )
    print(f"Done. Exported {n} image/caption pairs under {out}")


if __name__ == "__main__":
    main()
