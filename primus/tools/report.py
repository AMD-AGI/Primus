###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# ---- minimal table writer: auto by extension; stdout if not set; append supported ----

import csv
import gzip
import json
import os
from typing import Any, List


def _infer_fmt(path: str) -> str:
    """Infer output format from file extension."""
    p = path.lower()
    if p.endswith((".md", ".markdown")):
        return "md"
    if p.endswith(".csv") or p.endswith(".csv.gz"):
        return "csv"
    if p.endswith(".tsv") or p.endswith(".tsv.gz"):
        return "tsv"
    if p.endswith(".jsonl") or p.endswith(".jsonl.gz"):
        return "jsonl"
    return "md"  # default


def _is_gz(path: str) -> bool:
    """Check if file should be gzip compressed based on extension."""
    return path.lower().endswith(".gz")


def _md_header(header: List[str]) -> str:
    return "| " + " | ".join(header) + " |\n" + "|" + "|".join(["---"] * len(header)) + "|\n"


def _md_body(rows: List[List[Any]]) -> str:
    return "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)


def write_table_simple(output_file: str, rows: List[List[Any]], header: List[str], append: bool = False):
    """
    Minimal table writer for benchmark results.
    - If output_file == "" or "-" -> print Markdown table to stdout
    - Otherwise, infer format by extension: .md/.csv/.tsv/.jsonl (and optional .gz)
    - Append mode supported for csv/tsv/jsonl (rows only) and md (body only)
    """
    # stdout mode
    if not output_file or output_file == "-":
        content = _md_header(header) + _md_body(rows)
        print(content, end="")
        return

    fmt = _infer_fmt(output_file)
    gz = _is_gz(output_file)
    dirpath = os.path.dirname(output_file)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # choose open function
    if gz:
        open_fn = lambda path, mode: gzip.open(path, mode)
        text_mode_w, text_mode_a = "wt", "at"
    else:
        open_fn = lambda path, mode: open(path, mode, encoding="utf-8")
        text_mode_w, text_mode_a = "w", "a"

    # markdown
    if fmt == "md":
        if append and os.path.exists(output_file):
            with open_fn(output_file, text_mode_a) as f:
                f.write("\n" + _md_body(rows))
        else:
            with open_fn(output_file, text_mode_w) as f:
                f.write(_md_header(header) + _md_body(rows))
        print(f"[BENCH] Markdown saved: {output_file} ({'append' if append else 'overwrite'})")
        return

    # csv/tsv
    if fmt in ("csv", "tsv"):
        delim = "," if fmt == "csv" else "\t"
        file_exists = os.path.exists(output_file)
        mode = text_mode_a if append else text_mode_w
        with open_fn(output_file, mode) as f:
            w = csv.writer(f, delimiter=delim)
            if (not append) or (append and not file_exists):
                w.writerow(header)
            for r in rows:
                w.writerow([str(x) for x in r])
        print(f"[BENCH] {fmt.upper()} saved: {output_file} ({'append' if append else 'overwrite'})")
        return

    # jsonl
    if fmt == "jsonl":
        mode = text_mode_a if append else text_mode_w
        with open_fn(output_file, mode) as f:
            for r in rows:
                obj = {header[i]: r[i] for i in range(len(header))}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[BENCH] JSONL saved: {output_file} ({'append' if append else 'overwrite'})")
        return

    # fallback: markdown
    with open_fn(output_file, text_mode_w if not append else text_mode_a) as f:
        if append and os.path.getsize(output_file) > 0:
            f.write("\n" + _md_body(rows))
        else:
            f.write(_md_header(header) + _md_body(rows))
    print(f"[BENCH] Markdown (fallback) saved: {output_file} ({'append' if append else 'overwrite'})")
