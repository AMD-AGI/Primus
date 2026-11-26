###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fetch the embeddings, final norms and the first N decoder layers of Inkling-Small.

Ranged HTTP reads straight out of the released safetensors shards, so only the
needed bytes are downloaded: ~4.3 GB for the two dense layers, ~13 GB more
per MoE layer. Writes one ``model.safetensors`` with the release tensor names,
which ``test_inkling_small_first_layers_match_reference`` reads (layers 0-5,
one sliding/global attention period)::

    python tests/unit_tests/backends/megatron/inkling/fetch_release_layers.py \\
        --layers 6 --out /tmp/inkling_ref/small_6L
"""

import argparse
import json
import os
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors.torch import save_file

BASE = "https://huggingface.co/thinkingmachines/Inkling-Small/resolve/main/"
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}


def get(url, start=None, end=None):
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as r:
        return r.read()


def header(fname):
    n = struct.unpack("<Q", get(BASE + fname, 0, 7))[0]
    return 8 + n, json.loads(get(BASE + fname, 8, 7 + n))


def wanted(key, layers):
    if key in (
        "model.llm.embed.weight",
        "model.llm.embed_norm.weight",
        "model.llm.norm.weight",
        "model.llm.unembed.weight",
    ):
        return True
    parts = key.split(".")
    return key.startswith("model.llm.layers.") and int(parts[3]) < layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    index = json.loads(get(BASE + "model.safetensors.index.json"))["weight_map"]
    keys = sorted(k for k in index if wanted(k, args.layers))
    files = sorted({index[k] for k in keys})
    headers = {f: header(f) for f in files}

    def fetch(key):
        base, h = headers[index[key]]
        meta = h[key]
        s, e = meta["data_offsets"]
        raw = bytearray(get(BASE + index[key], base + s, base + e - 1))
        t = torch.frombuffer(raw, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"]).clone()
        print(f"{key} {meta['dtype']} {tuple(meta['shape'])}", flush=True)
        return key, t

    with ThreadPoolExecutor(8) as pool:
        tensors = dict(pool.map(fetch, keys))
    save_file(tensors, os.path.join(args.out, "model.safetensors"))
    print(
        f"wrote {len(tensors)} tensors, {sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()
