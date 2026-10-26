#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Build the Zebra-Llama Enhanced-ILD text as a Megatron indexed dataset, one document per conversation.

The reference stage-1 students were trained with TRL on packed 2048-token windows of a 20%
selection of ``JunxiongWang/sftdatasetv3``. This tool rebuilds that text for the packed recipes
(``hylo_m2_1B-enhanced-ild-packed.yaml``, ``hylo_mla_1B-enhanced-ild-packed.yaml``):

1. alignment-handbook ``get_datasets``: each split shuffled with seed 42;
2. ``apply_chat_template`` with the recipe's Llama-3 template (it never inserts a system message);
3. for train, a second ``shuffle(seed=42)`` and ``select(range(1, ratio * rows))``;
4. HumanEval decontamination of the templated text, on both splits;
5. TRL ``ConstantLengthDataset`` tokenization: ``add_special_tokens=True`` (a second BOS after
   the template's) and an EOS after every conversation.

Megatron's pretrain data path packs the documents into ``seq_length`` windows with attention
across documents, as ``ConstantLengthDataset`` does. The tool also counts the sequences
``ConstantLengthDataset`` itself would make, to check a selection against the reference's step
count (6934 steps of 96 for the train split)::

    python tools/hybrid/prepare_hylo_packed_data.py --split train --ratio 0.2 \\
        --tokenizer /path/to/Llama-3.2-1B-Instruct --out data/hylo/packed/train --workers 48
    python tools/hybrid/prepare_hylo_packed_data.py --split test \\
        --tokenizer /path/to/Llama-3.2-1B-Instruct --out data/hylo/packed/test

Writes ``<out>.bin``, ``<out>.idx``, and ``<out>-stats.json``. Needs ``datasets`` and
``transformers``; the dataset and HumanEval come from the Hugging Face Hub unless cached.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = "JunxiongWang/sftdatasetv3"
DEFAULT_DECONTAMINATION = "openai/openai_humaneval"
TEMPLATE = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = "
    "'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + "
    "'<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}{% endfor %}{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)
# Kept verbatim from alignment-handbook, including the adjacent literals in the last entry.
HUMAN_EVAL_STRINGS_OK = ["return x + y", "return len(string)", "return n**2", "return " ".join(strings)"]


def extract_docstring(prompt: str) -> str:
    if '"""' in prompt:
        if prompt.count('"""') == 2:
            return prompt.split('"""')[1].strip()
        if prompt.count('"""') == 4:
            return prompt.split('"""')[3].strip()
        raise ValueError("unexpected number of triple double quotes")
    if "'''" in prompt:
        if prompt.count("'''") != 2:
            raise ValueError("unexpected number of triple single quotes")
        return prompt.split("'''")[1].strip()
    raise ValueError("no docstring")


def normalize(text: str) -> str:
    return " ".join(text.split())


def contamination_strings(rows: Iterable[dict]) -> List[str]:
    """Normalized HumanEval docstrings and canonical solutions, as alignment-handbook builds them."""
    rows = list(rows)
    docstrings = [extract_docstring(row["prompt"]) for row in rows]
    solutions = [
        s for s in (row["canonical_solution"].strip() for row in rows) if s and s not in HUMAN_EVAL_STRINGS_OK
    ]
    return [normalize(s.lower()) for s in docstrings + solutions]


def is_contaminated(text: str, substrings: Sequence[str]) -> bool:
    content = normalize(text.lower())
    return any(s in content for s in substrings)


def select_rows(num_rows: int, split: str, ratio: float = 0.2, seed: int = 42) -> np.ndarray:
    """Original row indices in training order.

    ``datasets`` shuffles with ``np.random.default_rng(seed).permutation``. Every split is
    shuffled once; the train split is shuffled again with the same seed and then
    ``select(range(1, int(ratio * rows)))``, so row ``i`` of the result is ``p[p[i]]``.
    """
    perm = np.random.default_rng(seed).permutation(num_rows)
    if split != "train":
        return perm
    return perm[perm[1 : int(ratio * num_rows)]]


def constant_length_sequences(
    chars: Sequence[int],
    tokens: Sequence[int],
    seq_length: int,
    chars_per_token: float = 3.6,
    num_sequences: int = 1024,
) -> int:
    """Sequences TRL's ``ConstantLengthDataset`` makes: character buffers, remainders dropped per buffer."""
    limit = seq_length * chars_per_token * num_sequences
    total = buffer_chars = buffer_tokens = 0
    for c, t in zip(chars, tokens):
        buffer_chars += c
        buffer_tokens += t
        if buffer_chars >= limit:
            total += buffer_tokens // seq_length
            buffer_chars = buffer_tokens = 0
    return total + buffer_tokens // seq_length


# Worker state, set in the parent before the pool forks.
_DATASET = _TOKENIZER = _SUBSTRINGS = None


def _tokenize_part(job):
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    prefix, rows = job
    texts = [_TOKENIZER.apply_chat_template(m, tokenize=False) for m in _DATASET.select(rows)["messages"]]
    kept = [t for t in texts if not is_contaminated(t, _SUBSTRINGS)]
    ids = _TOKENIZER(kept, add_special_tokens=True, truncation=False)["input_ids"] if kept else []
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
    chars, counts = [], []
    for text, tokens in zip(kept, ids):
        tokens = list(tokens) + [_TOKENIZER.eos_token_id]
        builder.add_document(np.asarray(tokens, dtype=np.int32), [len(tokens)])
        chars.append(len(text))
        counts.append(len(tokens))
    builder.finalize(prefix + ".idx")
    head = list(ids[0][:6]) if ids else []
    return len(texts), len(kept), np.asarray(chars, np.int64), np.asarray(counts, np.int64), head


def main(argv: Optional[Sequence[str]] = None) -> dict:
    global _DATASET, _TOKENIZER, _SUBSTRINGS
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--out", required=True, help="output prefix for <out>.bin, <out>.idx, <out>-stats.json")
    p.add_argument("--tokenizer", required=True, help="Llama-3.2-1B-Instruct tokenizer (path or Hub name)")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--decontaminate-against", default=DEFAULT_DECONTAMINATION)
    p.add_argument("--ratio", type=float, default=0.2, help="train selection fraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-length", type=int, default=2048, help="window length, for the counts")
    p.add_argument("--batch", type=int, default=96, help="windows per step, for the step count")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--chunk", type=int, default=20000, help="conversations per worker task")
    a = p.parse_args(argv)

    megatron = os.environ.get("MEGATRON_PATH", str(_REPO_ROOT / "third_party" / "Megatron-LM"))
    if megatron not in sys.path:
        sys.path.insert(0, megatron)
    from datasets import load_dataset
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    from transformers import AutoTokenizer

    start = time.time()
    _DATASET = load_dataset(a.dataset, split=a.split)
    num_rows = len(_DATASET)
    rows = select_rows(num_rows, a.split, a.ratio, a.seed)
    if a.split == "train":
        expected = _DATASET.shuffle(seed=a.seed).shuffle(seed=a.seed).select(range(1, 4))["messages"]
        if expected != _DATASET.select(rows[:3])["messages"]:
            raise RuntimeError("the selection differs from datasets' double shuffle")
    _TOKENIZER = AutoTokenizer.from_pretrained(a.tokenizer)
    _TOKENIZER.chat_template = TEMPLATE
    _SUBSTRINGS = contamination_strings(load_dataset(a.decontaminate_against, split="test"))
    print(
        f"{a.split}: {num_rows} rows, {len(rows)} selected; eos {_TOKENIZER.eos_token_id} "
        f"bos {_TOKENIZER.bos_token_id}; {len(_SUBSTRINGS)} decontamination strings",
        flush=True,
    )

    out = Path(a.out)
    parts = out.parent / f"{out.name}-parts"
    shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir(parents=True)
    jobs = [
        (str(parts / f"part-{i:05d}"), rows[j : j + a.chunk])
        for i, j in enumerate(range(0, len(rows), a.chunk))
    ]
    with mp.get_context("fork").Pool(max(1, min(a.workers, len(jobs)))) as pool:
        results = pool.map(_tokenize_part, jobs, chunksize=1)
    print(f"tokenized {len(jobs)} parts in {time.time() - start:.0f}s", flush=True)

    builder = IndexedDatasetBuilder(str(out) + ".bin", dtype=np.int32)
    for prefix, _ in jobs:
        builder.add_index(prefix)
    builder.finalize(str(out) + ".idx")
    shutil.rmtree(parts)

    chars = np.concatenate([r[2] for r in results])
    counts = np.concatenate([r[3] for r in results])
    sequences = constant_length_sequences(chars.tolist(), counts.tolist(), a.seq_length)
    stats = {
        "split": a.split,
        "dataset": a.dataset,
        "raw_rows": num_rows,
        "selected": int(sum(r[0] for r in results)),
        "kept": int(sum(r[1] for r in results)),
        "tokens": int(counts.sum()),
        "mean_tokens": float(counts.mean()),
        "median_tokens": float(np.median(counts)),
        "seq_length": a.seq_length,
        "windows": int(counts.sum() // a.seq_length),
        "trl_sequences": sequences,
        "trl_steps": math.ceil(sequences / a.batch),
        "first_tokens": [int(t) for t in results[0][4]],
        "seconds": round(time.time() - start),
    }
    stats["decontaminated"] = stats["selected"] - stats["kept"]
    Path(str(out) + "-stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats), flush=True)
    return stats


if __name__ == "__main__":
    main()
