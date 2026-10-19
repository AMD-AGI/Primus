#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Convert a released Inkling checkpoint (Hugging Face safetensors) to Megatron.

    python tools/checkpoint_conversion/convert_inkling_hf_megatron.py \\
        --hf-dir /data/Inkling-Small \\
        --model-yaml primus/configs/models/megatron/inkling_small.yaml \\
        --output /data/inkling_small_megatron [--dry-run]

What it does:

1. Checks the model preset against the checkpoint's ``config.json``.
2. Builds the Megatron model on the meta device (no memory) to get the
   expected parameter names and shapes.
3. Maps every text-backbone tensor (``inkling_hf_mapping.iter_release_to_megatron``),
   streaming one tensor at a time from the shards, and checks each against
   step 2. ``model.mtp.*`` (multi-token prediction), ``model.visual.*`` and
   ``model.audio.*`` are skipped.
4. Writes a single-rank (TP=PP=EP=1) Megatron ``torch`` checkpoint under
   ``<output>/release/mp_rank_00/model_optim_rng.pt``; load it with
   ``--load <output> --ckpt-format torch --finetune --no-load-optim --no-load-rng``.

The embedding and LM head are trimmed or zero-padded to the vocabulary size
training builds: ``unpadded_vocab_size`` rounded up to a multiple of
``--make-vocab-size-divisible-by`` x ``--tensor-model-parallel-size`` (128 x 1
by default, matching ``trainer_base.yaml``). Pass the training run's values,
or ``--target-vocab-size``, if they differ.

``--dry-run`` stops after step 3, reading only the safetensors headers, so the
full mapping can be validated on a login node. Writing a resharded
``torch_dist`` checkpoint for multi-node runs is P3 work.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
from pathlib import Path
from typing import Dict, Tuple

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.append(str(_REPO / "third_party" / "Megatron-LM"))

import torch  # noqa: E402

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

# Kept in fp32 by Megatron (``TopKRouter._maintain_float32_expert_bias``).
_FP32_SUFFIXES = (".mlp.router.expert_bias",)


def read_safetensors_header(path: Path) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in header.items() if k != "__metadata__"}


def checkpoint_index(hf_dir: Path) -> Dict[str, str]:
    index = hf_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    return {k: "model.safetensors" for k in read_safetensors_header(hf_dir / "model.safetensors")}


def init_single_rank():
    if not torch.distributed.is_initialized():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend, init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0
        )
    from megatron.core import parallel_state as ps

    if not ps.model_parallel_is_initialized():
        ps.initialize_model_parallel(1, 1)


def check_preset(config, text_config) -> None:
    from primus.backends.megatron.core.models.inkling.inkling_hf_mapping import (
        megatron_config_from_release,
    )
    from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
        normalize_moe_layer_freq,
    )

    want = megatron_config_from_release(text_config)
    want["moe_layer_freq"] = normalize_moe_layer_freq(want["moe_layer_freq"], num_layers=want["num_layers"])
    want["num_moe_experts"] = want.pop("num_experts")
    del want["vocab_size"]  # Megatron pads the tokenizer vocab its own way
    bad = [
        f"{key}: preset={getattr(config, key)!r} checkpoint={value!r}"
        for key, value in want.items()
        if getattr(config, key) != value
    ]
    if bad:
        raise SystemExit("Model preset does not match the checkpoint:\n  " + "\n  ".join(bad))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-dir", required=True, type=Path)
    p.add_argument("--model-yaml", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--target-vocab-size",
        type=int,
        default=None,
        help="Megatron padded vocab (default: unpadded_vocab_size padded the way training pads it)",
    )
    p.add_argument("--make-vocab-size-divisible-by", type=int, default=128)
    p.add_argument("--tensor-model-parallel-size", type=int, default=1)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--dry-run", action="store_true", help="map shapes from the headers only; write nothing")
    args = p.parse_args(argv)
    if not args.dry_run and args.output is None:
        p.error("--output is required unless --dry-run")

    from primus.backends.megatron.core.models.inkling.inkling_builders import (
        build_inkling_model,
        inkling_config_from_model_yaml,
    )
    from primus.backends.megatron.core.models.inkling.inkling_hf_mapping import (
        InklingGeometry,
        iter_release_to_megatron,
        module_tensor_shapes,
        release_text_keys,
    )

    text_config = json.loads((args.hf_dir / "config.json").read_text()).get("text_config")
    weight_map = checkpoint_index(args.hf_dir)

    init_single_rank()
    config = inkling_config_from_model_yaml(str(args.model_yaml))
    check_preset(config, text_config)
    geo = InklingGeometry.from_megatron_config(config)

    headers: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    for shard in sorted(set(weight_map.values())):
        headers.update(read_safetensors_header(args.hf_dir / shard))
    vocab_rows = headers["model.llm.embed.weight"][1][0]
    unpadded = config.unpadded_vocab_size or vocab_rows
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    target_vocab = args.target_vocab_size or -(-unpadded // multiple) * multiple
    if target_vocab < unpadded:
        raise SystemExit(f"--target-vocab-size {target_vocab} would drop real tokens (unpadded {unpadded})")
    print(f"[inkling-convert] vocab: {vocab_rows} checkpoint rows -> {target_vocab} (unpadded {unpadded})")

    model = build_inkling_model(config, vocab_size=target_vocab, max_sequence_length=4096, meta=True)
    expected = module_tensor_shapes(model)
    extra_state = [k for k in model.state_dict() if k.endswith("_extra_state")]
    del model

    text_keys = release_text_keys(weight_map)
    used = set()
    handles = {}

    def meta_get(name):
        used.add(name)
        dtype, shape = headers[name]
        return torch.empty(shape, dtype=_DTYPES[dtype], device="meta")

    def real_get(name):
        from safetensors import safe_open

        used.add(name)
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(str(args.hf_dir / shard), framework="pt")
        return handles[shard].get_tensor(name)

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    state = {}
    seen = set()
    for key, tensor in iter_release_to_megatron(
        meta_get if args.dry_run else real_get, geo, target_vocab_size=target_vocab
    ):
        if key not in expected:
            raise SystemExit(f"converter produced {key}, which the Megatron model does not have")
        if tuple(tensor.shape) != expected[key]:
            raise SystemExit(f"{key}: converted {tuple(tensor.shape)} != model {expected[key]}")
        seen.add(key)
        if not args.dry_run:
            dtype = torch.float32 if key.endswith(_FP32_SUFFIXES) else out_dtype
            state[key] = tensor.to(dtype).contiguous().clone()

    missing = sorted(set(expected) - seen)
    unmapped = sorted(text_keys - used)
    skipped = sorted(set(weight_map) - text_keys)
    print(
        f"[inkling-convert] mapped {len(used)} released tensors -> {len(seen)} Megatron tensors; "
        f"skipped {len(skipped)} non-text tensors (MTP / vision / audio)"
    )
    if missing or unmapped:
        raise SystemExit(
            f"incomplete mapping: model keys without a source {missing[:5]}, unmapped sources {unmapped[:5]}"
        )
    if args.dry_run:
        print("[inkling-convert] dry run OK: every text tensor maps one-to-one onto the Megatron model")
        return 0

    for key in extra_state:
        state[key] = None
    ckpt_dir = args.output / "release" / "mp_rank_00"
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"model": state, "checkpoint_version": 3.0, "iteration": 0}, ckpt_dir / "model_optim_rng.pt")
    (args.output / "latest_checkpointed_iteration.txt").write_text("release")
    print(f"[inkling-convert] wrote {ckpt_dir / 'model_optim_rng.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
