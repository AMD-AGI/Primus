#!/usr/bin/env python3
"""
Consolidate a Primus FSDP-dtensor (distcp) checkpoint into the legacy
single-rank ``mp_rank_00/model_optim_rng.pt`` layout that the existing
``tools/convert_*_to_fla_hf.py`` converters consume.

Background
----------
When Primus trains with ``ckpt_format: fsdp_dtensor`` (Megatron-FSDP path,
used for the 1B Pure-GDN 100B run), Megatron writes one ``__N_0.distcp``
shard per data-parallel rank plus a ``.metadata`` file via
``torch.distributed.checkpoint``. Two transforms happen relative to the
legacy ``torch`` ckpt format:

1. All model parameters are nested under ``model.module.`` (the FSDP
   wrapper class layer).
2. Fused SwiGLU ``linear_fc1.weight`` (shape ``[2*intermediate, hidden]``)
   is **split** by FSDP into two halves at save time:
       * ``linear_fc1.weight_w`` = first half (gate proj, shape
         ``[intermediate, hidden]``)
       * ``linear_fc1.weight_v`` = second half (up proj, same shape)
   See ``third_party/Megatron-LM/megatron/core/transformer/fsdp_dtensor_checkpoint.py``
   ``split_swiglu_linear_fc1`` for the source-of-truth split.

This script reverses both transforms and writes a single
``mp_rank_00/model_optim_rng.pt`` file in the format
``convert_gdn_to_fla_hf.py`` / ``convert_kda_to_fla_hf.py`` /
``convert_gdn_hybrid_to_fla_hf.py`` already understand:

    {"model": {<flat parameter dict, no FSDP prefix, fused fc1>},
     "iteration": <int>,
     "checkpoint_version": <float>}

Usage
-----
    python3 tools/consolidate_distcp_to_torch.py \
        --distcp-dir output/amd/root/zebra_llama_1B_gdn_pure_100B-pretrain/checkpoints/iter_0095368 \
        --output-dir output/amd/root/zebra_llama_1B_gdn_pure_100B-pretrain/checkpoints_consolidated/iter_0095368

The output dir gets a ``mp_rank_00/model_optim_rng.pt`` file ready for the
HF converters. Memory: ~14 GB peak (full 1B model in fp32/bf16 on CPU).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys

FSDP_PREFIX = "model.module."
SWIGLU_W_SUFFIX = ".weight_w"
SWIGLU_V_SUFFIX = ".weight_v"


def _is_model_weight_key(key: str) -> bool:
    """Filter out optimizer / scheduler / RNG / iteration / args keys."""
    return key.startswith(FSDP_PREFIX)


def _strip_fsdp_prefix(key: str) -> str:
    assert key.startswith(FSDP_PREFIX), key
    return key[len(FSDP_PREFIX):]


def _enumerate_model_keys(distcp_dir: Path) -> tuple[list[str], int]:
    """Return (model_weight_keys, iteration) from the .metadata file."""
    reader = FileSystemReader(str(distcp_dir))
    md = reader.read_metadata()
    keys = sorted(k for k in md.state_dict_metadata.keys() if _is_model_weight_key(k))
    iteration = 0
    iter_re = re.search(r"iter_0*(\d+)$", distcp_dir.name)
    if iter_re:
        iteration = int(iter_re.group(1))
    return keys, iteration


def _flatten(prefix: str, obj, out: dict[str, torch.Tensor]) -> None:
    """Recursively flatten a nested dict (with dotted keys) into a single map."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten(new_prefix, v, out)
    else:
        out[prefix] = obj


def _load_tensors(distcp_dir: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    """Pull the listed keys out of the distcp shards into a single CPU dict.

    Uses torch's single-process DCP loader (``_load_state_dict_from_keys``);
    no torch.distributed init is required.  The loader returns a NESTED dict
    (because dotted keys are interpreted as paths), so we flatten it back.
    """
    print(f"[load] reading {len(keys)} tensors from {distcp_dir} ...")
    nested = _load_state_dict_from_keys(
        keys=set(keys),
        checkpoint_id=str(distcp_dir),
        storage_reader=FileSystemReader(str(distcp_dir)),
    )
    flat: dict[str, torch.Tensor] = {}
    _flatten("", nested, flat)
    # Drop anything we didn't ask for (the loader may pull adjacent BytesMetadata)
    flat = {k: v for k, v in flat.items() if k in set(keys)}
    print(f"[load] got {len(flat)} tensors back (flattened from nested dict)")
    if len(flat) != len(keys):
        missing = set(keys) - set(flat)
        if missing:
            raise RuntimeError(
                f"Loader returned {len(flat)}/{len(keys)} requested keys. "
                f"Missing examples: {list(missing)[:5]}"
            )
    return flat


def _refuse_swiglu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Re-concat ``...linear_fc1.weight_w`` + ``...linear_fc1.weight_v``
    pairs back into the original fused ``...linear_fc1.weight``.

    FSDP split puts ``weight_w`` first (gate) and ``weight_v`` second (up).
    Megatron's SwiGLU expects the same order: ``cat([gate, up], dim=0)``.
    """
    out = OrderedDict()
    pending_w: dict[str, torch.Tensor] = {}
    pending_v: dict[str, torch.Tensor] = {}

    for key, tensor in state.items():
        if key.endswith(SWIGLU_W_SUFFIX):
            base = key[: -len(SWIGLU_W_SUFFIX)] + ".weight"
            pending_w[base] = tensor
        elif key.endswith(SWIGLU_V_SUFFIX):
            base = key[: -len(SWIGLU_V_SUFFIX)] + ".weight"
            pending_v[base] = tensor
        else:
            out[key] = tensor

    fused = 0
    for base, w in pending_w.items():
        v = pending_v.pop(base, None)
        if v is None:
            raise KeyError(
                f"Found {base}_w but no matching {base}_v in checkpoint. "
                "SwiGLU split is incomplete."
            )
        if w.shape != v.shape:
            raise ValueError(
                f"Shape mismatch for SwiGLU pair {base}: "
                f"weight_w={tuple(w.shape)} vs weight_v={tuple(v.shape)}"
            )
        out[base] = torch.cat([w, v], dim=0)
        fused += 1

    if pending_v:
        raise KeyError(
            f"Dangling weight_v entries with no weight_w: {list(pending_v)[:5]}"
        )

    if fused:
        print(f"[fuse] re-fused {fused} SwiGLU linear_fc1 pairs "
              f"(weight_w + weight_v -> weight)")
    return out


def consolidate(distcp_dir: Path, output_dir: Path) -> Path:
    """Materialize a legacy mp_rank_00/model_optim_rng.pt from distcp shards."""
    keys, iteration = _enumerate_model_keys(distcp_dir)
    if not keys:
        raise RuntimeError(
            f"No model weight keys (model.module.*) found in {distcp_dir}. "
            "Is this really an FSDP-dtensor checkpoint?"
        )

    raw = _load_tensors(distcp_dir, keys)

    # Strip the FSDP wrapper prefix.
    stripped = OrderedDict((_strip_fsdp_prefix(k), v) for k, v in raw.items())
    print(f"[strip] removed '{FSDP_PREFIX}' prefix from {len(stripped)} keys")

    # Re-fuse SwiGLU fc1 if present.
    fused = _refuse_swiglu(stripped)

    # Pack into the converter-expected envelope.
    payload = {
        "model": fused,
        "iteration": iteration,
        "checkpoint_version": 3.0,  # arbitrary, matches Megatron's writer
        "args": None,                # converters don't read this
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_dir = output_dir / "mp_rank_00"
    rank_dir.mkdir(exist_ok=True)
    out_path = rank_dir / "model_optim_rng.pt"
    torch.save(payload, out_path)

    # Also drop a manifest for debugging.
    manifest = {
        "source_distcp_dir": str(distcp_dir),
        "iteration": iteration,
        "num_tensors": len(fused),
        "tensor_keys": sorted(fused.keys()),
    }
    with open(output_dir / "consolidation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[save] {out_path}  ({size_mb:.1f} MB, {len(fused)} tensors, iter={iteration})")
    print(f"[save] {output_dir / 'consolidation_manifest.json'}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distcp-dir", required=True, type=Path,
                    help="Primus iter dir holding the .distcp shards "
                         "(e.g. .../checkpoints/iter_0095368)")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Where to write the consolidated mp_rank_00 layout. "
                         "Typically '<distcp-dir-parent>_consolidated/<iter_name>'.")
    args = ap.parse_args()

    if not args.distcp_dir.is_dir():
        ap.error(f"--distcp-dir does not exist or is not a directory: {args.distcp_dir}")
    if not (args.distcp_dir / ".metadata").is_file():
        ap.error(f"No .metadata file in {args.distcp_dir} — "
                 "is this an FSDP-dtensor checkpoint?")

    print("=" * 78)
    print(" Primus FSDP-dtensor -> legacy mp_rank_00 consolidator")
    print("=" * 78)
    print(f"  source   = {args.distcp_dir}")
    print(f"  dest     = {args.output_dir}")
    print()

    out_path = consolidate(args.distcp_dir, args.output_dir)
    print()
    print("Done. Feed this directory to any of the FLA HF converters, e.g.:")
    print()
    print(f"    python3 tools/convert_gdn_to_fla_hf.py \\")
    print(f"        --checkpoint-path {args.output_dir} \\")
    print(f"        --output-dir output/gdn_pure_1B_fla_hf \\")
    print(f"        --config /home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs/gated_deltanet_1B_pure_100B.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
