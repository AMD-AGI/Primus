#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Rename a VGGT safetensors checkpoint to WorldMirror parameter names.

WorldMirror stage 1 starts from pretrained VGGT weights (arXiv 2510.10726,
"Decoupled Sequential Training"). The WorldMirror network reuses the VGGT
aggregator, camera head, point head and depth head under different module names,
and the training loaders copy tensors by exact name. This script writes a
``.safetensors`` file whose names match WorldMirror, so either trainer can load it
as ``pretrained``. Tensors that WorldMirror adds (depth/pose/ray embeds, normal
head, Gaussian head) are absent and keep their random initialisation. The VGGT
track head has no WorldMirror counterpart and is dropped.

No torch is needed: the safetensors container is a JSON header followed by raw bytes.

    python3 convert_vggt_to_worldmirror.py model.safetensors out.safetensors \
        [--reference worldmirror/model.safetensors]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

_EXACT = {
    "aggregator.camera_token": "visual_geometry_transformer.cam_token",
    "aggregator.register_token": "visual_geometry_transformer.reg_token",
}
# Ordered: first matching prefix wins.
_PREFIX = (
    ("aggregator.", "visual_geometry_transformer."),
    ("camera_head.trunk.", "cam_head.refine_net."),
    ("camera_head.embed_pose.", "cam_head.param_embed."),
    ("camera_head.empty_pose_tokens", "cam_head.init_token"),
    ("camera_head.poseLN_modulation.", "cam_head.adapt_norm_gen."),
    ("camera_head.pose_branch.", "cam_head.param_predictor."),
    ("camera_head.token_norm.", "cam_head.token_norm."),
    ("camera_head.trunk_norm.", "cam_head.out_norm."),
    ("point_head.", "pts_head."),
    ("depth_head.", "depth_head."),
)


def map_name(name: str) -> str | None:
    if name in _EXACT:
        return _EXACT[name]
    for old, new in _PREFIX:
        if name.startswith(old):
            return new + name[len(old) :]
    return None


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return header, 8 + size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", help="VGGT model.safetensors")
    parser.add_argument("dst", help="output .safetensors with WorldMirror names")
    parser.add_argument("--reference", help="WorldMirror safetensors used to check names and shapes")
    args = parser.parse_args()

    header, data_start = read_header(args.src)
    metadata = header.pop("__metadata__", None)

    reference = None
    if args.reference:
        reference, _ = read_header(args.reference)
        reference.pop("__metadata__", None)

    plan = []  # (new_name, entry)
    dropped, bad_shape, unknown = [], [], []
    for name, entry in header.items():
        new = map_name(name)
        if new is None:
            dropped.append(name)
            continue
        if reference is not None:
            if new not in reference:
                unknown.append((name, new))
                continue
            if list(reference[new]["shape"]) != list(entry["shape"]):
                bad_shape.append((name, new, entry["shape"], reference[new]["shape"]))
                continue
        plan.append((new, entry))

    if reference is not None:
        covered = {new for new, _ in plan}
        missing = sorted(set(reference) - covered)
        print(f"reference tensors: {len(reference)}  covered by VGGT: {len(covered)}  new in WorldMirror: {len(missing)}")
        prefixes = sorted({m.split(".")[0] for m in missing})
        print(f"  new tensors live in: {prefixes}")
    print(f"mapped: {len(plan)}  dropped (no WorldMirror counterpart): {len(dropped)}")
    if unknown or bad_shape:
        for item in unknown[:10]:
            print("  UNKNOWN target:", item)
        for item in bad_shape[:10]:
            print("  SHAPE mismatch:", item)
        print("refusing to write: names or shapes do not line up with the reference")
        return 1

    new_header, offset = {}, 0
    for new, entry in plan:
        begin, end = entry["data_offsets"]
        new_header[new] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    if metadata is not None:
        new_header["__metadata__"] = metadata
    blob = json.dumps(new_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)

    with open(args.src, "rb") as src, open(args.dst, "wb") as dst:
        dst.write(struct.pack("<Q", len(blob)))
        dst.write(blob)
        for _, entry in plan:
            begin, end = entry["data_offsets"]
            src.seek(data_start + begin)
            remaining = end - begin
            while remaining:
                chunk = src.read(min(remaining, 64 << 20))
                if not chunk:
                    raise RuntimeError("unexpected end of source file")
                dst.write(chunk)
                remaining -= len(chunk)
    print(f"wrote {args.dst} ({offset / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
