#!/usr/bin/env python3
"""Prepare 100/500/1K-scene DL3DV subsets for Primus WorldPlay SFT."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from worldplay_preprocess import (  # noqa: E402
    LatentExtractor,
    SceneReader,
    build_clip,
    clip_starts,
    download_scene_hashes,
    encode_video,
    meta_rows,
)

DL3DV_BATCHES = tuple(f"{index}K" for index in range(1, 12))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hunyuan-checkpoint-path",
        help="HunyuanVideo-1.5 snapshot; required unless --download-only.",
    )
    parser.add_argument("--batch", choices=DL3DV_BATCHES, default="1K")
    parser.add_argument(
        "--scene-count",
        type=int,
        choices=(100, 500, 1000),
        default=100,
        help="Number of DL3DV scenes to select from --batch.",
    )
    parser.add_argument(
        "--scene-hash",
        action="append",
        help="Explicit scene hash; repeatable and overrides --scene-count.",
    )
    parser.add_argument("--selection-seed", type=int, default=3208)
    parser.add_argument("--raw-dir", default="/data/dataset/dl3dv_raw")
    parser.add_argument("--output-dir", default="/data/dataset/worldplay_dl3dv_100")
    parser.add_argument("--num-frames", type=int, default=125)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--clips-per-scene", type=int, default=1)
    parser.add_argument(
        "--index-repeats",
        type=int,
        default=1,
        help="Repeat index rows without duplicating latent files.",
    )
    parser.add_argument(
        "--caption",
        default="A handheld camera moves through a real-world scene.",
    )
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--target-width", type=int, default=832)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def select_hashes(*, batch: str, count: int, seed: int) -> list[str]:
    available = [row["hash"] for row in meta_rows() if row["batch"] == batch]
    if len(available) < count:
        raise ValueError(
            f"DL3DV batch {batch} has {len(available)} scenes; cannot select {count}"
        )
    random.Random(seed).shuffle(available)
    return available[:count]


def write_negative_prompts(extractor, output_dir: Path):
    target = output_dir / "neg_prompts"
    target.mkdir(parents=True, exist_ok=True)
    prompt, mask, byt5, byt5_mask = extractor.encode_caption("")
    torch.save(
        {
            "negative_prompt_embeds": prompt.cpu(),
            "negative_prompt_mask": (
                mask.cpu()
                if mask is not None
                else torch.ones(prompt.shape[:2], dtype=torch.int64)
            ),
        },
        target / "hunyuan_neg_prompt.pt",
    )
    torch.save(
        {
            "byt5_text_states": byt5.cpu(),
            "byt5_text_mask": byt5_mask.cpu(),
        },
        target / "hunyuan_neg_byt5_prompt.pt",
    )


def main():
    args = parse_args()
    if args.num_frames < 109 or (args.num_frames - 1) % 4:
        raise ValueError("--num-frames must be 4*n+1 and yield at least 28 latents")
    if min(args.frame_stride, args.clips_per_scene, args.index_repeats) < 1:
        raise ValueError("stride, clips-per-scene, and repeats must be positive")
    if not args.download_only and not args.hunyuan_checkpoint_path:
        raise ValueError("--hunyuan-checkpoint-path is required for encoding")

    hashes = args.scene_hash or select_hashes(
        batch=args.batch,
        count=args.scene_count,
        seed=args.selection_seed,
    )
    sources = download_scene_hashes(hashes, args.raw_dir, args.hf_token)
    if args.download_only:
        print(f"Downloaded/reused {len(sources)} DL3DV scene archives")
        return

    output = Path(args.output_dir).expanduser().resolve()
    latents_dir = output / "latents"
    poses_dir = output / "poses"
    latents_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = LatentExtractor(
        args.hunyuan_checkpoint_path,
        device,
        (args.target_height, args.target_width),
    )

    clip_span = (args.num_frames - 1) * args.frame_stride + 1
    encoded = []
    skipped = []
    for source in sources:
        reader = SceneReader(source)
        try:
            frame_count = len(reader.data.get("frames", []))
            starts = clip_starts(frame_count, clip_span, args.clips_per_scene)
            if not starts:
                skipped.append(str(source))
                continue
            for start in tqdm(starts, desc=f"encode {source.stem}"):
                indices = [
                    start + offset * args.frame_stride
                    for offset in range(args.num_frames)
                ]
                images, poses = build_clip(
                    reader,
                    indices,
                    (args.target_height, args.target_width),
                )
                packed = encode_video(extractor, images, args.caption)
                clip_id = len(encoded)
                latent_path = latents_dir / f"{clip_id:06d}.pt"
                pose_path = poses_dir / f"{clip_id:06d}.json"
                torch.save(packed, latent_path)
                pose_path.write_text(json.dumps(poses))
                encoded.append(
                    {
                        "latent_path": str(latent_path),
                        "pose_path": str(pose_path),
                        "scene_hash": source.stem,
                    }
                )
        finally:
            reader.close()

    if not encoded:
        raise RuntimeError(
            f"No clips encoded; {len(skipped)} scenes had fewer than "
            f"{clip_span} registered frames"
        )
    rows = encoded * args.index_repeats
    (output / "train.json").write_text(json.dumps(rows, indent=2))
    write_negative_prompts(extractor, output)
    (output / "selection.json").write_text(
        json.dumps(
            {
                "batch": args.batch,
                "requested_scene_count": len(hashes),
                "selected_hashes": hashes,
                "encoded_clips": len(encoded),
                "skipped_scenes": skipped,
                "selection_seed": args.selection_seed,
            },
            indent=2,
        )
    )
    print(
        f"Wrote {len(encoded)} clips ({len(rows)} index rows) to "
        f"{output / 'train.json'}"
    )


if __name__ == "__main__":
    main()
