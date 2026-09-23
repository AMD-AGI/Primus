"""DL3DV download, scene reading, camera, and latent packing helpers.

Modified from Tencent HY-WorldPlay preprocessing source by AMD in 2026:
the reusable scene and packing paths are provided as an in-tree module.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DL3DV_REPO = "DL3DV/DL3DV-ALL-480P"
DL3DV_META_URL = (
    "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/"
    "cache/DL3DV-valid.csv"
)


def meta_rows():
    with urllib.request.urlopen(DL3DV_META_URL) as response:
        return list(csv.DictReader(io.TextIOWrapper(response, encoding="utf-8")))


def download_scene_hashes(scene_hashes, raw_dir, token):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download DL3DV scenes"
        ) from exc

    hash_to_batch = {row["hash"]: row["batch"] for row in meta_rows()}
    paths = []
    for scene_hash in scene_hashes:
        if scene_hash not in hash_to_batch:
            raise ValueError(f"Unknown DL3DV scene hash: {scene_hash}")
        batch = hash_to_batch[scene_hash]
        filename = f"{batch}/{scene_hash}.zip"
        local_zip = Path(raw_dir) / filename
        if local_zip.exists():
            print(f"Reusing {local_zip}")
            paths.append(local_zip)
            continue
        print(f"Downloading {DL3DV_REPO}/{filename} ...")
        paths.append(
            Path(
                hf_hub_download(
                    repo_id=DL3DV_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token,
                    local_dir=raw_dir,
                )
            )
        )
    return paths


class SceneReader:
    """Read transforms.json and images from an extracted scene or zip."""

    def __init__(self, source):
        self.source = Path(source)
        self.archive = None
        if self.source.is_file() and self.source.suffix.lower() == ".zip":
            self.archive = zipfile.ZipFile(self.source)
            names = self.archive.namelist()
            transforms = [name for name in names if name.endswith("transforms.json")]
            if not transforms:
                raise ValueError(f"No transforms.json in {self.source}")
            self.transforms_path = transforms[0]
            self.names = [name for name in names if not name.endswith("/")]
            self.data = json.loads(self.archive.read(self.transforms_path))
        elif self.source.is_dir():
            transforms = sorted(self.source.rglob("transforms.json"))
            if not transforms:
                raise ValueError(f"No transforms.json under {self.source}")
            self.transforms_path = transforms[0]
            self.data = json.loads(self.transforms_path.read_text())
            self.names = [
                path.relative_to(self.transforms_path.parent).as_posix()
                for path in self.transforms_path.parent.rglob("*")
                if path.is_file()
            ]
        else:
            raise FileNotFoundError(source)

        if self.archive is not None:
            self.base = str(PurePosixPath(self.transforms_path).parent)
            if self.base == ".":
                self.base = ""
        else:
            self.base = ""
        self._basename_map = {}
        for name in self.names:
            self._basename_map.setdefault(PurePosixPath(name).name, []).append(name)

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def _resolve(self, file_path):
        relative = file_path.lstrip("./")
        direct = f"{self.base}/{relative}" if self.base else relative
        candidates = [direct]
        if not PurePosixPath(relative).suffix:
            candidates.extend(direct + ext for ext in (".png", ".jpg", ".jpeg"))
        names = set(self.names)
        for candidate in candidates:
            if candidate in names:
                return candidate

        basename = PurePosixPath(relative).name
        basename_candidates = [basename]
        if not PurePosixPath(basename).suffix:
            basename_candidates.extend(
                basename + ext for ext in (".png", ".jpg", ".jpeg")
            )
        matches = []
        for item in basename_candidates:
            matches.extend(self._basename_map.get(item, []))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"Cannot resolve image {file_path!r} in {self.source}")

    def read_image(self, frame):
        name = self._resolve(frame["file_path"])
        if self.archive is not None:
            return Image.open(io.BytesIO(self.archive.read(name))).convert("RGB")
        return Image.open(self.transforms_path.parent / name).convert("RGB")


def _frame_intrinsic(metadata, frame, image_size, target_size):
    values = {**metadata, **frame}
    source_w = float(values.get("w", image_size[0]))
    source_h = float(values.get("h", image_size[1]))
    fx = float(values["fl_x"])
    fy = float(values.get("fl_y", fx))
    cx = float(values.get("cx", source_w / 2.0))
    cy = float(values.get("cy", source_h / 2.0))
    target_h, target_w = target_size
    sx, sy = target_w / source_w, target_h / source_h
    return [
        [fx * sx, 0.0, cx * sx],
        [0.0, fy * sy, cy * sy],
        [0.0, 0.0, 1.0],
    ]


def _opencv_w2c(transform_matrix):
    """Convert Nerfstudio/OpenGL c2w to WorldPlay's +Z-forward camera axes."""
    c2w = np.asarray(transform_matrix, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform_matrix, got {c2w.shape}")
    return np.linalg.inv(c2w @ np.diag([1.0, -1.0, -1.0, 1.0]))


def clip_starts(frame_count, clip_span, clips_per_scene):
    if frame_count < clip_span:
        return []
    max_start = frame_count - clip_span
    if clips_per_scene == 1:
        return [max_start // 2]
    return np.linspace(0, max_start, clips_per_scene, dtype=int).tolist()


@torch.no_grad()
def encode_video(extractor, images, caption):
    tensors = []
    for image in images:
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        tensor = (tensor / 255.0 - 0.5) * 2.0
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=extractor.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        tensors.append(tensor)
    video = torch.stack(tensors, dim=1).unsqueeze(0).to(extractor.device)
    extractor.vae.disable_tiling()
    latents = extractor.vae.encode(video).latent_dist.mode()
    latents.mul_(extractor.vae.config.scaling_factor)
    cond_latents = latents[:, :, :1].clone()

    ref_image = (tensors[0] + 1) * 127.5
    vision_states = extractor.vision_encoder.encode_images(
        ref_image.numpy().astype(np.uint8)
    ).last_hidden_state.to(device=extractor.device, dtype=torch.bfloat16)
    prompt, mask, byt5, byt5_mask = extractor.encode_caption(caption)
    return {
        "latent": latents.to(torch.bfloat16).cpu(),
        "prompt_embeds": prompt.cpu(),
        "image_cond": cond_latents.to(torch.bfloat16).cpu(),
        "vision_states": vision_states.cpu(),
        "prompt_mask": None if mask is None else mask.cpu(),
        "byt5_text_states": byt5.cpu(),
        "byt5_text_mask": byt5_mask.cpu(),
    }


def build_clip(reader, frame_indices, target_size):
    frames = reader.data["frames"]
    images, poses = [], {}
    for output_idx, frame_idx in enumerate(frame_indices):
        frame = frames[frame_idx]
        image = reader.read_image(frame)
        images.append(image)
        poses[str(output_idx)] = {
            "intrinsic": _frame_intrinsic(
                reader.data, frame, image.size, target_size
            ),
            "w2c": _opencv_w2c(frame["transform_matrix"]).tolist(),
        }
    return images, poses
