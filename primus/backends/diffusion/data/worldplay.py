###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Precomputed latent dataset for HY-WorldPlay autoregressive SFT."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


_ACTION_MAPPING = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}


def _resolve_path(value: str, index_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else index_dir / path


def _camera_center_normalize(w2c: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(w2c)
    first_inv = np.linalg.inv(c2w[0])
    return np.linalg.inv(np.asarray([first_inv @ pose for pose in c2w]))


def _rotation_xyz_degrees(matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert a proper rotation matrix to XYZ Euler angles without scipy."""
    sy = math.sqrt(float(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        y = math.atan2(float(-matrix[2, 0]), sy)
        z = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        x = math.atan2(float(-matrix[1, 2]), float(matrix[1, 1]))
        y = math.atan2(float(-matrix[2, 0]), sy)
        z = 0.0
    scale = 180.0 / math.pi
    return x * scale, y * scale, z * scale


def _actions_from_w2c(w2c: np.ndarray) -> torch.Tensor:
    """Discretize relative camera motion into WorldPlay's 9x9 action IDs."""
    c2w = np.linalg.inv(w2c)
    relative = np.zeros_like(c2w)
    relative[0] = c2w[0]
    relative[1:] = np.linalg.inv(c2w[:-1]) @ c2w[1:]
    translation = np.zeros((len(relative), 4), dtype=np.int64)
    rotation = np.zeros((len(relative), 4), dtype=np.int64)

    for index in range(1, len(relative)):
        direction = relative[index, :3, 3]
        norm = float(np.linalg.norm(direction))
        if norm > 0.01:
            angles = np.degrees(np.arccos(np.clip(direction / norm, -1.0, 1.0)))
            if angles[2] < 60:
                translation[index, 0] = 1
            elif angles[2] > 120:
                translation[index, 1] = 1
            if angles[0] < 60:
                translation[index, 2] = 1
            elif angles[0] > 120:
                translation[index, 3] = 1

        rot_x, rot_y, _ = _rotation_xyz_degrees(relative[index, :3, :3])
        if rot_y > 0.05:
            rotation[index, 0] = 1
        elif rot_y < -0.05:
            rotation[index, 1] = 1
        if rot_x > 0.05:
            rotation[index, 2] = 1
        elif rot_x < -0.05:
            rotation[index, 3] = 1

    move = torch.tensor([_ACTION_MAPPING[tuple(row)] for row in translation])
    view = torch.tensor([_ACTION_MAPPING[tuple(row)] for row in rotation])
    return move * 9 + view


def _memory_indices(
    w2c: np.ndarray,
    current: int,
    *,
    memory_frames: int,
    temporal_context: int = 12,
    prediction_frames: int = 4,
) -> list[int]:
    """Select spatially/view-aligned history plus recent temporal context."""
    recent = list(range(max(0, current - temporal_context), current))
    selected = set(recent)
    selected.update(range(min(4, current)))
    if len(selected) >= memory_frames:
        return sorted(selected)[-memory_frames:]

    c2w = np.linalg.inv(w2c)
    query = c2w[current : min(current + prediction_frames, len(c2w))]
    query_center = query[:, :3, 3].mean(axis=0)
    query_forward = query[:, :3, 2].mean(axis=0)
    query_forward /= max(float(np.linalg.norm(query_forward)), 1e-8)

    candidates: list[tuple[float, int]] = []
    for start in range(4, max(4, current - temporal_context), 4):
        pose = c2w[start]
        position_distance = float(np.linalg.norm(pose[:3, 3] - query_center))
        forward = pose[:3, 2]
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        angular_distance = 1.0 - float(np.clip(forward @ query_forward, -1.0, 1.0))
        candidates.append((position_distance + angular_distance, start))

    for _, start in sorted(candidates):
        selected.update(range(start, min(start + 4, current)))
        if len(selected) >= memory_frames:
            break
    return sorted(selected)[-memory_frames:]


class WorldPlayLatentProcessor:
    """Processor hook expected by the generic Primus diffusion trainer."""

    def prepare_batch(self, batch: dict[str, Any], device, dtype):
        floating = {
            "latent",
            "prompt_embed",
            "w2c",
            "intrinsic",
            "image_cond",
            "vision_states",
            "prompt_mask",
            "byt5_text_states",
            "byt5_text_mask",
            "i2v_mask",
        }
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                target_dtype = dtype if key in floating and value.is_floating_point() else value.dtype
                batch[key] = value.to(device=device, dtype=target_dtype, non_blocking=True)
        return batch


class WorldPlayLatentCollator:
    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
        return result


class WorldPlayLatentDataset(Dataset):
    """Load Hunyuan latents, embeddings, camera poses, and derived actions."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.index_path = Path(config["dataset_path"]).expanduser().resolve()
        with self.index_path.open() as handle:
            self.samples = json.load(handle)
        if not isinstance(self.samples, list) or not self.samples:
            raise ValueError(f"WorldPlay index must be a non-empty JSON list: {self.index_path}")

        self.window_frames = int(config.get("window_frames", 24))
        self.memory_frames = int(config.get("memory_frames", 20))
        self.max_frames = int(config.get("max_frames", 32))
        self.cfg_rate = float(config.get("cfg_rate", 0.1))
        self.memory_sample_rate = float(config.get("memory_sample_rate", 0.8))
        self.seed = int(config.get("data_seed", 42))
        if self.window_frames != self.memory_frames + 4:
            raise ValueError("WorldPlay requires window_frames == memory_frames + 4")

        self.neg_prompt = self._load_optional(config.get("negative_prompt_path"))
        self.neg_byt5 = self._load_optional(config.get("negative_byt5_prompt_path"))
        if self.cfg_rate > 0 and (self.neg_prompt is None or self.neg_byt5 is None):
            raise ValueError(
                "cfg_rate > 0 requires negative_prompt_path and negative_byt5_prompt_path"
            )

    def _load_optional(self, value: str | None):
        if not value:
            return None
        path = _resolve_path(value, self.index_path.parent)
        return torch.load(path, map_location="cpu", weights_only=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_poses(self, path: Path, latent_length: int) -> tuple[np.ndarray, torch.Tensor]:
        with path.open() as handle:
            poses = json.load(handle)
        keys = list(poses)
        required_rgb_index = 4 * (latent_length - 1)
        if len(keys) <= required_rgb_index:
            raise ValueError(
                f"{path} has {len(keys)} RGB poses; need at least {required_rgb_index + 1}"
            )
        intrinsics, w2c = [], []
        for latent_index in range(latent_length):
            rgb_index = 0 if latent_index == 0 else 4 * latent_index
            item = poses[keys[rgb_index]]
            intrinsic = np.asarray(item["intrinsic"], dtype=np.float64).copy()
            intrinsic[0, 0] /= intrinsic[0, 2] * 2
            intrinsic[1, 1] /= intrinsic[1, 2] * 2
            intrinsic[0, 2] = intrinsic[1, 2] = 0.5
            intrinsics.append(intrinsic)
            w2c.append(np.asarray(item["w2c"], dtype=np.float64))
        return _camera_center_normalize(np.asarray(w2c)), torch.tensor(np.asarray(intrinsics))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        latent_path = _resolve_path(row["latent_path"], self.index_path.parent)
        pose_path = _resolve_path(row["pose_path"], self.index_path.parent)
        packed = torch.load(latent_path, map_location="cpu", weights_only=True)
        latent = packed["latent"][0]
        latent_length = min(latent.shape[1], self.max_frames)
        latent = latent[:, :latent_length]
        if latent_length < self.window_frames:
            raise ValueError(
                f"{latent_path} has {latent_length} latents; need {self.window_frames}"
            )

        w2c, intrinsic = self._load_poses(pose_path, latent_length)
        action = _actions_from_w2c(w2c)
        rng = random.Random(self.seed + index)
        max_current = latent_length - 4
        use_memory = (
            rng.random() < self.memory_sample_rate
            and max_current >= self.window_frames
        )
        if use_memory:
            current = rng.randrange(self.window_frames // 4, max_current // 4 + 1) * 4
            selected = _memory_indices(
                w2c, current, memory_frames=self.memory_frames
            ) + list(range(current, current + 4))
            selected = selected[-self.window_frames :]
        else:
            selected = list(range(self.window_frames))

        latent = latent[:, selected]
        w2c_tensor = torch.tensor(w2c[selected])
        intrinsic = intrinsic[selected]
        action = action[selected]
        prompt_embed = packed["prompt_embeds"][0]
        prompt_mask = packed["prompt_mask"][0]
        byt5_states = packed["byt5_text_states"][0]
        byt5_mask = packed["byt5_text_mask"][0]
        if rng.random() < self.cfg_rate:
            prompt_embed = self.neg_prompt["negative_prompt_embeds"][0]
            prompt_mask = self.neg_prompt["negative_prompt_mask"][0]
            byt5_states = self.neg_byt5["byt5_text_states"][0]
            byt5_mask = self.neg_byt5["byt5_text_mask"][0]

        return {
            "latent": latent,
            "prompt_embed": prompt_embed,
            "prompt_mask": prompt_mask,
            "byt5_text_states": byt5_states,
            "byt5_text_mask": byt5_mask,
            "image_cond": packed["image_cond"][0],
            "vision_states": packed["vision_states"][0],
            "w2c": w2c_tensor,
            "intrinsic": intrinsic,
            "action": action,
            "i2v_mask": torch.ones_like(latent),
            "memory_sample": torch.tensor(use_memory),
            "video_path": str(pose_path),
        }

    def get_collator(self):
        return WorldPlayLatentCollator()
