###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import json

import torch

from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.data.worldplay import WorldPlayLatentDataset
from primus.backends.diffusion.models.worldplay.train_pipeline import (
    WorldPlayARTrainPipeline,
)
from primus.backends.diffusion.registry import DATASET_BUILDERS, MODEL_BUILDERS


def _write_dataset(tmp_path):
    latent_path = tmp_path / "clip.pt"
    pose_path = tmp_path / "poses.json"
    index_path = tmp_path / "train.json"
    torch.save(
        {
            "latent": torch.zeros(1, 2, 32, 2, 2),
            "prompt_embeds": torch.zeros(1, 3, 4),
            "prompt_mask": torch.ones(1, 3),
            "image_cond": torch.zeros(1, 2, 1, 2, 2),
            "vision_states": torch.zeros(1, 2, 4),
            "byt5_text_states": torch.zeros(1, 2, 4),
            "byt5_text_mask": torch.ones(1, 2),
        },
        latent_path,
    )
    poses = {}
    for index in range(125):
        w2c = torch.eye(4)
        w2c[2, 3] = -0.01 * index
        poses[str(index)] = {
            "intrinsic": [[832.0, 0.0, 416.0], [0.0, 832.0, 240.0], [0.0, 0.0, 1.0]],
            "w2c": w2c.tolist(),
        }
    pose_path.write_text(json.dumps(poses))
    index_path.write_text(
        json.dumps([{"latent_path": "clip.pt", "pose_path": "poses.json"}])
    )
    return index_path


def test_worldplay_is_registered():
    assert "worldplay" in MODEL_BUILDERS
    assert "worldplay" in DATASET_BUILDERS


def test_worldplay_argument_defaults_and_data_overrides():
    builder = DiffusionArgBuilder()
    builder.update(
        {
            "model": {"name": "worldplay", "config": {}},
            "data": {
                "dataset_path": "/data/train.json",
                "window_frames": 24,
                "memory_frames": 20,
                "max_frames": 64,
                "cfg_rate": 0.0,
            },
            "parallelism": {"sp_size": 8},
        }
    )
    result = builder.finalize()
    assert result.dataset["name"] == "worldplay"
    assert result.dataset["config"]["max_frames"] == 64
    assert result.dataset["config"]["cfg_rate"] == 0.0
    assert result.trainer["args"]["sp_size"] == 8
    assert result.trainer["args"]["save_strategy"] == "none"


def test_worldplay_dataset_loads_latents_poses_and_actions(tmp_path):
    dataset = WorldPlayLatentDataset(
        {
            "dataset_path": str(_write_dataset(tmp_path)),
            "window_frames": 24,
            "memory_frames": 20,
            "max_frames": 32,
            "cfg_rate": 0.0,
            "memory_sample_rate": 1.0,
            "data_seed": 1,
        }
    )
    sample = dataset[0]
    assert sample["latent"].shape == (2, 24, 2, 2)
    assert sample["w2c"].shape == (24, 4, 4)
    assert sample["intrinsic"].shape == (24, 3, 3)
    assert sample["action"].shape == (24,)
    assert sample["memory_sample"].item() is True
    collated = dataset.get_collator()([sample, sample])
    assert collated["latent"].shape == (2, 2, 24, 2, 2)


def test_worldplay_train_pipeline_builds_action_camera_forward():
    class FakeDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return (torch.zeros_like(kwargs["hidden_states"][:, :2]),)

    dit = FakeDiT()
    batch = {
        "latent": torch.zeros(1, 2, 24, 2, 2),
        "image_cond": torch.zeros(1, 2, 1, 2, 2),
        "prompt_embed": torch.zeros(1, 3, 4),
        "prompt_mask": torch.ones(1, 3),
        "vision_states": torch.zeros(1, 2, 4),
        "byt5_text_states": torch.zeros(1, 2, 4),
        "byt5_text_mask": torch.ones(1, 2),
        "w2c": torch.eye(4).repeat(1, 24, 1, 1),
        "intrinsic": torch.eye(3).repeat(1, 24, 1, 1),
        "action": torch.zeros(1, 24),
        "i2v_mask": torch.ones(1, 2, 24, 2, 2),
        "memory_sample": torch.tensor([True]),
    }
    output = WorldPlayARTrainPipeline().compute_loss(dit=dit, batch=batch)
    assert torch.isfinite(output["loss"])
    assert dit.kwargs["hidden_states"].shape == (1, 5, 24, 2, 2)
    assert dit.kwargs["action"].shape == (24,)
    assert dit.kwargs["viewmats"] is batch["w2c"]
