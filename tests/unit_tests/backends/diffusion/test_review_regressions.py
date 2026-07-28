from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from primus.backends.diffusion.data.collator import VisionCollator
from primus.backends.diffusion.data.dataset import WanVideoDataset
from primus.backends.diffusion.diffusion_pretrain_trainer import (
    DiffusionPretrainTrainer,
)


def test_vision_collator_uses_provided_attention_mask_without_input_ids():
    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0

    processor = SimpleNamespace(tokenizer=Tokenizer())
    collator = VisionCollator(processor=processor)

    batch = collator(
        [
            {"attention_mask": torch.tensor([1, 1]), "pixel_values": torch.ones(2)},
            {"attention_mask": torch.tensor([1]), "pixel_values": torch.zeros(2)},
        ]
    )

    torch.testing.assert_close(batch["attention_mask"], torch.tensor([[1, 1], [1, 0]]))
    torch.testing.assert_close(batch["pixel_values"], torch.tensor([[1.0, 1.0], [0.0, 0.0]]))


def test_wan_video_dataset_requires_config_with_dataset_path():
    with pytest.raises(ValueError, match="dataset config with dataset_path"):
        WanVideoDataset(processor=object())

    with pytest.raises(ValueError, match="config.dataset_path"):
        WanVideoDataset(processor=object(), config=SimpleNamespace(dataset_path=None))


def test_diffusion_setup_reports_missing_flux_dependencies(monkeypatch):
    missing = {"datasets", "huggingface_hub", "sentencepiece", "webdataset"}

    def fake_find_spec(package):
        return None if package in missing else object()

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)
    trainer = DiffusionPretrainTrainer.__new__(DiffusionPretrainTrainer)
    trainer.backend_args = SimpleNamespace(
        trainer={"args": {}},
        dataset={
            "name": "flux",
            "config": {
                "dataset_type": "raw",
                "dataset_format": "webdataset",
            },
        },
    )

    with pytest.raises(RuntimeError) as exc_info:
        trainer.setup()

    message = str(exc_info.value)
    for package in sorted(missing):
        assert package in message
