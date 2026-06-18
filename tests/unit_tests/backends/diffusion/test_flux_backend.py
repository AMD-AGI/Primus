from __future__ import annotations

import numpy as np
import pytest
import torch

from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.data.flux_precomputed import (
    FluxPrecomputedProcessor,
    FluxRawImageTextDataset,
    FluxRawImageTextProcessor,
)
from primus.backends.diffusion.models.registrations.flux import build_flux_model


def _finalize(params: dict):
    builder = DiffusionArgBuilder()
    builder.update(params)
    return builder.finalize()


def test_flux_argument_builder_selects_flux_defaults():
    args = _finalize(
        {
            "model": {"name": "flux", "config": {"config": {"model_variant": "flux-dev"}}},
            "training": {"steps": 7, "local_batch_size": 3},
            "data": {
                "dataset_path": "/tmp/precomputed",
                "dataset_type": "precomputed",
                "empty_encodings_path": "/tmp/empty",
                "prompt_dropout_prob": 0.25,
            },
        }
    )

    assert args.model["name"] == "flux"
    assert args.dataset["name"] == "flux"
    assert args.dataset["config"]["dataset_path"] == "/tmp/precomputed"
    assert args.dataset["config"]["processor_config"]["empty_encodings_path"] == "/tmp/empty"
    assert args.dataset["config"]["processor_config"]["prompt_dropout_prob"] == 0.25
    assert args.trainer["args"]["max_steps"] == 7
    assert args.trainer["args"]["per_device_train_batch_size"] == 3
    assert args.trainer["args"]["fsdp_transformer_layer_cls_to_wrap"] == "DoubleStreamBlock,SingleStreamBlock"


def test_flux_argument_builder_maps_raw_dataset_type():
    args = _finalize(
        {
            "model": {"name": "flux", "config": {"config": {"model_variant": "flux-dev"}}},
            "data": {
                "dataset_type": "raw",
                "dataset_format": "webdataset",
                "dataset_path": "/tmp/cc12m_test",
                "prompt_dropout_prob": 0.1,
                "img_size": 128,
            },
        }
    )

    assert args.dataset["config"]["dataset_type"] == "raw"
    assert args.dataset["config"]["dataset_format"] == "webdataset"
    assert args.dataset["config"]["processor_config"]["img_size"] == 128


def test_flux_raw_dataset_name_defaults():
    path, fmt = FluxRawImageTextDataset._resolve_dataset("cc12m-test", None, "webdataset")
    assert path.endswith("torchtitan-main/tests/assets/cc12m_test")
    assert fmt == "webdataset"

    path, fmt = FluxRawImageTextDataset._resolve_dataset("cc12m-wds", None, "webdataset")
    assert path == "pixparse/cc12m-wds"
    assert fmt == "hf_repo"


def test_flux_argument_builder_rejects_sequence_parallelism():
    with pytest.raises(ValueError, match="sp_size"):
        _finalize(
            {
                "model": {"name": "flux", "config": {"config": {"model_variant": "flux-dev"}}},
                "parallelism": {"sp_size": 2},
            }
        )


def test_flux_precomputed_processor_stacks_and_drops_empty_encodings(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    np.save(empty_dir / "t5_empty.npy", np.zeros((1, 3, 8), dtype=np.float32))
    np.save(empty_dir / "clip_empty.npy", np.zeros((1, 4), dtype=np.float32))

    processor = FluxPrecomputedProcessor(
        {
            "prompt_dropout_prob": 1.0,
            "empty_encodings_path": str(empty_dir),
        }
    )
    batch = [
        {
            "t5_encodings": torch.ones(3, 8),
            "clip_encodings": torch.ones(4),
            "mean": torch.zeros(1, 2, 2),
            "logvar": torch.zeros(1, 2, 2),
        },
        {
            "t5_encodings": torch.ones(3, 8),
            "clip_encodings": torch.ones(4),
            "mean": torch.zeros(1, 2, 2),
            "logvar": torch.zeros(1, 2, 2),
        },
    ]

    out = processor.prepare_batch(batch=batch, device=torch.device("cpu"), dtype=torch.float32)

    assert out["t5_encodings"].shape == (2, 3, 8)
    assert out["clip_encodings"].shape == (2, 4)
    assert torch.count_nonzero(out["t5_encodings"]) == 0
    assert torch.count_nonzero(out["clip_encodings"]) == 0


def test_flux_raw_processor_prepares_images_and_prompts():
    from PIL import Image

    processor = FluxRawImageTextProcessor({"img_size": 8, "prompt_dropout_prob": 1.0, "skip_low_resolution": False})
    image = Image.fromarray(np.full((6, 10, 3), 127, dtype=np.uint8))

    out = processor.prepare_batch(
        batch=[{"image": image, "prompt": "a test image"}],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert out["image"].shape == (1, 3, 8, 8)
    assert out["prompts"] == [""]


def test_tiny_flux_model_computes_precomputed_loss():
    model = build_flux_model(
        {
            "config": {
                "model_variant": "flux-dev",
                "guidance": 1.0,
                "params": {
                    "in_channels": 4,
                    "out_channels": 4,
                    "vec_in_dim": 4,
                    "context_in_dim": 8,
                    "hidden_size": 12,
                    "num_heads": 2,
                    "depth": 1,
                    "depth_single_blocks": 1,
                    "axes_dim": [2, 2, 2],
                },
            }
        }
    )
    batch = {
        "t5_encodings": torch.randn(2, 3, 8),
        "clip_encodings": torch.randn(2, 4),
        "mean": torch.randn(2, 1, 2, 2),
        "logvar": torch.zeros(2, 1, 2, 2),
    }

    outputs = model.forward_train(batch)

    assert outputs["loss"].ndim == 0
    assert torch.isfinite(outputs["loss"])


def test_tiny_flux_model_computes_raw_loss_with_dummy_encoders():
    class DummyAutoencoder(torch.nn.Module):
        def encode(self, image):
            return image[:, :1, :2, :2]

    class DummyT5(torch.nn.Module):
        def forward(self, prompts):
            return torch.zeros(len(prompts), 3, 8)

    class DummyClip(torch.nn.Module):
        def forward(self, prompts):
            return torch.zeros(len(prompts), 4)

    model = build_flux_model(
        {
            "config": {
                "model_variant": "flux-dev",
                "guidance": 1.0,
                "params": {
                    "in_channels": 4,
                    "out_channels": 4,
                    "vec_in_dim": 4,
                    "context_in_dim": 8,
                    "hidden_size": 12,
                    "num_heads": 2,
                    "depth": 1,
                    "depth_single_blocks": 1,
                    "axes_dim": [2, 2, 2],
                },
            }
        }
    )
    model.autoencoder = DummyAutoencoder()
    model.t5_encoder = DummyT5()
    model.clip_encoder = DummyClip()

    outputs = model.forward_train(
        {
            "image": torch.randn(2, 3, 4, 4),
            "prompts": ["cat", "dog"],
        }
    )

    assert outputs["loss"].ndim == 0
    assert torch.isfinite(outputs["loss"])
