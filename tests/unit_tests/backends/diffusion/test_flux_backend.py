from __future__ import annotations

import numpy as np
import pytest
import torch

from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.attention import (
    get_attention_backend,
    set_attention_backend,
)
from primus.backends.diffusion.data.flux_precomputed import (
    FluxPrecomputedProcessor,
    FluxRawImageTextDataset,
    FluxRawImageTextProcessor,
)
from primus.backends.diffusion.models.flux.adapter import FluxForTraining
from primus.backends.diffusion.models.flux.conditioner import HFEmbedder
from primus.backends.diffusion.models.flux.math import apply_rope
from primus.backends.diffusion.models.flux.math import attention as flux_attention
from primus.backends.diffusion.models.flux.math import rope
from primus.backends.diffusion.models.registrations.flux import build_flux_model
from primus.backends.diffusion.trainers.fsdp2 import FSDP2Trainer


def _finalize(params: dict):
    builder = DiffusionArgBuilder()
    builder.update(params)
    return builder.finalize()


def test_flux_argument_builder_selects_flux_defaults():
    args = _finalize(
        {
            "model": {"name": "flux.1-dev", "config": {}},
            "training": {"steps": 7, "local_batch_size": 3},
            "data": {
                "dataset_path": "/tmp/precomputed",
                "dataset_type": "precomputed",
                "empty_encodings_path": "/tmp/empty",
                "prompt_dropout_prob": 0.25,
            },
            "lr_scheduler": {
                "lr_scheduler_type": "constant_with_warmup",
                "warmup_steps": 11,
            },
        }
    )

    assert args.model["name"] == "flux.1-dev"
    assert args.dataset["name"] == "flux"
    assert args.dataset["config"]["dataset_path"] == "/tmp/precomputed"
    assert args.dataset["config"]["processor_config"]["empty_encodings_path"] == "/tmp/empty"
    assert args.dataset["config"]["processor_config"]["prompt_dropout_prob"] == 0.25
    assert args.trainer["args"]["max_steps"] == 7
    assert args.trainer["args"]["per_device_train_batch_size"] == 3
    assert args.trainer["args"]["lr_scheduler_type"] == "constant_with_warmup"
    assert args.trainer["args"]["warmup_steps"] == 11
    assert args.trainer["args"]["attention_backend"] == "flash_attn_aiter"
    assert args.trainer["args"]["fsdp_transformer_layer_cls_to_wrap"] == "DoubleStreamBlock,SingleStreamBlock"
    assert args.trainer["args"]["compile_transformer_blocks"] is True


def test_flux_argument_builder_maps_raw_dataset_type():
    args = _finalize(
        {
            "model": {"name": "flux.1-dev", "config": {}},
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
    assert path == "zirui3/cc12m-test"
    assert fmt == "hf_repo"

    path, fmt = FluxRawImageTextDataset._resolve_dataset("cc12m-test", "/tmp/cc12m_test", "webdataset")
    assert path == "/tmp/cc12m_test"
    assert fmt == "webdataset"

    path, fmt = FluxRawImageTextDataset._resolve_dataset("cc12m-wds", None, "webdataset")
    assert path == "pixparse/cc12m-wds"
    assert fmt == "hf_repo"


def test_flux_argument_builder_rejects_sequence_parallelism():
    with pytest.raises(ValueError, match="sp_size"):
        _finalize(
            {
                "model": {"name": "flux.1-dev", "config": {}},
                "parallelism": {"sp_size": 2},
            }
        )


def test_flux_attention_dispatch_matches_sdpa_layout():
    previous_backend = get_attention_backend()
    set_attention_backend("sdpa")
    try:
        torch.manual_seed(7)
        q = torch.randn(2, 3, 5, 4, dtype=torch.bfloat16)
        k = torch.randn(2, 3, 5, 4, dtype=torch.bfloat16)
        v = torch.randn(2, 3, 5, 4, dtype=torch.bfloat16)
        pos = torch.arange(5, dtype=torch.float32).repeat(2, 1)
        pe = rope(pos, dim=4, theta=10000).unsqueeze(1)

        actual = flux_attention(q, k, v, pe=pe)
        q_rope, k_rope = apply_rope(q, k, pe)
        expected = torch.nn.functional.scaled_dot_product_attention(q_rope, k_rope, v)
        expected = expected.transpose(1, 2).flatten(2)

        torch.testing.assert_close(actual, expected)
    finally:
        set_attention_backend(previous_backend)


def test_flux_hf_embedder_passes_attention_mask():
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs["padding"] == "max_length"
            return {
                "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long),
            }

    class FakeTextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.seen_attention_mask = None

        def forward(self, *, input_ids, attention_mask, output_hidden_states):
            self.seen_attention_mask = attention_mask
            return {"last_hidden_state": input_ids.float().unsqueeze(-1)}

    embedder = HFEmbedder.__new__(HFEmbedder)
    torch.nn.Module.__init__(embedder)
    embedder.tokenizer = FakeTokenizer()
    embedder.hf_module = FakeTextModel()
    embedder.output_key = "last_hidden_state"
    embedder.max_length = 3

    output = embedder(["short", "x"])

    torch.testing.assert_close(
        embedder.hf_module.seen_attention_mask,
        torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long),
    )
    assert output.shape == (2, 3, 1)


def test_flux_forward_uses_positional_scheduler():
    model = FluxForTraining(
        dit=torch.nn.Identity(),
        train_pipeline=object(),
        model_config=object(),
    )
    scheduler = object()
    captured = {}

    def forward_train(batch, scheduler=None):
        captured["batch"] = batch
        captured["scheduler"] = scheduler
        return {"loss": torch.tensor(0.0)}

    model.forward_train = forward_train
    batch = {"x": torch.tensor(1)}

    output = model(batch, scheduler)

    assert output["loss"].item() == 0.0
    assert captured == {"batch": batch, "scheduler": scheduler}


def test_fsdp2_compile_transformer_blocks_replaces_modules(monkeypatch):
    class CompiledBlock(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original

    root = torch.nn.Module()
    root.double_blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.ReLU()])
    root.single_blocks = torch.nn.ModuleList([torch.nn.Sigmoid()])
    compiled_inputs = []

    def fake_compile(module, *, fullgraph):
        assert fullgraph is True
        compiled_inputs.append(module)
        return CompiledBlock(module)

    monkeypatch.setattr(torch, "compile", fake_compile)
    trainer = FSDP2Trainer.__new__(FSDP2Trainer)
    trainer.rank = 1

    trainer._compile_transformer_blocks(root)

    assert len(compiled_inputs) == 3
    assert all(isinstance(block, CompiledBlock) for block in root.double_blocks)
    assert all(isinstance(block, CompiledBlock) for block in root.single_blocks)


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


def test_flux_precomputed_processor_rejects_mismatched_empty_encoding(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    # Empty T5 encoding has sequence length 5, but the batch samples use length 3.
    np.save(empty_dir / "t5_empty.npy", np.zeros((1, 5, 8), dtype=np.float32))
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
        }
    ]

    with pytest.raises(ValueError, match="empty T5 encoding shape"):
        processor.prepare_batch(batch=batch, device=torch.device("cpu"), dtype=torch.float32)


def test_flux_raw_processor_prepares_images_and_prompts():
    from PIL import Image

    processor = FluxRawImageTextProcessor(
        {"img_size": 8, "prompt_dropout_prob": 1.0, "skip_low_resolution": False}
    )
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
                "model_preset": "flux.1-dev",
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


def test_tiny_flux_schnell_model_computes_without_guidance():
    model = build_flux_model(
        {
            "config": {
                "model_preset": "flux.1-schnell",
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

    assert model.dit.params.guidance_embed is False
    assert model.model_config.guidance is None
    assert outputs["loss"].ndim == 0
    assert torch.isfinite(outputs["loss"])


def test_flux_position_ids_are_float32_regardless_of_model_dtype():
    model = build_flux_model(
        {
            "config": {
                "model_preset": "flux.1-dev",
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
    # Force a low-precision compute dtype; position ids must stay float32 so RoPE
    # grid indices are not corrupted (bf16 only represents integers up to 256).
    model.dit = model.dit.to(dtype=torch.bfloat16)

    captured = {}
    original_forward = model.dit.forward

    def capturing_forward(*args, **kwargs):
        captured["img_ids_dtype"] = kwargs["img_ids"].dtype
        captured["txt_ids_dtype"] = kwargs["txt_ids"].dtype
        return original_forward(*args, **kwargs)

    model.dit.forward = capturing_forward
    model.forward_train(
        {
            "t5_encodings": torch.randn(2, 3, 8, dtype=torch.bfloat16),
            "clip_encodings": torch.randn(2, 4, dtype=torch.bfloat16),
            "mean": torch.randn(2, 1, 2, 2, dtype=torch.bfloat16),
            "logvar": torch.zeros(2, 1, 2, 2, dtype=torch.bfloat16),
        }
    )

    assert captured["img_ids_dtype"] == torch.float32
    assert captured["txt_ids_dtype"] == torch.float32


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
                "model_preset": "flux.1-dev",
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
