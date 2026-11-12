###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from pathlib import Path

import pytest

from primus.backends.vision.argument_builder import VisionArgBuilder
from primus.backends.vision.registry import (
    DATASET_BUILDERS,
    MODEL_BUILDERS,
    TRAINER_BUILDERS,
    get_model_builder,
)
from primus.backends.vision.trainers.schedule import lr_factor
from primus.backends.vision.vision_adapter import VisionAdapter


def _build(params):
    builder = VisionArgBuilder()
    builder.update(params)
    return builder.finalize()


def test_vggt_preset_uses_its_learning_rate_and_dataset():
    args = _build({"model": {"name": "vggt", "config": {}}})
    assert args.model["name"] == "vggt"
    assert args.dataset["name"] == "vggt"
    assert args.trainer["name"] == "fsdp2"
    assert args.trainer["args"]["learning_rate"] == 5.0e-5
    assert args.trainer["args"]["parallelism_mode"] == "ddp"
    assert args.dataset["config"]["dataset_path"].endswith("/co3d")


def test_sections_override_trainer_and_data():
    args = _build(
        {
            "model": {"name": "synthetic", "config": {"embed_dim": 8}},
            "training": {"steps": 3, "output_dir": "/tmp/vision-out", "local_batch_size": 2},
            "parallelism": {"mode": "fsdp2"},
            "runtime": {"bf16": "false", "attention_backend": "sdpa"},
            "metrics": {"log_freq": 4},
        }
    )
    assert args.trainer["args"]["max_steps"] == 3
    assert args.trainer["args"]["output_dir"] == "/tmp/vision-out"
    assert args.trainer["args"]["parallelism_mode"] == "fsdp2"
    assert args.trainer["args"]["bf16"] is False
    assert args.trainer["args"]["logging_steps"] == 4
    assert args.dataset["config"]["batch_size"] == 2


def test_public_dataset_section_is_rejected():
    with pytest.raises(ValueError, match="data"):
        _build({"model": {"name": "vggt", "config": {}}, "dataset": {"name": "vggt"}})


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported vision model"):
        _build({"model": {"name": "not-a-model", "config": {}}})


def test_worldmirror_stage2_requires_a_checkpoint():
    with pytest.raises(ValueError, match="stage 2"):
        _build({"model": {"name": "worldmirror", "config": {"enable_gs": True, "pretrained": ""}}})
    args = _build(
        {"model": {"name": "worldmirror", "config": {"enable_gs": True, "pretrained": "/ckpt/stage1.pt"}}}
    )
    assert args.dataset["name"] == "worldmirror"
    assert args.model["config"]["pretrained"] == "/ckpt/stage1.pt"


def test_registry_names_match_the_builders():
    assert set(MODEL_BUILDERS) == {"vggt", "worldmirror", "synthetic"}
    assert set(DATASET_BUILDERS) == {"vggt", "worldmirror", "synthetic"}
    assert set(TRAINER_BUILDERS) == {"fsdp2"}
    with pytest.raises(KeyError, match="Unknown vision model"):
        get_model_builder("missing")


def test_adapter_reports_the_in_tree_package():
    adapter = VisionAdapter()
    assert adapter.detect_backend_version() == "in-tree"
    with pytest.raises(ValueError):
        adapter.load_trainer_class("rl")
    pytest.importorskip("torch")
    assert adapter.load_trainer_class("pretrain").__name__ == "VisionPretrainTrainer"
    assert adapter.load_trainer_class("posttrain").__name__ == "VisionPretrainTrainer"


def test_cosine_schedule_warms_up_and_ends_at_the_floor():
    assert lr_factor(0, scheduler_type="cosine", warmup_steps=10, total_steps=100, eta_min_ratio=0.1) == 0.1
    assert lr_factor(9, scheduler_type="cosine", warmup_steps=10, total_steps=100, eta_min_ratio=0.1) == 1.0
    end = lr_factor(100, scheduler_type="cosine", warmup_steps=10, total_steps=100, eta_min_ratio=0.1)
    assert end == pytest.approx(0.1)
    warmed = lr_factor(
        20,
        scheduler_type="constant_with_warmup",
        warmup_steps=10,
        total_steps=100,
        eta_min_ratio=0.1,
    )
    assert warmed == 1.0
    constant = lr_factor(5, scheduler_type="constant", warmup_steps=10, total_steps=100, eta_min_ratio=0.1)
    assert constant == 1.0


def test_vision_package_does_not_import_a_checkout():
    root = Path(__file__).resolve().parents[4] / "primus" / "backends" / "vision"
    owned = [
        path
        for path in root.rglob("*.py")
        if not path.relative_to(root).as_posix().startswith(("vggt_model/", "worldmirror/"))
    ]
    for path in owned:
        text = path.read_text()
        assert "third_party" not in text, path
        assert "sys.path" not in text, path
        assert "resolve_vggt_repo" not in text, path
    vggt = (root / "vggt_model" / "models" / "vggt.py").read_text()
    assert "from primus.backends.vision.vggt_model.models.aggregator import Aggregator" in vggt
    worldmirror = (root / "worldmirror" / "src" / "models" / "models" / "worldmirror.py").read_text()
    assert (
        "from primus.backends.vision.worldmirror.src.models.models.visual_transformer import"
        in worldmirror
    )
    assert "GaussianSplatRenderer" not in worldmirror.split("if self.enable_gs:", 1)[0]


def test_synthetic_model_trains_one_step(tmp_path: Path):
    torch = pytest.importorskip("torch")
    model = get_model_builder("synthetic")({"embed_dim": 8})
    from primus.backends.vision.registry import get_dataset_builder, get_trainer_builder

    dataset, processor = get_dataset_builder("synthetic")(
        {"length": 2, "views": 2, "image_size": 8, "batch_size": 1, "num_workers": 0}
    )
    before = [param.detach().clone() for param in model.parameters()]
    trainer = get_trainer_builder("fsdp2")(
        model=model,
        dataset=dataset,
        processor=processor,
        trainer_args={
            "max_steps": 1,
            "num_train_epochs": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-2,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1.0e-8,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "constant",
            "warmup_steps": 0,
            "eta_min_ratio": 0.1,
            "bf16": False,
            "gradient_checkpointing": False,
            "compile_transformer_blocks": False,
            "logging_steps": 1,
            "save_steps": 0,
            "save_total_limit": 1,
            "seed": 0,
            "output_dir": str(tmp_path),
            "parallelism_mode": "ddp",
            "profile": False,
        },
    )
    trainer.train()
    assert trainer.steps == 1
    assert trainer.skipped_steps == 0
    assert any(not torch.equal(old, param) for old, param in zip(before, model.parameters()))
    assert (tmp_path / "last.pt").is_file()
