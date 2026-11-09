###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus.backends.diffusion.trainers.base import BaseWanTrainer


def _make_trainer():
    trainer = BaseWanTrainer.__new__(BaseWanTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.rank = 0
    trainer.args = {
        "learning_rate": 1.0e-4,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
    }
    return trainer


def test_optimizer_falls_back_to_foreach_when_fused_raises_runtime_error(monkeypatch):
    calls = []

    def fake_adamw(**kwargs):
        calls.append(kwargs)
        if kwargs.get("fused"):
            raise RuntimeError("fused AdamW is unsupported")
        return kwargs

    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)

    optimizer = _make_trainer()._create_optimizer()

    assert calls[0]["fused"] is True
    assert calls[1]["foreach"] is True
    assert optimizer is calls[1]


def test_optimizer_falls_back_to_default_when_foreach_raises_runtime_error(monkeypatch):
    calls = []

    def fake_adamw(**kwargs):
        calls.append(kwargs)
        if kwargs.get("fused") or kwargs.get("foreach"):
            raise RuntimeError("optimized AdamW path is unsupported")
        return kwargs

    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)

    optimizer = _make_trainer()._create_optimizer()

    assert calls[0]["fused"] is True
    assert calls[1]["foreach"] is True
    assert "fused" not in calls[2]
    assert "foreach" not in calls[2]
    assert optimizer is calls[2]


@pytest.mark.parametrize(
    "mode,recipe,offset",
    [("performance_only", "0", 0), ("nemo_mlperf", "0", 1), ("performance_only", "1", 1)],
)
def test_warmup_offset_is_scoped_to_flux_benchmarks(monkeypatch, tmp_path, mode, recipe, offset):
    monkeypatch.setenv("FLUX_A6W4_RECIPE", recipe)
    monkeypatch.delenv("FLUX_FPROP_SWITCH_STEP", raising=False)
    monkeypatch.setattr(BaseWanTrainer, "_apply_parallelism", lambda self: None)
    monkeypatch.setattr(BaseWanTrainer, "_setup_mlperf", lambda self: None)
    dataset = torch.utils.data.TensorDataset(torch.zeros(8, 2))
    dataset.get_collator = lambda: None
    trainer = BaseWanTrainer(
        model=torch.nn.Linear(2, 2),
        args={
            "output_dir": str(tmp_path),
            "learning_rate": 1.0,
            "max_steps": 10,
            "lr_scheduler_type": "constant_with_warmup",
            "warmup_steps": 5,
            "dataloader_num_workers": 0,
            "performance_mode": mode,
            "mlperf_train_samples": 8,
            "mlperf_eval_total_samples": 8,
        },
        train_dataset=dataset,
        data_collator=None,
        processing_class=None,
        rank=0,
        world_size=1,
        local_rank=0,
        eval_dataset=dataset,
    )
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(offset / 5)
    assert [trainer.lr_scheduler.lr_lambdas[0](step) for step in range(6)] == pytest.approx(
        [min(1.0, (step + offset) / 5) for step in range(6)]
    )
