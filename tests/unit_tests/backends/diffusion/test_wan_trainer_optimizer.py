###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

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
