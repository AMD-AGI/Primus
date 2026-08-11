import json
from types import SimpleNamespace

import pytest
import torch.nn as nn

from primus.backends.megatron.data.diffusion.task_encoders.image import (
    _sample_key_fingerprint,
)
from primus.backends.megatron.flux_pretrain_trainer import (
    FluxPretrainTrainer,
    _precision_linear_class_census,
)
from primus.backends.megatron.patches.mlperf_warmup_patches import _is_resumed_training
from primus.backends.megatron.training.diffusion.forward_step import (
    _emit_batch_fingerprint,
)


def test_sample_key_fingerprint_is_order_sensitive():
    first = [
        SimpleNamespace(**{"__key__": "sample-a"}),
        SimpleNamespace(**{"__key__": "sample-b"}),
    ]
    second = list(reversed(first))

    assert _sample_key_fingerprint(first) != _sample_key_fingerprint(second)
    assert _sample_key_fingerprint(first) == _sample_key_fingerprint(first)


def test_sample_key_fingerprint_requires_energon_key():
    with pytest.raises(RuntimeError, match="sample has no __key__"):
        _sample_key_fingerprint([SimpleNamespace()])


def test_precision_linear_class_census_reports_exact_classes():
    mxfp4_column = type("MXFP4ColumnParallelLinear", (nn.Module,), {})()
    mxfp4_row = type("MXFP4RowParallelLinear", (nn.Module,), {})()
    float8_column = type("Float8ColumnParallelLinear", (nn.Module,), {})()
    model = nn.ModuleList([mxfp4_column, mxfp4_row, float8_column])

    assert _precision_linear_class_census(model) == {
        "MXFP4ColumnParallelLinear": 1,
        "MXFP4RowParallelLinear": 1,
        "Float8ColumnParallelLinear": 1,
        "Float8RowParallelLinear": 0,
    }


def test_emit_batch_fingerprint_is_fail_closed(monkeypatch):
    monkeypatch.setenv("PRIMUS_AUDIT_BATCH_FINGERPRINTS", "1")
    with pytest.raises(RuntimeError, match="no valid sample-key fingerprint"):
        _emit_batch_fingerprint({}, step_count=1)


def test_emit_batch_fingerprint_skips_synthetic_warmup(monkeypatch, capsys):
    monkeypatch.setenv("PRIMUS_AUDIT_BATCH_FINGERPRINTS", "1")
    monkeypatch.setenv("PRIMUS_SYNTHETIC_WARMUP_ACTIVE", "1")

    _emit_batch_fingerprint({}, step_count=1)

    assert capsys.readouterr().out == ""


def test_emit_batch_fingerprint_skips_validation(monkeypatch, capsys):
    monkeypatch.setenv("PRIMUS_AUDIT_BATCH_FINGERPRINTS", "1")

    _emit_batch_fingerprint({}, step_count=5, is_training=False)

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (0, False),
        (5, True),
        (None, False),
        (True, False),
    ],
)
def test_warmup_resume_detection(iteration, expected):
    assert _is_resumed_training(SimpleNamespace(iteration=iteration)) is expected


def test_diffusion_forward_step_exposes_counter_reset(monkeypatch):
    from megatron.core import num_microbatches_calculator

    trainer = FluxPretrainTrainer.__new__(FluxPretrainTrainer)
    trainer._forward_step_count = 2
    trainer._forward_step_count_initialized = True
    monkeypatch.setattr(
        num_microbatches_calculator,
        "get_num_microbatches",
        lambda: 1,
    )

    forward_step = trainer.get_forward_step()
    forward_step._primus_reset_forward_step_count(0)

    assert trainer._forward_step_count == 0
    assert trainer._forward_step_count_initialized is True


def test_emit_batch_fingerprint_logs_rank_local_payload(monkeypatch, capsys):
    from megatron.core import parallel_state

    monkeypatch.setenv("PRIMUS_AUDIT_BATCH_FINGERPRINTS", "1")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", lambda: 3)
    _emit_batch_fingerprint(
        {
            "_audit_sample_key_sha256": "a" * 64,
            "_audit_sample_count": 64,
        },
        step_count=6,
    )

    line = capsys.readouterr().out.strip()
    payload = json.loads(line.split("=", 1)[1])
    assert payload == {
        "data_parallel_rank": 3,
        "global_rank": 3,
        "sample_count": 64,
        "sample_keys_sha256": "a" * 64,
        "step": 6,
    }
