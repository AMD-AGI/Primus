import json
from types import SimpleNamespace

import pytest
import torch.nn as nn

from primus.backends.megatron.data.diffusion.task_encoders.image import (
    _sample_key_fingerprint,
)
from primus.backends.megatron.flux_pretrain_trainer import (
    FluxPretrainTrainer,
    _emit_precision_linear_class_census,
    _mxfp4_gemm_mode_census,
    _precision_linear_class_census,
)
from primus.backends.megatron.patches.mlperf_warmup_patches import _is_resumed_training
from primus.backends.megatron.training.diffusion.forward_step import (
    _emit_batch_fingerprint,
)

_FORWARD_STEP_LOGGER = "primus.backends.megatron.training.diffusion.forward_step"
_FLUX_TRAINER_LOGGER = "primus.backends.megatron.flux_pretrain_trainer"


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


def test_mxfp4_gemm_mode_census_reports_forward_backward_split():
    mxfp4_column = type("MXFP4ColumnParallelLinear", (nn.Module,), {})()
    mxfp4_column._forward_precision = "fp8"
    mxfp4_column._backward_is_fp8 = False
    mxfp4_row = type("MXFP4RowParallelLinear", (nn.Module,), {})()
    mxfp4_row._forward_precision = "bf16"
    mxfp4_row._backward_is_fp8 = False
    model = nn.ModuleList([mxfp4_column, mxfp4_row, nn.Linear(2, 2)])

    assert _mxfp4_gemm_mode_census(model) == {
        "bf16_forward_mxfp4_backward": 1,
        "fp8_forward_mxfp4_backward": 1,
    }


def test_precision_linear_class_census_emits_zero_counts_for_bf16(monkeypatch, caplog):
    from megatron.core import parallel_state

    monkeypatch.setenv("PRIMUS_AUDIT_LINEAR_CLASS_CENSUS", "1")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", lambda: 3)
    caplog.set_level("INFO", logger=_FLUX_TRAINER_LOGGER)

    _emit_precision_linear_class_census(nn.Linear(2, 2))

    line = next(
        record.message
        for record in caplog.records
        if record.message.startswith("PRIMUS_LINEAR_CLASS_CENSUS=")
    )
    payload = json.loads(line.split("=", 1)[1])
    assert payload == {
        "data_parallel_rank": 3,
        "global_rank": 3,
        "classes": {
            "MXFP4ColumnParallelLinear": 0,
            "MXFP4RowParallelLinear": 0,
            "Float8ColumnParallelLinear": 0,
            "Float8RowParallelLinear": 0,
        },
    }


def test_emit_batch_fingerprint_is_fail_closed(monkeypatch):
    monkeypatch.setenv("PRIMUS_AUDIT_BATCH_FINGERPRINTS", "1")
    with pytest.raises(RuntimeError, match="no valid sample-key fingerprint"):
        _emit_batch_fingerprint({}, step_count=1)


def test_emit_batch_fingerprint_skips_synthetic_warmup(monkeypatch, caplog):
    monkeypatch.setenv("PRIMUS_AUDIT_BATCH_FINGERPRINTS", "1")
    monkeypatch.setenv("PRIMUS_SYNTHETIC_WARMUP_ACTIVE", "1")
    caplog.set_level("INFO", logger=_FORWARD_STEP_LOGGER)

    _emit_batch_fingerprint({}, step_count=1)

    assert not any(record.message.startswith("PRIMUS_BATCH_FINGERPRINT=") for record in caplog.records)


def test_emit_batch_fingerprint_skips_validation(monkeypatch, caplog):
    monkeypatch.setenv("PRIMUS_AUDIT_BATCH_FINGERPRINTS", "1")
    caplog.set_level("INFO", logger=_FORWARD_STEP_LOGGER)

    _emit_batch_fingerprint({}, step_count=5, is_training=False)

    assert not any(record.message.startswith("PRIMUS_BATCH_FINGERPRINT=") for record in caplog.records)


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


def test_emit_batch_fingerprint_logs_rank_local_payload(monkeypatch, caplog):
    from megatron.core import parallel_state

    monkeypatch.setenv("PRIMUS_AUDIT_BATCH_FINGERPRINTS", "1")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", lambda: 3)
    caplog.set_level("INFO", logger=_FORWARD_STEP_LOGGER)
    _emit_batch_fingerprint(
        {
            "_audit_sample_key_sha256": "a" * 64,
            "_audit_sample_count": 64,
        },
        step_count=6,
    )

    marker_lines = [
        record.message
        for record in caplog.records
        if record.name == _FORWARD_STEP_LOGGER and record.message.startswith("PRIMUS_BATCH_FINGERPRINT=")
    ]
    assert len(marker_lines) == 1
    payload = json.loads(marker_lines[0].split("=", 1)[1])
    assert payload == {
        "data_parallel_rank": 3,
        "global_rank": 3,
        "sample_count": 64,
        "sample_keys_sha256": "a" * 64,
        "step": 6,
    }
