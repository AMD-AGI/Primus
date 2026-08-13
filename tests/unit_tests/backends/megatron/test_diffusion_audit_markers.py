import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
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
    _EMITTED_MODEL_WEIGHT_ITERATIONS,
    _emit_batch_fingerprint,
    _emit_model_weight_summary,
    _emit_model_weight_summary_after_forward,
    _encode_strict_json,
    _is_validation_forward,
    _model_weight_audit_context,
    _model_weight_iteration_coordinate,
    _parse_model_weight_steps,
    _pregenerated_diffusion_inputs,
    _read_strict_json,
    _sample_model_weights,
    _write_model_weight_summary_once,
)

_FORWARD_STEP_LOGGER = "primus.backends.megatron.training.diffusion.forward_step"
_FLUX_TRAINER_LOGGER = "primus.backends.megatron.flux_pretrain_trainer"
_UNSET = object()


@pytest.fixture(autouse=True)
def _clear_emitted_model_weight_iterations():
    _EMITTED_MODEL_WEIGHT_ITERATIONS.clear()
    yield
    _EMITTED_MODEL_WEIGHT_ITERATIONS.clear()


def _set_training_state(
    monkeypatch,
    *,
    iteration,
    train_iters=20_000,
    curr_iteration=_UNSET,
    num_microbatches=1,
):
    from megatron import training as megatron_training
    from megatron.core import num_microbatches_calculator

    args = SimpleNamespace(iteration=iteration, train_iters=train_iters)
    if curr_iteration is not _UNSET:
        args.curr_iteration = curr_iteration
    microbatch_state = {"value": num_microbatches}
    monkeypatch.setattr(megatron_training, "get_args", lambda: args)
    monkeypatch.setattr(
        num_microbatches_calculator,
        "get_num_microbatches",
        lambda: microbatch_state["value"],
    )
    return args, microbatch_state


def _enable_weight_audit(monkeypatch, output, *, steps, sample_size=3, rank=0):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.delenv("PRIMUS_SYNTHETIC_WARMUP_ACTIVE", raising=False)
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("PRIMUS_AUDIT_MODEL_WEIGHT_STEPS", steps)
    monkeypatch.setenv("PRIMUS_AUDIT_MODEL_WEIGHT_SAMPLE_SIZE", str(sample_size))
    monkeypatch.setenv("PRIMUS_AUDIT_MODEL_WEIGHT_PATH", str(output))


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

    marker_lines = [
        record.message
        for record in caplog.records
        if record.name == _FLUX_TRAINER_LOGGER and record.message.startswith("PRIMUS_LINEAR_CLASS_CENSUS=")
    ]
    assert len(marker_lines) == 1
    payload = json.loads(marker_lines[0].split("=", 1)[1])
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


def test_precision_linear_class_census_emits_mxfp4_gemm_modes(monkeypatch, caplog):
    from megatron.core import parallel_state

    monkeypatch.setenv("PRIMUS_AUDIT_LINEAR_CLASS_CENSUS", "1")
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", lambda: 5)
    caplog.set_level("INFO", logger=_FLUX_TRAINER_LOGGER)

    mxfp4_column = type("MXFP4ColumnParallelLinear", (nn.Module,), {})()
    mxfp4_column._forward_precision = "fp8"
    mxfp4_column._backward_is_fp8 = False
    _emit_precision_linear_class_census(nn.ModuleList([mxfp4_column]))

    line = next(
        record.message
        for record in caplog.records
        if record.message.startswith("PRIMUS_MXFP4_GEMM_MODE_CENSUS=")
    )
    assert json.loads(line.split("=", 1)[1]) == {
        "data_parallel_rank": 5,
        "global_rank": 5,
        "modes": {"fp8_forward_mxfp4_backward": 1},
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


def test_parse_model_weight_steps_is_fail_closed():
    assert _parse_model_weight_steps("5120, 8192") == {5120, 8192}
    with pytest.raises(RuntimeError, match="duplicate step 5120"):
        _parse_model_weight_steps("5120,5120")
    with pytest.raises(RuntimeError, match="positive integers"):
        _parse_model_weight_steps("5120,not-a-step")
    for invalid in ("", "0", "-1", "1,"):
        with pytest.raises(RuntimeError):
            _parse_model_weight_steps(invalid)


def test_sample_model_weights_is_deterministic_and_sensitive():
    model = nn.Sequential(nn.Linear(4, 3, bias=True), nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(
                torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter) + index
            )

    first = _sample_model_weights(model, sample_size=4)
    second = _sample_model_weights(model, sample_size=4)
    assert first == second
    assert first["sample_finite"]
    assert first["parameter_count"] == 3

    with torch.no_grad():
        model[0].weight.flatten()[0].add_(1)
    changed = _sample_model_weights(model, sample_size=4)
    first_by_name = {item["name"]: item for item in first["parameters"]}
    changed_by_name = {item["name"]: item for item in changed["parameters"]}
    assert changed_by_name["0.weight"]["sample_sha256"] != first_by_name["0.weight"]["sample_sha256"]


def test_sample_model_weights_serializes_nonfinite_values_as_strict_json():
    model = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.flatten()[0] = torch.nan
        model.weight.flatten()[2] = torch.inf

    summary = _sample_model_weights(model, sample_size=4)

    assert summary["sample_finite"] is False
    assert summary["sample_nonfinite_count"] == 2
    assert summary["sample_sum"] is None
    assert summary["sample_sum_squares"] is None
    assert summary["sample_absmax"] is None
    json.dumps(summary, allow_nan=False)


def test_model_weight_iteration_coordinate_uses_canonical_megatron_iteration(monkeypatch):
    args, microbatches = _set_training_state(
        monkeypatch,
        iteration=5120,
        num_microbatches=4,
    )

    assert _model_weight_iteration_coordinate() == (5120, 5121, 4, 20_000)

    microbatches["value"] = 8
    assert _model_weight_iteration_coordinate() == (5120, 5121, 8, 20_000)

    # The pinned Megatron loop keeps args.iteration at the restored checkpoint
    # and advances args.curr_iteration immediately before each train_step.
    args.curr_iteration = 8192
    assert _model_weight_iteration_coordinate() == (8192, 8193, 8, 20_000)


@pytest.mark.parametrize("iteration", [None, True, -1, 1.5])
def test_model_weight_iteration_coordinate_rejects_invalid_restored_iteration(monkeypatch, iteration):
    _set_training_state(monkeypatch, iteration=iteration)

    with pytest.raises(RuntimeError, match="args.iteration"):
        _model_weight_iteration_coordinate()


@pytest.mark.parametrize("num_microbatches", [None, True, 0, -1, 1.5])
def test_model_weight_iteration_coordinate_rejects_invalid_microbatch_metadata(monkeypatch, num_microbatches):
    _set_training_state(
        monkeypatch,
        iteration=0,
        num_microbatches=num_microbatches,
    )

    with pytest.raises(RuntimeError, match="positive integer"):
        _model_weight_iteration_coordinate()


@pytest.mark.parametrize("curr_iteration", [None, True, -1, 1.5])
def test_model_weight_iteration_coordinate_rejects_invalid_active_iteration(monkeypatch, curr_iteration):
    _set_training_state(
        monkeypatch,
        iteration=0,
        curr_iteration=curr_iteration,
    )

    with pytest.raises(RuntimeError, match="args.curr_iteration"):
        _model_weight_iteration_coordinate()


def test_emit_model_weight_summary_uses_resume_coordinate_and_suppresses_replays(
    monkeypatch, tmp_path, caplog
):
    output = tmp_path / "weights"
    model = nn.Linear(4, 2)
    _, microbatches = _set_training_state(
        monkeypatch,
        iteration=5120,
        num_microbatches=4,
    )
    _enable_weight_audit(monkeypatch, output, steps="5120,8192")
    caplog.set_level("INFO", logger=_FORWARD_STEP_LOGGER)

    # step_count is deliberately unrelated to the training-loop coordinate.
    _emit_model_weight_summary(model, step_count=999_999)

    record_path = output / "completed_iteration_0005120.json"
    record = json.loads(record_path.read_text())
    assert record["completed_training_iteration"] == 5120
    assert record["next_training_iteration"] == 5121
    assert record["forward_step_count"] == 999_999
    assert record["microbatch_index"] == 0
    assert record["num_microbatches"] == 4
    assert record["sample_size_per_parameter"] == 3
    assert record["sample_finite"]
    assert any(record.message.startswith("PRIMUS_MODEL_WEIGHT_SUMMARY=") for record in caplog.records)

    original = record_path.read_bytes()
    microbatches["value"] = 8
    _emit_model_weight_summary(model, step_count=1)
    assert record_path.read_bytes() == original
    assert list(output.glob("*.json")) == [record_path]
    assert not list(output.glob("*.tmp"))


def test_emit_model_weight_summary_uses_active_iteration_after_resume(monkeypatch, tmp_path):
    output = tmp_path / "weights"
    model = nn.Linear(2, 2)
    _set_training_state(
        monkeypatch,
        iteration=5120,
        curr_iteration=8192,
        num_microbatches=4,
    )
    _enable_weight_audit(monkeypatch, output, steps="8192")

    _emit_model_weight_summary(model, step_count=13)

    record = json.loads((output / "completed_iteration_0008192.json").read_text())
    assert record["completed_training_iteration"] == 8192
    assert record["next_training_iteration"] == 8193
    assert record["forward_step_count"] == 13


def test_emit_model_weight_summary_rejects_terminal_requested_iteration(monkeypatch, tmp_path):
    output = tmp_path / "weights"
    model = nn.Linear(2, 2)
    _set_training_state(
        monkeypatch,
        iteration=5120,
        train_iters=8192,
    )
    _enable_weight_audit(monkeypatch, output, steps="5120,8192")

    with pytest.raises(RuntimeError, match="iteration 8192 requires training through at least 8193"):
        _emit_model_weight_summary(model, step_count=5121)
    assert not output.exists()


@pytest.mark.parametrize("train_iters", [None, True, 0, -1, 1.5])
def test_emit_model_weight_summary_rejects_invalid_train_iters(monkeypatch, tmp_path, train_iters):
    output = tmp_path / "weights"
    _set_training_state(
        monkeypatch,
        iteration=0,
        train_iters=train_iters,
    )
    _enable_weight_audit(monkeypatch, output, steps="1")

    with pytest.raises(RuntimeError, match="args.train_iters"):
        _emit_model_weight_summary(nn.Linear(2, 2), step_count=1)
    assert not output.exists()


def test_emit_model_weight_summary_restart_allows_changed_replay_provenance(monkeypatch, tmp_path):
    output = tmp_path / "weights"
    model = nn.Linear(2, 2)
    _, microbatches = _set_training_state(monkeypatch, iteration=5120)
    _enable_weight_audit(monkeypatch, output, steps="5120")

    _emit_model_weight_summary(model, step_count=5121)
    _EMITTED_MODEL_WEIGHT_ITERATIONS.clear()
    original = (output / "completed_iteration_0005120.json").read_bytes()
    microbatches["value"] = 4
    _emit_model_weight_summary(model, step_count=7)

    assert (output / "completed_iteration_0005120.json").read_bytes() == original
    assert not list(output.glob("*.tmp"))


def test_emit_model_weight_summary_rejects_conflicting_restart(monkeypatch, tmp_path):
    output = tmp_path / "weights"
    model = nn.Linear(2, 2)
    _set_training_state(monkeypatch, iteration=5120)
    _enable_weight_audit(monkeypatch, output, steps="5120")

    _emit_model_weight_summary(model, step_count=5121)
    _EMITTED_MODEL_WEIGHT_ITERATIONS.clear()
    with torch.no_grad():
        model.weight.flatten()[0].add_(1)

    with pytest.raises(RuntimeError, match="already exists with different content"):
        _emit_model_weight_summary(model, step_count=5121)
    assert not list(output.glob("*.tmp"))


def test_emit_model_weight_summary_requires_rank_zero(monkeypatch, tmp_path):
    class TraversalForbidden:
        def named_parameters(self):
            raise AssertionError("rank one must not traverse parameters")

    output = tmp_path / "weights"
    _set_training_state(monkeypatch, iteration=5120)
    _enable_weight_audit(monkeypatch, output, steps="5120", rank=1)

    _emit_model_weight_summary(TraversalForbidden(), step_count=5121)

    assert not output.exists()


def test_rank_one_rejects_terminal_selection_before_model_traversal(monkeypatch, tmp_path):
    class TraversalForbidden:
        def named_parameters(self):
            raise AssertionError("terminal validation must precede model traversal")

    output = tmp_path / "weights"
    _set_training_state(
        monkeypatch,
        iteration=5120,
        train_iters=8192,
    )
    _enable_weight_audit(monkeypatch, output, steps="8192", rank=1)

    with pytest.raises(RuntimeError, match="iteration 8192 requires training through at least 8193"):
        _emit_model_weight_summary(TraversalForbidden(), step_count=5121)
    assert not output.exists()


def test_validation_forward_classification_is_batch_independent_across_tp_ranks():
    model = nn.Linear(2, 2)
    model.eval()

    # Loader and non-loader tensor-parallel ranks see a batch and None,
    # respectively, but both classify the model's eval forward identically.
    assert _is_validation_forward(model, {"timestep": torch.tensor([0])}) is True
    assert _is_validation_forward(model, None) is True

    model.train()
    assert _is_validation_forward(model, {"timestep": torch.tensor([0])}) is False
    assert _is_validation_forward(model, None) is False


@pytest.mark.parametrize("owns_batch", [True, False])
def test_tp_validation_timesteps_broadcast_on_every_rank(owns_batch):
    timestep = SimpleNamespace(is_cuda=True)

    class FakeTensorParallel:
        def __init__(self):
            self.calls = []

        def broadcast_data(self, keys, batch, dtype):
            self.calls.append((keys, batch, dtype))
            return {"timesteps": timestep}

    tensor_parallel = FakeTensorParallel()
    batch = {"timesteps": timestep} if owns_batch else None
    noise, observed_timesteps = _pregenerated_diffusion_inputs(
        batch,
        tp_size=2,
        is_validation=True,
        compute_dtype=torch.bfloat16,
        tensor_parallel=tensor_parallel,
    )

    assert noise is None
    assert observed_timesteps is timestep
    assert tensor_parallel.calls == [
        (["timesteps"], batch, torch.bfloat16),
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample", "0", "SAMPLE_SIZE"),
        ("sample", "4097", "SAMPLE_SIZE"),
        ("path", None, "PATH is required"),
        ("path", "relative/weights", "PATH must be absolute"),
        ("step_count", 0, "step_count"),
        ("step_count", True, "step_count"),
    ],
)
def test_rank_one_rejects_bad_audit_configuration_before_model_traversal(
    monkeypatch, tmp_path, field, value, message
):
    class TraversalForbidden:
        def named_parameters(self):
            raise AssertionError("configuration validation must precede model traversal")

    output = tmp_path / "weights"
    _set_training_state(monkeypatch, iteration=1, train_iters=2)
    _enable_weight_audit(monkeypatch, output, steps="1", rank=1)
    step_count = 1
    if field == "sample":
        monkeypatch.setenv("PRIMUS_AUDIT_MODEL_WEIGHT_SAMPLE_SIZE", value)
    elif field == "path":
        if value is None:
            monkeypatch.delenv("PRIMUS_AUDIT_MODEL_WEIGHT_PATH")
        else:
            monkeypatch.setenv("PRIMUS_AUDIT_MODEL_WEIGHT_PATH", value)
    else:
        step_count = value

    with pytest.raises(RuntimeError, match=message):
        _emit_model_weight_summary(TraversalForbidden(), step_count=step_count)
    assert not output.exists()


def test_rank_zero_reuses_prevalidated_sample_size_and_output_path(monkeypatch, tmp_path):
    output = tmp_path / "weights"
    model = nn.Linear(2, 2)
    _set_training_state(monkeypatch, iteration=1, train_iters=2)
    _enable_weight_audit(monkeypatch, output, steps="1", sample_size=2)
    context = _model_weight_audit_context(17, is_training=True)

    monkeypatch.setenv("PRIMUS_AUDIT_MODEL_WEIGHT_SAMPLE_SIZE", "invalid-after-preflight")
    monkeypatch.setenv("PRIMUS_AUDIT_MODEL_WEIGHT_PATH", "relative/after-preflight")
    _emit_model_weight_summary(
        model,
        step_count=17,
        audit_context=context,
    )

    record = json.loads((output / "completed_iteration_0000001.json").read_text())
    assert record["sample_size_per_parameter"] == 2
    assert record["forward_step_count"] == 17


def test_emit_model_weight_summary_env_unset_does_not_traverse_model(monkeypatch):
    class TraversalForbidden:
        def named_parameters(self):
            raise AssertionError("model traversal must stay behind the opt-in gate")

    monkeypatch.delenv("PRIMUS_AUDIT_MODEL_WEIGHT_STEPS", raising=False)

    def distributed_state_forbidden():
        raise AssertionError("distributed state must stay behind the opt-in gate")

    monkeypatch.setattr(torch.distributed, "is_initialized", distributed_state_forbidden)
    _emit_model_weight_summary(TraversalForbidden(), step_count=1)


def test_emit_model_weight_summary_skips_synthetic_warmup_before_traversal(monkeypatch, tmp_path):
    class TraversalForbidden:
        def named_parameters(self):
            raise AssertionError("synthetic warmup must not traverse parameters")

    output = tmp_path / "weights"
    _enable_weight_audit(monkeypatch, output, steps="1")
    monkeypatch.setenv("PRIMUS_SYNTHETIC_WARMUP_ACTIVE", "1")

    _emit_model_weight_summary(TraversalForbidden(), step_count=1)
    assert not output.exists()


def test_emit_model_weight_summary_skips_validation_before_traversal(monkeypatch, tmp_path):
    class TraversalForbidden:
        def named_parameters(self):
            raise AssertionError("validation must not traverse parameters")

    output = tmp_path / "weights"
    _enable_weight_audit(monkeypatch, output, steps="1")

    _emit_model_weight_summary(TraversalForbidden(), step_count=1, is_training=False)
    assert not output.exists()


def test_emit_model_weight_summary_does_not_mutate_cpu_rng(monkeypatch, tmp_path):
    output = tmp_path / "weights"
    model = nn.Linear(4, 2)
    _set_training_state(monkeypatch, iteration=1, train_iters=2)
    _enable_weight_audit(monkeypatch, output, steps="1")

    def cuda_seed_forbidden(*_args, **_kwargs):
        raise AssertionError("weight auditing must not reseed CUDA RNG")

    monkeypatch.setattr(torch.cuda, "manual_seed", cuda_seed_forbidden)
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()

    _emit_model_weight_summary(model, step_count=987)

    assert torch.equal(torch.random.get_rng_state(), before)


def test_model_weight_summary_observes_forward_pre_hook_weight(monkeypatch, tmp_path):
    output = tmp_path / "weights"
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)

    def refresh_parameter(module, _inputs):
        with torch.no_grad():
            module.weight.fill_(7)

    model.register_forward_pre_hook(refresh_parameter)
    _set_training_state(monkeypatch, iteration=5, train_iters=6)
    _enable_weight_audit(monkeypatch, output, steps="5", sample_size=1)
    context = _model_weight_audit_context(123, is_training=True)

    model_output = _emit_model_weight_summary_after_forward(
        model(torch.ones(1, 1)),
        model,
        step_count=123,
        audit_context=context,
    )

    record = json.loads((output / "completed_iteration_0000005.json").read_text())
    assert model_output.item() == 7
    assert record["sample_sum"] == 7
    assert record["parameters"][0]["sample_sum"] == 7


def _publish_and_capture(output, payload):
    try:
        _write_model_weight_summary_once(output, payload)
    except BaseException as error:
        return error
    return None


def test_atomic_concurrent_identical_publication_is_idempotent(monkeypatch, tmp_path):
    output = tmp_path / "summary.json"
    payload = {"version": 1, "value": 7}
    barrier = threading.Barrier(2)
    real_link = os.link

    def racing_link(*args, **kwargs):
        barrier.wait(timeout=5)
        return real_link(*args, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: _publish_and_capture(output, payload), range(2)))

    assert results == [None, None]
    assert _read_strict_json(output) == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_concurrent_different_publication_fails_closed(monkeypatch, tmp_path):
    output = tmp_path / "summary.json"
    payloads = [{"version": 1, "value": 7}, {"version": 1, "value": 8}]
    barrier = threading.Barrier(2)
    real_link = os.link

    def racing_link(*args, **kwargs):
        barrier.wait(timeout=5)
        return real_link(*args, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda payload: _publish_and_capture(output, payload), payloads))

    assert sum(result is None for result in results) == 1
    errors = [result for result in results if result is not None]
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "different content" in str(errors[0])
    assert _read_strict_json(output) in payloads
    assert not list(tmp_path.glob("*.tmp"))


def test_restart_canonicalizes_key_order_and_whitespace(tmp_path):
    output = tmp_path / "summary.json"
    original = b'{\n  "value": 7,\n  "version": 1\n}\n'
    output.write_bytes(original)

    _write_model_weight_summary_once(output, {"version": 1, "value": 7})

    assert output.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("existing", "payload"),
    [
        ({"version": True, "global_rank": False}, {"version": 1, "global_rank": 0}),
        ({"version": 1.0, "global_rank": 0}, {"version": 1, "global_rank": 0}),
    ],
)
def test_restart_rejects_json_type_conflicts(tmp_path, existing, payload):
    output = tmp_path / "summary.json"
    output.write_bytes(_encode_strict_json(existing))

    with pytest.raises(RuntimeError, match="different content"):
        _write_model_weight_summary_once(output, payload)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "existing",
    [
        {"version": 1, "forward_step_count": False, "num_microbatches": 1},
        {"version": 1, "forward_step_count": 7},
    ],
)
def test_restart_rejects_malformed_replay_provenance(tmp_path, existing):
    output = tmp_path / "summary.json"
    output.write_bytes(_encode_strict_json(existing))
    payload = {
        "version": 1,
        "forward_step_count": 7,
        "num_microbatches": 1,
    }

    with pytest.raises(RuntimeError, match="replay provenance|positive integer"):
        _write_model_weight_summary_once(output, payload)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("encoded", [b'{"value":NaN}\n', b'{"value":1e9999}\n'])
def test_read_strict_json_rejects_nonfinite_payloads(tmp_path, encoded):
    output = tmp_path / "summary.json"
    output.write_bytes(encoded)

    with pytest.raises(ValueError):
        _read_strict_json(output)


def test_read_strict_json_closes_descriptor_on_parse_failure(monkeypatch, tmp_path):
    output = tmp_path / "summary.json"
    output.write_text("{not-json}\n")
    real_open = os.open
    opened_descriptors = []

    def capture_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        opened_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", capture_open)
    with pytest.raises(json.JSONDecodeError):
        _read_strict_json(output)

    assert len(opened_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(opened_descriptors[0])


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW is unavailable")
def test_publication_rejects_symlink_target(tmp_path):
    target = tmp_path / "target.json"
    target.write_bytes(_encode_strict_json({"version": 1}))
    output = tmp_path / "summary.json"
    output.symlink_to(target)

    with pytest.raises(OSError):
        _write_model_weight_summary_once(output, {"version": 1})
    assert output.is_symlink()
    assert not list(tmp_path.glob("*.tmp"))


def test_publication_rejects_nonregular_target(tmp_path):
    output = tmp_path / "summary.json"
    output.mkdir()

    with pytest.raises((OSError, RuntimeError), match="regular file|directory"):
        _write_model_weight_summary_once(output, {"version": 1})
    assert not list(tmp_path.glob("*.tmp"))


def test_file_fsync_failure_closes_descriptor_and_removes_temporary(monkeypatch, tmp_path):
    output = tmp_path / "summary.json"
    fsync_descriptors = []

    def fail_file_fsync(descriptor):
        fsync_descriptors.append(descriptor)
        raise OSError("injected file fsync failure")

    monkeypatch.setattr(os, "fsync", fail_file_fsync)
    with pytest.raises(OSError, match="file fsync"):
        _write_model_weight_summary_once(output, {"version": 1})

    assert len(fsync_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(fsync_descriptors[0])
    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_directory_fsync_failure_closes_descriptor_and_removes_temporary(monkeypatch, tmp_path):
    output = tmp_path / "summary.json"
    real_fsync = os.fsync
    fsync_descriptors = []

    def fail_directory_fsync(descriptor):
        fsync_descriptors.append(descriptor)
        if len(fsync_descriptors) == 2:
            raise OSError("injected directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="directory fsync"):
        _write_model_weight_summary_once(output, {"version": 1})

    assert len(fsync_descriptors) == 2
    with pytest.raises(OSError):
        os.fstat(fsync_descriptors[1])
    assert output.exists()
    assert not list(tmp_path.glob("*.tmp"))
