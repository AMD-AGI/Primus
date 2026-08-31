###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Drive a full run's worth of MLPerf records and hand the file to the checker.

Every other test in this area asserts against a mock and therefore only says
that the code emitted what the test expected. This one uses the real
``mlperf_logging`` package end to end: real ``mllog`` writes the file, and
``mlperf_logging.compliance_checker`` reads it back under the same ruleset a
submission is judged with. It fails when the emitted log stops being a valid
one, including for reasons nobody wrote an assertion for.

Skipped when ``mlperf_logging`` is not installed, which is the case on plain
development hosts; the package ships in the training image.
"""

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

mlperf_logging = pytest.importorskip("mlperf_logging")

RULESET = "6.0.0"
GLOBAL_BATCH_SIZE = 512
MICRO_BATCH_SIZE = 64
EVAL_INTERVAL = 512
TARGET_VAL_LOSS = 0.586


def _has_ruleset() -> bool:
    from pathlib import Path

    root = Path(mlperf_logging.__file__).parent
    return (root / "compliance_checker" / f"training_{RULESET}" / "closed_flux1.yaml").exists()


pytestmark = pytest.mark.skipif(
    not _has_ruleset(),
    reason=f"installed mlperf_logging has no training_{RULESET} flux1 ruleset",
)


def _install_fake_megatron(monkeypatch):
    import sys

    megatron_mod = types.ModuleType("megatron")
    training_pkg = types.ModuleType("megatron.training")
    training_mod = types.ModuleType("megatron.training.training")
    global_vars_mod = types.ModuleType("megatron.training.global_vars")

    training_mod.train_step = lambda *a, **k: ({}, 0, False, False, 0, 0.0, 0, None)
    training_mod.training_log = lambda *a, **k: None
    training_mod.print_rank_last = lambda msg: None
    training_mod.get_tensorboard_writer = lambda: None
    training_mod.get_wandb_writer = lambda: None
    training_mod.setup_model_and_optimizer = lambda *a, **k: ("model", "optimizer", "scheduler")
    training_mod.build_train_valid_test_data_iterators = lambda *a, **k: ("train", "valid", "test")
    training_mod.get_model_config = lambda model: SimpleNamespace()
    training_mod.get_forward_backward_func = lambda: (lambda *a, **k: None)
    training_pkg.pretrain = lambda *a, **k: None

    training_pkg.training = training_mod
    megatron_mod.training = training_pkg

    megatron_args = SimpleNamespace(
        iteration=0,
        curr_iteration=0,
        consumed_train_samples=0,
        skipped_train_samples=0,
        train_iters=16000,
        eval_interval=EVAL_INTERVAL,
        do_valid=True,
        global_batch_size=GLOBAL_BATCH_SIZE,
        micro_batch_size=MICRO_BATCH_SIZE,
    )
    global_vars_mod.get_args = lambda: megatron_args
    training_pkg.get_args = global_vars_mod.get_args

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", global_vars_mod)

    return training_mod, megatron_args


def _make_ctx():
    """A context carrying the values closed_flux1.yaml pins exactly."""
    params = SimpleNamespace(
        mlperf_mode=True,
        warmup_train_steps=0,
        target_val_loss=TARGET_VAL_LOSS,
        global_batch_size=GLOBAL_BATCH_SIZE,
        micro_batch_size=MICRO_BATCH_SIZE,
        data_parallel_size=8,
        seed=42,
        log_interval=10,
        lr=2e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
        clip_grad=1.0,
        lr_warmup_iters=1600,
        eval_interval=EVAL_INTERVAL,
        eval_samples=29696,
        train_samples=1099776,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        transformer_impl="local",
        wall_clock_step_timer=False,
    )
    return SimpleNamespace(
        extra={"module_config": SimpleNamespace(params=params)},
        backend="megatron",
        phase="before_train",
    )


@pytest.fixture
def _run_environment(monkeypatch, tmp_path):
    from primus.backends.megatron.patches import mlperf_boundary

    mlperf_boundary.reset_for_tests()
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MLLOG_OUTPUT_FILE", str(tmp_path / "result_0.txt"))
    monkeypatch.setenv("MLPERF_CLEAR_CACHES", "true")
    monkeypatch.setenv("MLLOG_SUBMISSION_ORG", "AMD")
    monkeypatch.setenv("MLLOG_SUBMISSION_DIVISION", "closed")
    monkeypatch.setenv("MLLOG_SUBMISSION_PLATFORM", "MI355X")
    monkeypatch.setenv("EXP", "examples/megatron/configs/MI355X/diffusion/flux_mlperf.yaml")
    monkeypatch.setenv("MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR", "fp8")
    monkeypatch.setenv("MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN", "bfloat16")
    monkeypatch.setenv("MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM", "bfloat16")
    yield tmp_path / "result_0.txt"
    mlperf_boundary.reset_for_tests()
    _detach_mllog_handlers()


def _detach_mllog_handlers():
    """mllog's logger is a process-wide singleton; leave it as it was found."""
    from mlperf_logging import mllog

    logger = mllog.get_mllogger().logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _drive_a_run(mt, eval_losses):
    """Walk the run through the same call sequence Megatron would."""
    for module in (
        "primus.backends.megatron.patches.mlperf_logging_patches",
        "primus.backends.megatron.patches.mlperf_warmup_patches",
    ):
        pytest.MonkeyPatch().setattr(f"{module}.log_rank_0", lambda *a, **k: None)

    from primus.backends.megatron.patches.mlperf_logging_patches import (
        patch_mlperf_logging,
    )

    patch_mlperf_logging(_make_ctx())

    # Data iterators are built after the model, and this is where the clock
    # starts.
    mt.build_train_valid_test_data_iterators(None)

    iteration = 0
    for loss in eval_losses:
        for _ in range(EVAL_INTERVAL // 10):
            iteration += 10
            mt.training_log({"loss": 1.2}, {}, 2e-4, iteration, 1.0, False, False, 0.0, None, 0, None)

        mt.evaluate = lambda *a, _loss=loss, **k: ({"loss": _loss},)
        mt.evaluate_and_print_results(
            f"iteration {iteration}",
            lambda: None,
            None,
            [MagicMock()],
            iteration,
            None,
            MagicMock(),
        )


def _values(log_path, key):
    """Every value logged under ``key``, in emission order."""
    import json

    return [
        json.loads(line[line.index("{") :])["value"]
        for line in log_path.read_text().splitlines()
        if f'"key": "{key}"' in line
    ]


def _check(log_path):
    from mlperf_logging.compliance_checker import mlp_compliance

    checker = mlp_compliance.make_checker(usage="training", ruleset=RULESET, quiet=True, werror=False)
    valid, _system_id, _benchmark, _result = mlp_compliance.main(
        str(log_path), f"training_{RULESET}/common.yaml", checker
    )
    return valid


def test_a_converged_run_passes_the_compliance_checker(monkeypatch, _run_environment):
    log_path = _run_environment
    mt, _ = _install_fake_megatron(monkeypatch)

    # Real Megatron calls evaluate() from inside evaluate_and_print_results;
    # the capture wrapper in the patch depends on that.
    mt.evaluate_and_print_results = lambda *a, **k: mt.evaluate(*a, **k)

    _drive_a_run(mt, eval_losses=[0.9, 0.7, 0.5])

    assert log_path.exists(), "rank zero did not write the result file"
    assert _check(log_path), log_path.read_text()


def test_a_post_convergence_evaluation_cannot_pollute_the_log(monkeypatch, _run_environment):
    """The reported symptom, end to end, against the real checker.

    pretrain validates once more after the training loop exits. That evaluation
    draws fresh VAE and flow-matching noise, so it reports a different loss and
    can land above the target the run just met, leaving the log ending on a
    result that contradicts the one that stopped the run.

    Two things prevent it. do_valid is cleared on convergence, so pretrain
    skips the evaluation entirely; and the wrapper records nothing once
    run_stop has fired, for any caller that evaluates regardless. This asserts
    the first and then exercises the second, since the first alone would leave
    nothing to test.
    """
    log_path = _run_environment
    mt, megatron_args = _install_fake_megatron(monkeypatch)
    mt.evaluate_and_print_results = lambda *a, **k: mt.evaluate(*a, **k)

    _drive_a_run(mt, eval_losses=[0.9, 0.7, 0.5])

    assert megatron_args.do_valid is False, "pretrain would still run its post-training eval"

    # Evaluate anyway, at the loss the re-evaluation was observed to produce.
    mt.evaluate = lambda *a, **k: ({"loss": 0.91},)
    mt.evaluate_and_print_results(
        "iteration 1536 on validation set",
        lambda: None,
        None,
        [MagicMock()],
        1536,
        None,
        MagicMock(),
    )

    assert _check(log_path), log_path.read_text()
    assert _values(log_path, "eval_accuracy") == [0.9, 0.7, 0.5]
    assert _values(log_path, "run_stop") == ["success"]


def test_an_exhausted_run_fails_only_on_quality(monkeypatch, caplog, _run_environment):
    """A run that misses the target is rejected, and that is the right answer.

    ``closed_flux1.yaml`` requires at least one ``eval_accuracy`` at or below
    0.586, so no amount of well-formed logging makes a non-converged run pass.
    What the terminal ``run_stop`` buys is that the rejection is about the
    model and nothing else -- the log is otherwise complete, so the run can be
    read by the RCP checker and counted in the campaign instead of being an
    unparseable hole in it.
    """
    log_path = _run_environment
    mt, megatron_args = _install_fake_megatron(monkeypatch)
    mt.evaluate_and_print_results = lambda *a, **k: mt.evaluate(*a, **k)

    _drive_a_run(mt, eval_losses=[0.9, 0.8, 0.7])
    megatron_args.curr_iteration = 16000

    from primus.backends.megatron.patches.mlperf_logging_patches import (
        patch_mlperf_terminal_run_stop,
    )

    patch_mlperf_terminal_run_stop(_make_ctx())

    assert '"key": "run_stop"' in log_path.read_text()
    assert '"status": "aborted"' in log_path.read_text()

    with caplog.at_level("WARNING"):
        assert not _check(log_path)

    failures = [record.message for record in caplog.records if "Failed checks" in record.message]
    assert failures, "expected the checker to report why it rejected the log"
    assert all("eval_accuracy" in message for message in failures), failures


def test_a_run_without_a_terminal_record_is_rejected(monkeypatch, caplog, _run_environment):
    """Guards the guard: the checker really does fail a log with no run_stop."""
    log_path = _run_environment
    mt, _ = _install_fake_megatron(monkeypatch)
    mt.evaluate_and_print_results = lambda *a, **k: mt.evaluate(*a, **k)

    # Converges, so quality is not what is missing here.
    _drive_a_run(mt, eval_losses=[0.9, 0.5])
    assert '"key": "run_stop"' in log_path.read_text()

    without_run_stop = log_path.with_name("no_run_stop.txt")
    without_run_stop.write_text(
        "".join(line for line in log_path.read_text().splitlines(keepends=True) if "run_stop" not in line)
    )

    with caplog.at_level("WARNING"):
        assert not _check(without_run_stop)

    assert any("run_stop" in record.message for record in caplog.records)
