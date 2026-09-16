###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MLPerf logging and warmup patches.

Covers the production patch behavior (CPU-only): logging-patch monkey-patching
and idempotency, the INIT_STOP/RUN_START emission on the first post-warmup
training_log call, convergence detection, and the FP8/optimizer warmup helper
functions (_reset_fp8_te_spec, seed FP8 amax, optimizer neuter/restore/reset).
"""

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_MLLOG_CONSTANT_NAMES = {
    "INIT_START": "init_start",
    "INIT_STOP": "init_stop",
    "RUN_START": "run_start",
    "RUN_STOP": "run_stop",
    "SUBMISSION_BENCHMARK": "submission_benchmark",
    "SUBMISSION_ORG": "submission_org",
    "SUBMISSION_DIVISION": "submission_division",
    "SUBMISSION_PLATFORM": "submission_platform",
    "SUBMISSION_STATUS": "submission_status",
    "SEED": "seed",
    "GLOBAL_BATCH_SIZE": "global_batch_size",
    "TRAIN_SAMPLES": "train_samples",
    "EVAL_SAMPLES": "eval_samples",
    "GRADIENT_ACCUMULATION_STEPS": "gradient_accumulation_steps",
    "OPT_NAME": "opt_name",
    "OPT_BASE_LR": "opt_base_learning_rate",
    "EVAL_ACCURACY": "eval_accuracy",
    "EVAL_START": "eval_start",
    "EVAL_STOP": "eval_stop",
    "EPOCH_START": "epoch_start",
    "EPOCH_STOP": "epoch_stop",
    "BLOCK_START": "block_start",
    "BLOCK_STOP": "block_stop",
}


@pytest.fixture(autouse=True)
def _mlperf_submission_environment(monkeypatch, tmp_path):
    """Everything the logger now refuses to guess.

    The patch fails closed on each of these, so without them every test in
    this module would fail on identity rather than on what it is testing.
    """
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MLLOG_OUTPUT_FILE", str(tmp_path / "result_0.txt"))
    monkeypatch.setenv("MLPERF_CLEAR_CACHES", "true")
    monkeypatch.setenv("MLLOG_SUBMISSION_ORG", "AMD")
    monkeypatch.setenv("MLLOG_SUBMISSION_DIVISION", "closed")
    monkeypatch.setenv("MLLOG_SUBMISSION_PLATFORM", "MI355X")
    monkeypatch.setenv("EXP", "examples/megatron/configs/MI355X/diffusion/test.yaml")
    monkeypatch.setenv("MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR", "fp8")
    monkeypatch.setenv("MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN", "bfloat16")
    monkeypatch.setenv("MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM", "bfloat16")


@pytest.fixture(autouse=True)
def _clean_boundary():
    """The boundary keeps module-level state, so tests must not inherit it."""
    from primus.backends.megatron.patches import mlperf_boundary

    mlperf_boundary.reset_for_tests()
    yield
    mlperf_boundary.reset_for_tests()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_megatron(monkeypatch):
    """Install a fake megatron.training.training module into sys.modules."""
    import sys

    megatron_mod = types.ModuleType("megatron")
    training_pkg = types.ModuleType("megatron.training")
    training_mod = types.ModuleType("megatron.training.training")
    global_vars_mod = types.ModuleType("megatron.training.global_vars")

    def fake_train_step(fwd, data_iter, model, optimizer, sched, config, fwdbwd, iteration=None):
        return {}, 0, False, False, 0, 0.0, 0, None

    def fake_training_log(*args, **kwargs):
        return None

    def fake_eval(*args, **kwargs):
        return None

    def fake_evaluate(*args, **kwargs):
        return ({},)

    def fake_print_rank_last(msg):
        pass

    def fake_get_tb_writer():
        return None

    def fake_get_wandb_writer():
        return None

    training_mod.train_step = fake_train_step
    training_mod.training_log = fake_training_log
    training_mod.evaluate_and_print_results = fake_eval
    training_mod.evaluate = fake_evaluate
    training_mod.print_rank_last = fake_print_rank_last
    training_mod.get_tensorboard_writer = fake_get_tb_writer
    training_mod.get_wandb_writer = fake_get_wandb_writer

    # The three entry points the MLPerf boundary wraps, plus the two helpers
    # the relocated warmup reads off the training module.
    training_mod.setup_model_and_optimizer = lambda *a, **k: ("model", "optimizer", "scheduler")
    training_mod.build_train_valid_test_data_iterators = lambda *a, **k: ("train", "valid", "test")
    training_mod.get_model_config = lambda model: SimpleNamespace()
    training_mod.get_forward_backward_func = lambda: (lambda *a, **k: None)
    training_pkg.pretrain = lambda *a, **k: None

    training_pkg.training = training_mod
    megatron_mod.training = training_pkg

    # Store megatron args for get_args
    _megatron_args = SimpleNamespace(
        iteration=0,
        curr_iteration=0,
        consumed_train_samples=0,
        skipped_train_samples=0,
        train_iters=100,
        eval_interval=10,
        do_valid=True,
        global_batch_size=512,
        micro_batch_size=64,
    )
    global_vars_mod.get_args = lambda: _megatron_args

    training_pkg.get_args = global_vars_mod.get_args

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", global_vars_mod)

    return training_mod, _megatron_args


def _install_fake_grad_finalize(monkeypatch):
    """Point megatron.core.distributed.finalize_model_grads at a sentinel.

    Call this *before* ``_install_fake_megatron``: the warmup imports the
    callback lazily, and once ``megatron`` is a stub the real package can no
    longer be reached to swap the attribute on.
    """
    import importlib
    import sys

    def finalize_model_grads(*args, **kwargs):
        return None

    try:
        real = importlib.import_module("megatron.core.distributed")
    except ImportError:
        core_mod = types.ModuleType("megatron.core")
        core_mod.__path__ = []
        distributed_mod = types.ModuleType("megatron.core.distributed")
        distributed_mod.finalize_model_grads = finalize_model_grads
        core_mod.distributed = distributed_mod
        monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
        monkeypatch.setitem(sys.modules, "megatron.core.distributed", distributed_mod)
    else:
        monkeypatch.setattr(real, "finalize_model_grads", finalize_model_grads)

    return finalize_model_grads


def _warmup_actors(monkeypatch):
    """A model / optimizer / scheduler trio small enough to warm up on CPU."""
    import torch

    # The warmup brackets itself with device syncs; there is no device here.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    model = torch.nn.Linear(2, 2)
    optimizer = SimpleNamespace(
        param_groups=[{"betas": (0.9, 0.95), "weight_decay": 0.1, "step": 0}],
        state={},
        zero_grad=lambda set_to_none=True: None,
    )
    return [model], optimizer, SimpleNamespace(num_steps=0)


def _install_mock_mllog(monkeypatch, events=None):
    """Install a fake mlperf_logging.mllog and return (module, events).

    ``events`` accumulates ``(kind, key, value, metadata)`` in emission order,
    which is what most of the assertions below are about.
    """
    import sys

    if events is None:
        events = []

    def _record(kind):
        def _emit(key, value=None, metadata=None):
            events.append((kind, key, value, metadata))

        return _emit

    mock_mllogger = MagicMock()
    mock_mllogger.start = _record("start")
    mock_mllogger.end = _record("end")
    mock_mllogger.event = _record("event")

    mock_mllog_module = MagicMock()
    mock_mllog_module.get_mllogger.return_value = mock_mllogger
    mock_mllog_module.constants = SimpleNamespace(**_MLLOG_CONSTANT_NAMES)

    mock_mlperf_pkg = MagicMock()
    mock_mlperf_pkg.mllog = mock_mllog_module

    monkeypatch.setitem(sys.modules, "mlperf_logging", mock_mlperf_pkg)
    monkeypatch.setitem(sys.modules, "mlperf_logging.mllog", mock_mllog_module)

    return mock_mllog_module, events


def _keys(events):
    return [entry[1] for entry in events]


def _make_ctx(
    mlperf_mode=False,
    warmup_train_steps=0,
    target_val_loss=0.586,
    **kwargs,
):
    """Build a minimal PatchContext-like object."""
    params = SimpleNamespace(
        mlperf_mode=mlperf_mode,
        warmup_train_steps=warmup_train_steps,
        target_val_loss=target_val_loss,
        global_batch_size=kwargs.get("global_batch_size", 512),
        micro_batch_size=kwargs.get("micro_batch_size", 64),
        seed=kwargs.get("seed", 42),
        log_interval=kwargs.get("log_interval", 10),
        lr=kwargs.get("lr", 2e-4),
        adam_beta1=kwargs.get("adam_beta1", 0.9),
        adam_beta2=kwargs.get("adam_beta2", 0.95),
        adam_eps=kwargs.get("adam_eps", 1e-8),
        weight_decay=kwargs.get("weight_decay", 0.1),
        image_size=kwargs.get("image_size", 256),
        vae_latent_mode=kwargs.get("vae_latent_mode", "resample"),
        transformer_impl=kwargs.get("transformer_impl", "local"),
        use_fsdp2_fp8_all_gather=kwargs.get("use_fsdp2_fp8_all_gather", False),
        wall_clock_step_timer=False,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "global_batch_size",
                "micro_batch_size",
                "seed",
                "log_interval",
                "lr",
                "adam_beta1",
                "adam_beta2",
                "adam_eps",
                "weight_decay",
                "image_size",
                "vae_latent_mode",
                "transformer_impl",
                "use_fsdp2_fp8_all_gather",
            )
        },
    )
    module_config = SimpleNamespace(params=params)
    return SimpleNamespace(
        extra={"module_config": module_config},
        backend="megatron",
        phase="before_train",
    )


def _silence_log_rank_0(monkeypatch):
    for module in (
        "primus.backends.megatron.patches.mlperf_logging_patches",
        "primus.backends.megatron.patches.mlperf_warmup_patches",
    ):
        monkeypatch.setattr(f"{module}.log_rank_0", lambda *a, **k: None)


# ============================================================================
# Submission identity and precision disclosure
# ============================================================================


def test_precision_disclosures_are_explicit(monkeypatch):
    from primus.backends.megatron.patches.mlperf_logging_patches import (
        _precision_disclosures_from_env,
    )

    monkeypatch.setenv("MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR", "mxfp6")
    monkeypatch.setenv("MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN", "bfloat16")
    monkeypatch.setenv("MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM", "bfloat16")

    assert _precision_disclosures_from_env() == {
        "lowest_numerical_precision_in_linear": "mxfp6",
        "lowest_numerical_precision_in_attn": "bfloat16",
        "lowest_numerical_precision_in_comm": "bfloat16",
    }


def test_missing_precision_disclosure_fails_mlperf_startup(monkeypatch):
    from primus.backends.megatron.patches.mlperf_logging_patches import (
        _precision_disclosures_from_env,
    )

    for name in (
        "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR",
        "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN",
        "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="explicit precision disclosures"):
        _precision_disclosures_from_env()


def test_precision_value_outside_the_checker_enum_warns(monkeypatch, caplog):
    """mxfp6 is not yet an accepted disclosure, and a run must say so.

    The value is still emitted -- describing an MXFP6 run as fp8 would be
    worse than a log the checker rejects -- but it cannot pass silently.
    """
    from primus.backends.megatron.patches.mlperf_logging_patches import (
        _precision_disclosures_from_env,
    )

    monkeypatch.setenv("MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR", "mxfp6")

    with caplog.at_level("WARNING"):
        values = _precision_disclosures_from_env()

    assert values["lowest_numerical_precision_in_linear"] == "mxfp6"
    assert "mxfp6" in caplog.text
    assert "compliance checker" in caplog.text


@pytest.mark.parametrize(
    "variable",
    ["MLLOG_SUBMISSION_ORG", "MLLOG_SUBMISSION_DIVISION", "MLLOG_SUBMISSION_PLATFORM"],
)
def test_submission_identity_never_defaults(monkeypatch, variable):
    from primus.backends.megatron.patches.mlperf_logging_patches import (
        _submission_identity_from_env,
    )

    monkeypatch.delenv(variable, raising=False)

    with pytest.raises(RuntimeError, match=variable):
        _submission_identity_from_env()


def test_submission_division_must_be_a_real_division(monkeypatch):
    from primus.backends.megatron.patches.mlperf_logging_patches import (
        _submission_identity_from_env,
    )

    monkeypatch.setenv("MLLOG_SUBMISSION_DIVISION", "network")

    with pytest.raises(RuntimeError, match="closed"):
        _submission_identity_from_env()


# ============================================================================
# Patch registration
# ============================================================================


class TestLoggingPatchMonkeyPatching:
    """Verify that the logging patch replaces the expected functions."""

    def test_installs_wrappers(self, monkeypatch):
        mt, _ = _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _install_mock_mllog(monkeypatch)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        original_tl = mt.training_log
        original_eval = mt.evaluate_and_print_results
        original_prl = mt.print_rank_last

        patch_mlperf_logging(_make_ctx(mlperf_mode=True))

        assert mt.training_log is not original_tl
        assert mt.evaluate_and_print_results is not original_eval
        assert mt.print_rank_last is not original_prl
        assert getattr(mt, "_primus_mlperf_logging_installed", False) is True

    def test_idempotent(self, monkeypatch):
        mt, _ = _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _install_mock_mllog(monkeypatch)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        ctx = _make_ctx(mlperf_mode=True)
        mt._primus_mlperf_logging_installed = False

        patch_mlperf_logging(ctx)
        first_tl = mt.training_log
        first_eval = mt.evaluate_and_print_results

        patch_mlperf_logging(ctx)
        assert mt.training_log is first_tl
        assert mt.evaluate_and_print_results is first_eval

    def test_rank_zero_writes_the_log_to_a_file(self, monkeypatch, tmp_path):
        """The submitted artifact is a file, not whatever landed on stdout."""
        _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        mock_mllog, _ = _install_mock_mllog(monkeypatch)

        output = tmp_path / "result_3.txt"
        monkeypatch.setenv("MLLOG_OUTPUT_FILE", str(output))

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        patch_mlperf_logging(_make_ctx(mlperf_mode=True))

        mock_mllog.config.assert_called_once()
        assert mock_mllog.config.call_args.kwargs["filename"] == str(output)

    def test_missing_output_file_fails_startup(self, monkeypatch):
        _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _install_mock_mllog(monkeypatch)
        monkeypatch.delenv("MLLOG_OUTPUT_FILE", raising=False)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        with pytest.raises(RuntimeError, match="MLLOG_OUTPUT_FILE"):
            patch_mlperf_logging(_make_ctx(mlperf_mode=True))


# ============================================================================
# The measured-time boundary
# ============================================================================


class TestMeasuredTimeBoundary:
    """run_start must precede every read of the real dataset."""

    def test_transition_fires_before_the_data_iterators_are_built(self, monkeypatch):
        mt, _ = _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _, events = _install_mock_mllog(monkeypatch)

        order = []
        original_build = mt.build_train_valid_test_data_iterators
        mt.build_train_valid_test_data_iterators = lambda *a, **k: (
            order.append("build_data_iterators"),
            original_build(*a, **k),
        )[1]

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        patch_mlperf_logging(_make_ctx(mlperf_mode=True))
        events.clear()

        mt.build_train_valid_test_data_iterators(None)

        keys = _keys(events)
        assert "init_stop" in keys and "run_start" in keys
        assert keys.index("init_stop") < keys.index("run_start")
        # The wrapper records nothing itself, so the only way the dataset call
        # can be ordered against the log is that it has not happened yet.
        assert order == ["build_data_iterators"]

    def test_transition_fires_once(self, monkeypatch):
        """Virtual pipelining builds iterators per stage; the clock starts once."""
        mt, _ = _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _, events = _install_mock_mllog(monkeypatch)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        patch_mlperf_logging(_make_ctx(mlperf_mode=True))
        events.clear()

        mt.build_train_valid_test_data_iterators(None)
        mt.build_train_valid_test_data_iterators(None)

        assert _keys(events).count("run_start") == 1

    def test_pre_run_hooks_finish_before_the_clock_starts(self, monkeypatch):
        """Warmup is initialization, so it belongs on the init side of run_start."""
        mt, _ = _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _, events = _install_mock_mllog(monkeypatch)

        from primus.backends.megatron.patches import mlperf_boundary
        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        patch_mlperf_logging(_make_ctx(mlperf_mode=True))
        events.clear()

        mlperf_boundary.register_pre_run_hook(
            "fake_warmup", lambda: events.append(("hook", "warmup", None, None)), order=10
        )

        mt.build_train_valid_test_data_iterators(None)

        keys = _keys(events)
        assert keys.index("warmup") < keys.index("run_start")

    def test_model_optimizer_and_forward_step_are_captured(self, monkeypatch):
        """Warmup needs these, and nothing hands them to a before_train patch."""
        mt, _ = _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _install_mock_mllog(monkeypatch)

        from primus.backends.megatron.patches import mlperf_boundary
        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        patch_mlperf_logging(_make_ctx(mlperf_mode=True))

        import megatron.training as megatron_training_pkg

        def forward_step_func(*a, **k):
            return None

        megatron_training_pkg.pretrain(None, None, None, forward_step_func)
        mt.setup_model_and_optimizer(None, None)

        captured = mlperf_boundary.captured()
        assert captured["forward_step_func"] is forward_step_func
        assert captured["model"] == "model"
        assert captured["optimizer"] == "optimizer"
        assert captured["opt_param_scheduler"] == "scheduler"

    def test_training_log_backstop_does_not_restart_the_clock(self, monkeypatch):
        """The old first-training_log trigger stays, but must now be inert."""
        mt, _ = _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _, events = _install_mock_mllog(monkeypatch)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        patch_mlperf_logging(_make_ctx(mlperf_mode=True, log_interval=1))
        mt.build_train_valid_test_data_iterators(None)
        events.clear()

        mt.training_log({"loss": 0.5}, {}, 1e-4, 1, 1.0, False, False, 0.0, None, 0, None)

        keys = _keys(events)
        assert "init_stop" not in keys
        assert "run_start" not in keys

    def test_warmup_registers_at_the_boundary_in_mlperf_mode(self, monkeypatch):
        mt, _ = _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)

        from primus.backends.megatron.patches import mlperf_boundary
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            patch_mlperf_warmup,
        )

        original_train_step = mt.train_step
        patch_mlperf_warmup(_make_ctx(mlperf_mode=True, warmup_train_steps=2))

        assert mt.train_step is original_train_step, "warmup must not wrap train_step in MLPerf mode"
        assert [name for _order, name, _fn in mlperf_boundary._HOOKS] == ["mlperf_warmup"]

    def test_warmup_stays_on_train_step_outside_mlperf_mode(self, monkeypatch):
        """Development recipes keep the behavior they were tuned against."""
        mt, _ = _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)

        from primus.backends.megatron.patches import mlperf_boundary
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            patch_mlperf_warmup,
        )

        patch_mlperf_warmup(_make_ctx(mlperf_mode=False, warmup_train_steps=2))

        assert getattr(mt.train_step, "_primus_warmup_hook", False) is True
        assert mlperf_boundary._HOOKS == []

    def test_warmup_steps_run_with_a_grad_finalize_callback(self, monkeypatch):
        """Every warmup backward must have something that waits on its grad sync.

        The boundary warmup runs before Megatron's ``train()``, which is where
        ``config.finalize_model_grads_func`` is assigned. That callback is the
        only caller of ``finish_grad_sync()``, so if it is missing a warmup
        backward dispatches the data-parallel reduce-scatter and nothing ever
        waits on it. The first real step then asserts "Should not have multiple
        communication calls outstanding at once" and training never starts.
        """
        sentinel = _install_fake_grad_finalize(monkeypatch)
        _, megatron_args = _install_fake_megatron(monkeypatch)
        # The local-spec FP8 reset pulls in real Megatron enums, which the fake
        # megatron above does not provide and this test is not about.
        megatron_args.transformer_impl = "transformer_engine"
        _silence_log_rank_0(monkeypatch)
        model, optimizer, scheduler = _warmup_actors(monkeypatch)
        config = SimpleNamespace(finalize_model_grads_func=None)

        seen = []

        def _train_step(fwd, data_iter, mdl, opt, sched, cfg, fwdbwd, iteration=None):
            seen.append(cfg.finalize_model_grads_func)

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _run_warmup_and_restore,
        )

        _run_warmup_and_restore(
            warmup_steps=2,
            train_step_fn=_train_step,
            forward_step_func=lambda *a, **k: None,
            synthetic_iter=iter(()),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=config,
            forward_backward_func=lambda *a, **k: None,
            iteration=0,
        )

        assert seen == [sentinel, sentinel]
        assert config.finalize_model_grads_func is None, "warmup must leave the config as it found it"

    def test_warmup_keeps_a_finalize_callback_the_caller_already_set(self, monkeypatch):
        _install_fake_grad_finalize(monkeypatch)
        _, megatron_args = _install_fake_megatron(monkeypatch)
        megatron_args.transformer_impl = "transformer_engine"
        _silence_log_rank_0(monkeypatch)
        model, optimizer, scheduler = _warmup_actors(monkeypatch)

        def preexisting(*args, **kwargs):
            return None

        config = SimpleNamespace(finalize_model_grads_func=preexisting)

        seen = []

        def _train_step(fwd, data_iter, mdl, opt, sched, cfg, fwdbwd, iteration=None):
            seen.append(cfg.finalize_model_grads_func)

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _run_warmup_and_restore,
        )

        _run_warmup_and_restore(
            warmup_steps=1,
            train_step_fn=_train_step,
            forward_step_func=lambda *a, **k: None,
            synthetic_iter=iter(()),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=config,
            forward_backward_func=lambda *a, **k: None,
            iteration=0,
        )

        assert seen == [preexisting]
        assert config.finalize_model_grads_func is preexisting

    def test_warmup_restores_the_ddp_grad_ready_calibration(self, monkeypatch):
        """Warmup must not leave its own calibration behind for the first real step.

        Megatron's gradient buckets calibrate on their first batch, and from the second on issue
        the reduce-scatter only once that golden count recurs. Warmup batches consume the
        calibration, so the golden counts describe a synthetic step. Under gradient accumulation
        they then never recur: every parameter reports in and the collective is still never
        issued, which ``finish_grad_sync`` raises as "Communication call has not been issued for
        this bucket". Installing the grad-finalize callback does not cover this -- it was
        reproduced on 8x MI355X at accumulation 2 with that fix already applied.
        """
        _install_fake_grad_finalize(monkeypatch)
        _, megatron_args = _install_fake_megatron(monkeypatch)
        megatron_args.transformer_impl = "transformer_engine"
        _silence_log_rank_0(monkeypatch)
        model, optimizer, scheduler = _warmup_actors(monkeypatch)

        param = next(model[0].parameters())
        group = SimpleNamespace(
            is_first_batch=False,
            golden_per_param_grad_ready_counts={param: 1},
            per_param_grad_ready_counts={param: 1},
            grad_reduce_handle=None,
        )
        model[0].bucket_groups = [group]

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _run_warmup_and_restore,
        )

        _run_warmup_and_restore(
            warmup_steps=1,
            train_step_fn=lambda *a, **k: None,
            forward_step_func=lambda *a, **k: None,
            synthetic_iter=iter(()),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=SimpleNamespace(finalize_model_grads_func=None),
            forward_backward_func=lambda *a, **k: None,
            iteration=0,
        )

        assert group.is_first_batch is True, "the first real step must calibrate, not inherit"
        assert group.golden_per_param_grad_ready_counts == {}
        assert group.per_param_grad_ready_counts == {}

    def test_warmup_drains_an_outstanding_grad_reduce_handle(self, monkeypatch):
        """A collective left in flight by warmup must be awaited, not handed on.

        Expected to be a no-op now that the grad-finalize callback is installed for the warmup
        steps, so this pins the guard rather than a live failure: the handle would belong to a
        synthetic step whose gradients are about to be discarded, and the first real step cannot
        see why its bucket is busy.
        """
        _install_fake_grad_finalize(monkeypatch)
        _, megatron_args = _install_fake_megatron(monkeypatch)
        megatron_args.transformer_impl = "transformer_engine"
        _silence_log_rank_0(monkeypatch)
        model, optimizer, scheduler = _warmup_actors(monkeypatch)

        class _Handle:
            def __init__(self):
                self.waited = False

            def wait(self):
                self.waited = True

        handle = _Handle()
        group = SimpleNamespace(
            is_first_batch=False,
            golden_per_param_grad_ready_counts={},
            per_param_grad_ready_counts={},
            grad_reduce_handle=handle,
        )
        model[0].bucket_groups = [group]

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _run_warmup_and_restore,
        )

        _run_warmup_and_restore(
            warmup_steps=1,
            train_step_fn=lambda *a, **k: None,
            forward_step_func=lambda *a, **k: None,
            synthetic_iter=iter(()),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=SimpleNamespace(finalize_model_grads_func=None),
            forward_backward_func=lambda *a, **k: None,
            iteration=0,
        )

        assert handle.waited is True
        assert group.grad_reduce_handle is None


# ============================================================================
# Run lifecycle
# ============================================================================


def _patch_with_eval_loss(monkeypatch, loss):
    mt, megatron_args = _install_fake_megatron(monkeypatch)
    _silence_log_rank_0(monkeypatch)
    _, events = _install_mock_mllog(monkeypatch)

    mt.evaluate = lambda *a, **k: ({"loss": loss},)
    mt.evaluate_and_print_results = lambda *a, **k: mt.evaluate(*a, **k)

    from primus.backends.megatron.patches.mlperf_logging_patches import (
        patch_mlperf_logging,
    )

    megatron_args.train_iters = 5000
    patch_mlperf_logging(_make_ctx(mlperf_mode=True, target_val_loss=0.586))
    mt.build_train_valid_test_data_iterators(None)
    events.clear()
    return mt, megatron_args, events


def _run_eval(mt, iteration):
    mt.evaluate_and_print_results(
        f"iteration {iteration}",
        lambda: None,
        None,
        [MagicMock()],
        iteration,
        None,
        MagicMock(),
    )


class TestConvergenceDetection:
    """Verify convergence detection in evaluate_and_print_results wrapper."""

    def test_convergence_sets_train_iters(self, monkeypatch):
        mt, megatron_args, _ = _patch_with_eval_loss(monkeypatch, 0.500)
        _run_eval(mt, 512)
        assert megatron_args.train_iters == 512

    def test_convergence_clears_do_valid(self, monkeypatch):
        """Breaking the loop returns into pretrain, which validates once more.

        That evaluation is past run_stop and draws fresh noise, so it reports a
        different loss and can land above the target the run just met. do_valid
        is the flag pretrain branches on, so clearing it is what stops it.
        """
        mt, megatron_args, _ = _patch_with_eval_loss(monkeypatch, 0.500)
        assert megatron_args.do_valid is True

        _run_eval(mt, 512)

        assert megatron_args.do_valid is False

    def test_a_missed_target_leaves_do_valid_alone(self, monkeypatch):
        """A run still in progress must keep evaluating."""
        mt, megatron_args, _ = _patch_with_eval_loss(monkeypatch, 0.900)
        _run_eval(mt, 512)

        assert megatron_args.do_valid is True

    def test_convergence_emits_a_successful_run_stop(self, monkeypatch):
        mt, _, events = _patch_with_eval_loss(monkeypatch, 0.500)
        _run_eval(mt, 512)

        stops = [entry for entry in events if entry[1] == "run_stop"]
        assert len(stops) == 1
        assert stops[0][2] == "success"
        assert stops[0][3]["status"] == "success"

    def test_a_missed_target_reopens_the_block(self, monkeypatch):
        """block_stop fires on entry to eval; training resuming needs a new block."""
        mt, _, events = _patch_with_eval_loss(monkeypatch, 0.900)
        _run_eval(mt, 512)

        keys = _keys(events)
        assert keys.index("block_stop") < keys.index("eval_start")
        assert keys.index("eval_stop") < keys.index("block_start")
        assert "run_stop" not in keys

    def test_an_unreadable_loss_still_reopens_the_block(self, monkeypatch):
        mt, _, events = _patch_with_eval_loss(monkeypatch, 0.900)
        mt.evaluate = lambda *a, **k: ({},)
        events.clear()

        _run_eval(mt, 512)

        assert "block_start" in _keys(events)


class TestNothingIsLoggedAfterRunStop:
    """The submission log ends at run_stop, whoever evaluates afterwards.

    Clearing do_valid on convergence means pretrain's post-training validation
    never runs, so in practice nothing reaches the wrapper past run_stop. This
    guard is what makes that a property of the wrapper rather than of the order
    the callers happen to run in, which is why every test here calls the
    wrapper directly.
    """

    def test_an_evaluation_after_run_stop_emits_no_records(self, monkeypatch):
        mt, _, events = _patch_with_eval_loss(monkeypatch, 0.500)
        _run_eval(mt, 512)
        assert "run_stop" in _keys(events)
        events.clear()

        _run_eval(mt, 512)

        assert _keys(events) == []

    def test_a_second_run_stop_is_never_emitted(self, monkeypatch):
        """run_stop is EXACTLY_ONE, and the repeat evaluation also converges."""
        mt, _, events = _patch_with_eval_loss(monkeypatch, 0.500)
        _run_eval(mt, 512)

        _run_eval(mt, 512)

        assert _keys(events).count("run_stop") == 1

    def test_the_evaluation_itself_still_runs(self, monkeypatch):
        """The guard suppresses records, not the caller's evaluation."""
        mt, _, _ = _patch_with_eval_loss(monkeypatch, 0.500)
        _run_eval(mt, 512)

        calls = []
        mt.evaluate = lambda *a, **k: (calls.append(1), {"loss": 0.500})[1]

        _run_eval(mt, 512)

        assert calls == [1]

    def test_an_aborted_run_also_closes_the_log(self, monkeypatch):
        """converged would miss this; the guard keys on run_stop having fired."""
        mt, _, events = _patch_with_eval_loss(monkeypatch, 0.900)
        mt._primus_mlperf_logger.log_run_stop(success=False, global_step=5000)
        events.clear()

        _run_eval(mt, 5000)

        assert _keys(events) == []


class TestTerminalRunStop:
    """A run that never converges still has to produce a parseable log."""

    def _patch(self, monkeypatch):
        mt, megatron_args = _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _, events = _install_mock_mllog(monkeypatch)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        patch_mlperf_logging(_make_ctx(mlperf_mode=True))
        mt.build_train_valid_test_data_iterators(None)
        events.clear()
        return mt, megatron_args, events

    def test_exhausted_run_emits_an_aborted_run_stop(self, monkeypatch):
        mt, megatron_args, events = self._patch(monkeypatch)
        megatron_args.curr_iteration = 5000

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_terminal_run_stop,
        )

        patch_mlperf_terminal_run_stop(_make_ctx(mlperf_mode=True))

        stops = [entry for entry in events if entry[1] == "run_stop"]
        assert len(stops) == 1
        assert stops[0][2] == "aborted"
        assert stops[0][3]["status"] == "aborted"
        assert stops[0][3]["samples_count"] == 5000 * 512

    def test_a_converged_run_is_not_stopped_twice(self, monkeypatch):
        """run_stop is EXACTLY_ONE in the ruleset."""
        mt, megatron_args, events = self._patch(monkeypatch)

        mlperf_logger = mt._primus_mlperf_logger
        mlperf_logger.log_run_stop(success=True, global_step=512)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_terminal_run_stop,
        )

        patch_mlperf_terminal_run_stop(_make_ctx(mlperf_mode=True))

        assert _keys(events).count("run_stop") == 1


class TestHyperparameterRecords:
    """The values the closed_flux1 ruleset pins exactly."""

    def test_evaluation_frequency_is_reported_in_samples(self, monkeypatch):
        """closed_flux1.yaml requires evaluation_frequency == 262144."""
        _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _, events = _install_mock_mllog(monkeypatch)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        patch_mlperf_logging(_make_ctx(mlperf_mode=True, global_batch_size=512, eval_interval=512))

        frequency = [entry for entry in events if entry[1] == "evaluation_frequency"]
        assert len(frequency) == 1
        assert frequency[0][2] == 262144

    def test_config_filename_names_a_real_recipe(self, monkeypatch):
        _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _install_mock_mllog(monkeypatch)
        monkeypatch.delenv("EXP", raising=False)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        with pytest.raises(RuntimeError, match="EXP"):
            patch_mlperf_logging(_make_ctx(mlperf_mode=True))

    def test_cache_clear_is_reported_not_assumed(self, monkeypatch):
        _install_fake_megatron(monkeypatch)
        _silence_log_rank_0(monkeypatch)
        _install_mock_mllog(monkeypatch)
        monkeypatch.delenv("MLPERF_CLEAR_CACHES", raising=False)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        with pytest.raises(RuntimeError, match="MLPERF_CLEAR_CACHES"):
            patch_mlperf_logging(_make_ctx(mlperf_mode=True))


# ============================================================================
# Level 4: Helper function unit tests (CPU-only, no GPU required)
# ============================================================================


class TestResetFp8TeSpec:
    """Verify _reset_fp8_te_spec clears fp8_initialized and meta tensors."""

    def test_resets_fp8_initialized_flag(self):
        import torch

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_fp8_te_spec,
        )

        class FakeTeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fp8_initialized = True
                self.fp8_meta = {
                    "scaling_fwd": SimpleNamespace(
                        amax_history=torch.ones(16),
                        scale=torch.full((4,), 3.14),
                        scale_inv=torch.full((4,), 0.318),
                    ),
                    "scaling_bwd": SimpleNamespace(
                        amax_history=torch.ones(16),
                        scale=torch.full((4,), 2.71),
                        scale_inv=torch.full((4,), 0.369),
                    ),
                }

        model = torch.nn.Sequential(FakeTeModule(), FakeTeModule())
        count = _reset_fp8_te_spec([model])

        assert count == 2
        for module in model.modules():
            if hasattr(module, "fp8_initialized"):
                assert module.fp8_initialized is False
                for key in ("scaling_fwd", "scaling_bwd"):
                    tm = module.fp8_meta[key]
                    assert (tm.amax_history == 0.0).all()
                    assert (tm.scale == 1.0).all()
                    assert (tm.scale_inv == 1.0).all()

    def test_skips_reset_fp8_meta_tensors_shortcut(self):
        """
        TE 2.8.0.dev0's `reset_fp8_meta_tensors` unconditionally derefs `.scale`
        on the recipe state, which crashes on `Float8CurrentScalingRecipeState`
        (current/tensorwise scaling has no persistent state). The reset must
        therefore go through the recipe-agnostic manual path even when a TE
        module advertises the helper.
        """
        from types import SimpleNamespace

        import torch

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_fp8_te_spec,
        )

        reset_called = [False]

        class FakeTeCurrentScalingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fp8_initialized = True
                # Mimic Float8CurrentScalingRecipeState: no .scale, no .amax_history.
                self.fp8_meta = {
                    "scaling_fwd": SimpleNamespace(),
                    "scaling_bwd": SimpleNamespace(),
                }

            def reset_fp8_meta_tensors(self):
                reset_called[0] = True

        model = torch.nn.Sequential(FakeTeCurrentScalingModule())
        count = _reset_fp8_te_spec([model])

        assert not reset_called[
            0
        ], "reset_fp8_meta_tensors must NOT be called (would crash on current scaling)"
        assert count == 1
        assert model[0].fp8_initialized is False

    def test_falls_through_for_delayed_scaling_buffers(self):
        """
        Companion to ``test_skips_reset_fp8_meta_tensors_shortcut``: confirms
        the manual fallback still resets delayed-scaling buffers (.scale,
        .amax_history, .scale_inv) when they exist on the recipe state.
        """
        from types import SimpleNamespace

        import torch

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_fp8_te_spec,
        )

        reset_called = [False]

        class FakeTeDelayedScalingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fp8_initialized = True
                self.fp8_meta = {
                    "scaling_fwd": SimpleNamespace(
                        amax_history=torch.full((16,), 7.0),
                        scale=torch.full((4,), 2.71),
                        scale_inv=torch.full((4,), 0.369),
                    ),
                    "scaling_bwd": SimpleNamespace(
                        amax_history=torch.full((16,), 7.0),
                        scale=torch.full((4,), 2.71),
                        scale_inv=torch.full((4,), 0.369),
                    ),
                }

            def reset_fp8_meta_tensors(self):
                reset_called[0] = True

        model = torch.nn.Sequential(FakeTeDelayedScalingModule())
        count = _reset_fp8_te_spec([model])

        assert not reset_called[0]
        assert count == 1
        for key in ("scaling_fwd", "scaling_bwd"):
            tm = model[0].fp8_meta[key]
            assert (tm.amax_history == 0.0).all()
            assert (tm.scale == 1.0).all()
            assert (tm.scale_inv == 1.0).all()


class TestSeedFp8Amax:
    """Verify _seed_fp8_amax fills amax_history with the requested seed value."""

    def _make_te_model(self):
        import torch

        class FakeTeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fp8_meta = {
                    "scaling_fwd": SimpleNamespace(
                        amax_history=torch.zeros(16),
                    ),
                    "scaling_bwd": SimpleNamespace(
                        amax_history=torch.zeros(16),
                    ),
                }

        return torch.nn.Sequential(FakeTeModule(), FakeTeModule())

    def test_seeds_default_value(self):
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _seed_fp8_amax,
        )

        model = self._make_te_model()
        count = _seed_fp8_amax([model])

        assert count == 4
        for module in model.modules():
            if hasattr(module, "fp8_meta"):
                for key in ("scaling_fwd", "scaling_bwd"):
                    assert (module.fp8_meta[key].amax_history == 1.0).all()

    def test_seeds_custom_value(self):
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _seed_fp8_amax,
        )

        model = self._make_te_model()
        count = _seed_fp8_amax([model], seed_value=42.0)

        assert count == 4
        for module in model.modules():
            if hasattr(module, "fp8_meta"):
                for key in ("scaling_fwd", "scaling_bwd"):
                    assert (module.fp8_meta[key].amax_history == 42.0).all()


class TestNeuterRestoreOptimizer:
    """Verify _neuter_optimizer / _restore_optimizer roundtrip."""

    def test_roundtrip_preserves_hyperparams(self):
        import torch

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _neuter_optimizer,
            _restore_optimizer,
        )

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

        orig_betas = list(opt.param_groups[0]["betas"])
        orig_wd = opt.param_groups[0]["weight_decay"]

        wrapper = SimpleNamespace(optimizer=opt)
        saved = _neuter_optimizer(wrapper)

        assert opt.param_groups[0]["betas"] == [1.0, 1.0]
        assert opt.param_groups[0]["weight_decay"] == 0.0

        _restore_optimizer(wrapper, saved)

        assert list(opt.param_groups[0]["betas"]) == orig_betas
        assert opt.param_groups[0]["weight_decay"] == orig_wd

    def test_roundtrip_all_keys(self):
        """All 4 production keys roundtrip: betas, weight_decay, bias_correction, pre_mult_wd."""
        import torch

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _neuter_optimizer,
            _restore_optimizer,
        )

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        opt.param_groups[0]["betas"] = [0.9, 0.999]
        opt.param_groups[0]["weight_decay"] = 0.01
        opt.param_groups[0]["bias_correction"] = True
        opt.param_groups[0]["pre_mult_wd"] = 0.05

        wrapper = SimpleNamespace(optimizer=opt)
        saved = _neuter_optimizer(wrapper)

        assert opt.param_groups[0]["betas"] == [1.0, 1.0]
        assert opt.param_groups[0]["weight_decay"] == 0.0
        assert opt.param_groups[0]["bias_correction"] is False
        assert opt.param_groups[0]["pre_mult_wd"] == 0.0

        _restore_optimizer(wrapper, saved)

        assert opt.param_groups[0]["betas"] == [0.9, 0.999]
        assert opt.param_groups[0]["weight_decay"] == 0.01
        assert opt.param_groups[0]["bias_correction"] is True
        assert opt.param_groups[0]["pre_mult_wd"] == 0.05

    def test_multi_param_group_roundtrip(self):
        """Neuter/restore with 2 param groups preserves per-group values."""
        import torch

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _neuter_optimizer,
            _restore_optimizer,
        )

        model = torch.nn.Linear(4, 4)
        w_params = [model.weight]
        b_params = [model.bias]
        opt = torch.optim.Adam(
            [
                {"params": w_params, "weight_decay": 0.1},
                {"params": b_params, "weight_decay": 0.0},
            ],
            lr=1e-3,
            betas=(0.9, 0.999),
        )

        wrapper = SimpleNamespace(optimizer=opt)
        saved = _neuter_optimizer(wrapper)

        for g in opt.param_groups:
            assert g["betas"] == [1.0, 1.0]
            assert g["weight_decay"] == 0.0

        _restore_optimizer(wrapper, saved)

        assert opt.param_groups[0]["weight_decay"] == 0.1
        assert opt.param_groups[1]["weight_decay"] == 0.0
        for g in opt.param_groups:
            assert list(g["betas"]) == [0.9, 0.999]


class TestResetOptimizerState:
    """Verify _reset_optimizer_state clears step counters."""

    def test_resets_param_group_step(self):
        import torch

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_optimizer_state,
        )

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()

        opt.param_groups[0]["step"] = 42
        wrapper = SimpleNamespace(optimizer=opt)

        _reset_optimizer_state(wrapper)

        assert opt.param_groups[0]["step"] == 0

    def test_resets_per_param_state_step_tensor(self):
        import torch

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_optimizer_state,
        )

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()

        for state in opt.state.values():
            assert state["step"].item() > 0

        wrapper = SimpleNamespace(optimizer=opt)
        _reset_optimizer_state(wrapper)

        for state in opt.state.values():
            assert state["step"].item() == 0

    def test_handles_chained_optimizer(self):
        import torch

        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_optimizer_state,
        )

        m1 = torch.nn.Linear(4, 4)
        m2 = torch.nn.Linear(4, 4)
        opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)
        opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)

        opt1.param_groups[0]["step"] = 10
        opt2.param_groups[0]["step"] = 20

        w1 = SimpleNamespace(optimizer=opt1)
        w2 = SimpleNamespace(optimizer=opt2)
        chained = SimpleNamespace(chained_optimizers=[w1, w2])

        _reset_optimizer_state(chained)

        assert opt1.param_groups[0]["step"] == 0
        assert opt2.param_groups[0]["step"] == 0
