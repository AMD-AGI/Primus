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


# ============================================================================
# Level 1: Patch registration and conditions
# ============================================================================


class TestLoggingPatchMonkeyPatching:
    """Verify that the logging patch replaces the expected functions."""

    def test_installs_wrappers(self, monkeypatch):
        mt, _ = _install_fake_megatron(monkeypatch)

        monkeypatch.setattr(
            "primus.backends.megatron.patches.mlperf_logging_patches.log_rank_0",
            lambda *a, **k: None,
        )

        mock_mllog = MagicMock()
        mock_mllog.get_mllogger.return_value = MagicMock()
        mock_mllog.constants = SimpleNamespace(
            INIT_START="init_start",
            INIT_STOP="init_stop",
            RUN_START="run_start",
            RUN_STOP="run_stop",
            SUBMISSION_BENCHMARK="submission_benchmark",
            SUBMISSION_ORG="submission_org",
            SUBMISSION_DIVISION="submission_division",
            SUBMISSION_PLATFORM="submission_platform",
            SUBMISSION_STATUS="submission_status",
            SEED="seed",
            GLOBAL_BATCH_SIZE="global_batch_size",
            TRAIN_SAMPLES="train_samples",
            EVAL_SAMPLES="eval_samples",
            GRADIENT_ACCUMULATION_STEPS="gradient_accumulation_steps",
            OPT_NAME="opt_name",
            OPT_BASE_LR="opt_base_lr",
            EVAL_ACCURACY="eval_accuracy",
            EVAL_START="eval_start",
            EVAL_STOP="eval_stop",
            EPOCH_START="epoch_start",
            BLOCK_START="block_start",
            BLOCK_STOP="block_stop",
        )
        mock_mlperf_pkg = MagicMock()
        mock_mlperf_pkg.mllog = mock_mllog
        monkeypatch.setitem(__import__("sys").modules, "mlperf_logging", mock_mlperf_pkg)
        monkeypatch.setitem(__import__("sys").modules, "mlperf_logging.mllog", mock_mllog)

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        original_tl = mt.training_log
        original_eval = mt.evaluate_and_print_results
        original_prl = mt.print_rank_last

        ctx = _make_ctx(mlperf_mode=True)
        patch_mlperf_logging(ctx)

        assert mt.training_log is not original_tl
        assert mt.evaluate_and_print_results is not original_eval
        assert mt.print_rank_last is not original_prl
        assert getattr(mt, "_primus_mlperf_logging_installed", False) is True

    def test_idempotent(self, monkeypatch):
        mt, _ = _install_fake_megatron(monkeypatch)

        monkeypatch.setattr(
            "primus.backends.megatron.patches.mlperf_logging_patches.log_rank_0",
            lambda *a, **k: None,
        )

        mock_mllog = MagicMock()
        mock_mllog.get_mllogger.return_value = MagicMock()
        mock_mllog.constants = SimpleNamespace(
            INIT_START="init_start",
            INIT_STOP="init_stop",
            RUN_START="run_start",
            RUN_STOP="run_stop",
            SUBMISSION_BENCHMARK="submission_benchmark",
            SUBMISSION_ORG="submission_org",
            SUBMISSION_DIVISION="submission_division",
            SUBMISSION_PLATFORM="submission_platform",
            SUBMISSION_STATUS="submission_status",
            SEED="seed",
            GLOBAL_BATCH_SIZE="global_batch_size",
            TRAIN_SAMPLES="train_samples",
            EVAL_SAMPLES="eval_samples",
            GRADIENT_ACCUMULATION_STEPS="gradient_accumulation_steps",
            OPT_NAME="opt_name",
            OPT_BASE_LR="opt_base_lr",
            EVAL_ACCURACY="eval_accuracy",
            EVAL_START="eval_start",
            EVAL_STOP="eval_stop",
            EPOCH_START="epoch_start",
            BLOCK_START="block_start",
            BLOCK_STOP="block_stop",
        )
        mock_mlperf_pkg = MagicMock()
        mock_mlperf_pkg.mllog = mock_mllog
        monkeypatch.setitem(__import__("sys").modules, "mlperf_logging", mock_mlperf_pkg)
        monkeypatch.setitem(__import__("sys").modules, "mlperf_logging.mllog", mock_mllog)

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


class TestTrainingLogFirstCall:
    """Verify INIT_STOP + RUN_START fire on first post-warmup training_log."""

    def test_first_call_emits_init_stop_run_start(self, monkeypatch):
        mt, args = _install_fake_megatron(monkeypatch)

        monkeypatch.setattr(
            "primus.backends.megatron.patches.mlperf_logging_patches.log_rank_0",
            lambda *a, **k: None,
        )

        emitted_events = []

        mock_mllogger = MagicMock()
        mock_constants = SimpleNamespace(
            INIT_START="init_start",
            INIT_STOP="init_stop",
            RUN_START="run_start",
            RUN_STOP="run_stop",
            SUBMISSION_BENCHMARK="submission_benchmark",
            SUBMISSION_ORG="submission_org",
            SUBMISSION_DIVISION="submission_division",
            SUBMISSION_PLATFORM="submission_platform",
            SUBMISSION_STATUS="submission_status",
            SEED="seed",
            GLOBAL_BATCH_SIZE="global_batch_size",
            TRAIN_SAMPLES="train_samples",
            EVAL_SAMPLES="eval_samples",
            GRADIENT_ACCUMULATION_STEPS="gradient_accumulation_steps",
            OPT_NAME="opt_name",
            OPT_BASE_LR="opt_base_lr",
            EVAL_ACCURACY="eval_accuracy",
            EVAL_START="eval_start",
            EVAL_STOP="eval_stop",
            EPOCH_START="epoch_start",
            BLOCK_START="block_start",
            BLOCK_STOP="block_stop",
        )

        def track_start(key, value=None, metadata=None):
            emitted_events.append(("start", key))

        def track_end(key, value=None, metadata=None):
            emitted_events.append(("end", key))

        def track_event(key, value=None, metadata=None):
            emitted_events.append(("event", key))

        mock_mllogger.start = track_start
        mock_mllogger.end = track_end
        mock_mllogger.event = track_event

        mock_mllog_module = MagicMock()
        mock_mllog_module.get_mllogger.return_value = mock_mllogger
        mock_mllog_module.constants = mock_constants

        # Wire the top-level mlperf_logging mock so that
        # `from mlperf_logging import mllog` resolves correctly
        mock_mlperf_pkg = MagicMock()
        mock_mlperf_pkg.mllog = mock_mllog_module

        import sys as _sys

        monkeypatch.setitem(_sys.modules, "mlperf_logging", mock_mlperf_pkg)
        monkeypatch.setitem(_sys.modules, "mlperf_logging.mllog", mock_mllog_module)
        monkeypatch.setenv("RANK", "0")

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        ctx = _make_ctx(mlperf_mode=True, log_interval=1)
        patch_mlperf_logging(ctx)

        emitted_events.clear()

        # First call should emit INIT_STOP + RUN_START
        mt.training_log({"loss": 0.5}, {}, 1e-4, 1, 1.0, False, False, 0.0, None, 0, None)

        event_keys = [e[1] for e in emitted_events]
        assert "init_stop" in event_keys, f"INIT_STOP not emitted. Events: {emitted_events}"
        assert "run_start" in event_keys, f"RUN_START not emitted. Events: {emitted_events}"

        init_stop_idx = event_keys.index("init_stop")
        run_start_idx = event_keys.index("run_start")
        assert init_stop_idx < run_start_idx

        # Second call should NOT emit INIT_STOP/RUN_START again
        emitted_events.clear()
        mt.training_log({"loss": 0.4}, {}, 1e-4, 2, 1.0, False, False, 0.0, None, 0, None)

        event_keys_2 = [e[1] for e in emitted_events]
        assert "init_stop" not in event_keys_2
        assert "run_start" not in event_keys_2


# ============================================================================
# Level 3: Component tests
# ============================================================================


class TestConvergenceDetection:
    """Verify convergence detection in evaluate_and_print_results wrapper."""

    def test_convergence_sets_train_iters(self, monkeypatch):
        mt, megatron_args = _install_fake_megatron(monkeypatch)

        monkeypatch.setattr(
            "primus.backends.megatron.patches.mlperf_logging_patches.log_rank_0",
            lambda *a, **k: None,
        )

        # Mock mlperf_logging — wire .mllog attribute on the top-level package
        mock_mllogger = MagicMock()
        mock_constants = SimpleNamespace(
            INIT_START="init_start",
            INIT_STOP="init_stop",
            RUN_START="run_start",
            RUN_STOP="run_stop",
            SUBMISSION_BENCHMARK="submission_benchmark",
            SUBMISSION_ORG="submission_org",
            SUBMISSION_DIVISION="submission_division",
            SUBMISSION_PLATFORM="submission_platform",
            SUBMISSION_STATUS="submission_status",
            SEED="seed",
            GLOBAL_BATCH_SIZE="global_batch_size",
            TRAIN_SAMPLES="train_samples",
            EVAL_SAMPLES="eval_samples",
            GRADIENT_ACCUMULATION_STEPS="gradient_accumulation_steps",
            OPT_NAME="opt_name",
            OPT_BASE_LR="opt_base_lr",
            EVAL_ACCURACY="eval_accuracy",
            EVAL_START="eval_start",
            EVAL_STOP="eval_stop",
            EPOCH_START="epoch_start",
            BLOCK_START="block_start",
            BLOCK_STOP="block_stop",
        )
        mock_mllogger.start = MagicMock()
        mock_mllogger.end = MagicMock()
        mock_mllogger.event = MagicMock()

        mock_mllog_module = MagicMock()
        mock_mllog_module.get_mllogger.return_value = mock_mllogger
        mock_mllog_module.constants = mock_constants

        mock_mlperf_pkg = MagicMock()
        mock_mlperf_pkg.mllog = mock_mllog_module

        import sys as _sys

        monkeypatch.setitem(_sys.modules, "mlperf_logging", mock_mlperf_pkg)
        monkeypatch.setitem(_sys.modules, "mlperf_logging.mllog", mock_mllog_module)
        monkeypatch.setenv("RANK", "0")

        # Set evaluate to return a loss below target BEFORE patching.
        # Also make evaluate_and_print_results call evaluate() internally,
        # mirroring real Megatron behavior so _captured_loss gets populated.
        mt.evaluate = lambda *a, **k: ({"loss": 0.500},)

        def fake_eval_and_print(*a, **k):
            mt.evaluate(*a, **k)

        mt.evaluate_and_print_results = fake_eval_and_print

        from primus.backends.megatron.patches.mlperf_logging_patches import (
            patch_mlperf_logging,
        )

        target_val_loss = 0.586
        megatron_args.train_iters = 5000

        ctx = _make_ctx(mlperf_mode=True, target_val_loss=target_val_loss)
        patch_mlperf_logging(ctx)

        # Call eval at iteration 512
        mt.evaluate_and_print_results(
            "iteration 512",
            lambda: None,
            None,
            [MagicMock()],
            512,
            None,
            MagicMock(),
        )

        assert megatron_args.train_iters == 512


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
