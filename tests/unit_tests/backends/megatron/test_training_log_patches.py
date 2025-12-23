###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import types
from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches import training_log_patches as tl_patches


def _install_fake_megatron_training(monkeypatch: pytest.MonkeyPatch):
    """
    Install a fake `megatron.training.training` module into sys.modules so that
    training_log_patches can import and patch it.
    """
    import sys

    megatron_mod = types.ModuleType("megatron")
    training_pkg = types.ModuleType("megatron.training")
    training_mod = types.ModuleType("megatron.training.training")

    def fake_training_log(*args, **kwargs):
        return "ok"

    # Provide a minimal `get_model` stub so that any code which expects
    # `megatron.training.training.get_model` (e.g., Primus monkey patches in
    # `primus.pretrain.load_backend_trainer`) can safely import and patch this
    # fake training module during tests without raising AttributeError.
    def fake_get_model(*args, **kwargs):
        return None

    training_mod.training_log = fake_training_log
    training_mod.get_model = fake_get_model

    # Wire the package hierarchy: megatron.training.training
    training_pkg.training = training_mod
    megatron_mod.training = training_pkg

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.training"] = training_pkg
    sys.modules["megatron.training.training"] = training_mod

    return training_mod, fake_training_log


def _make_ctx(args=None, config=None, module_config=None):
    """
    Build a minimal PatchContext-like object for direct calls to patch functions.

    The current Megatron training_log patch expects:
        - ctx.extra["backend_args"] for runtime args
        - ctx.extra["module_config"] with a .params attribute for config

    In real usage, `module_config.params` is a SimpleNamespace-like object, not a
    raw dict, so we wrap dict configs into a SimpleNamespace for tests.
    """
    # Normalize module_config to have a .params attribute, matching real usage.
    if module_config is None and config is not None:
        if isinstance(config, dict):
            params = SimpleNamespace(**config)
        else:
            params = config
        module_config = SimpleNamespace(params=params)

    extra = {}
    if args is not None:
        extra["backend_args"] = args
    if module_config is not None:
        extra["module_config"] = module_config
    return SimpleNamespace(extra=extra)


def test_patch_training_log_skips_when_no_extensions(monkeypatch: pytest.MonkeyPatch):
    training_mod, original_fn = _install_fake_megatron_training(monkeypatch)

    # Disable ROCm stats via args/config so no extensions are created.
    args = SimpleNamespace(log_throughput=False)
    config = {}
    ctx = _make_ctx(args=args, config=config)

    # Patch log_rank_0 to avoid real logging.
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log_patches.log_rank_0",
        lambda *a, **k: None,
    )

    tl_patches.patch_training_log_unified(ctx)

    # training_log should remain the original function and not be wrapped.
    assert training_mod.training_log is original_fn
    assert not getattr(training_mod.training_log, "_primus_training_log_wrapper", False)


def test_patch_training_log_wraps_and_stacks_extensions(monkeypatch: pytest.MonkeyPatch):
    training_mod, original_fn = _install_fake_megatron_training(monkeypatch)

    # Enable ROCm stats so that a RocmMonitorExtension is created.
    args = SimpleNamespace(log_throughput=True)
    config = {"use_rocm_mem_info": True}
    ctx = _make_ctx(args=args, config=config)

    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log_patches.log_rank_0",
        lambda *a, **k: None,
    )

    # First application should wrap training_log once with one extension.
    tl_patches.patch_training_log_unified(ctx)
    wrapped_fn = training_mod.training_log

    assert wrapped_fn is not original_fn
    assert getattr(wrapped_fn, "_primus_training_log_wrapper", False)

    # Metadata should point back to original function and include one extension.
    orig = getattr(wrapped_fn, "_primus_original_training_log")
    exts = getattr(wrapped_fn, "_primus_extensions")
    assert orig is original_fn
    assert len(exts) == 1

    # Second application with the same config should stack another extension.
    tl_patches.patch_training_log_unified(ctx)
    wrapped_fn_2 = training_mod.training_log

    assert wrapped_fn_2 is not original_fn
    assert getattr(wrapped_fn_2, "_primus_training_log_wrapper", False)

    exts2 = getattr(wrapped_fn_2, "_primus_extensions")
    assert len(exts2) == 2


def test_rocm_monitor_hooked_print_rank_last_injects_stats(monkeypatch: pytest.MonkeyPatch):
    # Prepare fake torch and ROCm SMI helpers.
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            mem_get_info=lambda: (2 * 1024**3, 4 * 1024**3),
            current_device=lambda: 0,
        )
    )
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log_patches.torch",
        fake_torch,
    )

    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log_patches.get_rocm_smi_mem_info",
        lambda rank: (8 * 1024**3, 6 * 1024**3, 2 * 1024**3),
    )

    captured = []
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log_patches.log_rank_all",
        lambda msg, *a, **k: captured.append(msg),
    )

    # Enable both HIP and ROCm SMI stats.
    ext = tl_patches.RocmMonitorExtension(
        args=SimpleNamespace(),
        config={"use_rocm_mem_info": True, "use_rocm_mem_info_iters": []},
    )

    # Directly call hooked print; should not raise and should append stats.
    ext._hooked_print_rank_last("iter 1:")

    assert len(captured) == 1
    out = captured[0]
    assert "iter 1:" in out
    # Basic sanity checks that memory substrings are present.
    assert "hip mem usage/free/total/usage_ratio" in out
    assert "rocm mem usage/free/total/usage_ratio" in out


def test_rocm_monitor_hooked_print_rank_last_swallows_errors(monkeypatch: pytest.MonkeyPatch):
    # Force mem_get_info to raise and ensure we still log something.
    def _raise_oom():
        raise RuntimeError("OOM probe failed")

    def _raise_smi():
        raise RuntimeError("SMI failed")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            mem_get_info=_raise_oom,
            current_device=lambda: 0,
        )
    )
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log_patches.torch",
        fake_torch,
    )
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log_patches.get_rocm_smi_mem_info",
        lambda rank: _raise_smi(),
    )

    captured = []
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log_patches.log_rank_all",
        lambda msg, *a, **k: captured.append(msg),
    )

    ext = tl_patches.RocmMonitorExtension(
        args=SimpleNamespace(),
        config={"use_rocm_mem_info": True, "use_rocm_mem_info_iters": []},
    )

    # Should not raise despite internal errors and still call log_rank_all.
    ext._hooked_print_rank_last("iter 2:")

    assert len(captured) == 1
    assert captured[0].startswith("iter 2:")
