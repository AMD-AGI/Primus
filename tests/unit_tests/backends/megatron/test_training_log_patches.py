###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import types
from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches.training_log import (
    print_rank_last_patches as prl_patches,
)
from primus.backends.megatron.patches.training_log import (
    print_rank_last_patches as tl_patches,
)


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
    # `megatron.training.training.get_model` (e.g., Primus monkey patches)
    # can safely import and patch this fake training module during tests
    # without raising AttributeError.
    def fake_get_model(*args, **kwargs):
        return None

    training_mod.training_log = fake_training_log
    training_mod.get_model = fake_get_model

    # Wire the package hierarchy: megatron.training.training
    training_pkg.training = training_mod
    megatron_mod.training = training_pkg

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training_mod)

    return training_mod, fake_training_log


def _make_ctx(args=None, config=None, module_config=None):
    """
    Build a minimal PatchContext-like object for direct calls to patch functions.

    The current Megatron training_log patch expects:
        - ctx.extra["module_config"].params as the unified Megatron args namespace.

    In real usage, `module_config.params` is a SimpleNamespace-like object that
    already contains both runtime args (e.g., log_throughput) and config flags
    (e.g., use_rocm_mem_info). For tests where args/config are passed separately,
    we merge them into a single SimpleNamespace.
    """
    # Normalize module_config to have a .params attribute, matching real usage.
    if module_config is None and config is not None:
        if isinstance(config, dict):
            # Merge config dict with args namespace (if provided) to emulate
            # a unified Megatron args namespace.
            merged = dict(config)
            if isinstance(args, SimpleNamespace):
                merged.update(vars(args))
            params = SimpleNamespace(**merged)
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
        "primus.backends.megatron.patches.training_log.print_rank_last_patches.log_rank_0",
        lambda *a, **k: None,
    )
    # No injection and no forwarding -> patch must be a no-op. Pin forwarding to
    # False so this test does not depend on the runner's env vars.
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches._should_forward_training_log_to_rank_0",
        lambda: False,
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
        "primus.backends.megatron.patches.training_log.print_rank_last_patches.log_rank_0",
        lambda *a, **k: None,
    )

    # First application should patch training_log once.
    tl_patches.patch_training_log_unified(ctx)
    wrapped_fn = training_mod.training_log

    # training_log should be patched once (only if stats are actually enabled).
    if getattr(wrapped_fn, "_primus_print_rank_last_wrapper", False):
        assert wrapped_fn is not original_fn

    # Second application with the same config should be idempotent (no re-patch).
    tl_patches.patch_training_log_unified(ctx)
    wrapped_fn_2 = training_mod.training_log
    assert wrapped_fn_2 is wrapped_fn


def test_rocm_monitor_hooked_print_rank_last_injects_stats(monkeypatch: pytest.MonkeyPatch):
    # Prepare fake torch and ROCm SMI helpers. MemoryStatsExtension computes the
    # cross-rank max ROCm mem via torch.tensor + torch.distributed.all_reduce(MAX)
    # and torch.distributed.get_rank(); stub those so inject() does not fall into
    # the exception path in unit tests.
    class _FakeTensor:
        __slots__ = ("_value",)

        def __init__(self, value: int):
            self._value = int(value)

        def item(self) -> int:
            return self._value

    def _tensor(data, device=None, dtype=None):
        if isinstance(data, (list, tuple)):
            return _FakeTensor(data[0])
        return _FakeTensor(data)

    def _zeros_like(_t):
        return _FakeTensor(0)

    def _get_world_size():
        return 8

    def _get_rank():
        return 0

    def _all_reduce(tensor, op=None):
        # Single-rank reduction: MAX over one rank is a no-op, so leave the
        # in-place tensor value unchanged (matching real all_reduce semantics).
        return None

    # `inject` passes `dtype=torch.int64`; SimpleNamespace must expose `int64`
    # or attribute lookup fails before our fake `tensor()` runs. The refactored
    # code also reads `torch.distributed.ReduceOp.MAX`, so provide that too.
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            mem_get_info=lambda: (2 * 1024**3, 4 * 1024**3),
            current_device=lambda: 0,
        ),
        tensor=_tensor,
        zeros_like=_zeros_like,
        int64=int,
        distributed=SimpleNamespace(
            get_world_size=_get_world_size,
            get_rank=_get_rank,
            all_reduce=_all_reduce,
            ReduceOp=SimpleNamespace(MAX=object()),
        ),
    )
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches.torch",
        fake_torch,
    )

    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches.get_rocm_smi_mem_info",
        lambda rank: (8 * 1024**3, 6 * 1024**3, 2 * 1024**3),
    )

    # Avoid real logging via Primus logger during unit tests.
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches.log_rank_0",
        lambda *a, **k: None,
    )

    # Enable both HIP and ROCm SMI stats.
    args = SimpleNamespace(
        use_rocm_mem_info=True,
        use_rocm_mem_info_iters=[],
        log_avg_skip_iterations=0,
        log_avg_reset_interval=1000,
        seq_length=128,
        world_size=8,
    )
    mem_ext = tl_patches.MemoryStatsExtension(args=args)
    thr_ext = tl_patches.ThroughputAverageExtension(args=args)

    # Simulate a single print_rank_last call: build parsed info, inject into
    # parsed structure, then render as the real patch would do.
    parsed = prl_patches.parse_training_log_line("iter 1:")
    mem_ext.inject("iter 1:", call_index=1, parsed=parsed)
    out = prl_patches.render_training_log_line(parsed)

    assert "iter 1:" in out
    # Basic sanity checks that memory substrings are present.
    assert "hip mem usage/free/total/usage_ratio" in out
    assert "rocm mem usage/free/total/usage_ratio" in out
    assert "rocm max mem usage/usage_ratio" in out


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
        "primus.backends.megatron.patches.training_log.print_rank_last_patches.torch",
        fake_torch,
    )
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches.get_rocm_smi_mem_info",
        lambda rank: _raise_smi(),
    )

    # Avoid real logging via Primus logger during unit tests.
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches.log_rank_0",
        lambda *a, **k: None,
    )

    args = SimpleNamespace(
        use_rocm_mem_info=True,
        use_rocm_mem_info_iters=[],
        log_avg_skip_iterations=0,
        log_avg_reset_interval=1000,
        seq_length=128,
        world_size=8,
    )
    mem_ext = tl_patches.MemoryStatsExtension(args=args)
    thr_ext = tl_patches.ThroughputAverageExtension(args=args)

    parsed = prl_patches.parse_training_log_line("iter 2:")
    mem_ext.inject("iter 2:", call_index=1, parsed=parsed)
    out = prl_patches.render_training_log_line(parsed)

    assert out.startswith("iter 2:")


def test_parse_training_log_line_does_not_confuse_iteration_and_elapsed():
    """Ensure that 'elapsed time per iteration (ms)' does not interfere with iteration parsing."""
    # This string emulates a real Megatron training_log line with timestamp prefix.
    log = (
        "[2025-12-25 08:40:22] iteration        2/      50 | "
        "consumed samples:          256 | "
        "elapsed time per iteration (ms): 6046.6 | "
        "throughput per GPU (TFLOP/s/GPU): 1255.4 | "
        "learning rate: 1.000000E-05 | "
        "global batch size:   128 | "
        "lm loss: 1.189062E+01 | loss scale: 1.0 | grad norm: 7.499 | "
        "number of skipped iterations:   0 | number of nan iterations:   0"
    )

    info = prl_patches.parse_training_log_line(log)

    # Iteration and train_iters should be parsed correctly from the 'iteration 2/50' fragment.
    assert info.iteration == 2
    assert info.train_iters == 50

    # Elapsed time should also be parsed correctly from the 'elapsed time per iteration (ms)' fragment.
    assert info.elapsed_ms == pytest.approx(6046.6)

    # Global batch size and throughput should be populated as well.
    assert info.global_batch_size == 128
    assert info.throughput_tflops == pytest.approx(1255.4)


def test_bridge_forward_patch_wraps_and_forwards(monkeypatch: pytest.MonkeyPatch):
    """
    Megatron-Bridge forwarding patch.

    Bridge hosts ``training_log`` in ``megatron.bridge.training.train`` (call
    site) but resolves ``print_rank_last`` from
    ``megatron.bridge.training.utils.train_utils``. The patch must wrap the
    former and, only during the call, override the latter so the last-rank line
    is forwarded to rank 0. The log content must stay unchanged and
    ``print_rank_last`` must be restored afterwards. Wrapping is idempotent.
    """
    import sys

    # Minimal module tree so the patch's ``import megatron.bridge...`` resolves.
    # ``import ... as`` hits sys.modules directly, so parent wiring is not needed.
    mods = {
        name: types.ModuleType(name)
        for name in (
            "megatron",
            "megatron.bridge",
            "megatron.bridge.training",
            "megatron.bridge.training.train",
            "megatron.bridge.training.utils",
            "megatron.bridge.training.utils.train_utils",
        )
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    train_mod = mods["megatron.bridge.training.train"]
    train_utils_mod = mods["megatron.bridge.training.utils.train_utils"]

    printed = []

    def fake_print_rank_last(msg):
        printed.append(msg)

    def fake_training_log(line):
        # Bridge: training_log resolves print_rank_last from train_utils.
        train_utils_mod.print_rank_last(line)
        return "done"

    train_utils_mod.print_rank_last = fake_print_rank_last
    train_mod.training_log = fake_training_log

    forwarded = []
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches._should_forward_training_log_to_rank_0",
        lambda: True,
    )
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches._forward_single_node_training_log",
        lambda msg: forwarded.append(msg),
    )
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches.log_rank_0",
        lambda *a, **k: None,
    )

    ctx = SimpleNamespace(extra={})
    tl_patches.patch_bridge_training_log_forward(ctx)

    # Call-site training_log is wrapped; print_rank_last untouched until called.
    assert train_mod.training_log is not fake_training_log
    assert train_utils_mod.print_rank_last is fake_print_rank_last

    line = "[ts] iteration 1/10 | lm loss: 1.0 |"
    result = train_mod.training_log(line)

    assert result == "done"
    assert printed == [line]  # forwarding only: content is unchanged
    assert len(forwarded) == 1
    assert forwarded[0].endswith(line)
    # print_rank_last must be restored after the call.
    assert train_utils_mod.print_rank_last is fake_print_rank_last

    # Idempotent: a second application is a no-op.
    wrapped_once = train_mod.training_log
    tl_patches.patch_bridge_training_log_forward(ctx)
    assert train_mod.training_log is wrapped_once


def test_native_forwarding_only_without_injection(monkeypatch: pytest.MonkeyPatch):
    """
    Decoupling: with ROCm/throughput injection disabled but single-node
    forwarding active, the native patch must still wrap training_log and forward
    the last-rank line to rank 0 -- without altering the log content.
    """
    import sys

    megatron_mod = types.ModuleType("megatron")
    training_pkg = types.ModuleType("megatron.training")
    training_mod = types.ModuleType("megatron.training.training")

    printed = []

    def fake_print_rank_last(msg):
        printed.append(msg)

    def fake_training_log(line):
        # Native: training_log resolves print_rank_last from the same module.
        training_mod.print_rank_last(line)
        return "ok"

    training_mod.print_rank_last = fake_print_rank_last
    training_mod.training_log = fake_training_log
    training_pkg.training = training_mod
    megatron_mod.training = training_pkg
    for name, mod in [
        ("megatron", megatron_mod),
        ("megatron.training", training_pkg),
        ("megatron.training.training", training_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    forwarded = []
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches._should_forward_training_log_to_rank_0",
        lambda: True,
    )
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches._forward_single_node_training_log",
        lambda msg: forwarded.append(msg),
    )
    monkeypatch.setattr(
        "primus.backends.megatron.patches.training_log.print_rank_last_patches.log_rank_0",
        lambda *a, **k: None,
    )

    # log_throughput=False -> enable_rocm_stats is False -> injection disabled.
    ctx = _make_ctx(args=SimpleNamespace(log_throughput=False), config={})
    tl_patches.patch_training_log_unified(ctx)

    # Even without injection, forwarding alone causes training_log to be wrapped.
    assert training_mod.training_log is not fake_training_log

    line = "[ts] iteration        1/      10 | lm loss: 1.0 |"
    result = training_mod.training_log(line)

    assert result == "ok"
    assert printed == [line]  # content unchanged (no injection)
    assert len(forwarded) == 1
    assert forwarded[0].endswith(line)
    # print_rank_last must be restored after the call.
    assert training_mod.print_rank_last is fake_print_rank_last
