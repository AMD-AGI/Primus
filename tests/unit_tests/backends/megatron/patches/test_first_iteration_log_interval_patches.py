###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""megatron.training_log.first_iteration_ends_interval: an opt-in flag, off by default."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml

from primus.backends.megatron.patches.first_iteration_log_interval_patches import (
    _enabled,
    patch_first_iteration_log_interval,
    wrap_training_log,
)
from primus.core.patches import PatchContext

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _ctx(**params):
    ns = SimpleNamespace(**params)
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"backend_args": ns, "module_config": SimpleNamespace(params=ns)},
    )


def test_opt_in_only():
    assert not _enabled(_ctx())
    assert not _enabled(_ctx(log_first_iteration_ends_interval=False, model_type="olmo3"))
    assert _enabled(_ctx(log_first_iteration_ends_interval=True, model_type="gpt"))
    assert not _enabled(_ctx(log_first_iteration_ends_interval=True, mlperf_mode=True))
    # CLI / quoted-YAML strings are read as booleans, never as "non-empty means on"
    for off in ("false", "False", "0", ""):
        assert not _enabled(_ctx(log_first_iteration_ends_interval=off))
    for on in ("true", "TRUE", "1"):
        assert _enabled(_ctx(log_first_iteration_ends_interval=on))
    # mlperf_mode is read as the MLPerf logging patch reads it (truthiness), so the two never wrap
    # training_log together: any truthy value (also the string "false") turns this option off
    for mlperf in (True, "true", "false", 1):
        assert not _enabled(_ctx(log_first_iteration_ends_interval=True, mlperf_mode=mlperf))
    for mlperf in (False, None, "", 0):
        assert _enabled(_ctx(log_first_iteration_ends_interval=True, mlperf_mode=mlperf))
    import pytest

    with pytest.raises(ValueError, match="true or false"):
        _enabled(_ctx(log_first_iteration_ends_interval="ture"))


def test_default_is_off_in_module_config():
    cfg = yaml.safe_load(open(_REPO_ROOT / "primus/configs/modules/megatron/primus_megatron_module.yaml"))
    assert cfg["log_first_iteration_ends_interval"] is False


def _recorder(writers=None):
    """A stand-in with upstream's parameter names; it records (iteration, is_first_iteration) and,
    like upstream, writes iteration-time unless it is the first iteration."""
    seen = []

    def training_log(
        loss_dict,
        total_loss_dict,
        learning_rate,
        iteration,
        loss_scale=None,
        report_memory_flag=None,
        skipped_iter=None,
        grad_norm=None,
        params_norm=None,
        num_zeros_in_grad=None,
        max_attention_logit=None,
        pg_collection=None,
        is_first_iteration=False,
    ):
        seen.append((iteration, is_first_iteration))
        if writers is not None:
            tb, wb = writers.get_tensorboard_writer(), writers.get_wandb_writer()
            if not is_first_iteration:
                tb.add_scalar("iteration-time", 1.0, iteration)
                wb.log({"iteration-time": 1.0}, iteration)
            tb.add_scalar("lm loss", 2.0, iteration)
            wb.log({"lm loss": 2.0}, iteration)
        return "out"

    return training_log, seen


def test_first_iteration_on_the_schedule_is_logged_as_a_normal_one():
    """Upstream then resets the loss sums, counters and every logged timer itself."""
    original, seen = _recorder()
    log = wrap_training_log(original, lambda: 1)
    assert log.__name__ == "training_log" and log.__wrapped__ is original
    assert log(None, {}, 1e-4, 5, is_first_iteration=True) == "out"
    log(None, {}, 1e-4, 6, is_first_iteration=False)
    log(None, total_loss_dict={}, learning_rate=1e-4, iteration=7, is_first_iteration=True)
    # all positional, as an upstream call could pass them
    log(None, {}, 1e-4, 8, None, None, None, None, None, None, None, None, True)
    assert seen == [(5, False), (6, False), (7, False), (8, False)]


def test_first_iteration_off_the_schedule_is_unchanged():
    original, seen = _recorder()
    log = wrap_training_log(original, lambda: 10)
    log(None, {}, 1e-4, 5, is_first_iteration=True)
    log(None, {}, 1e-4, 10, is_first_iteration=True)
    assert seen == [(5, True), (10, False)]
    assert wrap_training_log(log, lambda: 1) is log  # not wrapped twice


def test_first_iteration_time_stays_out_of_tensorboard_and_wandb():
    """As upstream: the first iteration's iteration-time (start-up overhead) is not written; other
    scalars are. The writer getters are installed once and filter only inside that call, on that
    thread: another thread, or the same getters after the call, see the original writers."""
    import threading

    from primus.backends.megatron.patches.first_iteration_log_interval_patches import (
        _FirstIterationWriters,
    )

    tb_calls, wb_calls = [], []
    tb = SimpleNamespace(add_scalar=lambda tag, *a, **k: tb_calls.append(tag))
    wb = SimpleNamespace(log=lambda data, *a, **k: wb_calls.append(sorted(data)))
    writers = _FirstIterationWriters(lambda: tb, lambda: wb)
    other_thread = []

    def look_from_another_thread():
        other_thread.append(writers.get_tensorboard_writer() is tb)

    original, seen = _recorder(writers)

    def logging_with_a_thread(*a, **k):
        t = threading.Thread(target=look_from_another_thread)
        t.start()
        t.join()
        return original(*a, **k)

    logging_with_a_thread.__signature__ = __import__("inspect").signature(original)
    log = wrap_training_log(logging_with_a_thread, lambda: 1, writers=writers)
    log(None, {}, 1e-4, 1, is_first_iteration=True)
    assert seen == [(1, False)]
    assert tb_calls == ["lm loss"] and wb_calls == [["lm loss"]]
    assert other_thread == [True]  # the other thread got the unfiltered writer
    assert writers.get_tensorboard_writer() is tb  # after the call: the original writer
    log(None, {}, 1e-4, 2, is_first_iteration=False)
    assert tb_calls[1:] == ["iteration-time", "lm loss"]


def test_refuses_a_training_log_without_the_parameters():
    import pytest

    with pytest.raises(TypeError, match="is_first_iteration"):
        wrap_training_log(lambda loss_dict, iteration: None, lambda: 1)


def _fake_megatron(monkeypatch, upstream_src):
    """Stand-in megatron.training(.training) modules; ``upstream_src`` defines ``training_log`` in the
    fake training module's own namespace, as Megatron's does."""
    training_module = types.ModuleType("megatron.training.training")
    training_module.seen = []
    training_module.get_tensorboard_writer = lambda: None
    training_module.get_wandb_writer = lambda: None
    exec(upstream_src, vars(training_module))
    training_module.training_log.__module__ = training_module.__name__
    package = types.ModuleType("megatron.training")
    package.training = training_module
    package.get_args = lambda: SimpleNamespace(log_interval=1)
    if "megatron" not in sys.modules:
        monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    # `import megatron.training.training as m` reads the attributes of the parent packages
    monkeypatch.setattr(sys.modules["megatron"], "training", package, raising=False)
    monkeypatch.setitem(sys.modules, "megatron.training", package)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training_module)
    return training_module


_UPSTREAM = """
def training_log(loss_dict, total_loss_dict, learning_rate, iteration, is_first_iteration=False):
    writer, wandb = get_tensorboard_writer(), get_wandb_writer()
    seen.append((iteration, is_first_iteration))
"""


def test_registered_patch_installs_the_wrapper_once(monkeypatch):
    training_module = _fake_megatron(monkeypatch, _UPSTREAM)
    original, original_getter = training_module.training_log, training_module.get_tensorboard_writer

    ctx = _ctx(log_first_iteration_ends_interval=True)
    patch_first_iteration_log_interval(ctx)
    installed = training_module.training_log
    assert installed is not original and installed.__wrapped__ is original
    installed_getter = training_module.get_tensorboard_writer
    assert installed_getter is not original_getter and installed_getter() is None
    patch_first_iteration_log_interval(ctx)  # a second apply keeps the one wrapper and getters
    assert training_module.training_log is installed
    assert training_module.get_tensorboard_writer == installed_getter
    installed(None, {}, 1e-4, 3, is_first_iteration=True)
    assert training_module.seen == [(3, False)]


def test_getters_untouched_when_upstream_does_not_read_them(monkeypatch):
    """If upstream training_log does not look the writer getters up in its module, they are not
    replaced (a warning says so); the loss-interval part still works."""
    training_module = _fake_megatron(
        monkeypatch,
        """
def training_log(loss_dict, total_loss_dict, learning_rate, iteration, is_first_iteration=False):
    seen.append((iteration, is_first_iteration))
""",
    )
    getter = training_module.get_tensorboard_writer
    patch_first_iteration_log_interval(_ctx(log_first_iteration_ends_interval=True))
    assert training_module.get_tensorboard_writer is getter
    training_module.training_log(None, {}, 1e-4, 2, is_first_iteration=True)
    assert training_module.seen == [(2, False)]


def test_unknown_signature_skips_the_option_and_leaves_the_module_unchanged(monkeypatch):
    """An upstream training_log without the parameters: the option is skipped with a warning (a
    logging option never stops training) and no module attribute changes."""
    training_module = _fake_megatron(
        monkeypatch,
        """
def training_log(loss_dict, iteration):
    w = get_tensorboard_writer(), get_wandb_writer()
""",
    )
    before = {
        n: getattr(training_module, n) for n in ("training_log", "get_tensorboard_writer", "get_wandb_writer")
    }
    patch_first_iteration_log_interval(_ctx(log_first_iteration_ends_interval=True))
    assert {n: getattr(training_module, n) for n in before} == before


def test_installed_after_the_unified_patch_warns_and_skips(monkeypatch):
    """If the patch order changes and unified_patch's wrapper (no __wrapped__) is installed first,
    upstream's training_log cannot be found: the option is skipped, training_log stays the unified
    wrapper, and start-up does not fail."""
    training_module = _fake_megatron(monkeypatch, _UPSTREAM)
    upstream = training_module.training_log

    def primus_training_log(*args, **kwargs):  # as print_rank_last_patches.primus_training_log
        return upstream(*args, **kwargs)

    training_module.training_log = primus_training_log
    getter = training_module.get_tensorboard_writer
    patch_first_iteration_log_interval(_ctx(log_first_iteration_ends_interval=True))
    assert training_module.training_log is primus_training_log
    assert training_module.get_tensorboard_writer is getter
    training_module.training_log(None, {}, 1e-4, 3, is_first_iteration=True)
    assert training_module.seen == [(3, True)]  # upstream's behaviour


def test_works_after_a_pass_through_wrapper_without_wraps():
    """Installed after Primus's unified training_log patch (a plain ``*args, **kwargs`` wrapper with
    no functools.wraps), the call is bound with upstream's signature (``reference``)."""
    original, seen = _recorder()

    def primus_training_log(*args, **kwargs):  # as print_rank_last_patches.primus_training_log
        return original(*args, **kwargs)

    log = wrap_training_log(primus_training_log, lambda: 1, reference=original)
    log(None, {}, 1e-4, 5, is_first_iteration=True)
    log(None, {}, 1e-4, 6, None, None, None, None, None, None, None, None, True)
    assert seen == [(5, False), (6, False)]
    # a wrapper that keeps __wrapped__ needs no reference
    import functools

    wrapped = functools.wraps(original)(lambda *a, **k: original(*a, **k))
    wrap_training_log(wrapped, lambda: 1)(None, {}, 1e-4, 7, is_first_iteration=True)
    assert seen[-1] == (7, False)


def test_registered_patch_runs_before_the_unified_training_log_patch():
    import importlib

    from primus.backends.megatron.patches import (
        first_iteration_log_interval_patches as mod,
    )
    from primus.core.patches import PatchRegistry

    importlib.reload(mod)  # registers again (another test may clear the registry)
    assert PatchRegistry.get("megatron.training_log.first_iteration_ends_interval").priority < 50
