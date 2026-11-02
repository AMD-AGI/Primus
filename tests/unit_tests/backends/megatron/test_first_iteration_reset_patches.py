###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""CPU tests of the first-iteration-reset training_log patch.

The backport is tested on the real ``training_log`` of the pinned
third_party/Megatron-LM, vendored in pinned_megatron_training_log.py.txt (a test
checks it against the submodule when that is checked out). It is loaded with
stub globals (CPU tensors, a fake clock, a TensorBoard writer that records
values), so no GPU, submodule or Megatron install is needed. The registered patch is installed on a fake ``megatron.training.training``.
"""

import ast
import functools
import importlib.util
import inspect
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.patches.training_log import (
    first_iteration_reset_patches as fir,
)
from primus.core.patches.patch_registry import PatchRegistry

REPO = Path(__file__).resolve().parents[4]
MEGATRON_TRAINING = REPO / "third_party/Megatron-LM/megatron/training/training.py"
# training_log of the pinned Megatron-LM, vendored so these tests need no submodule
# (a .txt file, so formatters and linters keep it a verbatim copy).
FIXTURE = Path(__file__).with_name("pinned_megatron_training_log.py.txt")


def _training_log_node(text):
    return next(
        n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "training_log"
    )


def test_fixture_matches_pinned_megatron():
    """The vendored training_log is the submodule's (run where it is checked out)."""
    if not MEGATRON_TRAINING.is_file():
        pytest.skip("third_party/Megatron-LM is not checked out; the fixture is used")
    pinned, fixture = MEGATRON_TRAINING.read_text(), FIXTURE.read_text()
    assert ast.get_source_segment(pinned, _training_log_node(pinned)) == ast.get_source_segment(
        fixture, _training_log_node(fixture)
    )


class Clock:
    now = 0.0


class FakeTimer:
    """Megatron's Timer semantics (elapsed stops, optionally resets, restarts) on a fake clock."""

    def __init__(self):
        self.elapsed_s, self.started, self.start_t = 0.0, False, 0.0

    def start(self, barrier=False):
        self.started, self.start_t = True, Clock.now

    def stop(self, barrier=False):
        self.elapsed_s += Clock.now - self.start_t
        self.started = False

    def elapsed(self, reset=True, barrier=False):
        was = self.started
        if was:
            self.stop()
        value = self.elapsed_s
        if reset:
            self.elapsed_s = 0.0
        if was:
            self.start()
        return value


class FakeTimers:
    def __init__(self):
        self.timers = {}

    def __call__(self, name, log_level=None):
        return self.timers.setdefault(name, FakeTimer())

    def log(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass


class Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, step, float(value)))


class _Torch(types.ModuleType):
    """torch with CPU tensors for ``device='cuda'`` and a single-rank distributed."""

    def __init__(self):
        super().__init__("torch")
        self.float = torch.float
        self.distributed = SimpleNamespace(get_rank=lambda: 0, get_backend=lambda: "fake")
        self.cuda = SimpleNamespace(memory_stats=dict)

    @staticmethod
    def tensor(data, dtype=None, device=None):
        return torch.tensor(data, dtype=dtype)


def _megatron(log_interval, tmp_path, patched=False):
    """A module with the pinned training_log (from a file, so inspect.getsource
    works on it), its globals stubbed; with ``patched``, the patch's backport."""
    args = SimpleNamespace(
        log_interval=log_interval,
        tensorboard_log_interval=1,
        timing_log_level=0,
        micro_batch_size=1,
        data_parallel_size=1,
        seq_length=8,
        world_size=1,
        consumed_train_samples=0,
        skipped_train_samples=0,
        train_iters=100,
        log_throughput=True,
        log_timers_to_tensorboard=True,
        log_energy=False,
        record_memory_history=False,
        log_memory_interval=None,
        num_experts=None,
        mtp_num_layers=None,
        dsa_indexer_loss_coeff=None,
        log_loss_scale_to_tensorboard=False,
        log_world_size_to_tensorboard=False,
        log_memory_to_tensorboard=False,
        log_max_attention_logit=False,
        rl_use_sequence_packing=False,
    )
    m = SimpleNamespace(args=args, timers=FakeTimers(), writer=Writer(), printed=[])
    path = tmp_path / f"pinned_training_log_{len(list(tmp_path.iterdir()))}.py"
    path.write_text(FIXTURE.read_text())
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__.update(
        torch=_Torch(),
        datetime=__import__("datetime").datetime,
        has_rl_utils=False,
        get_args=lambda: args,
        get_timers=lambda: m.timers,
        get_tensorboard_writer=lambda: m.writer,
        get_wandb_writer=lambda: None,
        get_one_logger=lambda: None,
        get_energy_monitor=lambda: None,
        get_num_microbatches=lambda: 1,
        get_loaded_iteration=lambda: 0,
        one_logger_utils=SimpleNamespace(track_app_tag=lambda *a: None, track_e2e_metrics=lambda *a: None),
        reduce_max_stat_across_model_parallel_group=lambda x: x,
        num_floating_point_operations=lambda args, bs: 1e12,
        print_rank_last=m.printed.append,
        report_memory=lambda *a: None,
        report_theoretical_memory=lambda *a, **k: None,
        is_last_rank=lambda: True,
    )
    spec.loader.exec_module(mod)
    m.module = mod
    m.pinned = mod.training_log
    if patched:
        assert fir.apply_backport(mod) == "patched"
    m.training_log = mod.training_log
    return m


def _run(log_interval, start, iters, patched, tmp_path, init_s=100.0, step_s=1.0):
    """Megatron's train loop: a resumed run starts at ``start``; the first
    iteration also takes ``init_s`` of initialization time."""
    m = _megatron(log_interval, tmp_path, patched)
    m.timers("interval-time").start()
    total, first, report = {}, True, True
    for it in range(start + 1, start + iters + 1):
        Clock.now += step_s + (init_s if first else 0.0)
        report = m.training_log(
            {"lm loss": torch.tensor([float(it)])},
            total,
            1e-4,
            it,
            1.0,
            report,
            0,
            1.0,
            None,
            None,
            None,
            is_first_iteration=first,
        )
        first = False
    return m


def _logged(m, field):
    return [float(re.search(field + r": ([0-9.E+-]+)", s).group(1)) for s in m.printed]


def test_resume_logs_each_iteration_own_loss_and_time(tmp_path):
    # resumed at iteration 4, log_interval 1: the pinned code logs iteration 6 as
    # mean(5, 6), and its time includes the first iteration's initialization
    plain, patched = _run(1, 4, 4, False, tmp_path), _run(1, 4, 4, True, tmp_path)
    assert _logged(plain, "lm loss") == [5.0, 5.5, 7.0, 8.0]
    assert _logged(patched, "lm loss") == [5.0, 6.0, 7.0, 8.0]
    assert _logged(plain, r"elapsed time per iteration \(ms\)")[1] == pytest.approx(51000.0)
    assert _logged(patched, r"elapsed time per iteration \(ms\)") == pytest.approx(
        [101000.0, 1000, 1000, 1000]
    )


def _written(m, name):
    return {step: value for n, step, value in m.writer.scalars if n == name}


def test_tensorboard_writes(tmp_path):
    """The same scalars are written at the same steps. TensorBoard's lm loss is
    per iteration, so it does not change; the only changed value is the
    iteration time at step 6, which no longer includes the first iteration."""
    plain, patched = _run(1, 4, 4, False, tmp_path), _run(1, 4, 4, True, tmp_path)
    assert [s[:2] for s in patched.writer.scalars] == [s[:2] for s in plain.writer.scalars]
    assert _written(plain, "lm loss") == _written(patched, "lm loss") == {5: 5.0, 6: 6.0, 7: 7.0, 8: 8.0}
    # the first iteration's iteration-time is not written (as upstream)
    assert _written(plain, "iteration-time") == pytest.approx({6: 51.0, 7: 1.0, 8: 1.0})
    assert _written(patched, "iteration-time") == pytest.approx({6: 1.0, 7: 1.0, 8: 1.0})
    changed = {s[:2] for s, p in zip(plain.writer.scalars, patched.writer.scalars) if s != p}
    assert changed and {step for _, step in changed} == {6}
    assert ("iteration-time", 6) in changed and ("lm loss", 6) not in changed


def test_first_iteration_not_a_log_point_is_unchanged(tmp_path):
    # log_interval 2, resumed at 4: iteration 5 is the first (always printed) and
    # not a log point, so the pinned code already resets after it
    plain, patched = _run(2, 4, 4, False, tmp_path), _run(2, 4, 4, True, tmp_path)
    assert _logged(patched, "lm loss") == _logged(plain, "lm loss") == [5.0, 5.5, 7.5]
    assert _logged(patched, r"elapsed time per iteration \(ms\)") == _logged(
        plain, r"elapsed time per iteration \(ms\)"
    )
    assert patched.writer.scalars == plain.writer.scalars


def test_first_iteration_on_a_log_point_with_interval_above_1(tmp_path):
    # log_interval 2, resumed at 3: iteration 4 is the first and a log point. The
    # pinned code logs 6 as the mean of 4, 5 and 6 (and their time, also the
    # first iteration's initialization); the backport logs the mean of 5 and 6.
    plain, patched = _run(2, 3, 5, False, tmp_path), _run(2, 3, 5, True, tmp_path)
    assert _logged(plain, "lm loss") == [4.0, 5.0, 7.5]
    assert _logged(patched, "lm loss") == [4.0, 5.5, 7.5]
    assert _logged(plain, r"elapsed time per iteration \(ms\)") == pytest.approx([101000.0, 34333.3, 1000])
    assert _logged(patched, r"elapsed time per iteration \(ms\)") == pytest.approx([101000.0, 1000, 1000])


def test_backport_is_upstream_line_only(tmp_path):
    """The rebuilt function's AST differs from the pinned one only in the
    should_reset line; it keeps the module globals, name and signature."""
    m = _megatron(1, tmp_path, patched=True)
    assert m.training_log is not m.pinned
    assert m.training_log.__globals__ is m.pinned.__globals__
    assert (m.training_log.__name__, m.training_log.__module__) == (m.pinned.__name__, m.pinned.__module__)
    assert inspect.signature(m.training_log) == inspect.signature(m.pinned)
    pinned_src = inspect.getsource(m.pinned)
    expected = ast.dump(ast.parse(pinned_src.replace(fir.PINNED_LINE, fir.FIXED_LINE)))
    assert ast.dump(ast.parse(pinned_src)) != expected
    # the rebuilt code object is the compiled expected source
    ns = {}
    exec(
        compile(pinned_src.replace(fir.PINNED_LINE, fir.FIXED_LINE), "x", "exec"),
        dict(m.pinned.__globals__),
        ns,
    )
    assert m.training_log.__code__.co_code == ns["training_log"].__code__.co_code
    assert m.training_log.__code__.co_consts[1:] == ns["training_log"].__code__.co_consts[1:]


def test_check_training_log():
    def fixed(iteration, is_first_iteration=False):
        should_reset = (not is_first_iteration) or (iteration % args.log_interval == 0)  # noqa: F821,F841

    def changed(iteration, is_first_iteration=False):
        reset = True  # noqa: F841

    def spaced(iteration, is_first_iteration=False):
        should_reset = not is_first_iteration  # a comment  # noqa: F841

    @functools.wraps(fixed)
    def wrapper(*a, **k):
        return fixed(*a, **k)

    assert fir.check_training_log(fixed) == "fixed"  # upstream fix, spelled differently

    def fixed_multiline(iteration, is_first_iteration=False):
        should_reset = (  # noqa: F841
            iteration % args.log_interval == 0 or not is_first_iteration  # noqa: F821
        )

    def other_condition(iteration, is_first_iteration=False):
        # should_reset = not is_first_iteration or iteration % args.log_interval == 0
        should_reset = not is_first_iteration and iteration % args.log_interval  # noqa: F821,F841

    def twice(iteration, is_first_iteration=False):
        should_reset = not is_first_iteration  # noqa: F841
        should_reset = not is_first_iteration or iteration % args.log_interval == 0  # noqa: F821,F841

    def augmented(iteration, is_first_iteration=False):
        should_reset = not is_first_iteration
        should_reset |= iteration > 0  # noqa: F841

    def annotated(iteration, is_first_iteration=False):
        should_reset: bool = not is_first_iteration  # noqa: F841

    def walrus(iteration, is_first_iteration=False):
        should_reset = not is_first_iteration  # noqa: F841
        if should_reset := iteration > 0:  # noqa: F841
            pass

    assert fir.check_training_log(fixed_multiline) == "fixed"  # parenthesised, other order

    def plain_wrapper(*a, **k):  # a wrapper without functools.wraps (as Primus's mlperf one)
        return fixed(*a, **k)

    namespace = {}
    exec("def training_log(iteration):\n    pass\n", namespace)  # no source to read

    assert fir.check_training_log(fixed, fixed.__module__) == "fixed"
    for fn, match in (
        (changed, "does not have"),
        (spaced, "does not have"),
        (other_condition, "does not have"),  # a different condition, and the fix only in a comment
        (twice, "does not have"),
        (augmented, "does not have"),
        (annotated, "does not have"),
        (walrus, "does not have"),
        (wrapper, "not Megatron's function"),
        (plain_wrapper, "not Megatron's function"),
        (namespace["training_log"], "cannot read the source"),
    ):
        module = None if fn is namespace["training_log"] else "megatron.training.training"
        if fn in (changed, spaced, other_condition, twice, augmented, annotated, walrus):
            module = fn.__module__
        with pytest.raises(RuntimeError, match=match):
            fir.check_training_log(fn, module)


# ---------------------------------------------------------------------------
# The registered patch
# ---------------------------------------------------------------------------
def _ctx(enabled):
    params = SimpleNamespace(reset_log_averages_after_first_log=enabled)
    return SimpleNamespace(
        extra={"module_config": SimpleNamespace(params=params)}, backend="megatron", phase="before_train"
    )


def _fake_megatron(monkeypatch, training_log):
    mod = types.ModuleType("megatron.training.training")
    training_log.__module__ = mod.__name__  # as if defined there
    mod.training_log = training_log
    pkg = types.ModuleType("megatron.training")
    pkg.training = mod
    top = types.ModuleType("megatron")
    top.training = pkg
    monkeypatch.setitem(sys.modules, "megatron", top)
    monkeypatch.setitem(sys.modules, "megatron.training", pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", mod)
    return mod


def test_registration():
    p = PatchRegistry.get(fir.PATCH_ID)
    assert p is not None and p.backend == "megatron" and p.phase == "before_train"
    assert p.priority < 15  # rewrites Megatron's function before mlperf (15) and print_rank_last (50) wrap it
    assert p.applies_to(_ctx(True)) and not p.applies_to(_ctx(False))


def test_installs_once(monkeypatch, tmp_path):
    m = _megatron(1, tmp_path)
    mod = _fake_megatron(monkeypatch, m.pinned)
    fir.patch_first_iteration_reset(_ctx(True))
    fixed = mod.training_log
    assert fixed is not m.pinned
    assert fixed.__code__.co_code != m.pinned.__code__.co_code
    fir.patch_first_iteration_reset(_ctx(True))
    assert mod.training_log is fixed


def test_fails_when_line_missing(monkeypatch):
    def training_log(loss_dict, total_loss_dict):
        pass

    _fake_megatron(monkeypatch, training_log)
    with pytest.raises(RuntimeError, match="reset_log_averages_after_first_log"):
        fir.patch_first_iteration_reset(_ctx(True))


def test_already_fixed_megatron_is_left_alone(monkeypatch):
    def training_log(iteration, is_first_iteration=False):
        should_reset = not is_first_iteration or iteration % args.log_interval == 0  # noqa: F821,F841

    mod = _fake_megatron(monkeypatch, training_log)
    fir.patch_first_iteration_reset(_ctx(True))
    assert mod.training_log is training_log
