#!/usr/bin/env python3
# pyre-unsafe
"""DL_TEARDOWN_HARD_EXIT (clean_exit.py): a successful streaming run ends with
os._exit(0) after its teardown, so no daemon thread can be inside torch while
the interpreter finalizes. CPU only, standard library only:

  * hard_exit order: the trainer's exit hooks, the rank-0 line,
    stop_children, the log and stdio flush, then os._exit(code). os._exit and
    stop_children are patched in every in-process test: the gate runs all
    test files in ONE pytest process, so nothing here may kill that process's
    children or change its SIGCHLD handler (setUp/tearDown check the latter);
    the real EXIT_HOOKS entry is triton_spec._at_exit and resolves at call
    time;
  * stop_children resets SIGCHLD to SIG_DFL before it SIGKILLs every live
    multiprocessing child, never joins and never raises (signal.signal and
    multiprocessing.active_children patched); the reset is retried once if
    signal.signal raises (a pending raising handler) and called once if not;
  * a raising hook, or a raising log callable, never prevents the exit; a hook
    whose module is not loaded is skipped (nothing is imported at exit);
  * with the switch off, or in a TunableOp tuning / record-untuned session,
    hard_exit returns (the normal exit) and does nothing else;
  * the rank-0-only line, the kill switch values;
  * every exit-time registration in the package (atexit.register under any
    alias, threading._register_atexit, weakref.finalize,
    multiprocessing.util.Finalize with an exitpriority) is an EXIT_HOOKS
    entry (os._exit skips them all, so a new one must be run explicitly);
  * train_ranker calls _clean_exit.hard_exit(rank, ...) with the global rank
    exactly once, after the try/finally teardown (barrier + cleanup), never
    inside it, as the only statement under `if _clean_exit.enabled():` (no
    extra condition, so every rank takes it), and no handler in the teardown
    swallows a failure, so only the success path reaches it;
  * scripts/launch_slurm.sh forwards the switch once, into the trainer
    container;
  * real child processes: one whose daemon thread is blocked in a C call with
    the GIL released exits 0 at once through hard_exit, with its exit hook
    run, its buffered Python and C stdout flushed and its atexit callbacks
    skipped; one with forked multiprocessing children (one of them ignores
    SIGTERM) exits 0 and none of its children outlives it; one where a REAL
    SIGCHLD is pending, with a raising handler, when stop_children resets
    SIGCHLD: the first reset raises, the retry resets, both children die.
"""

import ast
import json
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import time
import types
import unittest
from unittest import mock

from generative_recommenders.dlrm_v4.train import clean_exit as ce

# recommendation/ (this file: recommendation/generative_recommenders/dlrm_v4/train/tests/)
_REPO = pathlib.Path(__file__).resolve().parents[4]
_LINE0 = (
    "[teardown] clean exit: os._exit(0) after barrier/destroy_process_group "
    "(DL_TEARDOWN_HARD_EXIT=1) rank=0"
)
_SIGCHLD = getattr(signal, "SIGCHLD", None)


class _Exited(Exception):
    """Raised by the patched os._exit, so hard_exit stops where the process
    would."""


class HardExitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = []
        self.mod = types.ModuleType("_clean_exit_test_hooks")
        sys.modules[self.mod.__name__] = self.mod
        self.sigchld = signal.getsignal(_SIGCHLD) if _SIGCHLD else None

    def tearDown(self) -> None:
        sys.modules.pop(self.mod.__name__, None)
        if _SIGCHLD:  # the test process's handler is never touched
            self.assertEqual(signal.getsignal(_SIGCHLD), self.sigchld)

    def _hook(self, name: str, exc: BaseException = None):
        def fn() -> None:
            self.events.append(("hook", name))
            if exc is not None:
                raise exc
        setattr(self.mod, name, fn)
        return (self.mod.__name__, name)

    def _run(self, hooks, rank: int = 0, code: int = 0, log=None):
        """hard_exit with the switch on and os._exit, stop_children,
        logging.shutdown and the stdio streams patched; returns the exit code
        os._exit received."""
        if log is None:
            def log(m: str) -> None:
                self.events.append(("log", m))

        def _exit(c: int) -> None:
            self.events.append(("exit", c))
            raise _Exited()

        def _stop() -> int:
            self.events.append(("stop_children",))
            return 0

        class _Stream:
            def __init__(s, name: str) -> None:
                s.name = name

            def write(s, x: str) -> int:
                return len(x)

            def flush(s) -> None:
                self.events.append(("flush", s.name))

        with mock.patch.dict(os.environ, {ce.ENV: "1"}), \
                mock.patch.object(ce, "EXIT_HOOKS", tuple(hooks)), \
                mock.patch.object(ce, "native_exit_writer", lambda *a, **k: ""), \
                mock.patch.object(ce, "stop_children", _stop), \
                mock.patch.object(ce.os, "_exit", _exit), \
                mock.patch.object(ce.logging, "shutdown",
                                  lambda: self.events.append(("shutdown",))), \
                mock.patch.object(ce.sys, "stdout", _Stream("stdout")), \
                mock.patch.object(ce.sys, "stderr", _Stream("stderr")):
            with self.assertRaises(_Exited):
                ce.hard_exit(rank, log=log, code=code)
        return [e[1] for e in self.events if e[0] == "exit"]

    def test_order_hooks_line_stop_children_flush_os_exit(self) -> None:
        codes = self._run([self._hook("a"), self._hook("b")])
        self.assertEqual(codes, [0])
        kinds = [e[0] for e in self.events]
        # hooks first (in EXIT_HOOKS order), then the one line, then the
        # children are stopped, then the flush, and os._exit last
        self.assertEqual(self.events[:4], [("hook", "a"), ("hook", "b"), ("log", _LINE0),
                                           ("stop_children",)])
        self.assertEqual(kinds.count("stop_children"), 1)
        self.assertEqual(kinds.index("shutdown"), 4)
        self.assertIn(("flush", "stdout"), self.events[5:-1])
        self.assertIn(("flush", "stderr"), self.events[5:-1])
        self.assertEqual(self.events[-1], ("exit", 0))
        self.assertEqual(ce.EXIT_FMT.format(code=0, rank=0), _LINE0)

    def test_exit_code_passed_through(self) -> None:
        self.assertEqual(self._run([self._hook("a")], code=3), [3])

    def test_raising_hook_does_not_prevent_exit(self) -> None:
        hooks = [self._hook("bad", RuntimeError("boom")),
                 self._hook("worse", SystemExit(7)),
                 self._hook("good")]
        self.assertEqual(self._run(hooks), [0])
        self.assertEqual([e for e in self.events if e[0] == "hook"],
                         [("hook", "bad"), ("hook", "worse"), ("hook", "good")])
        warns = [e[1] for e in self.events if e[0] == "log" and "exit hook" in e[1]]
        self.assertEqual(len(warns), 2)
        self.assertIn("RuntimeError: boom", warns[0])
        self.assertIn("SystemExit", warns[1])

    def test_raising_log_does_not_prevent_exit(self) -> None:
        def log(m: str) -> None:
            raise OSError("log sink gone")

        self.assertEqual(self._run([self._hook("bad", ValueError("x"))], log=log), [0])

    def test_unloaded_module_skipped(self) -> None:
        name = "_clean_exit_test_never_imported"
        self.assertNotIn(name, sys.modules)
        self.assertEqual(self._run([(name, "_at_exit"), self._hook("a")]), [0])
        self.assertNotIn(name, sys.modules)  # nothing imported at exit
        self.assertEqual([e for e in self.events if e[0] == "hook"], [("hook", "a")])

    def test_rank0_only_line(self) -> None:
        self._run([], rank=3)
        self.assertEqual([e for e in self.events if e[0] == "log"], [])
        self.assertIn(("stop_children",), self.events)  # every rank stops its children
        self.events.clear()
        self._run([], rank=0)
        self.assertEqual([e[1] for e in self.events if e[0] == "log"], [_LINE0])

    def test_kill_switch(self) -> None:
        self.assertEqual(ce.ENV, "DL_TEARDOWN_HARD_EXIT")
        self.assertTrue(ce.enabled({}))
        for v in ("1", "yes", "", "2", "on"):
            self.assertTrue(ce.enabled({ce.ENV: v}), v)
        for v in ("0", "false", "OFF", " no "):
            self.assertFalse(ce.enabled({ce.ENV: v}), v)
        with mock.patch.dict(os.environ, {ce.ENV: "0"}):
            self.assertFalse(ce.enabled())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ce.ENV, None)
            self.assertTrue(ce.enabled())

    def test_switch_off_returns_without_exiting(self) -> None:
        # A second guard behind the caller's enabled() check: with the switch
        # off, hard_exit itself keeps the normal exit and only logs OFF_LINE
        # on rank 0 (no hook, no kill, no flush, no os._exit).
        for rank in (0, 1):
            self.events.clear()
            with mock.patch.dict(os.environ, {ce.ENV: "0"}), \
                    mock.patch.object(ce, "EXIT_HOOKS", (self._hook("a"),)), \
                    mock.patch.object(ce, "native_exit_writer") as writer, \
                    mock.patch.object(ce, "stop_children") as stop, \
                    mock.patch.object(ce.os, "_exit") as osexit, \
                    mock.patch.object(ce, "_flush_all") as flush:
                ce.hard_exit(rank, log=lambda m: self.events.append(("log", m)))
            for m in (writer, stop, osexit, flush):
                m.assert_not_called()
            self.assertNotIn(("hook", "a"), self.events)
            logs = [e[1] for e in self.events if e[0] == "log"]
            self.assertEqual(logs, [ce.OFF_LINE] if rank == 0 else [])

    def test_triton_spec_hook_is_called(self) -> None:
        # The shipped entry, resolved at call time (stdlib-only module).
        from generative_recommenders.dlrm_v4.train import triton_spec

        key = ("generative_recommenders.dlrm_v4.train.triton_spec", "_at_exit")
        self.assertIn(key, ce.EXIT_HOOKS)
        self.assertTrue(callable(getattr(triton_spec, "_at_exit")))
        with mock.patch.dict(os.environ, {ce.ENV: "1"}), \
                mock.patch.object(triton_spec, "_at_exit") as hook, \
                mock.patch.object(ce, "native_exit_writer", lambda *a, **k: ""), \
                mock.patch.object(ce, "stop_children") as stop, \
                mock.patch.object(ce.os, "_exit") as osexit, \
                mock.patch.object(ce, "_flush_all"):
            ce.hard_exit(1, log=None)
        hook.assert_called_once_with()
        stop.assert_called_once_with()
        osexit.assert_called_once_with(0)

    def test_tunableop_session_keeps_normal_exit(self) -> None:
        why = "TunableOp tuning or record-untuned is ON (x)"
        for rank in (0, 2):
            self.events.clear()
            with mock.patch.dict(os.environ, {ce.ENV: "1"}), \
                    mock.patch.object(ce, "EXIT_HOOKS", (self._hook("a"),)), \
                    mock.patch.object(ce, "native_exit_writer", lambda *a, **k: why), \
                    mock.patch.object(ce, "stop_children") as stop, \
                    mock.patch.object(ce.os, "_exit") as osexit, \
                    mock.patch.object(ce, "_flush_all") as flush:
                ce.hard_exit(rank, log=lambda m: self.events.append(("log", m)))
            osexit.assert_not_called()  # returns: the normal interpreter exit
            stop.assert_not_called()  # the normal exit terminates and joins them
            flush.assert_not_called()
            self.assertNotIn(("hook", "a"), self.events)  # atexit runs them then
            logs = [e[1] for e in self.events if e[0] == "log"]
            self.assertEqual(logs, [ce.SKIP_FMT.format(why=why, rank=0)] if rank == 0 else [])

    def test_native_exit_writer(self) -> None:
        def fake(enabled, tuning, record=None, raises=False):
            def get(v):
                def f():
                    if raises:
                        raise AttributeError("no _cuda_tunableop bindings")
                    return v
                return f
            tun = types.SimpleNamespace(is_enabled=get(enabled), tuning_is_enabled=get(tuning))
            if record is not None:
                tun.record_untuned_is_enabled = get(record)
            return types.SimpleNamespace(cuda=types.SimpleNamespace(tunable=tun))

        self.assertIn("TunableOp", ce.native_exit_writer(fake(True, True, False)))
        self.assertIn("precaution", ce.native_exit_writer(fake(True, True, False)))
        self.assertIn("TunableOp", ce.native_exit_writer(fake(True, False, True)))
        self.assertEqual(ce.native_exit_writer(fake(True, False, False)), "")
        self.assertEqual(ce.native_exit_writer(fake(True, False)), "")  # no record API
        self.assertEqual(ce.native_exit_writer(fake(False, True, True)), "")
        self.assertEqual(ce.native_exit_writer(fake(True, True, True, raises=True)), "")
        self.assertIsInstance(ce.native_exit_writer(), str)  # this process: never raises

    def test_stop_children_resets_sigchld_then_kills_without_join(self) -> None:
        calls = []

        class _Child:
            def __init__(s, pid: int, exc: BaseException = None) -> None:
                s.pid, s.exc = pid, exc

            def kill(s) -> None:
                calls.append(("kill", s.pid))
                if s.exc is not None:
                    raise s.exc

            def terminate(s) -> None:
                calls.append(("terminate", s.pid))

            def join(s, *a, **k) -> None:
                calls.append(("join", s.pid))

        kids = [_Child(11), _Child(12, ProcessLookupError("gone")), _Child(13)]
        with mock.patch.object(ce.signal, "signal",
                               lambda sig, h: calls.append(("signal", sig, h))), \
                mock.patch.object(ce.multiprocessing, "active_children", lambda: list(kids)):
            n = ce.stop_children()
        # SIGCHLD first (torch's DataLoader handler would raise "worker killed"
        # in the main thread), then SIGKILL for each child; no terminate, no join
        self.assertEqual(calls, [("signal", signal.SIGCHLD, signal.SIG_DFL),
                                 ("kill", 11), ("kill", 12), ("kill", 13)])
        # The reset succeeded, so exactly ONE signal.signal call, before the
        # kills: the retry is only for a call that raised.
        self.assertEqual([c for c in calls if c[0] == "signal"],
                         [("signal", signal.SIGCHLD, signal.SIG_DFL)])
        self.assertEqual(calls[0][0], "signal")
        self.assertEqual(n, 2)

        def no_signal(*a, **k):
            raise ValueError("signal only works in main thread of the main interpreter")

        calls.clear()
        with mock.patch.object(ce.signal, "signal", no_signal), \
                mock.patch.object(ce.multiprocessing, "active_children", lambda: [_Child(21)]):
            self.assertEqual(ce.stop_children(), 1)  # the kill still happens
        self.assertEqual(calls, [("kill", 21)])
        class _Interrupt(BaseException):
            """A BaseException like KeyboardInterrupt; if it escaped, a real
            KeyboardInterrupt would abort the whole pytest session."""

        with mock.patch.object(ce.signal, "signal", lambda *a: None), \
                mock.patch.object(ce.multiprocessing, "active_children",
                                  mock.Mock(side_effect=_Interrupt)):
            self.assertEqual(ce.stop_children(), 0)  # never raises

    def test_stop_children_retries_sigchld_reset_once(self) -> None:
        # signal.signal() runs pending Python signal handlers before it changes
        # the disposition: with a pending SIGCHLD whose handler raises (torch's
        # "worker ... killed by signal"), the first reset raises and leaves that
        # handler installed. That run consumed the pending signal, so ONE retry
        # resets; the attempts are bounded (2) and the kills always follow.
        calls = []

        class _Child:
            def __init__(s, pid: int) -> None:
                s.pid = pid

            def kill(s) -> None:
                calls.append(("kill", s.pid))

        def resets(*outcomes):
            left = list(outcomes)  # per call: None = the reset works, else raised

            def fake_signal(sig, h):
                calls.append(("signal", sig, h))
                exc = left.pop(0)
                if exc is not None:
                    raise exc
            return fake_signal

        reset = ("signal", signal.SIGCHLD, signal.SIG_DFL)
        pending = RuntimeError("DataLoader worker (pid 7) is killed by signal: Killed.")
        # (raises, works): retried once and reset; (raises, raises): bounded at
        # 2 attempts. Either way both kills come after the 2nd attempt.
        for outcomes in ((pending, None), (pending, pending)):
            calls.clear()
            with mock.patch.object(ce.signal, "signal", resets(*outcomes)), \
                    mock.patch.object(ce.multiprocessing, "active_children",
                                      lambda: [_Child(31), _Child(32)]):
                n = ce.stop_children()  # never raises
            self.assertEqual(calls, [reset, reset, ("kill", 31), ("kill", 32)], repr(outcomes))
            self.assertEqual(n, 2)

        class _Interrupt(BaseException):
            """A BaseException like KeyboardInterrupt (a pending SIGINT); if it
            escaped, a real KeyboardInterrupt would abort the whole pytest session."""

        calls.clear()
        with mock.patch.object(ce.signal, "signal", resets(_Interrupt(), None)), \
                mock.patch.object(ce.multiprocessing, "active_children", lambda: [_Child(41)]):
            self.assertIsInstance(ce.stop_children(), int)  # never raises


def _dotted(node, aliases) -> str:
    """`a.b.c` -> "a.b.c" with the first name resolved through the module's
    imports (import x as y; from x import y)."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    return ".".join([aliases.get(node.id, node.id)] + parts[::-1])


def _exit_registrations(tree):
    """Callbacks a module registers to run at interpreter exit, i.e. the ones
    os._exit skips: atexit.register (call or decorator, any alias),
    threading._register_atexit, weakref.finalize, and
    multiprocessing.util.Finalize with an exitpriority (run by
    multiprocessing's _exit_function)."""
    aliases = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    top = a.name.split(".")[0]
                    aliases.setdefault(top, top)
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            for a in n.names:
                aliases[a.asname or a.name] = f"{n.module}.{a.name}"
    arg0 = ("atexit.register", "threading._register_atexit")
    out = set()

    def name(arg) -> str:
        if arg is None:
            return "?"
        return getattr(arg, "id", None) or ast.unparse(arg)

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = _dotted(n.func, aliases)
            if f in arg0:
                out.add(name(n.args[0] if n.args else None))
            elif f == "weakref.finalize":
                out.add(name(n.args[1] if len(n.args) > 1 else None))
            elif f == "multiprocessing.util.Finalize":
                prio = [k.value for k in n.keywords if k.arg == "exitpriority"]
                prio += n.args[4:5]
                if any(not (isinstance(p, ast.Constant) and p.value is None) for p in prio):
                    out.add(name(n.args[1] if len(n.args) > 1 else None))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(_dotted(d, aliases) == "atexit.register" for d in n.decorator_list):
                out.add(n.name)
    return out


class WiringTest(unittest.TestCase):
    def test_every_atexit_registration_is_an_exit_hook(self) -> None:
        # os._exit skips atexit and multiprocessing's _exit_function: every
        # callback the package registers for exit must be run by hard_exit
        # (EXIT_HOOKS), or the clean exit silently drops it.
        found = set()
        root = _REPO / "generative_recommenders"
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.relative_to(root).parts:
                continue
            text = path.read_text(errors="replace")
            low = text.lower()
            if "atexit" not in low and "finalize" not in low:
                continue
            mod = ".".join(path.relative_to(_REPO).with_suffix("").parts)
            found |= {(mod, cb) for cb in _exit_registrations(ast.parse(text))}
        self.assertEqual(found, set(ce.EXIT_HOOKS),
                         "exit-time registrations vs clean_exit.EXIT_HOOKS")

    def test_exit_registration_scanner(self) -> None:
        # The scanner itself: every form it must flag, and the ones it must not.
        src = textwrap.dedent(
            """
            import atexit
            import atexit as ae
            from atexit import register
            from atexit import register as reg
            import threading, weakref
            import multiprocessing.util
            import multiprocessing.util as mpu
            from multiprocessing import util
            from multiprocessing.util import Finalize
            atexit.register(a1)
            ae.register(a2)
            register(a3)
            reg(a4)
            threading._register_atexit(a5)
            weakref.finalize(obj, a6)
            multiprocessing.util.Finalize(obj, a7, exitpriority=0)
            mpu.Finalize(None, a8, args=(1,), exitpriority=100)
            util.Finalize(obj, a9, (), None, 5)
            Finalize(obj, a10, exitpriority=-1)
            Finalize(obj, n1)
            Finalize(obj, n2, exitpriority=None)
            other.register(n3)
            @atexit.register
            def a11(): pass
            @reg
            def a12(): pass
            """
        )
        self.assertEqual(_exit_registrations(ast.parse(src)),
                         {f"a{i}" for i in range(1, 13)})

    def test_train_ranker_calls_hard_exit_after_teardown_on_success_path(self) -> None:
        src = _REPO / "generative_recommenders/dlrm_v4/train/train_ranker.py"
        tree = ast.parse(src.read_text())

        def calls(node, name):
            return [c for c in ast.walk(node) if isinstance(c, ast.Call)
                    and getattr(c.func, "id", getattr(c.func, "attr", None)) == name]

        self.assertEqual(len(calls(tree, "hard_exit")), 1)
        main = next(f for f in tree.body if isinstance(f, ast.FunctionDef)
                    and f.name == "_main_func")
        # The teardown: the top-level try whose finally runs barrier + cleanup.
        tries = [(i, s) for i, s in enumerate(main.body) if isinstance(s, ast.Try)
                 and calls(ast.Module(body=s.finalbody, type_ignores=[]), "barrier")
                 and calls(ast.Module(body=s.finalbody, type_ignores=[]), "cleanup")]
        self.assertEqual(len(tries), 1)
        i_try, teardown = tries[0]
        self.assertEqual(calls(teardown, "hard_exit"), [])  # not in body/except/finally
        self.assertTrue(teardown.handlers)
        for h in teardown.handlers:  # a failure re-raises: it never passes the try
            self.assertIsInstance(h.body[-1], ast.Raise)
        self.assertEqual(teardown.orelse, [])
        # Nor does any handler nested in the try body swallow a failure: a
        # failed loop must not reach the exit (with rc 0).
        for s in teardown.body:
            for h in [n for n in ast.walk(s) if isinstance(n, ast.ExceptHandler)]:
                self.assertIsInstance(h.body[-1], ast.Raise, ast.unparse(h)[:300])
        # The call: a later top-level statement, under mode == streaming-train-eval
        # and _clean_exit.enabled(), with no try around it.
        after = main.body[i_try + 1:]
        where = [s for s in after if calls(s, "hard_exit")]
        self.assertEqual(len(where), 1)
        guard = where[0]
        self.assertIsInstance(guard, ast.If)
        self.assertEqual(ast.unparse(guard.test), "mode == 'streaming-train-eval'")
        self.assertFalse([n for n in ast.walk(guard) if isinstance(n, ast.Try)])
        inner = [n for n in ast.walk(guard) if isinstance(n, ast.If) and calls(n.test, "enabled")]
        self.assertEqual(len(inner), 1)
        # This module's switch, not another one (e.g. _pool_close.enabled()).
        self.assertEqual(ast.unparse(inner[0].test), "_clean_exit.enabled()")
        hx = calls(ast.Module(body=inner[0].body, type_ignores=[]), "hard_exit")
        self.assertEqual(len(hx), 1)
        # The global rank (only global rank 0 logs the line), and the run logger.
        self.assertEqual(ast.unparse(hx[0]), "_clean_exit.hard_exit(rank, log=logger.warning)")
        # The branch is exactly that call: an extra condition around it (e.g.
        # `if rank == 0:` or `if world_size > 1:`) would exit some ranks only.
        self.assertEqual([ast.unparse(s) for s in inner[0].body],
                         ["_clean_exit.hard_exit(rank, log=logger.warning)"])
        ranks = [s for s in main.body if isinstance(s, ast.Assign)
                 and [ast.unparse(t) for t in s.targets] == ["rank"]]
        self.assertEqual([ast.unparse(s.value) for s in ranks],
                         ["node_rank * gpus_per_node + local_rank"])
        self.assertIn("OFF_LINE", ast.unparse(ast.Module(body=inner[0].orelse,
                                                          type_ignores=[])))
        # Nothing between the teardown and the call (the exit follows the
        # teardown directly) and nothing after it in _main_func.
        self.assertIs(after[0], guard)
        self.assertIs(main.body[-1], guard)

    def test_launcher_forwards_switch_once(self) -> None:
        text = (_REPO / "scripts" / "launch_slurm.sh").read_text()
        lines = [s.strip() for s in text.splitlines()]
        fwd = "${%s:+-e %s=$%s}" % (ce.ENV, ce.ENV, ce.ENV)
        self.assertEqual(text.count(fwd), 1)
        at = [i for i, s in enumerate(lines) if s == fwd + " \\"]
        self.assertEqual(len(at), 1)
        # inside the worker `docker exec` that starts the trainer
        start = lines.index("-e LAUNCH_SLURM_PHASE=worker \\")
        end = next(i for i, s in enumerate(lines)
                   if "LAUNCH_SLURM_PHASE=worker bash scripts/launch_slurm.sh" in s)
        self.assertTrue(start < at[0] < end)


_CHILD = textwrap.dedent(
    """
    import atexit, ctypes, logging, sys, threading, time, types
    from generative_recommenders.dlrm_v4.train import clean_exit as ce

    started = threading.Event()

    def blocked():
        started.set()
        time.sleep(600)  # a C call with the GIL released, for the whole test

    threading.Thread(target=blocked, daemon=True).start()
    assert started.wait(30)
    time.sleep(0.05)  # let it enter the sleep
    atexit.register(lambda: print("ATEXIT-RAN", flush=True))
    hooks = types.ModuleType("child_hooks")
    hooks.at_exit = lambda: print("HOOK-RAN")
    sys.modules["child_hooks"] = hooks
    ce.EXIT_HOOKS = ce.EXIT_HOOKS + (("child_hooks", "at_exit"),)
    logging.basicConfig(level=logging.INFO)
    print("PY-STDOUT-BUFFERED")  # a pipe: block buffered, not flushed yet
    ctypes.CDLL(None).printf(b"C-STDOUT-BUFFERED\\n")
    ce.hard_exit(0, log=logging.getLogger("child").warning)
    print("NOT-REACHED", flush=True)
    """
)

# Forked multiprocessing children that would outlive the process: one daemon
# that ignores SIGTERM (only SIGKILL ends it), one non-daemon. Both stop
# holding the test's pipes first, so a survivor cannot block subprocess.run.
# The SIGCHLD handler raises like torch's DataLoader handler does.
_CHILD_MP = textwrap.dedent(
    """
    import multiprocessing, os, signal, sys, time
    from generative_recommenders.dlrm_v4.train import clean_exit as ce

    def worker(ignore_term, ready):
        if ignore_term:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
        ready.set()
        time.sleep(600)

    def starttime(pid):
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[19]

    ctx = multiprocessing.get_context("fork")
    kids = []
    for daemon, ignore_term in ((True, True), (False, False)):
        ready = ctx.Event()
        p = ctx.Process(target=worker, args=(ignore_term, ready), daemon=daemon)
        p.start()
        assert ready.wait(30)
        kids.append(p)

    def on_sigchld(signum, frame):
        sys.stderr.write("SIGCHLD-HANDLER-RAN\\n")
        sys.stderr.flush()
        raise RuntimeError("DataLoader worker (pid(s) ?) exited unexpectedly")

    signal.signal(signal.SIGCHLD, on_sigchld)
    print("CHILDREN " + " ".join("%d:%s" % (p.pid, starttime(p.pid)) for p in kids), flush=True)
    ce.hard_exit(0)
    print("NOT-REACHED", flush=True)
    """
)

# stop_children() while a REAL SIGCHLD is pending whose handler raises, like
# torch's DataLoader handler. signal.signal() runs pending Python handlers
# before it changes the disposition, and raises, with the disposition
# unchanged, if one raises. The wrapper (the signal.signal module attribute
# that clean_exit calls) trips the signal on its first call with kill(2)
# through ctypes and then calls the real signal.signal, all from C (starmap +
# operator.call), so no eval-breaker check runs the handler in between
# (os.kill would run it itself). Children as in _CHILD_MP.
_CHILD_PENDING = textwrap.dedent(
    """
    import ctypes, itertools, json, multiprocessing, operator, os, signal, time
    from generative_recommenders.dlrm_v4.train import clean_exit as ce

    def worker(ignore_term, ready):
        if ignore_term:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
        ready.set()
        time.sleep(600)

    def starttime(pid):
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[19]

    ctx = multiprocessing.get_context("fork")
    kids = []
    for daemon, ignore_term in ((True, True), (False, False)):
        ready = ctx.Event()
        p = ctx.Process(target=worker, args=(ignore_term, ready), daemon=daemon)
        p.start()
        assert ready.wait(30)
        kids.append(p)

    runs = []

    def on_sigchld(signum, frame):
        runs.append(signum)
        raise RuntimeError("DataLoader worker (pid(s) ?) is killed by signal: Killed.")

    signal.signal(signal.SIGCHLD, on_sigchld)
    print("CHILDREN " + " ".join("%d:%s" % (p.pid, starttime(p.pid)) for p in kids), flush=True)
    libc = ctypes.CDLL(None)
    real_signal = signal.signal
    calls = []

    def tripping_signal(sig, handler):
        calls.append(dict(sig=int(sig), handler=repr(handler), raised=None))
        if len(calls) > 1:
            return real_signal(sig, handler)
        try:
            return list(itertools.starmap(operator.call, [
                (libc.kill, os.getpid(), signal.SIGCHLD),
                (real_signal, sig, handler)]))[-1]
        except RuntimeError as e:
            calls[-1]["raised"] = "RuntimeError: %s" % e
            raise

    signal.signal = tripping_signal
    try:
        n = ce.stop_children()
    finally:
        signal.signal = real_signal
    print("RESULT " + json.dumps(dict(
        calls=calls, handler_runs=len(runs), kills=n,
        sigchld_dfl=signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL)), flush=True)
    os._exit(0)  # as hard_exit: the killed children are not joined
    """
)


def _child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop(ce.ENV, None)
    return env


def _alive(pid: int, starttime: str) -> bool:
    """The same process (pid and start time) still exists and is not a zombie."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
    except OSError:
        return False
    return fields[0] not in ("Z", "X") and fields[19] == starttime


class ChildProcessTest(unittest.TestCase):
    def test_daemon_thread_in_c_call_exits_zero(self) -> None:
        t0 = time.monotonic()
        p = subprocess.run([sys.executable, "-B", "-c", _CHILD], cwd=str(_REPO), env=_child_env(),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        dt = time.monotonic() - t0
        out, err = p.stdout.decode(), p.stderr.decode()
        self.assertEqual(p.returncode, 0, err[-2000:])
        self.assertLess(dt, 60.0)  # did not wait for the blocked thread
        self.assertIn("HOOK-RAN", out)
        self.assertIn("PY-STDOUT-BUFFERED", out)
        self.assertIn("C-STDOUT-BUFFERED", out)
        self.assertNotIn("ATEXIT-RAN", out)  # os._exit: atexit skipped
        self.assertNotIn("NOT-REACHED", out)
        self.assertIn(_LINE0, err)
        self.assertNotIn("terminate called", err)

    @unittest.skipUnless(os.path.isdir("/proc/self/task"), "needs Linux /proc")
    def test_multiprocessing_children_do_not_outlive_hard_exit(self) -> None:
        p = subprocess.run([sys.executable, "-B", "-c", _CHILD_MP], cwd=str(_REPO),
                           env=_child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=120)
        out, err = p.stdout.decode(), p.stderr.decode()
        line = [x for x in out.splitlines() if x.startswith("CHILDREN ")]
        self.assertEqual(len(line), 1, err[-2000:])
        kids = [(int(k.split(":")[0]), k.split(":")[1]) for k in line[0].split()[1:]]
        self.assertEqual(len(kids), 2)
        try:
            self.assertEqual(p.returncode, 0, err[-2000:])
            self.assertNotIn("NOT-REACHED", out)
            self.assertNotIn("SIGCHLD-HANDLER-RAN", err)
            deadline = time.monotonic() + 30.0
            alive = list(kids)
            while alive and time.monotonic() < deadline:
                alive = [k for k in alive if _alive(*k)]
                if alive:
                    time.sleep(0.1)
            self.assertEqual(alive, [], "multiprocessing children outlived hard_exit")
        finally:  # never leave a sleeper behind (same pid AND start time only)
            for pid, st in kids:
                if _alive(pid, st):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass

    @unittest.skipUnless(os.path.isdir("/proc/self/task"), "needs Linux /proc")
    def test_stop_children_resets_sigchld_with_raising_handler_pending(self) -> None:
        p = subprocess.run([sys.executable, "-B", "-c", _CHILD_PENDING], cwd=str(_REPO),
                           env=_child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=120)
        out, err = p.stdout.decode(), p.stderr.decode()
        line = [x for x in out.splitlines() if x.startswith("CHILDREN ")]
        self.assertEqual(len(line), 1, err[-2000:])
        kids = [(int(k.split(":")[0]), k.split(":")[1]) for k in line[0].split()[1:]]
        self.assertEqual(len(kids), 2)
        try:
            self.assertEqual(p.returncode, 0, err[-2000:])
            res = [x for x in out.splitlines() if x.startswith("RESULT ")]
            self.assertEqual(len(res), 1, out[-2000:])
            r = json.loads(res[0][len("RESULT "):])
            calls = r["calls"]
            # The first reset raised from the pending handler (its only run)
            # and left it installed; the retry, the 2nd and last call, reset it.
            self.assertIn("is killed by signal", calls[0]["raised"] or "", r)
            self.assertEqual(r["handler_runs"], 1, r)
            self.assertEqual(len(calls), 2, r)
            self.assertIsNone(calls[1]["raised"], r)
            self.assertEqual({(c["sig"], c["handler"]) for c in calls},
                             {(int(signal.SIGCHLD), repr(signal.SIG_DFL))}, r)
            self.assertTrue(r["sigchld_dfl"], r)
            self.assertEqual(r["kills"], 2, r)
            deadline = time.monotonic() + 30.0
            alive = list(kids)
            while alive and time.monotonic() < deadline:
                alive = [k for k in alive if _alive(*k)]
                if alive:
                    time.sleep(0.1)
            self.assertEqual(alive, [], "multiprocessing children outlived stop_children")
        finally:  # never leave a sleeper behind (same pid AND start time only)
            for pid, st in kids:
                if _alive(pid, st):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass


if __name__ == "__main__":
    unittest.main()
