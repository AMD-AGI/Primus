"""End a successful streaming run with os._exit (post-RUN_STOP teardown).

Ship14 item (69) (WBND K4 part B). Kill switch: DL_TEARDOWN_HARD_EXIT=0
(default 1).

WHY: on CPython 3.12, a thread that is inside a torch binding that released
the GIL (pybind11::gil_scoped_release, e.g. THPVariable_clear while it frees a
Tensor's storage) is ended with pthread_exit() when it re-acquires the GIL
after the main thread has entered Py_FinalizeEx. The forced unwind reaches the
noexcept ~gil_scoped_release frame and the C++ runtime calls std::terminate()
("terminate called without an active exception"): the rank dies with SIGABRT
and the torch.multiprocessing spawn parent stops the other ranks, so a run
that has already logged RUN_STOP ends with rc 1. Here that thread is a daemon
thread of the input pipeline: the train pool's "dl-teardown-close" helper
(pool_close.py) dropping the last reference to the train iterator or, with
the close off, a DataLoader pin-memory thread. A process that ends with
os._exit() never runs Py_FinalizeEx, so no thread can be unwound that way.

WHAT: train_ranker._main_func calls hard_exit() on the streaming-train-eval
success path only, after its try/finally teardown (barrier and
destroy_process_group) has completed; an exception never reaches it, so a
failed run keeps its traceback and nonzero exit code. In order:

  1. the trainer's own exit hooks (EXIT_HOOKS), which os._exit would skip:
     today triton_spec._at_exit (TSPEC monitor "exit" report, recorder summary
     row and close) and, with r16's B3, l11_do8._final / hstu_do8_registry._final
     (the DO8 producer's and the dO8 registry's "final=1" counter lines; with B4
     also l11_do8._only_final, the 'only' producer's line; with r16's NIB
     triton_hstu_linear._lnmd_mask_nib_final, the NIB writers' line). A hook
     runs only if its module is loaded; each is guarded;
  2. rank 0 logs one line (EXIT_FMT);
  3. stop_children(): SIGCHLD back to SIG_DFL, then SIGKILL, without a join,
     for every live multiprocessing child of this process (the DataLoader
     workers), which the normal exit would terminate and join (below). The
     reset is tried at most twice: signal.signal() first runs any pending
     Python signal handler, and a pending SIGCHLD whose handler raises
     (torch's) makes it raise before the disposition changes; that run
     consumed the pending signal, so the second attempt resets;
  4. logging.shutdown() (flushes and closes every logging handler: the run
     log, and the MLPerf log, which mllog writes through a logging
     FileHandler), then sys.stdout / sys.stderr and the C stdio buffers are
     flushed (_exit, unlike exit, does not flush them);
  5. os._exit(code).

hard_exit returns instead, and the normal interpreter exit follows, in two
cases (rank 0 logs why):
  - DL_TEARDOWN_HARD_EXIT is off (OFF_LINE). The caller checks enabled()
    first; this is a second guard, so the off switch always means the old
    exit;
  - TunableOp tuning or record-untuned is on, a non-scored BLAS tuning
    session (SKIP_FMT). A precaution: torch documents that it writes the
    results file when the application terminates, but the lab measured
    (k4fix K1 probe, torch 2.12 + ROCm 7.2) a complete results file after
    os._exit(0), i.e. it is written before exit. The record-untuned file was
    not probed.

What os._exit skips otherwise, and why that is safe after this teardown:
  - atexit callbacks. Every one the package registers is an EXIT_HOOKS entry,
    run in step 1
    (clean_exit_test.py fails if another atexit, threading._register_atexit,
    weakref.finalize or multiprocessing.util.Finalize(exitpriority=...)
    registration appears without an EXIT_HOOKS entry). logging's own hook is
    step 4.
  - multiprocessing's _exit_function and torch's persistent-worker hooks,
    which terminate and join the DataLoader workers. Step 3 replaces them.
    Without it a worker can outlive the rank indefinitely: a worker that
    close_pools did not signal (DL_TEARDOWN_CLOSE_POOLS=0, a pool the close
    leaves open, or a close helper that has not yet set the pool's done event
    when os._exit runs) leaves its loop through its manager watchdog with the
    done event unset, so it does not cancel its result-queue join; at its own
    exit it then joins that queue's feeder thread with no timeout, and if the
    feeder is blocked on a full result pipe (the read end stays open in every
    forked sibling) it never exits. Every worker also holds the
    torch.multiprocessing spawn sentinel, a pipe the rank inherited and its
    forked workers inherit from it, so the spawn parent's join(), the launcher
    and the job step wait for every worker of every rank, not only for the
    ranks. SIGKILL does not delay a worker that is already exiting.
  - Joining non-daemon threads and finalizers at interpreter teardown: no
    output of the trainer depends on them. The run log and the MLPerf log
    flush every record, the <LOG>.metrics.jsonl sink opens and closes its
    file per record, the TSPEC recorder is line buffered and closed in step
    1, checkpoints and profiler traces are written synchronously before the
    loop returns.
  - C++ static destructors (HIP runtime, RCCL): the process group is already
    destroyed; the kernel driver releases the process's GPU and RDMA
    resources when it exits, as for any exiting process.
  - Not handled (none is set by the launcher or the gin; use
    DL_TEARDOWN_HARD_EXIT=0 when their tail matters): TensorBoard events
    buffered since the writer's last flush (TENSORBOARD_LOG_PATH, default
    off); ENABLE_PYTORCH_EXECUTION_TRACE=1 (torch's ExecutionTraceObserver
    finalizes its file in __del__); native debug logs or external tracers
    that flush from exit handlers.

Nothing here runs before the loop has returned (after RUN_STOP when MLPerf
logging is on; MLPERF_LOGGING=0 runs take the same exit). Standard library
only (torch is consulted only if this process already loaded it), so the unit
test loads it on a CPU interpreter.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing
import os
import signal
import sys
from typing import Any, Callable, Mapping, Optional, Tuple

ENV: str = "DL_TEARDOWN_HARD_EXIT"
# Values that turn the hard exit off (normal interpreter exit). Anything else,
# or the variable unset, keeps it on: the parsing of pool_close.enabled().
_OFF = ("0", "false", "off", "no")

# The trainer's own atexit callbacks, run explicitly because os._exit skips
# atexit: (module, function). A module that is not loaded in this process
# registered nothing, so its hook is skipped.
EXIT_HOOKS: Tuple[Tuple[str, str], ...] = (
    ("generative_recommenders.dlrm_v4.train.triton_spec", "_at_exit"),
    # r16 B3 (ship14 compose defect F1): the DO8 producer and the dO8 registry
    # register their "final=1" counter lines with atexit on their first event;
    # both print only (no device work) and are no-ops in a process with no event.
    ("generative_recommenders.ops.hip.l11_do8", "_final"),
    ("generative_recommenders.ops.flydsl.hstu_do8_registry", "_final"),
    # r16 B4 (DO8-only, with B3): the 'only' producer's own "final=1" line.
    ("generative_recommenders.ops.hip.l11_do8", "_only_final"),
    # r14 MX6 forward trio (DLRM_HSTU_FP8_FWD_MXFP6, default off; yambda_5b.gin
    # item (92)): ops/hip/mx6_fwd registers its "[mx6-fwd] counters at exit"
    # line with atexit at import (knob on only); print only, no device work.
    ("generative_recommenders.ops.hip.mx6_fwd", "_at_exit"),
    # Item (118) PREPROC_DEDUP (DLRM_HSTU_FP8_PREPROC_DEDUP, default off): ops/preproc_dedup registers its
    # "[preproc-dedup] counters at exit" line with atexit when it arms (knob "1" only); print only, no device work.
    ("generative_recommenders.ops.preproc_dedup", "_at_exit"),
    # r16 NIB (DLRM_HSTU_FP8_LNMD_MASK_NIB, default off; yambda_5b.gin item
    # (94)): triton_hstu_linear registers its "[lnmd-mask-nib] counters: ...
    # final=1" line with atexit at the first NIB write; print only, no device
    # work, nothing unless the knob is "1" (then also in a run that never
    # armed: armed=0 and the declined count), at most once per process.
    (
        "generative_recommenders.ops.triton.triton_hstu_linear",
        "_lnmd_mask_nib_final",
    ),
    # r14 E-RM (DLRM_HSTU_FP8_BWD_ERM, default off; yambda_5b.gin item (97)):
    # ops/triton/dy_erm registers its "erm final=1" counter line with atexit at
    # its first site (knob on only); print only, no device work, and a no-op
    # with the knob off.
    ("generative_recommenders.ops.triton.dy_erm", "_final"),
)

# Grep-able log literals (rank 0).
EXIT_FMT: str = (
    "[teardown] clean exit: os._exit({code}) after barrier/destroy_process_group "
    "(DL_TEARDOWN_HARD_EXIT=1) rank={rank}"
)
SKIP_FMT: str = "[teardown] clean exit skipped: {why}; normal interpreter exit rank={rank}"
OFF_LINE: str = (
    "[teardown] DL_TEARDOWN_HARD_EXIT=0: normal interpreter exit "
    "(atexit, finalization)"
)

Log = Callable[[str], None]


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """$DL_TEARDOWN_HARD_EXIT: on unless 0 / false / off / no."""
    raw = (os.environ if env is None else env).get(ENV, "1")
    return str(raw).strip().lower() not in _OFF


def _say(log: Optional[Log], msg: str) -> None:
    if log is None:
        return
    try:
        log(msg)
    except Exception:  # logging must never stop the exit
        pass


def native_exit_writer(torch_mod: Any = None) -> str:
    """Why this process keeps the normal exit, or "". Today: TunableOp tuning
    or record-untuned is on (the check of blas_warmup / fp8_warmup). A
    precaution: torch documents that it writes these files when the process
    terminates, but the lab measured the results file complete after
    os._exit(0) (it is written before exit); the record-untuned file was not
    probed. torch is only consulted if it is already loaded. Never raises."""
    torch = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if torch is None:
        return ""
    try:
        tun = getattr(getattr(torch, "cuda", None), "tunable", None)
        if tun is None:  # `import torch` loads it (torch 2.12); just in case
            tun = importlib.import_module("torch.cuda.tunable")
        if tun.is_enabled() and (
            tun.tuning_is_enabled()
            or getattr(tun, "record_untuned_is_enabled", lambda: False)()
        ):
            return ("TunableOp tuning or record-untuned is ON (precaution; "
                    "the lab measured the results file is written before exit)")
    except Exception:  # e.g. a CPU build without the TunableOp bindings
        pass
    return ""


def run_exit_hooks(log: Optional[Log] = None) -> int:
    """Run the EXIT_HOOKS whose module is loaded; returns how many ran. Never
    raises: a failing hook is logged and the next one still runs."""
    n = 0
    for mod_name, fn_name in EXIT_HOOKS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        try:
            getattr(mod, fn_name)()
            n += 1
        except BaseException as e:  # the exit must go ahead
            _say(log, f"[teardown] exit hook {mod_name}.{fn_name} failed: "
                      f"{type(e).__name__}: {e}")
    return n


def stop_children() -> int:
    """SIGKILL every live multiprocessing child of this process (the DataLoader
    workers), without joining; returns how many were signalled. The normal
    exit's multiprocessing._exit_function terminates and joins them, and
    os._exit skips it (module docstring: an unsignalled worker can otherwise
    hang at its own exit and hold the spawn sentinel). SIGCHLD goes back to
    SIG_DFL first, so torch's DataLoader handler cannot raise "worker ...
    killed by signal" in this thread, which would cut this loop short or stop
    the exit. That reset is tried at most twice: signal.signal() runs pending
    Python signal handlers before it changes the disposition, so a pending
    SIGCHLD whose handler raises makes the first call raise with that handler
    still installed; the run consumed the pending signal, so the second call
    resets. Never raises. Only for the way to os._exit: it kills the
    children and changes the process's SIGCHLD disposition."""
    n = 0
    try:
        for _ in range(2):
            try:
                signal.signal(signal.SIGCHLD, signal.SIG_DFL)
                break
            except Exception:  # not the main thread, no SIGCHLD, or a handler raised
                continue
        for p in multiprocessing.active_children():
            try:
                p.kill()
                n += 1
            except Exception:
                pass
    except BaseException:  # the exit must go ahead
        pass
    return n


def _flush_all() -> None:
    try:
        logging.shutdown()
    except BaseException:
        pass
    for s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        try:
            if s is not None:
                s.flush()
        except BaseException:
            pass
    # C stdio buffers: exit() flushes them, _exit() does not. glibc's
    # fflush(NULL) takes each FILE lock (exit()'s flush does not), so a native
    # thread blocked while holding one could stall here; never observed.
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except BaseException:
        pass


def hard_exit(rank: int, log: Optional[Log] = None, code: int = 0) -> None:
    """Exit hooks, one rank-0 line, stop_children(), flush, then
    os._exit(code). Call only on the success path, after the teardown barrier
    and destroy_process_group (module docstring). Returns, for the normal
    interpreter exit, only when DL_TEARDOWN_HARD_EXIT is off (rank 0 logs
    OFF_LINE) or native_exit_writer() names a reason (rank 0 logs SKIP_FMT)."""
    if not enabled():
        if rank == 0:
            _say(log, OFF_LINE)
        return
    why = native_exit_writer()
    if why:
        if rank == 0:
            _say(log, SKIP_FMT.format(why=why, rank=rank))
        return
    run_exit_hooks(log)
    if rank == 0:
        _say(log, EXIT_FMT.format(code=code, rank=rank))
    stop_children()
    _flush_all()
    os._exit(code)
