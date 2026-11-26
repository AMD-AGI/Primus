"""Close the persistent input pools after RUN_STOP (teardown hygiene).

WBND SHIP-13 candidate. Kill switch: DL_TEARDOWN_CLOSE_POOLS=0 (default 1).

WHY: the streaming loop keeps its persistent train and eval DataLoader pools
(worker processes, plus a daemon pin-memory thread per pool when pin_memory is
on) alive until interpreter exit. torch 2.12 registers atexit cleanup only for
persistent workers (``_clean_up_worker``), not for the pin-memory thread, and
``_shutdown_workers()`` is a no-op once ``_utils.python_exit_status`` is set.
At a mid-window convergence stop the chained train pool still has up to
PREFETCH_FACTOR x NUM_WORKERS batches outstanding, so its pin thread can still
be pinning them while the process tears down. One ship12 run aborted at exit
after a successful RUN_STOP (rank 1 SIGABRT); this is the suspected race, not
a proven cause.

WHAT: ``close_pools`` runs once, after the train/eval loop has returned
(RUN_STOP already logged), on every rank, before the teardown barrier and
``destroy_process_group`` (train_ranker.py). For each pool the loop recorded:

  1. a loader wrapper with ``teardown_close()`` (``_PrefetchingWindowLoader``)
     stops its own background threads (the single-pool pusher, or the
     double-buffered ``_prepare`` threads), then shuts its pools down;
  2. a persistent DataLoader: ``_iterator._shutdown_workers()``. torch joins
     the pin-memory thread first, then stops and joins the workers;
  3. the references are dropped (``DataLoader._iterator = None``).

Every wait is bounded by ``timeout_s``. ``_shutdown_workers()`` runs on a
helper thread because its pin-thread join has no timeout. Each step is
wrapped: a failure is logged, never raised. A pool that is already shut down
is skipped, so a second call is a no-op. Rank 0 logs exactly one line,
``[teardown] input pools closed: N in X ms``, where N counts the worker pools
this call shut down.

Nothing here reads data, draws RNG, touches a tensor or runs a collective.
Stdlib only, duck-typed on torch's private iterator attributes ``_shutdown``
and ``_shutdown_workers``, so the unit test can load it on a CPU interpreter.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple

ENV: str = "DL_TEARDOWN_CLOSE_POOLS"
# Bound of each wait (thread join, one pool's shutdown), in seconds.
DEFAULT_TIMEOUT_S: float = 5.0
# Values that turn the close off. Anything else, or the variable unset, keeps
# it on (as $DLRM_HSTU_FP8_GC_FREEZE is parsed in train/utils.py).
_OFF = ("0", "false", "off", "no")

# Grep-able log literals (the falsifier census counts these per arm).
DONE_FMT: str = "[teardown] input pools closed: {n} in {ms:.0f} ms"
OFF_LINE: str = (
    "[teardown] DL_TEARDOWN_CLOSE_POOLS=0: input pools not closed "
    "(left to interpreter exit)"
)

Log = Callable[[str], None]


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """$DL_TEARDOWN_CLOSE_POOLS: on unless 0 / false / off / no."""
    raw = (os.environ if env is None else env).get(ENV, "1")
    return str(raw).strip().lower() not in _OFF


def _say(log: Optional[Log], msg: str) -> None:
    if log is None:
        return
    try:
        log(msg)
    except Exception:  # logging must never fail the teardown
        pass


def shutdown_iterator(
    it: Any, timeout_s: float = DEFAULT_TIMEOUT_S, warn: Optional[Log] = None
) -> int:
    """Bounded ``it._shutdown_workers()`` (pin-memory thread first, then the
    workers, in torch's order). Returns 1 if this call shut a live worker pool
    down, 0 if there was nothing to do, or if it failed or timed out (warned).
    """
    fn = getattr(it, "_shutdown_workers", None)
    if fn is None or getattr(it, "_shutdown", False):
        return 0
    err: List[BaseException] = []

    def _run() -> None:
        try:
            fn()
        except BaseException as e:  # reported by the caller's thread below
            err.append(e)

    try:
        th = threading.Thread(target=_run, name="dl-teardown-close", daemon=True)
        th.start()
        th.join(timeout_s)
    except Exception as e:
        _say(warn, f"[teardown] pool shutdown not run: {type(e).__name__}: {e}")
        return 0
    if th.is_alive():
        _say(
            warn,
            f"[teardown] pool shutdown still running after {timeout_s:.1f} s "
            f"({type(it).__name__}); continuing",
        )
        return 0
    if err:
        _say(
            warn,
            f"[teardown] pool shutdown failed: {type(err[0]).__name__}: {err[0]}",
        )
        return 0
    return 1


def _close_one(obj: Any, timeout_s: float, warn: Optional[Log]) -> int:
    closer = getattr(obj, "teardown_close", None)
    if callable(closer):  # _PrefetchingWindowLoader (train pools)
        return int(closer(timeout_s=timeout_s, warn=warn) or 0)
    if hasattr(obj, "_shutdown_workers"):  # a DataLoader iterator
        return shutdown_iterator(obj, timeout_s, warn)
    it = getattr(obj, "_iterator", None)  # a persistent DataLoader (eval pool)
    if it is None:
        return 0
    n = shutdown_iterator(it, timeout_s, warn)
    try:
        obj._iterator = None
    except Exception:
        pass
    return n


def close_pools(
    objs: Optional[Iterable[Any]],
    *,
    rank: int = 0,
    log: Optional[Log] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Tuple[int, float]:
    """Close every pool in ``objs`` (see the module docstring); returns
    ``(pools shut down, seconds)``. ``log`` gets the one summary line on rank
    0 and a warning per failure on any rank. A list ``objs`` is emptied, which
    drops its references to the pools."""
    t0 = time.perf_counter()

    def warn(msg: str) -> None:
        _say(log, f"{msg} rank={rank}")

    n = 0
    seen: List[Any] = []
    try:
        items = list(objs) if objs is not None else []
    except Exception as e:
        _say(warn, f"[teardown] pool list unreadable: {type(e).__name__}: {e}")
        items = []
    for obj in items:
        if obj is None or any(obj is s for s in seen):
            continue
        seen.append(obj)
        try:
            n += _close_one(obj, timeout_s, warn)
        except Exception as e:
            _say(
                warn,
                f"[teardown] closing {type(obj).__name__} failed: "
                f"{type(e).__name__}: {e}",
            )
    del seen, items
    if isinstance(objs, list):
        objs.clear()
    dt = time.perf_counter() - t0
    if rank == 0:
        _say(log, DONE_FMT.format(n=n, ms=dt * 1e3))
    return n, dt
