###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Watch the delayed-scaling amax and the scales it produces.

The FP8 scale update is the one place in the step that knows a tensor has gone
out of range, and in its production form it says nothing at all: a bad amax is
either clamped silently or divided by silently. This records what it saw and
what it decided, which is what turns "the loss went NaN somewhere after step
6,850" into a specific track, a specific module and a specific arithmetic path.

The three things worth recording separately, because the code treats them
differently and only one of them is guarded:

* **NaN amax.** Caught. ``sf = tl.where(amax == amax, sf, old_scale)`` keeps the
  previous scale, because ``x != x`` holds only for NaN.
* **Zero amax.** Caught, by the ``amax > 0`` test above it.
* **Infinite amax.** *Not* caught. Infinity passes both tests, so the kernel
  divides by it: ``sf = fp8_max / inf`` is exactly ``0.0``. A zero scale
  quantises every value to zero and gives the GEMM ``scale_inv = 1/0 = inf``,
  so the dequantised product is ``0 * inf = NaN``. See
  ``runs/issue220_late_bf16/scripts/_amax_guard_probe.py``, which reproduces the
  guard's arithmetic and shows the whole chain.

With ``fp8_amax_compute_algo: max`` and ``fp8_amax_history_len: 1024`` the scale
comes from the maximum over the history ring, so one infinity stays in the
window for 1024 steps. ``history_nonfinite`` is recorded for that reason: it is
the difference between a transient and a run that cannot recover.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import torch

ENV_WATCH = "PRIMUS_FP8_AMAX_WATCH"
ENV_WATCH_DIR = "PRIMUS_FP8_AMAX_WATCH_DIR"
ENV_HEARTBEAT = "PRIMUS_FP8_AMAX_WATCH_HEARTBEAT"

_TRACKS = ("input", "weight", "grad")

_ENABLED: Optional[bool] = None
_SINK = None
_STEP = 0
_LAST_HEARTBEAT = -1
_LOCAL_SNAPSHOT: Optional[torch.Tensor] = None
_FAILED = False
_ALARMED: set = set()


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get(ENV_WATCH, "0") == "1"
    return _ENABLED


def _heartbeat_every() -> int:
    try:
        return max(1, int(os.environ.get(ENV_HEARTBEAT, "100")))
    except (TypeError, ValueError):
        return 100


def _sink():
    """Per-rank file when a directory is given, else stderr.

    Never ``print``: Primus replaces it with a shim that drops the ``file``
    keyword and logs below the default level, so records sent that way create
    their file and then vanish into it.
    """
    global _SINK
    if _SINK is not None:
        return _SINK
    directory = os.environ.get(ENV_WATCH_DIR, "").strip()
    if not directory:
        return sys.stderr
    os.makedirs(directory, exist_ok=True)
    rank = os.environ.get("RANK", "0")
    _SINK = open(os.path.join(directory, f"amax_watch.rank{rank}.jsonl"), "a", buffering=1)
    _emit({"event": "watch_open", "pid": os.getpid(), "heartbeat": _heartbeat_every()})
    return _SINK


def _emit(record: dict) -> None:
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    record.setdefault("rank", int(os.environ.get("RANK", "0")))
    sink = _SINK if _SINK is not None else _sink()
    sink.write("[fp8-amax] " + json.dumps(record) + "\n")
    sink.flush()


def set_step(step: int) -> None:
    global _STEP
    _STEP = int(step)


def note_local_amaxes(staged_3n: torch.Tensor) -> None:
    """Keep this rank's own amaxes before the all-reduce overwrites them.

    Without this the record cannot say whether this rank spiked or merely
    received the maximum of a rank that did, and that distinction is the whole
    reason the 16-rank failure rate might differ from the 8-rank one.
    """
    global _LOCAL_SNAPSHOT
    if not enabled() or _FAILED:
        return
    try:
        _LOCAL_SNAPSHOT = staged_3n.detach().clone()
    except Exception:  # noqa: BLE001
        _LOCAL_SNAPSHOT = None


@torch.no_grad()
def observe(
    registry,
    reduced_3n: torch.Tensor,
    scales_before: Optional[torch.Tensor],
    scales_after: torch.Tensor,
    path: str,
) -> None:
    """Record one scale update.

    ``reduced_3n`` and the scale tensors are (3, N): three tracks by however
    many FP8 modules the model has. One host transfer per call, which is once
    per step rather than once per GEMM.
    """
    global _FAILED
    if not enabled() or _FAILED:
        return
    try:
        amax = reduced_3n.detach().float()
        after = scales_after.detach().float()
        local = _LOCAL_SNAPSHOT

        is_nan = torch.isnan(amax)
        is_inf = torch.isinf(amax)
        stats = {
            "nan": is_nan.sum(dim=1),
            "inf": is_inf.sum(dim=1),
            "zero_amax": (amax == 0).sum(dim=1),
            "zero_scale": (after == 0).sum(dim=1),
            # The finite maximum, so a single infinity does not hide the scale
            # the rest of the model is actually running at.
            "max_finite_amax": torch.where(
                torch.isfinite(amax), amax, torch.zeros_like(amax)
            ).amax(dim=1),
        }
        if scales_before is not None:
            stats["scale_kept"] = (after == scales_before.detach().float()).sum(dim=1)
        if local is not None and local.shape == amax.shape:
            local_f = local.float()
            # Did this rank supply the maximum, or receive it from another?
            stats["is_local_max"] = (
                torch.isclose(local_f.amax(dim=1), amax.amax(dim=1))
            ).to(torch.int64)
            stats["nan_local"] = torch.isnan(local_f).sum(dim=1)
            stats["inf_local"] = torch.isinf(local_f).sum(dim=1)

        history_bad = 0
        history = getattr(registry, "amax_history", None)
        if history is not None:
            history_bad = int((~torch.isfinite(history.detach())).sum().item())

        # One transfer for the lot.
        host = {name: value.cpu().tolist() for name, value in stats.items()}

        bad_tracks = [
            t
            for i, t in enumerate(_TRACKS)
            if host["nan"][i] or host["inf"][i] or host["zero_scale"][i]
        ]
        if bad_tracks:
            _alarm(host, bad_tracks, history_bad, path, registry)
        elif _STEP - _LAST_HEARTBEAT >= _heartbeat_every():
            _heartbeat(host, history_bad, path)
    except Exception as exc:  # noqa: BLE001
        _FAILED = True
        try:
            _emit({"event": "watch_failed", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:  # noqa: BLE001
            pass


def _per_track(host: dict, field: str) -> dict:
    values = host.get(field)
    if values is None:
        return {}
    return {t: values[i] for i, t in enumerate(_TRACKS)}


def _alarm(host: dict, bad_tracks: list, history_bad: int, path: str, registry) -> None:
    """One record per (track, kind) so a 1024-step-long failure stays readable."""
    kinds = tuple(
        k for k in ("nan", "inf", "zero_scale") if any(host[k][i] for i in range(len(_TRACKS)))
    )
    ident = (tuple(bad_tracks), kinds)
    repeat = ident in _ALARMED
    _ALARMED.add(ident)
    _emit(
        {
            "event": "amax_alarm",
            "step": _STEP,
            "path": path,
            "tracks": bad_tracks,
            "repeat": repeat,
            "nan_amax": _per_track(host, "nan"),
            "inf_amax": _per_track(host, "inf"),
            "zero_amax": _per_track(host, "zero_amax"),
            "zero_scale": _per_track(host, "zero_scale"),
            "scale_kept": _per_track(host, "scale_kept"),
            "max_finite_amax": _per_track(host, "max_finite_amax"),
            "nan_local": _per_track(host, "nan_local"),
            "inf_local": _per_track(host, "inf_local"),
            # 1 means this rank held the maximum for that track, so the spike is
            # local; 0 means it arrived from another rank through the MAX
            # all-reduce.
            "is_local_max": _per_track(host, "is_local_max"),
            # Non-zero means the poison is in the history ring, and with
            # algo=max it will keep setting the scale until it ages out.
            "history_nonfinite": history_bad,
            "history_len": getattr(registry, "history_len", None),
            "algo": getattr(registry, "algo", None),
            "reduce_amax": getattr(registry, "reduce_amax", None),
        }
    )


def _heartbeat(host: dict, history_bad: int, path: str) -> None:
    global _LAST_HEARTBEAT
    _LAST_HEARTBEAT = _STEP
    _emit(
        {
            "event": "amax_ok",
            "step": _STEP,
            "path": path,
            "max_finite_amax": _per_track(host, "max_finite_amax"),
            "zero_amax": _per_track(host, "zero_amax"),
            "is_local_max": _per_track(host, "is_local_max"),
            "history_nonfinite": history_bad,
        }
    )


def reset() -> None:
    """Drop all state and re-read the environment. For tests."""
    global _ENABLED, _SINK, _STEP, _LAST_HEARTBEAT, _LOCAL_SNAPSHOT, _FAILED
    _ENABLED = None
    _SINK = None
    _STEP = 0
    _LAST_HEARTBEAT = -1
    _LOCAL_SNAPSHOT = None
    _FAILED = False
    _ALARMED.clear()
