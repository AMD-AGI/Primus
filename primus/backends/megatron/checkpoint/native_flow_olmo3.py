###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 policy of the SFT conversion hook (``01_convert_checkpoints.py``).

The shared hook detects the family and calls :func:`run_native_flow` for OLMo-3; the other
families keep the hook's own flow. This module holds what is OLMo-3 specific:

* the training-layout check (the converter writes TP = PP = EP = 1 only);
* the converter options taken from the training config;
* the conversion record ``primus_native_convert.json`` that a reused checkpoint must match;
* the rank-0-converts / others-wait flow with fail-fast for the waiting nodes;
* an OLMo-3 source given as a hub id: rank 0 gets a complete snapshot, the other nodes wait for it.

No Megatron import at module level (the hook process imports this before Megatron is on the path).
"""

import hashlib
import json
import os
import re
import shutil
import socket
import time
from pathlib import Path

from primus.core.utils.arg_utils import as_int, is_unset

WAIT_TIMEOUT = 3600  # seconds; native_ckpt_wait_timeout overrides it
CONVERT_RECORD = "primus_native_convert.json"
TRACKER_FILE = "latest_checkpointed_iteration.txt"  # Megatron's marker of a complete checkpoint
# Bump when the converter changes the saved weights or args, so old checkpoints are not reused.
CONVERTER_VERSION = 1
# Record keys kept for information only: not compared when a checkpoint is reused. Every node
# computes the record from its own view of the source, and the realpath of hf_path differs between
# nodes that mount the HF directory at another path (symlinks, a node-local cache); the config
# sha256 and the weights fingerprint already identify the source.
_INFO_ONLY_KEYS = ("hf_path",)
_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.safetensors.index.json", "*.bin.index.json")


def config_int(opts: dict, key: str, default: int | None = None) -> int | None:
    """``opts[key]`` as an int (the shared ``as_int``: YAML / CLI values may be strings), else a
    clear ValueError. Unset values (the shared ``is_unset``: None, "", null / ~) give
    ``default``."""
    value = opts.get(key)
    if is_unset(value):
        return default
    return as_int(value, key)


def check_single_shard_layout(tp, pp, ep) -> None:
    """Refuse a layout other than TP = PP = EP = 1: the OLMo-3 converter runs in a 1-rank world
    and writes a single-shard checkpoint, which is not resharded at load."""
    if tuple(int(x) for x in (tp, pp, ep)) != (1, 1, 1):
        raise ValueError(
            "the OLMo-3 converter writes single-shard TP=PP=EP=1 checkpoints (it runs in a 1-rank "
            f"world); got TP={tp} PP={pp} EP={ep}"
        )


def single_shard_conflict(opts: dict) -> str | None:
    """The error message when the run trains with TP, PP or EP > 1, else None."""
    tp = config_int(opts, "tensor_model_parallel_size", 1)
    pp = config_int(opts, "pipeline_model_parallel_size", 1)
    ep = config_int(opts, "expert_model_parallel_size", 1)
    try:
        check_single_shard_layout(tp, pp, ep)
        return None
    except ValueError as e:
        return (
            f"Native OLMo-3 conversion: {e}, but this run trains with "
            f"tensor_model_parallel_size={tp}, pipeline_model_parallel_size={pp}, "
            f"expert_model_parallel_size={ep}; a legacy torch checkpoint is not resharded at load. "
            "Convert once with TP=PP=EP=1 and set pretrained_checkpoint to a checkpoint in the training "
            "layout. The torch -> torch_dist reshard step in docs/04-technical-guides/native-sft-lora.md "
            "is UNVERIFIED (experimental) for OLMo-3."
        )


def converter_options(opts: dict) -> dict:
    """Training-config settings the OLMo-3 converter takes.

    * ``make_vocab_size_divisible_by`` sets the padded vocab rows of the embedding and output
      weights, so it is always passed (Megatron's default 128 when the config does not set it).
    * ``rope_layout`` (``olmo3_rope_layout``): stored in the checkpoint args. ``auto`` is refused
      here (:func:`requested_rope_layout`); convert() refuses a layout that disagrees with the HF
      config.
    """
    extra = {"make_vocab_size_divisible_by": config_int(opts, "make_vocab_size_divisible_by", 128)}
    layout = requested_rope_layout(opts)
    if layout is not None:
        extra["rope_layout"] = layout
    return extra


def requested_rope_layout(opts: dict) -> str | None:
    """``olmo3_rope_layout`` for a conversion, normalised once (spaces stripped, lower case), so the
    converter and the conversion record get the same value: ``per_type``, ``yarn_all`` or None
    (unset). ``auto`` and other values raise here, at option parsing, before any lock is taken."""
    raw = opts.get("olmo3_rope_layout")
    if is_unset(raw):
        return None
    layout = str(raw).strip().lower()
    if layout == "auto":
        raise ValueError("olmo3_rope_layout must be per_type or yarn_all for a conversion, not auto")
    if layout not in ("per_type", "yarn_all"):
        raise ValueError(f"olmo3_rope_layout must be per_type or yarn_all, got {raw!r}")
    return layout


_SAMPLE_BLOCKS, _SAMPLE_BYTES = 16, 64 * 1024


def _sampled_digest(path: Path, size: int) -> bytes:
    """sha256 of ``_SAMPLE_BLOCKS`` blocks of ``_SAMPLE_BYTES`` spread evenly over the file (the
    first and the last included; the whole file when it is smaller), so ~1 MiB is read per file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        if size <= _SAMPLE_BLOCKS * _SAMPLE_BYTES:
            digest.update(fh.read())
        else:
            step = (size - _SAMPLE_BYTES) // (_SAMPLE_BLOCKS - 1)
            for k in range(_SAMPLE_BLOCKS):
                fh.seek(k * step)
                digest.update(fh.read(_SAMPLE_BYTES))
    return digest.digest()


def weights_fingerprint(hf_path: str) -> str:
    """sha256 over the weight files of ``hf_path``: per file its name, its size and its content
    (index files: all of it; weight files: 16 blocks of 64 KiB spread over the file, see
    :func:`_sampled_digest`, so a 7B model costs ~1 MiB of reads per shard, not ~15 GB).

    Two checkpoints of one architecture (base vs. instruct, two training snapshots) have the same
    index files and shard sizes but other bytes in every block, so they differ here. Limit: two
    weight sets that differ only outside the sampled blocks are not told apart."""
    root = Path(hf_path)
    digest = hashlib.sha256()
    files = sorted({p for g in _WEIGHT_GLOBS for p in root.glob(g) if p.is_file()})
    for p in files:
        size = p.stat().st_size
        digest.update(p.name.encode() + b"\0" + str(size).encode() + b"\0")
        if p.name.endswith(".index.json"):
            digest.update(hashlib.sha256(p.read_bytes()).digest())
        else:
            digest.update(_sampled_digest(p, size))
        digest.update(b"\0")
    return digest.hexdigest()


def _record_rope_layout(hf_path: str, opts: dict) -> str:
    """The concrete ``olmo3_rope_layout`` convert() saves, as convert() resolves it (the config's
    per-layer-type layout, else the requested layout, else ``per_type``); ``auto`` is refused as in
    convert(). No Megatron import."""
    from primus.backends.megatron.checkpoint import native_loader_olmo3 as conv

    requested = requested_rope_layout(opts)
    cfg = conv.common.read_hf_config(hf_path)
    full, cfg_layout, _ = conv._rope_config(cfg)
    if cfg_layout is not None:
        if requested is not None and requested != cfg_layout:
            raise ValueError(
                f"OLMo-3 config declares olmo3_rope_layout {cfg_layout!r}, but {requested!r} was requested"
            )
        return cfg_layout
    if requested == "yarn_all":
        yarn = conv._yarn_params(full, cfg)
        if yarn is None or yarn["factor"] <= 1.0:
            raise ValueError("olmo3_rope_layout 'yarn_all' needs YaRN rope parameters (factor > 1)")
    return requested or "per_type"


def convert_record(hf_path: str, opts: dict, dtype: str) -> dict:
    """The OLMo-3 conversion settings and source identity of a checkpoint."""
    cfg = Path(hf_path) / "config.json"
    return {
        "family": "olmo3",
        "converter_version": CONVERTER_VERSION,
        "hf_path": os.path.realpath(hf_path),
        "hf_config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg.is_file() else None,
        "hf_weights_fingerprint": weights_fingerprint(hf_path),
        "dtype": dtype,
        "make_vocab_size_divisible_by": converter_options(opts)["make_vocab_size_divisible_by"],
        "olmo3_rope_layout": _record_rope_layout(hf_path, opts),
    }


def stale_checkpoint_reason(megatron_path: Path, want: dict) -> str | None:
    """Why the OLMo-3 checkpoint at ``megatron_path`` cannot be reused for ``want``, else None."""
    record = Path(megatron_path) / CONVERT_RECORD
    if not record.is_file():
        return f"it has no {CONVERT_RECORD} (its conversion settings are unknown)"
    try:
        have = json.loads(record.read_text())
    except (OSError, ValueError) as e:
        return f"its {CONVERT_RECORD} is unreadable ({e})"
    diff = {k: (have.get(k), v) for k, v in want.items() if k not in _INFO_ONLY_KEYS and have.get(k) != v}
    if not diff:
        return None
    return "it was converted with other settings: " + ", ".join(
        f"{k}: {a!r} (checkpoint) vs {b!r} (this run)" for k, (a, b) in sorted(diff.items())
    )


# ---------------------------------------------------------------------------
# failure marker: rank 0 writes <out>.failed when it stops, so the other nodes stop too
# ---------------------------------------------------------------------------
LAUNCH_ID_KEYS = (
    "PRIMUS_LAUNCH_ID",
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    "SLURM_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _launch_id() -> str | None:
    """An id that all nodes of one launch share, joined from every variable that is set (in the
    order of ``LAUNCH_ID_KEYS``; no source wins over another), else None.

    ``PRIMUS_LAUNCH_ID`` (set by the caller), ``SLURM_JOB_ID`` with ``SLURM_STEP_ID`` and
    ``SLURM_RESTART_COUNT`` (another srun step or a requeue is another launch),
    ``TORCHELASTIC_RUN_ID`` and ``MASTER_ADDR:MASTER_PORT``. A later launch can have the same id
    (the same address and port outside a scheduler, a retry in one allocation), so a marker also
    has to be newer than the node's wait or survive rank 0's first step (:class:`FailureWatch`)."""
    parts = [f"{k}={os.environ[k]}" for k in LAUNCH_ID_KEYS if os.environ.get(k)]
    return ";".join(parts) if parts else None


def failed_marker(megatron_path) -> Path:
    return Path(f"{megatron_path}.failed")


def write_failure(megatron_path, reason: str) -> None:
    """Rank 0: record why it stopped (best effort; a failed write leaves the timeout)."""
    try:
        Path(megatron_path).parent.mkdir(parents=True, exist_ok=True)
        failed_marker(megatron_path).write_text(
            json.dumps({"reason": reason, "launch": _launch_id(), "time": time.time()}) + "\n"
        )
    except OSError:
        pass


def _storage_now(megatron_path, node_rank: int) -> float | None:
    """The current time of the storage that holds ``<out>.failed`` (the mtime of a file this node
    touches next to it), so it compares with the marker's mtime without clock skew between nodes.
    None if the file cannot be written."""
    stamp = Path(f"{megatron_path}.waiting.node{node_rank}")
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()
        return stamp.stat().st_mtime
    except OSError:
        return None
    finally:
        try:
            stamp.unlink(missing_ok=True)
        except OSError:
            pass


# A marker written before a waiting node took ``since`` (or in the same mtime second) is not known
# to be this launch's: it can be a stale marker of an earlier launch (also one with the same launch
# id, e.g. a retry inside one SLURM allocation or a constant PRIMUS_LAUNCH_ID). Rank 0 removes any
# marker as its first step (:func:`clear_failure`), so such a marker that is still in place,
# unchanged, this many seconds after the waiting node first saw it is rank 0's (rank 0 stopped
# early): the waiting node stops. The grace covers a rank 0 that starts later than the other nodes.
MARKER_GRACE = 600


def clear_failure(megatron_path) -> None:
    """Rank 0, first step: remove ``<out>.failed`` (of an earlier launch)."""
    try:
        failed_marker(megatron_path).unlink(missing_ok=True)
    except OSError:
        pass


def _marker(megatron_path):
    """``(reason, mtime, launch)`` of ``<out>.failed`` when it is readable and is not of another
    launch (the launch ids differ when both are set), else None."""
    marker = failed_marker(megatron_path)
    try:
        data = json.loads(marker.read_text())
        mtime = marker.stat().st_mtime
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    mine, theirs = _launch_id(), data.get("launch")
    if mine is not None and theirs is not None and mine != theirs:
        return None
    return str(data.get("reason")), mtime, theirs


def read_failure(megatron_path, since: float | None) -> str | None:
    """The reason in ``<out>.failed`` when the storage wrote it after ``since`` (a storage time
    from :func:`_storage_now`, so clock skew between nodes does not matter) and it is not of
    another launch, else None. An older marker is handled by :class:`FailureWatch`."""
    found = _marker(megatron_path)
    if found is None or since is None or found[1] <= since:
        return None
    return found[0]


class FailureWatch:
    """A waiting node's view of ``<out>.failed``; :meth:`check` returns rank 0's reason or None.

    * A marker the storage wrote after ``since`` (:func:`read_failure`) stops the node at once.
    * A marker written before ``since`` (or in the same mtime second, or with no storage time) is
      taken only when it is still in place and unchanged ``grace`` seconds after this node first
      saw it: rank 0 removes any marker as its first step (:func:`clear_failure`), so a stale
      marker of an earlier launch (also one with the same launch id) is gone by then.
    * A marker of another launch id is never taken.
    """

    def __init__(self, megatron_path, since: float | None, grace: float | None = None):
        self.path = megatron_path
        self.since = since
        self.grace = MARKER_GRACE if grace is None else grace
        self._seen = None  # (mtime, first seen on the monotonic clock) of an older marker

    def check(self) -> str | None:
        found = _marker(self.path)
        if found is None:
            self._seen = None
            return None
        reason, mtime, _ = found
        if self.since is not None and mtime > self.since:
            return reason
        now = time.monotonic()
        if self._seen is None or self._seen[0] != mtime:
            self._seen = (mtime, now)
        if now - self._seen[1] >= self.grace:
            return (
                f"{reason} (from {failed_marker(self.path)}, written before this node started to wait "
                f"and not cleared by rank 0 within {self.grace:g} s)"
            )
        return None


class ConversionFailed(RuntimeError):
    """Rank 0 stopped (a reason this node computed itself or read from ``<out>.failed``)."""


def output_not_empty_reason(megatron_path) -> str | None:
    """Rank 0 refuses to convert into a non-empty ``<out>`` without a checkpoint tracker (it could
    be anything); the waiting nodes compute the same reason."""
    megatron_path = Path(megatron_path)
    tracker = megatron_path / TRACKER_FILE
    try:
        busy = megatron_path.is_dir() and not tracker.exists() and any(megatron_path.iterdir())
    except OSError:
        busy = False
    if busy:
        return (
            f"{megatron_path} exists but holds no checkpoint (no {TRACKER_FILE}). "
            "Delete it or set native_ckpt_out to a new directory."
        )
    return None


def wait_for_checkpoint(
    megatron_path: Path,
    want: dict,
    timeout: float,
    poll: float = 5.0,
    node_rank: int = 1,
    since: float | None = None,
) -> None:
    """Other nodes: wait until the checkpoint is complete and matches ``want``.

    Stops early (``ConversionFailed``) when rank 0 stops, computing rank 0's refusals itself where
    it can, so a marker written before this node looked is not needed:

    * the checkpoint has a tracker, no conversion lock, and does not match;
    * ``<out>`` is not empty and has no tracker, and no lock (rank 0 refuses to write into it);
    * rank 0 wrote ``<out>.failed`` for this launch (:class:`FailureWatch`), e.g. the lock is held
      by another launch or the conversion raised.

    Else waits at most ``timeout`` seconds (monotonic clock); the TimeoutError says what was seen.
    No ``.done`` marker is read, so a ``.done`` of another launch neither releases nor blocks."""
    megatron_path = Path(megatron_path)
    tracker = megatron_path / TRACKER_FILE
    lock_file = lock_path(megatron_path)
    start = time.monotonic()
    if since is None:
        since = _storage_now(megatron_path, node_rank)
    watch = FailureWatch(megatron_path, since)
    while True:
        has_tracker = tracker.exists()  # once per poll, so the reason and the branch agree
        reason = stale_checkpoint_reason(megatron_path, want) if has_tracker else "no checkpoint yet"
        if reason is None:
            return
        locked = lock_file.exists()
        if has_tracker and not locked:
            raise ConversionFailed(
                f"the checkpoint at {megatron_path} cannot be reused: {reason} (rank 0 stops too)"
            )
        busy = output_not_empty_reason(megatron_path)
        if busy and not locked:
            raise ConversionFailed(f"{busy} (rank 0 stops too)")
        failure = watch.check()
        if failure is not None:
            raise ConversionFailed(f"rank 0 failed to convert the checkpoint: {failure}")
        if time.monotonic() - start >= timeout:
            raise TimeoutError(
                f"Timeout after {timeout:g} s waiting for rank 0 to prepare {megatron_path}: {reason}; "
                f"lock {lock_file} {'present' if locked else 'absent'}; no {failed_marker(megatron_path)} "
                "for this launch. Check rank 0's log; a slow conversion needs a larger native_ckpt_wait_timeout."
            )
        time.sleep(poll)


# ---------------------------------------------------------------------------
# conversion lock: one owner per output, stale locks of a dead process are taken over
# ---------------------------------------------------------------------------
def lock_path(megatron_path) -> Path:
    return Path(f"{megatron_path}.converting.lock")


def partial_path(megatron_path) -> Path:
    """Rank 0 converts into this directory and renames it to ``<out>`` when the checkpoint and
    its record are complete, so ``<out>`` never holds a partial checkpoint."""
    return Path(f"{megatron_path}.partial")


def _pid_namespace() -> str | None:
    """This process's PID namespace (``/proc/self/ns/pid``), so a pid is compared only with pids
    of the same namespace (two containers on one host have the same host name, not the same pids)."""
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (OSError, ValueError, TypeError):  # PermissionError is an OSError
        return True  # cannot tell: treat it as alive
    return True


def _lock_is_stale(held: dict) -> bool:
    """True only when the owner is certainly not running: same host and PID namespace, and its pid
    is dead or is this process's pid (this process does not hold the lock yet: a relaunch in a
    container with the same pid). Any other owner (another host or namespace, unreadable) counts
    as running."""
    if held.get("host") != socket.gethostname() or not held.get("pid"):
        return False
    if held.get("pidns") != _pid_namespace():
        return False
    try:
        if int(held["pid"]) == os.getpid():
            return True
    except (TypeError, ValueError):
        return False
    return not _pid_alive(held["pid"])


_LOCK_TOKENS: dict = {}  # lock path -> the token this process wrote into it


# creation attempts in acquire_lock: each failed one found a stale lock and took it aside (or saw
# another process take it aside first), then the next one tries to create the lock again
_LOCK_ATTEMPTS = 3


def acquire_lock(megatron_path) -> None:
    """Rank 0: create ``<out>.converting.lock`` with O_EXCL and the owner (host, PID namespace, pid,
    launch id, time, a random token). A lock whose owner is certainly not running
    (:func:`_lock_is_stale`) is taken over: it is first renamed to a unique name (only one process
    can rename it), then checked again; if it changed in between (a new owner), it is put back and
    this process stops. Any other existing lock raises ``ConversionFailed`` naming the owner and
    the file to delete, so two launches never convert into the same output."""
    lock = lock_path(megatron_path)
    token = os.urandom(8).hex()
    owner = {
        "host": socket.gethostname(),
        "pidns": _pid_namespace(),
        "pid": os.getpid(),
        "launch": _launch_id(),
        "time": time.time(),
        "token": token,
    }
    for _ in range(_LOCK_ATTEMPTS):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                raw = lock.read_text()
                held = json.loads(raw)
            except (OSError, ValueError):
                raw, held = None, {}
            if raw is None or not isinstance(held, dict) or not _lock_is_stale(held):
                raise ConversionFailed(
                    f"another conversion holds {lock} (owner: {held or 'unknown'}). If no conversion "
                    "is running (e.g. a killed launch on another node), delete the lock and launch again."
                )
            aside = Path(f"{lock}.stale.{token}")
            try:
                os.rename(lock, aside)  # atomic: one process takes the stale lock aside
            except FileNotFoundError:
                continue  # another process took it aside first: try to create again
            try:
                same = aside.read_text() == raw
            except OSError:
                same = False
            if not same:  # a new owner wrote the lock after it was read: give it back
                try:
                    os.link(aside, lock)
                except OSError:
                    pass
                aside.unlink(missing_ok=True)
                raise ConversionFailed(
                    f"another conversion took {lock} while this one checked it; launch again."
                )
            aside.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(owner) + "\n")
        _LOCK_TOKENS[str(lock)] = token
        return
    raise ConversionFailed(
        f"could not take the conversion lock {lock} in {_LOCK_ATTEMPTS} attempts (each found a stale "
        "lock and took it aside, or saw another process take it aside first); launch again."
    )


def release_lock(megatron_path) -> None:
    """Remove the lock if this process wrote it (its token is in the file)."""
    lock = lock_path(megatron_path)
    token = _LOCK_TOKENS.pop(str(lock), None)
    if token is None:
        return
    try:
        held = json.loads(lock.read_text())
    except (OSError, ValueError):
        return
    if isinstance(held, dict) and held.get("token") == token:
        lock.unlink(missing_ok=True)


def run_native_flow(
    hf_path: str, megatron_path, opts: dict, convert, dtype: str, node_rank: int, log
) -> None:
    """Rank 0 converts, the other nodes wait.

    Rank 0 takes ``<out>.converting.lock`` (:func:`acquire_lock`), converts into ``<out>.partial``
    (``convert(hf_path, str(partial), opts)``), writes ``primus_native_convert.json`` there and
    renames the directory to ``<out>``. So ``<out>`` holds a complete checkpoint with its record or
    nothing; a crash leaves only ``<out>.partial``, which the next lock owner deletes. A checkpoint
    with a tracker is reused only if its record matches this run; else rank 0 stops with what to do.
    When rank 0 stops (mismatch, lock held, the conversion raises) it writes ``<out>.failed`` with
    the reason, and the other nodes stop with it instead of waiting for ``native_ckpt_wait_timeout``.

    ``log`` has ``info`` / ``error`` / ``success`` callables. Raises ``ConversionFailed`` (rank 0
    mismatch or lock held, any node) or the conversion's exception (rank 0) or ``TimeoutError``
    (other nodes).
    """
    megatron_path = Path(megatron_path)
    tracker = megatron_path / TRACKER_FILE
    timeout = config_int(opts, "wait_timeout", WAIT_TIMEOUT)
    if node_rank == 0:
        # First step: a marker of an earlier launch is removed before anything else (see
        # FailureWatch: a waiting node takes an older marker only if it stays in place).
        clear_failure(megatron_path)
        since = None
    else:
        # The storage time before the (slow) record below: a marker rank 0 writes while this node
        # computes the record still counts.
        since = _storage_now(megatron_path, node_rank)
    try:
        want = convert_record(hf_path, opts, dtype)
    except Exception as e:
        if node_rank == 0:
            write_failure(megatron_path, f"conversion settings: {e}")
        raise

    if tracker.exists() and stale_checkpoint_reason(megatron_path, want) is None:
        log.info(f"Native Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        return
    if node_rank != 0:
        log.info(f"[RANK {node_rank}] Waiting for rank 0 to prepare the native checkpoint...")
        wait_for_checkpoint(megatron_path, want, timeout, node_rank=node_rank, since=since)
        log.success(f"[RANK {node_rank}] Checkpoint ready at {megatron_path}")
        return

    def stop(reason):
        write_failure(megatron_path, reason)
        raise ConversionFailed(reason)

    def state():
        """``(reusable, refusal)`` of ``<out>`` from one read of the tracker and the record:
        ``reusable`` when it holds the wanted checkpoint; else ``refusal`` says why rank 0 cannot
        convert into it (None: it can). Checked before and again after taking the lock, since
        another launch can finish between the two."""
        if tracker.exists():
            stale = stale_checkpoint_reason(megatron_path, want)
            if stale is None:
                return True, None
            return False, (
                f"Native Megatron checkpoint at {megatron_path} cannot be reused: {stale}. Delete it "
                "or set native_ckpt_out to a new directory to convert again."
            )
        return False, output_not_empty_reason(megatron_path)

    reusable, reason = state()
    if reusable:  # written after the check above (another launch)
        log.info(f"Native Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        return
    if reason:
        stop(reason)
    log.info("Converting HF checkpoint to Megatron format (native, bridge-free)...")
    megatron_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        acquire_lock(megatron_path)
    except ConversionFailed as e:
        stop(str(e))
    partial = partial_path(megatron_path)
    try:
        # again under the lock: a launch that held it may have written <out> meanwhile
        reusable, reason = state()
        if reusable:
            log.info(f"Native Megatron checkpoint written meanwhile at {megatron_path}, skipping conversion")
            return
        if reason:
            stop(reason)
        if partial.exists():  # left by a killed conversion; this process owns the lock now
            shutil.rmtree(partial)
        convert(hf_path, str(partial), opts)
        (partial / CONVERT_RECORD).write_text(json.dumps(want, indent=2, sort_keys=True) + "\n")
        if megatron_path.exists():
            megatron_path.rmdir()  # empty (checked under the lock)
        os.replace(partial, megatron_path)
        try:
            Path(f"{megatron_path}.done").touch()  # informational only; no node reads it
        except OSError as e:
            log.info(f"could not write {megatron_path}.done ({e}); the checkpoint is complete")
        log.success(f"Checkpoint prepared at {megatron_path}")
    except ConversionFailed:
        raise
    except BaseException as e:
        write_failure(megatron_path, f"{type(e).__name__}: {e}")
        raise
    finally:
        release_lock(megatron_path)


# --- OLMo-3 given as a Hugging Face hub id ---------------------------------------------------------

_HUB_ID = r"[A-Za-z0-9][\w.-]*/[\w.-]+"
_HUB_PATTERNS = ["*.json", "*.safetensors"]


def is_hub_id(hf_path) -> bool:
    """``org/name`` that is not a local path."""
    text = str(hf_path or "")
    return re.fullmatch(_HUB_ID, text) is not None and not os.path.exists(text)


_SHARD_NAME = re.compile(r".+-\d+-of-\d+\.safetensors")


def snapshot_incomplete_reason(path) -> str | None:
    """Why the snapshot directory cannot be converted (None: it can): it needs ``config.json`` and
    the ``*.safetensors`` weights; with a ``model.safetensors.index.json`` every shard it names.
    Sharded weight names (``*-00001-of-00003.safetensors``) without the index are incomplete."""
    path = Path(path)
    if not (path / "config.json").is_file():
        return f"{path} has no config.json"
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            return f"{index} cannot be read ({type(e).__name__}: {e})"
        missing = sorted(s for s in shards if not (path / s).is_file())
        if missing:
            return f"{path} lacks {len(missing)} weight shard(s) named in its index, e.g. {missing[0]}"
        return None
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        return f"{path} has no *.safetensors weights"
    # Shard names (model-00001-of-00003.safetensors) mean a sharded checkpoint, whose index is
    # needed to know the shard set: without it the download is not finished (the index may land
    # after the first shards while other nodes poll the shared cache).
    if any(_SHARD_NAME.fullmatch(w.name) for w in weights):
        return f"{path} has sharded weights but no model.safetensors.index.json (download not finished?)"
    return None


def _snapshot(hub_id: str, local_only: bool):
    """(path, None) for a complete snapshot of ``hub_id``, else (None, reason)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None, "huggingface_hub is not installed"
    try:
        path = snapshot_download(hub_id, allow_patterns=_HUB_PATTERNS, local_files_only=local_only)
    except Exception as e:  # not cached / no network
        where = "in the Hugging Face cache" if local_only else "on the Hugging Face hub"
        return None, f"{hub_id} not found {where} ({type(e).__name__}: {e})"
    reason = snapshot_incomplete_reason(path)
    return (None, reason) if reason else (path, None)


def hub_model_type_reason(hub_id: str) -> str | None:
    """Why ``hub_id`` is not an OLMo-3 checkpoint (None: its config.json says ``olmo3``). Reads only
    config.json (the cache first, else the hub), so a foreign model's weights are never downloaded."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return "huggingface_hub is not installed"
    config, errors = None, []
    for local_only in (True, False):
        try:
            path = snapshot_download(hub_id, allow_patterns=["config.json"], local_files_only=local_only)
            config = json.loads((Path(path) / "config.json").read_text())
            break
        except Exception as e:  # not cached / no network / not JSON
            errors.append(f"{type(e).__name__}: {e}")
    if not isinstance(config, dict):
        return f"cannot read the config.json of {hub_id} ({'; '.join(errors)})"
    model_type = config.get("model_type")
    if model_type != "olmo3":
        return (
            f"{hub_id} is model_type {model_type!r} (config.json), not olmo3; it cannot load into "
            "OLMo-3. Point hf_path (else tokenizer_model) at an OLMo-3 checkpoint"
        )
    return None


def resolve_hub_snapshot(hub_id: str, out, opts: dict, node_rank: int, log, poll: float = 5.0):
    """A complete local snapshot of the OLMo-3 hub id ``hub_id``, else None (the reason is logged).

    Only rank 0 downloads (the cache first, else the hub). The other nodes do not reach the
    network: they wait (``native_ckpt_wait_timeout``) until the snapshot is complete in the
    Hugging Face cache, which must be shared, as ``<out>`` must be. When rank 0 cannot get the
    snapshot it writes ``<out>.failed`` and the other nodes stop with it (:class:`FailureWatch`).
    A snapshot is complete with config.json and all its safetensors weights
    (:func:`snapshot_incomplete_reason`). Rank 0 first reads only config.json: a model that is not
    OLMo-3 stops there (:func:`hub_model_type_reason`), before any weights are downloaded."""
    if node_rank == 0:
        clear_failure(out)  # first step, as in run_native_flow (see FailureWatch)
        reason = hub_model_type_reason(hub_id)
        if reason is not None:
            write_failure(out, reason)
            log.error(reason)
            return None
        path, reason = _snapshot(hub_id, local_only=True)
        if path is None:
            log.info(f"{reason}; downloading the config and safetensors weights of {hub_id}")
            path, reason = _snapshot(hub_id, local_only=False)
        if path is None:
            reason = f"cannot get a complete snapshot of {hub_id}: {reason}"
            write_failure(out, reason)
            log.error(reason)
        return path
    timeout = config_int(opts, "wait_timeout", WAIT_TIMEOUT)
    watch = FailureWatch(out, _storage_now(out, node_rank))
    start = time.monotonic()
    log.info(f"[RANK {node_rank}] Waiting for rank 0 to put {hub_id} in the Hugging Face cache...")
    while True:
        path, reason = _snapshot(hub_id, local_only=True)
        if path is not None:
            return path
        failure = watch.check()
        if failure is not None:
            log.error(f"[RANK {node_rank}] rank 0 stopped: {failure}")
            return None
        if time.monotonic() - start >= timeout:
            log.error(
                f"[RANK {node_rank}] no complete snapshot of {hub_id} in the Hugging Face cache after "
                f"{timeout} s ({reason}); the nodes need a shared cache (HF_HOME)"
            )
            return None
        time.sleep(poll)
