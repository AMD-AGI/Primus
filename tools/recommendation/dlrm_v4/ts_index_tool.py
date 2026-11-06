#!/usr/bin/env python3
"""Offline build, verification and housekeeping for the PR 907 window index.

The trainer reads ``ts_indices_<tag>_W<sec>.npy`` and ``ts_offsets_<tag>_W<sec>.npy``
(built by ``generative_recommenders/dlrm_v4/datasets/offline_ts_indices.py``,
kept byte-identical to mlcommons/training#907) with only the PR's cheap
O(windows) load check. Everything this tool does happens OUTSIDE training, so
it adds no work between trainer start and end of training:

  prebuild      build the PR-format files with the PR builder (unchanged), then
                write a sidecar manifest (sizes, hashes, source fingerprints)
  verify        O(N) full check of the index against its source; records the
                result in the sidecar; nonzero exit on any failure
  check-staged  cheap check of a staged (node-local) copy against the sidecar:
                sizes, npy headers, offsets content, source fingerprints, and
                optionally the full-file hashes. stdlib only (no numpy), so the
                job driver can run it on the bare host before the trainer starts
  cleanup-temp  remove ``.ts-indices-*`` temp dirs left by killed builds
  status        print what exists and whether it is valid

Exit codes: 0 ok, 1 check failed, 2 usage/environment error, 3 refused
(target exists), 4 busy (a build holds the lock), 128+N killed by signal N.
"""

import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import struct
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

# The tool is typically run as root in the training container; loading the PR
# builder by path would otherwise drop a root-owned datasets/__pycache__/*.pyc
# into the repo. The tool writes only into the cache dir it is pointed at.
sys.dont_write_bytecode = True

TOOL_VERSION = "1"
SIDECAR_SCHEMA = 1
BUILDER_PATH = (
    Path(__file__).resolve().parent.parent
    / "generative_recommenders" / "dlrm_v4" / "datasets" / "offline_ts_indices.py"
)
# tempfile.TemporaryDirectory(prefix=".ts-indices-", dir=cache_dir) in the PR
# builder: the prefix plus tempfile's 8-character random suffix.
BUILDER_TEMP_RE = re.compile(r"^\.ts-indices-[A-Za-z0-9_]{8}$")
TOOL_TEMP_PREFIX = ".ts-index-tool-"
TOOL_TEMP_RE = re.compile(r"^\.ts-index-tool-.*\.tmp$")
FP_HEAD_BYTES = 4096
FP_BLOCK_BYTES = 1 << 16
FP_BLOCKS = 64
READ_BYTES = 8 << 20

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_REFUSED, EXIT_BUSY = 0, 1, 2, 3, 4


class ToolError(Exception):
    def __init__(self, msg, code=EXIT_USAGE):
        super().__init__(msg)
        self.code = code


class Interrupted(BaseException):
    """Raised from a signal handler so context managers (the PR builder's
    TemporaryDirectory included) run their cleanup."""

    def __init__(self, signum):
        super().__init__(f"signal {signum}")
        self.signum = signum


@contextmanager
def signals_raise():
    fired = []

    def handler(signum, _frame):
        if fired:  # already unwinding: let the cleanup finish
            return
        fired.append(signum)
        raise Interrupted(signum)

    sigs = [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]
    old = {s: signal.signal(s, handler) for s in sigs}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


# ----------------------------------------------------------------- names ---

def tag_for(history_length, min_history=None):
    """Same tag as build_offline_ts_indices and yambda's _positions_filename."""
    tag = f"L{history_length}"
    if min_history is not None and min_history != history_length:
        tag += f"_m{min_history}"
    return tag


def names(tag, window_seconds):
    return {
        "indices": f"ts_indices_{tag}_W{window_seconds}.npy",
        "offsets": f"ts_offsets_{tag}_W{window_seconds}.npy",
        "sidecar": f"ts_index_manifest_{tag}_W{window_seconds}.json",
        "positions": f"positions_{tag}.npy",
        "anchor_ts": f"anchor_ts_{tag}.npy",
        "flat_timestamps": "flat_timestamps.npy",
    }


def sidecar_name_ok(rec, expected):
    """A sidecar is data, not a path oracle: a file it describes must carry the
    name this tool derives for that tag/window, so a forged ``../x`` cannot make
    a check read (and pass on) a file outside the cache dir."""
    return rec.get("name", expected) == expected


def hash_algo_of(rec, problems, what):
    algo = rec.get("hash_algo", "sha256")
    if algo not in hashlib.algorithms_guaranteed:
        problems.append(f"{what}: sidecar hash_algo {algo!r} is not a supported hashlib algorithm")
        return None
    return algo


def load_builder():
    spec = importlib.util.spec_from_file_location("_pr907_offline_ts_indices", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path, algo="sha256"):
    h = hashlib.new(algo)
    buf = bytearray(READ_BYTES)
    view = memoryview(buf)
    with open(path, "rb", buffering=0) as f:
        while True:
            n = f.readinto(buf)
            if not n:
                break
            h.update(view[:n])
    return h.hexdigest()


def fingerprint(path):
    """Size, mtime, and a digest of the header plus 64 evenly spaced 64 KiB
    blocks (about 4 MiB read). Catches a regenerated source of the same length
    unless every sampled block happens to match; the mtime catches a rewrite
    in place. verify's full recompute is the authoritative check."""
    st = os.stat(path)
    size = st.st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(FP_HEAD_BYTES))
        if size > FP_BLOCK_BYTES:
            span = size - FP_BLOCK_BYTES
            for i in range(FP_BLOCKS):
                f.seek(span * i // (FP_BLOCKS - 1))
                h.update(f.read(FP_BLOCK_BYTES))
    return {
        "name": os.path.basename(path),
        "size": size,
        "mtime_ns": st.st_mtime_ns,
        "sampled_sha256": h.hexdigest(),
        "sample": f"head{FP_HEAD_BYTES}+{FP_BLOCKS}x{FP_BLOCK_BYTES}",
    }


def source_files(cache_dir, nm):
    """Files the PR builder reads, in the same precedence."""
    files = [nm["positions"]]
    files.append(nm["anchor_ts"] if (cache_dir / nm["anchor_ts"]).exists() else nm["flat_timestamps"])
    return files


# ------------------------------------------------- stdlib npy header read ---

def read_npy_header(path):
    """Parse an .npy header without numpy: (descr, fortran_order, shape, data_offset)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if len(magic) < 8 or magic[:6] != b"\x93NUMPY":
            raise ValueError(f"{path}: not an .npy file")
        major = magic[6]
        if major == 1:
            (hlen,) = struct.unpack("<H", f.read(2))
            start = 10
        elif major in (2, 3):
            (hlen,) = struct.unpack("<I", f.read(4))
            start = 12
        else:
            raise ValueError(f"{path}: unsupported .npy version {major}")
        raw = f.read(hlen)
        if len(raw) != hlen:
            raise ValueError(f"{path}: truncated .npy header")
    text = raw.decode("latin1")
    m_descr = re.search(r"'descr':\s*'([^']+)'", text)
    m_fort = re.search(r"'fortran_order':\s*(True|False)", text)
    m_shape = re.search(r"'shape':\s*\(([^)]*)\)", text)
    if not (m_descr and m_fort and m_shape):
        raise ValueError(f"{path}: unparseable .npy header")
    shape = tuple(int(x) for x in m_shape.group(1).replace(" ", "").split(",") if x)
    return m_descr.group(1), m_fort.group(1) == "True", shape, start + hlen


def check_int64_vector(path, problems, what):
    """Header says 1-D little-endian int64 and the file holds exactly that many bytes."""
    try:
        descr, fortran, shape, off = read_npy_header(path)
    except (OSError, ValueError) as e:
        problems.append(f"{what}: {e}")
        return None
    if descr not in ("<i8",) or fortran or len(shape) != 1:
        problems.append(f"{what}: header descr={descr} fortran={fortran} shape={shape}, want 1-D <i8")
        return None
    size = os.path.getsize(path)
    want = off + 8 * shape[0]
    if size != want:
        problems.append(f"{what}: file is {size} B, header implies {want} B (truncated or padded)")
        return None
    return shape[0], off


def read_int64_vector(path, n, off):
    import array

    a = array.array("q")
    with open(path, "rb") as f:
        f.seek(off)
        a.frombytes(f.read(8 * n))
    if sys.byteorder != "little":
        a.byteswap()
    return a


# ------------------------------------------------------------- sidecar ---

def write_json_atomic(path, obj):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=TOOL_TEMP_PREFIX, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load_sidecar(path):
    with open(path) as f:
        data = json.load(f)
    if data.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"{path}: unknown sidecar schema {data.get('schema')!r}")
    return data


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -------------------------------------------------------------- locking ---

@contextmanager
def try_build_lock(cache_dir):
    """Non-blocking exclusive lock on the builder's _positions_lock. Yields True
    if held (no build is running), False if another process holds it."""
    with open(cache_dir / "_positions_lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                yield False
                return
            raise
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# --------------------------------------------------------- cleanup-temp ---

def find_temps(cache_dir):
    out = []
    try:
        entries = list(os.scandir(cache_dir))
    except FileNotFoundError:
        return out
    for e in entries:
        if e.is_dir(follow_symlinks=False) and BUILDER_TEMP_RE.match(e.name):
            out.append(Path(e.path))
        elif e.is_file(follow_symlinks=False) and TOOL_TEMP_RE.match(e.name):
            out.append(Path(e.path))
    return sorted(out)


def disk_usage(path):
    if path.is_dir():
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.lstat(os.path.join(root, f)).st_blocks * 512
                except FileNotFoundError:
                    pass
        return total
    try:
        return os.lstat(path).st_blocks * 512
    except FileNotFoundError:
        return 0


def cleanup_temp(cache_dir, dry_run=False, log=print):
    """Remove leftover builder temp dirs and tool temp files. Only while holding
    the builder lock, so a build in progress is never touched."""
    temps = find_temps(cache_dir)
    if not temps:
        log(f"cleanup-temp: no leftovers in {cache_dir}")
        return EXIT_OK, 0
    if dry_run:
        for p in temps:
            log(f"cleanup-temp: would remove {p} ({disk_usage(p):,} B on disk)")
        return EXIT_OK, 0
    # Removing an entry needs write+search on its parent, and emptying a temp
    # dir needs the same on the dir itself. Check all of them before deleting
    # anything, so a non-owner gets a clean usage error, not a half-emptied dir.
    blocked = [str(q) for q in [cache_dir] + [p for p in temps if p.is_dir()]
               if not os.access(q, os.W_OK | os.X_OK)]
    if blocked:
        raise ToolError(f"cleanup-temp: uid {os.getuid()} cannot remove entries in {', '.join(blocked)} "
                        f"(run as the cache owner)", EXIT_USAGE)
    freed = 0
    try:
        with try_build_lock(cache_dir) as held:
            if not held:
                log(f"cleanup-temp: a build holds {cache_dir}/_positions_lock; not touching {len(temps)} temp path(s)")
                return EXIT_BUSY, 0
            for p in find_temps(cache_dir):
                used = disk_usage(p)
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink(missing_ok=True)
                freed += used
                log(f"cleanup-temp: removed {p} ({used:,} B on disk)")
    except OSError as e:
        raise ToolError(f"cleanup-temp: {e} (freed {freed:,} B before the error; run as the cache owner)",
                        EXIT_USAGE) from e
    return EXIT_OK, freed


# ---------------------------------------------------------------- verify ---

def _np():
    try:
        import numpy as np
    except ImportError as e:
        raise ToolError("numpy is required for prebuild/verify (run inside the training container)") from e
    return np


def full_check(cache_dir, nm, window_seconds, chunk_size, hash_algo="sha256", log=print):
    """O(N) check that the index is exactly what the PR builder produces for the
    current source. Returns (problems, facts)."""
    np = _np()
    problems, facts = [], {}
    ip, op = cache_dir / nm["indices"], cache_dir / nm["offsets"]
    for p in (ip, op, cache_dir / nm["positions"]):
        if not p.exists():
            problems.append(f"missing {p.name}")
    if problems:
        return problems, facts

    hdr_i = check_int64_vector(ip, problems, "indices")
    hdr_o = check_int64_vector(op, problems, "offsets")
    if problems:
        return problems, facts

    positions = np.load(cache_dir / nm["positions"], mmap_mode="r")
    has_anchor = (cache_dir / nm["anchor_ts"]).exists()
    ts_src = np.load(cache_dir / (nm["anchor_ts"] if has_anchor else nm["flat_timestamps"]), mmap_mode="r")
    for arr, what in ((positions, "positions"), (ts_src, "timestamps")):
        if arr.ndim != 1 or arr.dtype != np.int64:
            problems.append(f"source {what} is not 1-D int64")
    count = len(positions)
    if has_anchor and len(ts_src) != count:
        problems.append(f"anchor_ts has {len(ts_src):,} entries, positions has {count:,}")
    if problems:
        return problems, facts

    def ts_at(sel):  # timestamps of anchors at positions-array indices sel
        return ts_src[sel] if has_anchor else ts_src[positions[sel]]

    def ts_range(a, b):
        return ts_src[a:b] if has_anchor else ts_src[positions[a:b]]

    t0 = time.time()
    t_min, t_max = 0, -1
    for a in range(0, count, chunk_size):
        c = ts_range(a, min(a + chunk_size, count))
        t_min = int(c.min()) if a == 0 else min(t_min, int(c.min()))
        t_max = int(c.max()) if a == 0 else max(t_max, int(c.max()))
    windows = (t_max - t_min) // window_seconds + 1 if count else 0
    hist = np.zeros(windows, dtype=np.int64)
    for a in range(0, count, chunk_size):
        hist += np.bincount((ts_range(a, min(a + chunk_size, count)) - t_min) // window_seconds,
                            minlength=windows)
    facts.update(anchor_count=count, num_windows=windows, t_min=t_min, t_max=t_max,
                 timestamps_source="anchor_ts" if has_anchor else "flat_timestamps")
    log(f"verify: source pass {time.time() - t0:.1f}s, {count:,} anchors, {windows} windows")

    n_idx, _ = hdr_i
    n_off, _ = hdr_o
    if n_idx != count:
        problems.append(f"indices has {n_idx:,} entries, positions has {count:,}")
    if n_off != windows + 1:
        problems.append(f"offsets has {n_off} entries, source implies {windows + 1}")
    if problems:
        return problems, facts
    offsets = np.load(op)
    if offsets[0] != 0 or offsets[-1] != count:
        problems.append(f"offsets[0]={int(offsets[0])} offsets[-1]={int(offsets[-1])}, want 0 and {count}")
    if np.any(offsets[1:] < offsets[:-1]):
        problems.append("offsets not monotone")
    bad_w = np.flatnonzero(np.diff(offsets) != hist)
    if bad_w.size:
        problems.append(f"{bad_w.size} window count(s) differ from the source histogram, first w={int(bad_w[0])}")
    if problems:
        return problems, facts

    indices = np.load(ip, mmap_mode="r")
    h = hashlib.new(hash_algo)
    with open(ip, "rb") as f:  # hash = header bytes + data bytes = whole file
        h.update(f.read(hdr_i[1]))
    n_range = n_window = n_order = 0
    first = {}
    prev_idx, prev_w = None, None
    t0 = time.time()
    for a in range(0, count, chunk_size):
        b = min(a + chunk_size, count)
        idx = np.asarray(indices[a:b])
        h.update(memoryview(np.ascontiguousarray(idx)).cast("B"))
        # expected window of each slot from the offsets
        w_lo = int(np.searchsorted(offsets, a, side="right")) - 1
        w_hi = int(np.searchsorted(offsets, b - 1, side="right")) - 1
        ws = np.arange(w_lo, w_hi + 1)
        lens = np.minimum(offsets[ws + 1], b) - np.maximum(offsets[ws], a)
        wexp = np.repeat(ws, lens)
        ok = (idx >= 0) & (idx < count)
        nr = int(ok.size - np.count_nonzero(ok))
        if nr:
            n_range += nr
            first.setdefault("range", a + int(np.flatnonzero(~ok)[0]))
        safe = np.where(ok, idx, 0)
        wact = (ts_at(safe) - t_min) // window_seconds
        badw = ok & (wact != wexp)
        nw = int(np.count_nonzero(badw))
        if nw:
            n_window += nw
            first.setdefault("window", a + int(np.flatnonzero(badw)[0]))
        same = wexp[1:] == wexp[:-1]
        bad_order = same & (idx[1:] <= idx[:-1])
        no = int(np.count_nonzero(bad_order))
        if prev_idx is not None and prev_w == int(wexp[0]) and int(idx[0]) <= prev_idx:
            no += 1
            first.setdefault("order", a)
        if no:
            n_order += no
            if bad_order.any():
                first.setdefault("order", a + 1 + int(np.flatnonzero(bad_order)[0]))
        prev_idx, prev_w = int(idx[-1]), int(wexp[-1])
    log(f"verify: index pass {time.time() - t0:.1f}s")
    if n_range:
        problems.append(f"{n_range:,} entries out of [0, {count}), first at slot {first['range']}")
    if n_window:
        problems.append(f"{n_window:,} entries in the wrong window, first at slot {first['window']}")
    if n_order:
        problems.append(f"{n_order:,} order violations within a window, first at slot {first['order']}")
    facts["indices_hash"] = h.hexdigest()
    facts["offsets_hash"] = sha256_file(op, hash_algo)
    facts["indices_size"] = os.path.getsize(ip)
    facts["offsets_size"] = os.path.getsize(op)
    return problems, facts


def compare_sidecar(side, facts, problems, warnings):
    files = side.get("files", {})
    for key in ("indices", "offsets"):
        rec = files.get(key, {})
        if rec.get("size") != facts.get(f"{key}_size"):
            problems.append(f"{key}: size {facts.get(f'{key}_size')} != sidecar {rec.get('size')}")
        if rec.get("hash") != facts.get(f"{key}_hash"):
            problems.append(f"{key}: {rec.get('hash_algo')} differs from sidecar")
    for key in ("anchor_count", "num_windows"):
        if side.get(key) != facts.get(key):
            problems.append(f"{key}: {facts.get(key)} != sidecar {side.get(key)}")


def compare_sources(cache_dir, side, nm, problems, warnings, check_mtime):
    recs = side.get("source", {})
    if not recs:
        problems.append("sidecar has no source fingerprints")
    allowed = {nm["positions"], nm["anchor_ts"], nm["flat_timestamps"]}
    for name, rec in sorted(recs.items()):
        if name not in allowed:
            problems.append(f"sidecar source entry {name!r} is not one of {sorted(allowed)}")
            continue
        p = cache_dir / name
        if not p.exists():
            problems.append(f"source {name} missing")
            continue
        fp = fingerprint(p)
        if fp["size"] != rec.get("size"):
            problems.append(f"source {name}: size {fp['size']} != sidecar {rec.get('size')}")
        elif fp["sampled_sha256"] != rec.get("sampled_sha256"):
            problems.append(f"source {name}: sampled content differs from sidecar (stale index?)")
        elif fp["mtime_ns"] != rec.get("mtime_ns"):
            msg = f"source {name}: mtime differs from sidecar (rewritten or copied)"
            (problems if check_mtime else warnings).append(msg)


# --------------------------------------------------------------- commands ---

def make_sidecar(cache_dir, nm, args, hash_algo, facts, sources):
    builder_sha = sha256_file(BUILDER_PATH)
    return {
        "schema": SIDECAR_SCHEMA,
        "tool": "ts_index_tool.py",
        "tool_version": TOOL_VERSION,
        "builder": {"path": "generative_recommenders/dlrm_v4/datasets/offline_ts_indices.py",
                    "sha256": builder_sha, "upstream": "mlcommons/training#907 09612efd"},
        "created_utc": now_utc(),
        "cache_dir": str(cache_dir),
        "tag": tag_for(args.history_length, args.min_history),
        "history_length": args.history_length,
        "min_history": args.min_history,
        "window_seconds": args.window_seconds,
        "dtype": "<i8",
        "anchor_count": facts["anchor_count"],
        "num_windows": facts["num_windows"],
        "files": {
            "indices": {"name": nm["indices"], "size": facts["indices_size"],
                        "hash_algo": hash_algo, "hash": facts["indices_hash"]},
            "offsets": {"name": nm["offsets"], "size": facts["offsets_size"],
                        "hash_algo": hash_algo, "hash": facts["offsets_hash"]},
        },
        "source": sources,
        "verify": {"status": "not_run"},
    }


def common_paths(args):
    cache_dir = Path(args.cache_dir).resolve()
    if not cache_dir.is_dir():
        raise ToolError(f"{cache_dir}: not a directory")
    nm = names(tag_for(args.history_length, args.min_history), args.window_seconds)
    return cache_dir, nm


def cmd_prebuild(args, log=print):
    np = _np()
    cache_dir, nm = common_paths(args)
    ip, op, sp = cache_dir / nm["indices"], cache_dir / nm["offsets"], cache_dir / nm["sidecar"]
    if not (cache_dir / nm["positions"]).exists():
        raise ToolError(f"{cache_dir / nm['positions']} missing; build the positions cache first")
    if not os.access(cache_dir, os.W_OK):
        raise ToolError(f"{cache_dir} is not writable by uid {os.getuid()} (run as the cache owner)")
    exists = ip.exists() and op.exists()
    if exists and not args.force:
        raise ToolError(f"{ip.name} and {op.name} already exist; use verify (or verify --adopt), "
                        f"or prebuild --force to rebuild", EXIT_REFUSED)
    if sp.exists() and not args.force:
        raise ToolError(f"{sp.name} exists without a complete index; inspect, or use --force", EXIT_REFUSED)

    code, _ = cleanup_temp(cache_dir, log=log)
    if code == EXIT_BUSY:
        raise ToolError("another build holds the lock; retry when it finishes", EXIT_BUSY)

    positions = np.load(cache_dir / nm["positions"], mmap_mode="r")
    need = 8 * len(positions) + (64 << 20)
    free = shutil.disk_usage(cache_dir).free
    if free < need:
        raise ToolError(f"{cache_dir}: {free:,} B free, build needs {need:,} B "
                        f"(a full disk under the builder's mmap is a SIGBUS, not an exception)")
    del positions

    srcs = source_files(cache_dir, nm)
    before = {n: fingerprint(cache_dir / n) for n in srcs}
    builder = load_builder()
    pre_inode = ip.stat().st_ino if ip.exists() else None
    t0 = time.time()
    try:
        with signals_raise():
            if args.force:
                # Offsets are published last, so removing them marks the index
                # incomplete and makes the PR builder rebuild both files.
                sp.unlink(missing_ok=True)
                op.unlink(missing_ok=True)
            log(f"prebuild: building {ip.name} in {cache_dir} ({len(srcs)} source files)")
            builder.build_offline_ts_indices(
                cache_dir, args.history_length, args.min_history,
                window_seconds=args.window_seconds, chunk_size=args.chunk_size,
            )
    except BaseException:
        # The builder's TemporaryDirectory removed its own temp dir. If we died
        # between its two os.replace calls, drop the orphan indices file it
        # published so the cache does not hold a half-published index.
        if ip.exists() and not op.exists():
            try:
                if ip.stat().st_ino != pre_inode:
                    ip.unlink()
            except FileNotFoundError:
                pass
        find_and_report = find_temps(cache_dir)
        if find_and_report:
            log(f"prebuild: leftover temp paths after failure: {[str(p) for p in find_and_report]}")
        raise
    log(f"prebuild: build {time.time() - t0:.1f}s")
    after = {n: fingerprint(cache_dir / n) for n in srcs}
    if before != after:
        raise ToolError("source files changed during the build; index not trusted", EXIT_FAIL)

    with signals_raise():
        facts = {"indices_size": ip.stat().st_size, "offsets_size": op.stat().st_size,
                 "indices_hash": sha256_file(ip, args.hash), "offsets_hash": sha256_file(op, args.hash)}
        offsets = np.load(op)
        facts["num_windows"] = len(offsets) - 1
        facts["anchor_count"] = int(offsets[-1])
        side = make_sidecar(cache_dir, nm, args, args.hash, facts, after)
        write_json_atomic(sp, side)
    log(f"prebuild: wrote {sp}")
    if args.verify:
        return cmd_verify(args, log=log)
    return EXIT_OK


def cmd_verify(args, log=print):
    cache_dir, nm = common_paths(args)
    sp = cache_dir / nm["sidecar"]
    side = None
    problems, warnings = [], []
    if sp.exists():
        try:
            side = load_sidecar(sp)
        except (OSError, ValueError) as e:
            problems.append(f"sidecar unreadable: {e}")
    elif not args.adopt:
        problems.append(f"sidecar {sp.name} missing (use --adopt to record one after a passing check)")
    hash_algo = (side or {}).get("files", {}).get("indices", {}).get("hash_algo", args.hash)
    if hash_algo not in hashlib.algorithms_guaranteed:
        problems.append(f"sidecar hash_algo {hash_algo!r} is not a supported hashlib algorithm")
        hash_algo = args.hash
    if side is not None:
        for key in ("indices", "offsets"):
            if not sidecar_name_ok(side.get("files", {}).get(key, {}), nm[key]):
                problems.append(f"sidecar files.{key}.name is not {nm[key]!r}")
    t0 = time.time()
    p2, facts = full_check(cache_dir, nm, args.window_seconds, args.chunk_size, hash_algo, log=log)
    problems += p2
    if side is not None and not p2:
        compare_sidecar(side, facts, problems, warnings)
        if side.get("window_seconds") != args.window_seconds:
            problems.append(f"window_seconds {args.window_seconds} != sidecar {side.get('window_seconds')}")
        if args.rebind_source:
            warnings.append("source fingerprints re-recorded after a passing full check (--rebind-source)")
        else:
            compare_sources(cache_dir, side, nm, problems, warnings, check_mtime=False)
        if side.get("builder", {}).get("sha256") != sha256_file(BUILDER_PATH):
            warnings.append("builder source differs from the one recorded at build time "
                            "(content was still checked in full)")
    duration = round(time.time() - t0, 1)
    status = "failed" if problems else "passed"
    for w in warnings:
        log(f"verify: WARNING {w}")
    for p in problems:
        log(f"verify: FAIL {p}")
    log(f"verify: {status.upper()} in {duration}s")

    if not args.no_write:
        if side is None and args.adopt and not problems:
            srcs = source_files(cache_dir, nm)
            side = make_sidecar(cache_dir, nm, args, hash_algo, facts,
                                {n: fingerprint(cache_dir / n) for n in srcs})
            side["adopted"] = True
        if side is not None:
            if args.rebind_source and not problems:
                side["source"] = {n: fingerprint(cache_dir / n) for n in source_files(cache_dir, nm)}
            side["verify"] = {"status": status, "time_utc": now_utc(), "duration_s": duration,
                              "tool_version": TOOL_VERSION, "problems": problems[:20],
                              "warnings": warnings[:20],
                              "t_min": facts.get("t_min"), "t_max": facts.get("t_max"),
                              "timestamps_source": facts.get("timestamps_source")}
            try:
                write_json_atomic(sp, side)
            except OSError as e:
                log(f"verify: could not record result in {sp}: {e}")
                return EXIT_FAIL
    return EXIT_FAIL if problems else EXIT_OK


def cmd_check_staged(args, log=print):
    """stdlib only. Run in the job driver before the trainer starts."""
    dest = Path(args.dest).resolve()
    nm = names(tag_for(args.history_length, args.min_history), args.window_seconds)
    if args.sidecar and args.sidecar_dir:
        raise ToolError("check-staged: give --sidecar or --sidecar-dir, not both")
    if args.sidecar:
        sp = Path(args.sidecar)
    elif args.sidecar_dir:  # the authoritative copy, e.g. next to the NFS source
        sp = Path(args.sidecar_dir) / nm["sidecar"]
    else:
        sp = dest / nm["sidecar"]
    problems, warnings = [], []
    staged_sp = dest / nm["sidecar"]
    if sp.resolve() != staged_sp.resolve() and staged_sp.exists() and sp.exists():
        try:
            if staged_sp.read_bytes() != sp.read_bytes():
                warnings.append(f"staged sidecar {staged_sp.name} differs from {sp} (stale copy; checked against {sp})")
        except OSError:
            pass
    try:
        side = load_sidecar(sp)
    except FileNotFoundError:
        log(f"check-staged: FAIL sidecar {sp} missing")
        return EXIT_FAIL
    except (OSError, ValueError) as e:
        log(f"check-staged: FAIL sidecar {sp}: {e}")
        return EXIT_FAIL
    if side.get("window_seconds") != args.window_seconds or side.get("tag") != tag_for(args.history_length, args.min_history):
        problems.append("sidecar is for a different tag/window")
    vstat = side.get("verify", {}).get("status")
    if vstat != "passed":
        (warnings if args.allow_unverified else problems).append(f"sidecar verify status is {vstat!r}, not 'passed'")
    files = side.get("files", {})
    count = side.get("anchor_count")
    nwin = side.get("num_windows")
    for key in ("indices", "offsets"):
        rec = files.get(key, {})
        if not sidecar_name_ok(rec, nm[key]):
            problems.append(f"sidecar files.{key}.name {rec.get('name')!r} is not {nm[key]!r}")
            continue
        algo = hash_algo_of(rec, problems, key)
        if algo is None:
            continue
        p = dest / nm[key]
        if not p.exists():
            problems.append(f"{p.name} missing")
            continue
        size = p.stat().st_size
        if size != rec.get("size"):
            problems.append(f"{p.name}: {size} B, sidecar {rec.get('size')} B")
            continue
        hdr = check_int64_vector(p, problems, key)
        if hdr is None:
            continue
        n, off = hdr
        want_n = count if key == "indices" else (nwin + 1 if nwin is not None else None)
        if n != want_n:
            problems.append(f"{p.name}: {n} entries, sidecar implies {want_n}")
            continue
        if key == "offsets":
            offs = read_int64_vector(p, n, off)
            if offs[0] != 0 or offs[-1] != count or any(offs[i + 1] < offs[i] for i in range(n - 1)):
                problems.append(f"{p.name}: offsets not monotone from 0 to {count}")
            if sha256_file(p, algo) != rec.get("hash"):
                problems.append(f"{p.name}: hash differs from sidecar")
        elif args.hash:
            if sha256_file(p, algo) != rec.get("hash"):
                problems.append(f"{p.name}: hash differs from sidecar")
    if not args.no_source:
        compare_sources(dest, side, nm, problems, warnings, check_mtime=False)
    for w in warnings:
        log(f"check-staged: WARNING {w}")
    for p in problems:
        log(f"check-staged: FAIL {p}")
    log(f"check-staged: {'FAILED' if problems else 'OK'} {dest} "
        f"({'full hash' if args.hash else 'sizes+headers+offsets'}{'' if args.no_source else '+source fingerprints'})")
    return EXIT_FAIL if problems else EXIT_OK


def cmd_cleanup_temp(args, log=print):
    cache_dir = Path(args.cache_dir).resolve()
    if not cache_dir.is_dir():
        raise ToolError(f"{cache_dir}: not a directory")
    code, freed = cleanup_temp(cache_dir, dry_run=args.dry_run, log=log)
    if freed:
        log(f"cleanup-temp: freed {freed:,} B")
    return code


def cmd_status(args, log=print):
    cache_dir = Path(args.cache_dir).resolve()
    if not cache_dir.is_dir():
        raise ToolError(f"{cache_dir}: not a directory")
    rc = EXIT_OK
    pat = re.compile(r"^ts_(indices|offsets)_(.+)_W(\d+)\.npy$")
    keys = set()
    for e in os.scandir(cache_dir):
        m = pat.match(e.name)
        if m:
            keys.add((m.group(2), int(m.group(3))))
        m = re.match(r"^ts_index_manifest_(.+)_W(\d+)\.json$", e.name)
        if m:
            keys.add((m.group(1), int(m.group(2))))
    if not keys:
        log(f"status: no PR 907 index files in {cache_dir}")
    for tag, w in sorted(keys):
        nm = names(tag, w)
        ip, op, sp = cache_dir / nm["indices"], cache_dir / nm["offsets"], cache_dir / nm["sidecar"]
        state = "complete" if ip.exists() and op.exists() else (
            "INCOMPLETE (offsets missing; next init rebuilds)" if ip.exists() else
            "INCOMPLETE (indices missing; next init rebuilds)" if op.exists() else "absent")
        log(f"[{tag} W{w}] {state}")
        for p in (ip, op):
            if p.exists():
                log(f"  {p.name}: {p.stat().st_size:,} B")
        if sp.exists():
            try:
                side = load_sidecar(sp)
                v = side.get("verify", {})
                log(f"  sidecar: built {side.get('created_utc')}, verify={v.get('status')} {v.get('time_utc', '')}")
                probs = []
                for key in ("indices", "offsets"):
                    rec = side["files"][key]
                    if not sidecar_name_ok(rec, nm[key]):
                        probs.append(f"sidecar files.{key}.name {rec.get('name')!r} is not {nm[key]!r}")
                        continue
                    p = cache_dir / nm[key]
                    if not p.exists() or p.stat().st_size != rec["size"]:
                        probs.append(f"{key} size differs from sidecar")
                warns = []
                compare_sources(cache_dir, side, nm, probs, warns, check_mtime=False)
                for p in probs:
                    log(f"  STALE/INVALID: {p}")
                for p in warns:
                    log(f"  note: {p}")
                if probs or v.get("status") != "passed":
                    rc = EXIT_FAIL
            except (OSError, ValueError, KeyError) as e:
                log(f"  sidecar unreadable: {e}")
                rc = EXIT_FAIL
        else:
            log("  sidecar: none (run verify --adopt)")
            rc = EXIT_FAIL
    for p in find_temps(cache_dir):
        log(f"leftover temp: {p} ({disk_usage(p):,} B on disk) -> cleanup-temp")
        rc = EXIT_FAIL
    old = sorted(e.name for e in os.scandir(cache_dir) if e.name.startswith("window_index_"))
    if old:
        log(f"note: e93c4a4-era files unused by this backend: {', '.join(old)}")
    return rc


# ------------------------------------------------------------------- CLI ---

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def index_args(p, cache_flag=True):
        if cache_flag:
            p.add_argument("--cache-dir", required=True)
        p.add_argument("--history-length", type=int, required=True)
        p.add_argument("--min-history", type=int)
        p.add_argument("--window-seconds", type=int, default=86400)

    p = sub.add_parser("prebuild", help="build the index with the PR builder and write the sidecar")
    index_args(p)
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--hash", default="sha256", choices=sorted(hashlib.algorithms_guaranteed))
    p.add_argument("--force", action="store_true", help="rebuild even if the files exist")
    p.add_argument("--verify", action="store_true", help="run verify right after the build")
    p.add_argument("--adopt", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-write", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--rebind-source", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_prebuild)

    p = sub.add_parser("verify", help="O(N) full check; records the result in the sidecar")
    index_args(p)
    p.add_argument("--chunk-size", type=int, default=8_000_000)
    p.add_argument("--hash", default="sha256", choices=sorted(hashlib.algorithms_guaranteed),
                   help="hash for --adopt (an existing sidecar's algorithm wins)")
    p.add_argument("--adopt", action="store_true",
                   help="no sidecar: write one for the existing files if the full check passes")
    p.add_argument("--rebind-source", action="store_true",
                   help="after a passing full check, re-record source fingerprints")
    p.add_argument("--no-write", action="store_true", help="do not record the result")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("check-staged", help="cheap stdlib check of a staged copy against the sidecar")
    p.add_argument("dest", help="staged cache dir, e.g. /tmp/$USER/dlrm_data/processed_5b/hstu_cache_L4086")
    index_args(p, cache_flag=False)
    p.add_argument("--sidecar", help="sidecar to compare against (default: the one in dest)")
    p.add_argument("--sidecar-dir",
                   help="read the sidecar named for this tag/window from this dir instead of dest "
                        "(the stager passes the NFS source cache dir)")
    p.add_argument("--hash", action="store_true", help="also hash the full indices file (~18 GB read)")
    p.add_argument("--no-source", action="store_true", help="skip source fingerprints")
    p.add_argument("--allow-unverified", action="store_true", help="accept a sidecar whose verify did not pass")
    p.set_defaults(func=cmd_check_staged)

    p = sub.add_parser("cleanup-temp", help="remove .ts-indices-* temp dirs left by killed builds")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_cleanup_temp)

    p = sub.add_parser("status", help="print what exists and whether it is valid")
    p.add_argument("--cache-dir", required=True)
    p.set_defaults(func=cmd_status)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ToolError as e:
        print(f"ts_index_tool: {e}", file=sys.stderr)
        return e.code
    except Interrupted as e:
        print(f"ts_index_tool: interrupted by signal {e.signum}; temp files cleaned", file=sys.stderr)
        return 128 + e.signum


if __name__ == "__main__":
    sys.exit(main())
