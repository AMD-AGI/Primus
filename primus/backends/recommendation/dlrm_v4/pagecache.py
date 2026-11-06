"""Page-cache control for the staged yambda-5b cache (C1, MLPerf-legal cold start).

  flush DIR   sync + global drop_caches (privileged container) + per-file
              eviction, then VERIFY with mincore; exit 3 if >1% still resident.
              This is the MLPerf "flush the cache" step. Use before INIT.
  resident DIR  report how much of DIR is in the page cache (mincore, no faulting)
  drop  DIR   evict every file under DIR from the page cache
              (posix_fadvise DONTNEED; only OUR files, no root, no global
              drop_caches). Run BEFORE the trainer starts so the run really
              starts cold, as its logged cache_clear=True says.
  warm  DIR   stream the files sequentially into the page cache with a thread
              pool (hot files first). Run only AFTER the MLPerf run_start event:
              the reads then happen inside the timed region, so this is in-run
              I/O scheduling, not pre-loading data before the clock.

Why: the trainer gathers each sample's 20k-event history from mmap'd arrays.
Cold, those are random page faults that cap input at ~26-35k samples/s on this
cluster's dm/NVMe volume (~40k needed at 1n on MI355X), so ranks straggle in
the input all-to-all. The same files read sequentially stream at ~10 GB/s, so
the whole working set is resident ~25 s after run_start.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CHUNK = 64 << 20
# Read order: what the per-sample path touches, hottest first; the rest after.
PRIORITY = [
    "positions_L4086.npy", "ts_offsets_L4086_W86400.npy", "ts_indices_L4086_W86400.npy",
    "flat_uid.npy", "user_start.npy", "flat_item_ids.npy", "flat_timestamps.npy",
    "flat_is_listen_plus.npy", "flat_is_like.npy", "flat_is_skip.npy",
    "flat_played_ratio.npy", "flat_event_types.npy",
]


def files_in(d):
    fs = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    rank = {n: i for i, n in enumerate(PRIORITY)}
    return [os.path.join(d, f) for f in sorted(fs, key=lambda f: (rank.get(f, len(PRIORITY)), f))]


def drop(d):
    n = 0
    for p in files_in(d):
        fd = os.open(p, os.O_RDONLY)
        try:
            # Write back THIS file's dirty pages first (a freshly staged copy
            # may not be on disk yet): DONTNEED only evicts clean pages. Local
            # and per-file, unlike a global sync.
            os.fdatasync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        n += 1
    print(f"[pagecache] dropped {n} files under {d}", flush=True)


def resident(d, quiet=False):
    """Bytes of the files under d currently in the page cache, via mincore(2)
    on a PROT_READ/MAP_SHARED mapping: inspects residency without faulting any
    page in, so measuring does not warm the cache."""
    import ctypes
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    pg = os.sysconf("SC_PAGE_SIZE")
    tot = res = 0
    for p in files_in(d):
        size = os.path.getsize(p)
        if size == 0:
            continue
        fd = os.open(p, os.O_RDONLY)
        try:
            addr = libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), f"mmap failed for {p}")
            npages = (size + pg - 1) // pg
            vec = (ctypes.c_ubyte * npages)()
            if libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore failed for {p}")
            r = npages - bytes(vec).count(0)
            libc.munmap(addr, size)
        finally:
            os.close(fd)
        tot += size
        res += min(size, r * pg)
    if not quiet:
        print(f"[pagecache] resident {res / 1e9:.2f} of {tot / 1e9:.1f} GB "
              f"({100 * res / max(1, tot):.2f}%) under {d}", flush=True)
    return res, tot


def flush(d, max_resident_pct=1.0):
    """MLPerf 'flush the cache' before the run, as NVIDIA/AMD/Google submissions
    do: global drop_caches (needs a privileged container), plus a per-file
    DONTNEED on the dataset, then VERIFY with mincore and fail if more than
    max_resident_pct of the dataset is still cached. No global sync: the dataset
    is read-only, so its pages are clean and evict without one, while a global
    sync waits on every filesystem's dirty data (10+ min on contended nodes)."""
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        print("[pagecache] global drop_caches=3 done", flush=True)
    except OSError as e:
        print(f"[pagecache] WARNING global drop_caches failed ({e}); per-file only", flush=True)
    drop(d)
    res, tot = resident(d)
    if 100 * res / max(1, tot) > max_resident_pct:
        print(f"[pagecache] FAIL: {100 * res / tot:.2f}% of the dataset still resident "
              f"(> {max_resident_pct}%) -- not a cold start", flush=True)
        sys.exit(3)


def warm(d, threads=16):
    resident(d)  # residency at run_start, before warming: proves the cold start
    t0 = time.time()
    tasks = []
    for p in files_in(d):
        size = os.path.getsize(p)
        tasks += [(p, off, min(CHUNK, size - off)) for off in range(0, size, CHUNK)]
    local = __import__("threading").local()

    def read(task):
        p, off, ln = task
        if not hasattr(local, "buf"):
            local.buf = bytearray(CHUNK)
            local.fds = {}
        fd = local.fds.get(p)
        if fd is None:
            fd = local.fds[p] = os.open(p, os.O_RDONLY)
        return os.preadv(fd, [memoryview(local.buf)[:ln]], off)

    # tasks are consumed in submission order, so priority files land first
    with ThreadPoolExecutor(threads) as ex:
        total = sum(ex.map(read, tasks))
    print(f"[pagecache] warmed {total / 1e9:.1f} GB under {d} in {time.time() - t0:.1f}s "
          f"({total / 1e9 / max(1e-9, time.time() - t0):.1f} GB/s)", flush=True)


if __name__ == "__main__":
    mode, d = sys.argv[1], sys.argv[2]
    {"drop": drop, "warm": warm, "resident": resident, "flush": flush}[mode](d)
