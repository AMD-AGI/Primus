"""Unprivileged node page-cache cleanup ("balloon") before pinning host memory.

Run by the pinned-pool pre-grow (utils._pin_free_check, yambda_5b.gin item (121)) in a subprocess when a NUMA node
lacks free memory:  python -m generative_recommenders.dlrm_v4.train.host_mem_reclaim --target-free-gib N

Why: pinned memory cannot be evicted, so pinning on top of a nearly full page cache makes the kernel reclaim during
training, which stalls the ranks. `echo 3 > /proc/sys/vm/drop_caches` needs root, and container platforms often
forbid privileged containers, so instead: per NUMA node, allocate and touch anonymous memory bound to that node (the
kernel reclaims clean page cache to satisfy it), then free it all. Untimed, before RUN_START; no data or RNG touched.

Only as much as needed: a node whose MemFree already meets --target-free-gib is left alone, and a balloon stops once
it reaches that size (the kernel hands out free pages before it reclaims cache, so a balloon of size T leaves about T
free after release). Safety: grows in --chunk-gib steps and stops on a node once its free memory would fall below
--margin-gib with no evictable cache left, so it never needs to swap or OOM; shmem (tmpfs) is never counted as
evictable. If the job's cpuset forbids binding to a node, one unbound pass over the whole machine is used instead.

usage: python -m generative_recommenders.dlrm_v4.train.host_mem_reclaim [--target-free-gib 640] [--margin-gib 32]
       [--chunk-gib 8] [--target-file-gib 24] [--threads 32]
"""
import argparse
import ctypes
import mmap
import os
import sys
import threading
import time

MPOL_BIND = 2
libc = ctypes.CDLL(None, use_errno=True)
SYS_mbind = 237  # x86_64


def node_meminfo(n: int) -> dict:
    out = {}
    with open(f"/sys/devices/system/node/node{n}/meminfo") as f:
        for line in f:
            parts = line.split()
            # "Node 0 MemFree:  123 kB"
            out[parts[2].rstrip(":")] = int(parts[3]) * 1024
    return out


def evictable_file(m: dict) -> int:
    # clean page cache the kernel can drop: file pages minus tmpfs/shmem and dirty/writeback
    return max(0, m.get("FilePages", 0) - m.get("Shmem", 0) - m.get("Dirty", 0) - m.get("Writeback", 0))


def nodes() -> list:
    return sorted(int(d[4:]) for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit())


def touch(buf_addr: int, nbytes: int, threads: int) -> None:
    step = (nbytes + threads - 1) // threads
    ths = []
    for t in range(threads):
        off = t * step
        ln = min(step, nbytes - off)
        if ln <= 0:
            break
        th = threading.Thread(target=libc.memset, args=(ctypes.c_void_p(buf_addr + off), 1, ctypes.c_size_t(ln)))
        th.start()
        ths.append(th)
    for th in ths:
        th.join()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-free-gib", type=float, default=640.0)
    ap.add_argument("--margin-gib", type=float, default=32.0)
    ap.add_argument("--chunk-gib", type=float, default=8.0)
    ap.add_argument("--target-file-gib", type=float, default=24.0)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    G = 1 << 30
    margin, chunk, target = int(a.margin_gib * G), int(a.chunk_gib * G), int(a.target_file_gib * G)
    want = int(a.target_free_gib * G)
    t0 = time.time()
    held = []
    report = []
    unbound = []
    for n in nodes():
        m0 = node_meminfo(n)
        got = 0
        while True:
            m = node_meminfo(n)
            ev = evictable_file(m)
            free = m["MemFree"]
            if ev <= target:
                why = "evictable cache gone"
                break
            if m0["MemFree"] >= want and got == 0:
                why = f"already {m0['MemFree'] / G:.0f} GiB free"
                break
            if got >= want:
                why = f"balloon reached {want / G:.0f} GiB"
                break
            if free + ev - chunk < margin:
                why = "margin"
                break
            mm = mmap.mmap(-1, chunk, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            mask = ctypes.c_ulong(1 << n)
            rc = libc.syscall(SYS_mbind, ctypes.c_void_p(addr), ctypes.c_ulong(chunk), MPOL_BIND,
                              ctypes.byref(mask), ctypes.c_ulong(64), 0)
            if rc != 0:  # the job's cpuset.mems may exclude this node: fall back to one unbound global pass
                why = f"mbind errno {ctypes.get_errno()} -> unbound"
                mm.close()
                unbound.append(n)
                break
            touch(addr, chunk, a.threads)
            held.append(mm)
            got += chunk
        m1 = node_meminfo(n)
        report.append(f"node{n}: ballooned {got / G:.0f} GiB ({why}); free {m0['MemFree'] / G:.0f}->{m1['MemFree'] / G:.0f} GiB, "
                      f"evictable cache {evictable_file(m0) / G:.0f}->{evictable_file(m1) / G:.0f} GiB")
    if unbound:  # global pass: allocate unbound until the machine's evictable cache is gone or free hits the margin
        got = 0
        while True:
            ms = [node_meminfo(n) for n in nodes()]
            ev = sum(evictable_file(m) for m in ms)
            free = sum(m["MemFree"] for m in ms)
            if ev <= target * len(ms) or free + ev - chunk < margin * len(ms) or free < margin * len(ms):
                break
            mm = mmap.mmap(-1, chunk, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            touch(ctypes.addressof(ctypes.c_char.from_buffer(mm)), chunk, a.threads)
            held.append(mm)
            got += chunk
        report.append(f"global unbound pass ballooned {got / G:.0f} GiB")
    for mm in held:
        mm.close()
    time.sleep(1)
    after = ", ".join(f"node{n} free={node_meminfo(n)['MemFree'] / G:.0f}G file={node_meminfo(n)['FilePages'] / G:.0f}G" for n in nodes())
    print(f"[host-mem-reclaim] {'; '.join(report)}; after release: {after}; {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
