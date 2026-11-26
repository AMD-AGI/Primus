# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""Pre-RUN_START MADV_NOHUGEPAGE sweep of this rank's 4 KiB-mapped, huge-page-eligible private anonymous memory.

Code default OFF; the gin binding apply_env_bootstrap.DLRM_HSTU_FP8_THP_NOHUGE_SWEEP = True turns it ON.
The env kill switch is DLRM_HSTU_FP8_THP_NOHUGE_SWEEP=0, and a pre-set env var wins over gin. train_ranker
imports this module only when the knob is on.

WHY: a periodic single-GPU stall every 10.24 s. Under THP enabled=madvise, khugepaged wakes every
10 s and collapses up to 8 PMD ranges (16 MiB, pages_to_scan 4096) of 4 KiB-mapped memory in a
MADV_HUGEPAGE VMA of one process. When that memory is registered with the GPU driver as a KFD
USERPTR buffer, the collapse invalidates it, and the driver evicts every GPU queue of the process
until the restore worker re-validates. The rank's GPU stalls for 10-165 ms, and every rank then
waits for it at the next collective.

The 4 KiB-mapped memory in those VMAs is not fault-time fallback (thp_fault_fallback 0). These
VMAs were PMD-mapped THP until the DataLoader pre-start fork(), which shares them copy-on-write with
the persistent workers. The fork write-protects them, which invalidates the USERPTR registrations
and evicts the rank's queues once. The driver's restore then re-faults every registered page for
write. That breaks copy-on-write, splits the PMDs (thp_split_pmd) and copies the pages into 4 KiB
pages, all before RUN_START. From then on they are khugepaged's work, inside the timed region. The
pinned pool's power-of-two blocks allocated after the fork stay 100% PMD-mapped and are not
involved.

WHAT: once, after every pre-start fork and warmup and right before RUN_START:
1. Read /proc/self/smaps.
2. Select each private anonymous VMA (inode 0; path "", "[heap]" or "[anon:...]") that meets all of:
   - khugepaged may scan it under the PMD-size THP policy (see THP POLICY);
   - it contains at least one 2 MiB-aligned range;
   - it holds 4 KiB-mapped pages (Rss > AnonHugePages).
   Every other VMA keeps its policy. That includes VMAs whose populated pages are all PMD-mapped, so
   new pinned blocks and other huge-page-backed memory keep their huge-page fault path.
3. Re-read /proc/self/maps and keep only selected ranges that are still exactly one VMA. This
   shrinks the window for concurrent mapping changes from the ~0.5 s smaps read to the madvise
   loop; ranges that changed are skipped and counted as "stale".
4. Call madvise(MADV_NOHUGEPAGE) with each remaining VMA's own bounds.

With exact whole-VMA bounds, madvise only rewrites vm_flags: no page-table change, no huge-PMD
split, no MMU notifier, hence no queue eviction. khugepaged never scans a VM_NOHUGEPAGE range, and
MADV_COLLAPSE on one returns EINVAL. The policy stays on that range until something advises it
again: reuse of the range by its allocator inherits it (a re-fault there takes the 4 KiB path), while
brk growth of [heap] gets a new VMA without it, and a later MADV_HUGEPAGE on part of the range (e.g.
NumPy on a heap sub-range) re-enables huge pages for that part and splits the VMA. The sweep never
reverses a THP policy: it only adds VM_NOHUGEPAGE.

THP POLICY: the PMD-size policy is /sys/kernel/mm/transparent_hugepage/hugepages-<PMD>kB/enabled
when present and not "inherit", else /sys/kernel/mm/transparent_hugepage/enabled.
- "madvise" (the cluster's setting) and "always": only 'hg' VMAs are selected. That is the measured
  scope. Under "always", khugepaged could also collapse 4 KiB memory in VMAs without 'hg'; the sweep
  leaves those alone.
- "never": nothing is selected, and the sweep does nothing (khugepaged collapses nothing).
- An unreadable or unknown policy uses the same 'hg' rule and is reported as thp=unknown.

MARKERS (stdout only, one line per rank):
- "armed OK: [thp-sweep] ...": every madvise succeeded.
- "PARTIAL: [thp-sweep] ... errors E errnos={...} ...": some madvise calls failed.
- "NOT armed: [thp-sweep] ...": an error before or during the sweep; whatever was applied before
  the error stays.

COST: one /proc/self/smaps read (~0.5 s at ~23k VMAs), one /proc/self/maps read, and one madvise per
selected VMA, all in the untimed initialization.

NUMERICS: none. This is host memory page-size policy only; training math, samples and order are
unchanged.

MLPERF: reads process metadata (/proc/self/smaps, /proc/self/maps) and the THP policy in sysfs. It
never touches the dataset and uses no data-derived constant.

Never raises.
"""

import ctypes
import os
import time
from typing import Dict, List, Optional, Set, Tuple

ENV = "DLRM_HSTU_FP8_THP_NOHUGE_SWEEP"
MADV_NOHUGEPAGE = 15
_ON = ("1", "true", "on", "yes")
_THP = "/sys/kernel/mm/transparent_hugepage"
_MODES = ("always", "madvise", "never")

# One VMA: (lo, hi, perms, inode, path, rss_kb, ahp_kb, flags)
Vma = Tuple[int, int, str, int, str, int, int, Tuple[str, ...]]


def _say(msg: str) -> None:
    """One line on stdout, which the launcher tees into the run log. A single sink, because the
    launcher merges stderr into the same file and a logger line would double the marker. Never
    raises."""
    try:
        print(msg, flush=True)
    except Exception:  # noqa: BLE001 -- reporting must never break the run
        pass


def enabled() -> bool:
    return os.environ.get(ENV, "").strip().lower() in _ON


def _is_header(parts: List[str]) -> bool:
    if len(parts) < 5 or "-" not in parts[0] or parts[0].endswith(":"):
        return False
    perms = parts[1]
    return len(perms) == 4 and perms[3] in "ps"


def parse_smaps(text_lines) -> List[Vma]:  # pyre-ignore[2]: any iterable of str
    """VMAs from /proc/<pid>/smaps lines. Only the header, Rss, AnonHugePages and VmFlags are used.
    A header whose inode is not an integer gets inode -1, and _anon_private treats that VMA as not
    anonymous."""
    out: List[Vma] = []
    cur: Optional[List[object]] = None

    def _flush() -> None:
        if cur is not None:
            out.append(
                (int(cur[0]), int(cur[1]), str(cur[2]), int(cur[3]), str(cur[4]),  # pyre-ignore[6]
                 int(cur[5]), int(cur[6]), tuple(cur[7]))  # pyre-ignore[6]
            )

    for line in text_lines:
        parts = line.split()
        if not parts:
            continue
        if _is_header(parts):
            _flush()
            lo_s, hi_s = parts[0].split("-", 1)
            path = " ".join(parts[5:]) if len(parts) > 5 else ""
            try:
                inode = int(parts[4])
            except ValueError:
                inode = -1
            cur = [int(lo_s, 16), int(hi_s, 16), parts[1], inode, path, 0, 0, ()]
            continue
        if cur is None:
            continue
        key = parts[0]
        if key == "Rss:" and len(parts) >= 2:
            cur[5] = int(parts[1])
        elif key == "AnonHugePages:" and len(parts) >= 2:
            cur[6] = int(parts[1])
        elif key == "VmFlags:":
            cur[7] = tuple(parts[1:])
    _flush()
    return out


def _anon_private(v: Vma) -> bool:
    _lo, _hi, perms, inode, path, _rss, _ahp, _fl = v
    if perms[3] != "p" or inode != 0:
        return False
    return path == "" or path == "[heap]" or path.startswith("[anon:")


def select(vmas: List[Vma], thp_mode: str = "madvise", pmd: int = 2 * 1024 * 1024
           ) -> Tuple[List[Vma], Dict[str, int]]:
    """VMAs to set MADV_NOHUGEPAGE on, plus counters (KiB) for the arming line.

    Eligible: 'hg' under every policy but "never" (the measured scope); nothing under "never". VMAs
    without one fully contained pmd-aligned range are ignored (khugepaged can never collapse them).
    Selected: eligible + private anonymous + 4 KiB-mapped pages present (Rss > AnonHugePages).
    Kept: eligible private anonymous VMAs whose populated pages are all PMD-mapped (their THP fault
    path stays), and eligible shared / file-backed / [stack] mappings (not this sweep's business)."""
    sel: List[Vma] = []
    c: Dict[str, int] = {
        "n_vma": len(vmas), "sel_n": 0, "sel_kb": 0, "sel_4k_kb": 0, "sel_thp_kb": 0,
        "keep_thp_n": 0, "keep_thp_kb": 0, "keep_other_n": 0, "keep_other_kb": 0,
    }
    if thp_mode == "never":
        return sel, c
    for v in vmas:
        lo, hi, _perms, _inode, _path, rss, ahp, flags = v
        if "hg" not in flags:
            continue
        if (hi & ~(pmd - 1)) - ((lo + pmd - 1) & ~(pmd - 1)) < pmd:
            continue
        size_kb = (hi - lo) // 1024
        if not _anon_private(v):
            c["keep_other_n"] += 1
            c["keep_other_kb"] += size_kb
            continue
        four_k = max(0, rss - ahp)
        if four_k > 0:
            sel.append(v)
            c["sel_n"] += 1
            c["sel_kb"] += size_kb
            c["sel_4k_kb"] += four_k
            c["sel_thp_kb"] += ahp
        else:
            c["keep_thp_n"] += 1
            c["keep_thp_kb"] += size_kb
    return sel, c


def _read_word(path: str) -> Optional[str]:
    """The bracketed word of a sysfs THP policy file, or None."""
    try:
        with open(path) as f:
            txt = f.read()
        i, j = txt.index("["), txt.index("]")
        return txt[i + 1:j].strip()
    except (OSError, ValueError):
        return None


def pmd_size(thp_dir: str = _THP) -> int:
    try:
        with open(os.path.join(thp_dir, "hpage_pmd_size")) as f:
            v = int(f.read().strip())
        return v if v > 0 and (v & (v - 1)) == 0 else 2 * 1024 * 1024
    except (OSError, ValueError):
        return 2 * 1024 * 1024


def thp_mode(thp_dir: str = _THP, pmd: int = 2 * 1024 * 1024) -> str:
    """The PMD-size THP policy: "always" / "madvise" / "never", or "unknown". The per-size file
    (hugepages-<pmd>kB/enabled) wins unless it says "inherit"."""
    per = _read_word(os.path.join(thp_dir, "hugepages-%dkB" % (pmd // 1024), "enabled"))
    if per in _MODES:
        return per  # pyre-ignore[7]
    top = _read_word(os.path.join(thp_dir, "enabled"))
    return top if top in _MODES else "unknown"  # pyre-ignore[7]


def _live_ranges(maps_path: str = "/proc/self/maps") -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    with open(maps_path, "r", errors="replace") as f:
        for line in f:
            r = line.split(" ", 1)[0]
            lo_s, sep, hi_s = r.partition("-")
            if sep:
                out.add((int(lo_s, 16), int(hi_s, 16)))
    return out


def _libc() -> ctypes.CDLL:
    lib = ctypes.CDLL(None, use_errno=True)
    lib.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.madvise.restype = ctypes.c_int
    return lib


def sweep(rank: int = -1, smaps_path: str = "/proc/self/smaps", maps_path: str = "/proc/self/maps",
          thp_dir: str = _THP, mode: Optional[str] = None) -> Dict[str, object]:
    """Read smaps, select, re-check the ranges against maps, madvise(MADV_NOHUGEPAGE) each with its
    own bounds. Returns counters, including CLOCK_BOOTTIME stamps of the start and end (for matching
    against KFD event times). Raises on a read error; sweep_from_env turns that into a NOT-armed
    line."""
    boot0 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    t0 = time.monotonic()
    pmd = pmd_size(thp_dir)
    m = thp_mode(thp_dir, pmd) if mode is None else mode
    with open(smaps_path, "r", errors="replace") as f:
        vmas = parse_smaps(f)
    t_read = time.monotonic() - t0
    sel, c = select(vmas, m, pmd)
    live = _live_ranges(maps_path) if sel else set()
    todo = [v for v in sel if (v[0], v[1]) in live]
    lib = _libc()
    errs: Dict[int, int] = {}
    for lo, hi, *_rest in todo:
        if lib.madvise(ctypes.c_void_p(lo), hi - lo, MADV_NOHUGEPAGE) != 0:
            e = ctypes.get_errno() or -1
            errs[e] = errs.get(e, 0) + 1
    res: Dict[str, object] = dict(c)
    res.update(
        att_kb=sum((v[1] - v[0]) // 1024 for v in todo), att_4k_kb=sum(max(0, v[5] - v[6]) for v in todo),
        att_thp_kb=sum(v[6] for v in todo),
        rank=rank, stale=len(sel) - len(todo), attempted=len(todo), errors=sum(errs.values()),
        errnos=dict(errs), t_read=t_read, t_total=time.monotonic() - t0, mode=m, pmd=pmd,
        pid=os.getpid(), boot_ns0=boot0, boot_ns1=time.clock_gettime_ns(time.CLOCK_BOOTTIME),
    )
    return res


def _gib(kb: object) -> float:
    return float(kb) / (1024.0 * 1024.0)  # pyre-ignore[6]


def format_line(r: Dict[str, object]) -> str:
    head = "armed OK" if not r["errors"] else "PARTIAL"
    return (
        "%s: [thp-sweep] rank=%d before run_start: MADV_NOHUGEPAGE on %d VMA(s) %.2f GiB "
        "(4K-mapped %.2f GiB, THP-mapped %.2f GiB); kept hg: %d THP-only VMA(s) %.2f GiB, "
        "%d shared/file VMA(s) %.2f GiB; stale %d; errors %d%s; smaps %d VMAs read in %.3fs, total %.3fs; "
        "thp=%s pid=%d boot_ns0=%d boot_ns1=%d"
        % (
            head, r["rank"], r["attempted"], _gib(r["att_kb"]), _gib(r["att_4k_kb"]), _gib(r["att_thp_kb"]),
            r["keep_thp_n"], _gib(r["keep_thp_kb"]), r["keep_other_n"], _gib(r["keep_other_kb"]),
            r["stale"], r["errors"], (" errnos=%s" % r["errnos"]) if r["errors"] else "",
            r["n_vma"], r["t_read"], r["t_total"], r["mode"], r["pid"], r["boot_ns0"], r["boot_ns1"],
        )
    )


def sweep_from_env(rank: int = -1) -> Optional[Dict[str, object]]:
    """The trainer hook: a no-op (no line) unless DLRM_HSTU_FP8_THP_NOHUGE_SWEEP is 1/true/on/yes.
    Never raises."""
    try:
        if not enabled():
            return None
        r = sweep(rank=rank)
        _say(format_line(r))
        return r
    except Exception as e:  # noqa: BLE001 -- never raise: the run continues
        try:
            _say("NOT armed: [thp-sweep] rank=%s: %s: %s" % (rank, type(e).__name__, e))
        except Exception:  # noqa: BLE001
            pass
        return None
