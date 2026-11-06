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
"""Targeted HIP code-object preload before RUN_START (ship-14 item S14-d v2; default ON via the yambda_5b.gin binding).

ROCm 7.2.1 loads statically registered kernels lazily (HIP deferred loading, HIP_ENABLE_DEFERRED_LOADING unset = 1):
the FIRST launch of any kernel of a translation unit (TU) of libtorch_hip.so / fbgemm_gpu*.so digests that TU's fat
binary (COMGR unbundle + zstd decompression for libtorch_hip) and loads its gfx950 code object on the current device.
That costs ~1 ms per uncompressed MiB, and the TUs a production step needs (~40) are first launched inside training
step 1, i.e. inside the timed region (dlstart iter 34: ~1.5-2.2 s per rank, model estimate).

hipFuncGetAttributes(&attr, host_stub) on one kernel stub per needed TU does exactly that load and nothing else: it
launches no kernel, allocates no tensor, and touches no RNG, model, optimizer or data state. The stubs come from the
checked-in table hip_preload_table.jsonl (tools/recommendation/dlrm_v4/gen_hip_preload_table.py over the ship13 1n/2n/4n production
traces, plus the v2 rows listed in its header "amendments": step-1-only first-use code objects that steady-step traces
cannot show, rccl iter85 S14D85 section 6), keyed by library basename + mangled symbol:
  * mode "dlsym":  dlsym in the ALREADY-LOADED library (dlopen RTLD_NOLOAD; a library that is not loaded is skipped);
  * mode "symtab": local stubs (anonymous-namespace kernels are not exported): load base + st_value, used only when
    the loaded library's GNU build-id (read from its mapped ELF notes) equals the table's, and the base derived from
    a dlsym'd anchor symbol equals the library's offset-0 mapping in /proc/self/maps.
Every entry is fail-open: a missing library/symbol, a build-id mismatch or a HIP error is counted and skipped; a HIP
error is cleared with hipGetLastError so no later torch launch check sees it. Never raises.

Per software stack (T12, docs/04-technical-guides/recommendation/dlrm_v4/rocm-port.md): rows name symbols, TU indices and symtab offsets of ONE build, so there
is one checked-in table per stack (STACK_TABLES) and DLRM_HSTU_FP8_HIP_PRELOAD=1/all picks the table whose header
build-ids match the libraries this process has loaded (pick_table; build-ids read through dl_iterate_phdr, which also
covers TheRock's libtorch_hip whose program header table sits at the end of the file). ROCm 10.2 (TheRock) torch keeps
its device code in torch/.kpack/torch_gfx950.kpack (one zstd frame per TU, loaded per TU on first use through
librocm_kpack's kpack_load_code_object); the mechanism is unchanged: hipFuncGetAttributes on a stub of the TU. Measured
on 10.2 (coload_shim load log, compat image, 1 node): a code object loaded inside the timed region costs 20-70 ms of
host time whatever its size (about 4 ms during initialization), about 38 per rank without this module.

MLPerf (training_rules.adoc "Run Results" L545-549): model initialization before the clock may include work that does
not touch the dataset; code-object loading is library initialization (same class as the Triton/HIP compile, BLAS and
FlyDSL warmups already run before RUN_START; knowledge/mlperf-pre-runstart.md "Allowed before RUN_START": imports,
Triton/HIP compile/JIT). It reads no sample and leaves all training state bit-identical.

Knob (gin apply_env_bootstrap.DLRM_HSTU_FP8_HIP_PRELOAD, yambda_5b.gin item (S14-d); the env bootstrap writes it into
the environment before this module reads it; DLRM_HSTU_FP8_ prefix -> forwarded by scripts/launch_slurm.sh):
  DLRM_HSTU_FP8_HIP_PRELOAD   default ON via the yambda_5b.gin binding (S14-d v2: True -> "1"); code default unset =
                              off (no HIP lookup, no log line). A pre-set env value wins over the binding (_bind_bool):
                              0/off = off, i.e. "0" is the kill switch; 1/on = the checked-in table, entries of this run
                              shape (<nodes>n) only; "all" = every entry; any other value = path of a table file ("all"
                              and a path are env-only values).
  DLRM_HSTU_FP8_HIP_PRELOAD_REPORT=<dir>  optional per-rank JSON with per-entry milliseconds (diagnostic, env only).
Arming line (rank 0): "armed OK: [hip-preload] ...", or "[hip-preload] NOT armed: ..." when nothing could be loaded.
"""

import ctypes
import json
import logging
import os
import struct
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger: logging.Logger = logging.getLogger(__name__)

ENV = "DLRM_HSTU_FP8_HIP_PRELOAD"
REPORT_ENV = "DLRM_HSTU_FP8_HIP_PRELOAD_REPORT"
DEFAULT_TABLE: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hip_preload_table.jsonl")
# One checked-in table per software stack (rows hold symbols, bundle/TU indices and symtab offsets of one build).
# DLRM_HSTU_FP8_HIP_PRELOAD=1/all picks the table whose header build-ids match the libraries this process has
# loaded (pick_table); no match keeps DEFAULT_TABLE (its rows then fail open as before, and rank 0 adds the warning
# "[hip-preload] table=... matches the build-id of none of the N table libraries loaded").
#   hip_preload_table.jsonl            ROCm 7.2.1, torch 2.12.0+rocm7.2, FBGEMM 10b77573 (rocm/primus:v26.3 recipe)
#   hip_preload_table_rocm10.2.jsonl   TheRock ROCm 10.2.0a20261015, torch 2.16.0a0 (kpack device code), FBGEMM
#                                      nightly 2026.10.16 (UTD primus_rocm10.2_20261016 image)
STACK_TABLES: Tuple[str, ...] = (
    DEFAULT_TABLE,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "hip_preload_table_rocm10.2.jsonl"),
)
ARMED = "armed OK: [hip-preload]"
NOT_ARMED = "[hip-preload] NOT armed:"
_OFF = ("", "0", "off", "false", "no", "none")
_ON = ("1", "on", "true", "yes")
# An entry faster than this was already loaded (a hash lookup, ~5 us); the smallest real load measured 0.2 ms.
ALREADY_LOADED_MS = 0.05

RTLD_NOLOAD: int = getattr(os, "RTLD_NOLOAD", 4)
RTLD_LAZY: int = getattr(os, "RTLD_LAZY", 1)


def resolve_table(value: Optional[str]) -> Tuple[Optional[str], bool]:
    """env value -> (table path or None = off, use every shape). Pure."""
    v = (value or "").strip()
    if v.lower() in _OFF:
        return None, False
    if v.lower() in _ON:
        return DEFAULT_TABLE, False
    if v.lower() == "all":
        return DEFAULT_TABLE, True
    return v, True


def load_table(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int]:
    """(header, entries, n_bad_lines). A bad line is counted, not raised; a missing/unreadable file raises OSError."""
    header: Dict[str, Any] = {}
    entries: List[Dict[str, Any]] = []
    bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if not isinstance(row, dict):
                bad += 1
                continue
            kind = row.get("kind")
            if kind == "header":
                header = row
            elif (kind == "entry" and isinstance(row.get("lib"), str) and isinstance(row.get("sym"), str)
                  and row.get("mode") in ("dlsym", "symtab")
                  and (row["mode"] == "dlsym" or isinstance(row.get("value"), int))):
                entries.append(row)
            else:
                bad += 1
    return header, entries, bad


def read_header(path: str) -> Dict[str, Any]:
    """The header row of a table ({} if the file cannot be read or has none). Never raises."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                return row if isinstance(row, dict) and row.get("kind") == "header" else {}
    except (OSError, ValueError):
        pass
    return {}


def loaded_build_ids(basename: str, maps: Dict[str, Dict[str, int]]) -> List[str]:
    """GNU build-ids of every loaded copy of a library (read from its mapped ELF notes)."""
    out: List[str] = []
    for path in find_loaded(basename, maps):
        bid = memory_build_id(maps[path]["base"])
        if bid:
            out.append(bid)
    return out


def build_match(header: Dict[str, Any], maps: Dict[str, Dict[str, int]]) -> Tuple[int, int]:
    """(table libraries loaded with the table's build-id, table libraries loaded at all)."""
    hit = loaded = 0
    for name, meta in (header.get("libs") or {}).items():
        bids = loaded_build_ids(name, maps)
        if bids:
            loaded += 1
            if (meta or {}).get("build_id") in bids:
                hit += 1
    return hit, loaded


def pick_table(paths: Tuple[str, ...], maps: Dict[str, Dict[str, int]]) -> Tuple[str, Tuple[int, int]]:
    """The per-stack table for this process: the one with the most libraries loaded with its recorded build-id
    (ties and no match at all: the earlier path). Returns (path, (matched, loaded))."""
    best, best_m = paths[0], (-1, 0)
    for p in paths:
        if not os.path.exists(p):
            continue
        m = build_match(read_header(p), maps)
        if m[0] > best_m[0]:
            best, best_m = p, m
    if best_m[0] < 0:
        best_m = (0, 0)
    return best, best_m


def select_entries(header: Dict[str, Any], entries: List[Dict[str, Any]], shape: Optional[str],
                   use_all: bool) -> Tuple[List[Dict[str, Any]], int]:
    """Entries for this run shape ('1n', '2n', ...): those whose 'sets' contain it. A shape the table was not generated
    for, or use_all, keeps every entry. Returns (selected, n_other_shape)."""
    known = set((header.get("traces") or {}).keys())
    if use_all or not shape or shape not in known:
        return list(entries), 0
    sel = [e for e in entries if shape in (e.get("sets") or [shape])]
    return sel, len(entries) - len(sel)


# ----------------------------------------------------------------------------------------- process introspection
def mapped_libraries() -> Dict[str, Dict[str, int]]:
    """{path: {'base': start of the offset-0 mapping}} of file-backed mappings of this process."""
    out: Dict[str, Dict[str, int]] = {}
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) < 6:
                    continue
                path = parts[5].strip()
                if not path.startswith("/"):
                    continue
                lo = int(parts[0].split("-")[0], 16)
                if int(parts[2], 16) == 0:
                    cur = out.setdefault(path, {"base": lo})
                    cur["base"] = min(cur["base"], lo)
    except OSError:
        pass
    return out


def find_loaded(basename: str, maps: Dict[str, Dict[str, int]]) -> List[str]:
    return sorted(p for p in maps if os.path.basename(p) == basename)


def _notes_build_id(notes: bytes) -> Optional[str]:
    q = 0
    while q + 12 <= len(notes):
        namesz, descsz, ntype = struct.unpack_from("<III", notes, q)
        name = notes[q + 12:q + 12 + namesz]
        dq = q + 12 + ((namesz + 3) & ~3)
        if ntype == 3 and name.rstrip(b"\0") == b"GNU":
            return notes[dq:dq + descsz].hex()
        q = dq + ((descsz + 3) & ~3)
    return None


# preload_from_env snapshots loaded_object_build_ids() once for its build-id checks (one dl_iterate_phdr walk)
_BID_CACHE: Optional[Dict[int, str]] = None


class _DlPhdrInfo(ctypes.Structure):
    _fields_ = [("dlpi_addr", ctypes.c_uint64), ("dlpi_name", ctypes.c_char_p), ("dlpi_phdr", ctypes.c_void_p),
                ("dlpi_phnum", ctypes.c_uint16)]


def loaded_object_build_ids() -> Dict[int, str]:
    """{load base: GNU build-id} of every object the dynamic loader has mapped (dl_iterate_phdr: the loader's own
    pointer to each object's program headers, then its PT_NOTE segments in memory). Needed for libraries whose program
    header table is not inside their first PT_LOAD (TheRock's kpack post-processing appends a new table at the end of
    libtorch_hip.so: e_phoff is then a file offset with no fixed relation to the mapping). {} on any failure."""
    out: Dict[int, str] = {}
    try:
        libc = ctypes.CDLL(None)
        cb_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_DlPhdrInfo), ctypes.c_size_t, ctypes.c_void_p)

        def cb(info: Any, _size: int, _data: Any) -> int:
            try:
                i = info.contents
                ph = ctypes.string_at(i.dlpi_phdr, 56 * i.dlpi_phnum) if i.dlpi_phdr else b""
                for k in range(i.dlpi_phnum):
                    p_type, _fl, _off, p_vaddr, _pa, _fs, p_memsz, _al = struct.unpack_from("<IIQQQQQQ", ph, 56 * k)
                    if p_type == 4 and p_memsz:  # PT_NOTE
                        bid = _notes_build_id(ctypes.string_at(i.dlpi_addr + p_vaddr, p_memsz))
                        if bid:
                            out[int(i.dlpi_addr)] = bid
                            break
            except Exception:  # noqa: BLE001
                pass
            return 0

        libc.dl_iterate_phdr(cb_t(cb), None)
    except Exception:  # noqa: BLE001
        return {}
    return out


def memory_build_id(base: int) -> Optional[str]:
    """GNU build-id of the ELF image mapped at `base` (a loaded library's PT_NOTE). The caller has checked that `base`
    is the start of the library's offset-0 mapping. Uses the dynamic loader's program-header pointer
    (loaded_object_build_ids); falls back to e_phoff read from the mapped ELF header (valid when the program header
    table lies in the first PT_LOAD, as in an ordinary link)."""
    bid = (_BID_CACHE if _BID_CACHE is not None else loaded_object_build_ids()).get(base)
    if bid:
        return bid
    try:
        ehdr = ctypes.string_at(base, 64)
        if ehdr[:4] != b"\x7fELF" or ehdr[4] != 2:
            return None
        e_phoff = struct.unpack_from("<Q", ehdr, 0x20)[0]
        e_phentsize, e_phnum = struct.unpack_from("<HH", ehdr, 0x36)
        ph = ctypes.string_at(base + e_phoff, e_phentsize * e_phnum)
        for i in range(e_phnum):
            p_type, _fl, _off, p_vaddr, _pa, _fs, p_memsz, _al = struct.unpack_from("<IIQQQQQQ", ph, i * e_phentsize)
            if p_type != 4:  # PT_NOTE
                continue
            notes = ctypes.string_at(base + p_vaddr, p_memsz)
            q = 0
            while q + 12 <= len(notes):
                namesz, descsz, ntype = struct.unpack_from("<III", notes, q)
                name = notes[q + 12:q + 12 + namesz]
                dq = q + 12 + ((namesz + 3) & ~3)
                if ntype == 3 and name.rstrip(b"\0") == b"GNU":
                    return notes[dq:dq + descsz].hex()
                q = dq + ((descsz + 3) & ~3)
    except Exception:  # noqa: BLE001
        return None
    return None


class _Lib:
    """One already-loaded library: dlsym handle, and (if verified) the base for symtab entries."""

    def __init__(self, handle: Optional[ctypes.CDLL], path: Optional[str], symtab_base: Optional[int],
                 reason: str) -> None:
        self.handle = handle
        self.path = path
        self.symtab_base = symtab_base
        self.reason = reason  # why symtab entries are unusable ('' = usable)


def _dlsym(handle: ctypes.CDLL, sym: str) -> Optional[int]:
    try:
        return ctypes.cast(handle[sym], ctypes.c_void_p).value
    except (AttributeError, ValueError, TypeError):
        return None


def open_library(basename: str, meta: Dict[str, Any], maps: Dict[str, Dict[str, int]]) -> _Lib:
    paths = find_loaded(basename, maps)
    if not paths:
        return _Lib(None, None, None, "nolib")
    want = meta.get("build_id")
    chosen: Optional[_Lib] = None
    for path in paths:
        try:
            handle = ctypes.CDLL(path, mode=RTLD_NOLOAD | RTLD_LAZY)
        except OSError:
            continue
        base = maps[path]["base"]
        reason = ""
        anchor = meta.get("anchor") or {}
        a_addr = _dlsym(handle, anchor["sym"]) if anchor.get("sym") else None
        if a_addr is None or not isinstance(anchor.get("value"), int):
            reason = "anchor"
        elif a_addr - int(anchor["value"]) != base:
            reason = "base"
        elif not want or memory_build_id(base) != want:
            reason = "buildid"
        lib = _Lib(handle, path, base if not reason else None, reason)
        if not reason:
            return lib
        chosen = chosen or lib
    return chosen or _Lib(None, None, None, "nolib")


class _Hip:
    """hipFuncGetAttributes / hipGetLastError / hipGetDevice of the HIP runtime the kernels are registered with.
    Resolved with dlsym through the already-loaded libtorch_hip.so handle (its own dependency tree, i.e. exactly the
    libamdhip64 libtorch_hip binds to), else from the loaded libamdhip64 itself. Never dlopens a new library."""

    def __init__(self, maps: Dict[str, Dict[str, int]]) -> None:
        lib: Optional[ctypes.CDLL] = None
        self.path: str = ""
        for via in find_loaded("libtorch_hip.so", maps) + sorted(
                p for p in maps if os.path.basename(p).startswith("libamdhip64.so")):
            try:
                cand = ctypes.CDLL(via, mode=RTLD_NOLOAD | RTLD_LAZY)
                cand.hipFuncGetAttributes  # noqa: B018  (dlsym probe)
            except (OSError, AttributeError):
                continue
            lib, self.path = cand, via
            break
        if lib is None:
            raise OSError("no loaded library resolves hipFuncGetAttributes (libtorch_hip/libamdhip64 not loaded)")
        self._fga = lib.hipFuncGetAttributes
        self._fga.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._fga.restype = ctypes.c_int
        self._gle = lib.hipGetLastError
        self._gle.argtypes = []
        self._gle.restype = ctypes.c_int
        self._gdev = lib.hipGetDevice
        self._gdev.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._gdev.restype = ctypes.c_int
        # hipFuncAttributes is ~100 bytes in ROCm 7.x; 1 KiB leaves room for any layout growth.
        self._buf = ctypes.create_string_buffer(1024)

    def func_attributes(self, addr: int) -> int:
        return int(self._fga(self._buf, ctypes.c_void_p(addr)))

    def clear_error(self) -> int:
        return int(self._gle())

    def device(self) -> int:
        d = ctypes.c_int(-1)
        return d.value if self._gdev(ctypes.byref(d)) == 0 else -1


def preload_entries(entries: List[Dict[str, Any]], libs_meta: Dict[str, Dict[str, Any]],
                    func_attributes: Callable[[int], int], clear_error: Callable[[], int],
                    maps: Optional[Dict[str, Dict[str, int]]] = None,
                    opener: Callable[[str, Dict[str, Any], Dict[str, Dict[str, int]]], _Lib] = open_library,
                    ) -> Dict[str, Any]:
    """Core loop (testable without a GPU: pass fake func_attributes/clear_error). Fail-open per entry."""
    maps = mapped_libraries() if maps is None else maps
    cnt = {"ok": 0, "already": 0, "nolib": 0, "nosym": 0, "symtab_unverified": 0, "hip": 0, "error": 0}
    rows: List[Dict[str, Any]] = []
    libs: Dict[str, _Lib] = {}
    t_all = time.perf_counter()
    for e in entries:
        name = e["lib"]
        try:
            if name not in libs:
                libs[name] = opener(name, libs_meta.get(name) or {}, maps)
            lib = libs[name]
            if lib.handle is None:
                cnt["nolib"] += 1
                rows.append({"lib": name, "bundle": e.get("bundle"), "res": "nolib"})
                continue
            if e["mode"] == "dlsym":
                addr = _dlsym(lib.handle, e["sym"])
            elif lib.symtab_base is not None:
                addr = lib.symtab_base + int(e["value"])
            else:
                cnt["symtab_unverified"] += 1
                rows.append({"lib": name, "bundle": e.get("bundle"), "res": "symtab:" + lib.reason})
                continue
            if not addr:
                cnt["nosym"] += 1
                rows.append({"lib": name, "bundle": e.get("bundle"), "res": "nosym"})
                continue
            t = time.perf_counter()
            rc = func_attributes(addr)
            ms = (time.perf_counter() - t) * 1e3
            if rc != 0:
                clear_error()
                cnt["hip"] += 1
                rows.append({"lib": name, "bundle": e.get("bundle"), "res": f"hip{rc}", "ms": ms})
                continue
            cnt["ok"] += 1
            if ms < ALREADY_LOADED_MS:
                cnt["already"] += 1
            rows.append({"lib": name, "bundle": e.get("bundle"), "res": "ok", "ms": round(ms, 4),
                         "est_ms": e.get("est_ms")})
        except Exception as error:  # noqa: BLE001  (fail-open per entry)
            cnt["error"] += 1
            rows.append({"lib": name, "bundle": e.get("bundle"), "res": f"error:{type(error).__name__}"})
            try:
                clear_error()
            except Exception:  # noqa: BLE001
                pass
    return {"counts": cnt, "rows": rows, "seconds": time.perf_counter() - t_all,
            "libs": {k: {"path": v.path, "symtab": v.symtab_base is not None, "reason": v.reason}
                     for k, v in libs.items()}}


# --------------------------------------------------------------------- trainer HIP extensions (built per run)
# The trainer builds some HIP extensions at run time under tempfile.gettempdir()/dlrm_hstu_* (the onesweep sort, the
# FP8 wgrad CK kernels, ...). No checked-in table can name them (a new build per source or toolchain change), and their
# kernels load lazily at first use like any static HIP code: measured on ROCm 10.2, dlrm_hstu_onesweep_sort_rb10.so
# 30-48 ms per rank inside step 1. Their rows are derived from each loaded file here: one dlsym row per fat-binary
# bundle, an exported kernel handle registered first in that bundle (the rule of tools/recommendation/dlrm_v4/gen_hip_preload_table.py).
EXT_MARK = os.sep + "dlrm_hstu"
_DELETED = " (deleted)"
_ZSTD: Any = None


def _elf_sections(elf: bytes) -> List[Dict[str, int]]:
    e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    raw = [struct.unpack_from("<IIQQQQIIQQ", elf, e_shoff + i * e_shentsize) for i in range(e_shnum)]
    names = raw[e_shstrndx][4] if e_shstrndx < len(raw) else 0
    out: List[Dict[str, Any]] = []
    for (n, typ, _fl, _addr, off, size, link, _info, _al, ent) in raw:
        end = elf.index(b"\0", names + n)
        out.append({"name": elf[names + n:end].decode(errors="replace"), "type": typ, "off": off, "size": size,
                    "link": link, "ent": ent})
    return out


def _elf_symbols(elf: bytes, secs: List[Dict[str, Any]], sh_type: int) -> Dict[str, Tuple[int, int, int]]:
    """{name: (bind, visibility, st_shndx != 0)} of the SHT_SYMTAB(2) / SHT_DYNSYM(11) symbols."""
    out: Dict[str, Tuple[int, int, int]] = {}
    for s in secs:
        if s["type"] != sh_type or not s["ent"] or s["link"] >= len(secs):
            continue
        stoff = secs[s["link"]]["off"]
        for k in range(s["size"] // s["ent"]):
            st_name, st_info, st_other, st_shndx, _v, _sz = struct.unpack_from("<IBBHQQ", elf, s["off"] + k * s["ent"])
            end = elf.index(b"\0", stoff + st_name)
            out.setdefault(elf[stoff + st_name:end].decode(errors="replace"), (st_info >> 4, st_other & 3, st_shndx))
    return out


def _zstd_decompress(payload: bytes) -> Optional[bytes]:
    """libzstd through ctypes (None if libzstd is not available)."""
    global _ZSTD
    try:
        if _ZSTD is None:
            import ctypes.util

            lib = ctypes.CDLL(ctypes.util.find_library("zstd") or "libzstd.so.1")
            lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
            lib.ZSTD_getFrameContentSize.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
            lib.ZSTD_decompress.restype = ctypes.c_size_t
            lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
            lib.ZSTD_isError.restype = ctypes.c_uint
            lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
            _ZSTD = lib
        n = _ZSTD.ZSTD_getFrameContentSize(payload, len(payload))
        if n >= (1 << 63):
            return None
        out = ctypes.create_string_buffer(n)
        got = _ZSTD.ZSTD_decompress(out, n, payload, len(payload))
        return None if _ZSTD.ZSTD_isError(got) else out.raw[:got]
    except Exception:  # noqa: BLE001
        return None


def _bundle_kernels(ub: bytes) -> List[str]:
    """Kernel names ('<mangled>' of '<mangled>.kd') of the first amdgcn code object of an uncompressed offload
    bundle (every device target of a bundle carries the same kernels)."""
    if ub[:24] != b"__CLANG_OFFLOAD_BUNDLE__":
        return []
    n = struct.unpack_from("<Q", ub, 24)[0]
    q = 32
    for _ in range(n):
        eoff, esz, tl = struct.unpack_from("<QQQ", ub, q)
        triple = ub[q + 24:q + 24 + tl].decode(errors="replace")
        q += 24 + tl
        if "amdgcn" in triple and esz:
            co = ub[eoff:eoff + esz]
            if co[:4] != b"\x7fELF":
                return []
            secs = _elf_sections(co)
            names = set()
            for t in (2, 11):
                names.update(nm[:-3] for nm, (_b, _v, shndx) in _elf_symbols(co, secs, t).items()
                             if nm.endswith(".kd") and shndx)
            return sorted(names)
    return []


def extension_rows(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """(dlsym rows, bundles without an exported kernel handle) of one runtime-built extension library file."""
    with open(path, "rb") as f:
        elf = f.read()
    secs = _elf_sections(elf)
    fb = [s for s in secs if s["name"] == ".hip_fatbin" and s["type"] != 8]  # 8 = SHT_NOBITS
    if not fb:
        return [], 0
    sec = elf[fb[0]["off"]:fb[0]["off"] + fb[0]["size"]]
    dyn = _elf_symbols(elf, secs, 11)
    bundles: List[List[str]] = []
    pos = 0
    while True:
        i_plain, i_ccob = sec.find(b"__CLANG_OFFLOAD_BUNDLE__", pos), sec.find(b"CCOB", pos)
        cands = [i for i in (i_plain, i_ccob) if i >= 0]
        if not cands:
            break
        o = min(cands)
        if o == i_ccob:
            ver = struct.unpack_from("<H", sec, o + 4)[0]
            if ver == 3:
                tot, hdr = struct.unpack_from("<Q", sec, o + 8)[0], 32
            elif ver == 2:
                tot, hdr = struct.unpack_from("<I", sec, o + 8)[0], 24
            else:
                break
            ub = _zstd_decompress(bytes(sec[o + hdr:o + tot])) or b""
            pos = o + max(tot, 4)
        else:
            n = struct.unpack_from("<Q", sec, o + 24)[0]
            q, end = o + 32, o + 32
            for _ in range(n):
                eoff, esz, tl = struct.unpack_from("<QQQ", sec, q)
                q += 24 + tl
                end = max(end, o + eoff + esz, q)
            ub = bytes(sec[o:end])
            pos = end
        bundles.append(_bundle_kernels(ub))
    first: Dict[str, int] = {}
    for i, ks in enumerate(bundles):
        for k in ks:
            first.setdefault(k, i)
    rows: List[Dict[str, Any]] = []
    nostub = 0
    for i, ks in enumerate(bundles):
        own = [k for k in ks if first.get(k) == i]
        pick = next((k for k in own if k in dyn and dyn[k][0] in (1, 2) and dyn[k][1] in (0, 3) and dyn[k][2]), None)
        if pick is None:
            nostub += 1 if own else 0
            continue
        rows.append({"lib": path, "sym": pick, "mode": "dlsym", "bundle": i, "ext": True})
    return rows, nostub


def _open_path(name: str, meta: Dict[str, Any], maps: Dict[str, Dict[str, int]]) -> "_Lib":
    """open_library for extension rows: 'lib' is the loaded file's full path (several builds may share a name)."""
    try:
        return _Lib(ctypes.CDLL(name, mode=RTLD_NOLOAD | RTLD_LAZY), name, None, "symtab")
    except OSError:
        return _Lib(None, None, None, "nolib")


def preload_extensions(maps: Dict[str, Dict[str, int]], func_attributes: Callable[[int], int],
                       clear_error: Callable[[], int]) -> Dict[str, Any]:
    """Rows of every loaded trainer extension (path contains EXT_MARK, a .hip_fatbin section) through the same
    fail-open loop as the table rows. Never raises."""
    rows: List[Dict[str, Any]] = []
    libs = nostub = bad = 0
    seen = set()
    for mapped in sorted(maps):
        # A build cache raced by the local ranks (wgrad_ck.build: each rank compiles, then renames into place;
        # identical bytes, last rename wins) leaves the earlier ranks' mapping on a replaced file: /proc/self/maps
        # then shows "<path> (deleted)". The loader still knows the object by <path> (dlopen RTLD_NOLOAD matches the
        # loaded name) and the file now at <path> carries the same kernels.
        path = mapped[:-len(_DELETED)] if mapped.endswith(_DELETED) else mapped
        if EXT_MARK not in path or not path.endswith(".so") or path in seen:
            continue
        seen.add(path)
        try:
            r, n = extension_rows(path)
        except Exception:  # noqa: BLE001  (a file we cannot parse is skipped and counted)
            bad += 1
            continue
        libs += 1 if r or n else 0
        nostub += n
        rows += r
    if rows:
        res = _PRELOAD_ENTRIES(rows, {}, func_attributes, clear_error, maps, opener=_open_path)
    else:
        res = {"counts": {"ok": 0, "already": 0, "nolib": 0, "nosym": 0, "symtab_unverified": 0, "hip": 0,
                          "error": 0}, "rows": [], "seconds": 0.0, "libs": {}}
    return res | {"n_libs": libs, "nostub": nostub, "unparsed": bad, "n_rows": len(rows)}


_PRELOAD_ENTRIES = preload_entries  # the loop itself, for the extension rows (tests replace preload_entries)


def preload_from_env(rank: int = 0, world_size: int = 1, local_world_size: int = 1,
                     env: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """train_ranker hook: after torch.cuda.set_device + the library imports + every warmup, right before RUN_START.
    Off (code default when the env is unset; yambda_5b.gin binds it on) = returns None without importing or logging
    anything. Never raises."""
    env = os.environ if env is None else env
    path, use_all = resolve_table(env.get(ENV))
    if path is None:
        return None
    t0 = time.perf_counter()
    global _BID_CACHE
    try:
        maps = mapped_libraries()
        _BID_CACHE = loaded_object_build_ids()
        if path == DEFAULT_TABLE:  # "1"/"all": the checked-in table of the loaded stack (build-ids)
            path, builds = pick_table(STACK_TABLES, maps)
        else:
            builds = build_match(read_header(path), maps)
        header, entries, bad = load_table(path)
        shape = f"{max(1, world_size // max(1, local_world_size))}n"
        sel, other = select_entries(header, entries, shape, use_all)
        hip = _Hip(maps)
        dev = hip.device()
        res = preload_entries(sel, header.get("libs") or {}, hip.func_attributes, hip.clear_error, maps)
        try:  # the trainer's own runtime-built extensions: separate, so a failure here leaves the table result intact
            ext = preload_extensions(maps, hip.func_attributes, hip.clear_error)
        except Exception as ext_error:  # noqa: BLE001
            ext = {"counts": {"ok": 0, "already": 0, "nolib": 0, "nosym": 0, "symtab_unverified": 0, "hip": 0,
                              "error": 1}, "rows": [{"res": f"error:{type(ext_error).__name__}"}], "seconds": 0.0,
                   "n_libs": 1, "nostub": 0, "unparsed": 0, "n_rows": 0}
    except Exception as error:  # noqa: BLE001
        if rank == 0:
            logger.warning("%s %s: %s (run continues; step 1 loads the code objects lazily)", NOT_ARMED,
                           type(error).__name__, error)
        return {"armed": False, "error": repr(error)}
    finally:
        _BID_CACHE = None
    c = res["counts"]
    loaded_ms = sum(r.get("ms", 0.0) for r in res["rows"] if r.get("res") == "ok")
    failed = c["nolib"] + c["nosym"] + c["symtab_unverified"] + c["hip"] + c["error"]
    summary = {"armed": c["ok"] > 0, "table": path, "shape": shape, "selected": len(sel), "other_shape": other,
               "bad_lines": bad, "device": dev, "counts": c, "failed": failed, "seconds": time.perf_counter() - t0,
               "hip_seconds": res["seconds"], "loaded_ms": loaded_ms, "hip_via": hip.path, "libs": res["libs"],
               "builds": list(builds), "stack": header.get("stack"),
               "extensions": {k: ext[k] for k in ("counts", "seconds", "n_libs", "nostub", "unparsed", "n_rows")}}
    report = env.get(REPORT_ENV, "")
    if report and report.lower() not in _OFF:
        try:
            os.makedirs(report, exist_ok=True)
            with open(os.path.join(report, f"hip_preload_rank{rank}.json"), "w") as f:
                json.dump(summary | {"rows": res["rows"], "extension_rows": ext["rows"]}, f, indent=1)
        except OSError as error:
            logger.warning("[hip-preload] report not written: %s", error)
    msg = (f"table={os.path.basename(path)} shape={shape} entries={len(sel)} (other_shape={other} bad_lines={bad}) "
           f"ok={c['ok']} already_loaded={c['already']} failed={failed} (nolib={c['nolib']} nosym={c['nosym']} "
           f"symtab_unverified={c['symtab_unverified']} hip={c['hip']} error={c['error']}) "
           f"in {summary['seconds']:.2f}s (hip {res['seconds']:.2f}s) dev={dev} before run_start; "
           f"no kernel launch, no tensor/RNG state")
    if rank == 0:
        if c["ok"] > 0:
            logger.info("%s %s", ARMED, msg)
        else:
            logger.warning("%s %s", NOT_ARMED, msg)
        if builds[1] and not builds[0]:  # no table library is this build: no table for this software stack
            logger.warning("[hip-preload] table=%s matches the build-id of none of the %d table libraries loaded: "
                           "no checked-in table for this stack; its dlsym rows load whatever code objects those "
                           "symbols now belong to, its symtab rows are skipped. Regenerate the table for this stack "
                           "(tools/recommendation/dlrm_v4/gen_hip_preload_table.py)", os.path.basename(path), builds[1])
    elif failed:
        logger.warning("[hip-preload] rank %d: %s", rank, msg)
    ec = ext["counts"]
    ext_failed = ec["nolib"] + ec["nosym"] + ec["hip"] + ec["error"] + ext["unparsed"]
    if ext_failed or (ext["n_libs"] and rank == 0):  # unparsed-only failures (n_libs=0) are reported too
        logger.log(logging.WARNING if ext_failed else logging.INFO,
                   "[hip-preload] trainer extensions%s: libs=%d bundles=%d ok=%d already_loaded=%d failed=%d "
                   "(nosym=%d hip=%d error=%d unparsed=%d) no_exported_stub=%d in %.2fs",
                   "" if rank == 0 else f" rank {rank}", ext["n_libs"], ext["n_rows"], ec["ok"], ec["already"],
                   ext_failed, ec["nosym"], ec["hip"], ec["error"], ext["unparsed"], ext["nostub"], ext["seconds"])
    return summary
