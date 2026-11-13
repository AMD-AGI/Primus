#!/usr/bin/env python3
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
"""Generate generative_recommenders/dlrm_v4/train/hip_preload_table.jsonl (offline, CPU only).

The table lists, for every statically registered HIP fat-binary bundle (one per translation unit, "TU") whose kernels
a production training step launches, ONE host kernel stub that belongs to that bundle. At run time
(dlrm_v4/train/hip_preload.py, DLRM_HSTU_FP8_HIP_PRELOAD=1, the yambda_5b.gin default since S14-d v2)
hipFuncGetAttributes on that stub makes the HIP runtime
digest the bundle and load its gfx950 code object on the current device -- the work HIP's deferred loading
(ROCm 7.2.1 default, HIP_ENABLE_DEFERRED_LOADING unset) would otherwise do inside the first launch of any kernel of
that TU, i.e. inside training step 1.

Inputs:
  --lib PATH        a library with a .hip_fatbin section (libtorch_hip.so, fbgemm_gpu*.so) or a TheRock kpack library
                    (ROCm 10.2 torch: .hip_fatbin is NOBITS, .rocm_kpack_ref + HIPK wrappers point into
                    torch/.kpack/torch_<gfx>.kpack, one zstd frame per TU); repeatable
  --trace SET=GLOB  Kineto chrome traces of production steps; SET labels the run shape (1n, 2n, 4n); repeatable
  --loads SET=GLOB  code-object load logs of a production run (one coload.<pid>.log per process, written by the
                    LD_PRELOAD monitor scripts/cluster_a/coload_shim.c of kpack_load_code_object /
                    hsa_code_object_reader_create_from_memory / hsa_executable_load_agent_code_object), with
                    --run-start SET=<that run's mlperf.log>: every TU
                    loaded after RUN_START (step 1 with its once-per-process checks, eval, ...) becomes a row; this
                    replaces the hand-made "amendments" of the v2 table for a new stack. Repeatable.
  --stack LABEL     software stack recorded in the header (hip_preload.py picks a table by its libraries' build-ids)
  --out PATH        output .jsonl (first line kind=header, then one kind=entry row per bundle)

Method:
  1. Walk .hip_fatbin bundle by bundle (CCOB v2/v3 zstd-compressed or plain __CLANG_OFFLOAD_BUNDLE__), extract the
     gfx950 code object and its kernel descriptor symbols '<mangled>.kd'; demangle with c++filt (the trace names
     kernels as '<demangled> [clone .kd]').
  2. HIP registers a host stub once; the first registration wins and module constructors run in bundle order, so a
     kernel name belongs to the FIRST bundle (of its library) that contains it.
  3. Kernel names launched in the traces -> (library, first bundle). A demangled name present in several libraries
     (rocprim templates, host stubs not exported) is attributed by the launching op (fbgemm::* -> fbgemm libraries,
     else libtorch_hip); what stays ambiguous is kept as a union (an extra preload costs untimed init only).
  4. Per touched bundle pick a kernel whose first bundle is this one and whose host stub is a symbol of the library:
     prefer .dynsym (GLOBAL/WEAK, default/protected visibility; resolved with dlsym at run time, mode "dlsym"),
     else .symtab (local stubs, e.g. anonymous-namespace kernels; mode "symtab": load base + st_value, valid only
     when the loaded library's GNU build-id equals the one recorded in the header; the base comes from a dlsym'd
     anchor symbol and is cross-checked against /proc/self/maps). Bundles without any such symbol (stripped
     libraries) are listed in header.uncovered.
No sample data, no binaries: kernel/symbol names, sizes and build ids only.

S14-d v2 amendment (2026-10-11): the checked-in table also carries rows that NO steady-step trace shows. They are listed
in its header "amendments" (source: rccl iter85 S14D85 section 6; rows file sha256 51b052db...): libtorch_hip b10 and
b144 widened from 1n to every shape, and 7 symtab rows b2, b49, b55, b73, b74, b75, b77, the step-1-only first-use code
objects of once-per-process checks and self-tests (valid for libtorch_hip build-id 9889ac94... only). This script does
not produce them: regenerating from the steady traces alone (steps 42-46 / 52-56) DROPS those 7 rows and narrows b10 /
b144 back to 1n. After any regeneration, re-apply the amendment (append the rows file byte-identically, widen the two
rows, recompute n_entries / n_symtab / entries_by_set, keep the "amendments" record) or include step-1 (first-use)
traces in each shape's --trace glob. The 7 rows hold st_values of that one build: after a libtorch_hip rebuild re-derive
them (header "amendments" records rows_valid_for_build_id). Never re-append them to a table generated for another
build: the run-time build-id gate is per library (header), so stale values would pass it and only fail open as HIP
errors (no registered stub at that address).
"""
import argparse
import collections
import glob
import gzip
import json
import multiprocessing
import os
import re
import shutil
import struct
import subprocess
import sys
import time

GFX = "gfx950"
# First-use cost model (dlstart iter 34, ledger lazyload-bundle-cost-model; c23-18, fresh process, untraced):
#   libtorch_hip (compressed, multi-target): ms = -2.05 + 0.980 * uncompressed MiB
#   fbgemm (uncompressed, gfx950 only):      ms = 3.79 * gfx950 code-object MiB
COST_LIBTORCH = (-2.05, 0.980)
COST_FBGEMM = 3.79
#   TheRock kpack (ROCm 10.2 torch: one zstd frame per TU, gfx950 only): ms = offset + slope * code-object MiB, used
#   only without --loads. Measured (coload_shim log, compat run job 382981): a first-use load inside the timed region
#   costs 20-70 ms whatever its size (median 25 ms; about 4 ms during init). With --loads the header "cost_model" is
#   the measured one (each row's own median first-use load).
COST_KPACK = (25.0, 0.0)


# ----------------------------------------------------------------------------------------------- ELF helpers
def _sections(elf):
    e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    sh = []
    for i in range(e_shnum):
        sh.append(struct.unpack_from("<IIQQQQIIQQ", elf, e_shoff + i * e_shentsize))
    names_off = sh[e_shstrndx][4] if e_shstrndx < len(sh) else 0
    out = []
    for n, typ, fl, addr, off, size, link, info, al, ent in sh:
        e = elf.index(b"\0", names_off + n)
        out.append(
            {
                "name": elf[names_off + n : e].decode(errors="replace"),
                "type": typ,
                "addr": addr,
                "off": off,
                "size": size,
                "link": link,
                "ent": ent,
            }
        )
    return out


def _symbols(elf, secs, sh_type):
    """{name: (st_value, bind, vis, type)} of defined, nonzero symbols of SHT_SYMTAB(2) / SHT_DYNSYM(11)."""
    out = {}
    for s in secs:
        if s["type"] != sh_type or not s["ent"]:
            continue
        stoff = secs[s["link"]]["off"]
        for k in range(s["size"] // s["ent"]):
            st_name, st_info, st_other, st_shndx, st_value, _sz = struct.unpack_from(
                "<IBBHQQ", elf, s["off"] + k * s["ent"]
            )
            if not st_value or not st_shndx:
                continue
            e = elf.index(b"\0", stoff + st_name)
            nm = elf[stoff + st_name : e].decode(errors="replace")
            out.setdefault(nm, (st_value, st_info >> 4, st_other & 3, st_info & 0xF))
    return out


def build_id(elf, secs):
    for s in secs:
        if s["type"] != 7:  # SHT_NOTE
            continue
        q, end = s["off"], s["off"] + s["size"]
        while q + 12 <= end:
            namesz, descsz, ntype = struct.unpack_from("<III", elf, q)
            name = elf[q + 12 : q + 12 + namesz]
            dq = q + 12 + ((namesz + 3) & ~3)
            if ntype == 3 and name.rstrip(b"\0") == b"GNU":
                return elf[dq : dq + descsz].hex()
            q = dq + ((descsz + 3) & ~3)
    return None


def kd_symbols(co):
    if co[:4] != b"\x7fELF":
        return []
    secs = _sections(co)
    names = set()
    for t in (2, 11):
        for nm in _symbols(co, secs, t):
            if nm.endswith(".kd"):
                names.add(nm)
    return sorted(names)


# ----------------------------------------------------------------------------------------------- fat binary walk
_LIBZSTD = None


def _zstd_libzstd(payload, size_hint=None):
    """libzstd through ctypes (the ROCm 10.2 image has libzstd.so.1 but neither the zstandard module nor the CLI)."""
    global _LIBZSTD
    import ctypes
    import ctypes.util

    if _LIBZSTD is None:
        _LIBZSTD = ctypes.CDLL(ctypes.util.find_library("zstd") or "libzstd.so.1")
        _LIBZSTD.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        _LIBZSTD.ZSTD_getFrameContentSize.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        _LIBZSTD.ZSTD_decompress.restype = ctypes.c_size_t
        _LIBZSTD.ZSTD_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        _LIBZSTD.ZSTD_isError.restype = ctypes.c_uint
        _LIBZSTD.ZSTD_isError.argtypes = [ctypes.c_size_t]
    n = _LIBZSTD.ZSTD_getFrameContentSize(payload, len(payload))
    if n >= (1 << 63):  # ZSTD_CONTENTSIZE_UNKNOWN / _ERROR
        if not size_hint:
            raise RuntimeError("zstd frame without a content size")
        n = size_hint
    out = ctypes.create_string_buffer(n)
    got = _LIBZSTD.ZSTD_decompress(out, n, payload, len(payload))
    if _LIBZSTD.ZSTD_isError(got):
        raise RuntimeError(f"ZSTD_decompress failed ({got})")
    return out.raw[:got]


def _zstd_d(payload, size_hint=None):
    try:
        import zstandard  # type: ignore

        return zstandard.ZstdDecompressor().decompress(payload, max_output_size=1 << 34)
    except ImportError:
        pass
    try:
        return _zstd_libzstd(payload, size_hint)
    except OSError:
        p = subprocess.run(["zstd", "-d", "-c", "-q"], input=payload, capture_output=True, check=True)
        return p.stdout


def _msgpack(buf, q=0):
    """(value, next offset): the msgpack subset of TheRock kpack metadata (maps, arrays, str, bin, int, bool, nil)."""
    b = buf[q]
    if b <= 0x7F:
        return b, q + 1
    if b >= 0xE0:
        return b - 0x100, q + 1
    if 0x80 <= b <= 0x8F or b in (0xDE, 0xDF):
        n, q = (
            (b & 0x0F, q + 1)
            if b <= 0x8F
            else (struct.unpack_from(">H" if b == 0xDE else ">I", buf, q + 1)[0], q + (3 if b == 0xDE else 5))
        )
        out = {}
        for _ in range(n):
            k, q = _msgpack(buf, q)
            v, q = _msgpack(buf, q)
            out[k] = v
        return out, q
    if 0x90 <= b <= 0x9F or b in (0xDC, 0xDD):
        n, q = (
            (b & 0x0F, q + 1)
            if b <= 0x9F
            else (struct.unpack_from(">H" if b == 0xDC else ">I", buf, q + 1)[0], q + (3 if b == 0xDC else 5))
        )
        out = []
        for _ in range(n):
            v, q = _msgpack(buf, q)
            out.append(v)
        return out, q
    if 0xA0 <= b <= 0xBF or b in (0xD9, 0xDA, 0xDB, 0xC4, 0xC5, 0xC6):
        if b <= 0xBF:
            n, q = b & 0x1F, q + 1
        else:
            w = {0xD9: 1, 0xDA: 2, 0xDB: 4, 0xC4: 1, 0xC5: 2, 0xC6: 4}[b]
            n, q = int.from_bytes(buf[q + 1 : q + 1 + w], "big"), q + 1 + w
        raw = bytes(buf[q : q + n])
        return (raw if b in (0xC4, 0xC5, 0xC6) else raw.decode()), q + n
    fixed = {0xCC: ">B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q", 0xD0: ">b", 0xD1: ">h", 0xD2: ">i", 0xD3: ">q"}
    if b in fixed:
        f = fixed[b]
        return struct.unpack_from(f, buf, q + 1)[0], q + 1 + struct.calcsize(f)
    if b in (0xC0, 0xC2, 0xC3):
        return {0xC0: None, 0xC2: False, 0xC3: True}[b], q + 1
    raise ValueError(f"msgpack type 0x{b:02x} at {q} not supported")


KPACK_REF = ".rocm_kpack_ref"


def hipk_indices(elf, secs):
    """TU (code object) indices of the HIPK wrappers in .hipFatBinSegment, in section order. A TheRock 'kpack'
    library registers one wrapper {u32 'HIPK', u32 version, u64 metadata pointer, u64 TU index} per translation
    unit (one per classic fat-binary bundle); its device code lives in a separate .kpack archive."""
    seg = [s for s in secs if s["name"] == ".hipFatBinSegment"]
    out = []
    if not seg:
        return out
    s = seg[0]
    for q in range(s["off"], s["off"] + s["size"] - 23, 24):
        magic, _ver, _ptr, idx = struct.unpack_from("<4sIQQ", elf, q)
        if magic == b"HIPK":
            out.append(idx)
    return out


def walk_kpack(lib, elf, secs):
    """[{index, offset, kind, compressed, uncompressed, co_bytes, kernels_mangled}] of a TheRock kpack library (ROCm
    10.2 torch: .hip_fatbin is NOBITS, .rocm_kpack_ref = msgpack {kpack_search_paths, kernel_name}; the archive is
    KPAK v1 = header {magic, version, u64 TOC offset}, msgpack TOC {toc: {'<kernel_name>#<TU>': {<arch>: {ordinal,
    original_size}}}, zstd_offset}, blobs at zstd_offset = u32 count + count x (u32 size, zstd frame), frame i =
    ordinal i). Row i = TU index i (= the HIPK wrapper's index = the bundle index of the classic layout)."""
    ref_sec = [s for s in secs if s["name"] == KPACK_REF][0]
    ref, _ = _msgpack(elf, ref_sec["off"])
    name = ref["kernel_name"]
    path = None
    for sp in ref.get("kpack_search_paths") or []:
        cand = os.path.normpath(
            os.path.join(os.path.dirname(os.path.realpath(lib)), sp.replace("@GFXARCH@", GFX))
        )
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        raise SystemExit(f"{lib}: no kpack archive for {GFX} on {ref.get('kpack_search_paths')}")
    with open(path, "rb") as f:
        kp = f.read()
    magic, ver, toc_off = struct.unpack_from("<4sIQ", kp, 0)
    if magic != b"KPAK" or ver != 1:
        raise SystemExit(f"{path}: not a KPAK v1 archive ({magic!r}, {ver})")
    meta, _ = _msgpack(kp, toc_off)
    if meta.get("compression_scheme") != "zstd-per-kernel":
        raise SystemExit(f"{path}: compression {meta.get('compression_scheme')!r} not supported")
    zo = meta["zstd_offset"]
    n = struct.unpack_from("<I", kp, zo)[0]
    frames, q = [], zo + 4
    for _ in range(n):
        sz = struct.unpack_from("<I", kp, q)[0]
        frames.append((q + 4, sz))
        q += 4 + sz
    rows = []
    for pos, idx in enumerate(hipk_indices(elf, secs)):
        e = (meta["toc"].get(f"{name}#{idx}") or {}).get(GFX)
        r = {"index": pos, "tu": idx, "offset": None, "kind": "kpack", "kpack": os.path.basename(path)}
        if e is None:
            r.update(compressed=0, uncompressed=0, co_bytes=0, kernels_mangled=[])
        else:
            off, sz = frames[e["ordinal"]]
            co = _zstd_d(bytes(kp[off : off + sz]), e["original_size"])
            r.update(
                compressed=sz,
                uncompressed=len(co),
                co_bytes=len(co),
                kernels_mangled=kd_symbols(co),
                co_size=len(co),
                co_mid=_mid_sample(co),
            )
        rows.append(r)
    if [r["tu"] for r in rows] != list(range(len(rows))):
        raise SystemExit(f"{lib}: HIPK wrapper indices are not 0..N-1 in section order")
    return rows


def _mid_sample(co):
    """32 bytes at the middle of a code object (hex): with the size, the key that matches a code object logged by
    the load monitor (tools: coload_shim 'reader' lines carry size and an FNV-1a hash; see match_loads)."""
    h = len(co) // 2
    return co[h : h + 32].hex()


def fnv1a64(b):
    h = 1469598103934665603
    for x in b:
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _parse_offload_bundle(ub):
    assert ub[:24] == b"__CLANG_OFFLOAD_BUNDLE__", ub[:24]
    n = struct.unpack_from("<Q", ub, 24)[0]
    q, out, end = 32, [], 32
    for _ in range(n):
        eoff, esz, tl = struct.unpack_from("<QQQ", ub, q)
        q += 24
        out.append((ub[q : q + tl].decode(errors="replace"), eoff, esz))
        q += tl
        end = max(end, eoff + esz, q)
    return out, end


MAGIC = re.compile(rb"CCOB|__CLANG_OFFLOAD_BUNDLE__")


def walk_fatbin(lib):
    """[{index, offset, kind, compressed, uncompressed, co_bytes, kernels_mangled}] in section order. Sequential walk:
    the next bundle is searched only after the end of the current one (no false magic inside payloads)."""
    with open(lib, "rb") as f:
        elf = f.read()
    secs = _sections(elf)
    if any(s["name"] == KPACK_REF for s in secs):
        return elf, secs, walk_kpack(lib, elf, secs)
    fb = [s for s in secs if s["name"] == ".hip_fatbin" and s["type"] != 8]  # 8 = SHT_NOBITS
    if not fb:
        return elf, secs, []
    sec = elf[fb[0]["off"] : fb[0]["off"] + fb[0]["size"]]
    rows, pos = [], 0
    while True:
        m = MAGIC.search(sec, pos)
        if not m:
            break
        o = m.start()
        r = {"index": len(rows), "offset": o, "vaddr": fb[0]["addr"] + o}
        if sec[o : o + 4] == b"CCOB":
            ver, _meth = struct.unpack_from("<HH", sec, o + 4)
            if ver == 3:
                tot, unc, _h = struct.unpack_from("<QQQ", sec, o + 8)
                hdr = 32
            elif ver == 2:
                tot, unc, _h = struct.unpack_from("<IIQ", sec, o + 8)
                hdr = 24
            else:
                raise SystemExit(f"{lib}: unsupported CCOB version {ver} at {o}")
            ub = _zstd_d(sec[o + hdr : o + tot])
            r.update(kind="ccob", compressed=tot, uncompressed=unc)
            pos = o + tot
        else:
            ents, end = _parse_offload_bundle(sec[o:])
            ub = sec[o : o + end]
            r.update(kind="plain", compressed=None, uncompressed=end)
            pos = o + end
        ents, _ = _parse_offload_bundle(ub)
        co = [(eo, es) for t, eo, es in ents if GFX in t]
        r["co_bytes"] = co[0][1] if co else 0
        r["kernels_mangled"] = kd_symbols(ub[co[0][0] : co[0][0] + co[0][1]]) if co else []
        r["vaddr_end"] = fb[0]["addr"] + pos
        if co:
            r["co_size"], r["co_mid"] = co[0][1], _mid_sample(ub[co[0][0] : co[0][0] + co[0][1]])
        rows.append(r)
    return elf, secs, rows


def demangle(names):
    if not names:
        return []
    p = subprocess.run(["c++filt"], input="\n".join(names) + "\n", capture_output=True, text=True, check=True)
    out = p.stdout.splitlines()
    assert len(out) == len(names), (len(out), len(names))
    return out


# ----------------------------------------------------------------------------------------------- traces
def _trace_kernels(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    ev = d["traceEvents"] if isinstance(d, dict) else d
    cpu = {}
    for e in ev:
        if e.get("cat") == "cpu_op":
            x = (e.get("args") or {}).get("External id")
            if x is not None:
                cpu[x] = e.get("name", "")
    out = {}
    for e in ev:
        if e.get("cat") == "kernel":
            out.setdefault(e["name"], set()).add(cpu.get((e.get("args") or {}).get("External id"), ""))
    return path, {k: sorted(v) for k, v in out.items()}


def candidates(name, ctxs, libs):
    """(lib, first bundle) pairs a launched kernel name can load. A demangled name found in several libraries (rocprim
    templates; their host stubs are not exported, so no cross-library interposition) is attributed by the launching
    op: fbgemm::* -> the fbgemm libraries, any other op -> the non-fbgemm libraries (libtorch_hip); no usable context
    -> every candidate. Ambiguity left inside one group is kept as a union (an extra preload is untimed init).
    """
    cands = [(lib, L["name_first"][name]) for lib, L in libs.items() if name in L["name_first"]]
    if len(cands) <= 1:
        return cands
    out = set()
    for ctx in ctxs or [""]:
        if ctx.startswith("fbgemm::"):
            sel = [c for c in cands if c[0].startswith("fbgemm")]
        elif ctx:
            sel = [c for c in cands if not c[0].startswith("fbgemm")]
        else:
            sel = []
        out.update(sel or cands)
    return sorted(out)


# ----------------------------------------------------------------------------------------------- load logs
def mlperf_run_start_s(path):
    """Wall time (s) of the run_start event of an MLPerf log."""
    with open(path) as f:
        for line in f:
            if ":::MLLOG " in line:
                d = json.loads(line.split(":::MLLOG ", 1)[1])
                if d.get("key") == "run_start":
                    return d["time_ms"] / 1e3
    raise SystemExit(f"{path}: no run_start")


_LOAD_RE = re.compile(
    r"^(?P<t>\d+\.\d+) (?P<kind>kpack|reader|load|freeze) dur_ms=(?P<ms>[-\d.]+) rc=(?P<rc>-?\d+) "
    r"tid=(?P<tid>\d+)(?P<rest>.*)$"
)


def parse_load_logs(paths):
    """Events of code-object load logs written by an LD_PRELOAD load monitor (one file per process,
    'coload.<pid>.log'; lines '<wall s> kpack|reader|load|freeze dur_ms=<ms> rc=<rc> tid=<tid> key=value...').
    kpack: TheRock kpack_load_code_object (bin=<library path> idx=<TU>); reader:
    hsa_code_object_reader_create_from_memory (size=, lib=<dladdr library or ->, off=<address - load base>);
    load: hsa_executable_load_agent_code_object. Returns [{pid, t, kind, ms, rc, tid, **fields}] in file order.
    """
    out = []
    for p in paths:
        m = re.search(r"coload\.(\d+)\.log$", p)
        pid = int(m.group(1)) if m else -1
        with open(p, errors="replace") as f:
            for line in f:
                mm = _LOAD_RE.match(line.rstrip("\n"))
                if not mm:
                    continue
                ev = {
                    "pid": pid,
                    "t": float(mm["t"]),
                    "kind": mm["kind"],
                    "ms": float(mm["ms"]),
                    "rc": int(mm["rc"]),
                    "tid": int(mm["tid"]),
                }
                for kv in mm["rest"].split():
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        ev[k] = v
                out.append(ev)
    return out


def match_loads(events, libs, t_run_start):
    """Bundles loaded after RUN_START: {(lib, bundle): [(seconds after RUN_START, load ms), ...]} and the unmatched
    loads after RUN_START (Triton / FlyDSL modules, hipBLASLt code objects, ...: not statically registered, so not
    this table's business; returned for the report). A kpack line names (library, TU) directly; its code object
    then reaches the HSA reader from the heap (the next reader line of that thread: not counted twice). A classic
    bundle's code object is read in place (dladdr library + address inside the bundle) or, when compressed, from
    the heap after decompression (matched by its size; ambiguous sizes keep every candidate)."""
    by_vaddr = {}
    by_size = collections.defaultdict(list)
    for lib, L in libs.items():
        for r in L["rows"]:
            if r.get("vaddr") is not None:
                by_vaddr.setdefault(lib, []).append((r["vaddr"], r["vaddr_end"], r["index"]))
            if r.get("kind") != "kpack" and r.get("co_size"):
                by_size[r["co_size"]].append((lib, r["index"]))
    hits = collections.defaultdict(list)
    unmatched = []
    pending = {}  # (pid, tid) -> (lib, bundle, t, size) of the last kpack line
    open_hits = {}  # (pid, tid) -> hit records the next 'load' line of the thread completes (HSA load time)
    for ev in events:
        key_t = (ev["pid"], ev["tid"])
        after = ev["t"] - t_run_start
        if ev["kind"] == "load":
            for h in open_hits.pop(key_t, []):
                if isinstance(h, dict):
                    h["ms"] = round(h["ms"] + ev["ms"], 3)
                else:
                    h[1] = round(h[1] + ev["ms"], 3)
            continue
        if ev["kind"] == "kpack":
            lib = os.path.basename(ev.get("bin", ""))
            idx = int(ev.get("idx", -1))
            if lib in libs and 0 <= idx < len(libs[lib]["rows"]):
                if after >= 0:
                    hits[(lib, idx)].append([round(after, 3), ev["ms"]])
                    open_hits[key_t] = [hits[(lib, idx)][-1]]
                pending[key_t] = (lib, idx, after, int(ev.get("size", 0)))
            continue
        if ev["kind"] != "reader":
            continue
        size = int(ev.get("size", 0))
        pk = pending.pop(key_t, None)
        if pk is not None and pk[3] == size:  # the kpack code object reaching the HSA reader: same load
            if pk[2] >= 0:
                hits[(pk[0], pk[1])][-1][1] = round(hits[(pk[0], pk[1])][-1][1] + ev["ms"], 3)
            continue
        open_hits.pop(key_t, None)
        if after < 0:
            continue
        lib = os.path.basename(ev.get("lib", "-"))
        cands = []
        if lib in by_vaddr:
            off = int(ev.get("off", "0"), 16)
            cands = [(lib, i) for lo, hi, i in by_vaddr[lib] if lo <= off < hi]
        elif lib == "-":
            cands = by_size.get(size, [])
        if not cands:
            unmatched.append(
                {
                    "pid": ev["pid"],
                    "after_s": round(after, 3),
                    "size": size,
                    "lib": lib,
                    "ms": ev["ms"],
                    "kd": ev.get("kd", "-"),
                }
            )
            open_hits[key_t] = [unmatched[-1]]
            continue
        open_hits[key_t] = []
        for c in cands:
            hits[c].append([round(after, 3), ev["ms"]])
            open_hits[key_t].append(hits[c][-1])
    return hits, unmatched


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True)
    ap.add_argument("--trace", action="append", default=[], help="SET=GLOB")
    ap.add_argument(
        "--loads",
        action="append",
        default=[],
        help="SET=GLOB of code-object load logs (coload.<pid>.log); needs --run-start SET=<mlperf.log>",
    )
    ap.add_argument("--run-start", action="append", default=[], help="SET=<mlperf.log> of the --loads run")
    ap.add_argument(
        "--stack", default=None, help="stack label recorded in the header (e.g. rocm10.2-torch2.16)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--summary", default=None, help="optional json with the per-set bundle census")
    a = ap.parse_args(argv)
    if not a.trace and not a.loads:
        ap.error("need at least one --trace or --loads")
    t0 = time.time()

    # libraries
    libs = {}
    for lib in a.lib:
        base = os.path.basename(lib)
        elf, secs, rows = walk_fatbin(lib)
        dyn = _symbols(elf, secs, 11)
        tab = _symbols(elf, secs, 2)
        flat = [m for r in rows for m in r["kernels_mangled"]]
        dem = dict(zip(sorted(set(flat)), demangle(sorted(set(flat))))) if flat else {}
        first = {}
        for r in rows:
            for m in r["kernels_mangled"]:
                first.setdefault(m, r["index"])
        name_first = {}  # demangled '... [clone .kd]' -> first bundle
        for r in rows:
            for m in r["kernels_mangled"]:
                name_first.setdefault(dem[m], r["index"])
        # anchor: a defined GLOBAL FUNC in .dynsym (for the symtab-mode base); deterministic choice
        anchor = None
        for nm in sorted(dyn):
            v, bind, vis, typ = dyn[nm]
            if bind == 1 and vis in (0, 3) and typ == 2:
                anchor = {"sym": nm, "value": v}
                break
        libs[base] = {
            "path": lib,
            "rows": rows,
            "dyn": dyn,
            "tab": tab,
            "first": first,
            "name_first": name_first,
            "dem": dem,
            "build_id": build_id(elf, secs),
            "size": len(elf),
            "anchor": anchor,
            "has_symtab": any(s["type"] == 2 for s in secs),
        }
        print(
            f"{base}: {len(rows)} bundles, {sum(r['co_bytes'] for r in rows) / 2**20:.1f} MiB {GFX} code, "
            f"{len(flat)} kernels, dynsym {len(dyn)}, symtab {len(tab)}, build-id {libs[base]['build_id']}",
            flush=True,
        )
        del elf

    # traces
    sets = collections.OrderedDict()
    files = []
    for spec in a.trace:
        label, pat = spec.split("=", 1)
        fs = sorted(glob.glob(pat))
        if not fs:
            raise SystemExit(f"no trace matches {pat}")
        sets[label] = fs
        files += fs
    per_file = {}
    if files:
        with multiprocessing.Pool(a.jobs) as pool:
            per_file = dict(pool.map(_trace_kernels, sorted(set(files))))
    touched = collections.defaultdict(lambda: {"sets": set(), "names": set(), "loads": []})
    unmapped = collections.Counter()
    for label, fs in sets.items():
        for f in fs:
            for name, ctxs in per_file[f].items():
                cands = candidates(name, ctxs, libs)
                if not cands:
                    unmapped[name] += 1
                    continue
                for key in cands:
                    touched[key]["sets"].add(label)
                    touched[key]["names"].add(name)

    # code-object load logs: every bundle a process loaded after RUN_START (first use inside the timed region:
    # step 1 with its once-per-process checks, the eval, anything a steady-step trace cannot show)
    run_start = dict(spec.split("=", 1) for spec in a.run_start)
    load_sets = collections.OrderedDict()
    unmatched_loads = collections.OrderedDict()
    for spec in a.loads:
        label, pat = spec.split("=", 1)
        fs = sorted(glob.glob(pat))
        if not fs:
            raise SystemExit(f"no load log matches {pat}")
        if label not in run_start:
            raise SystemExit(f"--loads {label}=... needs --run-start {label}=<mlperf.log>")
        t_rs = mlperf_run_start_s(run_start[label])
        hits, unmatched = match_loads(parse_load_logs(fs), libs, t_rs)
        load_sets[label] = {
            "files": len(fs),
            "glob_example": fs[0],
            "run_start_ms": int(round(t_rs * 1e3)),
            "bundles": len(hits),
            "unmatched_after_run_start": len(unmatched),
        }
        unmatched_loads[label] = unmatched
        for key, evs in hits.items():
            L = libs[key[0]]
            ks = [
                L["dem"].get(m, m)
                for m in L["rows"][key[1]]["kernels_mangled"]
                if L["first"].get(m) == key[1]
            ]
            touched[key]["sets"].add(label)
            touched[key]["names"].add(
                "first use after RUN_START (load log): " + (sorted(ks)[0] if ks else "?")
            )
            touched[key]["loads"] += evs
        sets.setdefault(label, [])  # a load-log set; keep the trace files of a --trace set of the same label

    load_ms_all = sorted(x[1] for t in touched.values() for x in t["loads"])
    load_ms_med = load_ms_all[len(load_ms_all) // 2] if load_ms_all else None
    entries, uncovered = [], []
    for (lib, b), t in sorted(touched.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        L = libs[lib]
        r = L["rows"][b]
        own = [m[:-3] if m.endswith(".kd") else m for m in r["kernels_mangled"] if L["first"].get(m) == b]
        pick = None
        for m in own:
            if m in L["dyn"] and L["dyn"][m][1] in (1, 2) and L["dyn"][m][2] in (0, 3):
                pick = {"sym": m, "mode": "dlsym"}
                break
        if pick is None and L["anchor"] is not None:
            for m in own:
                if m in L["tab"]:
                    pick = {"sym": m, "mode": "symtab", "value": L["tab"][m][0]}
                    break
        unc, co = r["uncompressed"] or 0, r["co_bytes"] or 0
        if load_ms_all:  # measured stack: the row's own median first-use load, else the median of every load
            mine = sorted(x[1] for x in t["loads"])
            est = mine[len(mine) // 2] if mine else load_ms_med
        elif r.get("kind") == "kpack":
            est = COST_KPACK[0] + COST_KPACK[1] * co / 2**20
        elif lib.startswith("libtorch"):
            est = max(0.0, COST_LIBTORCH[0] + COST_LIBTORCH[1] * unc / 2**20)
        else:
            est = COST_FBGEMM * co / 2**20
        meta = {
            "lib": lib,
            "bundle": b,
            "unc_mib": round(unc / 2**20, 3),
            "co_mib": round(co / 2**20, 3),
            "est_ms": round(est, 1),
            "sets": sorted(t["sets"]),
            "example": sorted(t["names"])[0][:160],
        }
        if t[
            "loads"
        ]:  # measured first-use loads after RUN_START: (seconds after RUN_START, load ms), over processes
            meta["first_use_s"] = round(min(x[0] for x in t["loads"]), 3)
            meta["load_ms_max"] = round(max(x[1] for x in t["loads"]), 2)
            meta["load_ms_med"] = round(sorted(x[1] for x in t["loads"])[len(t["loads"]) // 2], 2)
            meta["load_procs"] = len(t["loads"])
        if pick is None:
            uncovered.append(
                meta | {"reason": "no host symbol (stripped library or no kernel registered first here)"}
            )
            continue
        entries.append({"kind": "entry"} | pick | meta)

    def census(rows):
        by = collections.OrderedDict()
        for lab in sets:
            rs = [x for x in rows if lab in x["sets"]]
            by[lab] = {"bundles": len(rs), "est_s": round(sum(x["est_ms"] for x in rs) / 1e3, 3)}
        return by

    header = {
        "kind": "header",
        "format": 1,
        "stack": a.stack,
        "generator": "tools/recommendation/dlrm_v4/gen_hip_preload_table.py",
        "note": (
            "one host kernel stub per statically registered HIP fat-binary bundle (translation unit) launched by "
            "the listed production traces or first loaded after RUN_START in the listed load logs (a TheRock "
            "kpack library: one TU = one HIPK wrapper = one code object of its .kpack archive); "
            "kernel/symbol names, sizes and build ids only, no data, no binaries. "
            "hipFuncGetAttributes on the stub = deferred code-object load of that bundle on the current device "
            "(dlrm_v4/train/hip_preload.py). Regenerate after any torch/fbgemm/ROCm rebuild or trainer op change."
        ),
        "arch": GFX,
        "libs": {
            lib: {
                "build_id": L["build_id"],
                "size": L["size"],
                "bundles": len(L["rows"]),
                "anchor": L["anchor"],
                "path_at_generation": L["path"],
            }
            | (
                {"kpack": L["rows"][0].get("kpack")}
                if L["rows"] and L["rows"][0].get("kind") == "kpack"
                else {}
            )
            for lib, L in libs.items()
        },
        "traces": {
            lab: (
                {"files": len(fs), "glob_example": fs[0] if fs else None}
                | ({"loads": load_sets[lab]} if lab in load_sets else {})
            )
            for lab, fs in sets.items()
        },
        "cost_model": (
            {
                "measured": "est_ms = the row's median first-use load after RUN_START over processes "
                "(kpack/reader + HSA load, load logs), else the median of every such load",
                "median_first_use_load_ms": load_ms_med,
                "loads": len(load_ms_all),
            }
            if load_ms_all
            else {
                "libtorch_ms_per_unc_mib": COST_LIBTORCH[1],
                "libtorch_ms_offset": COST_LIBTORCH[0],
                "fbgemm_ms_per_co_mib": COST_FBGEMM,
                "kpack_ms_offset": COST_KPACK[0],
                "kpack_ms_per_co_mib": COST_KPACK[1],
            }
        ),
        "n_entries": len(entries),
        "n_dlsym": sum(1 for e in entries if e["mode"] == "dlsym"),
        "n_symtab": sum(1 for e in entries if e["mode"] == "symtab"),
        "entries_by_set": census(entries),
        "uncovered": uncovered,
        "uncovered_by_set": census(uncovered),
        "unmapped_kernel_names": len(unmapped),
    }
    with open(a.out + ".tmp", "w") as f:
        f.write(json.dumps(header, sort_keys=True) + "\n")
        for e in entries:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    shutil.move(a.out + ".tmp", a.out)
    if a.summary:
        json.dump(
            {"header": header, "unmapped": unmapped.most_common(), "unmatched_loads": unmatched_loads},
            open(a.summary, "w"),
            indent=1,
        )
    print(
        f"wrote {a.out}: {len(entries)} entries ({header['n_dlsym']} dlsym, {header['n_symtab']} symtab), "
        f"uncovered {len(uncovered)}, by set {json.dumps(header['entries_by_set'])}, "
        f"unmapped names {len(unmapped)}, {time.time() - t0:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
