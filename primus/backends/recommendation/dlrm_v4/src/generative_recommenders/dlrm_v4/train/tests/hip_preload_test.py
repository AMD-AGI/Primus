#!/usr/bin/env python3
# pyre-strict
"""S14-d HIP code-object preload (dlrm_v4/train/hip_preload.py): code default off = no-op, the checked-in table parses
and is consistent, every failure is counted and skipped (fail-open), symtab entries need a verified build-id, and the
train_ranker hook sits after the allocator pre-grow and before RUN_START. S14-d v2 (yambda_5b.gin binds the knob on):
the v2 table's per-shape counts, its widened b10/b144 rows and its 7 amendment rows, a dry run of every row through
preload_entries / preload_from_env with fakes, and the arming lines the gin item documents.

CPU-only: the HIP call is replaced by fakes; dlsym/maps/build-id checks run against this process's libc."""

import collections
import ctypes
import hashlib
import importlib.util
import json
import logging
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from generative_recommenders.dlrm_v4.train import hip_preload as hp

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_SHAPES = ("1n", "2n", "4n")
# S14-d v2 (rccl iter85 S14D85 section 6): rows appended byte-identically from proposed_rows.jsonl (this sha256) and
# libtorch_hip bundles widened from 1n to every shape.
_V2_ROWS_SHA256 = "51b052db4c6fd655d76ab8195069abaec881ec298f8a5f78540da0f41543abff"
_V2_ADDED = (2, 49, 55, 73, 74, 75, 77)
_V2_WIDENED = (10, 144)
_V2_PER_SHAPE = {"1n": (46, 2), "2n": (48, 0), "4n": (48, 0)}  # (selected, other_shape)
_TORCH = "libtorch_hip.so"


def _libc() -> Dict[str, Any]:
    maps = hp.mapped_libraries()
    paths = [p for p in maps if os.path.basename(p).startswith("libc.so")]
    if not paths:
        raise unittest.SkipTest("libc not found in /proc/self/maps")
    path = paths[0]
    base = maps[path]["base"]
    malloc = ctypes.cast(ctypes.CDLL(path, mode=hp.RTLD_NOLOAD | hp.RTLD_LAZY)["malloc"], ctypes.c_void_p).value
    return {"maps": maps, "path": path, "name": os.path.basename(path), "base": base, "malloc": malloc}


class _Fake:
    def __init__(self, rc: int = 0, raise_on: int = -1) -> None:
        self.rc = rc
        self.raise_on = raise_on
        self.calls: List[int] = []
        self.cleared = 0

    def fga(self, addr: int) -> int:
        self.calls.append(addr)
        if len(self.calls) - 1 == self.raise_on:
            raise RuntimeError("boom")
        return self.rc

    def clear(self) -> int:
        self.cleared += 1
        return 0


class ResolveAndDisabledTest(unittest.TestCase):
    def test_resolve_table(self) -> None:
        for v in (None, "", "0", "off", "OFF", "false", "no", "none", " 0 "):
            self.assertEqual(hp.resolve_table(v), (None, False), v)
        self.assertEqual(hp.resolve_table("1"), (hp.DEFAULT_TABLE, False))
        self.assertEqual(hp.resolve_table("on"), (hp.DEFAULT_TABLE, False))
        self.assertEqual(hp.resolve_table("all"), (hp.DEFAULT_TABLE, True))
        self.assertEqual(hp.resolve_table("/x/t.jsonl"), ("/x/t.jsonl", True))

    def test_disabled_is_noop(self) -> None:
        called: List[str] = []
        orig_maps, orig_hip = hp.mapped_libraries, hp._Hip
        hp.mapped_libraries = lambda: called.append("maps") or {}  # type: ignore[assignment]
        hp._Hip = lambda *a: called.append("hip")  # type: ignore[assignment,misc]
        try:
            with self.assertNoLogs(hp.logger, level=logging.DEBUG):
                for env in ({}, {hp.ENV: "0"}, {hp.ENV: ""}, {hp.ENV: "off"}):
                    self.assertIsNone(hp.preload_from_env(rank=0, world_size=8, local_world_size=8, env=env))
        finally:
            hp.mapped_libraries, hp._Hip = orig_maps, orig_hip
        self.assertEqual(called, [])

    def test_missing_table_never_raises(self) -> None:
        with self.assertLogs(hp.logger, level=logging.WARNING) as cm:
            res = hp.preload_from_env(rank=0, env={hp.ENV: "/nonexistent/hip_preload_table.jsonl"})
        self.assertEqual(res and res["armed"], False)
        self.assertIn(hp.NOT_ARMED, "\n".join(cm.output))


class TableTest(unittest.TestCase):
    def test_checked_in_table(self) -> None:
        header, entries, bad = hp.load_table(hp.DEFAULT_TABLE)
        self.assertEqual(bad, 0)
        self.assertEqual(header.get("kind"), "header")
        self.assertEqual(header.get("n_entries"), len(entries))
        self.assertGreaterEqual(len(entries), 30)
        libs = header["libs"]
        shapes = set(header["traces"])
        self.assertTrue({"1n", "2n", "4n"} <= shapes)
        seen = set()
        for e in entries:
            key = (e["lib"], e["bundle"])
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertIn(e["lib"], libs)
            self.assertTrue(e["sets"] and set(e["sets"]) <= shapes, e)
            if e["mode"] == "symtab":
                self.assertIsInstance(e["value"], int)
                self.assertTrue(libs[e["lib"]]["build_id"])
                self.assertIsInstance(libs[e["lib"]]["anchor"]["value"], int)
        names = {e["lib"] for e in entries}
        self.assertIn("libtorch_hip.so", names)
        self.assertTrue(any(n.startswith("fbgemm_gpu") for n in names))
        for shape in ("1n", "2n", "4n"):
            sel, other = hp.select_entries(header, entries, shape, False)
            self.assertEqual(len(sel) + other, len(entries))
            self.assertGreaterEqual(len(sel), 30)

    def test_parse_counts_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.jsonl")
            with open(p, "w") as f:
                f.write(json.dumps({"kind": "header", "traces": {"1n": {}}, "libs": {}}) + "\n")
                f.write("not json\n[1, 2]\n\n")
                f.write(json.dumps({"kind": "entry", "lib": "a.so", "sym": "s", "mode": "dlsym", "sets": ["1n"]}) + "\n")
                f.write(json.dumps({"kind": "entry", "lib": "a.so", "sym": "t", "mode": "symtab"}) + "\n")  # no value
                f.write(json.dumps({"kind": "entry", "lib": "a.so", "sym": "u", "mode": "weird"}) + "\n")
            header, entries, bad = hp.load_table(p)
        self.assertEqual(header["kind"], "header")
        self.assertEqual([e["sym"] for e in entries], ["s"])
        self.assertEqual(bad, 4)

    def test_select_entries(self) -> None:
        header = {"traces": {"1n": {}, "2n": {}}}
        entries = [{"lib": "a", "sets": ["1n"]}, {"lib": "b", "sets": ["1n", "2n"]}, {"lib": "c", "sets": ["2n"]}]
        self.assertEqual([e["lib"] for e in hp.select_entries(header, entries, "1n", False)[0]], ["a", "b"])
        self.assertEqual(hp.select_entries(header, entries, "2n", False)[1], 1)
        self.assertEqual(len(hp.select_entries(header, entries, "8n", False)[0]), 3)  # unknown shape -> all
        self.assertEqual(len(hp.select_entries(header, entries, "1n", True)[0]), 3)


def _census(rows: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Dict[str, Any]]:
    """The per-set census rule of tools/recommendation/dlrm_v4/gen_hip_preload_table.py (header entries_by_set)."""
    out: Dict[str, Dict[str, Any]] = collections.OrderedDict()
    for lab in labels:
        rs = [x for x in rows if lab in x["sets"]]
        out[lab] = {"bundles": len(rs), "est_s": round(sum(x["est_ms"] for x in rs) / 1e3, 3)}
    return out


class TableV2Test(unittest.TestCase):
    """The S14-d v2 table: v1 rows + header.amendments (b10/b144 widened, 7 added symtab rows)."""

    def setUp(self) -> None:
        self.header, self.entries, self.bad = hp.load_table(hp.DEFAULT_TABLE)
        with open(hp.DEFAULT_TABLE) as f:
            self.lines: List[str] = f.read().splitlines()

    def test_per_shape_counts_and_census(self) -> None:
        h = self.header
        self.assertEqual((self.bad, len(self.entries)), (0, 48))
        self.assertEqual((h["n_entries"], h["n_dlsym"], h["n_symtab"]), (48, 26, 22))
        self.assertEqual(sum(1 for e in self.entries if e["mode"] == "symtab"), 22)
        for shape, (n, other) in _V2_PER_SHAPE.items():
            sel, o = hp.select_entries(h, self.entries, shape, False)
            self.assertEqual((len(sel), o), (n, other), shape)
            self.assertEqual(h["entries_by_set"][shape]["bundles"], n, shape)
        self.assertEqual(h["entries_by_set"], _census(self.entries, list(h["traces"])))
        self.assertEqual(len(hp.select_entries(h, self.entries, "8n", False)[0]), 48)  # unknown shape -> every row
        self.assertEqual(len(hp.select_entries(h, self.entries, "1n", True)[0]), 48)  # "all"

    def test_widened_and_added_rows(self) -> None:
        torch_rows = {e["bundle"]: e for e in self.entries if e["lib"] == _TORCH}
        for b in _V2_WIDENED + _V2_ADDED:
            e = torch_rows[b]
            self.assertEqual(e["sets"], list(_SHAPES), b)
            self.assertEqual(e["mode"], "symtab", b)
            self.assertTrue(isinstance(e["value"], int) and not isinstance(e["value"], bool) and e["value"] > 0, b)
        # appended byte-identically from the rows file, in its order, after every v1 row
        tail = self.lines[-len(_V2_ADDED):]
        self.assertEqual([json.loads(x)["bundle"] for x in tail], list(_V2_ADDED))
        self.assertEqual(hashlib.sha256(("\n".join(tail) + "\n").encode()).hexdigest(), _V2_ROWS_SHA256)
        # the only rows left out at 1n are the two 2n/4n-only v1 rows
        self.assertEqual(sorted((e["lib"], e["bundle"]) for e in self.entries if "1n" not in e["sets"]),
                         [("fbgemm_gpu_py.so", 37), (_TORCH, 98)])

    def test_amendment_record_and_build_id_gate(self) -> None:
        h = self.header
        meta = h["libs"][_TORCH]
        self.assertRegex(meta["build_id"], r"^[0-9a-f]{40}$")
        self.assertTrue(meta["anchor"]["sym"])
        self.assertIsInstance(meta["anchor"]["value"], int)
        (am,) = h["amendments"]
        self.assertEqual(am["rows_sha256"], _V2_ROWS_SHA256)
        self.assertEqual(am["rows_valid_for_build_id"], meta["build_id"])
        self.assertEqual([a["bundle"] for a in am["added"]], list(_V2_ADDED))
        self.assertEqual([w["bundle"] for w in am["widened"]], list(_V2_WIDENED))
        for shape in _SHAPES:
            new = [e for e in self.entries if e["lib"] == _TORCH and (
                e["bundle"] in _V2_ADDED or (e["bundle"] in _V2_WIDENED and shape != "1n"))]
            before = am["entries_by_set_before"][shape]["bundles"]
            self.assertEqual(h["entries_by_set"][shape]["bundles"] - before, len(new), shape)
            self.assertAlmostEqual(am["est_ms_added_by_set"][shape], sum(e["est_ms"] for e in new), places=6)


class _FakeHandle:
    """dlsym stand-in: one distinct fake address per symbol."""

    def __init__(self, base: int) -> None:
        self.base = base
        self.addr: Dict[str, int] = {}

    def __getitem__(self, sym: str) -> ctypes.c_void_p:
        return ctypes.c_void_p(self.addr.setdefault(sym, self.base + 16 * (len(self.addr) + 1)))


class _Opener:
    """open_library stand-in: every table library is 'loaded'; libtorch_hip's symtab base verified unless a reason is
    given (what open_library returns for a build-id / base / anchor mismatch)."""

    TORCH_BASE = 0x7F0000000000

    def __init__(self, torch_reason: str = "") -> None:
        self.torch_reason = torch_reason
        self.handles: Dict[str, _FakeHandle] = {}

    def __call__(self, name: str, meta: Dict[str, Any], maps: Dict[str, Dict[str, int]]) -> Any:
        h = self.handles.setdefault(name, _FakeHandle(0x100000000 * (len(self.handles) + 1)))
        if name == _TORCH and self.torch_reason:
            return hp._Lib(h, "/fake/" + name, None, self.torch_reason)  # type: ignore[arg-type]
        return hp._Lib(h, "/fake/" + name, self.TORCH_BASE if name == _TORCH else None, "")  # type: ignore[arg-type]


class _FakeHip:
    path = "/fake/" + _TORCH

    def __init__(self, maps: Dict[str, Dict[str, int]]) -> None:
        self.calls: List[int] = []

    def device(self) -> int:
        return 0

    def func_attributes(self, addr: int) -> int:
        self.calls.append(addr)
        return 0

    def clear_error(self) -> int:
        return 0


class DryRunV2Test(unittest.TestCase):
    """Every v2 row through the real preload_entries / preload_from_env with fake libraries and a fake HIP (CPU)."""

    def setUp(self) -> None:
        self.header, self.entries, _ = hp.load_table(hp.DEFAULT_TABLE)

    def test_every_row_resolves(self) -> None:
        for shape, (n, _other) in _V2_PER_SHAPE.items():
            sel, _ = hp.select_entries(self.header, self.entries, shape, False)
            f = _Fake()
            r = hp.preload_entries(sel, self.header["libs"], f.fga, f.clear, maps={}, opener=_Opener())
            c = r["counts"]
            self.assertEqual(c["ok"], n, shape)
            self.assertEqual(c["nolib"] + c["nosym"] + c["symtab_unverified"] + c["hip"] + c["error"], 0, (shape, c))
            self.assertEqual((len(f.calls), len(set(f.calls))), (n, n), shape)  # one distinct stub per bundle
            symtab = {_Opener.TORCH_BASE + e["value"] for e in sel if e["mode"] == "symtab"}
            self.assertEqual(len(symtab), 22, shape)
            self.assertTrue(symtab <= set(f.calls), shape)  # load base + st_value, as the table says

    def test_build_id_or_base_mismatch_skips_only_symtab_rows(self) -> None:
        sel, _ = hp.select_entries(self.header, self.entries, "2n", False)
        for reason in ("buildid", "base", "anchor"):
            f = _Fake()
            r = hp.preload_entries(sel, self.header["libs"], f.fga, f.clear, maps={},
                                   opener=_Opener(torch_reason=reason))
            c = r["counts"]
            self.assertEqual((c["symtab_unverified"], c["ok"]), (22, 26), reason)
            self.assertEqual(r["libs"][_TORCH]["reason"], reason)
            self.assertFalse({_Opener.TORCH_BASE + e["value"] for e in sel if e["mode"] == "symtab"} & set(f.calls))

    def _from_env(self, value: str, world: int, rank: int = 0, torch_reason: str = "") -> Any:
        real = hp.preload_entries
        op = _Opener(torch_reason)

        def entries_with_fake_libs(sel: List[Dict[str, Any]], libs: Dict[str, Any], fga: Any, clr: Any,
                                   maps: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, Any]:
            return real(sel, libs, fga, clr, maps, opener=op)

        with mock.patch.object(hp, "mapped_libraries", return_value={}), mock.patch.object(hp, "_Hip", _FakeHip), \
                mock.patch.object(hp, "preload_entries", entries_with_fake_libs):
            with self.assertLogs(hp.logger, level=logging.INFO) as cm:
                res = hp.preload_from_env(rank=rank, world_size=world, local_world_size=8, env={hp.ENV: value})
        return res, "\n".join(cm.output)

    def test_arming_lines_per_shape(self) -> None:
        for world, shape in ((8, "1n"), (16, "2n"), (32, "4n")):
            n, other = _V2_PER_SHAPE[shape]
            res, log = self._from_env("1", world)
            self.assertTrue(res and res["armed"], shape)
            self.assertIn(f"{hp.ARMED} table=hip_preload_table.jsonl shape={shape} entries={n} (other_shape={other} "
                          f"bad_lines=0) ok={n} ", log)
            self.assertIn(" failed=0 (nolib=0 nosym=0 symtab_unverified=0 hip=0 error=0) in ", log)
            self.assertNotIn(hp.NOT_ARMED, log)
            self.assertNotIn("[hip-preload] rank ", log)
        _res, log = self._from_env("all", 8)  # env-only value: every row at any shape
        self.assertIn("shape=1n entries=48 (other_shape=0 bad_lines=0) ok=48 ", log)

    def test_failure_marker_of_other_ranks(self) -> None:
        res, log = self._from_env("1", 16, rank=3, torch_reason="buildid")
        self.assertEqual(res and res["failed"], 22)
        self.assertIn("[hip-preload] rank 3: ", log)
        self.assertIn("symtab_unverified=22", log)
        self.assertNotIn(hp.ARMED, log)  # the arming line is rank 0's


class FailOpenTest(unittest.TestCase):
    def test_dlsym_missing_lib_missing_symbol_hip_error(self) -> None:
        c = _libc()
        entries = [
            {"lib": "libdoesnotexist_hip_preload.so", "sym": "x", "mode": "dlsym", "bundle": 0},
            {"lib": c["name"], "sym": "malloc", "mode": "dlsym", "bundle": 1},
            {"lib": c["name"], "sym": "no_such_symbol_hip_preload_xyz", "mode": "dlsym", "bundle": 2},
            {"lib": c["name"], "sym": "free", "mode": "dlsym", "bundle": 3},
        ]
        ok = _Fake(rc=0)
        r = hp.preload_entries(entries, {}, ok.fga, ok.clear, maps=c["maps"])
        self.assertEqual(r["counts"]["nolib"], 1)
        self.assertEqual(r["counts"]["nosym"], 1)
        self.assertEqual(r["counts"]["ok"], 2)
        self.assertEqual(ok.calls[0], c["malloc"])
        self.assertEqual(ok.cleared, 0)
        bad = _Fake(rc=98)  # hipErrorInvalidDeviceFunction
        r = hp.preload_entries(entries, {}, bad.fga, bad.clear, maps=c["maps"])
        self.assertEqual((r["counts"]["hip"], r["counts"]["ok"]), (2, 0))
        self.assertEqual(bad.cleared, 2)  # every HIP error is cleared (no sticky last error for torch)

    def test_exception_is_counted_and_loop_continues(self) -> None:
        c = _libc()
        entries = [{"lib": c["name"], "sym": s, "mode": "dlsym", "bundle": i} for i, s in enumerate(("malloc", "free"))]
        f = _Fake(rc=0, raise_on=0)
        r = hp.preload_entries(entries, {}, f.fga, f.clear, maps=c["maps"])
        self.assertEqual((r["counts"]["error"], r["counts"]["ok"]), (1, 1))
        self.assertEqual(len(f.calls), 2)

    def test_symtab_needs_verified_build_id(self) -> None:
        c = _libc()
        bid = hp.memory_build_id(c["base"])
        if not bid:
            self.skipTest("libc has no GNU build-id")
        off = c["malloc"] - c["base"]
        meta = {"anchor": {"sym": "malloc", "value": off}, "build_id": bid}
        entries = [{"lib": c["name"], "sym": "anything", "mode": "symtab", "value": off, "bundle": 0}]
        f = _Fake()
        r = hp.preload_entries(entries, {c["name"]: meta}, f.fga, f.clear, maps=c["maps"])
        self.assertEqual(r["counts"]["ok"], 1)
        self.assertEqual(f.calls, [c["base"] + off])
        for bad_meta, reason in (
            ({"anchor": {"sym": "malloc", "value": off}, "build_id": "00" * 20}, "buildid"),
            ({"anchor": {"sym": "malloc", "value": off + 16}, "build_id": bid}, "base"),
            ({"anchor": {"sym": "no_such_symbol_hip_preload_xyz", "value": off}, "build_id": bid}, "anchor"),
            ({}, "anchor"),
        ):
            f = _Fake()
            r = hp.preload_entries(entries, {c["name"]: bad_meta}, f.fga, f.clear, maps=c["maps"])
            self.assertEqual(r["counts"]["symtab_unverified"], 1, reason)
            self.assertEqual(f.calls, [], reason)
            self.assertEqual(r["libs"][c["name"]]["reason"], reason)


class HookOrderTest(unittest.TestCase):
    def test_train_ranker_hook_position(self) -> None:
        with open(os.path.join(os.path.dirname(hp.__file__), "train_ranker.py")) as f:
            src = f.read()
        i_pregrow = src.index("_alloc_pregrow.pregrow_before_run_start()")
        i_hook = src.index("_hip_preload.preload_from_env(")
        i_guard = src.index('_tspec.recheck_globals("run_start")')
        i_run = src.index("mlperf_logger.log_run_start(")
        i_setup = src.index("    setup(\n")
        i_fp8 = src.index("run_fp8_first_use_warmup(\n            device=device")
        self.assertTrue(i_setup < i_fp8 < i_pregrow < i_hook < i_guard < i_run)
        self.assertEqual(src.count("_hip_preload.preload_from_env("), 1)


class GeneratorTest(unittest.TestCase):
    def _gen(self) -> Any:
        path = os.path.join(_REPO, *[".."] * 5, "tools", "recommendation", "dlrm_v4", "gen_hip_preload_table.py")
        spec = importlib.util.spec_from_file_location("gen_hip_preload_table", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_candidates_attribution(self) -> None:
        g = self._gen()
        libs = {"libtorch_hip.so": {"name_first": {"k": 14, "only_torch": 3}},
                "fbgemm_gpu_py.so": {"name_first": {"k": 19}},
                "fbgemm_gpu_tbe_utils.so": {"name_first": {"k": 2}}}
        self.assertEqual(g.candidates("only_torch", ["aten::add"], libs), [("libtorch_hip.so", 3)])
        self.assertEqual(g.candidates("k", ["fbgemm::jagged_unique_indices"], libs),
                         [("fbgemm_gpu_py.so", 19), ("fbgemm_gpu_tbe_utils.so", 2)])
        self.assertEqual(g.candidates("k", ["aten::cumsum"], libs), [("libtorch_hip.so", 14)])
        self.assertEqual(len(g.candidates("k", [""], libs)), 3)
        self.assertEqual(g.candidates("absent", ["aten::add"], libs), [])

    def test_build_id_matches_memory_reader(self) -> None:
        g = self._gen()
        c = _libc()
        with open(c["path"], "rb") as f:
            elf = f.read()
        self.assertEqual(g.build_id(elf, g._sections(elf)), hp.memory_build_id(c["base"]))


# ------------------------------------------------------------------------------------------- per-stack tables (T12)
_ROCM102 = os.path.join(os.path.dirname(hp.DEFAULT_TABLE), "hip_preload_table_rocm10.2.jsonl")
_ROCM102_TORCH_BUILD = "888eba94a598d633c8929634793f6467d5ad2fab"  # TheRock 10.2.0a20261015 torch 2.16.0a0 wheel


class StackTablesTest(unittest.TestCase):
    """One checked-in table per software stack, picked by the build-ids of the libraries the process has loaded."""

    def test_stack_tables(self) -> None:
        self.assertEqual(hp.STACK_TABLES[0], hp.DEFAULT_TABLE)
        self.assertIn(_ROCM102, hp.STACK_TABLES)
        bids = []
        for p in hp.STACK_TABLES:
            self.assertTrue(os.path.exists(p), p)
            bids.append(hp.read_header(p)["libs"][_TORCH]["build_id"])
        self.assertEqual(len(set(bids)), len(bids))  # one table per libtorch_hip build

    def test_rocm102_table(self) -> None:
        header, entries, bad = hp.load_table(_ROCM102)
        self.assertEqual(bad, 0)
        self.assertEqual(header["kind"], "header")
        self.assertEqual(header["libs"][_TORCH]["build_id"], _ROCM102_TORCH_BUILD)
        self.assertEqual(header["libs"][_TORCH].get("kpack"), "torch_gfx950.kpack")
        self.assertEqual(header["n_entries"], len(entries))
        self.assertEqual(header["n_dlsym"], sum(1 for e in entries if e["mode"] == "dlsym"))
        self.assertEqual(header["n_symtab"], sum(1 for e in entries if e["mode"] == "symtab"))
        self.assertEqual(header["entries_by_set"], _census(entries, list(header["traces"])))
        self.assertIn("loads", header["traces"]["1n"])  # first use after RUN_START (load log) source recorded
        seen = set()
        for e in entries:
            key = (e["lib"], e["bundle"])
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertIn(e["lib"], header["libs"])
            self.assertTrue(e["sets"] and set(e["sets"]) <= set(header["traces"]), e)
            if e["mode"] == "symtab":
                self.assertEqual(e["lib"], _TORCH)  # FBGEMM nightly is stripped: dlsym rows only
                self.assertTrue(isinstance(e["value"], int) and e["value"] > 0, e)
        self.assertGreaterEqual(len(entries), 40)

    def test_rocm102_rows_dry_run(self) -> None:
        header, entries, _ = hp.load_table(_ROCM102)
        sel, other = hp.select_entries(header, entries, "1n", False)
        self.assertEqual((len(sel), other), (len(entries), 0))
        self.assertEqual(len(hp.select_entries(header, entries, "2n", False)[0]), len(entries))  # no 2n data: all
        f = _Fake()
        r = hp.preload_entries(sel, header["libs"], f.fga, f.clear, maps={}, opener=_Opener())
        c = r["counts"]
        self.assertEqual(c["ok"], len(sel))
        self.assertEqual(c["nolib"] + c["nosym"] + c["symtab_unverified"] + c["hip"] + c["error"], 0, c)
        self.assertEqual(len(set(f.calls)), len(sel))

    def _tables(self, d: str, bids: List[str]) -> List[str]:
        c = _libc()
        out = []
        for i, bid in enumerate(bids):
            p = os.path.join(d, f"t{i}.jsonl")
            with open(p, "w") as f:
                f.write(json.dumps({"kind": "header", "traces": {"1n": {}},
                                    "libs": {c["name"]: {"build_id": bid}, "libnotloaded_hip_preload.so":
                                             {"build_id": "11" * 20}}}) + "\n")
            out.append(p)
        return out

    def test_pick_table_by_build_id(self) -> None:
        c = _libc()
        real = hp.memory_build_id(c["base"])
        if not real:
            self.skipTest("libc has no GNU build-id")
        with tempfile.TemporaryDirectory() as d:
            wrong, right = self._tables(d, ["00" * 20, real])
            self.assertEqual(hp.build_match(hp.read_header(right), c["maps"]), (1, 1))
            self.assertEqual(hp.build_match(hp.read_header(wrong), c["maps"]), (0, 1))
            self.assertEqual(hp.pick_table((wrong, right), c["maps"]), (right, (1, 1)))
            self.assertEqual(hp.pick_table((right, wrong), c["maps"]), (right, (1, 1)))
            self.assertEqual(hp.pick_table((wrong, os.path.join(d, "missing.jsonl")), c["maps"]), (wrong, (0, 1)))
            self.assertEqual(hp.pick_table((wrong,), {}), (wrong, (0, 0)))  # nothing loaded: the first table

    def test_loader_build_ids(self) -> None:
        c = _libc()
        bids = hp.loaded_object_build_ids()
        with open(c["path"], "rb") as f:
            elf = f.read()
        g = GeneratorTest()._gen()
        want = g.build_id(elf, g._sections(elf))
        if not want:
            self.skipTest("libc has no GNU build-id")
        self.assertEqual(bids.get(c["base"]), want)
        self.assertEqual(hp.memory_build_id(c["base"]), want)

    def test_from_env_uses_the_picked_table(self) -> None:
        real = hp.preload_entries
        op = _Opener()

        def entries_with_fake_libs(sel: List[Dict[str, Any]], libs: Dict[str, Any], fga: Any, clr: Any,
                                   maps: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, Any]:
            return real(sel, libs, fga, clr, maps, opener=op)

        n = hp.read_header(_ROCM102)["n_entries"]
        with mock.patch.object(hp, "mapped_libraries", return_value={}), mock.patch.object(hp, "_Hip", _FakeHip), \
                mock.patch.object(hp, "preload_entries", entries_with_fake_libs), \
                mock.patch.object(hp, "pick_table", return_value=(_ROCM102, (14, 14))):
            with self.assertLogs(hp.logger, level=logging.INFO) as cm:
                res = hp.preload_from_env(rank=0, world_size=8, local_world_size=8, env={hp.ENV: "1"})
        log = "\n".join(cm.output)
        self.assertTrue(res and res["armed"])
        self.assertEqual(res and res["builds"], [14, 14])
        self.assertIn(f"{hp.ARMED} table=hip_preload_table_rocm10.2.jsonl shape=1n entries={n} (other_shape=0 "
                      f"bad_lines=0) ok={n} ", log)
        self.assertIn(" failed=0 (nolib=0 nosym=0 symtab_unverified=0 hip=0 error=0) in ", log)
        self.assertNotIn("matches the build-id of none", log)


class GeneratorKpackTest(unittest.TestCase):
    """tools/recommendation/dlrm_v4/gen_hip_preload_table.py on TheRock kpack libraries and code-object load logs (CPU, synthetic)."""

    def setUp(self) -> None:
        self.g = GeneratorTest()._gen()

    def test_msgpack_subset(self) -> None:
        # {"kernel_name": "lib/x.so", "paths": ["../.kpack/x_@GFXARCH@.kpack"], "n": 300, "neg": -2, "t": true}
        b = (b"\x85\xabkernel_name\xa8lib/x.so\xa5paths\x91\xbb../.kpack/x_@GFXARCH@.kpack"
             b"\xa1n\xcd\x01\x2c\xa3neg\xfe\xa1t\xc3")
        v, q = self.g._msgpack(b)
        self.assertEqual(q, len(b))
        self.assertEqual(v, {"kernel_name": "lib/x.so", "paths": ["../.kpack/x_@GFXARCH@.kpack"], "n": 300,
                             "neg": -2, "t": True})

    def test_hipk_indices(self) -> None:
        import struct as st

        seg = b"".join(st.pack("<4sIQQ", b"HIPK", 1, 0, i) for i in range(3))
        elf = b"\0" * 16 + seg
        secs = [{"name": ".hipFatBinSegment", "off": 16, "size": len(seg)}]
        self.assertEqual(self.g.hipk_indices(elf, secs), [0, 1, 2])
        self.assertEqual(self.g.hipk_indices(elf, []), [])

    def test_match_loads(self) -> None:
        libs = {"libtorch_hip.so": {"rows": [{"index": 0, "kind": "kpack", "co_size": 100},
                                             {"index": 1, "kind": "kpack", "co_size": 200}]},
                "fb.so": {"rows": [{"index": 0, "vaddr": 0x1000, "vaddr_end": 0x2000, "co_size": 50},
                                   {"index": 1, "vaddr": 0x2000, "vaddr_end": 0x3000, "co_size": 70}]}}
        t0 = 1000.0
        ev = [  # pid, t, kind, ms, tid, fields
            {"pid": 1, "t": 999.0, "kind": "kpack", "ms": 1.0, "rc": 0, "tid": 1, "bin": "/x/libtorch_hip.so",
             "idx": "0", "size": "100"},  # before RUN_START: not a target
            {"pid": 1, "t": 999.0, "kind": "reader", "ms": 1.0, "rc": 0, "tid": 1, "size": "100", "lib": "-"},
            {"pid": 1, "t": 1001.0, "kind": "kpack", "ms": 2.0, "rc": 0, "tid": 1, "bin": "/x/libtorch_hip.so",
             "idx": "1", "size": "200"},
            {"pid": 1, "t": 1001.0, "kind": "reader", "ms": 0.5, "rc": 0, "tid": 1, "size": "200", "lib": "-"},
            {"pid": 1, "t": 1001.0, "kind": "load", "ms": 30.0, "rc": 0, "tid": 1},
            {"pid": 2, "t": 1002.0, "kind": "reader", "ms": 0.2, "rc": 0, "tid": 7, "size": "70",
             "lib": "/y/fb.so", "off": "0x2100"},
            {"pid": 2, "t": 1002.0, "kind": "load", "ms": 40.0, "rc": 0, "tid": 7},
            {"pid": 2, "t": 1003.0, "kind": "reader", "ms": 0.3, "rc": 0, "tid": 7, "size": "999", "lib": "-",
             "kd": "triton_k.kd"},
            {"pid": 2, "t": 1003.0, "kind": "load", "ms": 5.0, "rc": 0, "tid": 7},
        ]
        hits, unmatched = self.g.match_loads(ev, libs, t0)
        self.assertEqual(sorted(hits), [("fb.so", 1), ("libtorch_hip.so", 1)])
        self.assertEqual(hits[("libtorch_hip.so", 1)], [[1.0, 32.5]])  # kpack + reader + HSA load
        self.assertEqual(hits[("fb.so", 1)], [[2.0, 40.2]])
        self.assertEqual([(u["size"], u["kd"], u["ms"]) for u in unmatched], [(999, "triton_k.kd", 5.3)])

    def test_parse_load_logs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "coload.123.log")
            with open(p, "w") as f:
                f.write("1793682826.663228 kpack dur_ms=2.427 rc=0 tid=783 bin=/a/libtorch_hip.so idx=7 size=35368\n")
                f.write("garbage\n")
                f.write("1793682826.7 load dur_ms=4.0 rc=0 tid=783 agent=1\n")
            ev = self.g.parse_load_logs([p])
        self.assertEqual([(e["pid"], e["kind"], e.get("idx")) for e in ev], [(123, "kpack", "7"), (123, "load", None)])
        self.assertAlmostEqual(ev[0]["ms"], 2.427)


class ExtensionSweepTest(unittest.TestCase):
    """Trainer HIP extensions built at run time (tempfile.gettempdir()/dlrm_hstu_*): rows derived from the file."""

    def test_deleted_mapping_and_filter(self) -> None:
        seen: List[str] = []

        def rows(path: str) -> Any:
            seen.append(path)
            return [{"lib": path, "sym": "k", "mode": "dlsym", "bundle": 0, "ext": True}], 0

        maps = {"/tmp/dlrm_hstu_fp8_wgrad_ck/ab/libwgrad_ck.so (deleted)": {"base": 1},
                "/tmp/dlrm_hstu_x/liby.so": {"base": 2}, "/usr/lib/libz.so": {"base": 3},
                "/tmp/dlrm_hstu_x/notes.txt": {"base": 4}}
        f = _Fake()
        with mock.patch.object(hp, "extension_rows", rows), \
                mock.patch.object(hp, "_open_path", lambda n, m, mp: hp._Lib(_FakeHandle(0x1000), n, None, "")):
            r = hp.preload_extensions(maps, f.fga, f.clear)
        self.assertEqual(seen, ["/tmp/dlrm_hstu_fp8_wgrad_ck/ab/libwgrad_ck.so", "/tmp/dlrm_hstu_x/liby.so"])
        self.assertEqual((r["n_libs"], r["n_rows"], r["counts"]["ok"]), (2, 2, 2))

    def test_no_extensions_no_calls(self) -> None:
        f = _Fake()
        r = hp.preload_extensions({}, f.fga, f.clear)
        self.assertEqual((r["n_rows"], r["counts"]["ok"], f.calls), (0, 0, []))

    def test_unparsed_only_is_reported(self) -> None:
        # every extension file unparsable (n_libs=0, unparsed=1): its rank still logs the failure
        zero = {"ok": 0, "already": 0, "nolib": 0, "nosym": 0, "symtab_unverified": 0, "hip": 0, "error": 0}
        ext = {"counts": dict(zero), "rows": [], "seconds": 0.0, "libs": {}, "n_libs": 0, "nostub": 0,
               "unparsed": 1, "n_rows": 0}
        table = {"counts": dict(zero, ok=1), "rows": [], "seconds": 0.0, "libs": {}}
        with mock.patch.object(hp, "mapped_libraries", return_value={}), mock.patch.object(hp, "_Hip", _FakeHip), \
                mock.patch.object(hp, "preload_entries", return_value=table), \
                mock.patch.object(hp, "preload_extensions", return_value=ext):
            with self.assertLogs(hp.logger, level=logging.INFO) as cm:
                hp.preload_from_env(rank=3, world_size=8, local_world_size=8, env={hp.ENV: "1"})
        self.assertTrue(any("[hip-preload] trainer extensions rank 3:" in m and "unparsed=1" in m for m in cm.output),
                        cm.output)

    def test_rows_of_a_built_extension(self) -> None:
        import shutil
        import subprocess

        hipcc = shutil.which("hipcc") or ""
        sdk = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ctypes.__file__))), "site-packages",
                           "_rocm_sdk_devel", "bin", "hipcc")
        hipcc = hipcc or (sdk if os.path.exists(sdk) else "")
        if not hipcc:
            self.skipTest("no hipcc")
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "k.hip")
            with open(src, "w") as f:
                f.write('#include <hip/hip_runtime.h>\n'
                        'template <int N> __global__ void kern(float* x) { x[threadIdx.x] *= N; }\n'
                        'extern "C" void launch(float* x) { kern<2><<<1, 32>>>(x); kern<3><<<1, 32>>>(x); }\n')
            so = os.path.join(d, "libk.so")
            p = subprocess.run([hipcc, src, "--offload-arch=gfx950", "-O1", "-shared", "-fPIC", "-o", so],
                               capture_output=True, text=True, timeout=600)
            if p.returncode != 0:
                self.skipTest("hipcc failed: " + p.stderr[-300:])
            rows, nostub = hp.extension_rows(so)
        self.assertEqual(nostub, 0)
        self.assertEqual(len(rows), 1)  # one bundle (one TU), one stub
        self.assertIn(rows[0]["sym"], ("_Z4kernILi2EEvPf", "_Z4kernILi3EEvPf"))


if __name__ == "__main__":
    unittest.main()
