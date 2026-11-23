#!/usr/bin/env python3
"""TSPEC (dlrm_v4/train/triton_spec.py) tests. CPU only, no GPU, no kernel launch.

Pure-python tests always run: env parsing / kill switch, the typed constexpr
codec (with stand-in classes, nothing is put into sys.modules), L1-key option
split, merge (dedup, determinism, torn lines, phase union), phase selection,
LPT shards, the file barrier, the stdlib-only merge CLI, launcher forwarding and
the gin binding.

Tests that need triton run when it is importable (the gate container: Triton
3.6 + torch, no GPU; a host check: Triton 3.8 without torch; the UTD ROCm 10.2
image: Triton 3.9 + torch). They compile tiny kernels for gfx950 ON THE CPU:
JITFunction.run is replaced by an emulation of its L1-miss path (jit.py 3.8
:726-766 / 3.6 :695-729 / 3.9 :733-788: debug/instrumentation (3.9: and
fpsan_homomorphic_casts) kwargs, binder, compute_cache_key, kernel_cache lookup,
_pack_args, _do_compile, the used-globals check) and each kernel's per-device
binder is built for a fixed GPUTarget, so nothing touches a driver. The
cross-version translation is also checked between two real builds with their
own run() (T11 work: tools/xver_check.py, 3.6.0 <-> 3.9.0: every translated row
an L1 hit, identical kernel hashes). Tensors are stand-ins with data_ptr /
ptr_range / dtype. If this image cannot compile for gfx950 on the CPU at all,
the compile tests skip with that reason.
"""

import enum
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List, Tuple
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_REC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))  # .../recommendation
if _REC not in sys.path:
    sys.path.insert(0, _REC)

from generative_recommenders.dlrm_v4.train import triton_spec as T  # noqa: E402

_TSPY = os.path.join(_REC, "generative_recommenders", "dlrm_v4", "train", "triton_spec.py")


def _try_import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


triton = _try_import("triton")
HAVE_TRITON = triton is not None and hasattr(triton, "jit")
torch = _try_import("torch") if HAVE_TRITON else None


# --------------------------------------------------------------------------
# Stand-ins for the codec (shape of triton.language.core.dtype / constexpr and
# torch.dtype: class name + __module__ are what the codec keys on).
# --------------------------------------------------------------------------
class _TlDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"triton.language.{self.name}"

    def __eq__(self, o: Any) -> bool:
        return isinstance(o, _TlDtype) and o.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class _TlConstexpr:
    def __init__(self, value: Any) -> None:
        self.value = value.value if isinstance(value, _TlConstexpr) else value

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"


for _c, _n in ((_TlDtype, "dtype"), (_TlConstexpr, "constexpr")):
    _c.__name__ = _n
    _c.__module__ = "triton.language.core"


class _TorchDtype:
    def __init__(self, n: str) -> None:
        self.n = n

    def __str__(self) -> str:
        return f"torch.{self.n}"


_TorchDtype.__name__ = "dtype"
_TorchDtype.__module__ = "torch"
_TL_NS = type("tl_ns", (), {"dtype": _TlDtype, "constexpr": _TlConstexpr})
_TORCH_NS = type("torch_ns", (), {"bfloat16": _TorchDtype("bfloat16"),
                                  "float32": _TorchDtype("float32")})


class _FakeJit:  # duck-typed JIT callable (only used when triton is absent)
    cache_key = "c" * 64

    def __init__(self) -> None:
        def act(x: Any) -> Any:
            return x

        act.__module__ = "pkg.mod"
        act.__qualname__ = "act"
        self.fn = act


class _Color(enum.IntEnum):
    RED = 1


# ==========================================================================
# Pure python
# ==========================================================================
class EnvTest(unittest.TestCase):

    def test_kill_switch_and_parsing(self) -> None:
        for v in (None, "", "0", " 0 ", "off", "OFF", "false", "none"):
            env = {} if v is None else {"TRITON_SPEC_REPLAY": v, "TRITON_SPEC_RECORD": v}
            with mock.patch.dict(os.environ, env, clear=False):
                if v is None:
                    os.environ.pop("TRITON_SPEC_REPLAY", None)
                    os.environ.pop("TRITON_SPEC_RECORD", None)
                self.assertIsNone(T.env_replay_table(), v)
                self.assertIsNone(T.env_record_dir(), v)
        with mock.patch.dict(os.environ, {"TRITON_SPEC_REPLAY": "/t/x.jsonl",
                                          "TRITON_SPEC_RECORD": "/r"}):
            self.assertEqual(T.env_replay_table(), "/t/x.jsonl")
            self.assertEqual(T.env_record_dir(), "/r")
        for v, want in (("", ("post", True)), ("post", ("post", True)), ("ALL", ("all", True)),
                        ("pre", ("post", False))):
            with mock.patch.dict(os.environ, {"TRITON_SPEC_REPLAY_PHASES": v}):
                self.assertEqual(T.env_replay_phases(), want, v)
        for v, want in (("1", True), ("true", True), ("On", True), ("0", False), ("", False),
                        ("2", False)):
            with mock.patch.dict(os.environ, {"TRITON_SPEC_MONITOR": v}):
                self.assertEqual(T.env_monitor(), want, v)

    def test_off_paths_are_silent_and_never_import_triton(self) -> None:
        T._reset_for_tests()
        with mock.patch.dict(os.environ, {"TRITON_SPEC_REPLAY": "0", "TRITON_SPEC_RECORD": "0",
                                          "TRITON_SPEC_MONITOR": "0"}):
            with mock.patch.object(T, "check_triton", side_effect=AssertionError("imported")):
                self.assertEqual(T.replay_from_env(rank=3), {"armed": False, "reason": "off"})
                self.assertEqual(T.install_from_env(rank=3), {"record": False, "monitor": False})
                T.on_run_stop()
                self.assertEqual(T.monitor_report(log=False), "[triton-spec] monitor off")
                self.assertEqual(T.recheck_globals("run_start"), 0)
                T.mark_run_start()
        T._reset_for_tests()

    def test_resume_is_not_armed_without_touching_triton(self) -> None:
        T._reset_for_tests()
        with mock.patch.dict(os.environ, {"TRITON_SPEC_REPLAY": "/nonexistent/tspec.jsonl"}), \
                mock.patch.object(T, "check_triton", side_effect=AssertionError("imported")), \
                self.assertLogs(T.logger, level="WARNING") as cm:
            c = T.replay_from_env(rank=6, local_world=1, resume=True)
        self.assertEqual(c, {"armed": False, "reason": "resume"})
        self.assertTrue(any("[triton-spec] NOT armed: replay: resume relaunch" in m
                            and m.endswith("rank=6") for m in cm.output), cm.output)

    def test_missing_table_fails_open(self) -> None:
        with mock.patch.dict(os.environ, {"TRITON_SPEC_REPLAY": "/nonexistent/tspec.jsonl"}):
            with self.assertLogs(T.logger, level="WARNING") as cm:
                c = T.replay_from_env(rank=5, local_world=1)
        self.assertFalse(c["armed"])
        self.assertTrue(any("[triton-spec] NOT armed: replay: table not found" in m
                            and m.endswith("rank=5") for m in cm.output), cm.output)


class CodecTest(unittest.TestCase):

    def rt(self, v: Any, **kw: Any) -> Any:
        enc = T.encode_value(v)
        return T.decode_value(json.loads(json.dumps(enc)), tl=_TL_NS, torch=_TORCH_NS, **kw)

    def test_scalars_keep_type_and_str(self) -> None:
        for v in [None, True, False, 0, 1, 16, 2 ** 40, -3, 1e-6, float("inf"), 0.1, "fp32", "",
                  (1, (2, True)), ("a", None, 2.5)]:
            d = self.rt(v)
            self.assertIs(type(d), type(v), v)
            self.assertEqual(str(d), str(v), v)  # ASTSource.hash uses str()
            self.assertEqual(repr(d), repr(v), v)  # the L1 key uses repr()

    def test_bool_is_not_int(self) -> None:
        self.assertEqual(T.encode_value(True)["t"], "bool")
        self.assertEqual(T.encode_value(1)["t"], "int")
        self.assertEqual(str(self.rt(True)), "True")

    def test_stand_in_dtypes_constexpr(self) -> None:
        d = self.rt(_TlDtype("bf16"))
        self.assertEqual((str(d), repr(d)), ("bf16", "triton.language.bf16"))
        c = self.rt(_TlConstexpr((4, _TlDtype("fp32"))))
        self.assertEqual(repr(c), "constexpr[(4, triton.language.fp32)]")
        t = self.rt(_TorchDtype("bfloat16"))
        self.assertEqual(str(t), "torch.bfloat16")

    def test_jit_by_reference(self) -> None:
        j = _FakeJit()
        enc = T.encode_value(j) if not HAVE_TRITON else {
            "t": "jit", "module": "pkg.mod", "qualname": "act", "ck": j.cache_key}
        self.assertEqual(enc, {"t": "jit", "module": "pkg.mod", "qualname": "act",
                               "ck": "c" * 64})
        seen: List[Any] = []
        got = T.decode_value(enc, resolve_jit=lambda m, q: j,
                             on_jit=lambda d, o: seen.append((d["ck"], o)))
        self.assertIs(got, j)
        self.assertEqual(seen, [("c" * 64, j)])

    def test_unsupported(self) -> None:
        import collections

        NT = collections.namedtuple("NT", "a b")
        for v in (_Color.RED, [1, 2], {"a": 1}, NT(1, 2), object()):
            with self.assertRaises(T.Unsupported, msg=repr(v)):
                T.encode_value(v)
        with self.assertRaises(T.Unsupported):
            T.encode_compile_args("k", {}, {"BLOCK": 4}, {}, {})  # non-tuple path

    def test_strict_decode(self) -> None:
        for bad in ({"t": "int", "v": "1"}, {"t": "bool", "v": 1}, {"t": "float", "v": 1.0},
                    {"t": "nope"}, "CORRUPT", {"t": "jit", "module": "m"}, {"t": "tuple", "v": 3}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                T.decode_value(bad, resolve_jit=lambda m, q: None, tl=_TL_NS, torch=_TORCH_NS)

    def test_compile_args_round_trip_keeps_order(self) -> None:
        sig = {"X": "*bf16", "N": "i32", "D": "constexpr", "T": ("*fp32", "i64"), "F": "constexpr"}
        cx = {(2,): 512, (4,): True}
        attrs = {(0,): [["tt.divisibility", 16], ["tt.pointer_range", 32]], (1,): [], (3, 1): []}
        opts = {"num_warps": 1, "extern_libs": (("ocml", "/x/ocml.bc"), ("ockl", "/x/ockl.bc")),
                "supported_fp8_dtypes": ("fp8e4b8", "fp8e4nv"), "arch": "gfx950"}
        e = T.encode_compile_args("KEY", sig, cx, attrs, opts)
        e = json.loads(json.dumps(e, sort_keys=True))  # the table is written with sort_keys
        self.assertEqual([p[0] for p in e["signature"]], list(sig))
        self.assertEqual({p[0]: T._tuplify(p[1]) for p in e["signature"]}, sig)
        at2 = {tuple(p): v for p, v in e["attrs"]}
        self.assertEqual(str(at2), str(attrs))  # str(attrs) enters ASTSource.hash
        cx2 = {tuple(p): T.decode_value(v) for p, v in e["constexprs"]}
        self.assertEqual([(k, type(v)) for k, v in cx2.items()], [((2,), int), ((4,), bool)])
        self.assertEqual(T._tuplify(e["options"])["extern_libs"], opts["extern_libs"])


class KeySplitTest(unittest.TestCase):

    def test_split(self) -> None:
        spec = "[('*fp32', 'DS'), ('i32', ''), ('constexpr', 128)]"
        opts = {"num_warps": 4, "num_stages": 2, "debug": False, "instrumentation_mode": ""}
        got = T.split_key_options(spec + str(opts))
        self.assertEqual(got, (spec, opts))
        self.assertEqual(T.split_key_options("[]{}"), ("[]", {}))
        # "]{" inside a string constexpr of the specialization part
        spec2 = "[('constexpr', ']{x'), ('i32', 'D')]"
        self.assertEqual(T.split_key_options(spec2 + str(opts)), (spec2, opts))
        for bad in ("no-dict", "[(1,)]{'a': object()}", 5, "[(1,)]{1: 2"):
            self.assertIsNone(T.split_key_options(bad), bad)


def _rec_row(q: str, key: str, s: Any, phase: str = "pre", hit: bool = False,
             version: Any = "3.6.0", **extra: Any) -> Dict[str, Any]:
    e = {"schema": T.SCHEMA, "kind": "compile", "module": "m", "qualname": q, "name": q,
         "pid": 7, "rank": 0, "local_rank": 0, "t_rel": 1.0, "compile_s": s, "phase": phase,
         "disk_hit": hit, "warmup": False}
    if version is not None:  # None: the record file's header supplies it
        e["triton_version"] = version
    e.update(T.encode_compile_args(key, {"X": "*bf16"}, {}, {(0,): []}, {"num_warps": 4}))
    e.update(extra)
    return e


def _write(path: str, objs: List[Any], raw_tail: str = "") -> None:
    with open(path, "w") as f:
        for o in objs:
            f.write(json.dumps(o) + "\n")
        f.write(raw_tail)


class MergeTest(unittest.TestCase):

    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.d = self.td.name

    def tearDown(self) -> None:
        self.td.cleanup()

    def p(self, n: str) -> str:
        return os.path.join(self.d, n)

    def _inputs(self) -> Tuple[str, str]:
        hdr = {"kind": "header", "schema": T.SCHEMA, "triton_version": "3.6.0",
               "triton_key_sha": "K", "env": {}, "knobs": {"debug": False}}
        _write(self.p("a.jsonl"), [hdr, _rec_row("k1", "x", 0.5, version=None),
                                   _rec_row("k2", "y", 2.0, version=None),
                                   _rec_row("k1", "x", 0.1, hit=True, version=None)],
               raw_tail='{"kind": "compile", "torn\n')
        _write(self.p("b.jsonl"), [
            hdr, _rec_row("k1", "x", 0.7, phase="post", name="k1b"),
            {"kind": "compile", "unsupported": "x", "key": "z", "module": "m", "qualname": "k3"},
            _rec_row("k4", "w", 9.0, phase="post", via="replay"),
            _rec_row("k5", "v", 1.0, version="3.8.0"),
            {"kind": "summary", "rows": 3}])
        return self.p("a.jsonl"), self.p("b.jsonl")

    def test_dedup_phase_union_last_wins_strip(self) -> None:
        a, b = self._inputs()
        st = T.merge(self.p("t.jsonl"), [a, b])
        self.assertEqual((st["rows"], st["post"], st["pre"], st["kernels"], st["files"]),
                         (3, 1, 2, 3, 2))
        self.assertEqual((st["torn"], st["unsupported"], st["replay"]), (1, 1, 1))
        hdr, rows, torn = T.load_table(self.p("t.jsonl"))
        self.assertEqual((hdr["kind"], hdr["n"], hdr["n_post"], torn), ("table", 3, 1, 0))
        self.assertEqual(hdr["triton_versions"], ["3.6.0", "3.8.0"])
        r1 = [r for r in rows if r["qualname"] == "k1"][0]
        self.assertEqual((r1["n_seen"], r1["phase"], r1["compile_s"], r1["compile_cold"]),
                         (3, "post", 0.7, True))
        self.assertEqual(r1["name"], "k1b")  # content: last occurrence wins
        for k in T._VOLATILE:
            self.assertNotIn(k, r1)
        self.assertFalse(any(r["qualname"] == "k4" for r in rows))  # via=replay dropped

    def test_deterministic_and_order_independent(self) -> None:
        a, b = self._inputs()
        T.merge(self.p("t1.jsonl"), [a, b])
        T.merge(self.p("t2.jsonl"), [a, b])
        with open(self.p("t1.jsonl")) as f1, open(self.p("t2.jsonl")) as f2:
            self.assertEqual(f1.read(), f2.read())
        # no conflicting content -> input order does not matter for the rows
        _write(self.p("c.jsonl"), [_rec_row("k7", "q", 1.0)])
        _write(self.p("d.jsonl"), [_rec_row("k8", "r", 2.0, phase="post")])
        T.merge(self.p("t3.jsonl"), [self.p("c.jsonl"), self.p("d.jsonl")])
        T.merge(self.p("t4.jsonl"), [self.p("d.jsonl"), self.p("c.jsonl")])
        r3 = T.load_table(self.p("t3.jsonl"))[1]
        r4 = T.load_table(self.p("t4.jsonl"))[1]
        self.assertEqual(r3, r4)

    def test_table_as_input_and_cli(self) -> None:
        a, b = self._inputs()
        T.merge(self.p("t.jsonl"), [a, b])
        _write(self.p("new.jsonl"), [_rec_row("k9", "n", 3.0, phase="post")])
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        out = subprocess.run([sys.executable, "-S", "-B", _TSPY, "merge", self.p("t2.jsonl"),
                              self.p("t.jsonl"), self.p("new.jsonl")], capture_output=True,
                             text=True, env=env, cwd=self.d, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("merged rows=4 post=2 pre=2 kernels=4 files=2", out.stdout.strip())
        r1 = [r for r in T.load_table(self.p("t2.jsonl"))[1] if r["qualname"] == "k1"][0]
        self.assertEqual(r1["n_seen"], 3)  # n_seen carried through the earlier table
        bad = subprocess.run([sys.executable, "-S", "-B", _TSPY, "merge"], capture_output=True,
                             text=True, env=env, timeout=120)
        self.assertEqual(bad.returncode, 2)

    def test_no_readable_input_keeps_existing_table(self) -> None:
        a, b = self._inputs()
        T.merge(self.p("t.jsonl"), [a, b])
        before = _read(self.p("t.jsonl"))
        st = T.merge(self.p("t.jsonl"), [self.p("missing.jsonl")])
        self.assertEqual((st["written"], st["files"], st["rows"]), (False, 0, 0))
        _write(self.p("empty.jsonl"), [{"kind": "header", "schema": T.SCHEMA}])
        st = T.merge(self.p("t.jsonl"), [self.p("empty.jsonl")])
        self.assertEqual((st["written"], st["files"], st["rows"]), (False, 1, 0))
        self.assertEqual(_read(self.p("t.jsonl")), before)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        out = subprocess.run([sys.executable, "-S", "-B", _TSPY, "merge", self.p("t.jsonl"),
                              self.p("missing.jsonl")], capture_output=True, text=True, env=env,
                             timeout=120)
        self.assertEqual(out.returncode, 1, out.stderr)
        self.assertIn("merged rows=0 post=0 pre=0 kernels=0 files=0", out.stdout)
        self.assertIn("NOT written", out.stderr)
        self.assertEqual(_read(self.p("t.jsonl")), before)

    def test_world_sizes_targets_and_header_variants(self) -> None:
        tgt = {"backend": "hip", "arch": "gfx950", "warp_size": 64}
        h1 = {"kind": "header", "schema": T.SCHEMA, "triton_version": "3.6.0", "env": {},
              "knobs": {"debug": False, "use_buffer_ops": True}}
        h2 = dict(h1, env={"TRITON_DEFAULT_FP_FUSION": "0"})
        _write(self.p("w8.jsonl"), [h1, _rec_row("k1", "x", 1.0, world_size=8, target=tgt)])
        _write(self.p("w16.jsonl"), [h2, _rec_row("k1", "x", 1.0, world_size=16, target=tgt),
                                     _rec_row("k2", "y", 1.0, world_size=16)])
        st = T.merge(self.p("t.jsonl"), [self.p("w8.jsonl"), self.p("w16.jsonl")])
        self.assertEqual((st["rows"], st["env_variants"], st["knob_variants"]), (2, 2, 1))
        hdr, rows, _ = T.load_table(self.p("t.jsonl"))
        by = {r["qualname"]: r for r in rows}
        self.assertEqual((by["k1"]["world_sizes"], by["k2"]["world_sizes"]), ([8, 16], [16]))
        self.assertNotIn("world_size", by["k1"])
        self.assertEqual(by["k1"]["target"], tgt)
        self.assertEqual((hdr["world_sizes"], len(hdr["env_seen"]), len(hdr["knobs_seen"])),
                         ([8, 16], 2, 1))
        # a table as input keeps every variant
        T.merge(self.p("t2.jsonl"), [self.p("t.jsonl")])
        self.assertEqual(len(T.load_table(self.p("t2.jsonl"))[0]["env_seen"]), 2)

    def test_key_knob_and_env_drift(self) -> None:
        rec = {"debug": False, "instrumentation_mode": "", "use_buffer_ops": True,
               "default_fp_fusion": True}
        self.assertEqual(T.key_knob_drift({"knobs": rec}, dict(rec)), "")
        self.assertEqual(T.key_knob_drift({"knobs": rec}, dict(rec, default_fp_fusion=False)), "")
        d = T.key_knob_drift({"knobs": rec}, dict(rec, use_buffer_ops=False))
        self.assertIn("use_buffer_ops", d)
        self.assertIn("debug", T.key_knob_drift({"knobs": dict(rec, debug=True)}, rec))
        self.assertEqual(T.key_knob_drift({"knobs_seen": [dict(rec, debug=True), rec]}, rec), "")
        self.assertEqual(T.key_knob_drift({"knobs": dict(rec, debug=None)}, rec), "")  # unknown
        self.assertEqual(T.key_knob_drift({}, rec), "")
        self.assertEqual(T.env_drift({"env": {"A": "1"}}, {"A": "1"}), [])
        self.assertEqual(T.env_drift({"env": {"A": "1"}}, {"A": "2", "B": "x"}), ["A", "B"])
        self.assertEqual(T.env_drift({"env_seen": [{"A": "2"}, {}]}, {"A": "2"}), [])

    def test_select_and_shard(self) -> None:
        a, b = self._inputs()
        _, rows, _ = T.load_table(a)  # a raw record file: version from its header
        self.assertEqual({r["triton_version"] for r in rows}, {"3.6.0"})
        T.merge(self.p("t.jsonl"), [a, b])
        _, rows, _ = T.load_table(self.p("t.jsonl"))
        sel, other = T.select_rows(rows, "post", "3.6.0")
        self.assertEqual(([r["qualname"] for r in sel], other), (["k1"], 1))
        sel, other = T.select_rows(rows, "all", "3.6.0")
        self.assertEqual(([r["qualname"] for r in sel], other), (["k1", "k2"], 1))
        rr = [_rec_row(f"k{i}", str(i), s) for i, s in
              enumerate([5, 4, 3, 3, 2, 2, 1, 1, 1, 0.5, None, float("nan")])]
        sh = T.lpt_shard(rr, 4)
        self.assertEqual(sorted(i for s in sh for i in s), list(range(len(rr))))
        loads = [sum((T._num(rr[i]["compile_s"]) or 1.0) for i in s) for s in sh]
        self.assertLessEqual(max(loads) - min(loads), 5.0)
        self.assertEqual(sh, T.lpt_shard(list(rr), 4))  # every rank computes the same shards
        self.assertEqual(T.lpt_shard(rr, 1), [sorted(range(len(rr)), key=lambda i: (
            -(T._num(rr[i]["compile_s"]) or 1.0), T.row_id(rr[i])))])
        # rows only seen as disk hits (warm record run) weigh the median cold time
        mix = [dict(_rec_row("c1", "1", 9.0), compile_cold=True),
               dict(_rec_row("c2", "2", 3.0), compile_cold=True),
               dict(_rec_row("c3", "3", 1.0), compile_cold=True)] + \
            [dict(_rec_row(f"w{i}", f"w{i}", 0.01), compile_cold=False) for i in range(4)]
        sh = T.lpt_shard(mix, 2)
        loads = [sum({9.0: 9.0, 3.0: 3.0, 1.0: 1.0}.get(mix[i]["compile_s"], 3.0) for i in s)
                 for s in sh]
        self.assertEqual(sorted(loads), [12.0, 13.0])  # warm rows at 0.01 s: [4.04, 9.0]


def _row36(key_spec: str, kw: Dict[str, Any], sig: List[List[Any]], attrs: List[List[Any]],
           cx: List[List[Any]], version: str = "3.6.0", **extra: Any) -> Dict[str, Any]:
    r = {"schema": T.SCHEMA, "kind": "compile", "module": "m", "qualname": "q", "name": "q",
         "triton_version": version, "phase": "post", "key": key_spec + str(kw),
         "call_opts": [[k, v] for k, v in kw.items()], "signature": sig, "attrs": attrs,
         "constexprs": cx, "kernel_hash": "h" * 64, "options": {"num_warps": 4},
         "fn_cache_key": "f" * 64, "compile_s": 0.5}
    r.update(extra)
    return r


def _parse_attr(desc: str) -> List[List[Any]]:  # HIPBackend.parse_attr (I:175-179 N:429-433)
    return ([["tt.divisibility", 16]] if "D" in desc else []) + (
        [["tt.pointer_range", 32]] if "S" in desc else [])


class XverTest(unittest.TestCase):
    """Cross-version rows (T11): translate_row / translate_attrs / select_rows_xver."""

    SPEC = "[('*fp32', 'DS'), ('i32', None), ('constexpr', 1), ('constexpr', 'DS')]"
    SIG = [["X", "*fp32"], ["N", "i32"], ["ONE", "constexpr"], ["MODE", "constexpr"]]
    CX = [[[2], {"t": "int", "v": 1}], [[3], {"t": "str", "v": "DS"}]]
    KW36 = {"num_warps": 4, "debug": False, "instrumentation_mode": ""}
    KW39 = dict(KW36, fpsan_homomorphic_casts=False)

    def test_env(self) -> None:
        for v, want in ((None, True), ("", True), ("1", True), ("on", True), ("0", False),
                        ("off", False), ("NO", False)):
            env = {} if v is None else {"TRITON_SPEC_REPLAY_XVER": v}
            with mock.patch.dict(os.environ, env):
                if v is None:
                    os.environ.pop("TRITON_SPEC_REPLAY_XVER", None)
                self.assertEqual(T.env_replay_xver(), want, v)

    def test_translate_key_both_ways(self) -> None:
        r = _row36(self.SPEC, self.KW36, self.SIG, [], self.CX)
        t = T.translate_row(r, "3.9.0", {"fpsan_homomorphic_casts": False})
        self.assertEqual(t["key"], self.SPEC + str(self.KW39))  # run() appends it last
        self.assertEqual(t["call_opts"], [[k, v] for k, v in self.KW39.items()])
        self.assertEqual((t["triton_version"], t["xver_from"]), ("3.9.0", "3.6.0"))
        self.assertNotIn("kernel_hash", t)  # another build's binary
        self.assertNotIn("options", t)  # re-derived from the call kwargs
        self.assertEqual(r["key"], self.SPEC + str(self.KW36))  # input untouched
        back = T.translate_row(dict(t, triton_version="3.9.0"), "3.6.0", {})
        self.assertEqual((back["key"], back["call_opts"]), (r["key"], r["call_opts"]))
        # a user kwarg placed after debug keeps its position; fpsan still goes last
        kw = {"debug": False, "num_warps": 2, "instrumentation_mode": ""}
        t = T.translate_row(_row36(self.SPEC, kw, self.SIG, [], self.CX), "3.9.1",
                            {"fpsan_homomorphic_casts": True})
        self.assertTrue(t["key"].endswith("'instrumentation_mode': '', "
                                          "'fpsan_homomorphic_casts': True}"), t["key"])

    def test_translate_rejects(self) -> None:
        r = _row36(self.SPEC, self.KW36, self.SIG, [], self.CX)
        with self.assertRaises(ValueError):  # 3.8 rules not verified for translation
            T.translate_row(dict(r, triton_version="3.8.0"), "3.9.0", {})
        with self.assertRaises(ValueError):
            T.translate_row(r, "3.10.0", {})
        with self.assertRaises(ValueError):  # the knob's current value is needed
            T.translate_row(r, "3.9.0", {})
        with self.assertRaises(ValueError):  # recorded with fpsan on: no 3.6 equivalent
            T.translate_row(_row36(self.SPEC, dict(self.KW39, fpsan_homomorphic_casts=True),
                                   self.SIG, [], self.CX, version="3.9.0"), "3.6.0", {})
        with self.assertRaises(ValueError):  # call_opts are not the key's kwargs
            T.translate_row(dict(r, call_opts=[["num_warps", 4]]), "3.9.0",
                            {"fpsan_homomorphic_casts": False})
        with self.assertRaises(ValueError):
            T.translate_row(dict(r, key="no-dict-suffix"), "3.9.0",
                            {"fpsan_homomorphic_casts": False})

    def test_translate_attrs(self) -> None:
        cx = {(2,): 1, (3,): "DS"}
        a36 = [((0,), _parse_attr("DS")), ((3,), _parse_attr("DS"))]  # 3.6: str constexpr parsed
        a39 = [((0,), _parse_attr("DS")), ((2,), []), ((3,), [])]  # 3.9: every constexpr, ""
        self.assertEqual(T.translate_attrs(self.SIG, a36, cx, "3.6.0", "3.9.0", _parse_attr), a39)
        self.assertEqual(T.translate_attrs(self.SIG, a39, cx, "3.9.0", "3.6.0", _parse_attr), a36)
        self.assertEqual(T.translate_attrs(self.SIG, a36, cx, "3.6.0", "3.6.0", _parse_attr), a36)
        sig_t = [["T", [["i32", "i32"]]], ["C", "constexpr"]]  # tuple param: nested paths kept
        a = [((0, 0), [["tt.divisibility", 16]]), ((0, 1), [])]
        self.assertEqual(T.translate_attrs(sig_t, a, {(1,): 4}, "3.6.0", "3.9.0", _parse_attr),
                         a + [((1,), [])])
        with self.assertRaises(ValueError):
            T.translate_attrs(self.SIG, a36, cx, "3.8.0", "3.9.0", _parse_attr)

    def test_select_rows_xver(self) -> None:
        own = _row36(self.SPEC, self.KW39, self.SIG, [], self.CX, version="3.9.0")
        dup36 = _row36(self.SPEC, self.KW36, self.SIG, [], self.CX)  # same call, 3.6 record
        new36 = _row36(self.SPEC.replace("'DS'), ('i32'", "'S'), ('i32'"), self.KW36, self.SIG,
                       [], self.CX)
        pre36 = dict(_row36(self.SPEC, dict(self.KW36, num_warps=8), self.SIG, [], self.CX),
                     phase="pre")
        v38 = dict(new36, triton_version="3.8.0")
        rows = [dup36, new36, pre36, v38, own]
        knobs = {"fpsan_homomorphic_casts": False}
        sel, st = T.select_rows_xver(rows, "post", "3.9.0", knobs)
        self.assertEqual([(r["triton_version"], r.get("xver_from")) for r in sel],
                         [("3.9.0", None), ("3.9.0", "3.6.0")])
        self.assertEqual(sel[1]["key"], new36["key"][:new36["key"].rfind("{")] + str(self.KW39))
        self.assertEqual((st["other"], st["xver"], st["xver_dup"], st["xver_bad"],
                          st["xver_phase"]), (4, 1, 1, 1, 1), st)
        sel, st = T.select_rows_xver(rows, "all", "3.9.0", knobs)
        self.assertEqual((len(sel), st["xver"], st["xver_phase"]), (3, 2, 0))
        # an own PRE row also covers the other version's POST row with that key
        own_pre = dict(own, phase="pre")
        sel, st = T.select_rows_xver([dup36, own_pre], "post", "3.9.0", knobs)
        self.assertEqual((sel, st["xver_dup"]), ([], 1))
        # nothing of this version: everything translatable comes from the others
        sel, st = T.select_rows_xver([dup36, new36], "post", "3.9.0", knobs)
        self.assertEqual(len(sel), 2)
        sel, st = T.select_rows_xver([own], "post", "3.6.0", {})  # and back
        self.assertEqual((len(sel), sel[0]["key"]), (1, dup36["key"]))

    def test_version_only_launch_kwargs(self) -> None:
        """opsel-fix: llvm_fn_attrs exists only in 3.9's HIPOptions and the call
        sites pass it only there (gfx950 no-packed-FP32 launches). 3.9 -> 3.6 drops
        it; a 3.6 row of a call whose own 3.9 row carries it is covered (xver_dup),
        not replayed as a never-hit key."""
        opt = "target-features=-packed-fp32-ops"
        kw39o = {"llvm_fn_attrs": opt, **self.KW39}  # Autotuner: call-site kwarg first
        own = _row36(self.SPEC, kw39o, self.SIG, [], self.CX, version="3.9.0")
        back = T.translate_row(own, "3.6.0", {})
        self.assertEqual((back["key"], back["call_opts"]),
                         (self.SPEC + str(self.KW36), [[k, v] for k, v in self.KW36.items()]))
        dup36 = _row36(self.SPEC, self.KW36, self.SIG, [], self.CX)
        knobs = {"fpsan_homomorphic_casts": False}
        sel, st = T.select_rows_xver([dup36, own], "post", "3.9.0", knobs)
        self.assertEqual(([r["key"] for r in sel], st["xver"], st["xver_dup"]),
                         ([own["key"]], 0, 1))
        # dq8 form: the kwarg after the launch kwargs, before run()'s kwargs
        kwd = {"num_warps": 4, "llvm_fn_attrs": opt, "debug": False,
               "instrumentation_mode": "", "fpsan_homomorphic_casts": False}
        own_d = _row36(self.SPEC, kwd, self.SIG, [], self.CX, version="3.9.0")
        sel, st = T.select_rows_xver([dup36, own_d], "post", "3.9.0", knobs)
        self.assertEqual((len(sel), st["xver_dup"]), (1, 1))
        # no own row for that call: translated unchanged (fail-open, first-use compile)
        sel, st = T.select_rows_xver([dup36], "post", "3.9.0", knobs)
        self.assertEqual((len(sel), sel[0]["key"]), (1, self.SPEC + str(self.KW39)))
        # 3.6 replays never see the 3.9-only kwarg
        sel, st = T.select_rows_xver([own, own_d], "post", "3.6.0", {})
        self.assertEqual(([r["key"] for r in sel], st["xver_dup"]),
                         ([self.SPEC + str(self.KW36)], 1))

    def test_merge_keeps_provenance_and_versions(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            t1 = os.path.join(d, "t1.jsonl")
            _write(t1, [{"kind": "table", "schema": T.SCHEMA, "triton_versions": ["3.6.0"],
                         "source": "synthetic 1n/2n/4n", "recorded_runs": ["1n_x"]},
                        _rec_row("k1", "x", 1.0, phase="post")])
            r39 = os.path.join(d, "r39.jsonl")
            _write(r39, [{"kind": "header", "schema": T.SCHEMA, "triton_version": "3.9.0",
                          "knobs": {"fpsan_homomorphic_casts": False}},
                         _rec_row("k1", "x", 2.0, phase="post", version=None)])
            out = os.path.join(d, "t2.jsonl")
            st = T.merge(out, [t1, r39])
            self.assertEqual(st["rows"], 2)  # one row per Triton version
            hdr, rows, _ = T.load_table(out)
            self.assertEqual(hdr["triton_versions"], ["3.6.0", "3.9.0"])
            self.assertEqual(hdr["provenance"], [{"source": "synthetic 1n/2n/4n",
                                                  "recorded_runs": ["1n_x"],
                                                  "triton_versions": ["3.6.0"]}])
            T.merge(os.path.join(d, "t3.jsonl"), [out])  # carried through later merges
            self.assertEqual(T.load_table(os.path.join(d, "t3.jsonl"))[0]["provenance"],
                             hdr["provenance"])

    def test_merge_replace_version(self) -> None:
        """Re-recording one stack: --replace-version drops that version's rows (and
        the provenance of their campaign) from input TABLES; record files and a
        row-less provenance header given now are kept; other versions untouched."""
        with tempfile.TemporaryDirectory() as d:
            p = lambda n: os.path.join(d, n)  # noqa: E731
            _write(p("t.jsonl"), [
                {"kind": "table", "schema": T.SCHEMA, "triton_versions": ["3.6.0", "3.9.0"],
                 "provenance": [{"source": "old36", "triton_versions": ["3.6.0"]},
                                {"source": "old39", "triton_versions": ["3.9.0"]}]},
                dict(_rec_row("a", "x", 1.0, phase="post"), n_seen=0),  # class-completed row
                _rec_row("b", "y", 1.0, phase="post", version="3.9.0"),
                _rec_row("stale", "z", 1.0, phase="post", version="3.9.0")])
            _write(p("prov.jsonl"), [{"kind": "table", "schema": T.SCHEMA,
                                      "triton_versions": ["3.9.0"], "source": "new39"}])
            _write(p("r.jsonl"), [{"kind": "header", "schema": T.SCHEMA, "triton_version": "3.9.0"},
                                  _rec_row("b", "y", 2.0, phase="post", version=None)])
            st = T.merge(p("o.jsonl"), [p("t.jsonl"), p("prov.jsonl"), p("r.jsonl")],
                         replace_versions=["3.9.0"])
            self.assertEqual((st["rows"], st["replaced"]), (2, 2))
            hdr, rows, _ = T.load_table(p("o.jsonl"))
            self.assertEqual(sorted((r["qualname"], r["triton_version"], r["n_seen"]) for r in rows),
                             [("a", "3.6.0", 0), ("b", "3.9.0", 1)])  # n_seen 0 kept
            self.assertEqual(sorted(x["source"] for x in hdr["provenance"]), ["new39", "old36"])
            out = subprocess.run([sys.executable, "-S", "-B", _TSPY, "merge", "--replace-version",
                                  "3.9.0", p("o2.jsonl"), p("t.jsonl"), p("r.jsonl")],
                                 capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("replaced: 2 rows of Triton 3.9.0", out.stderr)
            self.assertIn("merged rows=2 ", out.stdout)
            st = T.merge(p("o3.jsonl"), [p("t.jsonl")])  # without it nothing is dropped
            self.assertEqual((st["rows"], st["replaced"]), (3, 0))


class BarrierTest(unittest.TestCase):

    def test_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(T.file_barrier(d, "t0", 0, 1, 1.0), "single")
            t = time.monotonic()
            self.assertEqual(T.file_barrier(d, "t1", 0, 2, 0.2), "timeout")
            self.assertLess(time.monotonic() - t, 5.0)
            res: Dict[int, str] = {}

            def member(i: int) -> None:
                res[i] = T.file_barrier(d, "t2", i, 3, 30.0)

            th = [threading.Thread(target=member, args=(i,)) for i in range(3)]
            for x in th:
                x.start()
            for x in th:
                x.join(60)
            self.assertEqual(res, {0: "ok", 1: "ok", 2: "ok"})
            T.barrier_arrive(d, "t3", 1)  # an early-failing rank only arrives
            self.assertEqual(T.file_barrier(d, "t3", 0, 2, 5.0), "ok")
        self.assertEqual(T.launch_token(), T.launch_token())


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


class LauncherGinTest(unittest.TestCase):
    _VARS = ("TRITON_SPEC_RECORD", "TRITON_SPEC_REPLAY", "TRITON_SPEC_REPLAY_PHASES",
             "TRITON_SPEC_MONITOR")
    # env-only knobs (no apply_env_bootstrap binding) that launch_slurm.sh must still forward
    _LAUNCHER_ONLY_VARS = ("TRITON_SPEC_REPLAY_XVER",)

    def test_launcher_forwards_and_validates(self) -> None:
        sh = os.path.join(_REC, "scripts", "launch_slurm.sh")
        if not os.path.exists(sh):
            self.skipTest("no scripts/launch_slurm.sh in this checkout")
        txt = _read(sh)
        loops = [ln for ln in txt.splitlines() if ln.strip().startswith("for _ts_v in ")]
        for v in self._VARS + self._LAUNCHER_ONLY_VARS:
            line = "${%s:+-e %s=$%s}" % (v, v, v)
            self.assertTrue(line in txt, f"launch_slurm.sh does not forward {v}")
            self.assertTrue(any(v in ln.replace(";", " ").split() for ln in loops),
                            f"launch_slurm.sh does not validate {v}")
        self.assertTrue("^[A-Za-z0-9_./=+-]+$" in txt, "no TRITON_SPEC_* value validation")

    def test_gin_binding_and_bootstrap_param(self) -> None:
        txt = _read(os.path.join(_REC, "generative_recommenders", "dlrm_v4", "train", "gin",
                                 "yambda_5b.gin"))
        table = "generative_recommenders/ops/triton/triton_spec_table.jsonl"
        self.assertTrue(f'\napply_env_bootstrap.TRITON_SPEC_REPLAY = "{table}"\n' in txt,
                        f"gin binding apply_env_bootstrap.TRITON_SPEC_REPLAY = \"{table}\" missing")
        self.assertTrue(os.path.exists(os.path.join(_REC, table)), f"replay table {table} missing")
        self.assertTrue("# (57) TSPEC, a Triton specialization table" in txt, "gin comment block missing")
        src = _read(os.path.join(_REC, "generative_recommenders", "dlrm_v4", "train",
                                 "_env_bootstrap.py"))
        for v in self._VARS:
            self.assertTrue(f'_bind_value("{v}", {v})' in src, f"_env_bootstrap: no {v}")
        gin = _try_import("gin")
        if gin is None:
            return
        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

        import inspect

        params = inspect.signature(eb.apply_env_bootstrap).parameters
        self.assertIn("TRITON_SPEC_REPLAY", params)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRITON_SPEC_REPLAY", None)
            eb._bind_value("TRITON_SPEC_REPLAY", "0")  # gin default
            self.assertIsNone(T.env_replay_table())
            os.environ["TRITON_SPEC_REPLAY"] = "/x/t.jsonl"  # pre-set env wins
            eb._bind_value("TRITON_SPEC_REPLAY", "0")
            self.assertEqual(T.env_replay_table(), "/x/t.jsonl")


# ==========================================================================
# Triton (CPU compile, no GPU)
# ==========================================================================
if HAVE_TRITON:
    import triton.language as tl

    SCALE = tl.constexpr(2)

    @triton.jit
    def _tsp_act(x):
        return x * 2.0

    @triton.jit
    def tsp_add(X, Y, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        tl.store(Y + offs, tl.load(X + offs, mask=m) + 1.0, mask=m)

    @triton.jit
    def tsp_apply(X, Y, N, F: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        tl.store(Y + offs, F(tl.load(X + offs, mask=m)), mask=m)

    @triton.jit
    def tsp_scale(X, Y, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        tl.store(Y + offs, tl.load(X + offs, mask=m) * SCALE, mask=m)

    @triton.autotune(configs=[triton.Config({"BLOCK": 64}, num_warps=2)], key=["N"])
    @triton.jit
    def tsp_tuned(X, Y, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        tl.store(Y + offs, tl.load(X + offs, mask=m) - 1.0, mask=m)

    @triton.heuristics(values={"BLOCK": lambda a: 128})
    @triton.jit
    def tsp_heur(X, Y, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        tl.store(Y + offs, tl.load(X + offs, mask=m) * 0.5, mask=m)

    def _make_local_kernel():  # a kernel that only lives in a closure (<locals>)
        @triton.jit
        def tsp_local(X, Y, N, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = offs < N
            tl.store(Y + offs, tl.load(X + offs, mask=m) + 3.0, mask=m)

        return tsp_local

    TSP_LOCAL = _make_local_kernel()

    _KERNELS = [_tsp_act, tsp_add, tsp_apply, tsp_scale, tsp_tuned.fn, tsp_heur.fn, TSP_LOCAL]

    # globals-guard kernels (own tables; not in _KERNELS)
    GAIN = tl.constexpr(2.0)
    GACT = tl.constexpr(3.0)

    @triton.jit
    def _tsp_ghelp(x):  # a @triton.jit helper that reads a module global
        return x * GAIN

    @triton.jit
    def tsp_gk(X, Y, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        tl.store(Y + offs, _tsp_ghelp(tl.load(X + offs, mask=m)), mask=m)

    @triton.jit
    def tsp_gk2(X, Y, N, BLOCK: tl.constexpr):  # never replayed; same helper
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        tl.store(Y + offs, _tsp_ghelp(tl.load(X + offs, mask=m)) + 1.0, mask=m)

    @triton.jit
    def _tsp_gact(x):  # a JIT-callable constexpr that reads a module global
        return x * GACT

    _GKERNELS = [tsp_gk, tsp_gk2, _tsp_ghelp, _tsp_gact, tsp_apply]


class _FDT:  # torch-like dtype: str() is what triton canonicalizes
    def __init__(self, n: str) -> None:
        self.n = n

    def __str__(self) -> str:
        return "torch." + self.n

    __repr__ = __str__

    def __hash__(self) -> int:
        return hash(("tspec-test-dtype", self.n))

    def __eq__(self, o: Any) -> bool:
        return isinstance(o, _FDT) and o.n == self.n


class _FT:  # tensor stand-in: 16-byte aligned, small storage ('D' + 'S' classes)
    def __init__(self, ptr: int = 1 << 20) -> None:
        self.dtype = _FDT("float32")
        self._ptr = ptr

    def data_ptr(self) -> int:
        return self._ptr

    def ptr_range(self) -> int:
        return 1 << 20


def _cpu_binder(fn: Any) -> Tuple[Any, ...]:
    """JITFunction.create_binder (jit.py 3.8 :689-700 / 3.6 :658-669) for a fixed
    gfx950 target instead of driver.active."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import compiler as tc
    from triton.runtime.jit import create_function_from_signature

    target = GPUTarget("hip", "gfx950", 64)
    backend = tc.make_backend(target)
    fn.CompiledKernel, fn.compile, fn.ASTSource = tc.CompiledKernel, tc.compile, tc.ASTSource
    return {}, {}, target, backend, create_function_from_signature(fn.signature, fn.params,
                                                                   backend)


def _emu_run(self: Any, *args: Any, grid: Any = None, warmup: bool = False, **kwargs: Any) -> Any:
    """JITFunction.run up to the launch (jit.py 3.8 :726-766 / 3.6 :695-729 / 3.9 :733-788)."""
    from triton import knobs
    from triton.runtime.jit import compute_cache_key

    kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
    kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    if hasattr(knobs.compilation, "fpsan_homomorphic_casts"):  # 3.9 (jit.py N:736)
        kwargs["fpsan_homomorphic_casts"] = knobs.compilation.fpsan_homomorphic_casts
    device = 0
    if device not in self.device_caches:
        self.device_caches[device] = _cpu_binder(self)
    kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
    bound_args, specialization, options = binder(*args, **kwargs)
    key = compute_cache_key(kernel_key_cache, specialization, options)
    kernel = kernel_cache.get(key, None)
    if kernel is None:
        options, signature, constexprs, attrs = self._pack_args(backend, kwargs, bound_args,
                                                                specialization, options)
        kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
    not_present = object()
    for (name, _), (val, globals_dict) in self.used_global_vals.items():
        if (new_val := globals_dict.get(name, not_present)) != val:
            raise RuntimeError(f"Global variable {name} has changed since we compiled this "
                               f"kernel, from {val} to {new_val}")
    return kernel


_CPU_COMPILE: Dict[str, Any] = {}


def _cpu_compile_reason() -> str:
    """'' when this image can compile a gfx950 kernel on the CPU, else why not."""
    if "reason" not in _CPU_COMPILE:
        try:
            from triton.backends.compiler import GPUTarget
            from triton.compiler.compiler import ASTSource, compile as tcompile

            with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ,
                                                                     {"TRITON_CACHE_DIR": d}):
                src = ASTSource(tsp_add, {"X": "*fp32", "Y": "*fp32", "N": "i32",
                                          "BLOCK": "constexpr"}, {(3,): 64}, {})
                k = tcompile(src, target=GPUTarget("hip", "gfx950", 64), options={})
                if not k.asm.get("hsaco"):
                    _CPU_COMPILE["reason"] = "no hsaco produced"
                    return _CPU_COMPILE["reason"]
                # The stand-in tensors through THIS build's native specializer and the
                # emulated L1-miss path, as every compile test below does: an
                # incompatible build skips the class with the reason instead of failing.
                from triton.runtime.jit import JITFunction

                saved = (dict(tsp_add.device_caches), tsp_add.hash, tsp_add.used_global_vals)
                try:
                    tsp_add.device_caches.clear()
                    with mock.patch.object(JITFunction, "run", _emu_run):
                        kern = tsp_add[(1,)](_FT(), _FT(2 << 20), 1000, BLOCK=64)
                    if not kern.asm.get("hsaco"):
                        _CPU_COMPILE["reason"] = "binder path: no hsaco produced"
                        return _CPU_COMPILE["reason"]
                finally:
                    tsp_add.device_caches.clear()
                    tsp_add.device_caches.update(saved[0])
                    tsp_add.hash, tsp_add.used_global_vals = saved[1], saved[2]
                _CPU_COMPILE["reason"] = ""
        except Exception as ex:
            _CPU_COMPILE["reason"] = f"{type(ex).__name__}: {ex}"[:300]
    return _CPU_COMPILE["reason"]


def _calls_pre() -> None:
    x, y = _FT(), _FT(2 << 20)
    tsp_add[(1,)](x, y, 1000, BLOCK=128)
    tsp_tuned[(1,)](x, y, 1000)


def _calls_post() -> None:
    x, y = _FT(), _FT(2 << 20)
    tsp_add[(1,)](x, y, 1024, BLOCK=128)  # N % 16 == 0: another class than 1000
    tsp_apply[(1,)](x, y, 1000, F=_tsp_act, BLOCK=128)
    tsp_heur[(1,)](x, y, 1000)
    tsp_scale[(1,)](x, y, 1000, BLOCK=128)
    TSP_LOCAL[(1,)](x, y, 1000, BLOCK=64)


@unittest.skipUnless(HAVE_TRITON, "triton not importable")
class TritonResolveCodecTest(unittest.TestCase):

    def test_resolve_kernel(self) -> None:
        mod = __name__
        self.assertIs(T.resolve_kernel(mod, "tsp_add"), tsp_add)
        self.assertIs(T.resolve_kernel(mod, "tsp_tuned"), tsp_tuned.fn)  # Autotuner unwrap
        self.assertIs(T.resolve_kernel(mod, "tsp_heur"), tsp_heur.fn)  # Heuristics unwrap
        with self.assertRaises(Exception):
            T.resolve_kernel(mod, "no_such_kernel")
        with self.assertRaises(LookupError):  # attribute bound to another kernel
            T.resolve_kernel(mod, "TSP_LOCAL")
        with self.assertRaises(LookupError):  # not under the import prefixes, not loaded
            T.resolve_kernel("tspec_no_such_module_xyz", "k")
        ok, reason, version = T.check_triton()
        self.assertTrue(ok, reason)
        self.assertEqual(version, triton.__version__)

    def test_real_codec(self) -> None:
        for v in (tl.bfloat16, tl.float8e4nv, tl.constexpr(4), tl.constexpr((tl.int32, 2)),
                  (tl.float32, True)):
            enc = json.loads(json.dumps(T.encode_value(v)))
            d = T.decode_value(enc)
            self.assertEqual((str(d), repr(d)), (str(v), repr(v)))
        with self.assertRaises(T.Unsupported):
            T.encode_value(tl.pointer_type(tl.float32))
        enc = T.encode_value(_tsp_act)
        self.assertEqual((enc["t"], enc["qualname"], enc["ck"]),
                         ("jit", "_tsp_act", _tsp_act.cache_key))
        if torch is not None:
            enc = json.loads(json.dumps(T.encode_value(torch.bfloat16)))
            self.assertIs(T.decode_value(enc), torch.bfloat16)


@unittest.skipUnless(HAVE_TRITON, "triton not importable")
class TritonReplayTest(unittest.TestCase):
    """record -> merge -> replay -> the same calls are all L1 hits, compiled from
    the current source into the identical binary."""

    @classmethod
    def setUpClass(cls) -> None:
        why = _cpu_compile_reason()
        if why:
            raise unittest.SkipTest(f"this image cannot compile gfx950 kernels on the CPU: {why}")
        from triton.runtime.jit import JITFunction

        cls.JITFunction = JITFunction
        cls.td = tempfile.TemporaryDirectory()
        cls.dir = cls.td.name
        cls.env_patch = mock.patch.dict(os.environ, {})  # restores TRITON_CACHE_DIR after
        cls.env_patch.start()
        cls.run_patch = mock.patch.object(JITFunction, "run", _emu_run)
        cls.run_patch.start()
        try:
            cls.hsaco = cls._record()
        except Exception:
            T._reset_for_tests()
            cls.run_patch.stop()
            cls.env_patch.stop()
            cls.td.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        T._reset_for_tests()
        cls.run_patch.stop()
        cls.env_patch.stop()
        cls.td.cleanup()

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _fresh(cache_dir: str) -> None:
        T._reset_for_tests()
        os.environ["TRITON_CACHE_DIR"] = cache_dir
        for f in _KERNELS:
            f.device_caches.clear()
            f.hash = None
            f.used_global_vals = {}
            f.device_caches[0] = _cpu_binder(f)

    @classmethod
    def _record(cls) -> Dict[str, bytes]:
        rec_dir = os.path.join(cls.dir, "rec")
        with mock.patch.dict(os.environ, {"TRITON_CACHE_DIR": os.path.join(cls.dir, "c_rec")}):
            cls._fresh(os.path.join(cls.dir, "c_rec"))
            assert T.install_recorder(rec_dir, rank=0, local_rank=0)
            assert T.install_monitor(rank=0)
            _calls_pre()
            T.mark_run_start()
            _calls_post()
            cls.monitor_line = T.monitor_report(log=False)
            hs = {}
            for f in _KERNELS:
                for k, kern in f.device_caches[0][0].items():
                    hs[k] = kern.asm["hsaco"]
            T._reset_for_tests()
        files = [os.path.join(rec_dir, x) for x in os.listdir(rec_dir)]
        cls.merge_stats = T.merge(os.path.join(cls.dir, "table.jsonl"), files)
        return hs

    def _counting(self) -> List[str]:
        calls: List[str] = []
        orig = self.JITFunction._do_compile

        def counted(fn: Any, *a: Any, **k: Any) -> Any:
            calls.append(fn.__name__)
            return orig(fn, *a, **k)

        p = mock.patch.object(self.JITFunction, "_do_compile", counted)
        p.start()
        self.addCleanup(p.stop)
        return calls

    def _replay(self, table: str, tag: str, **kw: Any) -> Dict[str, Any]:
        self._fresh(os.path.join(self.dir, f"c_{tag}_{time.monotonic_ns()}"))
        kw.setdefault("phases", "all")
        kw.setdefault("prime", False)
        return T.replay(table, rank=0, local_rank=0, local_world=1, device=0, log=False, **kw)

    def _table_rows(self) -> List[Dict[str, Any]]:
        with open(os.path.join(self.dir, "table.jsonl")) as f:
            return [json.loads(x) for x in f if x.strip()]

    def _write_table(self, objs: List[Dict[str, Any]], name: str) -> str:
        p = os.path.join(self.dir, name)
        _write(p, objs)
        return p

    # -- tests ---------------------------------------------------------------
    def test_record_and_merge(self) -> None:
        st = self.merge_stats
        self.assertEqual((st["rows"], st["pre"], st["post"], st["kernels"]), (7, 2, 5, 6))
        self.assertTrue(self.monitor_line.startswith(
            "[triton-spec] monitor post-run_start: compiles=5 kernels="), self.monitor_line)
        rows = self._table_rows()
        self.assertEqual(rows[0]["kind"], "table")
        by = {r["qualname"]: r for r in rows[1:] if r["phase"] == "post"}
        self.assertTrue(by["tsp_add"]["same_instance"])
        self.assertFalse(by["_make_local_kernel.<locals>.tsp_local"]["same_instance"])
        self.assertEqual(by["tsp_scale"]["globals"], [[__name__, "SCALE", "constexpr[2]"]])
        self.assertTrue(by["tsp_add"]["call_opts"])  # options re-derivable from the key
        self.assertEqual(by["tsp_add"]["target"], {"backend": "hip", "arch": "gfx950",
                                                   "warp_size": 64})
        enc = [e for _, e in by["tsp_apply"]["constexprs"] if e["t"] == "jit"][0]
        self.assertEqual(enc["ck"], _tsp_act.cache_key)
        self.assertIn(enc["ck"], by["tsp_apply"]["key"])

    def test_replay_all_rows_are_l1_hits_with_identical_binaries(self) -> None:
        c = self._replay(os.path.join(self.dir, "table.jsonl"), "all")
        self.assertTrue(c["armed"], c)
        self.assertEqual((c["rows"], c["resolved"], c["unresolved"], c["compiled"], c["errors"]),
                         (7, 7, 0, 7, 0), c)
        self.assertEqual((c["hash_same"], c["hash_known"], c["src_changed"]), (7, 7, 0), c)
        self.assertEqual((c["opts_drift"], c["fallback_opts"], c["rekeyed"], c["already"]),
                         (0, 0, 0, 0), c)
        self.assertEqual(len(c["slowest"]), 5)  # per-row times, slowest first
        self.assertEqual(c["slowest"], sorted(c["slowest"], key=lambda x: -x[1]))
        replayed = {k: kern.asm["hsaco"] for f in _KERNELS
                    for k, kern in f.device_caches[0][0].items()}
        self.assertEqual(set(replayed), set(self.hsaco))
        for k, blob in replayed.items():
            self.assertEqual(blob, self.hsaco[k], k)  # bitwise: the binary the run compiles
        calls = self._counting()
        _calls_pre()
        _calls_post()
        self.assertEqual(calls, [])

    def test_phases_post(self) -> None:
        c = self._replay(os.path.join(self.dir, "table.jsonl"), "post", phases="post")
        self.assertEqual((c["rows"], c["compiled"], c["errors"], c["unresolved"]), (5, 5, 0, 0))
        calls = self._counting()
        _calls_post()
        self.assertEqual(calls, [])
        _calls_pre()
        self.assertEqual(sorted(calls), ["tsp_add", "tsp_tuned"])

    def test_monitor_counts_only_post_run_start_misses(self) -> None:
        self._replay(os.path.join(self.dir, "table.jsonl"), "mon")
        self.assertTrue(T.install_monitor(rank=4))
        _calls_pre()
        T.mark_run_start()
        _calls_post()
        x, y = _FT(), _FT()
        tsp_add[(1,)](x, y, 1000, BLOCK=256)  # a specialization the record run never saw
        with self.assertLogs(T.logger, level="INFO") as cm:
            T.on_run_stop()
            T.on_run_stop()  # once per process
        lines = [m for m in cm.output if "monitor post-run_start" in m]
        self.assertEqual(len(lines), 1, cm.output)
        self.assertTrue(lines[0].endswith(
            "[triton-spec] monitor post-run_start: compiles=1 kernels=tsp_add:1 rank=4"),
            lines[0])

    def test_corrupt_and_unknown_rows_fail_open(self) -> None:
        objs = self._table_rows()
        rows = objs[1:]
        i = [j for j, r in enumerate(rows) if r["qualname"] == "tsp_heur"][0]
        bad = dict(rows[i])
        for k in ("signature", "constexprs", "attrs", "options"):
            bad[k] = "CORRUPT"
        unk = dict(rows[0], qualname="tsp_no_such_kernel", name="tsp_no_such_kernel",
                   key=rows[0]["key"] + " unknown")
        stale = dict(rows[[j for j, r in enumerate(rows) if r["qualname"] == "tsp_add"][0]])
        stale["signature"] = [["X", "*fp32"], ["Y", "*fp32"], ["N", "i32"]]  # a param removed
        stale["key"] = stale["key"] + " stale"
        torn = '{"kind": "compile", "module": "x'
        p = self._write_table([objs[0]] + rows[:i] + [bad] + rows[i + 1:] + [unk, stale],
                              "bad.jsonl")
        with open(p, "a") as f:
            f.write(torn + "\n")
        self._fresh(os.path.join(self.dir, "c_bad"))
        with self.assertLogs(T.logger, level="INFO") as cm:
            c = T.replay(p, rank=2, local_rank=0, local_world=1, device=0, phases="all",
                         prime=False)
        self.assertTrue(c["armed"], c)
        self.assertEqual((c["rows"], c["errors"], c["unresolved"], c["resolved"]), (9, 1, 3, 6), c)
        self.assertEqual(c["unresolved_by"], {"decode": 1, "missing": 1, "signature": 1}, c)
        self.assertEqual(c["torn"], 1)
        armed = [m for m in cm.output if "armed OK: [triton-spec] replay table=" in m]
        self.assertEqual(len(armed), 1, cm.output)
        self.assertIn("rows=9 resolved=6 unresolved=3 compiled=6 primed=0 errors=1 in ", armed[0])
        self.assertTrue(armed[0].endswith(" rank=2"), armed[0])
        calls = self._counting()
        _calls_pre()
        _calls_post()
        self.assertEqual(calls, ["tsp_heur"])  # only the corrupted row compiles, at first use

    def test_prime_failure_restores_kernel_state(self) -> None:
        from triton.compiler.compiler import CompiledKernel

        def boom(self: Any) -> None:
            self._run = "half-initialized"
            raise RuntimeError("no GPU in this test")

        with mock.patch.object(CompiledKernel, "_init_handles", boom):
            c = self._replay(os.path.join(self.dir, "table.jsonl"), "prime", prime=True)
        self.assertTrue(c["armed"])
        self.assertEqual((c["compiled"], c["primed"], c["errors"]), (7, 0, 7), c)
        for f in _KERNELS:
            for kern in f.device_caches[0][0].values():
                self.assertIsNone(kern._run)  # the first launch retries _init_handles itself
                self.assertIsNone(kern.module)

    def test_prime_out_of_resources_is_not_an_error(self) -> None:
        from triton.compiler.compiler import CompiledKernel
        from triton.runtime.errors import OutOfResources

        def oor(self: Any) -> None:
            self._run = "half-initialized"
            raise OutOfResources(1 << 20, 65536, "shared memory")

        with mock.patch.object(CompiledKernel, "_init_handles", oor):
            c = self._replay(os.path.join(self.dir, "table.jsonl"), "oor", prime=True)
        self.assertTrue(c["armed"], c)
        self.assertEqual((c["compiled"], c["primed"], c["prime_oor"], c["errors"]), (7, 0, 7, 0), c)
        for f in _KERNELS:
            for kern in f.device_caches[0][0].values():
                self.assertIsNone(kern._run)  # the first launch raises it itself, as without TSPEC

    def test_other_target_rows_are_skipped_and_not_armed(self) -> None:
        objs = self._table_rows()
        out = [objs[0]] + [dict(r, target={"backend": "hip", "arch": "gfx942", "warp_size": 64})
                           for r in objs[1:]]
        p = self._write_table(out, "target.jsonl")
        self._fresh(os.path.join(self.dir, f"c_tgt_{time.monotonic_ns()}"))
        with self.assertLogs(T.logger, level="INFO") as cm:
            c = T.replay(p, rank=3, local_rank=0, local_world=1, device=0, phases="all",
                         prime=False)
        self.assertFalse(c["armed"], c)
        self.assertEqual((c["unresolved_by"], c["compiled"], c["errors"]), ({"target": 7}, 0, 0), c)
        self.assertTrue(any("[triton-spec] NOT armed: replay: no row compiled" in m
                            and m.endswith("rank=3") for m in cm.output), cm.output)
        self.assertFalse(any("armed OK" in m for m in cm.output), cm.output)
        self.assertTrue(any("replay detail:" in m for m in cm.output), cm.output)

    def test_key_knob_drift_is_not_armed(self) -> None:
        objs = self._table_rows()
        hdr = dict(objs[0])
        kn = dict(hdr.get("knobs") or {})
        if not isinstance(kn.get("use_buffer_ops"), bool):
            self.skipTest(f"use_buffer_ops not recorded as a bool: {kn}")
        kn["use_buffer_ops"] = not kn["use_buffer_ops"]
        hdr["knobs"], hdr["knobs_seen"] = kn, [kn]
        c = self._replay(self._write_table([hdr] + objs[1:], "knobs.jsonl"), "knobs")
        self.assertFalse(c["armed"], c)
        self.assertIn("use_buffer_ops", c["reason"])
        self.assertEqual(c["compiled"], 0)

    def test_changed_global_row_is_skipped(self) -> None:
        mod = sys.modules[__name__]
        old = mod.SCALE
        try:
            mod.SCALE = tl.constexpr(3)
            c = self._replay(os.path.join(self.dir, "table.jsonl"), "glob")
        finally:
            mod.SCALE = old
        self.assertEqual((c["unresolved"], c["unresolved_by"], c["compiled"]),
                         (1, {"globals": 1}, 6), c)

    def test_global_changed_after_replay_is_evicted(self) -> None:
        mod = sys.modules[__name__]
        old = mod.SCALE
        c = self._replay(os.path.join(self.dir, "table.jsonl"), "evict")
        self.assertEqual(c["compiled"], 7)
        try:
            mod.SCALE = tl.constexpr(3)
            T.mark_run_start()
            self.assertEqual(T._STATE.globals_changed, 1)
            calls = self._counting()
            x, y = _FT(), _FT(2 << 20)
            tsp_scale[(1,)](x, y, 1000, BLOCK=128)  # without the eviction: RuntimeError
            self.assertEqual(calls, ["tsp_scale"])
        finally:
            mod.SCALE = old
            tsp_scale.hash = None
            tsp_scale.used_global_vals = {}

    def test_jit_constexpr_rekey(self) -> None:
        objs = self._table_rows()
        fake = "f" * 64
        out = [objs[0]]
        for r in objs[1:]:
            s = json.dumps(r).replace(_tsp_act.cache_key, fake)
            out.append(json.loads(s))
        p = self._write_table(out, "rekey.jsonl")
        c = self._replay(p, "rekey")
        self.assertEqual((c["rekeyed"], c["errors"], c["unresolved"]), (1, 0, 0), c)
        calls = self._counting()
        _calls_post()
        self.assertEqual(calls, [])

    def test_missing_jit_constexpr_callee_is_stale(self) -> None:
        objs = self._table_rows()
        out = [objs[0]] + [json.loads(json.dumps(r).replace('"qualname": "_tsp_act"',
                                                            '"qualname": "_tsp_gone"'))
                           for r in objs[1:]]
        c = self._replay(self._write_table(out, "gone.jsonl"), "gone")
        self.assertEqual((c["unresolved_by"], c["errors"], c["compiled"]), ({"missing": 1}, 0, 6), c)

    def test_replay_from_env_logs_contract_line(self) -> None:
        self._fresh(os.path.join(self.dir, f"c_env_{time.monotonic_ns()}"))
        env = {"TRITON_SPEC_REPLAY": os.path.join(self.dir, "table.jsonl"),
               "TRITON_SPEC_REPLAY_PHASES": "bogus"}
        with mock.patch.dict(os.environ, env), mock.patch.object(T, "_prime", lambda k: None), \
                self.assertLogs(T.logger, level="INFO") as cm:
            c = T.replay_from_env(rank=1, local_rank=0, local_world=1, device=0)
        self.assertEqual((c["armed"], c["phases"], c["rows"], c["primed"], c["errors"]),
                         (True, "post", 5, 5, 0), c)
        self.assertTrue(any("TRITON_SPEC_REPLAY_PHASES='bogus'" in m for m in cm.output))
        armed = [m for m in cm.output if "armed OK: [triton-spec] replay table=" in m]
        self.assertEqual(len(armed), 1, cm.output)
        self.assertRegex(armed[0], r"rows=5 resolved=5 unresolved=0 compiled=5 primed=5 errors=0 "
                                   r"in [0-9.]+s \(A [0-9.]+s, B [0-9.]+s\) rank=1$")
        self.assertTrue(any("[triton-spec] replay detail: " in m and " slowest=" in m
                            for m in cm.output), cm.output)

    def test_two_local_ranks_shard_and_barrier(self) -> None:
        table = os.path.join(self.dir, "table.jsonl")
        cache = os.path.join(self.dir, f"c_shared_{time.monotonic_ns()}")
        tok = f"tok{time.monotonic_ns()}"
        res: Dict[int, Dict[str, Any]] = {}
        for lr in (1, 0):  # rank 1 waits for rank 0 and times out; rank 0 finds it arrived
            self._fresh(cache)  # one disk cache, as in the container
            res[lr] = T.replay(table, phases="all", rank=lr, local_rank=lr, local_world=2,
                               device=0, prime=False, log=False, barrier_dir=self.dir,
                               barrier_token=tok, barrier_timeout_s=0.3)
        _, rows, _ = T.load_table(table)
        shards = T.lpt_shard(rows, 2)
        self.assertEqual((res[1]["shard"], res[0]["shard"]), (len(shards[1]), len(shards[0])))
        self.assertEqual(len(shards[0]) + len(shards[1]), 7)
        self.assertEqual((res[1]["barrier"], res[0]["barrier"]), ("timeout", "ok"))
        for lr in (0, 1):
            self.assertEqual((res[lr]["compiled"], res[lr]["errors"], res[lr]["unresolved"]),
                             (7, 0, 0), res[lr])

    def test_other_triton_version_and_bad_schema_not_armed(self) -> None:
        objs = self._table_rows()
        p = self._write_table([objs[0]] + [dict(r, triton_version="0.0.0") for r in objs[1:]],
                              "oldver.jsonl")
        c = self._replay(p, "ver")
        self.assertFalse(c["armed"])
        self.assertIn("rows_for_other_triton=7", c["reason"])
        p = self._write_table([dict(objs[0], schema=1)] + objs[1:], "schema.jsonl")
        c = self._replay(p, "schema")
        self.assertFalse(c["armed"])
        self.assertIn("schema", c["reason"])

    def _foreign_table(self) -> Tuple[str, str]:
        """This build's table, as the OTHER verified version would have recorded it
        (translate_row + translate_attrs): (path, other version)."""
        mine = triton.__version__
        if T._mm(mine) not in T._XVER:
            self.skipTest(f"Triton {mine} has no cross-version rules")
        other = "3.9.0" if T._mm(mine) == "3.6" else "3.6.0"
        objs = self._table_rows()
        out = [dict(objs[0], triton_versions=[other])]
        for r in objs[1:]:
            t = T.translate_row(r, other, {"fpsan_homomorphic_casts": False})
            t.pop("xver_from")
            cx = {tuple(p): e.get("v") for p, e in r["constexprs"] if e.get("t") == "str"}
            t["attrs"] = [[list(p), v] for p, v in T.translate_attrs(
                r["signature"], [(tuple(p), v) for p, v in r["attrs"]], cx, mine, other,
                _parse_attr)]
            out.append(t)
        return self._write_table(out, f"foreign_{other}.jsonl"), other

    def test_cross_version_rows_are_l1_hits_with_identical_binaries(self) -> None:
        p, other = self._foreign_table()
        c = self._replay(p, "xver", xver=True)
        self.assertTrue(c["armed"], c)
        self.assertEqual((c["rows"], c["xver"], c["compiled"], c["xver_compiled"], c["errors"],
                          c["unresolved"], c["xver_from"]), (7, 7, 7, 7, 0, 0, [other]), c)
        self.assertEqual((c["hash_known"], c["src_changed"], c["opts_drift"]), (0, 0, 0), c)
        replayed = {k: kern.asm["hsaco"] for f in _KERNELS
                    for k, kern in f.device_caches[0][0].items()}
        self.assertEqual(set(replayed), set(self.hsaco))  # this build's own L1 keys
        for k, blob in replayed.items():
            self.assertEqual(blob, self.hsaco[k], k)
        calls = self._counting()
        _calls_pre()
        _calls_post()
        self.assertEqual(calls, [])

    def test_cross_version_off_is_the_old_behaviour(self) -> None:
        p, _ = self._foreign_table()
        c = self._replay(p, "xver_off", xver=False)
        self.assertFalse(c["armed"])
        self.assertIn("rows_for_other_triton=7", c["reason"])
        with mock.patch.dict(os.environ, {"TRITON_SPEC_REPLAY_XVER": "0"}):
            c = self._replay(p, "xver_env_off")
        self.assertFalse(c["armed"])


@unittest.skipUnless(HAVE_TRITON, "triton not importable")
class TritonGlobalsGuardTest(unittest.TestCase):
    """A global changed between the replay and the first launch (recheck_globals):
    a @triton.jit helper's global (without the guard: 3.8 merges it into the
    kernel's frozen globals -> RuntimeError at launch; 3.6 merges it only when the
    helper was hashed first, jit.py I:99-111 -> a silent L1 hit on the binary built
    with the old value) and a JIT-callable constexpr's global (never re-checked at
    launch: without the guard a stale L1 hit). Both must compile at first use."""

    @classmethod
    def setUpClass(cls) -> None:
        why = _cpu_compile_reason()
        if why:
            raise unittest.SkipTest(f"this image cannot compile gfx950 kernels on the CPU: {why}")
        from triton.runtime.jit import JITFunction

        cls.JITFunction = JITFunction
        cls.td = tempfile.TemporaryDirectory()
        cls.env_patch = mock.patch.dict(os.environ, {})
        cls.env_patch.start()
        cls.run_patch = mock.patch.object(JITFunction, "run", _emu_run)
        cls.run_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        T._reset_for_tests()
        for f in _GKERNELS:
            f.device_caches.clear()
            f.hash = None
            f.used_global_vals = {}
        cls.run_patch.stop()
        cls.env_patch.stop()
        cls.td.cleanup()

    def _fresh(self, tag: str) -> None:
        T._reset_for_tests()
        os.environ["TRITON_CACHE_DIR"] = os.path.join(self.td.name, f"c_{tag}_{time.monotonic_ns()}")
        for f in _GKERNELS:
            f.device_caches.clear()
            f.hash = None
            f.used_global_vals = {}
            f.device_caches[0] = _cpu_binder(f)

    def _table(self, tag: str, call: Any) -> str:
        self._fresh(tag + "_rec")
        rec = os.path.join(self.td.name, f"rec_{tag}")
        self.assertTrue(T.install_recorder(rec, rank=0, local_rank=0))
        T.mark_run_start()
        call()
        T._reset_for_tests()
        table = os.path.join(self.td.name, f"table_{tag}.jsonl")
        st = T.merge(table, [os.path.join(rec, x) for x in os.listdir(rec)])
        self.assertEqual(st["rows"], 1, st)
        self._fresh(tag + "_rep")
        c = T.replay(table, phases="all", rank=0, local_rank=0, local_world=1, device=0,
                     prime=False, log=False)
        self.assertEqual((c["compiled"], c["errors"], c["unresolved"]), (1, 0, 0), c)
        return table

    def _counting(self) -> List[str]:
        calls: List[str] = []
        orig = self.JITFunction._do_compile

        def counted(fn: Any, *a: Any, **k: Any) -> Any:
            calls.append(fn.__name__)
            return orig(fn, *a, **k)

        p = mock.patch.object(self.JITFunction, "_do_compile", counted)
        p.start()
        self.addCleanup(p.stop)
        return calls

    def test_helper_global_changed_after_replay(self) -> None:
        mod = sys.modules[__name__]
        x, y = _FT(), _FT(2 << 20)
        self._table("help", lambda: tsp_gk[(1,)](x, y, 1024, BLOCK=64))
        self.assertTrue(any(o is _tsp_ghelp for o, _ in T._STATE.guarded))  # helper tracked
        old, old_hash = mod.GAIN, _tsp_ghelp.hash
        try:
            mod.GAIN = tl.constexpr(3.0)
            self.assertEqual(T.recheck_globals("run_start"), 2)  # helper + kernel (key moved)
            self.assertNotEqual(_tsp_ghelp.hash, old_hash)  # recomputed with the current GAIN
            self.assertEqual(T._STATE.globals_changed, 2)
            calls = self._counting()
            tsp_gk[(1,)](x, y, 1024, BLOCK=64)  # no guard: 3.8 RuntimeError, 3.6 stale L1 hit
            tsp_gk2[(1,)](x, y, 1024, BLOCK=64)  # a kernel the replay never touched
            self.assertEqual(calls, ["tsp_gk", "tsp_gk2"])
            T.mark_run_start()  # the final check already ran: only the timestamp
            self.assertEqual(T._STATE.globals_changed, 2)
        finally:
            mod.GAIN = old

    def test_constexpr_callable_global_changed_after_replay(self) -> None:
        mod = sys.modules[__name__]
        x, y = _FT(), _FT(2 << 20)
        self._table("cact", lambda: tsp_apply[(1,)](x, y, 1000, F=_tsp_gact, BLOCK=128))
        old = mod.GACT
        try:
            mod.GACT = tl.constexpr(5.0)
            calls = self._counting()
            T.mark_run_start()  # no recheck_globals("run_start") yet: the fallback runs here
            self.assertEqual(T._STATE.globals_changed, 1)  # _tsp_gact; tsp_apply reads no global
            tsp_apply[(1,)](x, y, 1000, F=_tsp_gact, BLOCK=128)  # without the guard: stale L1 hit
            self.assertEqual(calls, ["tsp_apply"])
        finally:
            mod.GACT = old

    def test_unchanged_globals_keep_the_replayed_entries(self) -> None:
        x, y = _FT(), _FT(2 << 20)
        self._table("same", lambda: tsp_gk[(1,)](x, y, 1024, BLOCK=64))
        self.assertEqual(T.recheck_globals("pre_warmup"), 0)
        self.assertEqual(T.recheck_globals("run_start"), 0)
        calls = self._counting()
        tsp_gk[(1,)](x, y, 1024, BLOCK=64)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    print(f"triton={'%s %s' % (triton.__version__, triton.__file__) if HAVE_TRITON else 'absent'} "
          f"torch={'present' if torch is not None else 'absent'}")
    unittest.main(verbosity=2)
