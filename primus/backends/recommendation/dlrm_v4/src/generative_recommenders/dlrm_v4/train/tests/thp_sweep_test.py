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

# pyre-strict
"""THP NOHUGEPAGE sweep (dlrm_v4/train/thp_sweep.py; gin apply_env_bootstrap.DLRM_HSTU_FP8_THP_NOHUGE_SWEEP).

  PARSE   smaps headers (names with spaces, [heap], [anon:x], files, shared, "(deleted)", a bad inode) and the
          Rss / AnonHugePages / VmFlags fields; other fields ignored.
  SELECT  eligible ('hg' under every policy but never; nothing under never) + private anonymous
          (inode 0; "", [heap], [anon:...]) + at least one aligned 2 MiB range + 4 KiB-mapped pages -> selected;
          all-huge -> kept; shared / file / [stack] / bad inode -> kept (counted); too small / no flag -> ignored.
  POLICY  per-size hugepages-<pmd>kB/enabled wins unless "inherit"; unreadable -> "unknown" (narrow rule).
  LIVE    on this process's real mappings: a touched MADV_HUGEPAGE region flips to 'nh' with Rss/AnonHugePages
          unchanged, an untouched one and a shared one keep 'hg'; MADV_COLLAPSE on the flipped one -> EINVAL.
  STALE   a selected range that is no longer exactly one VMA in /proc/self/maps is skipped and counted.
          The 4 KiB targets get MADV_COLD before MADV_HUGEPAGE, so khugepaged (which collapses only referenced
          pages) cannot race the test on an idle node, where the test's first MADV_HUGEPAGE wakes it at once;
          a collapse that still lands before the sweep is a skip (environment), never a pass or a failure.
  MARKER  "armed OK" (errors 0) / "PARTIAL" (errors) / "NOT armed" (exception); one line, stdout only; a print
          failure never raises; unset / "0" / "false" -> nothing at all.
  WIRING  apply_env_bootstrap declares the knob once and binds it with one _bind_bool call; gin binds it once to
          True; env wins over gin in a clean child interpreter (unset -> "1", "1" -> "1", "0" -> "0");
          train_ranker imports the module only under the env test, inside try/except, once, after the pre-start
          fork, startup preload and pre-grow (and the HIP preload, on trees that have one), and before the
          RUN_START log (and the TSPEC globals recheck, on trees that have one).
CPU only; no torch import."""

import ast
import builtins
import contextlib
import ctypes
import io
import json
import mmap
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

from generative_recommenders.dlrm_v4.train import thp_sweep as ts

try:
    import gin  # noqa: F401
except ImportError:  # pragma: no cover
    gin = None  # type: ignore[assignment]

_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(_TRAIN, "..", "..", ".."))
_GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_RANKER = os.path.join(_TRAIN, "train_ranker.py")
KNOB = "DLRM_HSTU_FP8_THP_NOHUGE_SWEEP"
MADV_COLD, MADV_HUGEPAGE, MADV_COLLAPSE = 20, 14, 25
HUGE = 2 << 20
LINE_RX = re.compile(
    r"^(armed OK|PARTIAL): \[thp-sweep\] rank=(\d+) before run_start: MADV_NOHUGEPAGE on (\d+) VMA\(s\) ([\d.]+) GiB "
    r"\(4K-mapped ([\d.]+) GiB, THP-mapped ([\d.]+) GiB\); kept hg: (\d+) THP-only VMA\(s\) ([\d.]+) GiB, "
    r"(\d+) shared/file VMA\(s\) ([\d.]+) GiB; stale (\d+); errors (\d+)( errnos=\{[^}]*\})?; smaps (\d+) VMAs "
    r"read in ([\d.]+)s, total ([\d.]+)s; thp=(always|madvise|never|unknown) pid=(\d+) boot_ns0=(\d+) "
    r"boot_ns1=(\d+)$"
)


def _hdr(lo: int, hi: int, perms: str = "rw-p", inode: str = "0", path: str = "") -> str:
    return f"{lo:x}-{hi:x} {perms} 00000000 00:00 {inode}" + (f"                 {path}" if path else "")


def _vma(lo: int, size_kb: int, rss: int, ahp: int, flags: str, perms: str = "rw-p", inode: str = "0",
         path: str = "") -> List[str]:
    hi = lo + size_kb * 1024
    return [
        _hdr(lo, hi, perms, inode, path),
        f"Size:           {size_kb} kB",
        "KernelPageSize:        4 kB",
        f"Rss:            {rss} kB",
        "Pss:                   0 kB",
        f"AnonHugePages:  {ahp} kB",
        "Swap:                  0 kB",
        f"VmFlags: {flags}",
    ]


B0 = 0x7E0000000000


def _fixture() -> List[str]:
    b = B0
    lines: List[str] = []
    lines += _vma(b + 0x00000000, 323732, 323732, 0, "rd wr mr mw me ac sd hg")            # A 4K-mapped -> sel
    lines += _vma(b + 0x20000000, 262144, 262144, 262144, "rd wr mr mw me ac sd hg")       # B full THP -> keep
    lines += _vma(b + 0x40000000, 65536, 8192, 8192, "rd wr mr mw me ac sd hg")            # C partial THP -> keep
    lines += _vma(b + 0x50000000, 65536, 8192, 4096, "rd wr mr mw me ac sd hg")            # D partial + 4K -> sel
    lines += _vma(b + 0x60000000, 65536, 65536, 0, "rd wr sh mr mw me ms sd hg", perms="rw-s")  # E shared
    lines += _vma(b + 0x70000000, 4096, 4096, 0, "rd mr mw me sd hg", perms="r--p", inode="123",
                  path="/usr/lib/my lib.so")                                                # F file
    lines += _vma(b + 0x80000000, 65536, 65536, 0, "rd wr mr mw me ac sd")                 # G no flag
    lines += _vma(b + 0x90000000, 32160, 32160, 24576, "rd wr mr mw me ac sd hg", path="[heap]")  # H heap -> sel
    lines += _vma(b + 0xA0000000, 8192, 8192, 0, "rd wr mr mw me ac sd hg", path="[anon:x y]")    # I named -> sel
    lines += _vma(b + 0xB0000000, 8192, 8192, 0, "rd wr mr mw me gd ac hg", path="[stack]")       # J stack
    lines += _vma(b + 0xC0000000, 4096, 4096, 0, "rd wr mr mw me ac sd nh")                # K nh already
    lines += _vma(b + 0xD0000000 + 4096, 2048, 2048, 0, "rd wr mr mw me ac sd hg")         # L < 1 aligned PMD
    lines += _vma(b + 0xE0000000, 8192, 8192, 0, "rd wr mr mw me ac sd hg", inode="x?")   # M bad inode -> keep
    lines += _vma(b + 0xF0000000, 8192, 8192, 0, "rd wr sh mr mw me sd hg", perms="rw-s", inode="77",
                  path="/memfd:thunk (deleted)")                                            # N deleted memfd
    return lines


def _idx(sel: List[Any]) -> List[int]:
    return sorted((v[0] - B0) >> 28 for v in sel)


class ParseSelectTest(unittest.TestCase):
    def test_parse(self) -> None:
        v = ts.parse_smaps(_fixture())
        self.assertEqual(len(v), 14)
        self.assertEqual(v[5][4], "/usr/lib/my lib.so")
        self.assertEqual(v[5][3], 123)
        self.assertEqual(v[8][4], "[anon:x y]")
        self.assertEqual(v[12][3], -1)
        self.assertEqual(v[13][4], "/memfd:thunk (deleted)")
        self.assertEqual(v[0][5:7], (323732, 0))
        self.assertIn("hg", v[0][7])

    def test_select_madvise(self) -> None:
        sel, c = ts.select(ts.parse_smaps(_fixture()))
        self.assertEqual(_idx(sel), [0x0, 0x5, 0x9, 0xA])  # A D H I
        self.assertEqual(c["sel_n"], 4)
        self.assertEqual(c["keep_thp_n"], 2)  # B C
        self.assertEqual(c["keep_other_n"], 5)  # E F J M N
        self.assertEqual(c["sel_4k_kb"], 323732 + 4096 + (32160 - 24576) + 8192)
        self.assertEqual(c["sel_thp_kb"], 4096 + 24576)

    def test_modes(self) -> None:
        v = ts.parse_smaps(_fixture())
        self.assertEqual(ts.select(v, "never")[1]["sel_n"], 0)
        self.assertEqual(_idx(ts.select(v, "unknown")[0]), [0x0, 0x5, 0x9, 0xA])  # narrow rule
        self.assertEqual(_idx(ts.select(v, "always")[0]), [0x0, 0x5, 0x9, 0xA])  # same measured 'hg' scope

    def test_policy_reader(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ts.thp_mode(d), "unknown")
            self.assertEqual(ts.pmd_size(d), 2 << 20)
            with open(os.path.join(d, "enabled"), "w") as f:
                f.write("always [madvise] never\n")
            with open(os.path.join(d, "hpage_pmd_size"), "w") as f:
                f.write("2097152\n")
            self.assertEqual(ts.thp_mode(d), "madvise")
            os.makedirs(os.path.join(d, "hugepages-2048kB"))
            with open(os.path.join(d, "hugepages-2048kB", "enabled"), "w") as f:
                f.write("always [inherit] madvise never\n")
            self.assertEqual(ts.thp_mode(d), "madvise")
            with open(os.path.join(d, "hugepages-2048kB", "enabled"), "w") as f:
                f.write("[always] inherit madvise never\n")
            self.assertEqual(ts.thp_mode(d), "always")
            with open(os.path.join(d, "enabled"), "w") as f:
                f.write("garbage\n")
            os.unlink(os.path.join(d, "hugepages-2048kB", "enabled"))
            self.assertEqual(ts.thp_mode(d), "unknown")


def _libc() -> ctypes.CDLL:
    lib = ctypes.CDLL(None, use_errno=True)
    lib.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.madvise.restype = ctypes.c_int
    return lib


def _thp_madvise_host() -> bool:
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled") as f:
            return "[madvise]" in f.read()
    except OSError:
        return False


def _region(size: int, touch: bool, shared: bool = False, touch_first: bool = False) -> Tuple[mmap.mmap, int]:
    """A 2 MiB-aligned MADV_HUGEPAGE range. touch_first: fault the pages BEFORE the advice (then MADV_COLD), so under THP
    "madvise" they are 4 KiB pages in an 'hg' VMA, the production target shape (deterministic)."""
    flags = (mmap.MAP_SHARED if shared else mmap.MAP_PRIVATE) | mmap.MAP_ANONYMOUS
    m = mmap.mmap(-1, size + HUGE, flags=flags)
    base = ctypes.addressof(ctypes.c_char.from_buffer(m))
    off = (-base) % HUGE
    a = base + off
    if touch_first:
        for i in range(0, size, 4096):
            m[off + i] = 1
        # Old PTEs, folios not referenced: khugepaged skips the range (SCAN_LACK_REFERENCED_PAGE), so it
        # cannot collapse the target between this advice and the sweep. Rss and VmFlags are unchanged.
        assert _libc().madvise(ctypes.c_void_p(a), size, MADV_COLD) == 0
    assert _libc().madvise(ctypes.c_void_p(a), size, MADV_HUGEPAGE) == 0
    if touch and not touch_first:
        for i in range(0, size, 4096):
            m[off + i] = 1
    return m, a


def _info(addr: int) -> Optional[Tuple[Any, ...]]:
    with open("/proc/self/smaps") as f:
        for v in ts.parse_smaps(f):
            if v[0] <= addr < v[1]:
                return v
    return None


@unittest.skipUnless(_thp_madvise_host(), "needs THP enabled=madvise for the deterministic 4 KiB target shape")
class LiveSweepTest(unittest.TestCase):
    def test_live(self) -> None:
        size = 16 << 20
        m2, a2 = _region(size, touch=False)
        m3, a3 = _region(size, touch=True, shared=True)
        m1, a1 = _region(size, touch=True, touch_first=True)  # the target last: shortest window
        before = (_info(a1), _info(a2), _info(a3))
        for b in before:
            self.assertIsNotNone(b)
            self.assertIn("hg", b[7])  # pyre-ignore[16]
        assert before[0] is not None
        if before[0][6] != 0:
            self.skipTest("khugepaged collapsed the target before the sweep (environment, not the sweep)")
        self.assertEqual((before[0][5], before[0][6]), (size // 1024, 0))  # fully 4 KiB-mapped target
        r = ts.sweep(rank=3, mode="madvise")
        self.assertEqual(r["errors"], 0, r)
        self.assertGreaterEqual(int(r["attempted"]), 1)  # pyre-ignore[6]
        i1, i2, i3 = _info(a1), _info(a2), _info(a3)
        assert i1 is not None and i2 is not None and i3 is not None
        if i1[6] > before[0][6]:  # only a collapse raises AnonHugePages; MADV_NOHUGEPAGE never collapses
            self.skipTest("khugepaged collapsed the target during the sweep (environment, not the sweep)")
        self.assertIn("nh", i1[7])
        self.assertNotIn("hg", i1[7])
        self.assertEqual(i1[5:7], before[0][5:7])  # Rss / AnonHugePages unchanged
        rc = _libc().madvise(ctypes.c_void_p(a1), size, MADV_COLLAPSE)
        self.assertEqual((rc, ctypes.get_errno()), (-1, 22))
        self.assertIn("hg", i2[7])  # untouched: no 4 KiB pages, keeps its THP fault path
        self.assertIn("hg", i3[7])  # shared: not this sweep's business
        del m1, m2, m3

    def test_stale_range_skipped(self) -> None:
        size = 16 << 20
        m1, a1 = _region(size, touch=True, touch_first=True)
        info = _info(a1)
        assert info is not None
        if info[6] != 0:
            self.skipTest("khugepaged collapsed the target before the sweep (environment, not the sweep)")
        self.assertGreater(info[5], info[6])
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            with open("/proc/self/maps") as src:
                for line in src:
                    lo, _, _hi = line.split(" ", 1)[0].partition("-")
                    if int(lo, 16) != info[0]:
                        f.write(line)
        try:
            r = ts.sweep(rank=1, mode="madvise", maps_path=f.name)
        finally:
            os.unlink(f.name)
        self.assertGreaterEqual(int(r["stale"]), 1)  # pyre-ignore[6]
        i1 = _info(a1)
        assert i1 is not None
        self.assertIn("hg", i1[7])  # skipped, not advised
        del m1


class MarkerTest(unittest.TestCase):
    def _run(self, env: Dict[str, Optional[str]]) -> Tuple[Any, str]:
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(buf):
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            r = ts.sweep_from_env(rank=5)
        return r, buf.getvalue()

    def test_off(self) -> None:
        for v in (None, "0", "false", "", "off", "no"):
            r, out = self._run({KNOB: v})
            self.assertIsNone(r, v)
            self.assertEqual(out, "", v)

    def test_on(self) -> None:
        r, out = self._run({KNOB: "1"})
        self.assertIsNotNone(r)
        lines = out.splitlines()
        self.assertEqual(len(lines), 1, out)
        m = LINE_RX.match(lines[0])
        self.assertIsNotNone(m, lines[0])
        assert m is not None
        self.assertEqual(m.group(1), "armed OK")
        self.assertEqual(m.group(12), "0")
        self.assertLessEqual(int(m.group(19)), int(m.group(20)))  # boot_ns0 <= boot_ns1

    def test_partial(self) -> None:
        fake = mock.MagicMock()
        fake.madvise.return_value = -1
        sel_vma = (0x7F0000000000, 0x7F0000400000, "rw-p", 0, "", 4096, 0, ("rd", "wr", "hg"))
        with mock.patch.object(ts, "_libc", return_value=fake), \
                mock.patch.object(ts, "parse_smaps", return_value=[sel_vma]), \
                mock.patch.object(ts, "_live_ranges", return_value={(sel_vma[0], sel_vma[1])}), \
                mock.patch.object(ts, "thp_mode", return_value="madvise"), \
                mock.patch.object(ctypes, "get_errno", return_value=12):
            r, out = self._run({KNOB: "1"})
        self.assertIsNotNone(r)
        m = LINE_RX.match(out.strip())
        self.assertIsNotNone(m, out)
        assert m is not None
        self.assertEqual((m.group(1), m.group(3), m.group(12)), ("PARTIAL", "1", "1"))
        self.assertIn("errnos={12: 1}", out)

    def test_error_never_raises(self) -> None:
        with mock.patch.object(ts, "sweep", side_effect=OSError(13, "Permission denied")):
            r, out = self._run({KNOB: "1"})
        self.assertIsNone(r)
        self.assertEqual(out.strip(), "NOT armed: [thp-sweep] rank=5: PermissionError: [Errno 13] Permission denied")

    def test_print_failure_never_raises(self) -> None:
        def boom(*_a: Any, **_k: Any) -> None:
            raise OSError(5, "EIO")
        with mock.patch.dict(os.environ, {KNOB: "1"}), mock.patch.object(builtins, "print", boom):
            self.assertIsNotNone(ts.sweep_from_env(rank=2))
            with mock.patch.object(ts, "sweep", side_effect=RuntimeError("x")):
                self.assertIsNone(ts.sweep_from_env(rank=2))


_CHILD = textwrap.dedent(
    """
    import json, os, sys
    root, gin_path = sys.argv[1], sys.argv[2]
    import gin
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    gin.parse_config_file(gin_path, skip_unknown=True)
    boot.apply_env_bootstrap()
    print("THPS_CHILD " + json.dumps({"v": os.environ.get("%s"), "torch": "torch" in sys.modules,
                                      "mod": "generative_recommenders.dlrm_v4.train.thp_sweep" in sys.modules}))
    """ % KNOB
)


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DLRM_", "TRITON_", "HSTU_", "GR_"))}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_ROOT, env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, _ROOT, _GIN], env=env, cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("THPS_CHILD ")]
    got = json.loads(out[0][len("THPS_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


class WiringTest(unittest.TestCase):
    def test_bootstrap(self) -> None:
        src = open(_BOOT).read()
        tree = ast.parse(src)
        fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap"]
        self.assertEqual(len(fn), 1)
        args = [a.arg for a in fn[0].args.args + fn[0].args.kwonlyargs]
        self.assertEqual(args.count(KNOB), 1)
        self.assertEqual(len(re.findall(r'_bind_bool\("%s", %s\)' % (KNOB, KNOB), src)), 1)

    def test_gin(self) -> None:
        txt = open(_GIN).read()
        self.assertEqual(re.findall(r"^apply_env_bootstrap\.%s\s*=\s*(\S+)\s*$" % KNOB, txt, re.M), ["True"])
        self.assertIn('"armed OK: [thp-sweep] rank=R before run_start:', txt)
        self.assertIn('"PARTIAL: [thp-sweep]"', txt)
        self.assertIn('"NOT armed: [thp-sweep]"', txt)
        self.assertIn("Kill switch: DLRM_HSTU_FP8_THP_NOHUGE_SWEEP=0", txt)

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_env_wins(self) -> None:
        for preset, want in (({}, "1"), ({KNOB: "1"}, "1"), ({KNOB: "0"}, "0")):
            rc, log, got = _run_child(preset)
            self.assertEqual(rc, 0, log)
            assert got is not None, log
            self.assertEqual(got["v"], want, (preset, log))
            self.assertFalse(got["mod"])  # the bootstrap never imports the sweep module

    def test_ranker_hook(self) -> None:
        src = open(_RANKER).read()
        call = "_thp_sweep.sweep_from_env(rank=rank)"
        self.assertEqual(src.count(call), 1)
        pos = src.index(call)
        # Anchors every tree has. The HIP preload and the TSPEC globals recheck (ship14+) are required only
        # on trees that import them (this port also targets the ship12-era WBND testbed); keyed on the import
        # line, so a renamed or dropped call on such a tree fails instead of skipping the check.
        for before in ("prestart_streaming_loaders(", "run_fp8_first_use_warmup(", "run_startup_preload(prestarted",
                       "_alloc_pregrow.pregrow_before_run_start()"):
            self.assertLess(src.index(before), pos, before)
        if "import hip_preload as _hip_preload" in src:
            self.assertLess(src.index("_hip_preload.preload_from_env("), pos)
        self.assertGreater(src.index("mlperf_logger.log_run_start("), pos)
        if "import triton_spec as _tspec" in src:
            self.assertGreater(src.index('_tspec.recheck_globals("run_start")'), pos)
        tree = ast.parse(src)
        imps = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and any(
            a.name == "thp_sweep" for a in n.names)]
        self.assertEqual(len(imps), 1)
        # the import sits in a try inside an `if` whose test reads the knob from the environment
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        node, chain = imps[0], []
        while node in parents:
            node = parents[node]
            chain.append(node)
        self.assertTrue(any(isinstance(n, ast.Try) for n in chain))
        ifs = [n for n in chain if isinstance(n, ast.If)]
        self.assertTrue(ifs and KNOB in ast.unparse(ifs[0].test) and "os.environ" in ast.unparse(ifs[0].test))

    def test_module_has_no_reversal_and_one_sink(self) -> None:
        tree = ast.parse(open(ts.__file__).read())
        names = {n.attr if isinstance(n, ast.Attribute) else n.id
                 for n in ast.walk(tree) if isinstance(n, (ast.Attribute, ast.Name))}
        self.assertNotIn("prctl", names)
        self.assertNotIn("MADV_HUGEPAGE", names)
        self.assertNotIn("logger", names)
        self.assertNotIn("logging", names)
        madv = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "madvise"]
        self.assertEqual(len(madv), 1)
        self.assertEqual(ast.unparse(madv[0].args[2]), "MADV_NOHUGEPAGE")
        self.assertEqual(ts.MADV_NOHUGEPAGE, 15)


if __name__ == "__main__":
    unittest.main()
