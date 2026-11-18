#!/usr/bin/env python3
# pyre-strict
"""DLRM_HSTU_FP8_FWD_MXFP6_RING6 (yambda_5b.gin item (96); r14 RING6; BITWISE; default off).

RING6 is a compile-time switch of the MXFP6 forward K=512 kernels (ops/hip/k512_mx6_gemm.cu, k512_mx6_u.cu:
K512MX6_RING undefined or 5 = the shipped 5-slot LDS ring, 6 = the 6-slot ring with the activation item two K steps
ahead) plus its loader plumbing (ops/hip/k512_mx6_gemm.py). CPU only: hipcc (subprocess.run) and the library loader
(ctypes.CDLL) are replaced by fakes, so nothing is compiled or launched. Pinned here:

  GIN   exactly one binding, False or True, right after item (92)'s; the (96) RING6 item block names the bound value
        ("bound False below" / "bound True below"), documents WHAT / NUMERICS / BUILD / ARMED / Must be ABSENT / the
        kill switch, stays within 80 columns and carries no measured number (a gin-on flip edits only the gin item).
  BOOT  apply_env_bootstrap declares the parameter once (Optional[bool] = None) and binds it with one env-wins
        _bind_bool; a pre-set env value wins both ways; a clean child running train_ranker's Phase 1 (parse
        yambda_5b.gin, apply_env_bootstrap) exports the bound value, and a pre-set "0" / "1" survives; the launcher
        forwards the DLRM_HSTU_FP8_ prefix (the kill switch reaches the container).
  FLAGS the loader adds -DK512MX6_RING=6 to exactly the GEMM N=1536 / N=1024 and U builds when the knob is "1"
        (read once; only "1" arms it) and changes nothing when it is off: the hipcc command lines equal the shipped
        ones element for element (pinned literally below), so does every library file name.
  KEYS  a RING6 library lives under its own key ((N, "ring6")), file name (..._ring6.so) and build directory
        (pid<n>_ring6_*): one built with one setting is never served for the other, a failure latched for one never
        latches the other; the weight shadow (library, key, cache entries) is the same for both settings.
  ROUTE both ways. Knob on: a RING6 library whose impl string does not report the RING6 build (a source without the
        switch, or an injected -DK512MX6_RING=5 after the loader's define) or reports the secondary RING6S variant is
        refused (latched, never served). Knob off: a GEMM / U library whose impl string reports RING6 at all (the
        define injected outside the loader: HIPCC_COMPILE_FLAGS_APPEND, which hipcc appends to every command, or a
        source defaulting to RING6) is refused the same way, before RUN_START, with an error naming the knob and the
        library (VQK / VK: the prebuild fails; U: U alone is latched).
  ARMED prebuild() with the knob on builds every library once, before RUN_START (later kernel() calls compile
        nothing), and prints ONE rank-tagged "[k512-mx6] RING6 ARMED: ..." line; with the knob off it prints nothing
        new. A U routing failure latches U only and the line says so. The build locks follow the library keys.
  CHECK the knob-on self-check paths of prebuild(): a failed GEMM self-check drops and latches every RING6 library
        (none is served); a failed U self-check drops and latches the RING6 U library only (the prebuilt and ARMED
        lines say so; VQK / VK stay RING6); the U self-check runs on the RING6 U library; the self-check message names
        the RING6 library.
  LDS   the per-kernel lds= of the "[k512-mx6] prebuilt:" line (each library's "LDS Size [bytes/block]" compiler
        remark, parsed by the shipped resource_summary) marks the arm: 73728 for k512_mx6_kernel (N=1024 / 1536) and
        k512_mx6_u_kernel when the knob is on, 61440 when it is off; the weight shadow's is unchanged.
  SRC   the two .cu files are GPU lab job 71320's proven sources with one comment line refreshed (sha256 pins; r14
        iter65 proved the refresh codegen-neutral on the CPU: identical preprocessed translation units and
        cuid-normalised .s for K512MX6_RING undefined / 5 / 6; a comment-only edit needs new pins plus that check, any
        other edit a new GPU gate), default the switch to 5, carry the impl tags the loader checks, and the loader
        never reaches RING6S."""

import ast
import contextlib
import hashlib
import importlib
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

# recommendation/ (this file: recommendation/generative_recommenders/dlrm_v4/train/tests/)
_REPO = pathlib.Path(__file__).resolve().parents[4]
_GR = _REPO / "generative_recommenders"
_GIN = _GR / "dlrm_v4/train/gin/yambda_5b.gin"
_BOOT = _GR / "dlrm_v4/train/_env_bootstrap.py"
_HIP = _GR / "ops/hip"
_LOADER = _HIP / "k512_mx6_gemm.py"
_GEMM_CU = _HIP / "k512_mx6_gemm.cu"
_U_CU = _HIP / "k512_mx6_u.cu"
_LAUNCHER = _REPO / "scripts/launch_slurm.sh"
_KNOB = "DLRM_HSTU_FP8_FWD_MXFP6_RING6"
_PREV_BINDING = "apply_env_bootstrap.DLRM_HSTU_FP8_FWD_MXFP6 "  # item (92)'s binding (any value)
_HEAD = "# (96) RING6 "
_DEFINE = "-DK512MX6_RING=6"
_REAL_GEMM = "generative_recommenders.ops.hip.k512_mx6_gemm"
# sha256 of the RING6 sources: lab job 71320 (GO_INMODEL, bitwise vs production for VQK, VK and U) built
# d46c8de0... / cb005dda... (r14 probes/iter64/ring6/src); these files differ from them only in one comment line each
# (the stale "lab-only until a GPU gate passes" now cites the job), proven codegen-neutral on the CPU (r14
# probes/iter65/ring6_v2: preprocessed device / host translation units and cuid-normalised .s identical for
# K512MX6_RING undefined, 5 and 6). Any further edit changes these pins: a comment-only edit needs that CPU check, any
# other edit a new GPU gate.
_SHA_GEMM_CU = "34574038d11f13343f52d0a6a323269b88bd42f5cb9cacfa01ed24284645d995"
_SHA_U_CU = "e0366b32af2ccf84ba70028f711c2d44e626ea4acad7ab203d4c78712dd86428"
_LDS = {False: 5 * 12288, True: 6 * 12288}  # per-workgroup LDS of the shipped / RING6 GEMM and U kernels (12 KiB slots)
# impl strings of the fake libraries (the tags the real k512_mx6_gemm_impl() / k512_mx6_u_impl() return; SrcTest
# checks that the .cu files carry them)
_IMPL = {
    ("gemm", False): "k512mx6 (fake): 128x256 tile, 5-slot 60 KiB LDS ring, swapped MFMA",
    ("gemm", True): "k512mx6 (fake): 128x256 tile, RING6 6-slot 72 KiB LDS ring (A two K steps ahead), swapped MFMA",
    ("u", False): "k512_mx6_u (fake): 2-WG mainloop (5 x 12 KiB LDS ring, scaled MFMA)",
    ("u", True): "k512_mx6_u (fake): 2-WG mainloop (RING6 6 x 12 KiB LDS ring, A two K steps ahead, scaled MFMA)",
    ("wshadow", False): "k512_mx6_wshadow (fake)",
}


def _try_import(name: str) -> Optional[Any]:  # pyre-ignore[3]
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - environment specific
        return None


torch = _try_import("torch")
gin = _try_import("gin")


@contextlib.contextmanager
def _capture():  # pyre-ignore[3]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def _gin_lines() -> List[str]:
    return _GIN.read_text().splitlines()


def _binding_index(lines: List[str]) -> int:
    hits = [i for i, ln in enumerate(lines) if re.match(rf"^\s*apply_env_bootstrap\.{_KNOB}\s*=", ln)]
    if len(hits) != 1:
        raise AssertionError(f"expected one {_KNOB} binding, found {len(hits)}")
    return hits[0]


# =============================================================================================== GIN
class GinItemTest(unittest.TestCase):
    def test_gin_binds_once_with_item_text(self) -> None:
        # Exactly one boolean binding, and the item text names the bound value ("bound False below" / "bound True
        # below"): the gin-on flip edits the gin item only (it never needs to edit this test).
        lines = _gin_lines()
        i_bind = _binding_index(lines)
        bound = re.fullmatch(rf"apply_env_bootstrap\.{_KNOB} = (True|False)", lines[i_bind])
        self.assertIsNotNone(bound, lines[i_bind])
        assert bound is not None
        names = [m.group(1) for m in (re.match(r"^\s*apply_env_bootstrap\.(\w+)\s*=", ln) for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        i_head = max(i for i in range(i_bind) if lines[i].startswith(_HEAD))
        block = lines[i_head:i_bind]
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = " ".join(" ".join(ln.lstrip("#").split()) for ln in block)
        self.assertIn(f"bound {bound.group(1)} below", text)
        self.assertNotIn(f"bound {'False' if bound.group(1) == 'True' else 'True'} below", text)
        for marker in ("(96) RING6", "item (92)", "WHAT:", "NUMERICS: BITWISE", "BUILD:", "Code default OFF",
                       "ops/hip/k512_mx6_gemm.cu", "ops/hip/k512_mx6_u.cu", "ops/hip/k512_mx6_gemm.py",
                       "K512MX6_RING", _DEFINE, "N=1536", "N=1024", "U", "libk512_mx6_wshadow.so",
                       "before RUN_START", "no compile after RUN_START", "INERT",
                       "ARMED (every rank, once, before RUN_START)", '"[k512-mx6] RING6 ARMED:', "[rank N]",
                       "libk512_mx6_gemm_N1024_ring6.so, libk512_mx6_gemm_N1536_ring6.so, libk512_mx6_u_ring6.so",
                       "Must be ABSENT", '"RING6 routing check failed"', '"U N=512 not RING6"',
                       '"[k512-mx6] prebuild FAILED"', "Only the value 1 arms it", "read once per process",
                       f"Kill switch: env {_KNOB}=0", "pre-set env value wins over this binding",
                       'INERT unless item (92) arms the MXFP6 forward: no build and no "[k512-mx6]" line',
                       f'"env bootstrap: ... {_KNOB}=<v>"', "Must be ABSENT (both settings)",
                       'ARM MARKER (both settings): the per-kernel lds= of the "[k512-mx6] prebuilt:" line',
                       f"lds={_LDS[True]} (six 12 KiB slots) for k512_mx6_kernel (N=1024 and N=1536) and "
                       f"k512_mx6_u_kernel (N=512) when armed, lds={_LDS[False]} (five) when off",
                       'Off (binding False, or the kill switch): no "[k512-mx6]" line names RING6',
                       "A GEMM or U library that reports RING6 while off", "HIPCC_COMPILE_FLAGS_APPEND",
                       "fails the routing check: fail closed"):
            self.assertIn(marker, text, marker)
        # absence wording is scoped to [k512-mx6] lines: _bind_bool logs "env bootstrap: ...RING6=<v>" in every run
        for unscoped in ("INERT (no line", "no RING6 line"):
            self.assertNotIn(unscoped, text, unscoped)
        # no measured number in gin text (timings, ratios, percentages, register counts live in the evidence)
        self.assertIsNone(re.search(r"\d\.\d|%|\bms\b|\bVGPR", text), text)

    def test_item_follows_item_92(self) -> None:
        lines = _gin_lines()
        i_bind = _binding_index(lines)
        i_head = max(i for i in range(i_bind) if lines[i].startswith(_HEAD))
        self.assertTrue(lines[i_head - 1].startswith("# ----"), lines[i_head - 1])
        self.assertTrue(lines[i_head - 2].startswith(_PREV_BINDING), lines[i_head - 2])
        self.assertTrue(lines[i_bind + 1].startswith("#"), lines[i_bind + 1])


# =============================================================================================== BOOT
_CHILD = textwrap.dedent(
    """
    import json, os, sys
    root, gin_path = sys.argv[1], sys.argv[2]
    import gin
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
    boot.apply_env_bootstrap()
    print("RING6_GIN_CHILD " + json.dumps({"env": {k: v for k, v in os.environ.items() if "MXFP6" in k},
                                           "torch_imported": "torch" in sys.modules}, sort_keys=True))
    """
)


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DLRM_", "TRITON_", "HSTU_", "GR_"))}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_REPO), env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, str(_REPO), str(_GIN)], env=env, cwd=str(_REPO),
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("RING6_GIN_CHILD ")]
    got = json.loads(out[0][len("RING6_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


class BootstrapTest(unittest.TestCase):
    def test_one_parameter_one_env_wins_bind(self) -> None:
        tree = ast.parse(_BOOT.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(names.count(_KNOB), 1)
        arg = next(a for a in fn.args.args if a.arg == _KNOB)
        self.assertEqual(ast.unparse(arg.annotation) if arg.annotation else None, "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[_KNOB], ast.Constant) and defaults[_KNOB].value is None)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == _KNOB]
        self.assertEqual(len(calls), 1)
        (call,) = calls
        self.assertEqual(getattr(call.func, "id", None), "_bind_bool")  # env wins; True -> "1"
        self.assertEqual([getattr(a, "id", None) for a in call.args[1:]], [_KNOB])

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_bootstrap_exports_and_env_wins(self) -> None:
        import inspect

        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

        self.assertIsNone(inspect.signature(eb.apply_env_bootstrap).parameters[_KNOB].default)
        cases = [  # (pre-set env, bound value, expected env after)
            (None, None, None),
            (None, False, "0"),
            (None, True, "1"),
            ("1", False, "1"),  # a pre-set launcher value wins over gin
            ("0", True, "0"),  # the kill switch
        ]
        for preset, bound, want in cases:
            env = {} if preset is None else {_KNOB: preset}
            with mock.patch.dict(os.environ, env, clear=True):
                eb.apply_env_bootstrap(**{_KNOB: bound})
                self.assertEqual(os.environ.get(_KNOB), want, (preset, bound))

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_gin_run_exports_the_binding_and_env_wins(self) -> None:
        lines = _gin_lines()
        bound = lines[_binding_index(lines)].split("=")[1].strip()
        want = {"True": "1", "False": "0"}[bound]
        for preset, expect in (({}, want), ({_KNOB: "0"}, "0"), ({_KNOB: "1"}, "1")):
            rc, log, got = _run_child(preset)
            self.assertEqual(rc, 0, log)
            assert got is not None, log
            self.assertEqual(got["env"].get(_KNOB), expect, preset)
            self.assertFalse(got["torch_imported"])

    def test_launcher_forwards_the_prefix(self) -> None:
        # the kill switch reaches the container: launch_slurm.sh forwards every DLRM_HSTU_FP8_* variable the caller set
        if not _LAUNCHER.is_file():
            self.skipTest("no scripts/launch_slurm.sh in this checkout")
        self.assertTrue(_KNOB.startswith("DLRM_HSTU_FP8_"))
        self.assertIn('for v in "${!DLRM_HSTU_FP8_@}"; do', _LAUNCHER.read_text())


# =============================================================================================== loader fakes
class _FakeFn:
    def __init__(self, result: Any) -> None:  # pyre-ignore[2]
        self.result = result
        self.restype: Any = None
        self.argtypes: Any = None

    def __call__(self, *args: Any) -> Any:  # pyre-ignore[2,3]
        return self.result


class _FakeLib:
    """What the fake ctypes.CDLL returns: the entry symbol and the impl string the fake build wrote."""

    def __init__(self, path: str) -> None:
        meta = json.loads(pathlib.Path(path).read_text())
        self.syms: Dict[str, _FakeFn] = {meta["entry"]: _FakeFn(0), meta["impl_symbol"]: _FakeFn(meta["impl"].encode())}

    def __getattr__(self, name: str) -> Any:  # pyre-ignore[3]
        syms = self.__dict__.get("syms", {})
        if name in syms:
            return syms[name]
        raise AttributeError(name)


def _remarks(source: pathlib.Path, kernels: List[Tuple[str, int]]) -> str:
    """The -Rpass-analysis=kernel-resource-usage remarks hipcc prints while building a library, in the real format
    (register numbers are placeholders; the LDS size is what the "[k512-mx6] prebuilt:" line reports as lds=)."""
    tag = " [-Rpass-analysis=kernel-resource-usage]"
    lines = []
    for name, lds in kernels:
        at = f"{source}:1:1: remark:"
        lines.append(f"{at} Function Name: {name}{tag}")
        for field, value in (("TotalSGPRs", 32), ("VGPRs", 128), ("AGPRs", 0), ("ScratchSize [bytes/lane]", 0),
                             ("Dynamic Stack", "False"), ("Occupancy [waves/SIMD]", 2), ("SGPRs Spill", 0),
                             ("VGPRs Spill", 0), ("LDS Size [bytes/block]", lds)):
            lines.append(f"{at}     {field}: {value}{tag}")
    return "\n".join(lines) + "\n"


class _LoaderCase(unittest.TestCase):
    """The real k512_mx6_gemm with hipcc and ctypes.CDLL faked: every build writes a small JSON 'library' naming its
    entry symbol and impl string and prints the kernel-resource-usage remarks (LDS 73728 for a RING6 GEMM / U build,
    61440 for a shipped one). Like hipcc, the fake appends $HIPCC_COMPILE_FLAGS_APPEND to the command and the last
    -DK512MX6_RING= wins: a build with =6 reports the RING6 impl unless the case simulates a source without the switch
    (honours=False); default_ring6_fn simulates a source that defaults to RING6; spin=True the RING6S variant."""

    def setUp(self) -> None:
        if torch is None:
            self.skipTest("torch not installed")
        self.G = G = importlib.import_module(_REAL_GEMM)
        self.commands: List[List[str]] = []
        self.honours = True  # the source honours the switch (a RING6 build reports RING6)
        self.honours_fn = lambda command: self.honours  # per build (prebuild builds in parallel threads)
        self.default_ring6_fn = lambda command: False  # the source defaults to RING6 (e.g. a *_SRC override)
        self.spin = False
        self.tmp = tempfile.TemporaryDirectory(prefix="mx6_loader_test_")
        self.addCleanup(self.tmp.cleanup)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for patch in (
            mock.patch.object(G, "_BUILD_ROOT", pathlib.Path(self.tmp.name)),
            mock.patch.object(G, "_build_dir", None),
            mock.patch.dict(G._kernels, clear=True),
            mock.patch.dict(G._failures, clear=True),
            mock.patch.dict(G._key_locks, clear=True),
            mock.patch.object(G, "_prebuilt", False),
            mock.patch.object(G, "_ring6", None),
            mock.patch.object(G, "_ring6_override", None),
            mock.patch.object(G, "_ring6_logged", False),
            mock.patch.object(G.subprocess, "run", self._fake_run),
            mock.patch.object(G.ctypes, "CDLL", _FakeLib),
            mock.patch.dict(os.environ, {"RANK": "0"}),
        ):
            stack.enter_context(patch)
        os.environ.pop(_KNOB, None)
        os.environ.pop("DLRM_HSTU_FP8_FWD_MXFP6_GEMM_SRC", None)
        os.environ.pop("HIPCC_COMPILE_FLAGS_APPEND", None)
        # item (110) BSR: off in these cases whatever the environment says (mx6_bsr_cpu_test pins BSR)
        os.environ.pop("DLRM_HSTU_FP8_FWD_MXFP6_BSR", None)
        if hasattr(G, "_bsr"):
            stack.enter_context(mock.patch.object(G, "_bsr", None))

    def _fake_run(self, command: List[str], **kwargs: Any) -> Any:  # pyre-ignore[2,3]
        self.commands.append(list(command))
        out = pathlib.Path(command[command.index("-o") + 1])
        source = pathlib.Path(command[1])
        flags = list(command) + os.environ.get("HIPCC_COMPILE_FLAGS_APPEND", "").split()  # what hipcc compiles with
        rings = [a for a in flags if a.startswith("-DK512MX6_RING=")]
        ring6 = bool(rings and rings[-1] == _DEFINE and self.honours_fn(command)) or self.default_ring6_fn(command)
        lds = _LDS[ring6]
        if "-DK512_MX6_WSHADOW" in command:
            meta = {"entry": "k512_mx6_wshadow", "impl_symbol": "k512_mx6_wshadow_impl",
                    "impl": _IMPL[("wshadow", False)]}
            kernels = [(f"_ZN8k512mx6w23k512_mx6_wshadow_kernelILb{b}EEEvPKvxxiiiiiiiiiPhS3_S3_Pf", 24) for b in (0, 1)]
        elif source.name == "k512_mx6_u.cu":
            meta = {"entry": "k512_mx6_u_bf16_bias", "impl_symbol": "k512_mx6_u_impl", "impl": _IMPL[("u", ring6)]}
            kernels = [("_Z17k512_mx6_u_kernelILi1ELi0ELi0ELi1EEvPKhS1_S1_S1_PtPKtPKfS6_i", lds)]
        else:
            n = next(int(a.split("=")[1]) for a in command if a.startswith("-DHK_N="))
            meta = {"entry": f"k512_mx6_e4m3_bias_N{n}", "impl_symbol": "k512_mx6_gemm_impl",
                    "impl": _IMPL[("gemm", ring6)]}
            kernels = [(f"_Z15k512_mx6_kernelILi{n}EEvPKhS1_PhPKtPKfS6_S1_S1_i", lds)]
        if ring6 and self.spin:
            meta["impl"] = meta["impl"].replace("ahead", "ahead; RING6S: scale loads pinned")
        out.write_text(json.dumps(meta))
        return subprocess.CompletedProcess(command, 0, stdout=_remarks(source, kernels))

    def set_knob(self, value: Optional[str]) -> None:
        """The environment as the bootstrap leaves it, read afresh (the loader reads it once per process)."""
        if value is None:
            os.environ.pop(_KNOB, None)
        else:
            os.environ[_KNOB] = value
        self.G.set_ring6_for_testing(None)

    def shipped_command(self, key: Any, output: pathlib.Path) -> List[str]:  # pyre-ignore[2]
        """The hipcc command of the shipped loader (ship15.1 k512_mx6_gemm.py, ba1cb37c), element for element."""
        G = self.G
        if key == G.WSHADOW_KEY:
            source, defines = G.source_path(), ["-DK512_MX6_WSHADOW"]
        elif key == G.U_N:
            source, defines = G.u_source_path(), []
        else:
            source, defines = G.source_path(), [f"-DHK_N={key}", "-DHK_K=512"]
        return ["/opt/rocm/bin/hipcc", str(source), "--offload-arch=gfx950", "-std=c++20", "-w", "-O3",
                "-ffp-contract=off", "-I/opt/rocm/include/hip", *defines, "-mllvm",
                "-amdgpu-sched-strategy=iterative-ilp", "-shared", "-fPIC", "-Rpass-analysis=kernel-resource-usage",
                "-o", str(output)]

    def build_all(self) -> Dict[Any, List[str]]:  # pyre-ignore[3]
        """_build every library key once; {key: hipcc command}."""
        G = self.G
        got: Dict[Any, List[str]] = {}
        for key in (1536, 1024, G.WSHADOW_KEY) + ((G.U_N,) if G.U_N in G.SUPPORTED_N else ()):
            n0 = len(self.commands)
            G._build(key)
            self.assertEqual(len(self.commands), n0 + 1)
            got[key] = self.commands[-1]
        return got


# =============================================================================================== FLAGS
class LoaderFlagsTest(_LoaderCase):
    def test_names(self) -> None:
        G = self.G
        self.assertEqual(G.RING6_KNOB, _KNOB)
        self.assertEqual(G.RING6_DEFINE, _DEFINE)
        self.assertEqual(G.KNOB + "_RING6", _KNOB)
        self.assertEqual(G.U_N in G.SUPPORTED_N, _U_CU.is_file())

    def test_only_one_arms_and_absent_is_off(self) -> None:
        for value, want in (("1", True), ("0", False), ("", False), ("true", False), ("True", False), (" 1", False),
                            ("2", False), (None, False)):
            self.set_knob(value)
            self.assertEqual(self.G.ring6_requested(), want, value)

    def test_read_once(self) -> None:
        self.set_knob("1")
        self.assertTrue(self.G.ring6_requested())
        os.environ[_KNOB] = "0"  # a later change of the environment does not flip a running process
        self.assertTrue(self.G.ring6_requested())
        self.assertEqual(self.G._lib_key(1536), (1536, "ring6"))

    def test_knob_off_commands_are_the_shipped_ones(self) -> None:
        G = self.G
        for value in (None, "0"):
            self.set_knob(value)
            with mock.patch.object(G, "_build_dir", None):
                got = self.build_all()
                out_dir = G._process_build_dir()
                self.assertTrue(out_dir.name.startswith(f"pid{os.getpid()}_"), out_dir.name)
                self.assertNotIn("ring6", out_dir.name)
            names = {1536: "libk512_mx6_gemm_N1536.so", 1024: "libk512_mx6_gemm_N1024.so",
                     G.WSHADOW_KEY: "libk512_mx6_wshadow.so", G.U_N: "libk512_mx6_u.so"}
            for key, command in got.items():
                tmp = out_dir / f".{names[key]}.{os.getpid()}.tmp.so"
                self.assertEqual(command, self.shipped_command(key, tmp), (value, key))
                self.assertTrue((out_dir / names[key]).is_file(), names[key])
                self.assertFalse([a for a in command if "K512MX6" in a], command)

    def test_knob_on_adds_the_define_to_exactly_the_gemm_and_u_builds(self) -> None:
        G = self.G
        self.set_knob("1")
        got = self.build_all()
        out_dir = G._process_build_dir()
        self.assertTrue(out_dir.name.startswith(f"pid{os.getpid()}_ring6_"), out_dir.name)
        names = {1536: "libk512_mx6_gemm_N1536_ring6.so", 1024: "libk512_mx6_gemm_N1024_ring6.so",
                 G.WSHADOW_KEY: "libk512_mx6_wshadow.so", G.U_N: "libk512_mx6_u_ring6.so"}
        with_define = set()
        for key, command in got.items():
            tmp = out_dir / f".{names[key]}.{os.getpid()}.tmp.so"
            shipped = self.shipped_command(key, tmp)
            if key == G.WSHADOW_KEY:
                self.assertEqual(command, shipped)  # the weight shadow is the shipped build
            else:
                # the shipped command with -DK512MX6_RING=6 as hipcc_command's extra flag (right before -o)
                self.assertEqual(command, shipped[:-2] + [_DEFINE] + shipped[-2:], key)
            if _DEFINE in command:
                with_define.add(names[key])
            self.assertEqual(command.count(_DEFINE), 1 if key != G.WSHADOW_KEY else 0)
            self.assertFalse([a for a in command if "R6_SPIN" in a or "K512MX6_RING=5" in a], command)
            self.assertTrue((out_dir / names[key]).is_file(), names[key])
        self.assertEqual(with_define, {"libk512_mx6_gemm_N1536_ring6.so", "libk512_mx6_gemm_N1024_ring6.so",
                                       "libk512_mx6_u_ring6.so"})


# =============================================================================================== KEYS / ROUTE
class KeySeparationTest(_LoaderCase):
    def test_a_library_is_never_served_for_the_other_setting(self) -> None:
        G = self.G
        self.set_knob("0")
        prod = G.kernel(1536)
        self.assertFalse(prod.ring6)
        self.assertNotIn("RING6", prod.impl)
        self.assertEqual(sorted(map(str, G._kernels)), ["1536"])
        self.set_knob("1")
        ring6 = G.kernel(1536)  # a fresh RING6 build, not the shipped library
        self.assertEqual(len(self.commands), 2)
        self.assertIsNot(ring6, prod)
        self.assertTrue(ring6.ring6)
        self.assertIn("RING6 6-slot 72 KiB LDS ring", ring6.impl)
        self.assertEqual((ring6.key, ring6.n), (1536, 1536))
        self.assertTrue(ring6.path.endswith("/libk512_mx6_gemm_N1536_ring6.so"), ring6.path)
        self.assertTrue(prod.path.endswith("/libk512_mx6_gemm_N1536.so"), prod.path)
        self.assertEqual(set(G._kernels), {1536, (1536, "ring6")})
        self.assertIs(G.kernel(1536), ring6)  # served from the table: no rebuild
        self.set_knob("0")
        self.assertIs(G.kernel(1536), prod)
        self.assertEqual(len(self.commands), 2)

    def test_failure_latches_are_separate(self) -> None:
        G = self.G
        self.honours = False  # a source without the switch: the define has no effect
        self.set_knob("1")
        with self.assertRaisesRegex(RuntimeError, "RING6 routing check failed: libk512_mx6_gemm_N1024_ring6.so"):
            G.kernel(1024)
        self.assertIn("RING6 routing check failed", G.failure(1024) or "")
        self.assertNotIn((1024, "ring6"), G._kernels)
        with self.assertRaisesRegex(RuntimeError, "RING6 routing check failed"):
            G.kernel(1024)  # latched: no second build
        self.assertEqual(len(self.commands), 1)
        self.set_knob("0")
        self.assertIsNone(G.failure(1024))  # the shipped library is not latched by the RING6 failure
        self.assertFalse(G.kernel(1024).ring6)
        self.assertEqual(len(self.commands), 2)

    def test_ring6s_is_refused(self) -> None:
        G = self.G
        if G.U_N not in G.SUPPORTED_N:
            self.skipTest("no k512_mx6_u.cu in this tree")
        self.spin = True
        self.set_knob("1")
        with self.assertRaisesRegex(RuntimeError, "RING6 routing check failed: libk512_mx6_u_ring6.so"):
            G.kernel(G.U_N)
        self.assertIn("RING6S: scale loads pinned", G.failure(G.U_N) or "")
        with self.assertRaisesRegex(RuntimeError, "RING6 routing check failed: libk512_mx6_gemm_N1536_ring6.so"):
            G.kernel(1536)

    def test_weight_shadow_is_shared(self) -> None:
        G = self.G
        specs = {}
        for value in ("0", "1"):
            self.set_knob(value)
            self.assertEqual(G._lib_key(G.WSHADOW_KEY), G.WSHADOW_KEY)
            self.assertEqual(G._extra_flags(G.WSHADOW_KEY), [])
            specs[value] = G._spec(G.WSHADOW_KEY)
        self.assertEqual(specs["0"], specs["1"])
        self.set_knob("0")
        shadow = G.wshadow_kernel()
        self.set_knob("1")
        self.assertIs(G.wshadow_kernel(), shadow)  # the same library serves both settings
        self.assertEqual(len(self.commands), 1)

    def test_shadow_cache_entries_serve_both_settings(self) -> None:
        # The shadow cache (ops/tests/k512_mx6_gemm_cache_test.py) holds the weight-shadow library's outputs; that
        # library and the GEMM operand formats do not change with the knob, so an entry stays valid across it.
        G = self.G
        calls: List[int] = []

        def fake_refresh(weight: Any, layout: Any, **kw: Any) -> Any:  # pyre-ignore[2,3]
            calls.append(id(weight))
            return G.mx6_weight_reference(weight, layout)

        w = torch.nn.Parameter(torch.randn((512, 2048), generator=torch.Generator().manual_seed(3)) * 0.02)
        G.clear_shadow_cache()
        self.addCleanup(G.clear_shadow_cache)
        with mock.patch.object(G, "refresh_weight_shadow", new=fake_refresh):
            self.set_knob("0")
            a = G.mx6_weight(w, ((512, 2048),))
            self.set_knob("1")
            b = G.mx6_weight(w, ((512, 2048),))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(x is y for x, y in zip(a, b)))


# =============================================================================================== ARMED
class _PrebuildCase(_LoaderCase):
    """_LoaderCase plus one fake gfx950 device, with both GPU self-checks stubbed to pass (a test may patch them
    again to fail)."""

    def setUp(self) -> None:
        super().setUp()
        G = self.G
        props = types.SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        self.real_selfcheck_gemm = G._selfcheck_gemm  # the loader's own (stubbed below)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for patch in (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_properties", return_value=props),
            mock.patch.object(G, "_selfcheck_shadow", lambda device: None),
            mock.patch.object(G, "_selfcheck_gemm", lambda device, n, rows=257: None),
        ):
            stack.enter_context(patch)

    def prebuild(self) -> str:
        with _capture() as out:
            self.assertTrue(self.G.prebuild())
        return out.getvalue()

    def prebuild_raw(self) -> Tuple[str, Optional[Exception]]:
        """prebuild() that may raise PrebuildError: (stdout, the error or None)."""
        error: Optional[Exception] = None
        with _capture() as out:
            try:
                self.G.prebuild()
            except self.G.PrebuildError as e:
                error = e
        return out.getvalue(), error

    def line(self, text: str, head: str) -> str:
        """The one stdout line starting with `head`."""
        hits = [ln for ln in text.splitlines() if ln.startswith(head)]
        self.assertEqual(len(hits), 1, (head, text))
        return hits[0]

    @contextlib.contextmanager
    def fresh_tables(self):  # pyre-ignore[3]
        """A further prebuild within one test: an empty library table, latches, build locks and build directory."""
        G = self.G
        with mock.patch.dict(G._kernels, clear=True), mock.patch.dict(G._failures, clear=True), \
                mock.patch.dict(G._key_locks, clear=True), mock.patch.object(G, "_build_dir", None):
            yield


class PrebuildTest(_PrebuildCase):

    def test_knob_on_builds_before_run_start_and_arms_once(self) -> None:
        G = self.G
        self.set_knob("1")
        text = self.prebuild()
        keys = [1024, 1536, G.WSHADOW_KEY] + ([G.U_N] if G.U_N in G.SUPPORTED_N else [])
        self.assertEqual(len(self.commands), len(keys))
        armed = [ln for ln in text.splitlines() if "RING6" in ln]
        self.assertEqual(len(armed), 1, text)
        line = armed[0]
        libs = "libk512_mx6_gemm_N1024_ring6.so, libk512_mx6_gemm_N1536_ring6.so" + (
            ", libk512_mx6_u_ring6.so" if G.U_N in G.SUPPORTED_N else "")
        self.assertTrue(line.startswith(f"[k512-mx6] RING6 ARMED: {_KNOB}=1 -> {_DEFINE} in {libs}; impl routing OK"),
                        line)
        self.assertIn("weight shadow libk512_mx6_wshadow.so as shipped", line)
        self.assertTrue(line.endswith(" [rank 0]"), line)
        if G.U_N in G.SUPPORTED_N:
            self.assertNotIn("U N=512 not RING6", line)
        # the gin item quotes this line's head
        gin_text = " ".join(" ".join(ln.lstrip("#").split()) for ln in _gin_lines())
        self.assertIn(f'"[k512-mx6] RING6 ARMED: {_KNOB}=1 -> {_DEFINE} in {libs}; impl routing OK', gin_text)
        self.assertIn("[k512-mx6] prebuilt: N=1024,1536 + weight shadow", text)
        build_dir = G._process_build_dir()
        self.assertTrue(build_dir.name.startswith(f"pid{os.getpid()}_ring6_"), build_dir.name)
        self.assertIn(f"(dir {build_dir})", text)
        # everything is built: the training forward compiles nothing after RUN_START
        for n in (1536, 1024) + ((G.U_N,) if G.U_N in G.SUPPORTED_N else ()):
            self.assertTrue(G.kernel(n).ring6)
        G.wshadow_kernel()
        self.assertEqual(len(self.commands), len(keys))
        self.assertNotIn("RING6", self.prebuild())  # once per process; no rebuild
        self.assertEqual(len(self.commands), len(keys))

    def test_knob_off_prints_nothing_new(self) -> None:
        G = self.G
        self.set_knob(None)
        text = self.prebuild()
        self.assertNotIn("RING6", text)
        self.assertNotIn("_ring6.so", text)
        self.assertNotIn("ring6", G._process_build_dir().name)
        self.assertFalse([c for c in self.commands if any("K512MX6" in a for a in c)])
        self.assertFalse([k for k in G._kernels if isinstance(k, tuple)])

    def test_routing_failure_fails_the_prebuild(self) -> None:
        G = self.G
        self.honours = False
        self.set_knob("1")
        with _capture() as out:
            with self.assertRaisesRegex(G.PrebuildError, "RING6 routing check failed"):
                G.prebuild()
        self.assertIn("[k512-mx6] prebuild FAILED: ", out.getvalue())
        self.assertNotIn("RING6 ARMED", out.getvalue())

    def test_u_routing_failure_latches_u_only(self) -> None:
        G = self.G
        if G.U_N not in G.SUPPORTED_N:
            self.skipTest("no k512_mx6_u.cu in this tree")
        self.honours_fn = lambda command: pathlib.Path(command[1]).name != "k512_mx6_u.cu"
        self.set_knob("1")
        text = self.prebuild()
        line = next(ln for ln in text.splitlines() if "RING6 ARMED" in ln)
        self.assertIn(f"{_DEFINE} in libk512_mx6_gemm_N1024_ring6.so, libk512_mx6_gemm_N1536_ring6.so;", line)
        self.assertIn("; U N=512 not RING6 (K=512 MXFP6 GEMM N=512 library unavailable: RING6 routing check failed",
                      line)
        self.assertIn("RING6 routing check failed", G.failure(G.U_N) or "")

    def test_build_locks_follow_the_library_keys(self) -> None:
        # _get takes each library's build lock under the library's own key: the two settings never share one
        G = self.G
        keys = [1024, 1536, G.WSHADOW_KEY] + ([G.U_N] if G.U_N in G.SUPPORTED_N else [])
        for value, ring6 in ((None, False), ("1", True)):
            with self.fresh_tables():
                self.set_knob(value)
                self.prebuild()
                want = {(k, "ring6") if ring6 and k != G.WSHADOW_KEY else k for k in keys}
                self.assertEqual(set(G._key_locks), want, value)


# =============================================================================================== LDS
class ArmMarkerTest(_PrebuildCase):
    def test_prebuilt_line_lds_marks_the_arm(self) -> None:
        # The per-kernel lds= of the "[k512-mx6] prebuilt:" line is each served library's own compiler remark (the
        # shipped resource_summary): 73728 for the RING6 GEMM / U builds, 61440 for the shipped ones, so the log shows
        # the arm although the kernel names are the same; the weight shadow's is the same in both settings.
        G = self.G
        u = (G.U_N,) if G.U_N in G.SUPPORTED_N else ()
        for value, ring6 in ((None, False), ("0", False), ("1", True)):
            with self.fresh_tables():
                self.set_knob(value)
                pre = self.line(self.prebuild(), "[k512-mx6] prebuilt:")
                for n in (1024, 1536) + u:
                    kern = "k512_mx6_u_kernel" if n == G.U_N else "k512_mx6_kernel"
                    found = re.findall(rf"; {n} [0-9.]+s \[{kern}: [^\]]*?\blds=(\d+)\]", pre)
                    self.assertEqual(found, [str(_LDS[ring6])], (value, n, pre))
                shadow = re.findall(r"; wshadow [0-9.]+s \[([^\]]*)\]", pre)
                self.assertEqual(len(shadow), 1, pre)
                self.assertEqual(re.findall(r"\blds=(\d+)", shadow[0]), ["24", "24"], pre)


# =============================================================================================== CHECK
class KnobOnSelfCheckTest(_PrebuildCase):
    """The knob-on self-check paths of prebuild() (the cases above stub both GPU self-checks to pass)."""

    def setUp(self) -> None:
        super().setUp()
        self.set_knob("1")

    def test_gemm_selfcheck_failure_latches_every_ring6_library(self) -> None:
        G = self.G
        keys = [1024, 1536, G.WSHADOW_KEY] + ([G.U_N] if G.U_N in G.SUPPORTED_N else [])
        with mock.patch.object(G, "_selfcheck_gemm", lambda device, n, rows=257: "bad bytes" if n == 1024 else None):
            text, error = self.prebuild_raw()
        self.assertIsNotNone(error)
        self.assertIn("[k512-mx6] prebuild FAILED: bad bytes", text)
        self.assertNotIn("RING6 ARMED", text)
        self.assertEqual(list(G._kernels), [])  # nothing stays loaded: no RING6 library can be served
        self.assertEqual(set(G._failures), {G._lib_key(k) for k in keys})
        for n in [k for k in keys if k != G.WSHADOW_KEY]:
            self.assertIn("self-check failed: bad bytes", G.failure(n) or "", n)
            with self.assertRaisesRegex(RuntimeError, "self-check failed: bad bytes"):
                G.kernel(n)
        self.assertEqual(len(self.commands), len(keys))  # latched: kernel() rebuilds nothing

    def test_u_selfcheck_failure_latches_the_ring6_u_library_only(self) -> None:
        G = self.G
        if G.U_N not in G.SUPPORTED_N:
            self.skipTest("no k512_mx6_u.cu in this tree")
        with mock.patch.object(G, "_selfcheck_gemm", lambda device, n, rows=257: "u bad" if n == G.U_N else None):
            text, error = self.prebuild_raw()
        self.assertIsNone(error)
        pre = self.line(text, "[k512-mx6] prebuilt:")
        self.assertIn(f"(U N={G.U_N}: u bad)", pre)
        self.assertNotIn(f"+ U N={G.U_N}", pre)
        armed = self.line(text, "[k512-mx6] RING6 ARMED:")
        self.assertIn(f"{_DEFINE} in libk512_mx6_gemm_N1024_ring6.so, libk512_mx6_gemm_N1536_ring6.so;", armed)
        self.assertIn(f"U N={G.U_N} not RING6 (u bad)", armed)
        self.assertNotIn((G.U_N, "ring6"), G._kernels)  # never served to the U site
        self.assertEqual(set(G._failures), {(G.U_N, "ring6")})
        self.assertIn("U self-check failed: u bad", G.failure(G.U_N) or "")
        with self.assertRaisesRegex(RuntimeError, "U self-check failed: u bad"):
            G.kernel(G.U_N)
        self.assertTrue(G.kernel(1536).ring6 and G.kernel(1024).ring6)

    def test_u_selfcheck_runs_on_the_ring6_u_library(self) -> None:
        G = self.G
        seen: List[int] = []
        with mock.patch.object(G, "_selfcheck_gemm", lambda device, n, rows=257: seen.append(n)):
            text, error = self.prebuild_raw()
        self.assertIsNone(error)
        u = [G.U_N] if G.U_N in G.SUPPORTED_N else []
        self.assertEqual(sorted(seen), sorted([1024, 1536] + u))
        if u:
            self.assertIn(f" + U N={G.U_N} in ", self.line(text, "[k512-mx6] prebuilt:"))

    def test_selfcheck_message_names_the_ring6_library(self) -> None:
        G = self.G
        self.assertTrue(G.kernel(1536).ring6)
        with mock.patch.object(G, "k512_mx6", lambda *a, **k: k["out"].view(torch.uint8).fill_(1)), \
                mock.patch.object(torch.cuda, "synchronize", lambda *a: None):
            message = self.real_selfcheck_gemm(torch.device("cpu"), 1536)
        self.assertIsNotNone(message)
        self.assertIn("/libk512_mx6_gemm_N1536_ring6.so)", message or "")


# =============================================================================================== ROUTE (knob off)
class KnobOffGuardTest(_PrebuildCase):
    """The routing check the other way round: with the knob OFF every GEMM / U library must be the shipped build."""

    def test_an_injected_define_is_refused(self) -> None:
        G = self.G
        self.set_knob(None)
        refused = rf"RING6 routing check failed: {_KNOB} is off but libk512_mx6_gemm_N1536\.so k512_mx6_gemm_impl\(\)"
        with mock.patch.dict(os.environ, {"HIPCC_COMPILE_FLAGS_APPEND": _DEFINE}):
            with self.assertRaisesRegex(RuntimeError, refused + " = .*RING6 6-slot"):
                G.kernel(1536)
            with self.assertRaisesRegex(RuntimeError, "RING6 routing check failed"):
                G.kernel(1536)  # latched: no second build
        self.assertEqual(len(self.commands), 1)
        self.assertNotIn(_DEFINE, self.commands[0])  # the loader added nothing; hipcc appended the define
        self.assertEqual(dict(G._kernels), {})
        self.assertEqual(list(G._failures), [1536])  # the shipped key (the knob is off)
        self.assertIn(f"{_KNOB} is off but libk512_mx6_gemm_N1536.so ", G.failure(1536) or "")

    def test_an_injected_define_fails_the_prebuild_before_run_start(self) -> None:
        G = self.G
        self.set_knob("0")
        with mock.patch.dict(os.environ, {"HIPCC_COMPILE_FLAGS_APPEND": _DEFINE}):
            text, error = self.prebuild_raw()
        self.assertIsNotNone(error)
        self.assertIn(f"RING6 routing check failed: {_KNOB} is off", str(error))
        failed = self.line(text, "[k512-mx6] prebuild FAILED: ")
        for lib in ("libk512_mx6_gemm_N1024.so", "libk512_mx6_gemm_N1536.so"):
            self.assertIn(f"{_KNOB} is off but {lib} ", failed)
        self.assertNotIn("prebuilt:", text)
        self.assertNotIn("RING6 ARMED", text)
        for n in (1024, 1536) + ((G.U_N,) if G.U_N in G.SUPPORTED_N else ()):
            self.assertNotIn(n, G._kernels)
            self.assertIn(f"RING6 routing check failed: {_KNOB} is off", G.failure(n) or "", n)
            with self.assertRaisesRegex(RuntimeError, "RING6 routing check failed"):
                G.kernel(n)

    def test_a_u_library_reporting_ring6_is_latched_alone(self) -> None:
        # e.g. $DLRM_HSTU_FP8_FWD_MXFP6_U_SRC naming a source that defaults to RING6: U is optional, so U alone is
        # latched (U stays FP8: item (92)'s option-C fallback) and VQK / VK are the shipped builds
        G = self.G
        if G.U_N not in G.SUPPORTED_N:
            self.skipTest("no k512_mx6_u.cu in this tree")
        self.set_knob(None)
        self.default_ring6_fn = lambda command: pathlib.Path(command[1]).name == "k512_mx6_u.cu"
        text, error = self.prebuild_raw()
        self.assertIsNone(error)
        pre = self.line(text, "[k512-mx6] prebuilt:")
        self.assertIn(f"(U N={G.U_N}: K=512 MXFP6 GEMM N={G.U_N} library unavailable: RING6 routing check failed: "
                      f"{_KNOB} is off but libk512_mx6_u.so k512_mx6_u_impl() = ", pre)
        self.assertNotIn("RING6 ARMED", text)
        self.assertNotIn(G.U_N, G._kernels)
        self.assertEqual(list(G._failures), [G.U_N])
        for n in (1024, 1536):
            self.assertFalse(G.kernel(n).ring6)
            self.assertNotIn("RING6", G.kernel(n).impl)

    def test_ring6s_reported_with_the_knob_off_is_refused(self) -> None:
        G = self.G
        self.set_knob(None)
        self.spin = True
        self.default_ring6_fn = lambda command: True  # a source that defaults to the RING6S variant
        with self.assertRaisesRegex(RuntimeError, rf"{_KNOB} is off but libk512_mx6_gemm_N1024\.so .*RING6S"):
            G.kernel(1024)

    def test_knob_on_with_an_injected_ring5_is_refused(self) -> None:
        # the same injection against the knob-on direction: an appended -DK512MX6_RING=5 comes after the loader's
        # define and wins (the last -D of a macro counts), so the library is the shipped ring: refused
        G = self.G
        self.set_knob("1")
        with mock.patch.dict(os.environ, {"HIPCC_COMPILE_FLAGS_APPEND": "-DK512MX6_RING=5"}):
            with self.assertRaisesRegex(RuntimeError, r"RING6 routing check failed: libk512_mx6_gemm_N1536_ring6\.so"):
                G.kernel(1536)
        self.assertEqual(list(G._failures), [(1536, "ring6")])


# =============================================================================================== SRC
class SrcTest(unittest.TestCase):
    def test_sources_are_the_gpu_proven_bytes(self) -> None:
        self.assertEqual(hashlib.sha256(_GEMM_CU.read_bytes()).hexdigest(), _SHA_GEMM_CU)
        self.assertEqual(hashlib.sha256(_U_CU.read_bytes()).hexdigest(), _SHA_U_CU)

    def test_switch_defaults_to_the_shipped_ring_and_tags_the_impl(self) -> None:
        for path, tag in ((_GEMM_CU, "RING6 6-slot 72 KiB LDS ring"), (_U_CU, "RING6 6 x 12 KiB LDS ring")):
            text = path.read_text()
            self.assertEqual(text.count("#ifndef K512MX6_RING\n#define K512MX6_RING 5\n#endif\n"), 1, path.name)
            self.assertIn('#error "K512MX6_RING must be 5 (production ring) or 6 (RING6)"', text)
            self.assertIn(tag, text, path.name)
            self.assertIn("#define K512MX6_R6_SPIN 0\n", text)
        self.assertIn("5-slot 60 KiB LDS ring", _GEMM_CU.read_text())
        self.assertIn("(5 x 12 KiB LDS ring", _U_CU.read_text())

    def test_loader_checks_the_impl_tags_and_never_reaches_ring6s(self) -> None:
        src = _LOADER.read_text()
        self.assertNotIn("R6_SPIN", src)
        self.assertIn('RING6_IMPL: Dict[str, str] = {"gemm": "RING6 6-slot 72 KiB LDS ring", '
                      '"u": "RING6 6 x 12 KiB LDS ring"}', src)
        self.assertIn('RING6_DEFINE: str = "-DK512MX6_RING=6"', src)
        self.assertIn('RING6_KNOB: str = KNOB + "_RING6"', src)


if __name__ == "__main__":
    unittest.main()
