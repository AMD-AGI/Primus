#!/usr/bin/env python3
# pyre-strict
"""DLRM_HSTU_FP8_FWD_MXFP6_BSR (yambda_5b.gin item (110); r14 BSR; BITWISE; code default off).

BSR is the persistent MXFP6 K=512 forward GEMM with register-resident weights (ops/hip/k512_mx6_bsr.cu, r14 iter68
design O-D2, lab arm bsr1) served per site (N=1536 V/Q/K, N=1024 targets V/K, N=512 U) by the loader
(ops/hip/k512_mx6_gemm.py) once its pre-RUN_START self-check passed. CPU only: hipcc (subprocess.run) and the library
loader (ctypes.CDLL) are replaced by fakes, the GPU launches by a fake where a test needs one, so nothing is compiled or
launched. Pinned here:

  PARSE  the knob table (unset / "" / "0" / "false" off; "1" / "true" every site at every topology; "1n" every site
         on a single-node job only; a comma list of 1536 / 1024 / 512 those sites; anything else off plus ONE
         rank-tagged WARNING line), read once per process; the tests-only override.
  TOPO   mode "1n" (main's ruling; the cand_v1b binding): single node iff WORLD_SIZE == LOCAL_WORLD_SIZE (else the
         launcher's GPUS_PER_NODE, which scripts/launch_slurm.sh exports instead; both set must agree), the visible GPU
         count equals it and an initialized process group agrees; otherwise (2n / 4n, a missing, unparsable or
         conflicting value such as a stale LOCAL_WORLD_SIZE=16 next to GPUS_PER_NODE=8, a GPU-count or group mismatch)
         nothing is built or armed and ONE rank-tagged "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=<n>
         (LOCAL_WORLD_SIZE=<m>)" line is the only difference from a knob-off run.
  FLAGS  each BSR library's source (k512_mx6_bsr.cu next to the loader), file name (libk512_mx6_bsr_N<N>.so), key
         ((N, "bsr")), defines (-DHK_N / -DHK_K=512 or -DBSR_U=1, plus -DBSR_S=16 -DBSR_SCL=1 -DBSR_VMS=1) and hipcc
         command (the shipped one with those defines; never the RING6 define, with RING6 on or off).
  ROUTE  a BSR library whose impl string lacks "BSR O-D2" or reports RING6 is refused (latched under its own key);
         the RING6 routing checks never apply to BSR keys.
  ARMED  prebuild() builds the BSR libraries in its parallel pool next to the shipped ones (which it keeps), runs the
         BSR self-check against the site's shipped library, then kernel(N) serves BSR: ONE rank-tagged "[k512-mx6] BSR
         ARMED: ..." line; the "prebuilt:" line lists each armed library's resources (lds=102400). Without a
         prebuild, kernel(N) serves the shipped library and builds no BSR library.
  FALLBACK per site: a BSR build, routing or self-check failure (or exception) latches BSR for that site only: the
         site keeps its shipped library, ONE rank-tagged "[k512-mx6] BSR DECLINED N=<n>: <reason> (serving the shipped
         kernel)" line, the other sites keep BSR, prebuild() still returns True (never PrebuildError, never FP8).
  OFF    knob off ("0", "", "false", unset): hipcc command lines, file names, keys, latches, build locks and log
         lines equal the shipped ones (pinned literally) and a run with the BSR code path cut out.
  RING6  RING6 on + BSR on: the BSR libraries get no RING6 define and no RING6 check; the other sites and the fallback
         libraries keep RING6; the BSR self-check compares against the RING6 library the site would serve.
  CHECK  _selfcheck_bsr (real, on CPU tensors with a fake launch): zeros check, guard rows, the full-output bitwise
         comparison, the operands (deterministic ramps, the site's weight layout, nonzero bias, sa / sb).
  GIN    exactly one binding right after item (96)'s, a string: "0" (the testbed form, default OFF) or "1n" (the
         cand_v1b form); the (110) block names the bound value, the modes, the markers, the kill switch and the
         synthetic-only self-check, within 80 columns, no measured number.
  BOOT   apply_env_bootstrap declares the parameter once (Optional[str] = None) with one env-wins _bind_value (the
         string exported verbatim); a clean child running train_ranker's Phase 1 exports the bound value and a
         pre-set value survives.
  SRC    the in-tree k512_mx6_bsr.cu is the shipped source byte for byte (sha256 pin); its code (comments stripped,
         whitespace normalized) is pinned too (the shipped revision differs from the source verified bitwise on GPU
         in comment text and two host impl() string literals only, and builds byte-identical device code objects);
         it carries the impl tags and the switches the loader relies on.
"""

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
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

# recommendation/ (this file: recommendation/generative_recommenders/dlrm_v4/train/tests/)
_REPO = pathlib.Path(__file__).resolve().parents[4]
_GR = _REPO / "generative_recommenders"
_GIN = _GR / "dlrm_v4/train/gin/yambda_5b.gin"
_BOOT = _GR / "dlrm_v4/train/_env_bootstrap.py"
_HIP = _GR / "ops/hip"
_LOADER = _HIP / "k512_mx6_gemm.py"
_BSR_CU = _HIP / "k512_mx6_bsr.cu"
_LAUNCHER = _REPO / "scripts/launch_slurm.sh"
_KNOB = "DLRM_HSTU_FP8_FWD_MXFP6_BSR"
_RING6_KNOB = "DLRM_HSTU_FP8_FWD_MXFP6_RING6"
_PREV_BINDING = "apply_env_bootstrap.DLRM_HSTU_FP8_FWD_MXFP6_RING6 "  # item (96)'s binding (any value)
_HEAD = "# (110) BSR "
_RING6_DEFINE = "-DK512MX6_RING=6"
_REAL_GEMM = "generative_recommenders.ops.hip.k512_mx6_gemm"
# sha256 of the shipped k512_mx6_bsr.cu. It differs from the source verified bitwise on GPU (against the RING6
# builds of the shipped kernels) in comment text and in the two host impl() string literals only, and builds
# byte-identical device code objects for the three BSR builds. Any edit changes this pin.
_SHA_BSR_CU = "ee3f11fc2c828b4fbcf1275d734a80e212a7d8627f48fd277165e8e088040f3c"
# sha256 of _bsr_code_text() of the shipped source (comments stripped, whitespace normalized), re-pinned for v2.1:
# its code differs from that of v1, the source verified bitwise on GPU, only in the two host impl() string literals
# (k512_mx6_gemm_impl, k512_mx6_u_impl; host code). The gfx950 device code objects of the three BSR builds are
# byte-identical, proven by the code-object hashes recorded in ROW_110_bsr_v2_1.md. A comment-only edit keeps this
# pin; a code edit changes it and needs a new bitwise check on GPU (a host-only edit: a device code identity proof).
_SHA_BSR_CU_CODE = "68a03717988d9ef6747d7e0f583b304441ce4e10f1a88c3fbd45c8df9fe9decc"
# code whose result depends on a line number or a file name. _bsr_code_text() drops line positions and indentation, so
# "same code => same device code" also needs the code to hold none of it (and a build without debug info: FlagsTest
# pins the hipcc command element for element).
_LINE_DEPENDENT = re.compile(r"__LINE__|__FILE__|__FILE_NAME__|__BASE_FILE__|__COUNTER__|__DATE__|__TIME__|"
                             r"__TIMESTAMP__|__builtin_(?:LINE|FILE|COLUMN)|source_location|#\s*line\b|"
                             r"\bassert\s*\(|printf\s*\(")
_SWITCH = ["-DBSR_S=16", "-DBSR_SCL=1", "-DBSR_VMS=1"]  # lab arm bsr1 (16 ring slots, scales via LDS, strict waits)
_BSR_LDS = 102400  # per-workgroup LDS of the BSR kernels (16 x 6 KiB activation slots + the scale slots)
_SHIP_LDS = {False: 5 * 12288, True: 6 * 12288}  # shipped / RING6 GEMM and U kernels
_IMPL = {
    "gemm": "k512mx6 (fake): 128x256 tile, 5-slot 60 KiB LDS ring, swapped MFMA",
    "gemm_ring6": "k512mx6 (fake): 128x256 tile, RING6 6-slot 72 KiB LDS ring (A two K steps ahead), swapped MFMA",
    "u": "k512_mx6_u (fake): 2-WG mainloop (5 x 12 KiB LDS ring, scaled MFMA)",
    "u_ring6": "k512_mx6_u (fake): 2-WG mainloop (RING6 6 x 12 KiB LDS ring, A two K steps ahead, scaled MFMA)",
    "wshadow": "k512_mx6_wshadow (fake)",
    "bsr": "k512mx6 BSR O-D2 (fake): persistent 256 x 8-wave WGs, 64x512 tile, weights REGISTER-resident",
    "bsr_u": "k512_mx6_u BSR O-D2 (fake): persistent 256 x 8-wave WGs, bf16 + bias epilogue (k512_mx6_u.cu)",
}
_SITES = (1536, 1024, 512)
_BSR_NAMES = {n: f"libk512_mx6_bsr_N{n}.so" for n in _SITES}


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


def G_parse(raw: str) -> Any:  # pyre-ignore[3]
    """The loader's parse_bsr (the gin binding must be a valid knob value)."""
    return importlib.import_module(_REAL_GEMM).parse_bsr(raw)


def _binding_index(lines: List[str]) -> int:
    hits = [i for i, ln in enumerate(lines) if re.match(rf"^\s*apply_env_bootstrap\.{_KNOB}\s*=", ln)]
    if len(hits) != 1:
        raise AssertionError(f"expected one {_KNOB} binding, found {len(hits)}")
    return hits[0]


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


def _shipped_command(source: pathlib.Path, defines: List[str], output: pathlib.Path,
                     extra: Optional[List[str]] = None) -> List[str]:
    """The hipcc command of the shipped loader (ship15.1 k512_mx6_gemm.py), element for element."""
    return ["/opt/rocm/bin/hipcc", str(source), "--offload-arch=gfx950", "-std=c++20", "-w", "-O3",
            "-ffp-contract=off", "-I/opt/rocm/include/hip", *defines, "-mllvm",
            "-amdgpu-sched-strategy=iterative-ilp", "-shared", "-fPIC", "-Rpass-analysis=kernel-resource-usage",
            *(extra or []), "-o", str(output)]


class _LoaderCase(unittest.TestCase):
    """The real k512_mx6_gemm with hipcc and ctypes.CDLL faked (every build writes a small JSON 'library' naming its
    entry symbol and impl string, and prints the kernel-resource-usage remarks: LDS 102400 for a BSR build, 73728 /
    61440 for a RING6 / shipped GEMM or U build). fail_fn(command) makes a build fail (rc 1); impl_fn(command) may
    replace the impl string a build reports (a routing failure). Every module-level table and once-flag is restored
    after each test."""

    def setUp(self) -> None:
        if torch is None:
            self.skipTest("torch not installed")
        self.G = G = importlib.import_module(_REAL_GEMM)
        self.commands: List[List[str]] = []
        self.fail_fn: Callable[[List[str]], bool] = lambda command: False
        self.impl_fn: Callable[[List[str]], Optional[str]] = lambda command: None
        self.tmp = tempfile.TemporaryDirectory(prefix="mx6_loader110_")  # no "bsr" in the paths the logs show
        self.addCleanup(self.tmp.cleanup)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        utils = importlib.import_module("generative_recommenders.ops.utils")
        patches = [
            mock.patch.object(G, "_BUILD_ROOT", pathlib.Path(self.tmp.name)),
            mock.patch.object(G, "_build_dir", None),
            mock.patch.dict(G._kernels, clear=True),
            mock.patch.dict(G._failures, clear=True),
            mock.patch.dict(G._key_locks, clear=True),
            mock.patch.object(G, "_prebuilt", False),
            mock.patch.object(G, "_ring6", None),
            mock.patch.object(G, "_ring6_override", None),
            mock.patch.object(G, "_ring6_logged", False),
            mock.patch.object(G, "_bsr", None),
            mock.patch.object(G, "_bsr_raw", None),
            mock.patch.object(G, "_bsr_override", None),
            mock.patch.dict(G._bsr_armed, clear=True),
            mock.patch.dict(G._bsr_declined, clear=True),
            mock.patch.object(G, "_bsr_logged", False),
            mock.patch.object(G.subprocess, "run", self._fake_run),  # never the real hipcc
            mock.patch.object(G.ctypes, "CDLL", _FakeLib),  # never a real library load
            mock.patch.dict(os.environ, {"RANK": "0"}),
            # host-independent (the harness runs this file on a GPU node too): no process group, whatever an earlier
            # test of the session left behind; a test that needs one patches it again
            mock.patch.object(torch.distributed, "is_initialized", return_value=False),
        ]
        if hasattr(utils, "_ARM_PRINT_RANK"):
            patches.append(mock.patch.object(utils, "_ARM_PRINT_RANK", None))
        for patch in patches:
            stack.enter_context(patch)
        for name in (_KNOB, _RING6_KNOB, "DLRM_HSTU_FP8_FWD_MXFP6_GEMM_SRC", "HIPCC_COMPILE_FLAGS_APPEND",
                     "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GPUS_PER_NODE"):
            os.environ.pop(name, None)

    def _fake_run(self, command: List[str], **kwargs: Any) -> Any:  # pyre-ignore[2,3]
        self.commands.append(list(command))
        out = pathlib.Path(command[command.index("-o") + 1])
        source = pathlib.Path(command[1])
        if self.fail_fn(command):
            return subprocess.CompletedProcess(command, 1, stdout=f"{source}:1:1: error: synthetic build failure\n")
        flags = list(command) + os.environ.get("HIPCC_COMPILE_FLAGS_APPEND", "").split()  # what hipcc compiles with
        rings = [a for a in flags if a.startswith("-DK512MX6_RING=")]
        ring6 = bool(rings and rings[-1] == _RING6_DEFINE)
        if source.name == "k512_mx6_bsr.cu":  # the BSR source has no ring switch: a ring define changes nothing
            n = next(int(a.split("=")[1]) for a in command if a.startswith("-DHK_N="))
            u = "-DBSR_U=1" in command
            meta = {"entry": "k512_mx6_u_bf16_bias" if u else f"k512_mx6_e4m3_bias_N{n}",
                    "impl_symbol": "k512_mx6_u_impl" if u else "k512_mx6_gemm_impl",
                    "impl": _IMPL["bsr_u" if u else "bsr"]}
            kernels = [(f"_Z19k512_mx6_bsr_kernelILi{n}EEvPKhS1_PhPKtPKfS6_S1_S1_i", _BSR_LDS)]
        elif "-DK512_MX6_WSHADOW" in command:
            meta = {"entry": "k512_mx6_wshadow", "impl_symbol": "k512_mx6_wshadow_impl", "impl": _IMPL["wshadow"]}
            kernels = [(f"_ZN8k512mx6w23k512_mx6_wshadow_kernelILb{b}EEEvPKvxxiiiiiiiiiPhS3_S3_Pf", 24) for b in (0, 1)]
        elif source.name == "k512_mx6_u.cu":
            meta = {"entry": "k512_mx6_u_bf16_bias", "impl_symbol": "k512_mx6_u_impl",
                    "impl": _IMPL["u_ring6" if ring6 else "u"]}
            kernels = [("_Z17k512_mx6_u_kernelILi1ELi0ELi0ELi1EEvPKhS1_S1_S1_PtPKtPKfS6_i", _SHIP_LDS[ring6])]
        else:
            n = next(int(a.split("=")[1]) for a in command if a.startswith("-DHK_N="))
            meta = {"entry": f"k512_mx6_e4m3_bias_N{n}", "impl_symbol": "k512_mx6_gemm_impl",
                    "impl": _IMPL["gemm_ring6" if ring6 else "gemm"]}
            kernels = [(f"_Z15k512_mx6_kernelILi{n}EEvPKhS1_PhPKtPKfS6_S1_S1_i", _SHIP_LDS[ring6])]
        impl = self.impl_fn(command)
        if impl is not None:
            meta["impl"] = impl
        out.write_text(json.dumps(meta))
        return subprocess.CompletedProcess(command, 0, stdout=_remarks(source, kernels))

    def set_knob(self, value: Optional[str], ring6: Optional[str] = None) -> None:
        """The environment as the bootstrap leaves it, read afresh (the loader reads both knobs once per process)."""
        for name, v in ((_KNOB, value), (_RING6_KNOB, ring6)):
            if v is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = v
        self.G.set_bsr_for_testing(None)
        self.G.set_ring6_for_testing(None)

    def by_output(self) -> Dict[str, List[str]]:
        """{library file name: hipcc command} of every build so far."""
        got: Dict[str, List[str]] = {}
        for command in self.commands:
            tmp = pathlib.Path(command[command.index("-o") + 1]).name  # .<libname>.<pid>.tmp.so
            got[re.sub(r"^\.(.*)\.\d+\.tmp\.so$", r"\1", tmp)] = command
        return got


class _PrebuildCase(_LoaderCase):
    """_LoaderCase plus one fake gfx950 device; the shipped GPU self-checks pass, the BSR operands are dummies and the
    BSR self-check returns self.bsr_check(n) (None = pass); every BSR self-check call is recorded."""

    def setUp(self) -> None:
        super().setUp()
        G = self.G
        props = types.SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        self.bsr_checks: List[Tuple[int, str, str]] = []  # (N, BSR library, reference library)
        self.bsr_check: Callable[[int], Optional[str]] = lambda n: None
        self.gemm_checks: List[Tuple[int, Optional[str]]] = []
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for patch in (  # one fake gfx950 device with 8 visible GPUs on every host (CPU venv or GPU node)
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=8),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_properties", return_value=props),
            mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None),
            mock.patch.object(G, "_selfcheck_shadow", lambda device: None),
            mock.patch.object(G, "_selfcheck_gemm", self._fake_selfcheck_gemm),
            mock.patch.object(G, "_bsr_activation", lambda m, device, block=4096: ("x6", "xs")),
            mock.patch.object(G, "_selfcheck_bsr", self._fake_selfcheck_bsr),
        ):
            stack.enter_context(patch)

    def _fake_selfcheck_gemm(self, device: Any, n: int, rows: int = 257,  # pyre-ignore[2]
                             kern: Any = None) -> Optional[str]:  # pyre-ignore[2]
        self.gemm_checks.append((int(n), None if kern is None else pathlib.Path(kern.path).name))
        return None

    def _fake_selfcheck_bsr(self, device: Any, n: int, bsr: Any, ref: Any, acts: Any = None,  # pyre-ignore[2]
                            ms: Any = None) -> Optional[str]:  # pyre-ignore[2]
        self.assertEqual(sorted(acts or {}), sorted(self.G.BSR_SELFCHECK_M))
        self.bsr_checks.append((int(n), pathlib.Path(bsr.path).name, pathlib.Path(ref.path).name))
        return self.bsr_check(int(n))

    def prebuild(self) -> str:
        with _capture() as out:
            self.assertTrue(self.G.prebuild())
        return out.getvalue()

    def lines(self, text: str, needle: str) -> List[str]:
        return [ln for ln in text.splitlines() if needle in ln]

    def line(self, text: str, head: str) -> str:
        hits = [ln for ln in text.splitlines() if ln.startswith(head)]
        self.assertEqual(len(hits), 1, (head, text))
        return hits[0]

    def u_sites(self) -> Tuple[int, ...]:
        return (512,) if self.G.U_N in self.G.SUPPORTED_N else ()

    def shipped_name(self, n: int, ring6: bool = False) -> str:
        stem = "libk512_mx6_u" if n == 512 else f"libk512_mx6_gemm_N{n}"
        return f"{stem}_ring6.so" if ring6 else f"{stem}.so"


# =============================================================================================== PARSE
class ParseTest(_LoaderCase):
    def test_names(self) -> None:
        G = self.G
        self.assertEqual(G.BSR_KNOB, _KNOB)
        self.assertEqual(G.KNOB + "_BSR", _KNOB)
        self.assertEqual(list(G.BSR_SWITCH), _SWITCH)
        self.assertEqual(G.BSR_IMPL, "BSR O-D2")
        self.assertEqual(G.BSR_SITES, _SITES)
        self.assertEqual(G.bsr_source_path(), _BSR_CU)

    def test_table(self) -> None:
        G = self.G
        every, off = frozenset(_SITES), frozenset()
        cases = [  # (raw, sites, single-node only, valid)
            (None, off, False, True), ("", off, False, True), ("0", off, False, True), ("false", off, False, True),
            ("FALSE", off, False, True), ("False", off, False, True), (" 0 ", off, False, True),
            ("1", every, False, True), ("true", every, False, True), ("True", every, False, True),
            ("TRUE", every, False, True), (" 1\n", every, False, True),
            ("1n", every, True, True), ("1N", every, True, True), (" 1n ", every, True, True),
            ("1536", frozenset({1536}), False, True), ("1024", frozenset({1024}), False, True),
            ("512", frozenset({512}), False, True), ("1536,1024", frozenset({1536, 1024}), False, True),
            ("512, 1536", frozenset({512, 1536}), False, True), ("1536,1536", frozenset({1536}), False, True),
            ("1536,1024,512", every, False, True),
        ]
        for bad in ("2", "yes", "on", "off", "none", "1536,", ",1536", "1536;1024", "1536,2048", "-1536", "1,1536",
                    "0x600", "1536 1024", "1e3", "+1536", "1536.0", "all", "2n", "1n,1536", "1536n", "n1", "1 n"):
            cases.append((bad, off, False, False))
        for raw, sites, one_node, valid in cases:
            self.assertEqual(G.parse_bsr(raw), (sites, one_node, valid), raw)
        self.assertEqual(G.BSR_MODE_1N, "1n")

    def test_read_once_and_override(self) -> None:
        G = self.G
        self.set_knob("1536,512")
        self.assertEqual(G.bsr_sites(), frozenset({1536, 512}))
        os.environ[_KNOB] = "0"  # a later change of the environment does not flip a running process
        self.assertEqual(G.bsr_sites(), frozenset({1536, 512}))
        G.set_bsr_for_testing(None)  # re-read
        self.assertEqual(G.bsr_sites(), frozenset())
        G.set_bsr_for_testing("1024")  # the override wins over the environment
        os.environ[_KNOB] = "1"
        self.assertEqual(G.bsr_sites(), frozenset({1024}))
        G.set_bsr_for_testing(None)
        self.assertEqual(G.bsr_sites(), frozenset(_SITES))

    def test_invalid_value_warns_once_and_is_off(self) -> None:
        G = self.G
        with _capture() as out:
            self.set_knob("yes")
            self.assertEqual(G.bsr_sites(), frozenset())
            self.assertEqual(G.bsr_sites(), frozenset())
        warn = [ln for ln in out.getvalue().splitlines() if ln.strip()]
        self.assertEqual(warn, [f"[k512-mx6] WARNING: {_KNOB}='yes' is not 0 / 1 / 1n / false / true or a comma list "
                                f"of 1536, 1024, 512: BSR off (serving the shipped kernels) [rank 0]"])
        with _capture() as out:
            self.set_knob("1024")
            self.assertEqual(G.bsr_sites(), frozenset({1024}))
            self.set_knob(None)
            self.assertEqual(G.bsr_sites(), frozenset())
        self.assertEqual(out.getvalue(), "")  # valid values print nothing


# =============================================================================================== FLAGS
class FlagsTest(_LoaderCase):
    def test_spec_per_site(self) -> None:
        G = self.G
        for ring6 in (None, "1"):
            self.set_knob("1", ring6=ring6)
            for n in (1536, 1024):
                self.assertEqual(G._spec((n, "bsr")), (_BSR_CU, _BSR_NAMES[n], [f"-DHK_N={n}", "-DHK_K=512"] + _SWITCH,
                                                       f"k512_mx6_e4m3_bias_N{n}", "k512_mx6_gemm_impl"))
            self.assertEqual(G._spec((512, "bsr")), (_BSR_CU, _BSR_NAMES[512], ["-DHK_N=512", "-DBSR_U=1"] + _SWITCH,
                                                     "k512_mx6_u_bf16_bias", "k512_mx6_u_impl"))
            for n in _SITES:
                self.assertEqual(G._bsr_key(n), (n, "bsr"))
                self.assertEqual(G._lib_key((n, "bsr")), (n, "bsr"))  # never a RING6 key
                self.assertEqual(G._extra_flags((n, "bsr")), [])  # never the RING6 define
        with self.assertRaisesRegex(RuntimeError, "is not built"):
            G._spec((2048, "bsr"))
        with mock.patch.object(G, "bsr_source_path", return_value=pathlib.Path(self.tmp.name) / "absent.cu"):
            with self.assertRaisesRegex(RuntimeError, "BSR source absent: no k512_mx6_bsr.cu next to k512_mx6_gemm.py"):
                G._spec((1536, "bsr"))

    def test_commands_and_names(self) -> None:
        G = self.G
        for ring6 in (None, "1"):
            self.set_knob("1", ring6=ring6)
            self.commands.clear()
            with mock.patch.object(G, "_build_dir", None):
                for n in _SITES:
                    G._build((n, "bsr"))
                out_dir = G._process_build_dir()
            got = self.by_output()
            self.assertEqual(sorted(got), sorted(_BSR_NAMES.values()))
            for n in _SITES:
                name = _BSR_NAMES[n]
                defines = ([f"-DHK_N={n}", "-DHK_K=512"] if n != 512 else ["-DHK_N=512", "-DBSR_U=1"]) + _SWITCH
                tmp = out_dir / f".{name}.{os.getpid()}.tmp.so"
                self.assertEqual(got[name], _shipped_command(_BSR_CU, defines, tmp), (ring6, n))
                self.assertNotIn(_RING6_DEFINE, got[name])
                self.assertTrue((out_dir / name).is_file(), name)


# =============================================================================================== ROUTE
class RouteTest(_LoaderCase):
    def test_bsr_library_is_accepted(self) -> None:
        G = self.G
        for ring6 in (None, "1"):  # no RING6 check on a BSR key, in either RING6 setting
            with mock.patch.dict(G._kernels, clear=True), mock.patch.dict(G._failures, clear=True):
                self.set_knob("1", ring6=ring6)
                for n in _SITES:
                    kern = G._get((n, "bsr"))
                    self.assertTrue(kern.bsr)
                    self.assertFalse(kern.ring6)
                    self.assertEqual((kern.key, kern.n), ((n, "bsr"), n))
                    self.assertIn("BSR O-D2", kern.impl)
                    self.assertTrue(kern.path.endswith("/" + _BSR_NAMES[n]), kern.path)
                    self.assertEqual(kern.source, str(_BSR_CU))
                    self.assertIn("lds=102400", kern.resources)
                self.assertEqual(dict(G._failures), {})

    def test_refused_impls_are_latched_under_the_bsr_key(self) -> None:
        G = self.G
        self.set_knob("1")
        cases = {
            1536: "k512mx6 (fake): 128x256 tile, 5-slot 60 KiB LDS ring, swapped MFMA",  # a shipped build
            1024: "k512mx6 BSR O-D2 (fake) RING6 6-slot 72 KiB LDS ring",  # reports RING6
            512: "k512_mx6_u (fake): RING6 6 x 12 KiB LDS ring",  # a RING6 U build
        }
        self.impl_fn = lambda command: next((v for n, v in cases.items() if f"-DHK_N={n}" in command
                                             and pathlib.Path(command[1]).name == "k512_mx6_bsr.cu"), None)
        for n in _SITES:
            refused = rf"BSR routing check failed: {_BSR_NAMES[n]} k512_mx6_(gemm|u)_impl\(\) = "
            with self.assertRaisesRegex(RuntimeError, refused):
                G._get((n, "bsr"))
            self.assertIn("BSR routing check failed", G._failures[(n, "bsr")])
            self.assertTrue(G._failures[(n, "bsr")].startswith(f"K=512 MXFP6 BSR GEMM N={n} library unavailable: "))
            with self.assertRaisesRegex(RuntimeError, "BSR routing check failed"):
                G._get((n, "bsr"))  # latched: no second build
            self.assertIsNone(G.failure(n))  # the shipped key is not latched
        self.assertEqual(len(self.commands), 3)
        self.assertEqual(set(G._failures), {(n, "bsr") for n in _SITES})
        self.assertEqual(dict(G._kernels), {})


# =============================================================================================== ARMED
class ArmedTest(_PrebuildCase):
    def test_knob_on_arms_every_site_once(self) -> None:
        G = self.G
        self.set_knob("1")
        text = self.prebuild()
        sites = (1536, 1024) + self.u_sites()
        shipped = {"libk512_mx6_gemm_N1536.so", "libk512_mx6_gemm_N1024.so", "libk512_mx6_wshadow.so"} | (
            {"libk512_mx6_u.so"} if self.u_sites() else set())
        self.assertEqual(set(self.by_output()), shipped | {_BSR_NAMES[n] for n in sites})  # the shipped ones kept
        self.assertEqual(self.bsr_checks, [(n, _BSR_NAMES[n], self.shipped_name(n)) for n in sites])
        for n in sites:
            kern = G.kernel(n)
            self.assertTrue(kern.bsr and kern.path.endswith("/" + _BSR_NAMES[n]), kern.path)
            self.assertIs(kern, G._kernels[(n, "bsr")])
        armed = self.lines(text, "BSR ARMED")
        self.assertEqual(len(armed), 1, text)
        refs = ", ".join(self.shipped_name(n) for n in sites)
        self.assertEqual(armed[0], f"[k512-mx6] BSR ARMED: {_KNOB}=1 -> {', '.join(_BSR_NAMES[n] for n in sites)}; "
                                   f"impl routing OK (BSR O-D2); self-check zeros + bitwise vs shipped ({refs}) at "
                                   f"M=64,257,4096,4219,65536,65665 OK [rank 0]")
        self.assertEqual(self.lines(text, "DECLINED"), [])
        pre = self.line(text, "[k512-mx6] prebuilt:")
        for n in sites:
            found = re.findall(rf"; BSR {n} [0-9.]+s \[k512_mx6_bsr_kernel: [^\]]*?\blds=(\d+)\]", pre)
            self.assertEqual(found, [str(_BSR_LDS)], (n, pre))
        self.assertLess(text.index("[k512-mx6] prebuilt:"), text.index("BSR ARMED"))
        # once per process: a second prebuild builds nothing, checks nothing and prints no BSR line
        n_commands, n_checks = len(self.commands), len(self.bsr_checks)
        again = self.prebuild()
        self.assertEqual((len(self.commands), len(self.bsr_checks)), (n_commands, n_checks))
        self.assertEqual(self.lines(again, "BSR ARMED") + self.lines(again, "DECLINED"), [])
        self.assertEqual(len(re.findall(r"; BSR \d+ ", self.line(again, "[k512-mx6] prebuilt:"))), len(sites))

    def test_subset_arms_only_its_sites(self) -> None:
        G = self.G
        self.set_knob("1024")
        text = self.prebuild()
        self.assertEqual([c for c in self.by_output() if "bsr" in c], [_BSR_NAMES[1024]])
        self.assertEqual(self.bsr_checks, [(1024, _BSR_NAMES[1024], "libk512_mx6_gemm_N1024.so")])
        self.assertTrue(G.kernel(1024).bsr)
        for n in (1536,) + self.u_sites():
            self.assertFalse(G.kernel(n).bsr)
            self.assertTrue(G.kernel(n).path.endswith("/" + self.shipped_name(n)))
        self.assertIn(f"BSR ARMED: {_KNOB}=1024 -> {_BSR_NAMES[1024]}; ", self.line(text, "[k512-mx6] BSR ARMED:"))
        self.assertEqual(len(re.findall(r"; BSR \d+ ", self.line(text, "[k512-mx6] prebuilt:"))), 1)

    def test_no_bsr_without_a_prebuild(self) -> None:
        # kernel() never builds or serves an unchecked BSR library: before / without prebuild() it is the shipped one
        G = self.G
        self.set_knob("1")
        with _capture() as out:
            for n in (1536, 1024) + self.u_sites():
                self.assertFalse(G.kernel(n).bsr)
        self.assertEqual(out.getvalue(), "")
        self.assertFalse([c for c in self.commands if "k512_mx6_bsr.cu" in c[1]])
        self.prebuild()  # a later prebuild arms the sites
        for n in (1536, 1024) + self.u_sites():
            self.assertTrue(G.kernel(n).bsr)

    def test_build_locks_follow_the_library_keys(self) -> None:
        G = self.G
        self.set_knob("1536")
        self.prebuild()
        want = {1024, 1536, G.WSHADOW_KEY, (1536, "bsr")} | set(self.u_sites())
        self.assertEqual(set(G._key_locks), want)

    def test_u_site_absent(self) -> None:
        # U is a BSR site only when the U site exists (option C); the ARMED line says so
        G = self.G
        with mock.patch.object(G, "SUPPORTED_N", (1024, 1536)):
            self.set_knob("1")
            text = self.prebuild()
        self.assertEqual(sorted(c for c in self.by_output() if "bsr" in c),
                         sorted([_BSR_NAMES[1536], _BSR_NAMES[1024]]))
        armed = self.line(text, "[k512-mx6] BSR ARMED:")
        self.assertIn(f"-> {_BSR_NAMES[1536]}, {_BSR_NAMES[1024]}; ", armed)
        self.assertTrue(armed.endswith(" OK; U N=512 not BSR (U site absent) [rank 0]"), armed)
        self.assertEqual(self.lines(text, "DECLINED"), [])


# =============================================================================================== TOPO
class TopologyTest(_PrebuildCase):
    """Mode "1n" (main's ruling): on a single-node job only."""

    @contextlib.contextmanager
    def fresh(self, env: Dict[str, str], value: str = "1n", devices: int = 8):  # pyre-ignore[3]
        """One more process in this test: empty library tables / latches / locks / build directory, the topology
        variables `env` (the others unset), `devices` visible GPUs (torch.cuda.device_count; the fake device is
        available), the BSR knob `value` read afresh, a fresh command record."""
        G = self.G
        with contextlib.ExitStack() as stack:
            for patch in (mock.patch.dict(G._kernels, clear=True), mock.patch.dict(G._failures, clear=True),
                          mock.patch.dict(G._key_locks, clear=True), mock.patch.object(G, "_build_dir", None),
                          mock.patch.object(torch.cuda, "device_count", return_value=devices)):
                stack.enter_context(patch)
            for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "GPUS_PER_NODE"):
                os.environ.pop(name, None)
            os.environ.update(env)
            self.commands.clear()
            self.bsr_checks.clear()
            self.set_knob(value)
            yield

    def bsr_lines(self, text: str) -> List[str]:
        return [ln for ln in text.splitlines() if "BSR" in ln and not ln.startswith("[k512-mx6] prebuilt:")]

    def off_run(self) -> str:
        """stdout of a knob-off prebuild in fresh tables (build times / directory normalized)."""
        with self.fresh({}, value="0"):
            return self.normalized(self.prebuild())

    def normalized(self, text: str) -> str:
        text = text.replace(str(self.G._process_build_dir()), "<dir>")
        return re.sub(r"[0-9]+\.[0-9]s", "<t>s", text)

    def test_single_node_arms(self) -> None:
        G = self.G
        sites = (1536, 1024) + self.u_sites()
        cases = [  # (environment, visible GPUs, LOCAL_WORLD_SIZE as the ARMED line reports it)
            ({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8"}, 8, "8"),
            ({"WORLD_SIZE": "8", "GPUS_PER_NODE": "8"}, 8, "8 from GPUS_PER_NODE"),  # what launch_slurm.sh exports
            ({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8", "GPUS_PER_NODE": "8"}, 8, "8"),  # the normal 8/8/8 case
            ({"WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}, 1, "1"),
        ]
        for env, devices, lws in cases:
            with self.fresh(env, devices=devices):
                text = self.prebuild()
                for n in sites:
                    self.assertTrue(G.kernel(n).bsr, (env, n))
                lines = self.bsr_lines(text)
                self.assertEqual(len(lines), 1, (env, text))
                self.assertTrue(lines[0].startswith(f"[k512-mx6] BSR ARMED: {_KNOB}=1n (mode 1n: single node, "
                                                    f"WORLD_SIZE={env['WORLD_SIZE']} LOCAL_WORLD_SIZE={lws}) -> "),
                                lines[0])

    def test_gin_quotes_the_shipped_mode_line(self) -> None:
        # the ARMED line of the shipped mode as the gin item quotes it (launch_slurm.sh at one node: WORLD_SIZE 8,
        # GPUS_PER_NODE 8)
        if not self.u_sites():
            self.skipTest("no k512_mx6_u.cu in this tree")
        with self.fresh({"WORLD_SIZE": "8", "GPUS_PER_NODE": "8"}):
            line = self.line(self.prebuild(), "[k512-mx6] BSR ARMED:")
        gin_text = " ".join(" ".join(ln.lstrip("#").split()) for ln in _gin_lines())
        quoted = re.search(r'"(\[k512-mx6\] BSR ARMED: [^"]*)" ending in " \[rank N\]"', gin_text)
        self.assertIsNotNone(quoted, gin_text)
        assert quoted is not None
        template = re.escape(quoted.group(1)).replace(re.escape("<n>"), "8").replace(re.escape("<shipped libraries>"),
                                                                                     "[^)]+").replace(
            re.escape("<list>"), "[0-9,]+")
        self.assertRegex(line, "^" + template + r" \[rank 0\]$")
        self.assertIn(f"-> {', '.join(_BSR_NAMES[n] for n in _SITES)}; ", line)

    def test_multi_node_and_unknown_topology_are_off(self) -> None:
        G = self.G
        off = self.off_run()
        cases = [  # (environment, the one OFF line without " [rank 0]")
            ({"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"},
             "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=16 (LOCAL_WORLD_SIZE=8)"),
            ({"WORLD_SIZE": "32", "GPUS_PER_NODE": "8"},
             "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=32 (LOCAL_WORLD_SIZE=8 from GPUS_PER_NODE)"),
            ({"LOCAL_WORLD_SIZE": "8"},
             "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=<unset> (LOCAL_WORLD_SIZE=8): WORLD_SIZE or "
             "LOCAL_WORLD_SIZE / GPUS_PER_NODE missing or unparsable: treated as multi-node"),
            ({"WORLD_SIZE": "8"},
             "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=8 (LOCAL_WORLD_SIZE=<unset> from GPUS_PER_NODE): "
             "WORLD_SIZE or LOCAL_WORLD_SIZE / GPUS_PER_NODE missing or unparsable: treated as multi-node"),
            ({}, "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=<unset> (LOCAL_WORLD_SIZE=<unset> from "
                 "GPUS_PER_NODE): WORLD_SIZE or LOCAL_WORLD_SIZE / GPUS_PER_NODE missing or unparsable: treated as "
                 "multi-node"),
            ({"WORLD_SIZE": "eight", "LOCAL_WORLD_SIZE": "8"},
             "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=eight (LOCAL_WORLD_SIZE=8): WORLD_SIZE or "
             "LOCAL_WORLD_SIZE / GPUS_PER_NODE missing or unparsable: treated as multi-node"),
            ({"WORLD_SIZE": "0", "LOCAL_WORLD_SIZE": "0"},
             "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=0 (LOCAL_WORLD_SIZE=0): WORLD_SIZE or "
             "LOCAL_WORLD_SIZE / GPUS_PER_NODE not positive: treated as multi-node"),
            # a stale LOCAL_WORLD_SIZE next to the launcher's GPUS_PER_NODE (review): never single-node
            ({"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "16", "GPUS_PER_NODE": "8"},
             "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=16 (LOCAL_WORLD_SIZE=16): conflicting "
             "LOCAL_WORLD_SIZE/GPUS_PER_NODE (16 != 8): treated as multi-node"),
            ({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8", "GPUS_PER_NODE": "x"},
             "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=8 (LOCAL_WORLD_SIZE=8): WORLD_SIZE or "
             "LOCAL_WORLD_SIZE / GPUS_PER_NODE missing or unparsable: treated as multi-node"),
            # the visible device count must equal the ranks of one node
            ({"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "16"},
             "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=16 (LOCAL_WORLD_SIZE=16): torch.cuda.device_count() 8 "
             "!= LOCAL_WORLD_SIZE 16: treated as multi-node"),
        ]
        for env, want in cases:
            with self.fresh(env):
                text = self.prebuild()
                self.assertEqual(self.bsr_lines(text), [want + " [rank 0]"], env)
                self.assertFalse([c for c in self.commands if pathlib.Path(c[1]).name == "k512_mx6_bsr.cu"], env)
                self.assertEqual((dict(G._bsr_armed), self.bsr_checks), ({}, []), env)
                for n in (1536, 1024) + self.u_sites():
                    self.assertFalse(G.kernel(n).bsr, (env, n))
                # the OFF line is the only difference from a knob-off run
                self.assertEqual(self.normalized(text).replace(want + " [rank 0]\n", ""), off, env)
                self.assertEqual(self.bsr_lines(self.prebuild()), [], env)  # once per process

    def test_device_count_must_match(self) -> None:
        # the launcher's environment (WORLD_SIZE 8, GPUS_PER_NODE 8) with a different number of visible GPUs
        G = self.G
        for devices, armed in ((8, True), (4, False)):
            with self.fresh({"WORLD_SIZE": "8", "GPUS_PER_NODE": "8"}, devices=devices):
                lines = self.bsr_lines(self.prebuild())
                self.assertEqual(G.kernel(1536).bsr, armed, devices)
                if armed:
                    self.assertTrue(lines[0].startswith("[k512-mx6] BSR ARMED: "), lines)
                else:
                    self.assertEqual(lines, ["[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=8 (LOCAL_WORLD_SIZE=8 "
                                             "from GPUS_PER_NODE): torch.cuda.device_count() 4 != GPUS_PER_NODE 8: "
                                             "treated as multi-node [rank 0]"])

    def test_single_node_table(self) -> None:
        # single_node() itself; without a visible GPU the device count is not consulted
        G = self.G
        dist = torch.distributed
        cases = [  # (environment, GPU available, visible GPUs, single node, the reason or "")
            ({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8", "GPUS_PER_NODE": "8"}, True, 8, True, ""),
            ({"WORLD_SIZE": "8", "GPUS_PER_NODE": "8"}, True, 8, True, ""),
            ({"WORLD_SIZE": "8", "GPUS_PER_NODE": "8"}, False, 0, True, ""),
            ({"WORLD_SIZE": "8", "GPUS_PER_NODE": "8"}, True, 4, False,
             "torch.cuda.device_count() 4 != GPUS_PER_NODE 8"),
            ({"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "16", "GPUS_PER_NODE": "8"}, True, 16, False,
             "conflicting LOCAL_WORLD_SIZE/GPUS_PER_NODE (16 != 8)"),
            ({"WORLD_SIZE": "16", "GPUS_PER_NODE": "8"}, True, 8, False, ""),  # an ordinary 2-node job
            ({"WORLD_SIZE": "8"}, True, 8, False, "missing or unparsable"),
            ({"GPUS_PER_NODE": "8"}, True, 8, False, "missing or unparsable"),
        ]
        for env, available, devices, want, reason in cases:
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(torch.cuda, "is_available", return_value=available), \
                    mock.patch.object(torch.cuda, "device_count", return_value=devices), \
                    mock.patch.object(dist, "is_initialized", return_value=False):
                for name in {"WORLD_SIZE", "LOCAL_WORLD_SIZE", "GPUS_PER_NODE"} - set(env):
                    os.environ.pop(name, None)
                single, why = G.single_node()
                self.assertEqual(single, want, (env, available, devices, why))
                if reason:
                    self.assertIn(reason, why, env)
                else:
                    self.assertEqual(why, "", env)

    def test_process_group_cross_check(self) -> None:
        G = self.G
        dist = torch.distributed
        for group, armed in ((8, True), (16, False)):
            with self.fresh({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8"}), \
                    mock.patch.object(dist, "is_available", return_value=True), \
                    mock.patch.object(dist, "is_initialized", return_value=True), \
                    mock.patch.object(dist, "get_world_size", return_value=group), \
                    mock.patch.object(dist, "get_rank", return_value=0):
                text = self.prebuild()
                self.assertEqual(G.kernel(1536).bsr, armed, group)
                lines = self.bsr_lines(text)
                if armed:
                    self.assertTrue(lines[0].startswith("[k512-mx6] BSR ARMED: "), lines)
                else:
                    self.assertEqual(lines, ["[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=8 (LOCAL_WORLD_SIZE=8): "
                                             "torch.distributed world size 16 != LOCAL_WORLD_SIZE 8: treated as "
                                             "multi-node [rank 0]"])

    def test_mode_1_and_site_lists_ignore_the_topology(self) -> None:
        G = self.G
        for value, sites in (("1", (1536, 1024) + self.u_sites()), ("1024", (1024,))):
            with self.fresh({"WORLD_SIZE": "32", "GPUS_PER_NODE": "8"}, value=value):
                text = self.prebuild()
                self.assertEqual([n for n in (1536, 1024) + self.u_sites() if G.kernel(n).bsr], list(sites), value)
                self.assertTrue(self.line(text, "[k512-mx6] BSR ARMED:").startswith(
                    f"[k512-mx6] BSR ARMED: {_KNOB}={value} -> "), text)
                self.assertNotIn("BSR OFF", text)

    def test_off_and_unset_print_no_bsr_line(self) -> None:
        for value in ("0", "", "false"):
            with self.fresh({"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"}, value=value):
                self.assertEqual(self.bsr_lines(self.prebuild()), [], value)
        with self.fresh({"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"}, value="0"):
            os.environ.pop(_KNOB, None)
            self.G.set_bsr_for_testing(None)
            self.assertEqual(self.bsr_lines(self.prebuild()), [])


# =============================================================================================== FALLBACK
class FallbackTest(_PrebuildCase):
    def check_declined(self, text: str, n: int, reason: str) -> None:
        G = self.G
        declined = self.lines(text, "BSR DECLINED")
        self.assertEqual(len(declined), 1, text)
        self.assertTrue(declined[0].startswith(f"[k512-mx6] BSR DECLINED N={n}: {reason}"), declined[0])
        self.assertTrue(declined[0].endswith(" (serving the shipped kernel) [rank 0]"), declined[0])
        self.assertNotIn(n, G._bsr_armed)
        self.assertNotIn((n, "bsr"), G._kernels)
        self.assertIn((n, "bsr"), G._failures)
        kern = G.kernel(n)  # the shipped library serves the site
        self.assertFalse(kern.bsr)
        self.assertTrue(kern.path.endswith("/" + self.shipped_name(n)), kern.path)
        self.assertIsNone(G.failure(n))
        others = [m for m in (1536, 1024) + self.u_sites() if m != n]
        for m in others:
            self.assertTrue(G.kernel(m).bsr, m)
        armed = self.line(text, "[k512-mx6] BSR ARMED:")
        self.assertIn(f"-> {', '.join(_BSR_NAMES[m] for m in others)}; ", armed)
        self.assertNotIn(_BSR_NAMES[n], armed)
        self.assertNotIn(f"; BSR {n} ", self.line(text, "[k512-mx6] prebuilt:"))
        # once per process: a second prebuild neither rebuilds nor re-prints it
        n_commands = len(self.commands)
        again = self.prebuild()
        self.assertEqual(len(self.commands), n_commands)
        self.assertEqual(self.lines(again, "DECLINED"), [])
        self.assertFalse(G.kernel(n).bsr)

    def test_build_failure(self) -> None:
        self.fail_fn = lambda command: pathlib.Path(command[1]).name == "k512_mx6_bsr.cu" and "-DHK_N=1024" in command
        self.set_knob("1")
        text = self.prebuild()
        self.check_declined(text, 1024, "K=512 MXFP6 BSR GEMM N=1024 library unavailable: hipcc rc=1 for "
                                        "libk512_mx6_bsr_N1024.so (log ")
        self.assertIn("synthetic build failure", self.lines(text, "BSR DECLINED")[0])
        self.assertEqual([c[0] for c in self.bsr_checks], [1536] + list(self.u_sites()))

    def test_routing_failure(self) -> None:
        if not self.u_sites():
            self.skipTest("no k512_mx6_u.cu in this tree")
        self.impl_fn = lambda command: ("k512_mx6_u (fake): 2-WG mainloop" if pathlib.Path(command[1]).name ==
                                        "k512_mx6_bsr.cu" and "-DBSR_U=1" in command else None)
        self.set_knob("1")
        text = self.prebuild()
        self.check_declined(text, 512, "K=512 MXFP6 BSR GEMM N=512 library unavailable: BSR routing check failed: "
                                       "libk512_mx6_bsr_N512.so k512_mx6_u_impl() = ")

    def test_selfcheck_failure(self) -> None:
        self.bsr_check = lambda n: "bitwise check M=4219: 3 of 6483 output bytes differ" if n == 1536 else None
        self.set_knob("1")
        text = self.prebuild()
        self.check_declined(text, 1536, "self-check failed: bitwise check M=4219: 3 of 6483 output bytes differ "
                                        "(serving")
        self.assertIn("self-check failed: bitwise check", self.G._failures[(1536, "bsr")])

    def test_selfcheck_exception(self) -> None:
        def boom(n: int) -> Optional[str]:
            if n == 1024:
                raise RuntimeError("k512_mx6 N=1024 M=257 failed: hipError 98")
            return None

        self.bsr_check = boom
        self.set_knob("1")
        text = self.prebuild()
        self.check_declined(text, 1024, "self-check failed: self-check raised RuntimeError: k512_mx6 N=1024 M=257 "
                                        "failed: hipError 98")

    def test_every_site_declined_keeps_the_prebuild(self) -> None:
        G = self.G
        self.bsr_check = lambda n: f"bad {n}"
        self.set_knob("1")
        text = self.prebuild()  # still True: BSR never fails the prebuild
        sites = (1536, 1024) + self.u_sites()
        self.assertEqual([ln.split(":")[0] for ln in self.lines(text, "BSR DECLINED")],
                         [f"[k512-mx6] BSR DECLINED N={n}" for n in sites])
        self.assertEqual(self.lines(text, "BSR ARMED"), [])
        self.assertEqual(re.findall(r"; BSR \d+ ", self.line(text, "[k512-mx6] prebuilt:")), [])
        for n in sites:
            self.assertFalse(G.kernel(n).bsr)
        self.assertEqual(dict(G._bsr_armed), {})

    def test_operand_failure_declines_and_keeps_the_prebuild(self) -> None:
        G = self.G

        def no_memory(m: int, device: Any, block: int = 4096) -> Any:  # pyre-ignore[2,3]
            raise RuntimeError("HIP out of memory")

        self.set_knob("1536")
        with mock.patch.object(G, "_bsr_activation", no_memory):
            text = self.prebuild()
        self.assertIn("[k512-mx6] BSR DECLINED N=1536: self-check operands failed: RuntimeError: HIP out of memory "
                      "(serving the shipped kernel) [rank 0]", text)
        self.assertFalse(G.kernel(1536).bsr)

    def test_shipped_u_failure_declines_bsr_u(self) -> None:
        # without the site's shipped library there is no bitwise reference: BSR U is declined and U behaves as without
        # the knob (latched: item (92)'s option-C fallback keeps U on the shipped FP8 GEMM)
        G = self.G
        if not self.u_sites():
            self.skipTest("no k512_mx6_u.cu in this tree")

        def u_bad(device: Any, n: int, rows: int = 257, kern: Any = None) -> Optional[str]:  # pyre-ignore[2]
            return "u bad" if n == 512 and kern is None else None

        self.set_knob("1")
        with mock.patch.object(G, "_selfcheck_gemm", u_bad):
            text = self.prebuild()
        self.assertIn("[k512-mx6] BSR DECLINED N=512: the site's shipped library is not available (K=512 MXFP6 U "
                      "self-check failed: u bad) (serving the shipped kernel) [rank 0]", text)
        with self.assertRaisesRegex(RuntimeError, "U self-check failed: u bad"):
            G.kernel(512)
        self.assertTrue(G.kernel(1536).bsr and G.kernel(1024).bsr)


# =============================================================================================== OFF
_PREBUILT_OFF = re.compile(  # the shipped "prebuilt:" line (build times normalized to <t>)
    r"^\[k512-mx6\] prebuilt: N=1024,1536 \+ weight shadow( \+ U N=512)? in <t>s \(dir \S+\); self-check \(shadow "
    r"ramps vs torch reference, GEMMs on zeros\) OK; 1024 <t>s \[k512_mx6_kernel: [^\]]*\]; 1536 <t>s "
    r"\[k512_mx6_kernel: [^\]]*\]; wshadow <t>s \[[^\]]*\](; 512 <t>s \[k512_mx6_u_kernel: [^\]]*\])?$")


class KnobOffTest(_PrebuildCase):
    def run_prebuild(self, value: Optional[str], ring6: Optional[str], cut: bool) -> Dict[str, Any]:
        """One prebuild in fresh tables: everything a knob-off run must keep as shipped. cut=True cuts the BSR code
        path out (the knob is never read; the arming / logging hooks must not run)."""
        G = self.G

        def must_not_run(*args: Any, **kwargs: Any) -> None:  # pyre-ignore[2]
            raise AssertionError("BSR code path ran")

        self.commands.clear()
        with contextlib.ExitStack() as stack:
            for patch in (mock.patch.dict(G._kernels, clear=True), mock.patch.dict(G._failures, clear=True),
                          mock.patch.dict(G._key_locks, clear=True), mock.patch.object(G, "_build_dir", None)):
                stack.enter_context(patch)
            if cut:
                for patch in (mock.patch.object(G, "bsr_sites", lambda: frozenset()),
                              mock.patch.object(G, "_arm_bsr", must_not_run),
                              mock.patch.object(G, "_bsr_decline", must_not_run),
                              mock.patch.object(G, "_log_bsr_armed", lambda: None)):
                    stack.enter_context(patch)
            self.set_knob(value, ring6=ring6)
            text = self.prebuild()
            build_dir = str(G._process_build_dir())
            served = {n: pathlib.Path(G.kernel(n).path).name for n in (1536, 1024) + self.u_sites()}
            state = {"keys": sorted(map(str, G._kernels)), "locks": sorted(map(str, G._key_locks)),
                     "failures": dict(G._failures), "armed": dict(G._bsr_armed), "served": served,
                     "commands": sorted([a.replace(build_dir, "<dir>") for a in c] for c in self.commands),  # parallel
                     "text": re.sub(r"[0-9]+\.[0-9]s", "<t>s", text.replace(build_dir, "<dir>")),
                     "bsr_checks": list(self.bsr_checks), "gemm_checks": list(self.gemm_checks)}
        self.bsr_checks.clear()
        self.gemm_checks.clear()
        return state

    def test_off_equals_the_shipped_run(self) -> None:
        G = self.G
        for ring6 in (None, "1"):
            reference = self.run_prebuild(None, ring6, cut=True)
            for value in (None, "0", "", "false", "FALSE", " 0 "):
                self.assertEqual(self.run_prebuild(value, ring6, cut=False), reference, (value, ring6))
            on6 = ring6 == "1"
            # and that run is the shipped one, pinned literally
            names = {1536: self.shipped_name(1536, on6), 1024: self.shipped_name(1024, on6),
                     G.WSHADOW_KEY: "libk512_mx6_wshadow.so"}
            if self.u_sites():
                names[512] = self.shipped_name(512, on6)
            want_commands = []
            for key, name in names.items():
                if key == G.WSHADOW_KEY:
                    source, defines, extra = G.source_path(), ["-DK512_MX6_WSHADOW"], []
                elif key == 512:
                    source, defines, extra = G.u_source_path(), [], [_RING6_DEFINE] if on6 else []
                else:
                    source, defines, extra = G.source_path(), [f"-DHK_N={key}", "-DHK_K=512"], (
                        [_RING6_DEFINE] if on6 else [])
                want_commands.append(_shipped_command(source, defines, pathlib.Path(f"<dir>/.{name}.{os.getpid()}"
                                                                                    f".tmp.so"), extra))
            self.assertEqual(sorted(reference["commands"]), sorted(want_commands), ring6)
            self.assertEqual(reference["keys"], sorted(str((k, "ring6") if on6 and k != G.WSHADOW_KEY else k)
                                                       for k in names))
            self.assertEqual(reference["locks"], reference["keys"])
            self.assertEqual(reference["failures"], {})
            self.assertEqual(reference["armed"], {})
            self.assertEqual(reference["bsr_checks"], [])
            self.assertTrue(all(kern is None for _, kern in reference["gemm_checks"]))
            self.assertEqual(reference["served"], {n: names[n] for n in reference["served"]})
            lines = [ln for ln in reference["text"].splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2 if on6 else 1, lines)
            self.assertRegex(lines[0], _PREBUILT_OFF)
            self.assertNotRegex(reference["text"], "(?i)bsr")
            if on6:
                self.assertTrue(lines[1].startswith(f"[k512-mx6] RING6 ARMED: {_RING6_KNOB}=1 -> "), lines[1])

    def test_kernel_never_reads_the_knob_when_nothing_is_armed(self) -> None:
        G = self.G
        self.set_knob(None)
        with mock.patch.object(G, "bsr_sites", side_effect=AssertionError("knob read")):
            for n in (1536, 1024) + self.u_sites():
                self.assertFalse(G.kernel(n).bsr)


# =============================================================================================== RING6
class Ring6Test(_PrebuildCase):
    def test_ring6_and_bsr(self) -> None:
        G = self.G
        self.set_knob("1536", ring6="1")
        text = self.prebuild()
        got = self.by_output()
        self.assertNotIn(_RING6_DEFINE, got[_BSR_NAMES[1536]])
        self.assertEqual(got[_BSR_NAMES[1536]].count("-DHK_N=1536"), 1)
        for n in (1536, 1024) + self.u_sites():  # every shipped library of a site is the RING6 build, as without BSR
            self.assertEqual(got[self.shipped_name(n, True)].count(_RING6_DEFINE), 1, n)
        self.assertNotIn(_RING6_DEFINE, got["libk512_mx6_wshadow.so"])
        self.assertTrue(G.kernel(1536).bsr and not G.kernel(1536).ring6)
        for n in (1024,) + self.u_sites():
            self.assertTrue(G.kernel(n).ring6 and not G.kernel(n).bsr, n)
        # the bitwise reference of the BSR site is the RING6 library it would serve without BSR
        self.assertEqual(self.bsr_checks, [(1536, _BSR_NAMES[1536], self.shipped_name(1536, True))])
        self.assertIn(f"vs shipped ({self.shipped_name(1536, True)}) at ", self.line(text, "[k512-mx6] BSR ARMED:"))
        ring6_line = self.line(text, "[k512-mx6] RING6 ARMED:")
        self.assertNotIn("bsr", ring6_line)
        self.assertEqual(set(G._kernels), {(1536, "bsr"), (1536, "ring6"), (1024, "ring6"), G.WSHADOW_KEY}
                         | {(n, "ring6") for n in self.u_sites()})

    def test_injected_ring6_define_does_not_touch_bsr(self) -> None:
        # an appended -DK512MX6_RING=6 (HIPCC_COMPILE_FLAGS_APPEND) with RING6 on: the BSR source has no ring switch
        G = self.G
        self.set_knob("1", ring6="1")
        with mock.patch.dict(os.environ, {"HIPCC_COMPILE_FLAGS_APPEND": _RING6_DEFINE}):
            for n in _SITES:
                self.assertTrue(G._get((n, "bsr")).bsr)


# =============================================================================================== CHECK
class _Launches:
    """A fake launch(): the shipped library writes a deterministic function of its operands into the M rows of `out`
    (zeros on all-zero operands, like the real kernels); the BSR library writes the same unless `mutate` says
    otherwise. Records every call."""

    def __init__(self, bsr_path: str) -> None:
        self.bsr_path = bsr_path
        self.mutate: Optional[str] = None
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, kern: Any, x6: Any, xs: Any, w6: Any, ws: Any, bias: Any, sa: Any, sb: Any, m: int,  # noqa
                 out: Any) -> Any:  # pyre-ignore[2,3]
        assert torch is not None
        self.calls.append({"path": kern.path, "x6": x6, "xs": xs, "w6": w6, "ws": ws, "bias": bias, "sa": sa,
                           "sb": sb, "m": m, "out": out})
        u8 = out.view(torch.uint8)
        width = u8.shape[1]
        zeros = int(torch.count_nonzero(x6)) == 0 and int(torch.count_nonzero(w6)) == 0  # the zeros check
        if zeros:
            u8.zero_()
        else:
            seed = (int(x6.to(torch.int64).sum()) * 31 + int(w6.to(torch.int64).sum()) * 7 + int(xs.sum())) % 97
            rows = torch.arange(m, dtype=torch.int64).view(-1, 1)
            cols = torch.arange(width, dtype=torch.int64).view(1, -1)
            u8.copy_(((rows * 13 + cols * 5 + seed) % 127).to(torch.uint8))  # never 0xFF
        if kern.path != self.bsr_path or self.mutate is None:
            return out
        if self.mutate == "zeros" and zeros:
            u8.fill_(1)
        elif self.mutate == "byte" and not zeros:
            u8[100, 3] ^= 1
        elif self.mutate == "row" and not zeros:
            u8[m - 1].fill_(0xFF)  # the last row left unwritten
        elif self.mutate == "guard" and not zeros:
            u8.as_strided((m + 1, width), (width, 1))[m, 0] = 0  # the first byte past row M
        return out


class SelfCheckTest(_LoaderCase):
    """The real _selfcheck_bsr on CPU tensors: hipcc / ctypes faked (_LoaderCase), launch() faked (_Launches),
    operand validation and the weight-shadow kernel replaced by their CPU equivalents."""

    def setUp(self) -> None:
        super().setUp()
        G = self.G
        self.set_knob("1")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for patch in (
            mock.patch.object(G, "ineligible_reason", lambda *a: None),
            mock.patch.object(G, "refresh_weight_shadow", lambda w, layout, **kw: G.mx6_weight_reference(w, layout)),
            mock.patch.object(torch.cuda, "synchronize", lambda *a: None),
        ):
            stack.enter_context(patch)
        self.cpu = torch.device("cpu")

    def run_check(self, n: int, mutate: Optional[str], ms: Tuple[int, ...] = (257, 300)) -> Tuple[Optional[str], Any]:
        G = self.G
        if n == 512 and G.U_N not in G.SUPPORTED_N:
            self.skipTest("no k512_mx6_u.cu in this tree")
        bsr, ref = G._get((n, "bsr")), G._get(n)
        launches = _Launches(bsr.path)
        launches.mutate = mutate
        with mock.patch.object(G, "launch", launches):
            return G._selfcheck_bsr(self.cpu, n, bsr, ref, None, ms), launches

    def test_pass_and_operands(self) -> None:
        G = self.G
        for n in (1536, 1024, 512):
            reason, launches = self.run_check(n, None)
            self.assertIsNone(reason, n)
            # zeros check on the BSR library first, then per M: the shipped library, then BSR
            bsr_path, ref_path = G._kernels[(n, "bsr")].path, G._kernels[n].path
            self.assertEqual([c["path"] for c in launches.calls], [bsr_path] + [ref_path, bsr_path] * 2)
            w6, ws, sb = G.mx6_weight_reference(G._ramp_weight(self.cpu, torch.float32), G._BSR_LAYOUT[n])
            dtype = torch.bfloat16 if n == 512 else torch.float8_e4m3fn
            width = n * (2 if n == 512 else 1)
            for call, m in zip(launches.calls[1:], (257, 257, 300, 300)):
                self.assertEqual(call["m"], m)
                self.assertEqual((tuple(call["out"].shape), call["out"].dtype), ((m, n), dtype))
                self.assertEqual(call["out"].untyped_storage().nbytes(), (m + 2) * width)  # two guard rows
                self.assertTrue(G.bitwise_equal(call["w6"], w6) and G.bitwise_equal(call["ws"], ws))
                self.assertTrue(G.bitwise_equal(call["sb"], sb))
                self.assertEqual(float(call["sa"].item()), 0.8125)
                self.assertEqual(call["bias"].dtype, torch.bfloat16)
                self.assertEqual(int(torch.count_nonzero(call["bias"])), n)
                x6, xs = G.mx6_activation_reference(G._bsr_ramp_rows(0, m, self.cpu))
                self.assertTrue(G.bitwise_equal(call["x6"], x6) and G.bitwise_equal(call["xs"], xs))

    def test_refusals(self) -> None:
        cases = {
            "byte": r"^bitwise check M=257: 1 of \d+ output bytes differ from libk512_mx6_gemm_N1536\.so "
                    r"\(libk512_mx6_bsr_N1536\.so\)$",
            "row": r"^bitwise check M=257: 1536 of \d+ output bytes differ from ",
            "guard": r"^bitwise check M=257: 1 guard-row bytes past row M were written "
                     r"\(libk512_mx6_bsr_N1536\.so\)$",
            "zeros": r"^zeros check: GEMM N=1536 self-check on zeros wrote \d+ nonzero bytes \(\S+/libk512_mx6_bsr_N1536"
                     r"\.so\)$",
        }
        for mutate, want in cases.items():
            reason, _ = self.run_check(1536, mutate)
            self.assertIsNotNone(reason, mutate)
            self.assertRegex(reason or "", want)

    def test_u_rows_are_bf16(self) -> None:
        reason, launches = self.run_check(512, "byte")
        self.assertRegex(reason or "", r"^bitwise check M=257: 1 of \d+ output bytes differ from libk512_mx6_u\.so ")
        self.assertEqual(launches.calls[1]["out"].dtype, torch.bfloat16)

    def test_activation_blocks_equal_the_one_shot_reference(self) -> None:
        G = self.G
        for m in (1, 129, 700):
            x6, xs = G._bsr_activation(m, self.cpu, block=256)
            r6, rs = G.mx6_activation_reference(G._bsr_ramp_rows(0, m, self.cpu))
            self.assertTrue(G.bitwise_equal(x6, r6) and G.bitwise_equal(xs, rs), m)

    def test_operand_coverage(self) -> None:
        # deterministic ramps: no two quantized rows of the largest self-check alike, many block exponents, zero
        # blocks, both signs, exact nonzero bias; the e4m3 outputs reach saturation, subnormals and +-0 (unquantized
        # estimate)
        G = self.G
        m = max(G.BSR_SELFCHECK_M) + 2  # with the two guard rows
        rows = []
        for r0 in range(0, m, 8192):
            xb = G._bsr_ramp_rows(r0, min(m, r0 + 8192), self.cpu).reshape(-1, 16, 32)
            e = G._ref_e8m0_exp(xb.abs().amax(-1))
            q = G._ref_q_e2m3(torch.clamp(xb / torch.exp2(e).unsqueeze(-1), -7.5, 7.5))
            rows.append(torch.cat([q.reshape(-1, 512), e.reshape(-1, 16)], 1))
        self.assertEqual(int(torch.unique(torch.cat(rows), dim=0).shape[0]), m)
        y = G._bsr_ramp_rows(0, 1024, self.cpu)
        x6, xs = G.mx6_activation_reference(y)
        self.assertGreaterEqual(int(torch.unique(xs).numel()), 24)
        blocks = y.view(1024, 16, 32).abs().amax(-1)
        self.assertGreater(int((blocks == 0).sum()), 0)
        self.assertTrue(bool((y < 0).any()) and bool((y > 0).any()))
        for n in _SITES:
            b = G._bsr_bias(n, self.cpu)
            self.assertEqual((b.dtype, tuple(b.shape)), (torch.bfloat16, (n,)))
            k = torch.arange(n, dtype=torch.int64)
            exact = (((k * 13) % 29 - 14).double() + 0.5) / 8.0 * torch.exp2(-4.0 * (k % 4).double())
            self.assertTrue(torch.equal(b.double(), exact))
            self.assertEqual(int(torch.count_nonzero(b)), n)
        w = G._ramp_weight(self.cpu, torch.float32)
        acc = y.double() @ w[:, 512:2048].double()
        q = torch.clamp(acc * 0.8125 + G._bsr_bias(1536, self.cpu).double(), -448, 448).float()
        codes = q.to(torch.float8_e4m3fn).view(torch.uint8) & 0x7F
        self.assertGreater(int((codes == 0x7E).sum()), 0)  # saturated
        self.assertGreater(int(((codes >> 3) == 0).sum() - (codes == 0).sum()), 0)  # subnormal
        self.assertGreater(int((codes == 0).sum()), 0)  # +-0


# =============================================================================================== GIN
class GinItemTest(unittest.TestCase):
    def test_gin_binds_once_with_item_text(self) -> None:
        # a string: "0" in the testbed form (default OFF), "1n" in the cand_v1b form (main's ruling), "1" in the
        # ship17 final form (r14 BSR production rung FLIP_OK at 2n and 4n; PREREG_bsrprod cross-scale RECOMMEND)
        lines = _gin_lines()
        i_bind = _binding_index(lines)
        bound = re.fullmatch(rf'apply_env_bootstrap\.{_KNOB} = "(0|1n|1)"', lines[i_bind])
        self.assertIsNotNone(bound, lines[i_bind])
        assert bound is not None
        names = [m.group(1) for m in (re.match(r"^\s*apply_env_bootstrap\.(\w+)\s*=", ln) for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        i_head = max(i for i in range(i_bind) if lines[i].startswith(_HEAD))
        block = lines[i_head:i_bind]
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = " ".join(" ".join(ln.lstrip("#").split()) for ln in block)
        self.assertIn(f'Code default OFF; bound "{bound.group(1)}" below', text)
        for other in sorted({"0", "1n", "1"} - {bound.group(1)}):
            self.assertNotIn(f'bound "{other}" below', text)
        if torch is not None:  # the binding is a valid knob value: off, or every site on a single node
            self.assertEqual(G_parse(bound.group(1)), (frozenset(), False, True) if bound.group(1) == "0"
                             else (frozenset(_SITES), bound.group(1) == "1n", True))
        for marker in ("(110) BSR (r14)", "item (92)", "WHAT:", "NUMERICS: BITWISE", "BUILD:", "Code default OFF",
                       "ops/hip/k512_mx6_bsr.cu", "ops/hip/k512_mx6_gemm.py", "register-resident weights",
                       "-DHK_N=1536 -DHK_K=512", "-DHK_N=1024 -DHK_K=512", "-DHK_N=512 -DBSR_U=1",
                       "-DBSR_S=16 -DBSR_SCL=1 -DBSR_VMS=1", "never -DK512MX6_RING=6", "libk512_mx6_wshadow.so",
                       "libk512_mx6_bsr_N1536.so, libk512_mx6_bsr_N1024.so, libk512_mx6_bsr_N512.so",
                       "before RUN_START", "no compile after RUN_START",
                       "synthetic inputs only: deterministic ramps, no dataset read, no model state and no RNG",
                       "two guard rows", '"BSR O-D2"', "PER-SITE FALLBACK", "never the FP8 path",
                       "never a failed prebuild",
                       '"[k512-mx6] BSR DECLINED N=<n>: <reason> (serving the shipped kernel) [rank N]"',
                       'INERT unless item (92) arms the MXFP6 forward: no build and no "[k512-mx6]" line',
                       f'"env bootstrap: ... {_KNOB}=<v>"', "ARMED (every rank, once, before RUN_START)",
                       '"[k512-mx6] BSR ARMED:', '" [rank N]"', 'ARM MARKER: the "[k512-mx6] prebuilt:" line',
                       "lds=102400", 'Must be ABSENT: "BSR DECLINED", "[k512-mx6] WARNING"',
                       '"[k512-mx6] prebuild FAILED"', "a comma list of 1536, 1024, 512", "Read once per process",
                       'TOPOLOGY: mode "1n" arms BSR only on a single-node job',
                       "LOCAL_WORLD_SIZE, else GPUS_PER_NODE", "the two must agree when both are set",
                       "the visible GPU count equals them",
                       "missing, unparsable or conflicting value counts as multi-node",
                       "at six M (64-row aligned and tail sizes)",
                       "a recoverable BSR failure (build, load, routing, self-check mismatch, a Python exception)",
                       "A device fault or a hang inside a BSR kernel during the self-check aborts or hangs the job "
                       "before RUN_START, as for any kernel",
                       "Read once per process at startup, like every knob",
                       "the loader's hipcc commands, library names, keys, kernels and log lines are the shipped ones",
                       "the bootstrap's own \"env bootstrap: ...\" line for this binding is the one new line",
                       "resident RCCL CTA", "the BSR schedule is static",
                       '"[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=<n> (LOCAL_WORLD_SIZE=<m>) [rank N]"',
                       'VALUES (a string, exported verbatim', '"1" every site at every topology',
                       '"1n" every site on a single-node job only',
                       'with "1n" on one node also "BSR OFF by gin mode 1n"',
                       'Off ("0", or the kill switch): no "[k512-mx6]" line names BSR',
                       f"Kill switch: env {_KNOB}=0", "pre-set env value wins over this binding"):
            self.assertIn(marker, text, marker)
        # no measured number in gin text (timings, ratios, percentages, register counts live in the evidence)
        self.assertIsNone(re.search(r"\d\.\d|%|\bms\b|\bVGPR", text), text)

    def test_item_follows_item_96(self) -> None:
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
    print("BSR_GIN_CHILD " + json.dumps({"env": {k: v for k, v in os.environ.items() if "MXFP6" in k},
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
    out = [x for x in r.stdout.splitlines() if x.startswith("BSR_GIN_CHILD ")]
    got = json.loads(out[0][len("BSR_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


class BootstrapTest(unittest.TestCase):
    def test_one_parameter_one_env_wins_bind(self) -> None:
        tree = ast.parse(_BOOT.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(names.count(_KNOB), 1)
        self.assertEqual(names.index(_KNOB), names.index(_RING6_KNOB) + 1)  # next to item (96)'s
        arg = next(a for a in fn.args.args if a.arg == _KNOB)
        self.assertEqual(ast.get_source_segment(_BOOT.read_text(), arg.annotation) if arg.annotation else None,
                         "Optional[str]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[_KNOB], ast.Constant) and defaults[_KNOB].value is None)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == _KNOB]
        self.assertEqual(len(calls), 1)
        (call,) = calls
        self.assertEqual(getattr(call.func, "id", None), "_bind_value")  # env wins; the string verbatim
        self.assertEqual([getattr(a, "id", None) for a in call.args[1:]], [_KNOB])

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_bootstrap_exports_and_env_wins(self) -> None:
        import inspect

        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

        self.assertIsNone(inspect.signature(eb.apply_env_bootstrap).parameters[_KNOB].default)
        cases = [  # (pre-set env, bound value, expected env after)
            (None, None, None),  # code default: unset = off
            (None, "0", "0"),  # the testbed binding
            (None, "1n", "1n"),  # the cand_v1b binding
            (None, "1", "1"),
            ("1n", "0", "1n"),  # a pre-set launcher value wins over gin (the in-model A/B arms it so)
            ("1", "0", "1"),
            ("1536,1024", "0", "1536,1024"),  # env-only site list
            ("0", "1n", "0"),  # the kill switch
        ]
        for preset, bound, want in cases:
            env = {} if preset is None else {_KNOB: preset}
            with mock.patch.dict(os.environ, env, clear=True):
                eb.apply_env_bootstrap(**{_KNOB: bound})
                self.assertEqual(os.environ.get(_KNOB), want, (preset, bound))

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_gin_run_exports_the_binding_and_env_wins(self) -> None:
        lines = _gin_lines()
        want = ast.literal_eval(lines[_binding_index(lines)].split("=", 1)[1].strip())  # "0", "1n" or "1"
        self.assertIn(want, ("0", "1n", "1"))
        for preset, expect in (({}, want), ({_KNOB: "0"}, "0"), ({_KNOB: "1"}, "1"), ({_KNOB: "1n"}, "1n"),
                               ({_KNOB: "1024"}, "1024")):
            rc, log, got = _run_child(preset)
            self.assertEqual(rc, 0, log)
            assert got is not None, log
            self.assertEqual(got["env"].get(_KNOB), expect, preset)
            self.assertFalse(got["torch_imported"])

    def test_launcher_forwards_the_prefix(self) -> None:
        if not _LAUNCHER.is_file():
            self.skipTest("no scripts/launch_slurm.sh in this checkout")
        self.assertTrue(_KNOB.startswith("DLRM_HSTU_FP8_"))
        self.assertIn('for v in "${!DLRM_HSTU_FP8_@}"; do', _LAUNCHER.read_text())


# =============================================================================================== SRC
def _bsr_code_text(source: str) -> str:
    """The code of a C/C++ source: comments removed (string and character literals kept verbatim, so a '//' or '/*'
    inside one is code; a '//' comment runs to the end of the line, across backslash-newline splices; a comment counts
    as a blank), then whitespace normalized outside literals (each blank run is one space, no blank at a line start
    or end, no empty line). ValueError on an unterminated literal or block comment."""
    out: List[str] = []

    def blank() -> None:
        if out and out[-1] not in (" ", "\n"):
            out.append(" ")

    i, n = 0, len(source)
    while i < n:
        c, two = source[i], source[i:i + 2]
        if two == "//":
            j = source.find("\n", i)
            while j > 0 and source[j - 1] == "\\":
                j = source.find("\n", j + 1)
            i = n if j < 0 else j
            blank()
        elif two == "/*":
            j = source.find("*/", i + 2)
            if j < 0:
                raise ValueError("unterminated block comment")
            i = j + 2
            blank()
        elif c in "\"'":
            j = i + 1
            while j < n and source[j] != c:
                if source[j] == "\n":
                    raise ValueError("unterminated literal")
                j += 2 if source[j] == "\\" else 1
            if j >= n:
                raise ValueError("unterminated literal")
            out.append(source[i:j + 1])
            i = j + 1
        elif c == "\n":
            if out and out[-1] == " ":
                out.pop()
            if out and out[-1] != "\n":
                out.append("\n")
            i += 1
        elif c.isspace():
            blank()
            i += 1
        else:
            out.append(c)
            i += 1
    if out and out[-1] == " ":
        out.pop()
    if out and out[-1] != "\n":
        out.append("\n")
    return "".join(out)


class SrcTest(unittest.TestCase):
    def test_source_is_the_shipped_bytes(self) -> None:
        self.assertEqual(hashlib.sha256(_BSR_CU.read_bytes()).hexdigest(), _SHA_BSR_CU)

    def test_revision_is_comment_only(self) -> None:
        # the stripper: literals are code (a '//' or '/*' inside one stays), comments, blank runs and empty lines go
        sample = 'int a = 1;  // c \\\n still c\n/* b\n */ const char* s = "//k/*  x"; char q = \'"\';\n\n  x(); \n'
        self.assertEqual(_bsr_code_text(sample), 'int a = 1;\nconst char* s = "//k/*  x"; char q = \'"\';\nx();\n')
        self.assertEqual(_bsr_code_text("a  /* x */ b // y\n"), "a b\n")
        with self.assertRaises(ValueError):
            _bsr_code_text('const char* s = "open;\n')
        # the guard finds line- or file-dependent code (not static_assert, not a comment)
        guard = _bsr_code_text('static_assert(a, "b");  // __LINE__\n#line 7\nassert(c); printf("%d", __FILE__);\n')
        self.assertEqual(_LINE_DEPENDENT.findall(guard), ["#line", "assert(", "printf(", "__FILE__"])
        # the shipped source carries the pinned code (the code verified bitwise on GPU but for two host impl() string
        # literals); none of it depends on a line position, so a comment-only revision builds the same device code
        code = _bsr_code_text(_BSR_CU.read_text())
        self.assertEqual(_LINE_DEPENDENT.findall(code), [])
        self.assertEqual(hashlib.sha256(code.encode("utf-8")).hexdigest(), _SHA_BSR_CU_CODE)

    def test_source_carries_the_switches_and_impl_tags(self) -> None:
        text = _BSR_CU.read_text()
        for guard in ('#error "define HK_N (1536 VQK, 1024 VK, or 512 with BSR_U=1)"',
                      '#error "BSR_U=1 (U bf16 + bias epilogue) requires HK_N=512"',
                      "#ifndef BSR_S\n#define BSR_S 16\n#endif", "#ifndef BSR_VMS\n#define BSR_VMS 0\n#endif",
                      "#ifndef BSR_SCL\n#define BSR_SCL 0\n#endif",
                      'extern "C" int K512MX6_CAT(k512_mx6_e4m3_bias_N, HK_N)(',
                      'extern "C" const char* k512_mx6_gemm_impl(void)', 'extern "C" int k512_mx6_u_bf16_bias(',
                      'extern "C" const char* k512_mx6_u_impl(void)'):
            self.assertIn(guard, text, guard)
        impls = re.findall(r'const char\* k512_mx6_(?:gemm|u)_impl\(void\) \{\s*return ((?:"[^"]*"\s*)+);', text)
        self.assertEqual(len(impls), 2)
        for impl in impls:
            joined = "".join(re.findall(r'"([^"]*)"', impl))
            self.assertIn("BSR O-D2", joined)
            self.assertNotIn("RING6", joined)
        # no ring switch (the macro is named in header comments only): an injected RING6 define changes nothing
        code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("//")]
        self.assertEqual([ln for ln in code if "K512MX6_RING" in ln], [])

    def test_loader_pins(self) -> None:
        src = _LOADER.read_text()
        self.assertIn('BSR_KNOB: str = KNOB + "_BSR"', src)
        self.assertIn('BSR_SWITCH: Tuple[str, ...] = ("-DBSR_S=16", "-DBSR_SCL=1", "-DBSR_VMS=1")', src)
        self.assertIn('BSR_IMPL: str = "BSR O-D2"', src)
        self.assertIn('BSR_SOURCE_NAME: str = "k512_mx6_bsr.cu"', src)


if __name__ == "__main__":
    unittest.main()
