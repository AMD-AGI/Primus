#!/usr/bin/env python3
# pyre-strict
"""DLRM_HSTU_FP8_FWD_MXFP6 (yambda_5b.gin item (92); DECLARED PRECISION CHANGE; default off).

CPU-only (nothing is built or launched):
* knob plumbing: the gin binding ships False, the bootstrap parameter exports "0" / "1" / nothing,
  a pre-set env value wins, launch_slurm.sh forwards the prefix, the adapter reads the knob at import;
* the producer CONTRACT: the adapter's producer call (ops/hip/mx6_fwd.py ``_producer_call``, the only
  place that builds it) binds the REAL ``ln_store_mx6`` signature -- from the real module
  (inspect.signature().bind() of the exact arguments, also through ``ln_store`` with a recorder) and,
  without torch, from its AST; the same pattern for pr7's GEMM module once it lands (skips until then);
* the adapter declines cleanly -- None, one log line per kind, no exception -- while pr7's GEMM module
  or the r14 producer is absent, and raises only under ..._STRICT=1;
* a failed prebuild latches every site off; the producer's own eligibility verdict declines a call
  without latching and its failure() latches; pr7's k512_mx6 must write `out` in place;
* weight shadows are keyed like the FP8 ones (id / data_ptr / layout / weight._version), and the audit
  catches a write that skipped the version bump;
* the first-call guard: row windows (head, 2^31 offsets, last tile), rel L2 / per-tile / slope /
  non-finite, acceptance, latching, a failing reference, and the device ARMED line only once every
  site is decided; the end-of-run counters line;
* the torch reference of the MX6 activation operand (ops/hip/mx6_ref.py) against an independent
  pure-Python encoder of the documented byte layout, and its once-rounded fp32 fma against exact
  rationals;
* knob-off structure of the call sites: every MX6 arm is `if _MX6_FWD and ...:` with the shipped
  statement as its `else`, the adapter is imported only under the knob, and the knob is defined below
  the module's @triton.jit kernels (their starting lines enter Triton's cache key).
The full base-vs-edited AST equality (fold the knob at False, compare with the shipped tree) is the
out-of-tree check probes/iter49/fp6fwd/integ/verify_knob_off.py of the r14 session."""

import ast
import contextlib
import importlib
import inspect
import io
import math
import os
import pathlib
import random
import struct
import sys
import types
import unittest
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

# recommendation/ (this file: recommendation/generative_recommenders/dlrm_v4/train/tests/)
_REPO = pathlib.Path(__file__).resolve().parents[4]
_GR = _REPO / "generative_recommenders"
_GIN = _GR / "dlrm_v4/train/gin/yambda_5b.gin"
_BOOT = _GR / "dlrm_v4/train/_env_bootstrap.py"
_CALL_SITE = _GR / "ops/triton/triton_hstu_preprocess_and_attention.py"
_TRITON_FP8 = _GR / "ops/triton/triton_fp8.py"
_ADAPTER = _GR / "ops/hip/mx6_fwd.py"
_PRODUCER_FILE = _GR / "ops/hip/ln_store_mx6.py"
_LAUNCHER = _REPO / "scripts/launch_slurm.sh"
_KNOB = "DLRM_HSTU_FP8_FWD_MXFP6"
_BINDING = f"apply_env_bootstrap.{_KNOB} = True"  # ship gin flip v3 (r14 iter57 DRAFT companion)
_ABSENT_GEMM = "generative_recommenders.ops.hip._mx6_absent_gemm_for_test"
_ABSENT_PRODUCER = "generative_recommenders.ops.hip._mx6_absent_producer_for_test"
_REAL_PRODUCER = "generative_recommenders.ops.hip.ln_store_mx6"
_REAL_GEMM = "generative_recommenders.ops.hip.k512_mx6_gemm"
_FAKE_GEMM = "_mx6_fwd_test_fake_gemm"
_FAKE_PRODUCER = "_mx6_fwd_test_fake_producer"


def _try_import(name: str) -> Optional[Any]:  # pyre-ignore[3]
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - environment specific
        return None


torch = _try_import("torch")
gin = _try_import("gin")


def _mx6() -> Any:  # pyre-ignore[3]
    from generative_recommenders.ops.hip import mx6_fwd

    return mx6_fwd


@contextlib.contextmanager
def _capture():  # pyre-ignore[3]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


# =============================================================================================== knob plumbing
class KnobPlumbingTest(unittest.TestCase):
    def test_gin_ships_the_binding_on(self) -> None:
        txt = _GIN.read_text()
        lines = [ln for ln in txt.splitlines() if ln.startswith(f"apply_env_bootstrap.{_KNOB} ")]
        self.assertEqual(lines, [_BINDING])
        self.assertIn("# (92) MXFP6 forward K=512 trio", txt)
        item = txt[txt.index("# (92) MXFP6 forward K=512 trio"): txt.index(_BINDING)]
        flat = " ".join(" ".join(line.lstrip("#").split()) for line in item.splitlines())
        for needle in ("DECLARED", "PRECISION CHANGE", "ARMED", "Must be ABSENT", "Kill switch",
                       '"[mx6-fwd] DECLINED"', "-> REJECTED", "DLRM_HSTU_FP8_FWD_MXFP6=0",
                       "[ln-store-mx6] prebuilt in", "ARMED wgrad_ck xt_tiled producer ln_store_mx6",
                       "[ln-store-mx6] prebuild FAILED", "[mx6-fwd] counters at exit",
                       "pool created lazily", "no TM_NONEXP pool block"):
            self.assertIn(needle, flat, needle)

    def test_bootstrap_source_binds_the_knob(self) -> None:
        src = _BOOT.read_text()
        self.assertIn(f"{_KNOB}: Optional[bool] = None,", src)
        self.assertEqual(src.count(f'_bind_bool("{_KNOB}", {_KNOB})'), 1)

    def test_launcher_forwards_the_prefix(self) -> None:
        if not _LAUNCHER.exists():
            self.skipTest("no scripts/launch_slurm.sh in this checkout")
        self.assertIn('for v in "${!DLRM_HSTU_FP8_@}"; do', _LAUNCHER.read_text())

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_bootstrap_exports_and_env_wins(self) -> None:
        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

        param = inspect.signature(eb.apply_env_bootstrap).parameters[_KNOB]
        self.assertIsNone(param.default)
        cases = [  # (pre-set env, bound value, expected env after)
            (None, None, None),
            (None, False, "0"),
            (None, True, "1"),
            ("1", False, "1"),  # a pre-set launcher value wins over gin
            ("0", True, "0"),
        ]
        for preset, bound, want in cases:
            env = {} if preset is None else {_KNOB: preset}
            with mock.patch.dict(os.environ, env, clear=True):
                eb.apply_env_bootstrap(**{_KNOB: bound})
                self.assertEqual(os.environ.get(_KNOB), want, (preset, bound))

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_shipped_gin_line_parses_and_exports_one(self) -> None:
        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

        gin.clear_config()
        try:
            gin.parse_config(_BINDING)
            with mock.patch.dict(os.environ, {}, clear=True):
                eb.apply_env_bootstrap()
                self.assertEqual(os.environ.get(_KNOB), "1")
        finally:
            gin.clear_config()

    @unittest.skipIf(torch is None, "torch not installed")
    def test_adapter_reads_the_knob_at_import(self) -> None:
        mx6 = _mx6()
        keep = {k: v for k, v in os.environ.items() if not k.startswith(_KNOB)}
        try:
            for value, want in ((None, False), ("0", False), ("true", False), ("1", True)):
                env = dict(keep)
                if value is not None:
                    env[_KNOB] = value
                with mock.patch.dict(os.environ, env, clear=True), _capture() as out, \
                        mock.patch("atexit.register") as register:
                    importlib.reload(mx6)
                self.assertEqual(mx6.ENABLED, want, value)
                self.assertEqual("[mx6-fwd] requested:" in out.getvalue(), want, value)
                self.assertEqual(register.called, want, value)  # the end-of-run counters line
        finally:
            importlib.reload(mx6)  # back to the process environment


# =============================================================================================== contract
def _signature_from_ast(path: pathlib.Path, name: str) -> inspect.Signature:
    """inspect.Signature of a top-level function, from source (no import, no torch)."""
    fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    a = fn.args
    params: List[inspect.Parameter] = []
    positional = a.posonlyargs + a.args
    first_default = len(positional) - len(a.defaults)
    for i, arg in enumerate(positional):
        kind = (inspect.Parameter.POSITIONAL_ONLY if i < len(a.posonlyargs)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD)
        default = None if i >= first_default else inspect.Parameter.empty
        params.append(inspect.Parameter(arg.arg, kind, default=default))
    if a.vararg is not None:
        params.append(inspect.Parameter(a.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        params.append(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                                        default=inspect.Parameter.empty if default is None else None))
    if a.kwarg is not None:
        params.append(inspect.Parameter(a.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(params)


def _adapter_producer_call_shape() -> Tuple[int, Tuple[str, ...]]:
    """(number of positional arguments, every keyword) of mx6_fwd's producer call, from source."""
    tree = ast.parse(_ADAPTER.read_text())
    kw: Tuple[str, ...] = ()
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_PRODUCER_KW":
            kw = tuple(ast.literal_eval(node.value))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_producer_call")
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    assert isinstance(ret.value, ast.Tuple) and isinstance(ret.value.elts[0], ast.Tuple)
    return len(ret.value.elts[0].elts), kw


class ProducerContractAstTest(unittest.TestCase):
    """No torch: the producer's ln_store_mx6 signature (from its source) binds the adapter's
    positional count and every keyword the adapter may pass (REVIEW B1: `xt8=` vs `xt_out=`)."""

    def setUp(self) -> None:
        if not _PRODUCER_FILE.exists():
            self.skipTest("the r14 producer (ops/hip/ln_store_mx6.py) is not in this tree")

    def test_adapter_keywords_bind_the_producer_signature(self) -> None:
        sig = _signature_from_ast(_PRODUCER_FILE, "ln_store_mx6")
        npos, kw = _adapter_producer_call_shape()
        self.assertEqual(npos, 6)
        self.assertIn("xt_out", kw)
        self.assertNotIn("xt8", kw)
        sig.bind(*range(npos), **{k: None for k in kw})  # TypeError = the contract is broken
        for k in kw:
            sig.bind(*range(npos), **{k: None})

    def test_required_api_is_defined(self) -> None:
        names = {n.name for n in ast.parse(_PRODUCER_FILE.read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        tree = ast.parse(_ADAPTER.read_text())
        required = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.AnnAssign)
                        and isinstance(n.target, ast.Name) and n.target.id == "_REQUIRED_API")
        for attr in required["producer"]:
            self.assertIn(attr, names)


def _fake_outputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    """What the real ln_store_mx6 returns for these arguments (shapes / dtypes / identities), on CPU."""
    assert torch is not None
    x = args[0]
    m = int(x.shape[0])
    t = (m + 127) // 128
    e4 = torch.float8_e4m3fn

    def get(key: str, make: Any) -> Any:  # pyre-ignore[2,3]
        value = kwargs.get(key)
        return make() if value is None else value

    x6 = get("x6_out", lambda: torch.zeros((t, 4, 12288), dtype=torch.uint8))
    xs = get("xs_out", lambda: torch.full((t, 4, 512), 127, dtype=torch.uint8))
    xt = get("xt_out", lambda: torch.zeros((t, 512, 128), dtype=torch.uint8).view(e4))
    mean = get("mean_out", lambda: torch.zeros(m))
    rstd = get("rstd_out", lambda: torch.ones(m))
    want_x8 = kwargs.get("want_x8", False) or kwargs.get("x8_out") is not None
    x8 = get("x8_out", lambda: torch.zeros((m, 512), dtype=torch.uint8).view(e4)) if want_x8 else None
    return (x6, xs, xt, mean, rstd, x8)


@unittest.skipIf(torch is None, "torch not installed")
class ProducerContractTest(unittest.TestCase):
    """The REAL producer module (its import needs only torch and builds nothing): every call the
    adapter makes binds its signature."""

    def setUp(self) -> None:
        try:
            self.L6 = importlib.import_module(_REAL_PRODUCER)
        except ImportError as error:
            self.skipTest(f"the r14 producer is not importable in this tree ({error})")
        self.mx6 = _mx6()
        m = self.mx6
        saved = (m.ENABLED, m.GEMM_MODULE, m.PRODUCER_MODULE, m.STRICT, m.SHADOW_AUDIT)
        self.addCleanup(lambda: m.reset_for_testing(enabled=saved[0], gemm_module=saved[1],
                                                    producer_module=saved[2], strict=saved[3],
                                                    shadow_audit=saved[4]))
        m.reset_for_testing(enabled=True, gemm_module=_ABSENT_GEMM, producer_module=_REAL_PRODUCER,
                            strict=False, shadow_audit=0)
        self.x = torch.zeros((300, 512), dtype=torch.bfloat16)
        self.w = torch.ones(512, dtype=torch.bfloat16)
        self.b = torch.zeros(512, dtype=torch.bfloat16)

    def test_required_api_present(self) -> None:
        for attr in self.mx6._REQUIRED_API["producer"]:
            self.assertTrue(callable(getattr(self.L6, attr, None)), attr)
        self.assertIsNone(self.mx6.module_error("producer"))

    def test_every_adapter_call_binds_the_real_signature(self) -> None:
        sig = inspect.signature(self.L6.ln_store_mx6)
        e4 = torch.float8_e4m3fn
        for want_x8 in (False, True):
            for xt in (None, torch.empty((3, 512, 128), dtype=e4)):
                for poison in (False, True):
                    outs = (self.mx6._poisoned_outputs(300, 384, torch.device("cpu"), False, want_x8)
                            if poison else None)
                    args, kwargs = self.mx6._producer_call(self.x, self.w, self.b, 1e-6, 45.5, 384,
                                                           want_x8=want_x8, xt8=xt, outs=outs)
                    sig.bind(*args, **kwargs)
                    self.assertNotIn("xt8", kwargs)
                    self.assertEqual("xt_out" in kwargs, xt is not None)
                    self.assertLessEqual(set(kwargs), set(self.mx6._PRODUCER_KW))

    def test_adapter_ln_store_reaches_the_real_signature(self) -> None:
        sig = inspect.signature(self.L6.ln_store_mx6)
        seen: List[List[str]] = []

        def recorder(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:  # pyre-ignore[2]
            sig.bind(*args, **kwargs)  # TypeError = the adapter broke the contract
            seen.append(sorted(kwargs))
            return _fake_outputs(args, kwargs)

        mx6 = self.mx6
        with mock.patch.object(self.L6, "ln_store_mx6", recorder):
            p = mx6.ln_store(self.x, self.w, self.b, 1e-6, 45.5, want_xt8=True, want_x8=True,
                             kind="selfcheck", xt_empty=mx6._plain_empty)
            self.assertEqual(tuple(p.xt8.shape), (3, 512, 128))
            p = mx6.ln_store(self.x, self.w, self.b, 1e-6, 45.5, want_xt8=False, want_x8=False,
                             kind="selfcheck")
            self.assertIsNone(p.xt8)
            p = mx6.ln_store(self.x, self.w, self.b, 1e-6, 45.5, want_xt8=True, want_x8=False,
                             kind="check", poison=True)
        self.assertEqual(seen, [["want_x8", "xt_out"], ["want_x8"],
                                ["mean_out", "rstd_out", "want_x8", "x6_out", "xs_out", "xt_out"]])

    def test_eligibility_failure_kernel_prebuild_signatures(self) -> None:
        inspect.signature(self.L6.ineligible_reason).bind(self.x, self.w, self.b)
        inspect.signature(self.L6.failure).bind()
        inspect.signature(self.L6.kernel).bind()
        inspect.signature(self.L6.prebuild).bind()
        # CPU operands: the producer's own verdict is a reason, which the adapter turns into a decline
        self.assertIsNotNone(self.mx6._producer_operand_reason(self.L6, self.x, self.w, self.b))
        fields = getattr(self.L6.Kernel, "__dataclass_fields__", {})
        for name in ("impl", "source", "source_sha256"):  # what _producer_desc logs
            self.assertIn(name, fields)


@unittest.skipIf(torch is None, "torch not installed")
class GemmContractTest(unittest.TestCase):
    """pr7's MX6 GEMM module: the adapter's calls bind its real signatures. Skips until the module
    (ops/hip/k512_mx6_gemm.py) lands in the tree."""

    def setUp(self) -> None:
        try:
            self.G = importlib.import_module(_REAL_GEMM)
        except ImportError as error:
            self.skipTest(f"pr7's MX6 GEMM module is not in this tree yet ({error})")
        self.mx6 = _mx6()

    def test_required_api_and_calls_bind(self) -> None:
        mx6 = self.mx6
        for attr in mx6._REQUIRED_API["gemm"]:
            self.assertTrue(hasattr(self.G, attr), attr)
        u8 = torch.uint8
        x6 = torch.zeros((3, 4, 12288), dtype=u8)
        xs = torch.full((3, 4, 512), 127, dtype=u8)
        w6 = torch.zeros((6, 2, 4, 12288), dtype=u8)
        ws = torch.full((48, 4, 4), 127, dtype=u8)
        one = torch.ones((1, 1))
        out = torch.empty((300, 1536), dtype=torch.float8_e4m3fn)
        args, kwargs = mx6._gemm_call(x6, xs, w6, ws, torch.zeros(1536, dtype=torch.bfloat16), one, one,
                                      300, 1536, out)
        inspect.signature(self.G.k512_mx6).bind(*args, **kwargs)
        inspect.signature(self.G.mx6_weight).bind(torch.zeros((512, 2048)), ((512, 2048),))
        inspect.signature(self.G.kernel).bind(1536)
        if hasattr(self.G, "prebuild"):
            inspect.signature(self.G.prebuild).bind()


# =============================================================================================== adapter
def _ln_args(**kw: Any) -> Dict[str, Any]:
    assert torch is not None
    args = dict(
        x=torch.zeros((300, 512), dtype=torch.bfloat16),
        norm_weight=torch.ones(512, dtype=torch.bfloat16),
        norm_bias=torch.zeros(512, dtype=torch.bfloat16),
        eps=1e-6,
        fp8_scale_value=0.01,
        uvqk_weight=torch.zeros((512, 2048)),
        uvqk_bias=torch.zeros(2048),
        qkv_frozen=True,
        grad=True,
        store_t=True,
        pad_to=128,
        tiled_xt=True,
    )
    args.update(kw)
    return args


@unittest.skipIf(torch is None, "torch not installed")
class _AdapterCase(unittest.TestCase):
    gemm_module: str = _ABSENT_GEMM
    producer_module: str = _ABSENT_PRODUCER

    def setUp(self) -> None:
        self.mx6 = _mx6()
        m = self.mx6
        saved = (m.ENABLED, m.GEMM_MODULE, m.PRODUCER_MODULE, m.STRICT, m.SHADOW_AUDIT)
        self.addCleanup(self._restore, saved)
        m.reset_for_testing(enabled=True, gemm_module=self.gemm_module,
                            producer_module=self.producer_module, strict=False, shadow_audit=0)

    def _restore(self, saved: Tuple[Any, ...]) -> None:
        enabled, gemm, producer, strict, audit = saved
        self.mx6.reset_for_testing(enabled=enabled, gemm_module=gemm, producer_module=producer,
                                   strict=strict, shadow_audit=audit)

    def uvqk(self, **kw: Any) -> Any:  # pyre-ignore[3]
        args = _ln_args(**kw)
        args.setdefault("u_columns", 512)
        args.setdefault("pack_saved", True)
        return self.mx6.uvqk_ln_store(**args)

    def vk(self, **kw: Any) -> Any:  # pyre-ignore[3]
        args = _ln_args(**kw)
        args.setdefault("vk_layout", ((512, 1024), (1536, 2048)))
        args.setdefault("pack_handed", True)
        return self.mx6.vk_ln_store(**args)

    @contextlib.contextmanager
    def operands_ok(self):  # pyre-ignore[3]
        """CPU tensors: let a call pass the device / scale-mode / xT-layout checks."""
        m = self.mx6
        with mock.patch.object(m, "_operand_reason", return_value=None), \
                mock.patch.object(m, "_scale_mode_reason", return_value=None), \
                mock.patch.object(m, "_xt_layout_reason", return_value=None):
            yield


class AdapterDeclineTest(_AdapterCase):
    """pr7's GEMM module and the r14 producer are absent (module names that do not exist)."""

    def test_not_ready_names_the_missing_module(self) -> None:
        cpu = torch.device("cpu")
        reason = self.mx6.not_ready_reason(cpu)
        self.assertIsNotNone(reason)
        self.assertIn("not importable", reason)
        self.assertFalse(self.mx6.is_ready(cpu))
        self.assertIn("not importable", self.mx6.site_reason(cpu, "VQK"))

    def test_ln_arms_decline_once_and_return_none(self) -> None:
        with _capture() as out:
            self.assertIsNone(self.uvqk())
            self.assertIsNone(self.uvqk())
            self.assertIsNone(self.vk())
        lines = [ln for ln in out.getvalue().splitlines() if "DECLINED" in ln]
        self.assertEqual(len(lines), 1, out.getvalue())  # one line per kind (the producer module)
        self.assertIn(_ABSENT_PRODUCER, lines[0])
        self.assertIn("-> shipped FP8 path", lines[0])
        self.assertEqual(self.mx6.counters().get("declined"), 3)

    def test_expected_skips_are_not_declines(self) -> None:
        with _capture() as out:
            self.assertIsNone(self.uvqk(qkv_frozen=False))  # calibration window
            self.assertIsNone(self.uvqk(grad=False))  # eval / no-grad forward
            self.assertIsNone(self.vk(qkv_frozen=False))
        text = out.getvalue()
        self.assertNotIn("DECLINED", text)
        self.assertEqual(text.count("not engaged (expected)"), 3, text)

    def test_call_structure_declines(self) -> None:
        with _capture() as out:
            self.assertIsNone(self.uvqk(store_t=False))
            self.assertIsNone(self.uvqk(pack_saved=False))
            self.assertIsNone(self.vk(pack_handed=False))
            self.assertIsNone(self.uvqk(u_columns=256))
        text = out.getvalue()
        for kind in ("uvqk:store_t", "uvqk:pack", "vk:pack", "uvqk:shape"):
            self.assertIn(kind, text)

    def test_strict_raises(self) -> None:
        self.mx6.reset_for_testing(strict=True)
        with _capture(), self.assertRaises(self.mx6.StrictError):
            self.uvqk()

    def test_prebuild_declines_without_modules(self) -> None:
        with _capture() as out:
            self.assertFalse(self.mx6.prebuild())
        self.assertIn("prebuild DECLINED", out.getvalue())
        self.mx6.reset_for_testing(strict=True)
        with _capture(), self.assertRaises(self.mx6.PrebuildError):
            self.mx6.prebuild()

    def test_knob_off_is_silent_and_inert(self) -> None:
        self.mx6.reset_for_testing(enabled=False)
        with _capture() as out:
            self.assertIsNone(self.uvqk())
            self.assertIsNone(self.vk())
            self.assertFalse(self.mx6.prebuild())
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(self.mx6.counters(), {})


# ------------------------------------------------------------------------------------------- fake modules
class _FakeKernel:
    impl = "fake"
    source = "/fake/ln_store_mx6.cu"
    source_sha256 = "f" * 64


def _install_fakes() -> Tuple[types.ModuleType, types.ModuleType]:
    """API-complete stand-ins for pr7's GEMM module and the r14 producer (CPU, no math). The
    producer has the REAL ln_store_mx6 signature (REVIEW B1)."""
    assert torch is not None
    gemm = types.ModuleType(_FAKE_GEMM)
    gemm.SUPPORTED_N = (1024, 1536)  # no U yet
    gemm.calls = []  # pyre-ignore[16]

    def kernel(n: int) -> str:
        return f"fake-k512-mx6-N{n}"

    def mx6_weight(weight: Any, layout: Any) -> Tuple[Any, Any, Any]:  # pyre-ignore[2]
        gemm.calls.append(("mx6_weight", tuple(layout), int(weight._version)))
        n = sum(b - a for a, b in layout)
        return (torch.zeros((n // 256, 2, 4, 12288), dtype=torch.uint8),
                torch.full((n // 32, 4, 4), 127, dtype=torch.uint8),
                torch.ones((1, 1)))

    def k512_mx6(x6: Any, xs: Any, w6: Any, ws: Any, bias: Any, sa: Any, sb: Any,  # pyre-ignore[2]
                 m: int, n: int, out: Any = None) -> Any:
        gemm.calls.append(("k512_mx6", m, n))
        return out

    gemm.kernel = kernel  # pyre-ignore[16]
    gemm.mx6_weight = mx6_weight  # pyre-ignore[16]
    gemm.k512_mx6 = k512_mx6  # pyre-ignore[16]

    prod = types.ModuleType(_FAKE_PRODUCER)
    prod.calls = []  # pyre-ignore[16]
    prod.reason = None  # pyre-ignore[16]
    prod.fail = None  # pyre-ignore[16]

    def ln_store_mx6(x: Any, weight: Any, bias: Any, eps: float, inv_scale: float,  # pyre-ignore[2]
                     mp: Optional[int] = None, *, want_x8: bool = False, return_y: bool = False,
                     xt_out: Any = None, x6_out: Any = None, xs_out: Any = None, mean_out: Any = None,
                     rstd_out: Any = None, x8_out: Any = None, y_out: Any = None) -> Tuple[Any, ...]:
        kwargs = dict(want_x8=want_x8, xt_out=xt_out, x6_out=x6_out, xs_out=xs_out, mean_out=mean_out,
                      rstd_out=rstd_out, x8_out=x8_out)
        prod.calls.append(sorted(k for k, v in kwargs.items() if v is not None and k != "want_x8"))
        return _fake_outputs((x,), kwargs)

    prod.ln_store_mx6 = ln_store_mx6  # pyre-ignore[16]
    prod.ineligible_reason = lambda x, weight, bias: prod.reason  # pyre-ignore[16]
    prod.failure = lambda: prod.fail  # pyre-ignore[16]
    prod.kernel = lambda: _FakeKernel  # pyre-ignore[16]
    prod.prebuild = lambda: True  # pyre-ignore[16]
    sys.modules[_FAKE_GEMM] = gemm
    sys.modules[_FAKE_PRODUCER] = prod
    return gemm, prod


class AdapterFakeModulesTest(_AdapterCase):
    gemm_module = _FAKE_GEMM
    producer_module = _FAKE_PRODUCER

    def setUp(self) -> None:
        self.gemm, self.prod = _install_fakes()
        self.addCleanup(sys.modules.pop, _FAKE_GEMM, None)
        self.addCleanup(sys.modules.pop, _FAKE_PRODUCER, None)
        super().setUp()

    def plan(self, m: int = 256) -> Any:  # pyre-ignore[3]
        return self.mx6.Plan(kind="uvqk", device=torch.device("cpu"), m=m,
                             x6=torch.zeros(((m + 127) // 128, 4, 12288), dtype=torch.uint8),
                             xs=torch.full(((m + 127) // 128, 4, 512), 127, dtype=torch.uint8),
                             xt8=None, mean=torch.zeros(m), rstd=torch.ones(m), x8=None, wkey=(11, 22))

    def test_readiness_per_site(self) -> None:
        cpu = torch.device("cpu")
        self.assertTrue(self.mx6.is_ready(cpu, (1536, 1024)))
        self.assertIsNone(self.mx6.site_reason(cpu, "VQK"))
        self.assertIsNone(self.mx6.site_reason(cpu, "VK"))
        self.assertIn("N=512 not in", self.mx6.site_reason(cpu, "U"))  # U stays FP8

    def test_operand_checks_follow_module_checks(self) -> None:
        with _capture() as out:
            self.assertIsNone(self.uvqk())  # CPU tensors: modules pass, the device check declines
        self.assertIn("uvqk:device", out.getvalue())

    # ---- REVIEW m1: the producer's own eligibility verdict and failure() ----------------------
    def test_producer_ineligible_declines_the_call_without_latching(self) -> None:
        from generative_recommenders.ops.triton import wgrad_ck

        self.prod.reason = "x / weight / bias not 16-byte aligned"
        with self.operands_ok(), _capture() as out:
            self.assertIsNone(self.uvqk())
        self.assertIn("uvqk:producer_operands", out.getvalue())
        self.assertIn("16-byte aligned", out.getvalue())
        self.assertEqual(self.mx6._producer_error, {})
        self.assertEqual(self.prod.calls, [])  # declined before the producer ran
        self.prod.reason = None
        with self.operands_ok(), mock.patch.object(self.mx6, "_producer_check", return_value=None), \
                mock.patch.object(wgrad_ck, "note_tiled_producer"), _capture():
            plan = self.uvqk()
        self.assertIsNotNone(plan)
        self.assertIsNotNone(plan.x8)  # the guard is pending, and U stays FP8
        self.assertEqual(self.prod.calls, [["xt_out"]])  # the step-tiled xT8 the adapter owns

    def test_producer_failure_latches_the_producer(self) -> None:
        cpu = torch.device("cpu")
        self.prod.fail = "ln_store_mx6 library unavailable: hipcc rc=1"
        self.assertIn("producer latched off by itself", self.mx6.site_reason(cpu, "VQK"))
        self.assertIn("hipcc rc=1", self.mx6.not_ready_reason(cpu))

    # ---- REVIEW M4: a failed prebuild latches every site --------------------------------------
    def test_failed_prebuild_latches_every_site(self) -> None:
        cpu = torch.device("cpu")
        with mock.patch.object(self.mx6, "_prebuild", return_value="producer self-check on ramp: {'x6': 3}"), \
                _capture() as out:
            self.assertFalse(self.mx6.prebuild())
        self.assertIn("prebuild DECLINED", out.getvalue())
        self.assertIn("latched for this process", out.getvalue())
        for site in ("U", "VQK", "VK"):
            self.assertIn("prebuild failed", self.mx6.site_reason(cpu, site))
        with self.operands_ok(), _capture() as out:
            self.assertIsNone(self.uvqk())
            self.assertIsNone(self.vk())
        self.assertIn("uvqk:VQK", out.getvalue())
        self.assertIn("vk:VK", out.getvalue())
        self.assertEqual(self.prod.calls, [])

    def test_prebuild_exception_latches_and_skips_do_not(self) -> None:
        cpu = torch.device("cpu")
        with mock.patch.object(self.mx6, "_prebuild", return_value="skipped: no GPU"), _capture() as out:
            self.assertFalse(self.mx6.prebuild())
        self.assertNotIn("DECLINED", out.getvalue())
        self.assertIsNone(self.mx6.site_reason(cpu, "VQK"))
        with mock.patch.object(self.mx6, "_prebuild", side_effect=TypeError("boom")), \
                mock.patch.object(torch.cuda, "is_available", return_value=False), _capture() as out:
            self.assertFalse(self.mx6.prebuild())
        self.assertIn("prebuild DECLINED: TypeError: boom", out.getvalue())
        self.assertIn("prebuild failed", self.mx6.site_reason(cpu, "VK"))
        self.mx6.reset_for_testing(strict=True)
        with mock.patch.object(self.mx6, "_prebuild", return_value="GEMM N=1536 self-check on zeros wrote 7 nonzero bytes"), \
                _capture(), self.assertRaises(self.mx6.PrebuildError):
            self.mx6.prebuild()
        self.assertIn("prebuild failed", self.mx6.site_reason(cpu, "VQK"))

    def test_import_time_gates_must_agree(self) -> None:
        fake = types.ModuleType("generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention")
        fake._MX6_FWD = False  # pyre-ignore[16]
        with mock.patch.dict(sys.modules, {fake.__name__: fake}), _capture() as out:
            self.assertIsNotNone(self.mx6._gate_reason())
        self.assertIn("ERROR import-time gates", out.getvalue())

    # ---- REVIEW M3: the self-check / check buffers never come from the TM_NONEXP pool ----------
    def test_poisoned_calls_never_use_the_tm_pool(self) -> None:
        from generative_recommenders.ops.triton import fp8_tm_alloc

        x, w, b = _ln_args()["x"], torch.ones(512, dtype=torch.bfloat16), torch.zeros(512, dtype=torch.bfloat16)
        with mock.patch.object(fp8_tm_alloc, "empty_xt", side_effect=AssertionError("TM pool")):
            p = self.mx6.ln_store(x, w, b, 1e-6, 1.0, want_xt8=True, want_x8=True, kind="selfcheck", poison=True)
            self.assertEqual(tuple(p.xt8.shape), (3, 512, 128))
            seen = []
            p = self.mx6.ln_store(x, w, b, 1e-6, 1.0, want_xt8=True, kind="check",
                                  xt_empty=lambda *a: seen.append(a) or self.mx6._plain_empty(*a))
            self.assertEqual(len(seen), 1)
        self.assertEqual(self.prod.calls[0], ["mean_out", "rstd_out", "x6_out", "x8_out", "xs_out", "xt_out"])

    def test_pad_to_other_than_128_declines(self) -> None:
        reason = self.mx6._xt_layout_reason(True, 3072, pad_to=64)
        self.assertIsNotNone(reason)
        self.assertIn("pad_to=64", reason)

    # ---- REVIEW m3: pr7's k512_mx6 must write `out` -------------------------------------------
    def test_launch_requires_the_output_written_in_place(self) -> None:
        w = torch.nn.Parameter(torch.randn(512, 2048))
        plan = self.plan()
        self.assertIsNotNone(self.mx6._module("gemm"))
        e4 = torch.float8_e4m3fn
        bias = torch.zeros(1536, dtype=torch.bfloat16)
        out = self.mx6._launch("VQK", plan, w, ((512, 2048),), bias, torch.ones((1, 1)), e4)
        self.assertEqual((tuple(out.shape), out.dtype), ((256, 1536), e4))
        self.gemm.k512_mx6 = lambda *a, **k: torch.zeros((256, 1536), dtype=e4)  # pyre-ignore[16]
        with self.assertRaises(RuntimeError):
            self.mx6._launch("VQK", plan, w, ((512, 2048),), bias, torch.ones((1, 1)), e4)

    # ---- weight shadows -------------------------------------------------------------------------
    def test_weight_shadow_is_version_keyed(self) -> None:
        w = torch.nn.Parameter(torch.randn(512, 2048))
        vqk, vk = ((512, 2048),), ((512, 1024), (1536, 2048))
        a = self.mx6.weight_shadow(w, vqk)
        b = self.mx6.weight_shadow(w, vqk)
        self.assertTrue(all(x is y for x, y in zip(a, b)))  # hit: the same tensors
        self.mx6.weight_shadow(w, vk)  # another layout: its own entry
        self.assertEqual([c[0] for c in self.gemm.calls].count("mx6_weight"), 2)
        with torch.no_grad():
            w.add_(1.0)  # an autograd-visible in-place update bumps _version
        c = self.mx6.weight_shadow(w, vqk)
        self.assertFalse(c[0] is a[0])
        self.assertEqual([c[0] for c in self.gemm.calls].count("mx6_weight"), 3)
        w2 = torch.nn.Parameter(w.detach().clone())  # same shape, another tensor: its own key
        self.mx6.weight_shadow(w2, vqk)
        self.assertEqual([c[0] for c in self.gemm.calls].count("mx6_weight"), 4)

    def test_shadow_audit_catches_an_unversioned_write(self) -> None:
        self.mx6.reset_for_testing(shadow_audit=1)
        w = torch.nn.Parameter(torch.randn(512, 2048))
        layout = ((512, 2048),)
        self.mx6.weight_shadow(w, layout)
        version = w._version
        w.data.add_(1.0)  # raw write: the Parameter's version counter does not move
        if w._version != version:  # pragma: no cover - torch semantics changed
            self.skipTest(".data writes bump the version on this torch")
        with _capture() as out:
            self.mx6.weight_shadow(w, layout)  # a hit -> audited
        self.assertIn("STALE weight shadow", out.getvalue())
        self.assertIn("latched", self.mx6.site_reason(torch.device("cpu"), "VQK"))

    # ---- REVIEW m4: the guard -------------------------------------------------------------------
    def test_guard_windows(self) -> None:
        mx6 = self.mx6
        m = 2_376_002
        last = (m // 128 * 128, m)
        for n, item, boundary in ((1536, 1, 1_398_101), (1024, 1, 2_097_152), (512, 2, 2_097_152)):
            w = mx6.guard_windows(m, n, item)
            self.assertEqual(w[0], (0, mx6.GUARD_HEAD_ROWS))
            self.assertEqual(w[-1], last)
            self.assertTrue(any(a <= boundary < b for a, b in w), (n, w))
            self.assertTrue(all(a % 128 == 0 and (b % 128 == 0 or b == m) for a, b in w), w)
            self.assertLess(sum(b - a for a, b in w), 66_000)  # bounded memory at production M
        self.assertEqual(mx6.guard_windows(3000, 1536, 1), [(0, 3000)])
        self.assertEqual(mx6.guard_windows(0, 1536, 1), [])
        cw = mx6.check_windows(m)
        self.assertEqual((cw[0], cw[-1]), ((0, mx6.CHECK_HEAD_ROWS), last))
        self.assertTrue(any(a <= 2_097_152 < b for a, b in cw))  # byte 2^31 of the bf16 input
        self.assertEqual(mx6.check_windows(4097), [(0, 4097)])

    def test_guard_metrics(self) -> None:
        mx6 = self.mx6
        g = torch.Generator().manual_seed(58)
        ref = (torch.randn((8192 + 77, 64), generator=g) + 0.5).to(torch.bfloat16)
        rel, tile, finite, slope = mx6.guard_metrics(ref, ref, chunk_rows=1024)
        self.assertEqual((rel, tile, finite, slope), (0.0, 0.0, True, 1.0))
        noisy = (ref.float() + 0.05 * torch.randn(ref.shape, generator=g)).to(torch.bfloat16)  # design-sized
        rel, tile, finite, slope = mx6.guard_metrics(noisy, ref, chunk_rows=1000)
        self.assertTrue(finite and rel < mx6.GUARD_REL_L2_MAX and tile < mx6.GUARD_TILE_REL_L2_MAX)
        self.assertLess(abs(slope - 1.0), mx6.GUARD_SLOPE_MAX)
        rel, _, _, _ = mx6.guard_metrics((ref.float() * 2).to(torch.bfloat16), ref)  # one binade
        self.assertGreater(rel, mx6.GUARD_REL_L2_MAX)
        # REVIEW m4.2: a non-power-of-two scale lost to an E8M0 operand (1.2 seen as 1.0): rel L2 ~0.17
        # passes the rel bound; the slope catches it.
        scaled = (ref.float() / 1.2).to(torch.bfloat16)
        rel, tile, finite, slope = mx6.guard_metrics(scaled, ref)
        self.assertLess(rel, mx6.GUARD_REL_L2_MAX)
        self.assertGreater(abs(slope - 1.0), mx6.GUARD_SLOPE_MAX)
        stats = mx6.guard_stats(scaled, lambda a, b: ref[a:b], [(0, ref.shape[0])])
        self.assertFalse(mx6.guard_accepts(stats))
        bad_tile = ref.clone()
        bad_tile[128 * 17:128 * 18] = torch.randn((128, 64), generator=g).to(torch.bfloat16)
        rel, tile, _, _ = mx6.guard_metrics(bad_tile, ref, chunk_rows=1024)
        self.assertLess(rel, mx6.GUARD_REL_L2_MAX)  # invisible globally ...
        self.assertGreater(tile, mx6.GUARD_TILE_REL_L2_MAX)  # ... caught per tile
        nan = ref.clone()
        nan[5, 3] = float("nan")
        self.assertFalse(mx6.guard_metrics(nan, ref)[2])
        # non-finite values OUTSIDE the windows are still counted (full output, bit patterns)
        outside = ref.clone()
        outside[4000, 1] = float("inf")
        stats = mx6.guard_stats(outside, lambda a, b: ref[a:b], [(0, 128)])
        self.assertEqual(stats["nonfinite"], 1)
        e4 = torch.float8_e4m3fn
        r8 = ref.float().clamp(-448, 448).to(e4)
        self.assertEqual(mx6.guard_metrics(r8, r8)[:2], (0.0, 0.0))
        n8 = r8.clone()
        n8.view(torch.uint8)[7, 7] = 0x7F  # the e4m3fn NaN code
        self.assertFalse(mx6.guard_metrics(n8, r8)[2])

    def test_first_call_accepts_rejects_and_survives_a_failing_reference(self) -> None:
        mx6 = self.mx6
        cpu = torch.device("cpu")
        plan = self.plan()
        ref = torch.randn((256, 32)).to(torch.bfloat16)
        good = ref.clone()
        bad = (ref.float() * 3).to(torch.bfloat16)
        full_calls: List[int] = []

        def full() -> Any:  # pyre-ignore[3]
            full_calls.append(1)
            return ref

        with _capture() as out:
            self.assertIs(mx6._first_call("VQK", plan, lambda: good, lambda a, b: ref[a:b], full, 1), good)
            self.assertIs(mx6._first_call("VK", plan, lambda: bad, lambda a, b: ref[a:b], full, 1), ref)

            def oom(a: int, b: int) -> Any:  # pyre-ignore[3]
                raise RuntimeError("HIP out of memory (synthetic)")

            self.assertIs(mx6._first_call("U", plan, lambda: good, oom, full, 2), ref)
        text = out.getvalue()
        self.assertIn("guard VQK on cuda:-1", text)
        self.assertIn("-> armed", text)
        self.assertIn("-> REJECTED", text)
        self.assertIn("ERROR guard U on cuda:-1: raised RuntimeError", text)
        self.assertEqual(len(full_calls), 2)  # the full FP8 call only on a rejection / failure
        self.assertIn((-1, "VQK", (11, 22)), mx6._site_armed)
        self.assertIsNone(mx6.site_reason(cpu, "VQK"))
        self.assertIn("guard rejected", mx6.site_reason(cpu, "VK"))
        self.assertIn("guard raised", mx6.site_reason(cpu, "U"))
        # per weight: another weight's first VQK call is guarded again
        self.assertTrue(mx6._guard_pending(cpu, "VQK", (33, 44)))
        self.assertFalse(mx6._guard_pending(cpu, "VQK", (11, 22)))

    # ---- REVIEW m5 / m6: the device ARMED line and the counters ---------------------------------
    def test_device_line_waits_for_every_site(self) -> None:
        mx6 = self.mx6
        cpu = torch.device("cpu")
        mx6._site_armed.add((-1, "VQK", (1, 2)))
        with _capture() as out:
            mx6._log_armed(cpu)
        self.assertNotIn("ARMED on", out.getvalue())  # VK's first guard has not run
        mx6._site_armed.add((-1, "VK", (1, 2)))
        with _capture() as out:
            mx6._log_armed(cpu)
            mx6._log_armed(cpu)
        text = out.getvalue()
        self.assertEqual(text.count("ARMED on cuda:-1"), 1)
        self.assertIn("U:N=512:FP8 (N=512 not in", text)
        self.assertIn("VQK:N=1536:mx6 VK:N=1024:mx6", text)
        self.assertIn("impl='fake' src=ln_store_mx6.cu@ffffffffffff", text)
        self.assertNotIn("ERROR", text)

    def test_device_line_reports_a_latched_required_site(self) -> None:
        mx6 = self.mx6
        cpu = torch.device("cpu")
        mx6._site_armed.add((-1, "VQK", (1, 2)))
        mx6._latch(cpu, "VK", "guard rejected: rel_l2=0.99")
        with _capture() as out:
            mx6._log_armed(cpu)
        self.assertIn("ERROR NOT ARMED on cuda:-1", out.getvalue())
        self.assertIn("VK:N=1024:FP8 (latched: guard rejected", out.getvalue())

    def test_counters_line(self) -> None:
        mx6 = self.mx6
        mx6._count("served_VQK")
        mx6._count("planned_uvqk")
        with _capture() as out:
            line = mx6.report_counters("test")
        self.assertIn("[mx6-fwd] counters at test", out.getvalue())
        for needle in ("planned_uvqk=1", "served_VQK=1", "served_VK=0", "served_U=0", "declined=0",
                       "recompute_VQK=0", "shadows built=0"):
            self.assertIn(needle, line)


_REF_PRODUCER = "_mx6_fwd_test_reference_producer"


def _cpu_store_t(x: Any, weight: Any, bias: Any, eps: float, inv_scale: float,  # pyre-ignore[2]
                 pad_to: int = 128, tiled_xt: bool = True) -> Tuple[Any, ...]:
    """CPU stand-in of the shipped store_t: (y8, mean, rstd, step-tiled yT) from torch statistics."""
    from generative_recommenders.ops.hip import mx6_ref

    xf = x.float()
    mean = xf.mean(1)
    rstd = torch.rsqrt(xf.var(1, unbiased=False) + eps)
    y = mx6_ref.ln_y_exact(x, mean, rstd, weight, bias, inv_scale)
    return mx6_ref.e4m3_sat(y), mean, rstd, mx6_ref.xt8_reference(y)


def _install_reference_producer(defects: Dict[str, bool]) -> types.ModuleType:
    """CPU producer with the real signature whose bytes are the stand-in store_t's and mx6_ref's, with
    switchable defects (REVIEW M2): a 1-ulp mean, an x6 byte wrong only in the no-x8 instantiation, an xs byte
    wrong in both."""
    from generative_recommenders.ops.hip import mx6_ref

    prod = types.ModuleType(_REF_PRODUCER)

    def ln_store_mx6(x: Any, weight: Any, bias: Any, eps: float, inv_scale: float,  # pyre-ignore[2]
                     mp: Optional[int] = None, *, want_x8: bool = False, return_y: bool = False,
                     xt_out: Any = None, x6_out: Any = None, xs_out: Any = None, mean_out: Any = None,
                     rstd_out: Any = None, x8_out: Any = None, y_out: Any = None) -> Tuple[Any, ...]:
        y8, mean, rstd, yt = _cpu_store_t(x, weight, bias, eps, inv_scale)
        y = mx6_ref.ln_y_exact(x, mean, rstd, weight, bias, inv_scale)
        x6r, xsr = mx6_ref.x6_xs_reference(y)
        want = want_x8 or x8_out is not None
        if defects.get("x6_no_x8") and not want:
            x6r.view(-1)[4321] ^= 1
        if defects.get("xs"):
            xsr.view(-1)[77] ^= 1
        if defects.get("mean"):
            mean[7] = torch.nextafter(mean[7], mean[7] + 1)

        def put(dst: Any, src: Any) -> Any:  # pyre-ignore[2,3]
            if dst is None:
                return src
            dst.copy_(src)
            return dst

        return (put(x6_out, x6r), put(xs_out, xsr), put(xt_out, yt), put(mean_out, mean), put(rstd_out, rstd),
                put(x8_out, y8) if want else None)

    prod.ln_store_mx6 = ln_store_mx6  # pyre-ignore[16]
    prod.ineligible_reason = lambda x, weight, bias: None  # pyre-ignore[16]
    prod.failure = lambda: None  # pyre-ignore[16]
    prod.kernel = lambda: _FakeKernel  # pyre-ignore[16]
    sys.modules[_REF_PRODUCER] = prod
    return prod


class ProducerCheckCpuTest(_AdapterCase):
    """The first-call producer check (REVIEW M2) end to end on CPU: the store_t stand-in, the row-slice
    reference and the other kernel instantiation into poisoned buffers."""

    gemm_module = _FAKE_GEMM
    producer_module = _REF_PRODUCER

    def setUp(self) -> None:
        _install_fakes()
        self.addCleanup(sys.modules.pop, _FAKE_GEMM, None)
        self.addCleanup(sys.modules.pop, _FAKE_PRODUCER, None)
        self.addCleanup(sys.modules.pop, _REF_PRODUCER, None)
        super().setUp()

    def run_check(self, **defects: bool) -> Tuple[Any, str]:  # pyre-ignore[3]
        from generative_recommenders.ops.triton import triton_layer_norm, wgrad_ck

        _install_reference_producer(defects)
        g = torch.Generator().manual_seed(11)
        x = (torch.randn((3000, 512), generator=g) * 2 + 0.2).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(512, generator=g)).to(torch.bfloat16)
        b = (0.05 * torch.randn(512, generator=g)).to(torch.bfloat16)
        with self.operands_ok(), _capture() as out, \
                mock.patch.object(triton_layer_norm, "triton_weighted_layer_norm_fwd_fp8_store_t", _cpu_store_t), \
                mock.patch.object(wgrad_ck, "note_tiled_producer"):
            plan = self.uvqk(x=x, norm_weight=w, norm_bias=b, fp8_scale_value=1.0 / 37.5)
        return plan, out.getvalue()

    def test_clean_producer_passes(self) -> None:
        plan, text = self.run_check()
        self.assertIsNotNone(plan, text)
        self.assertIn("producer check on cuda:-1: M=3000 x8=yes+no differing: vs store_t mean 0, rstd 0, "
                      "xT8 0 B, x8 0 B; x6 / xs vs reference on 3000 rows 0 / 0 B", text)
        self.assertIn("-> ok", text)
        self.assertIn(-1, self.mx6._producer_checked)

    def test_each_defect_latches_the_producer(self) -> None:
        for defect, needle in (("mean", "mean 1,"), ("x6_no_x8", "other instantiation x6 1,"),
                               ("xs", "3000 rows 0 / 1 B")):
            self.mx6.reset_for_testing()
            plan, text = self.run_check(**{defect: True})
            self.assertIsNone(plan, defect)
            self.assertIn("ERROR producer check on cuda:-1", text)
            self.assertIn(needle, text, defect)
            self.assertIn("uvqk:producer", text)
            self.assertIn("producer check failed", self.mx6.site_reason(torch.device("cpu"), "VQK"))


# =============================================================================================== reference
def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _f32_from_bits(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def _round_to_f32(value: Fraction) -> float:
    """RNE of an exact rational to fp32 (finite, normal range or zero)."""
    if value == 0:
        return 0.0
    near = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    u = _f32_bits(near)
    cands = [_f32_from_bits(u + d) for d in (-1, 0, 1)]
    return min(cands, key=lambda c: (abs(Fraction(c) - value), _f32_bits(c) & 1))


def _py_code_of(g: float) -> int:
    """E2M3 code of a non-negative grid value: exponent bits [3, 5), mantissa bits [0, 3)."""
    if g < 1.0:
        return int(round(g * 8))
    e = int(math.floor(math.log2(g))) + 1
    return (e << 3) | int(round((g / 2.0 ** (e - 1) - 1.0) * 8))


_E2M3_CODES = sorted((g, _py_code_of(g)) for g in {(m / 8.0) for m in range(8)} | {
    (2.0 ** (e - 1)) * (1 + m / 8.0) for e in (1, 2, 3) for m in range(8)})


def _py_code(v: float) -> int:
    """E2M3 RNE (ties to the even code), |v| <= 7.5; a negative value rounding to 0 is +0."""
    a = abs(v)
    g, code = min(_E2M3_CODES, key=lambda gc: (abs(gc[0] - a), gc[1] & 1))
    return code | (32 if (v < 0 and g != 0.0) else 0)


def _py_e8m0(amax: float) -> int:
    if amax == 0.0:
        return 0
    e = math.ceil(math.log2(amax / 7.5))
    while amax > 7.5 * 2.0 ** e:
        e += 1
    while amax <= 7.5 * 2.0 ** (e - 1):
        e -= 1
    return max(-127, min(127, e))


def _py_x6_xs(rows: List[List[float]]) -> Tuple[List[int], List[int]]:
    """MX6-T / MX6-A-v1 bytes of y rows, written from the layout text (r14 prod RESULT.md section 1),
    not from the proto code: field f of a 32-value block at bits [6f, 6f+6) of 24 LE bytes; per
    (tile T, K step S) HEAD [0, 8192): 16-B head of block q of tile row R at 64R + 16*(q ^ ((R>>2)&3));
    TAIL [8192, 12288): 8-B tail at 8192 + 32R + 8*(q ^ (2*((R>>3)&1))); xs byte [T, S, (q*16 + r)*8 + m]
    = E8M0 of row 128T + 16m + r, block 4S + q; pad rows: zero codes, byte 127."""
    m = len(rows)
    t_count = (m + 127) // 128
    x6 = [0] * (t_count * 4 * 12288)
    xs = [127] * (t_count * 4 * 512)
    for row in range(m):
        t, rr = divmod(row, 128)
        for blk in range(16):
            vals = rows[row][32 * blk:32 * blk + 32]
            e = _py_e8m0(max(abs(v) for v in vals))
            packed = 0
            for f, v in enumerate(vals):
                packed |= _py_code(v / 2.0 ** e) << (6 * f)
            by = [(packed >> (8 * k)) & 255 for k in range(24)]
            s, q = divmod(blk, 4)
            base = (t * 4 + s) * 12288
            for b in range(16):
                x6[base + 64 * rr + 16 * (q ^ ((rr >> 2) & 3)) + b] = by[b]
            for b in range(8):
                x6[base + 8192 + 32 * rr + 8 * (q ^ (2 * ((rr >> 3) & 1))) + b] = by[16 + b]
            mm, r = divmod(rr, 16)
            xs[(t * 4 + s) * 512 + (q * 16 + r) * 8 + mm] = e + 127
    return x6, xs


@unittest.skipIf(torch is None, "torch not installed")
class Mx6ReferenceTest(unittest.TestCase):
    """ops/hip/mx6_ref.py (the in-model producer check's reference)."""

    def test_fma_is_rounded_once(self) -> None:
        from generative_recommenders.ops.hip import mx6_ref

        f32 = torch.float32
        # a*b = 1 - 2^-40 exactly; + (2^24 + 2) is just below the fp32 midpoint 2^24 + 3: RNE gives
        # 2^24 + 2, the fp64 sum rounds to the midpoint and a second RNE gives 2^24 + 4.
        a = torch.tensor([1 + 2.0**-20], dtype=f32)
        b = torch.tensor([1 - 2.0**-20], dtype=f32)
        c = torch.tensor([2.0**24 + 2], dtype=f32)
        self.assertEqual(float((a.double() * b.double() + c.double()).float()), 2.0**24 + 4)
        self.assertEqual(float(mx6_ref.fma_f32_exact(a, b, c)), 2.0**24 + 2)
        self.assertEqual(float(mx6_ref.fma_f32_exact(-a, b, -c)), -(2.0**24 + 2))
        rng = random.Random(49)
        n = 3000
        av = [rng.gauss(0, 1) * 2.0 ** rng.randint(-8, 8) for _ in range(n)]
        bv = [rng.gauss(1, 0.3) for _ in range(n)]
        cv = [rng.gauss(0, 1) * 2.0 ** rng.randint(-30, 30) for _ in range(n)]
        at = torch.tensor(av, dtype=f32)
        bt = torch.tensor(bv, dtype=torch.bfloat16).float()  # bf16 LN weights
        ct = torch.tensor(cv, dtype=f32)
        got = mx6_ref.fma_f32_exact(at, bt, ct).tolist()
        bad = 0
        for i in range(n):
            exact = Fraction(float(at[i])) * Fraction(float(bt[i])) + Fraction(float(ct[i]))
            if exact != 0 and _round_to_f32(exact) != got[i]:
                bad += 1
        self.assertEqual(bad, 0)

    def test_ln_y_matches_the_formula(self) -> None:
        from generative_recommenders.ops.hip import mx6_ref

        g = torch.Generator().manual_seed(4)
        x = (torch.randn((40, 512), generator=g) * 3).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(512, generator=g)).to(torch.bfloat16)
        b = (0.1 * torch.randn(512, generator=g)).to(torch.bfloat16)
        mean = x.float().mean(1)
        rstd = torch.rsqrt(x.float().var(1, unbiased=False) + 1e-6)
        y = mx6_ref.ln_y_exact(x, mean, rstd, w, b, 45.5, chunk=16)
        inv = struct.unpack("<f", struct.pack("<f", 45.5))[0]
        for r in (0, 17, 39):
            for c in (0, 100, 511):
                t = _round_to_f32(Fraction(_round_to_f32(Fraction(float(x[r, c])) - Fraction(float(mean[r]))))
                                  * Fraction(float(rstd[r])))
                f = _round_to_f32(Fraction(t) * Fraction(float(w[c])) + Fraction(float(b[c])))
                self.assertEqual(float(y[r, c]), _round_to_f32(Fraction(f) * Fraction(inv)), (r, c))

    def test_x6_xs_match_an_independent_encoder(self) -> None:
        from generative_recommenders.ops.hip import mx6_ref

        g = torch.Generator().manual_seed(1949)
        m = 130  # one full tile and a 2-row tail tile (126 pad rows)
        y = torch.randn((m, 512), generator=g)
        y *= torch.exp2(torch.randint(-12, 12, (m, 16, 1), generator=g).float()).repeat_interleave(32, 1).view(m, 512)
        y[3, 0:32] = 0.0  # all-zero block: byte 127
        y[4, 32:64] = 7.5 * 2.0**3  # amax exactly 7.5 * 2^e
        y[5, 64:96] = torch.linspace(-7.5, 7.5, 32) * 2.0**-4
        y[5, 70] = -0.0625 * 2.0**-4 * 0.4  # a small negative value: code +0, not -0
        y[6, 96:128] = (torch.arange(32).float() + 0.5) * 0.25  # RNE ties on the 0.25 / 0.5 grid
        y[7, 128:160] = torch.tensor([0.9375, 1.9375, 3.875, -0.9375, -1.9375, -3.875] + [7.5] * 26)  # binade ties
        y[129, :] = -1.0e-3
        x6, xs = mx6_ref.x6_xs_reference(y)
        ex6, exs = _py_x6_xs(y.tolist())
        self.assertEqual(tuple(x6.shape), (2, 4, 12288))
        self.assertEqual(tuple(xs.shape), (2, 4, 512))
        self.assertEqual(x6.view(-1).tolist(), ex6)
        self.assertEqual(xs.view(-1).tolist(), exs)

    def test_slice_check_windows_and_detection(self) -> None:
        """The first-call producer check's x6 / xs slice check (REVIEW M2) on CPU: window placement, tile
        indexing across several windows, a defect inside a window caught, one outside not looked at."""
        from generative_recommenders.ops.hip import mx6_fwd as mx6
        from generative_recommenders.ops.hip import mx6_ref

        g = torch.Generator().manual_seed(7)
        m = 3000
        x = (torch.randn((m, 512), generator=g) * 2).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(512, generator=g)).to(torch.bfloat16)
        b = (0.05 * torch.randn(512, generator=g)).to(torch.bfloat16)
        mean = x.float().mean(1)
        rstd = torch.rsqrt(x.float().var(1, unbiased=False) + 1e-6)
        y = mx6_ref.ln_y_exact(x, mean, rstd, w, b, 37.5)
        x6, xs = mx6_ref.x6_xs_reference(y)
        plan = mx6.Plan(kind="uvqk", device=torch.device("cpu"), m=m, x6=x6.clone(), xs=xs.clone(), xt8=None,
                        mean=mean, rstd=rstd, x8=None)
        y8 = mx6_ref.e4m3_sat(y)
        # small stand-ins for 65,536 rows and 2^31 bytes, so the windows are several and distinct
        with mock.patch.object(mx6, "CHECK_HEAD_ROWS", 256), mock.patch.object(mx6, "_TWO31", 1_000_000):
            windows = mx6.check_windows(m)
            self.assertEqual(windows, [(0, 256), (768, 1152), (1792, 2176), (2432, 3000)])
            bad = {k: int(v) for k, v in mx6._slice_check(plan, x, w, b, 37.5, y8).items()}
            self.assertEqual(bad, {"x6_ref": 0, "xs_ref": 0, "ref_vs_store_t": 0, "rows": 256 + 384 + 384 + 568})
            plan.x6[7, 2, 100] ^= 1  # tile 7 = rows 896..1023: inside the window around row 976
            plan.xs[11, 1, 5] ^= 1  # tile 11 = rows 1408..1535: outside every window
            bad = {k: int(v) for k, v in mx6._slice_check(plan, x, w, b, 37.5, y8).items()}
            self.assertEqual((bad["x6_ref"], bad["xs_ref"]), (1, 0))
            y8.view(torch.uint8)[2999, 3] ^= 1  # a "store_t" byte the emulation disagrees with
            bad = {k: int(v) for k, v in mx6._slice_check(plan, x, w, b, 37.5, y8).items()}
            self.assertEqual(bad["ref_vs_store_t"], 1)

    def test_xt8_reference_layout(self) -> None:
        from generative_recommenders.ops.hip import mx6_ref

        y = torch.arange(130 * 512, dtype=torch.float32).view(130, 512) % 97 - 48
        xt = mx6_ref.xt8_reference(y)
        self.assertEqual(tuple(xt.shape), (2, 512, 128))
        e8 = mx6_ref.e4m3_sat(y).view(torch.uint8)
        self.assertEqual(int(xt.view(torch.uint8)[1, 7, 1]), int(e8[129, 7]))
        self.assertEqual(int(xt.view(torch.uint8)[0, 300, 5]), int(e8[5, 300]))
        self.assertEqual(int(torch.count_nonzero(xt.view(torch.uint8)[1, :, 2:])), 0)


# =============================================================================================== structure
def _is_gate(test: ast.expr, gate: str) -> bool:
    if isinstance(test, ast.Name):
        return test.id == gate
    return (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
            and isinstance(test.values[0], ast.Name) and test.values[0].id == gate)


def _mentions(node: ast.AST, names: Tuple[str, ...]) -> List[str]:
    return sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in names})


class _Fold(ast.NodeTransformer):
    def __init__(self, gate: str) -> None:
        self.gate = gate

    def visit_If(self, node: ast.If) -> Any:  # pyre-ignore[3]
        if _is_gate(node.test, self.gate):  # knob off: the `else` statements (none: removed)
            out: List[Any] = []
            for stmt in node.orelse:
                visited = self.visit(stmt)
                out.extend(visited if isinstance(visited, list) else [visited])
            return out
        return self.generic_visit(node)


class KnobOffStructureTest(unittest.TestCase):
    _MX6_NAMES = ("_mx6_fwd", "mx6", "vqk_mx6", "vk_mx6", "_MX6_FWD")

    def _tree(self, path: pathlib.Path) -> ast.Module:
        return ast.parse(path.read_text(), filename=str(path))

    def test_call_site_arms_are_gated_with_a_shipped_else(self) -> None:
        tree = self._tree(_CALL_SITE)
        gates = {}
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.If) and _mentions(node.test, ("_MX6_FWD",)):
                    self.assertTrue(_is_gate(node.test, "_MX6_FWD"), f"line {node.lineno}: not `_MX6_FWD and ...`")
                    self.assertTrue(node.orelse, f"line {node.lineno}: MX6 arm without the shipped else")
                    # r16 B4 (item (88)) splits the ordinary layer's backward into backward ->
                    # _backward_impl (try/finally around the dO8 registry release); the
                    # MX6 recompute arm then sits in _backward_impl, still the backward body.
                    gates[node.lineno] = {"_backward_impl": "backward"}.get(fn.name, fn.name)
        # ordinary layer: forward LN / U / V/Q/K / record + backward recompute; targets-only layer:
        # forward LN / V/K / record + backward recompute
        self.assertEqual(sorted(gates.values()), ["backward"] * 2 + ["forward"] * 7)
        for node in ast.walk(tree):  # nothing else in the module reads the adapter
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "_mx6_fwd":
                self.assertIn(node.attr, ("uvqk_ln_store", "vk_ln_store", "gemm_u", "gemm_vqk", "gemm_vk",
                                          "record_contract", "recompute_vqk", "recompute_vk"))

    def test_folded_call_site_has_no_mx6_names(self) -> None:
        tree = _Fold("_MX6_FWD").visit(self._tree(_CALL_SITE))
        tree.body = [s for s in tree.body if not (
            isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id == "_MX6_FWD")]
        self.assertEqual(_mentions(tree, self._MX6_NAMES), [])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn("mx6_fwd", [a.name for a in node.names])

    def test_knob_is_read_once_below_the_triton_kernels(self) -> None:
        tree = self._tree(_CALL_SITE)
        defs = [s for s in tree.body if isinstance(s, ast.AnnAssign)
                and isinstance(s.target, ast.Name) and s.target.id == "_MX6_FWD"]
        self.assertEqual(len(defs), 1)
        want = ast.parse(f'os.environ.get("{_KNOB}", "0") == "1"', mode="eval").body
        self.assertEqual(ast.dump(defs[0].value), ast.dump(want))
        kernels = [n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and any(
            "triton" in ast.unparse(d) for d in n.decorator_list)]
        self.assertTrue(kernels)
        self.assertGreater(defs[0].lineno, max(kernels))
        imports = [s for s in tree.body if isinstance(s, ast.If) and _is_gate(s.test, "_MX6_FWD")]
        self.assertEqual(len(imports), 1)
        self.assertEqual(ast.unparse(imports[0].body[0]),
                         "from generative_recommenders.ops.hip import mx6_fwd as _mx6_fwd")
        top_imports = [a.name for s in tree.body if isinstance(s, ast.ImportFrom) for a in s.names]
        self.assertNotIn("mx6_fwd", top_imports)

    def test_prebuild_hook_is_gated(self) -> None:
        tree = self._tree(_TRITON_FP8)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prebuild_extensions")
        hooks = [n for n in ast.walk(fn) if isinstance(n, ast.If) and _mentions(n.test, ("_HSTU_FP8_FWD_MXFP6",))]
        self.assertEqual(len(hooks), 1)
        self.assertTrue(_is_gate(hooks[0].test, "_HSTU_FP8_FWD_MXFP6"))
        folded = _Fold("_HSTU_FP8_FWD_MXFP6").visit(fn)
        self.assertNotIn("mx6_fwd", ast.unparse(folded))

    def test_reference_module_is_reached_only_through_the_adapter(self) -> None:
        for path in (_CALL_SITE, _TRITON_FP8):
            self.assertNotIn("mx6_ref", path.read_text())

    def test_call_site_calls_bind_the_adapter_signatures(self) -> None:
        """Every `_mx6_fwd.<fn>(...)` call of the call-site module binds that adapter function's signature
        (no torch: both from source), so a renamed or dropped adapter parameter fails here, not in a run."""
        calls = [n for n in ast.walk(self._tree(_CALL_SITE)) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "_mx6_fwd"]
        self.assertEqual(len(calls), 9)
        for call in calls:
            sig = _signature_from_ast(_ADAPTER, call.func.attr)
            sig.bind(*range(len(call.args)), **{kw.arg: None for kw in call.keywords})


if __name__ == "__main__":
    unittest.main()
