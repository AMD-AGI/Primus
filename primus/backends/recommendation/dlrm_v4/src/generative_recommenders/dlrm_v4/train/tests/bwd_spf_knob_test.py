#!/usr/bin/env python3
# pyre-strict
"""BWD SPF lab knob (R15 iter75 tscv2): DLRM_HSTU_FP8_LAB_BWD_SPF parsing, its
dependency on DLRM_HSTU_FP8_LAB_FWD_SPF, the backward arm check
(ops/fp8_qkv_spf_bwd.arm_check_bwd = fp8_qkv_spf.arm_check + the range
[-40, -7]), agreement with bwd_dkdv's own guard (tsc_exponent, read from the
kernel source), the stray-token strip, the per-call checks and the ARMED line
of the launcher (hstu_flydsl_mha._bwd_spf_tokens, driven through
flydsl_hstu_attention_bwd with a stub kernel that stops at compile), and the
fail-closed refusal in try_flydsl_attention_bwd.

CPU-only: nothing compiles or launches a kernel (the kernel side is the CPU
AOT of R15 iter75 tscv2). The host tests need only the standard library and
run under Python 3.9; the launcher tests skip where torch / hstu_flydsl_mha
are not importable. All values are synthetic (grid points of the SPF grid)."""

import ast
import contextlib
import io
import logging
import math
import os
import random
import re
import struct
import unittest
from unittest import mock

_BK = "DLRM_HSTU_FP8_LAB_BWD_SPF"
_FK = "DLRM_HSTU_FP8_LAB_FWD_SPF"
_ALPHA = 1.0 / (128 ** 0.5)  # stu.py: attention_dim 128, attn_alpha unset


def _mods():  # pyre-ignore[3]
    from generative_recommenders.ops import fp8_qkv_spf, fp8_qkv_spf_bwd

    return fp8_qkv_spf, fp8_qkv_spf_bwd


def _gr_dir() -> str:
    _, spfb = _mods()
    return os.path.dirname(os.path.dirname(os.path.abspath(spfb.__file__)))


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _from_bits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b))[0]


def _kernel_guard():  # pyre-ignore[3]
    """LOG2E, TSC_K_MIN, TSC_K_MAX and tsc_exponent, executed from the
    hstu_bwd_dkdv.py source text (the module itself needs flydsl)."""
    path = os.path.join(_gr_dir(), "ops", "flydsl", "hstu_bwd_dkdv.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    keep, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = set()
            for t in node.targets:
                names |= {e.id for e in (t.elts if isinstance(t, ast.Tuple)
                                          else [t]) if isinstance(e, ast.Name)}
            hit = names & {"LOG2E", "TSC_K_MIN", "TSC_K_MAX"}
            if hit:
                keep.append(node)
                found |= hit
        elif isinstance(node, ast.FunctionDef) and node.name == "tsc_exponent":
            keep.append(node)
            found.add("tsc_exponent")
    assert found == {"LOG2E", "TSC_K_MIN", "TSC_K_MAX", "tsc_exponent"}, found
    ns = {"math": math, "struct": struct}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


@contextlib.contextmanager
def _env(**kv):  # pyre-ignore[2,3]
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class BwdSpfHostTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spf, self.spfb = _mods()

    def test_knob_parse_every_value(self) -> None:
        spfb = self.spfb
        for v in (None, "", "0"):
            for fv in (None, "", "0", "snap", "1"):
                env = {} if v is None else {_BK: v}
                if fv is not None:
                    env[_FK] = fv
                self.assertEqual(spfb.bwd_spf_mode(env), "off", (v, fv))
        for fv in ("snap", "1"):
            self.assertEqual(spfb.bwd_spf_mode({_BK: "1", _FK: fv}), "fold")
        for bad in ("2", "true", "on", " 1", "1 ", "snap", "fold", "yes",
                    "-1", "tsc", "01", "1.0", "00", " "):
            for fv in (None, "snap", "1"):
                env = {_BK: bad} if fv is None else {_BK: bad, _FK: fv}
                with self.assertRaises(ValueError, msg=(bad, fv)):
                    spfb.bwd_spf_mode(env)
        with _env(**{_BK: "1", _FK: "snap"}):
            self.assertEqual(spfb.bwd_spf_mode(), "fold")
        with _env(**{_BK: None, _FK: None}):
            self.assertEqual(spfb.bwd_spf_mode(), "off")

    def test_fwd_spf_dependency(self) -> None:
        spfb = self.spfb
        for fv in (None, "", "0"):
            env = {_BK: "1"} if fv is None else {_BK: "1", _FK: fv}
            with self.assertRaisesRegex(ValueError, _FK):
                spfb.bwd_spf_mode(env)
        with self.assertRaises(ValueError):         # invalid forward value
            spfb.bwd_spf_mode({_BK: "1", _FK: "on"})

    def test_arm_check_pass_on_grid(self) -> None:
        spf, spfb = self.spf, self.spfb
        s15 = spf.grid_scale(-15, _ALPHA)
        self.assertEqual(_bits(s15), 0x3C7D75E0)
        self.assertEqual(spfb.arm_check_bwd(s15, _ALPHA), -15)
        self.assertEqual(spf.arm_check(s15, _ALPHA), -15)
        for j in range(spfb.BWD_J_MIN, spfb.BWD_J_MAX + 1):
            self.assertEqual(spfb.arm_check_bwd(spf.grid_scale(j, _ALPHA),
                                                _ALPHA), j)

    def test_arm_check_fail_off_grid(self) -> None:
        spf, spfb = self.spf, self.spfb
        s15 = spf.grid_scale(-15, _ALPHA)
        for bad in (0.0146325, spf.f32(0.0146325), 0.0173, 2.0 ** -6,
                    _from_bits(_bits(s15) + 1), _from_bits(_bits(s15) - 1),
                    0.0, -s15, float("nan"), float("inf")):
            with self.assertRaisesRegex(RuntimeError, "SPF grid",
                                        msg=bad) as cm:
                spfb.arm_check_bwd(bad, _ALPHA)
            # names this knob and the backward range; x4c19's error chained
            msg = str(cm.exception)
            self.assertTrue(msg.startswith(_BK + "=1: "), msg)
            self.assertIn("backward range [-40, -7]", msg)
            self.assertIsInstance(cm.exception.__cause__, RuntimeError)
            self.assertIn(_FK, str(cm.exception.__cause__))

    def test_arm_check_fail_out_of_range(self) -> None:
        spf, spfb = self.spf, self.spfb
        for j in (-6, -5, -2, 0, 7, 103, -41, -60, -108):
            s = spf.grid_scale(j, _ALPHA)
            self.assertEqual(spf.arm_check(s, _ALPHA), j)  # forward accepts
            with self.assertRaisesRegex(RuntimeError, "backward fold's range"):
                spfb.arm_check_bwd(s, _ALPHA)
        for j in (spf.J_MIN - 1, spf.J_MAX + 1):           # forward refuses
            with self.assertRaises(RuntimeError) as cm:
                spfb.arm_check_bwd(spf.grid_scale(j, _ALPHA), _ALPHA)
            self.assertIn(_BK + "=1: ", str(cm.exception))
            self.assertIn("backward range [-40, -7]", str(cm.exception))

    def test_range_choice(self) -> None:
        spf, spfb = self.spf, self.spfb
        self.assertEqual((spfb.BWD_J_MIN, spfb.BWD_J_MAX), (-40, -7))
        # -7 is the largest j under the vf3p guard qkv_scale <= 0.25
        self.assertLessEqual(spf.grid_scale(-7, _ALPHA), 0.25)
        self.assertGreater(spf.grid_scale(-6, _ALPHA), 0.25)
        self.assertEqual(spfb.VF3P_QKV_SCALE_MAX, 0.25)
        self.assertTrue(spf.J_MIN <= spfb.BWD_J_MIN and spfb.BWD_J_MAX <= spf.J_MAX)
        # same bound as the launcher's vf3p guard (source text)
        with open(os.path.join(_gr_dir(), "ops", "flydsl",
                               "hstu_flydsl_mha.py")) as f:
            self.assertIn("_VF3P_QKV_SCALE_MAX = 0.25\n", f.read())

    def test_kernel_guard_agrees(self) -> None:
        spf, spfb = self.spf, self.spfb
        g = _kernel_guard()
        self.assertEqual((g["TSC_K_MIN"], g["TSC_K_MAX"]),
                         (spfb.BWD_J_MIN, spfb.BWD_J_MAX))
        self.assertEqual(g["LOG2E"], spf.LOG2E)

        def arm(x):  # pyre-ignore[2,3]
            try:
                return spfb.arm_check_bwd(x, _ALPHA)
            except RuntimeError:
                return None

        n_acc = n = 0
        for j in range(-130, 130):
            s = spf.grid_scale(j, _ALPHA)
            if s is None:
                continue
            for d in (-3, -2, -1, 0, 1, 2, 3):
                x = _from_bits(_bits(s) + d)
                kj = g["tsc_exponent"](_ALPHA, x)
                self.assertEqual(kj, arm(x), (j, d, x))
                n += 1
                n_acc += kj is not None
        self.assertEqual(n_acc, spfb.BWD_J_MAX - spfb.BWD_J_MIN + 1)
        rng = random.Random(75)
        for _ in range(20000):
            x = spf.f32(2.0 ** rng.uniform(-30.0, 2.0))
            self.assertEqual(g["tsc_exponent"](_ALPHA, x), arm(x), x)
        # Input domain: the kernel rounds qkv_scale to f32 (as FlyDSL marshals
        # it), the arm check needs an exact f32 (the launcher passes the f32
        # buffer's host value). A double next to s_-15 is refused only here.
        s15 = spf.grid_scale(-15, _ALPHA)
        near = s15 * (1.0 + 2.0 ** -40)
        self.assertEqual(g["tsc_exponent"](_ALPHA, near), -15)
        self.assertIsNone(arm(near))

    def test_body_reason(self) -> None:
        spfb = self.spfb
        ship = "ilp,msat,nobr,rfl,undef,vf3p,vf4".split(",")
        self.assertIsNone(spfb.body_reason({"msplit": 1}, "k128", ship))
        for cfg, mfma, toks, pat in (
                ({"msplit": 1}, "k128", [t for t in ship if t != "vf4"], "vf4"),
                ({"msplit": 1}, "k128", [t for t in ship if t != "msat"], "msat"),
                ({"msplit": 2}, "k128", ship, "msplit"),
                ({}, "k128", ship, "msplit"),
                ({"msplit": 1}, "", ship, "k128"),
                ({}, "", [], "vf4")):
            why = spfb.body_reason(cfg, mfma, toks)
            self.assertIsNotNone(why)
            self.assertIn(pat, why)

    def test_warmup_scale(self) -> None:
        spfb = self.spfb
        s = 2.0 ** -6
        for env in ({}, {_BK: "0"}, {_BK: "", _FK: "1"}):
            self.assertIs(spfb.warmup_scale(s, _ALPHA, env), s)
        for fv in ("snap", "1"):
            w = spfb.warmup_scale(s, _ALPHA, {_BK: "1", _FK: fv})
            self.assertEqual(spfb.arm_check_bwd(w, _ALPHA), -14)
            self.assertGreaterEqual(w, s)
        with self.assertRaises(ValueError):
            spfb.warmup_scale(s, _ALPHA, {_BK: "1"})

    def test_lines(self) -> None:
        spf, spfb = self.spf, self.spfb
        s15 = spf.grid_scale(-15, _ALPHA)
        line = spfb.armed_line(-15, s15, "snap", "ilp,msat,tsc,vf4")
        for part in (_BK + "=1 ARMED", "j=-15 e8m0_byte=112", "0x3c7d75e0",
                     _FK + "=snap", "valu=ilp,msat,tsc,vf4"):
            self.assertIn(part, line)
        self.assertNotIn("\n", line)
        st = spfb.strip_line("DLRM_HSTU_FP8_BWD_VALU", "msat,tsc", "")
        self.assertIn("tsc", st)
        self.assertIn("stripped", st)
        self.assertIn("<unset>", st)

    def test_selftest(self) -> None:
        self.assertEqual(self.spfb._selftest(verbose=False), 0)

    def test_bound_once_as_ship17_item_106(self) -> None:
        # ship17 row (106): one apply_env_bootstrap parameter and one gin
        # binding (yambda_5b.gin, bound True); no other gin file binds it.
        gr = _gr_dir()
        with open(os.path.join(gr, "dlrm_v4", "train", "_env_bootstrap.py")) as f:
            boot = f.read()
        self.assertEqual(boot.count(f"    {_BK}: Optional[bool] = None,"), 1)
        gin_dir = os.path.join(gr, "dlrm_v4", "train", "gin")
        binds = {}
        for fn in sorted(os.listdir(gin_dir)):
            if fn.endswith(".gin"):
                with open(os.path.join(gin_dir, fn)) as f:
                    for ln in f:
                        if ln.startswith(f"apply_env_bootstrap.{_BK} "):
                            binds.setdefault(fn, []).append(ln.strip())
        self.assertEqual(
            binds, {"yambda_5b.gin": [f"apply_env_bootstrap.{_BK} = True"]})

    def test_launcher_forwards_the_prefix(self) -> None:
        sh = os.path.join(os.path.dirname(_gr_dir()), "scripts",
                          "launch_slurm.sh")
        if not os.path.exists(sh):
            self.skipTest("no scripts/launch_slurm.sh in this tree (Primus launches with primus-cli)")
        with open(sh) as f:
            txt = f.read()
        self.assertIn('for v in "${!DLRM_HSTU_FP8_@}"; do', txt)
        self.assertIn('FP8_ENV_ARGS="$FP8_ENV_ARGS -e $v=$val"', txt)
        self.assertRegex(txt, r"\n\s+\$FP8_ENV_ARGS \\\n")
        self.assertTrue(_BK.startswith("DLRM_HSTU_FP8_"))


# ----------------------------------------------------------- launcher tests
try:
    import torch

    from generative_recommenders.ops.flydsl import hstu_do8_registry as reg
    from generative_recommenders.ops.flydsl import hstu_do_fp8
    from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
except Exception as error:  # noqa: BLE001 - torch / triton absent (py3.9)
    torch = reg = hstu_do_fp8 = mha = None  # pyre-ignore[9]
    _MHA_ERR = repr(error)
else:
    _MHA_ERR = ""

_MSS = "mid,sb,wsl,sgv5,la,xt,rs2,lc,ek,nk,wh,mc,dzr,bst,ob0,obj"
_VALU_SHIPPED = "ilp,msat,nobr,rfl,undef,vf3p,vf4"
_VALU_TSC = "ilp,msat,nobr,rfl,tsc,undef,vf3p,vf4"
# The production bwd attention env (the gin's apply_env_bootstrap values of
# the bwd_dkdv path) plus every knob this test drives, unset.
_C8_ENV = {
    "HSTU_ATTN_BWD": "flydsl",
    "DLRM_HSTU_FP8_BWD_FULL": "1",
    "DLRM_HSTU_FP8_BWD_QK_MFMA": "fp8",
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT": "1",
    "DLRM_HSTU_FP8_BWD_EPILOGUE": "reuse_pk",
    "DLRM_HSTU_FP8_BWD_VALU": "undef,rfl,ilp,msat,nobr",
    "DLRM_HSTU_FP8_BWD_VF3P": "1",
    "DLRM_HSTU_FP8_BWD_VF4": "1",
    "DLRM_HSTU_FP8_BWD_MSPLIT": "1",
    "DLRM_HSTU_FP8_BWD_MSPLIT_SCHED": _MSS,
    "DLRM_HSTU_FP8_BWD_PIPE": "2",
    "DLRM_HSTU_FP8_BWD_FULL_DSC_K": "-5",
    "DLRM_HSTU_FP8_BWD_WS_AUX": "2",
    "DLRM_HSTU_FP8_BWD_WS_LAYOUT": "l128t",
    **{k: None for k in (
        "DLRM_HSTU_FP8_BWD_K128", "DLRM_HSTU_FP8_BWD_FULL_HYBRID",
        "DLRM_HSTU_FP8_BWD_FULL_DOBLK", "DLRM_HSTU_FP8_BWD_FULL_SAT",
        "DLRM_HSTU_FP8_BWD_GORDER", "DLRM_HSTU_FP8_BWD_PK",
        "DLRM_HSTU_FP8_BWD_CENSUS", "DLRM_E1_BWD_CENSUS",
        "DLRM_HSTU_FP8_BWD_FULL_STATS", "DLRM_HSTU_FP8_BWD_DUMP",
        "DLRM_HSTU_FP8_BWD_DUMP_CALLS", "HSTU_ATTN_BWD_CHECK",
        "DLRM_HSTU_FP8_ATTN_INTGLUE", "DLRM_HSTU_FP8_BWD_FULL_SSC_K",
        "DLRM_HSTU_FP8_BWD_FULL_SSC_LOG2", "DLRM_HSTU_FP8_BWD_FULL_DSC_LOG2",
        _BK, _FK)},
}
_H, _D, _N = 4, 128, 3058


class _Stop(Exception):
    """Raised by the stub compile: everything the test checks happens before."""


class _StubKernel:
    def __init__(self, shipped, guard) -> None:  # pyre-ignore[2]
        self.SHIPPED = dict(shipped)
        self.TSC_K_MIN = guard["TSC_K_MIN"]
        self.TSC_K_MAX = guard["TSC_K_MAX"]
        self.compiles = []

    def dq_acc_dtype(self):  # pyre-ignore[3]
        return torch.bfloat16

    def cvt_scale_is_exact(self, s) -> bool:  # pyre-ignore[2]
        return False

    def compile_hstu_bwd_dkdv(self, heads, dim, alpha, **kw):  # pyre-ignore
        self.compiles.append(dict(kw, heads=heads, dim=dim, alpha=alpha))
        raise _Stop()


def _shipped():  # pyre-ignore[3]
    path = os.path.join(_gr_dir(), "ops", "flydsl", "hstu_bwd_dkdv.py")
    with open(path) as f:
        body = ast.parse(f.read()).body
    for node in body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "SHIPPED" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no SHIPPED in hstu_bwd_dkdv.py")


def _stub_quantize(dout, seq_offsets, N):  # pyre-ignore[2,3]
    f = dout.float()
    return (f.clamp(-448, 448).to(torch.float8_e4m3fn),
            torch.full(dout.shape[:2], 127, dtype=torch.uint8))


class _Logs(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.msgs = []

    def emit(self, record) -> None:  # pyre-ignore[2]
        self.msgs.append(record.getMessage())


@unittest.skipIf(mha is None, f"hstu_flydsl_mha not importable: {_MHA_ERR}")
class BwdSpfLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spf, self.spfb = _mods()
        self.s15 = self.spf.grid_scale(-15, _ALPHA)
        self.s14 = self.spf.grid_scale(-14, _ALPHA)
        self.kernel = _StubKernel(_shipped(), _kernel_guard())
        self._saved = (mha._load_kernel, mha.unsupported_hardware_reason,
                       hstu_do_fp8.quantize_do_e4m3_blk,
                       set(mha._logged_layouts), dict(mha._fp8_bwd_spf_logged))
        mha._load_kernel = lambda: self.kernel
        mha.unsupported_hardware_reason = lambda: None
        hstu_do_fp8.quantize_do_e4m3_blk = _stub_quantize
        mha._logged_layouts.clear()
        mha._fp8_bwd_spf_logged.clear()
        regenv = {getattr(reg, n): None for n in
                  ("KNOB_ENV", "TAKE_ENV", "ONLY_ENV", "POISON_ENV")
                  if isinstance(getattr(reg, n, None), str)}
        self._cenv = _env(**_C8_ENV, **regenv)
        self._cenv.__enter__()
        self.logs = _Logs()
        mha.logger.addHandler(self.logs)
        self._lvl = mha.logger.level
        mha.logger.setLevel(logging.WARNING)

    def tearDown(self) -> None:
        mha.logger.removeHandler(self.logs)
        mha.logger.setLevel(self._lvl)
        self._cenv.__exit__(None, None, None)
        (mha._load_kernel, mha.unsupported_hardware_reason,
         hstu_do_fp8.quantize_do_e4m3_blk, ll, sl) = self._saved
        mha._logged_layouts.clear()
        mha._logged_layouts.update(ll)
        mha._fp8_bwd_spf_logged.clear()
        mha._fp8_bwd_spf_logged.update(sl)

    # one backward call; returns the compile kwargs (None: raised before)
    def _bwd(self, scale, e4m3=True, **env):  # pyre-ignore[2,3]
        T = 16
        offs = torch.tensor([0, 7, T], dtype=torch.int64)
        if e4m3:
            vqk = torch.zeros((T, 3 * _H * _D), dtype=torch.uint8).view(
                torch.float8_e4m3fn)
        else:
            vqk = torch.zeros((T, 4 * _H * _D), dtype=torch.bfloat16)[
                :, _H * _D:]
        v, q, k = (x.view(T, _H, _D) for x in vqk.split(_H * _D, dim=1))
        g = torch.zeros((T, 4 * _H * _D), dtype=torch.bfloat16)
        dv, dq, dk = (g[:, a:a + _H * _D].view(T, _H, _D)
                      for a in (512, 1024, 1536))
        dout = torch.zeros((T, _H, _D), dtype=torch.bfloat16)
        n0 = len(self.kernel.compiles)
        with _env(**env):
            try:
                mha.flydsl_hstu_attention_bwd(dout, q, k, v, offs, _N, _ALPHA,
                                              out=(dq, dk, dv),
                                              qkv_scale=scale)
            except _Stop:
                pass
        self.assertLessEqual(len(self.kernel.compiles), n0 + 1)
        return self.kernel.compiles[-1] if len(self.kernel.compiles) > n0 \
            else None

    def _spf_lines(self, word):  # pyre-ignore[2,3]
        return [m for m in self.logs.msgs if _BK in m and word in m]

    def test_off_is_the_stack(self) -> None:
        base = self._bwd(0.0173)
        self.assertEqual(base["valu"], _VALU_SHIPPED)
        self.assertEqual((base["msplit"], base["ms_sched"], base["f8b"],
                          base["mu2"], base["dqsplit"]), (1, _MSS, "all",
                                                          "k128", True))
        for knob in (None, "", "0"):
            for fv in (None, "snap", "1"):
                for s in (0.0173, self.s15, 2.0 ** -6):
                    kw = self._bwd(s, **{_BK: knob, _FK: fv})
                    self.assertEqual(kw, base, (knob, fv, s))
        self.assertFalse([m for m in self.logs.msgs if _BK in m])

    def test_stray_token_stripped_off(self) -> None:
        base = self._bwd(0.0173)
        for valu in ("undef,rfl,ilp,msat,nobr,tsc", "tsc,undef,rfl,ilp,msat,nobr",
                     "undef,rfl,ilp,msat,nobr,TSC"):
            kw = self._bwd(0.0173, DLRM_HSTU_FP8_BWD_VALU=valu)
            self.assertEqual(kw, base, valu)                   # never armed
        lines = self._spf_lines("stripped")
        self.assertEqual(len(lines), 1, lines)                 # one line
        self.assertIn("DLRM_HSTU_FP8_BWD_VALU", lines[0])
        self.assertFalse(self._spf_lines("ARMED"))

    def test_stray_token_from_another_path(self) -> None:
        # a "tsc" already in config["valu"] at the hook (here: the kernel's
        # SHIPPED dict, with the VALU env list off) is stripped too
        self.kernel.SHIPPED["valu"] = "msat,tsc"
        kw = self._bwd(0.0173, DLRM_HSTU_FP8_BWD_VALU="0")
        self.assertEqual(kw["valu"], "msat")
        self.kernel.SHIPPED["valu"] = "tsc"
        kw = self._bwd(0.0173, DLRM_HSTU_FP8_BWD_VALU="0")
        self.assertNotIn("valu", kw)
        lines = self._spf_lines("stripped")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn('config["valu"]', lines[0])

    def test_fold_arms_once(self) -> None:
        base = self._bwd(0.0173)
        on = {_BK: "1", _FK: "snap"}
        kw = self._bwd(self.s15, **on)
        self.assertEqual(kw["valu"], _VALU_TSC)
        self.assertEqual(dict(kw, valu=_VALU_SHIPPED), base)  # only the token
        armed = self._spf_lines("ARMED")
        self.assertEqual(len(armed), 1)
        self.assertIn("j=-15 e8m0_byte=112", armed[0])
        self.assertIn("valu=" + _VALU_TSC, armed[0])
        # per call: another snapped j and the full fold arm (FWD_SPF=1)
        kw = self._bwd(self.s14, **{_BK: "1", _FK: "1"})
        self.assertEqual(kw["valu"], _VALU_TSC)
        self.assertEqual(len(self._spf_lines("ARMED")), 1)    # once/process
        # a stray env token under "1": stripped, then added by the knob
        kw = self._bwd(self.s15, DLRM_HSTU_FP8_BWD_VALU="undef,rfl,ilp,msat,nobr,tsc",
                       **on)
        self.assertEqual(kw["valu"], _VALU_TSC)

    def test_fold_per_call_grid_check(self) -> None:
        on = {_BK: "1", _FK: "snap"}
        self.assertIsNotNone(self._bwd(self.s15, **on))
        n = len(self.kernel.compiles)
        # a double next to s_-15, not an f32 (it rounds to s_-15): refused as
        # passed; the launcher must not round the scale to f32 first
        near = self.s15 * (1.0 + 2.0 ** -40)
        self.assertNotEqual(near, self.s15)
        self.assertEqual(self.spf.f32(near), self.s15)
        for bad, pat in ((0.0173, "SPF grid"), (2.0 ** -6, "SPF grid"),
                         (_from_bits(_bits(self.s15) + 1), "SPF grid"),
                         (near, "SPF grid"),
                         (self.spf.grid_scale(-6, _ALPHA), "range"),
                         (self.spf.grid_scale(-41, _ALPHA), "range")):
            with self.assertRaisesRegex(RuntimeError, pat, msg=bad) as cm:
                self._bwd(bad, **on)
            self.assertIn(_BK, str(cm.exception))
        self.assertEqual(len(self.kernel.compiles), n)   # before compile
        self.assertIsNotNone(self._bwd(self.s15, **on))

    def test_fold_needs_fwd_spf(self) -> None:
        for fv in (None, "", "0"):
            with self.assertRaisesRegex(ValueError, _FK):
                self._bwd(self.s15, **{_BK: "1", _FK: fv})
        # at parse: also on a bf16 (calibration) call
        with self.assertRaisesRegex(ValueError, _FK):
            self._bwd(1.0, e4m3=False, **{_BK: "1", _FK: None})

    def test_invalid_values(self) -> None:
        for bad in ("2", "true", "snap", "on", " 1"):
            for e4 in (True, False):
                with self.assertRaises(ValueError, msg=(bad, e4)):
                    self._bwd(self.s15 if e4 else 1.0, e4m3=e4,
                              **{_BK: bad, _FK: "snap"})

    def test_fold_body_fail_closed(self) -> None:
        on = {_BK: "1", _FK: "snap"}
        for env, pat in (({"DLRM_HSTU_FP8_BWD_VF4": "0"}, "vf4"),
                         ({"DLRM_HSTU_FP8_BWD_VALU": "undef,rfl,ilp,nobr"}, "msat"),
                         ({"DLRM_HSTU_FP8_BWD_MSPLIT": "2"}, "msplit"),
                         ({"DLRM_HSTU_FP8_BWD_MSPLIT": "0"}, "msplit"),
                         ({"DLRM_HSTU_FP8_BWD_QK_MFMA": "bf16"}, "vf4"),
                         ({"DLRM_HSTU_FP8_BWD_FULL": None}, "vf4")):
            with self.assertRaisesRegex(ValueError, pat, msg=env):
                self._bwd(self.s15, **on, **env)
        self.assertFalse(self._spf_lines("ARMED"))

    def test_bf16_calls_not_armed(self) -> None:
        kw = self._bwd(1.0, e4m3=False, **{_BK: "1", _FK: "snap"})
        self.assertNotIn("tsc", str(kw.get("valu", "")))
        self.assertFalse(self._spf_lines("ARMED"))

    def test_kernel_range_must_agree(self) -> None:
        self.kernel.TSC_K_MAX = -2
        with self.assertRaisesRegex(RuntimeError, "guard range"):
            self._bwd(self.s15, **{_BK: "1", _FK: "snap"})

    def test_unserved_e4m3_call_refused(self) -> None:
        T = 16
        offs = torch.tensor([0, 7, T], dtype=torch.int64)

        def call(e4m3):  # pyre-ignore[2,3]
            dt = torch.float8_e4m3fn if e4m3 else torch.bfloat16
            q = torch.zeros((T, _H, _D), dtype=torch.uint8).view(dt) if e4m3 \
                else torch.zeros((T, _H, _D), dtype=dt)
            z = torch.zeros((T, _H, _D), dtype=torch.bfloat16)
            return mha.try_flydsl_attention_bwd(
                dout=z, q=q, k=q, v=q, dq=z.clone(), dk=z.clone(),
                dv=z.clone(), seq_offsets=offs, num_targets=None, N=_N,
                alpha=_ALPHA, max_attn_len=0, contextual_seq_len=0,
                num_softmax_heads=0, qkv_scale=self.s15)

        with _env(HSTU_ATTN_BWD="triton"):
            with _env(**{_BK: "1", _FK: "snap"}):
                with self.assertRaisesRegex(ValueError, "not served"):
                    call(True)
                self.assertFalse(call(False))          # calibration: Triton
            for knob in (None, "0"):
                with _env(**{_BK: knob, _FK: "snap"}):
                    self.assertFalse(call(True))

    def test_warmup_scale_helper(self) -> None:
        s = 2.0 ** -6
        with _env(**{_BK: None}):
            self.assertIs(mha.fp8_bwd_spf_warmup_scale(s, _ALPHA), s)
        with _env(**{_BK: "1", _FK: "snap"}):
            w = mha.fp8_bwd_spf_warmup_scale(s, _ALPHA)
        self.assertEqual(self.spfb.arm_check_bwd(w, _ALPHA), -14)

    def _fw(self):  # pyre-ignore[3]
        try:
            from generative_recommenders.dlrm_v4.train import fp8_warmup
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"fp8_warmup not importable here: {error}")
        return fp8_warmup

    def test_warmup_keeps_the_latch(self) -> None:
        fw = self._fw()
        key = ("generative_recommenders.ops.flydsl.hstu_flydsl_mha",
               "_fp8_bwd_spf_logged")
        self.assertNotIn(key[1], fw._MHA_SETS + fw._MHA_DICTS)
        self.assertIn(key, fw._KEEP_GLOBALS)
        with open(fw.__file__) as f:
            src = f.read()
        self.assertEqual(len(re.findall(
            r"fp8_bwd_spf_warmup_scale\(s, alpha\)", src)), 3)
        self.assertNotIn("qkv_scale=s,", src)
        # _run: the latch before take(), the reprint right after restore()
        run = src[src.index("\ndef _run("):]
        i_before = run.index("bwd_spf_before = dict(mha._fp8_bwd_spf_logged)")
        i_take = run.index("guards.take(verify=verify)")
        i_restore = run.index("        guards.restore()\n")
        i_re = run.index("_spf_bwd_armed_reprint(mha, bwd_spf_before)")
        self.assertLess(i_before, i_take)
        self.assertLess(i_restore, i_re)
        self.assertNotIn("guards.", run[i_restore + 25:i_re])
        self.assertEqual(src.count("_spf_bwd_armed_reprint("), 2)  # def + _run

    def test_f2_capture_then_reprint(self) -> None:
        # F2 through the real _Guards (mha logger muted) and _Runner (stdout
        # captured): the ARMED line is latched but not logged; after restore()
        # _spf_bwd_armed_reprint logs it once; later calls log nothing.
        fw = self._fw()
        on = {_BK: "1", _FK: "snap"}

        class _Bufs:
            def reset(self) -> None:
                pass

        r = fw._Runner(torch.device("cpu"), _Bufs(), True, set(), False)

        def f2() -> None:
            self.assertEqual(self._bwd(self.s14, **on)["valu"], _VALU_TSC)

        guards = fw._Guards()
        out = io.StringIO()
        with mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None), \
                contextlib.redirect_stdout(out):
            before = dict(mha._fp8_bwd_spf_logged)
            guards.take(verify=True)
            try:
                r.run("F2", "flydsl_bwd_full_fp8", "bwd_spf_knob_test", f2)
                inside = list(self.logs.msgs)
            finally:
                guards.restore()
            n = fw._spf_bwd_armed_reprint(mha, before)
            # a later warm-up pass (latch kept) logs nothing
            n2 = fw._spf_bwd_armed_reprint(mha, dict(mha._fp8_bwd_spf_logged))
        guards.verify_globals()
        self.assertEqual((r.ok, r.failed), (1, 0))
        self.assertEqual(inside, [])                # muted during the item
        self.assertEqual((n, n2), (1, 0))
        armed = self._spf_lines("ARMED")
        self.assertEqual(armed, ["[hstu-attn] " + mha._fp8_bwd_spf_logged["armed"]])
        self.assertIn("j=-14 e8m0_byte=113", armed[0])
        self.assertEqual(guards.globals_changed, [])
        self.assertEqual(out.getvalue(), "")
        # the timed region: production calls at j=-15 log no ARMED line
        for _ in range(3):
            self.assertEqual(self._bwd(self.s15, **on)["valu"], _VALU_TSC)
        self.assertEqual(len(self._spf_lines("ARMED")), 1)

    def test_guards_keep_the_latch(self) -> None:
        fw = self._fw()
        guards = fw._Guards()
        guards.take(verify=True)
        try:
            # what _bwd_spf_tokens does on its first armed call
            mha._fp8_bwd_spf_logged["armed"] = "bwd_spf_knob_test sentinel"
        finally:
            guards.restore()
        guards.verify_globals()
        self.assertEqual(guards.globals_changed, [])
        self.assertEqual(guards.changed, [])
        self.assertEqual(mha._fp8_bwd_spf_logged,
                         {"armed": "bwd_spf_knob_test sentinel"})

    def test_real_kernel_module(self) -> None:
        try:
            from generative_recommenders.ops.flydsl import hstu_bwd_dkdv as kb
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_bwd_dkdv not importable here: {error}")
        self.assertEqual((kb.TSC_K_MIN, kb.TSC_K_MAX),
                         (self.spfb.BWD_J_MIN, self.spfb.BWD_J_MAX))
        for j in range(-45, 0):
            s = self.spf.grid_scale(j, _ALPHA)
            ok = self.spfb.BWD_J_MIN <= j <= self.spfb.BWD_J_MAX
            self.assertEqual(kb.tsc_exponent(_ALPHA, s), j if ok else None)


if __name__ == "__main__":
    unittest.main()
