#!/usr/bin/env python3
# pyre-strict
"""DLRM_HSTU_FP8_BWD_OSNB (yambda_5b.gin item (116) OSNB; R15 out-gemm snake order; BITWISE; code default OFF).

The knob is an env-bootstrap effect only: where item (101) (DLRM_HSTU_FP8_BWD_GSPLIT) appends "gsp,xtd" to the
MSPLIT_SCHED tokens item (27) appends to item (19)'s gin-bound list, dlrm_v4/train/_env_bootstrap.py appends the
bwd_dkdv tokens "osn,osnb" right after them; the FlyDSL launcher passes DLRM_HSTU_FP8_BWD_MSPLIT_SCHED to the kernel
build unchanged. CPU only: no torch, no GPU, no FlyDSL. Pinned here:

  ENV   apply_env_bootstrap in a scrubbed os.environ with the item (19)/(27)/(101) bindings read from yambda_5b.gin:
        OSNB on -> MSPLIT_SCHED == the 16 shipped tokens + "gsp,xtd,osn,osnb" and one INFO line right after item
        (101)'s; off, unbound or the env kill switch "0" (only "1" arms it) -> item (101)'s 18-token string and no
        OSNB line; the knob changes no other env value; item (101) off (unbound, False or its kill switch "0") ->
        the 16-token string; item (101) declined or item (27) off -> no token; a pre-set MSPLIT_SCHED is used
        verbatim (GSPLIT and OSNB both decline). Every refusal is exactly one
        "env bootstrap: DLRM_HSTU_FP8_BWD_OSNB declined: <reason>" WARNING.
  GIN   exactly one binding (True or False), its item block directly after item (101)'s binding; the block names
        the bound value, documents WHAT / IMPACT / NUMERICS: BIT-IDENTICAL / COLD START / ARMED (with the 20-token
        env echo) / Must be ABSENT / the kill switch / Evidence, stays within 80 columns and carries no measured
        number and no absolute path.
  BOOT  one parameter (Optional[bool] = None) right after DLRM_HSTU_FP8_BWD_GSPLIT, bound by one env-wins
        _bind_bool right after GSPLIT's and read back only as == "1"; osn,osnb are appended only inside item
        (101)'s gsp,xtd branch; the module imports no torch; a clean child running train_ranker's Phase 1
        (gin.parse_config_file(yambda_5b.gin, skip_unknown=True); apply_env_bootstrap()) exports the string of the
        bound values with the matching log lines, and both kill switches and a pre-set string win.
  KERN  ops/flydsl/hstu_bwd_dkdv.py accepts osn and osnb and asserts "osn needs gsp" / "osnb needs osn"; the gsp
        out-gemm loop of ms_out_g, run on stand-ins, issues the snake order under osn (each consecutive pair of
        MFMAs shares an operand) and one sched_barrier(0x7F6) before every out-gemm MFMA but the first under osnb,
        while every accumulator keeps exactly its shipped (A, B) operands.
"""

import ast
import importlib
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

# recommendation/ (this file: recommendation/generative_recommenders/dlrm_v4/train/tests/)
_REPO = pathlib.Path(__file__).resolve().parents[4]
_GR = _REPO / "generative_recommenders"
_GIN = _GR / "dlrm_v4/train/gin/yambda_5b.gin"
_BOOT = _GR / "dlrm_v4/train/_env_bootstrap.py"
_KERNEL = _GR / "ops/flydsl/hstu_bwd_dkdv.py"

_KNOB = "DLRM_HSTU_FP8_BWD_OSNB"
_GSPLIT = "DLRM_HSTU_FP8_BWD_GSPLIT"
_ATTN_V2 = "DLRM_HSTU_FP8_BWD_ATTN_V2"
_MSS = "DLRM_HSTU_FP8_BWD_MSPLIT_SCHED"
_HEAD = re.compile(r"^# \(\w+\) OSNB ")
# item (19)'s list "mid,sb,wsl,sgv5,la" + item (27)'s tokens (VALU msat -> mc; layout l128t -> dzr), as shipped in
# ship16 (544e691); item (101) appends gsp,xtd and this knob osn,osnb after them.
_SHIPPED = "mid,sb,wsl,sgv5,la,xt,rs2,lc,ek,nk,wh,mc,dzr,bst,ob0,obj"
_GSPX = _SHIPPED + ",gsp,xtd"
_OSNB = _GSPX + ",osn,osnb"
_ACTED = "env bootstrap: DLRM_HSTU_FP8_BWD_OSNB=1 (MSPLIT_SCHED += osn,osnb)"
_DECLINED = "env bootstrap: DLRM_HSTU_FP8_BWD_OSNB declined: "
_GS_ACTED = "env bootstrap: DLRM_HSTU_FP8_BWD_GSPLIT=1 (MSPLIT_SCHED += gsp,xtd)"
_GS_DECLINED = "env bootstrap: DLRM_HSTU_FP8_BWD_GSPLIT declined: "
# the apply_env_bootstrap bindings of items (19), (27) and (101) (and the ws layout/aux item (27) defaults)
_BASE_NAMES = (
    "DLRM_HSTU_FP8_BWD_VALU",
    "DLRM_HSTU_FP8_BWD_MSPLIT",
    _MSS,
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED",
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT",
    _ATTN_V2,
    "DLRM_HSTU_FP8_BWD_WS_LAYOUT",
    "DLRM_HSTU_FP8_BWD_WS_AUX",
    _GSPLIT,
)


def _try_import(name: str) -> Optional[Any]:  # pyre-ignore[3]
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - environment specific
        return None


gin = _try_import("gin")


def _gin_lines() -> List[str]:
    return _GIN.read_text().splitlines()


def _gin_bindings() -> Dict[str, List[str]]:
    """Every live `apply_env_bootstrap.<NAME> = <value>` line of yambda_5b.gin, by name (raw value text)."""
    out: Dict[str, List[str]] = {}
    for ln in _gin_lines():
        m = re.match(r"^apply_env_bootstrap\.(\w+)\s*=\s*(.+?)\s*$", ln)
        if m:
            out.setdefault(m.group(1), []).append(m.group(2))
    return out


def _gin_base_kwargs() -> Dict[str, Any]:  # pyre-ignore[3]
    """Items (19)/(27)/(101)'s gin values as apply_env_bootstrap kwargs (each bound at most once; literals only)."""
    b = _gin_bindings()
    kw: Dict[str, Any] = {}
    for name in _BASE_NAMES:
        vals = b.get(name, [])
        if len(vals) > 1:
            raise AssertionError(f"{name} bound {len(vals)} times in {_GIN}")
        if vals:
            kw[name] = ast.literal_eval(vals[0])
    return kw


def _binding_index(lines: List[str], knob: str) -> int:
    hits = [i for i, ln in enumerate(lines) if re.match(rf"^\s*apply_env_bootstrap\.{knob}\s*=", ln)]
    if len(hits) != 1:
        raise AssertionError(f"expected one {knob} binding, found {len(hits)}")
    return hits[0]


def _bound_value(knob: str) -> str:
    lines = _gin_lines()
    m = re.fullmatch(rf"apply_env_bootstrap\.{knob} = (True|False)", lines[_binding_index(lines, knob)])
    if m is None:
        raise AssertionError(lines[_binding_index(lines, knob)])
    return m.group(1)


def _apply_fn() -> ast.FunctionDef:
    tree = ast.parse(_BOOT.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")


# =============================================================================================== ENV
@unittest.skipIf(gin is None, "gin-config not installed (_env_bootstrap imports gin)")
class EnvEffectsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.eb = importlib.import_module("generative_recommenders.dlrm_v4.train._env_bootstrap")

    def _boot(self, preset: Dict[str, str], **overrides: Any) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
        """apply_env_bootstrap(items (19)/(27)/(101) from the gin, then `overrides`) in an os.environ holding only
        `preset`; returns (the env after, [(level, message)] of the bootstrap's log records)."""
        kw = _gin_base_kwargs()
        kw.update(overrides)
        with mock.patch.dict(os.environ, preset, clear=True):
            with self.assertLogs(self.eb.logger, level="INFO") as cm:
                self.eb.apply_env_bootstrap(**kw)
            env = dict(os.environ)
        return env, [(r.levelname, r.getMessage()) for r in cm.records]

    def _osnb_lines(self, recs: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        acted = [r for r in recs if r[1] == _ACTED]
        declined = [r for r in recs if r[1].startswith(_DECLINED)]
        return acted, declined

    def _assert_declined(self, recs: List[Tuple[str, str]], reason: str) -> None:
        acted, declined = self._osnb_lines(recs)
        self.assertEqual(acted, [], recs)
        self.assertEqual(len(declined), 1, recs)
        level, msg = declined[0]
        self.assertEqual(level, "WARNING")
        self.assertTrue(msg[len(_DECLINED):].startswith(reason), msg)

    def test_on_appends_osn_osnb_after_gsp_xtd(self) -> None:
        env, recs = self._boot({}, **{_GSPLIT: True, _KNOB: True})
        self.assertEqual(env.get(_MSS), _OSNB)
        self.assertEqual(len(_OSNB.split(",")), 20)
        self.assertEqual(env.get(_KNOB), "1")
        self.assertEqual(env.get(_GSPLIT), "1")
        acted, declined = self._osnb_lines(recs)
        self.assertEqual(acted, [("INFO", _ACTED)])
        self.assertEqual(declined, [])
        msgs = [m for _, m in recs]
        self.assertEqual(msgs.index(_ACTED), msgs.index(_GS_ACTED) + 1)  # right after item (101)'s line
        self.assertEqual([m for m in msgs if m.startswith(_GS_DECLINED)], [])
        self.assertIn(("INFO", f"env bootstrap: {_MSS}={_OSNB}"), recs)
        # the item (19)/(27) values the 20 tokens rely on
        self.assertEqual(env.get("DLRM_HSTU_FP8_BWD_MSPLIT"), "1")
        self.assertEqual(env.get("DLRM_HSTU_FP8_BWD_WS_LAYOUT"), "l128t")

    def test_off_unbound_is_item_101_string(self) -> None:
        for bound, want_knob in ((False, "0"), (None, None)):
            env, recs = self._boot({}, **{_GSPLIT: True, _KNOB: bound})
            self.assertEqual(env.get(_MSS), _GSPX, bound)
            self.assertEqual(env.get(_KNOB), want_knob, bound)
            self.assertEqual(self._osnb_lines(recs), ([], []), bound)
            self.assertIn(("INFO", _GS_ACTED), recs)

    def test_knob_changes_nothing_else(self) -> None:
        on, _ = self._boot({}, **{_GSPLIT: True, _KNOB: True})
        off, _ = self._boot({}, **{_GSPLIT: True, _KNOB: False})
        unbound, _ = self._boot({}, **{_GSPLIT: True, _KNOB: None})
        diff_on = {k for k in set(on) | set(unbound) if on.get(k) != unbound.get(k)}
        diff_off = {k for k in set(off) | set(unbound) if off.get(k) != unbound.get(k)}
        self.assertEqual(diff_on, {_KNOB, _MSS})
        self.assertEqual(diff_off, {_KNOB})

    def test_env_kill_switch_wins(self) -> None:
        for preset in ("0", "true", ""):  # only "1" arms it
            env, recs = self._boot({_KNOB: preset}, **{_GSPLIT: True, _KNOB: True})
            self.assertEqual(env.get(_MSS), _GSPX, preset)
            self.assertEqual(env.get(_KNOB), preset)
            self.assertEqual(self._osnb_lines(recs), ([], []), preset)
            self.assertIn(("INFO", f"env bootstrap: honoring pre-set {_KNOB}={preset} (overrides gin binding)"), recs)
        env, recs = self._boot({_KNOB: "1"}, **{_GSPLIT: True, _KNOB: False})  # a pre-set "1" arms it over False
        self.assertEqual(env.get(_MSS), _OSNB)
        self.assertEqual(self._osnb_lines(recs), ([("INFO", _ACTED)], []))

    def test_gsplit_off_declines(self) -> None:
        # item (101) unbound, bound False, or its env kill switch "0": item (27)'s 16 tokens and one WARNING
        for preset, bound in (({}, False), ({}, None), ({_GSPLIT: "0"}, True)):
            env, recs = self._boot(preset, **{_GSPLIT: bound, _KNOB: True})
            self.assertEqual(env.get(_MSS), _SHIPPED, (preset, bound))
            self.assertEqual(env.get(_KNOB), "1")
            self.assertNotIn(("INFO", _GS_ACTED), recs)
            self._assert_declined(recs, f"{_GSPLIT} is off")

    def test_gsplit_declined_declines(self) -> None:
        tail = _SHIPPED[len("mid,sb,wsl,sgv5,la"):]  # item (27)'s tokens
        for preset, kw, want in (({"DLRM_HSTU_FP8_BWD_WS_LAYOUT": "c64"}, {}, _SHIPPED.replace(",dzr", "")),
                                 ({"DLRM_HSTU_FP8_BWD_MSPLIT": "2"}, {}, _SHIPPED),
                                 ({}, {_MSS: "wsl,sgv5,la"}, "wsl,sgv5,la" + tail),
                                 ({}, {_MSS: "mid,sb,wsl,sgv5,la,ntm"}, "mid,sb,wsl,sgv5,la,ntm" + tail)):
            env, recs = self._boot(preset, **{_GSPLIT: True, _KNOB: True, **kw})
            self.assertEqual(env.get(_MSS), want, (preset, kw))
            self.assertEqual(len([r for r in recs if r[1].startswith(_GS_DECLINED)]), 1, recs)
            self._assert_declined(recs, f"{_GSPLIT} declined")

    def test_preset_msplit_sched_wins_verbatim(self) -> None:
        for preset in ("mid,sb,wsl", _GSPX, _OSNB):
            env, recs = self._boot({_MSS: preset}, **{_GSPLIT: True, _KNOB: True})
            self.assertEqual(env.get(_MSS), preset)
            gs_declined = [r for r in recs if r[1].startswith(_GS_DECLINED)]
            self.assertEqual(len(gs_declined), 1, recs)
            self.assertIn(f"{_MSS} is pre-set", gs_declined[0][1])
            self._assert_declined(recs, f"{_GSPLIT} declined")

    def test_attn_v2_off_declines(self) -> None:
        item19 = _gin_base_kwargs()[_MSS]
        for preset, attn, gs in (({}, False, True), ({}, None, True), ({_ATTN_V2: "0"}, True, True),
                                 ({}, False, False)):
            env, recs = self._boot(preset, **{_GSPLIT: gs, _KNOB: True, _ATTN_V2: attn})
            self.assertEqual(env.get(_MSS), item19, (preset, attn, gs))
            self._assert_declined(recs, f"{_ATTN_V2} is off")


# =============================================================================================== GIN
class GinItemTest(unittest.TestCase):
    def test_item_101_bindings_the_strings_assume(self) -> None:
        kw = _gin_base_kwargs()
        self.assertEqual(kw.get(_MSS), "mid,sb,wsl,sgv5,la")
        self.assertEqual(kw.get("DLRM_HSTU_FP8_BWD_MSPLIT"), 1)
        self.assertIn("msat", str(kw.get("DLRM_HSTU_FP8_BWD_VALU")).split(","))
        self.assertIs(kw.get(_ATTN_V2), True)
        self.assertIn(kw.get("DLRM_HSTU_FP8_BWD_WS_LAYOUT", "l128t"), ("l128t",))
        self.assertIn(kw.get(_GSPLIT), (True, False))

    def test_gin_binds_once_directly_after_item_101(self) -> None:
        lines = _gin_lines()
        i_bind = _binding_index(lines, _KNOB)
        bound = _bound_value(_KNOB)
        dups = sorted(n for n, v in _gin_bindings().items() if len(v) > 1)
        self.assertEqual(dups, [])
        heads = [i for i in range(i_bind) if _HEAD.match(lines[i])]
        self.assertTrue(heads, "no '# (<n>) OSNB' item header before the binding")
        i_head = heads[-1]
        # directly after item (101)'s block: its binding, "#", the rule line, then this header
        self.assertEqual(_binding_index(lines, _GSPLIT), i_head - 3)
        self.assertEqual(lines[i_head - 2], "#")
        self.assertTrue(lines[i_head - 1].startswith("# ----"), lines[i_head - 1])
        self.assertIn(" OSNB (r15): ", lines[i_head])
        block = lines[i_head:i_bind]
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block + [lines[i_bind]] if len(ln) > 80], [])
        text = " ".join(" ".join(ln.lstrip("#").split()) for ln in block)
        self.assertIn(f"bound {bound} below", text)
        self.assertNotIn(f"bound {'False' if bound == 'True' else 'True'} below", text)
        for marker in ("OSNB", "item (101)", "ops/flydsl/hstu_bwd_dkdv.py", "dlrm_v4/train/_env_bootstrap.py",
                       "Code default OFF", "WHAT:", '"gsp,xtd"', '"osn,osnb"', '"osn"', '"osnb"', "sched_barrier",
                       "IMPACT:", "NUMERICS: BIT-IDENTICAL", "COLD START:", "before RUN_START",
                       "No Triton kernel changes", "ARMED (every rank, once):", f'"{_ACTED}"', "Must be ABSENT:",
                       '"DLRM_HSTU_FP8_BWD_OSNB declined:"', '"DLRM_HSTU_FP8_BWD_GSPLIT declined:"',
                       f"Opt out / kill switch: {_KNOB}=0", "a pre-set env value wins over this binding",
                       "only the value 1 arms it", "Evidence:"):
            self.assertIn(marker, text, marker)
        # the env echo (wrapped over two lines in the gin text)
        self.assertIn(f"{_MSS}={_OSNB}", re.sub(r",\s+", ",", text))
        # no measured number in gin text (timings, ratios, percentages, register counts live in the evidence)
        self.assertIsNone(re.search(r"\d\.\d|%|\bms\b|\bVGPR|\bpJ\b|\bGB\b|\bJ/", text), text)
        # no absolute path (compliance gate G2 scans gin text for them)
        self.assertNotIn("/apps/", text)
        self.assertNotIn("/home/", text)


# =============================================================================================== BOOT
_CHILD = textwrap.dedent(
    """
    import json, logging, os, sys
    root, gin_path = sys.argv[1], sys.argv[2]
    import gin
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    recs = []

    class _Keep(logging.Handler):
        def emit(self, record):
            recs.append([record.levelname, record.getMessage()])

    boot.logger.addHandler(_Keep())
    boot.logger.setLevel(logging.INFO)
    gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
    boot.apply_env_bootstrap()
    keep = ("DLRM_HSTU_FP8_BWD_OSNB", "DLRM_HSTU_FP8_BWD_GSPLIT", "DLRM_HSTU_FP8_BWD_MSPLIT_SCHED",
            "DLRM_HSTU_FP8_BWD_ATTN_V2")
    log = [r for r in recs if "_BWD_OSNB" in r[1] or "_BWD_GSPLIT" in r[1]]
    print("OSNB_GIN_CHILD " + json.dumps({"env": {k: os.environ.get(k) for k in keep}, "log": log,
                                          "torch_imported": "torch" in sys.modules}, sort_keys=True))
    """
)


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DLRM_", "TRITON_", "HSTU_", "GR_", "PYTORCH_"))}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_REPO), env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, str(_REPO), str(_GIN)], env=env, cwd=str(_REPO),
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("OSNB_GIN_CHILD ")]
    got = json.loads(out[0][len("OSNB_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


class BootstrapTest(unittest.TestCase):
    def test_one_parameter_right_after_gsplit_one_env_wins_bind(self) -> None:
        fn = _apply_fn()
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(names.count(_KNOB), 1)
        self.assertEqual(names.index(_KNOB), names.index(_GSPLIT) + 1)
        arg = next(a for a in fn.args.args if a.arg == _KNOB)
        self.assertEqual(ast.unparse(arg.annotation) if arg.annotation else None, "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[_KNOB], ast.Constant) and defaults[_KNOB].value is None)
        binds = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id.startswith("_bind") and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == _KNOB]
        self.assertEqual(len(binds), 1)
        (call,) = binds
        self.assertEqual(call.func.id, "_bind_bool")  # env wins; True -> "1"
        self.assertEqual([getattr(a, "id", None) for a in call.args[1:]], [_KNOB])

        def _bind_of(stmt: ast.stmt) -> Optional[str]:
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "_bind_bool"
                    and stmt.value.args and isinstance(stmt.value.args[0], ast.Constant)):
                return stmt.value.args[0].value
            return None

        order = [_bind_of(s) for s in fn.body]
        self.assertEqual(order.count(_GSPLIT), 1)
        self.assertEqual(order.index(_KNOB), order.index(_GSPLIT) + 1)  # the statement right after GSPLIT's bind
        # the knob is read back only as the exact "1" (only the value 1 arms it)
        reads = [n for n in ast.walk(fn) if isinstance(n, ast.Compare) and isinstance(n.left, ast.Call)
                 and n.left.args and isinstance(n.left.args[0], ast.Constant) and n.left.args[0].value == _KNOB]
        self.assertEqual(len(reads), 1)
        self.assertEqual(ast.unparse(reads[0]), f"os.environ.get('{_KNOB}', '0') == '1'")
        # this module must stay importable before the heavy imports: no torch import anywhere in it
        tree = ast.parse(_BOOT.read_text())
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("torch", imported)

    def test_osn_osnb_appended_only_in_the_gsp_branch(self) -> None:
        fn = _apply_fn()
        adds: Dict[Tuple[str, ...], List[ast.AugAssign]] = {}
        for n in ast.walk(fn):
            if (isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == "_ms_add"
                    and isinstance(n.value, ast.List)):
                key = tuple(e.value for e in n.value.elts if isinstance(e, ast.Constant))
                adds.setdefault(key, []).append(n)
        self.assertEqual(len(adds.get(("gsp", "xtd"), [])), 1)
        self.assertEqual(len(adds.get(("osn", "osnb"), [])), 1)
        (gsp,) = adds[("gsp", "xtd")]
        (osn,) = adds[("osn", "osnb")]
        found = False
        for n in ast.walk(fn):
            for field in ("body", "orelse"):
                block = getattr(n, field, None)
                if isinstance(block, list) and any(s is gsp for s in block):
                    i = next(k for k, s in enumerate(block) if s is gsp)
                    found = found or any(isinstance(s, ast.If) and ast.unparse(s.test) == "_osnb"
                                         and any(t is osn for t in s.body) for s in block[i + 1:])
        self.assertTrue(found, "osn,osnb must be appended by an `if _osnb:` after gsp,xtd in the same block")

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_gin_phase1_exports_the_bound_string_and_env_wins(self) -> None:
        gs_on = _bound_value(_GSPLIT) == "True"
        os_on = _bound_value(_KNOB) == "True"
        base = _GSPX if gs_on else _SHIPPED
        want = ("1" if os_on else "0", _OSNB if (gs_on and os_on) else base)
        cases = (
            ({}, want),
            ({_KNOB: "0"}, ("0", base)),  # kill switch
            ({_KNOB: "1"}, ("1", _OSNB if gs_on else _SHIPPED)),  # a pre-set "1" arms it whatever the binding
            ({_GSPLIT: "0"}, (want[0], _SHIPPED)),  # item (101)'s kill switch turns this item off too
            ({_MSS: "mid,sb,wsl"}, (want[0], "mid,sb,wsl")),  # a pre-set string is used verbatim
        )
        for preset, (knob, mss) in cases:
            rc, log, got = _run_child(preset)
            self.assertEqual(rc, 0, log)
            assert got is not None, log
            self.assertEqual(got["env"].get(_KNOB), knob, preset)
            self.assertEqual(got["env"].get(_MSS), mss, preset)
            self.assertEqual(got["env"].get(_ATTN_V2), "1", preset)
            self.assertFalse(got["torch_imported"])
            msgs = [m for _, m in got["log"]]
            appended = mss.endswith(",osn,osnb")
            self.assertEqual(msgs.count(_ACTED), 1 if appended else 0, (preset, msgs))
            declined = [m for m in msgs if m.startswith(_DECLINED)]
            self.assertEqual(len(declined), 1 if (knob == "1" and not appended) else 0, (preset, msgs))
            if not preset and gs_on and os_on:  # the shipped gin: ARMED lines present, refusals absent
                self.assertEqual(msgs.count(_GS_ACTED), 1, msgs)
                self.assertEqual([m for m in msgs if m.startswith(_GS_DECLINED)], [], msgs)


# =============================================================================================== KERN
class KernelContractTest(unittest.TestCase):
    """The bootstrap appends osn,osnb only where gsp is appended; the kernel asserts the same dependency."""

    def test_kernel_accepts_and_asserts_osn_osnb(self) -> None:
        src = _KERNEL.read_text()
        for needle in ('MS_OSN = "osn" in ms_tok', 'MS_OSNB = "osnb" in ms_tok',
                       'assert not (MS_OSN and not MS_GSP), "osn needs gsp"',
                       'assert not (MS_OSNB and not MS_OSN), "osnb needs osn"'):
            self.assertIn(needle, src, needle)
        allow = re.search(r"assert set\(ms_tok\) <= \{([^}]*)", src)
        self.assertIsNotNone(allow)
        assert allow is not None
        for tok in ('"gsp"', '"xtd"', '"osn"', '"osnb"'):
            self.assertIn(tok, allow.group(1))

    def test_out_gemm_order_on_stand_ins(self) -> None:
        tree = ast.parse(_KERNEL.read_text())
        fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "ms_out_g"]
        self.assertEqual(len(fns), 1)
        loops = [n for n in ast.walk(fns[0]) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                 and n.target.id == "jj" and ast.unparse(n.iter) == "range_constexpr(NJG)"]
        self.assertEqual(len(loops), 1)
        code = compile(ast.Module(body=[loops[0]], type_ignores=[]), str(_KERNEL), "exec")

        def run(osn: bool, osnb: bool, la: int, njg: int = 8, nt: int = 2) -> List[Tuple[Any, ...]]:
            ev: List[Tuple[Any, ...]] = []

            class _Rocdl:
                @staticmethod
                def sched_barrier(mask: int) -> None:
                    ev.append(("barrier", mask))

            def _mfma(a: str, b: str, c: int, sa: str, sb: str) -> int:
                ev.append(("mfma", a, b, c, sa, sb))
                return c

            ns: Dict[str, Any] = {
                "range_constexpr": range, "NJG": njg, "NT": nt, "la": la, "MS_OSN": osn, "MS_OSNB": osnb,
                "bq": {j: f"B{j}" for j in range(min(la, njg - 1) + 1)}, "par": None,
                "ms_bop_g": lambda par, j: f"B{j}", "a": [f"A{n}" for n in range(nt)],
                "acc": list(range(nt * njg)), "sa": "sa", "sb": "sb", "rocdl": _Rocdl, "mfma_k128_sc": _mfma,
            }
            exec(code, ns)  # noqa: S102 - the kernel's own loop on stand-ins
            return ev

        for la in (0, 1):  # drain (no lookahead) and loop (obj lookahead of one d tile)
            base, osn, osnb = run(False, False, la), run(True, False, la), run(True, True, la)
            mf = {k: [e for e in v if e[0] == "mfma"] for k, v in (("base", base), ("osn", osn), ("osnb", osnb))}
            # every accumulator gets exactly one MFMA, with the shipped (A, B, scales)
            per_acc = {k: {e[3]: e[1:3] + e[4:] for e in v} for k, v in mf.items()}
            self.assertEqual(sorted(per_acc["base"]), list(range(16)))
            self.assertEqual(per_acc["osn"], per_acc["base"])
            self.assertEqual(per_acc["osnb"], per_acc["base"])
            self.assertTrue(all(len(v) == 16 for v in mf.values()))
            # base: nt = 0, 1 in every d tile, no barrier
            self.assertEqual([e[1:3] for e in mf["base"]], [(f"A{n}", f"B{j}") for j in range(8) for n in (0, 1)])
            self.assertEqual(len(base), 16)
            # osn: snake order (odd d tiles nt = 1, 0); each consecutive pair shares A or B; no barrier
            snake = [(f"A{n}", f"B{j}") for j in range(8) for n in ((0, 1) if j % 2 == 0 else (1, 0))]
            self.assertEqual([e[1:3] for e in mf["osn"]], snake)
            self.assertEqual(len(osn), 16)
            pairs = list(zip(snake, snake[1:]))
            self.assertEqual(sum(1 for p, q in pairs if p[0] == q[0] or p[1] == q[1]), 15)
            # osnb: the same order, one sched_barrier(0x7F6) before every MFMA but the first
            self.assertEqual([e[1:3] for e in mf["osnb"]], snake)
            self.assertEqual(osnb[0][0], "mfma")
            self.assertEqual([e[0] for e in osnb[1:]], ["barrier", "mfma"] * 15)
            self.assertEqual({e[1] for e in osnb if e[0] == "barrier"}, {0x7F6})


if __name__ == "__main__":
    unittest.main()
