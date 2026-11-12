#!/usr/bin/env python3
# pyre-strict
"""DLRM_HSTU_FP8_BWD_GSPLIT (yambda_5b.gin item (101) GSPX; R15 G-split; BITWISE; code default OFF).

The knob is an env-bootstrap effect only: with item (27) (DLRM_HSTU_FP8_BWD_ATTN_V2) on, dlrm_v4/train/_env_bootstrap.py
appends the bwd_dkdv tokens "gsp,xtd" to the MSPLIT_SCHED tokens item (27) appends to item (19)'s gin-bound list; the
FlyDSL launcher passes DLRM_HSTU_FP8_BWD_MSPLIT_SCHED to the kernel build unchanged. CPU only: no torch, no GPU, no
FlyDSL. Pinned here:

  ENV   apply_env_bootstrap in a scrubbed os.environ with item (19)'s and item (27)'s bindings read from yambda_5b.gin:
        GSPLIT on -> MSPLIT_SCHED == the 16 shipped tokens + "gsp,xtd" and one INFO line; off, unbound or the env
        kill switch "0" -> the shipped 16-token string and no GSPLIT line; the knob changes no other env value; a
        pre-set MSPLIT_SCHED is used verbatim; ATTN_V2 off -> item (19)'s list, no token; a ws layout other than l128t,
        MSPLIT other than 1, an item (19) list without mid,sb or with a token gsp rejects -> no gsp/xtd. Every
        refusal is exactly one "env bootstrap: DLRM_HSTU_FP8_BWD_GSPLIT declined: <reason>" WARNING.
  GIN   exactly one binding (True or False); its item block names the bound value, documents WHAT / IMPACT /
        NUMERICS: BIT-IDENTICAL / ARMED (with the 18-token env echo) / Must be ABSENT / the kill switch, stays within
        80 columns and carries no measured number; item (19)/(27)'s bindings are the ones the strings below assume.
  BOOT  one parameter (Optional[bool] = None) bound by one env-wins _bind_bool; the module imports no torch; a clean
        child running train_ranker's Phase 1 (gin.parse_config_file(yambda_5b.gin, skip_unknown=True);
        apply_env_bootstrap()) exports the string of the bound value, and the kill switch and a pre-set string win.
  KERN  ops/flydsl/hstu_bwd_dkdv.py still accepts gsp and xtd and asserts exactly what the bootstrap checks; the
        launcher still reads MSPLIT_SCHED per call; launch_slurm.sh forwards the DLRM_HSTU_FP8_ prefix (kill switch).
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
_MHA = _GR / "ops/flydsl/hstu_flydsl_mha.py"
_LAUNCHER = _REPO / "scripts/launch_slurm.sh"

_KNOB = "DLRM_HSTU_FP8_BWD_GSPLIT"
_MSS = "DLRM_HSTU_FP8_BWD_MSPLIT_SCHED"
_HEAD = re.compile(r"^# \(\w+\) GSPX ")
# item (19)'s list "mid,sb,wsl,sgv5,la" + item (27)'s tokens (VALU msat -> mc; layout l128t -> dzr), as shipped in
# ship16 (544e691); the knob appends gsp,xtd after them.
_SHIPPED = "mid,sb,wsl,sgv5,la,xt,rs2,lc,ek,nk,wh,mc,dzr,bst,ob0,obj"
_GSPX = _SHIPPED + ",gsp,xtd"
_ACTED = "env bootstrap: DLRM_HSTU_FP8_BWD_GSPLIT=1 (MSPLIT_SCHED += gsp,xtd)"
_DECLINED = "env bootstrap: DLRM_HSTU_FP8_BWD_GSPLIT declined: "
# the apply_env_bootstrap bindings of items (19) and (27) (and the ws layout/aux item (27) defaults)
_ATTN_NAMES = (
    "DLRM_HSTU_FP8_BWD_VALU",
    "DLRM_HSTU_FP8_BWD_MSPLIT",
    _MSS,
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED",
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT",
    "DLRM_HSTU_FP8_BWD_ATTN_V2",
    "DLRM_HSTU_FP8_BWD_WS_LAYOUT",
    "DLRM_HSTU_FP8_BWD_WS_AUX",
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


def _gin_attn_kwargs() -> Dict[str, Any]:  # pyre-ignore[3]
    """Items (19)/(27)'s gin values as apply_env_bootstrap kwargs (each bound at most once; literals only)."""
    b = _gin_bindings()
    kw: Dict[str, Any] = {}
    for name in _ATTN_NAMES:
        vals = b.get(name, [])
        if len(vals) > 1:
            raise AssertionError(f"{name} bound {len(vals)} times in {_GIN}")
        if vals:
            kw[name] = ast.literal_eval(vals[0])
    return kw


def _binding_index(lines: List[str]) -> int:
    hits = [i for i, ln in enumerate(lines) if re.match(rf"^\s*apply_env_bootstrap\.{_KNOB}\s*=", ln)]
    if len(hits) != 1:
        raise AssertionError(f"expected one {_KNOB} binding, found {len(hits)}")
    return hits[0]


def _bound_value() -> str:
    lines = _gin_lines()
    m = re.fullmatch(rf"apply_env_bootstrap\.{_KNOB} = (True|False)", lines[_binding_index(lines)])
    if m is None:
        raise AssertionError(lines[_binding_index(lines)])
    return m.group(1)


# =============================================================================================== ENV
@unittest.skipIf(gin is None, "gin-config not installed (_env_bootstrap imports gin)")
class EnvEffectsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.eb = importlib.import_module("generative_recommenders.dlrm_v4.train._env_bootstrap")

    def _boot(self, preset: Dict[str, str], **overrides: Any) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
        """apply_env_bootstrap(items (19)/(27) from the gin, then `overrides`) in an os.environ holding only
        `preset`; returns (the env after, [(level, message)] of the bootstrap's log records)."""
        kw = _gin_attn_kwargs()
        kw.update(overrides)
        with mock.patch.dict(os.environ, preset, clear=True):
            with self.assertLogs(self.eb.logger, level="INFO") as cm:
                self.eb.apply_env_bootstrap(**kw)
            env = dict(os.environ)
        return env, [(r.levelname, r.getMessage()) for r in cm.records]

    def _gsplit_lines(self, recs: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        acted = [r for r in recs if r[1] == _ACTED]
        declined = [r for r in recs if r[1].startswith(_DECLINED)]
        return acted, declined

    def _assert_declined(self, recs: List[Tuple[str, str]], *reasons: str) -> None:
        acted, declined = self._gsplit_lines(recs)
        self.assertEqual(acted, [], recs)
        self.assertEqual(len(declined), 1, recs)
        level, msg = declined[0]
        self.assertEqual(level, "WARNING")
        for reason in reasons:
            self.assertIn(reason, msg[len(_DECLINED):])

    def test_on_appends_gsp_xtd(self) -> None:
        env, recs = self._boot({}, **{_KNOB: True})
        self.assertEqual(env.get(_MSS), _GSPX)
        self.assertEqual(env.get(_KNOB), "1")
        acted, declined = self._gsplit_lines(recs)
        self.assertEqual(acted, [("INFO", _ACTED)])
        self.assertEqual(declined, [])
        self.assertIn(("INFO", f"env bootstrap: {_MSS}={_GSPX}"), recs)
        # the item (19)/(27) values the 18 tokens rely on
        self.assertEqual(env.get("DLRM_HSTU_FP8_BWD_MSPLIT"), "1")
        self.assertEqual(env.get("DLRM_HSTU_FP8_BWD_WS_LAYOUT"), "l128t")

    def test_off_unbound_is_the_shipped_string(self) -> None:
        for bound, want_knob in ((False, "0"), (None, None)):
            env, recs = self._boot({}, **{_KNOB: bound})
            self.assertEqual(env.get(_MSS), _SHIPPED, bound)
            self.assertEqual(env.get(_KNOB), want_knob, bound)
            self.assertEqual(self._gsplit_lines(recs), ([], []), bound)

    def test_knob_changes_nothing_else(self) -> None:
        on, _ = self._boot({}, **{_KNOB: True})
        off, _ = self._boot({}, **{_KNOB: False})
        unbound, _ = self._boot({}, **{_KNOB: None})
        diff_on = {k for k in set(on) | set(unbound) if on.get(k) != unbound.get(k)}
        diff_off = {k for k in set(off) | set(unbound) if off.get(k) != unbound.get(k)}
        self.assertEqual(diff_on, {_KNOB, _MSS})
        self.assertEqual(diff_off, {_KNOB})

    def test_env_kill_switch_wins(self) -> None:
        for preset in ("0", "true", ""):  # only "1" arms it
            env, recs = self._boot({_KNOB: preset}, **{_KNOB: True})
            self.assertEqual(env.get(_MSS), _SHIPPED, preset)
            self.assertEqual(env.get(_KNOB), preset)
            self.assertEqual(self._gsplit_lines(recs), ([], []), preset)
            self.assertIn(("INFO", f"env bootstrap: honoring pre-set {_KNOB}={preset} (overrides gin binding)"), recs)
        env, _ = self._boot({_KNOB: "1"}, **{_KNOB: False})  # a pre-set "1" arms it over a False binding
        self.assertEqual(env.get(_MSS), _GSPX)

    def test_preset_msplit_sched_wins_verbatim(self) -> None:
        for preset in ("mid,sb,wsl", _SHIPPED, _SHIPPED + ",gsp,xtd,gss2"):
            env, recs = self._boot({_MSS: preset}, **{_KNOB: True})
            self.assertEqual(env.get(_MSS), preset)
            self._assert_declined(recs, f"{_MSS} is pre-set")

    def test_attn_v2_off_adds_no_token(self) -> None:
        item19 = _gin_attn_kwargs()[_MSS]
        for preset, bound in (({}, False), ({}, None), ({"DLRM_HSTU_FP8_BWD_ATTN_V2": "0"}, True)):
            env, recs = self._boot(preset, **{_KNOB: True, "DLRM_HSTU_FP8_BWD_ATTN_V2": bound})
            self.assertEqual(env.get(_MSS), item19, (preset, bound))
            self._assert_declined(recs, "DLRM_HSTU_FP8_BWD_ATTN_V2 is off")

    def test_layout_not_l128t_declines(self) -> None:
        # item (27) drops dzr without l128t as well; gsp/xtd are not added
        no_dzr = _SHIPPED.replace(",dzr", "")
        for preset, bound in (({}, "l128"), ({"DLRM_HSTU_FP8_BWD_WS_LAYOUT": "c64"}, None),
                              ({"DLRM_HSTU_FP8_BWD_WS_LAYOUT": "0"}, None)):
            env, recs = self._boot(preset, **{_KNOB: True, "DLRM_HSTU_FP8_BWD_WS_LAYOUT": bound})
            self.assertEqual(env.get(_MSS), no_dzr, (preset, bound))
            self._assert_declined(recs, "DLRM_HSTU_FP8_BWD_WS_LAYOUT=")

    def test_msplit_not_1_declines(self) -> None:
        for preset, bound in (({"DLRM_HSTU_FP8_BWD_MSPLIT": "0"}, 1), ({"DLRM_HSTU_FP8_BWD_MSPLIT": "2"}, 1),
                              ({}, 2), ({}, None)):
            env, recs = self._boot(preset, **{_KNOB: True, "DLRM_HSTU_FP8_BWD_MSPLIT": bound})
            self.assertEqual(env.get(_MSS), _SHIPPED, (preset, bound))
            self._assert_declined(recs, "DLRM_HSTU_FP8_BWD_MSPLIT=")

    def test_item19_list_without_mid_sb_declines(self) -> None:
        tail = _SHIPPED[len("mid,sb,wsl,sgv5,la"):]  # item (27)'s tokens
        for item19, want, reason in (("wsl,sgv5,la", "wsl,sgv5,la" + tail, "lacks mid,sb"),
                                     ("mid,wsl,sgv5,la", "mid,wsl,sgv5,la" + tail, "lacks sb"),
                                     ("0", "0", "lacks mid,sb"),  # an off item (19) list stays off
                                     (None, None, "lacks mid,sb")):
            env, recs = self._boot({}, **{_KNOB: True, _MSS: item19})
            self.assertEqual(env.get(_MSS), want, item19)
            self._assert_declined(recs, reason)

    def test_item19_list_with_rejected_token_declines(self) -> None:
        for tok in ("ntm", "oh", "xe", "so", "anti"):
            item19 = "mid,sb,wsl,sgv5,la," + tok
            env, recs = self._boot({}, **{_KNOB: True, _MSS: item19})
            self.assertNotIn("gsp", env.get(_MSS, "").split(","), tok)
            self.assertNotIn("xtd", env.get(_MSS, "").split(","), tok)
            self._assert_declined(recs, "has " + tok)

    def test_reasons_are_joined_in_one_line(self) -> None:
        env, recs = self._boot({_MSS: "mid,sb", "DLRM_HSTU_FP8_BWD_WS_LAYOUT": "c64"}, **{_KNOB: True})
        self.assertEqual(env.get(_MSS), "mid,sb")
        self._assert_declined(recs, f"{_MSS} is pre-set", "DLRM_HSTU_FP8_BWD_WS_LAYOUT=c64")


# =============================================================================================== GIN
class GinItemTest(unittest.TestCase):
    def test_item19_27_bindings_match_the_expected_strings(self) -> None:
        kw = _gin_attn_kwargs()
        self.assertEqual(kw.get(_MSS), "mid,sb,wsl,sgv5,la")
        self.assertEqual(kw.get("DLRM_HSTU_FP8_BWD_MSPLIT"), 1)
        self.assertIn("msat", str(kw.get("DLRM_HSTU_FP8_BWD_VALU")).split(","))
        self.assertIs(kw.get("DLRM_HSTU_FP8_BWD_ATTN_V2"), True)
        self.assertIn(kw.get("DLRM_HSTU_FP8_BWD_WS_LAYOUT", "l128t"), ("l128t",))

    def test_gin_binds_once_with_item_text(self) -> None:
        lines = _gin_lines()
        i_bind = _binding_index(lines)
        bound = _bound_value()
        dups = sorted(n for n, v in _gin_bindings().items() if len(v) > 1)
        self.assertEqual(dups, [])
        heads = [i for i in range(i_bind) if _HEAD.match(lines[i])]
        self.assertTrue(heads, "no '# (<n>) GSPX' item header before the binding")
        i_head = heads[-1]
        self.assertTrue(lines[i_head - 1].startswith("# ----"), lines[i_head - 1])
        block = lines[i_head:i_bind]
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = " ".join(" ".join(ln.lstrip("#").split()) for ln in block)
        self.assertIn(f"bound {bound} below", text)
        self.assertNotIn(f"bound {'False' if bound == 'True' else 'True'} below", text)
        for marker in ("GSPX", "items (19) and (27)", "ops/flydsl/hstu_bwd_dkdv.py", "dlrm_v4/train/_env_bootstrap.py",
                       "Code default OFF", "WHAT:", '"gsp"', '"xtd"', "IMPACT:", "NUMERICS: BIT-IDENTICAL",
                       "items (19)+(27)", "COLD START:", "before RUN_START", "ARMED (every rank, once):",
                       '"env bootstrap: DLRM_HSTU_FP8_BWD_GSPLIT=1 (MSPLIT_SCHED += gsp,xtd)"',
                       "Must be ABSENT:", '"DLRM_HSTU_FP8_BWD_GSPLIT declined:"',
                       '"honoring pre-set DLRM_HSTU_FP8_BWD_MSPLIT_SCHED"',
                       f"Opt out / kill switch: {_KNOB}=0", "a pre-set env value wins over this binding",
                       "only the value 1 arms it"):
            self.assertIn(marker, text, marker)
        # the env echo (wrapped over two lines in the gin text)
        self.assertIn(f"{_MSS}={_GSPX}", re.sub(r",\s+", ",", text))
        # no measured number in gin text (timings, ratios, percentages, register counts live in the evidence)
        self.assertIsNone(re.search(r"\d\.\d|%|\bms\b|\bVGPR|\bpJ\b|\bGB\b", text), text)
        # no absolute path (compliance gate G2 scans gin text for them)
        self.assertNotIn("/apps/", text)
        self.assertNotIn("/home/", text)


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
    keep = ("DLRM_HSTU_FP8_BWD_GSPLIT", "DLRM_HSTU_FP8_BWD_MSPLIT_SCHED", "DLRM_HSTU_FP8_BWD_ATTN_V2")
    print("GSPLIT_GIN_CHILD " + json.dumps({"env": {k: os.environ.get(k) for k in keep},
                                            "torch_imported": "torch" in sys.modules}, sort_keys=True))
    """
)


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DLRM_", "TRITON_", "HSTU_", "GR_", "PYTORCH_"))}
    # item (116) OSNB (gin ON) appends osn,osnb after this knob's gsp,xtd: its env kill switch keeps these
    # cases on item (101)'s own string (osnb_bootstrap_test.py pins the composed one)
    env["DLRM_HSTU_FP8_BWD_OSNB"] = "0"
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_REPO), env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, str(_REPO), str(_GIN)], env=env, cwd=str(_REPO),
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("GSPLIT_GIN_CHILD ")]
    got = json.loads(out[0][len("GSPLIT_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
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
        binds = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id.startswith("_bind") and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == _KNOB]
        self.assertEqual(len(binds), 1)
        (call,) = binds
        self.assertEqual(call.func.id, "_bind_bool")  # env wins; True -> "1"
        self.assertEqual([getattr(a, "id", None) for a in call.args[1:]], [_KNOB])
        # the knob is read back only as the exact "1" (only the value 1 arms it)
        reads = [n for n in ast.walk(fn) if isinstance(n, ast.Compare) and isinstance(n.left, ast.Call)
                 and n.left.args and isinstance(n.left.args[0], ast.Constant) and n.left.args[0].value == _KNOB]
        self.assertEqual(len(reads), 1)
        self.assertEqual(ast.unparse(reads[0]), f"os.environ.get('{_KNOB}', '0') == '1'")
        # this module must stay importable before the heavy imports: no torch import anywhere in it
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("torch", imported)

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_gin_phase1_exports_the_bound_string_and_env_wins(self) -> None:
        want_on = {"True": ("1", _GSPX), "False": ("0", _SHIPPED)}[_bound_value()]
        cases = (
            ({}, want_on),
            ({_KNOB: "0"}, ("0", _SHIPPED)),  # kill switch
            ({_KNOB: "1"}, ("1", _GSPX)),  # a pre-set "1" arms it whatever the binding
            ({_MSS: "mid,sb,wsl"}, (want_on[0], "mid,sb,wsl")),  # a pre-set string is used verbatim
        )
        for preset, (knob, mss) in cases:
            rc, log, got = _run_child(preset)
            self.assertEqual(rc, 0, log)
            assert got is not None, log
            self.assertEqual(got["env"].get(_KNOB), knob, preset)
            self.assertEqual(got["env"].get(_MSS), mss, preset)
            self.assertEqual(got["env"].get("DLRM_HSTU_FP8_BWD_ATTN_V2"), "1", preset)
            self.assertFalse(got["torch_imported"])


# =============================================================================================== KERN
class KernelContractTest(unittest.TestCase):
    """The bootstrap's refusal list mirrors the kernel's asserts; if the kernel changes them, revisit the bootstrap."""

    def test_kernel_accepts_and_asserts_what_the_bootstrap_checks(self) -> None:
        src = _KERNEL.read_text()
        for needle in ('MS_GSP = "gsp" in ms_tok', 'MS_XTD = "xtd" in ms_tok',
                       '"gsp needs msplit=1 mid,sb (no ntm/oh/xe/so/anti)"',
                       '"xtd needs msplit=1, xt and the l128t ws layout"'):
            self.assertIn(needle, src, needle)
        allow = re.search(r"assert set\(ms_tok\) <= \{([^}]*)", src)
        self.assertIsNotNone(allow)
        assert allow is not None
        self.assertIn('"gsp"', allow.group(1))
        self.assertIn('"xtd"', allow.group(1))

    def test_launcher_reads_the_string_per_call(self) -> None:
        src = _MHA.read_text()
        self.assertIn(f'os.environ.get("{_MSS}", "").strip()', src)
        self.assertIn('config["ms_sched"]', src)

    def test_launcher_forwards_the_prefix(self) -> None:
        if not _LAUNCHER.is_file():
            self.skipTest("no scripts/launch_slurm.sh in this checkout")
        self.assertTrue(_KNOB.startswith("DLRM_HSTU_FP8_"))
        self.assertIn('for v in "${!DLRM_HSTU_FP8_@}"; do', _LAUNCHER.read_text())


if __name__ == "__main__":
    unittest.main()
