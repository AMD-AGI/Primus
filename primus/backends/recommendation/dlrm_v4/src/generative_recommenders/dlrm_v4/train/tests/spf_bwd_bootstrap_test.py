#!/usr/bin/env python3
# pyre-strict
"""DLRM_HSTU_FP8_LAB_BWD_SPF (yambda_5b.gin item (106), SPF fold bwd; PRECISION CLASS; code default OFF).

Ship-17 row (106) binds the backward half of the SPF fold in gin. The knob rides on item (105)
(DLRM_HSTU_FP8_LAB_FWD_SPF in {"snap", "1"}): dlrm_v4/train/_env_bootstrap.py sets it to "1" only when the
forward knob is armed, and otherwise sets nothing and logs one "declined" WARNING. A pre-set env value wins:
"0" is the kill switch. The per-call behaviour (scale check, body check, the "tsc" token, the ARMED line) is
pinned by bwd_spf_knob_test.py. CPU only: no torch, no GPU, no FlyDSL. Pinned here:

  ENV   apply_env_bootstrap in a scrubbed os.environ: True + FWD_SPF snap or 1 -> "1" and one INFO line;
        True without it (unset, 0, fold, ...) -> unset and one WARNING naming the forward value; False -> "0";
        unbound -> nothing. A pre-set value wins ("0" kill switch; "1" is honoured verbatim). The knob changes
        no other env value.
  GIN   exactly one binding (True); its item block documents WHAT / IMPACT / NUMERICS / COLD START / ARMED /
        Must be ABSENT / the kill switch, stays within 80 columns and carries no measured number or absolute
        path. The ARMED texts it quotes are the bootstrap's and the launcher's.
  BOOT  one parameter (Optional[bool] = None); the module imports no torch; a clean child running
        train_ranker's Phase 1 (gin.parse_config_file(yambda_5b.gin, skip_unknown=True); apply_env_bootstrap())
        exports "1" when the forward knob is armed (by gin item (105) or a pre-set env), declines otherwise,
        and honours the kill switch.
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

_REPO = pathlib.Path(__file__).resolve().parents[4]
_GR = _REPO / "generative_recommenders"
_GIN = _GR / "dlrm_v4/train/gin/yambda_5b.gin"
_BOOT = _GR / "dlrm_v4/train/_env_bootstrap.py"
_SPFB = _GR / "ops/fp8_qkv_spf_bwd.py"

_KNOB = "DLRM_HSTU_FP8_LAB_BWD_SPF"
_FWD = "DLRM_HSTU_FP8_LAB_FWD_SPF"
_ACTED = f"env bootstrap: {_KNOB}=1"
_DECLINED = f"env bootstrap: {_KNOB} declined: "
_HEAD = re.compile(r"^# \(106\) SPF fold bwd ")


def _try_import(name: str) -> Optional[Any]:  # pyre-ignore[3]
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - environment specific
        return None


gin = _try_import("gin")


def _gin_lines() -> List[str]:
    return _GIN.read_text().splitlines()


def _binding_index(lines: List[str]) -> int:
    hits = [i for i, ln in enumerate(lines) if re.match(rf"^\s*apply_env_bootstrap\.{_KNOB}\s*=", ln)]
    if len(hits) != 1:
        raise AssertionError(f"expected one {_KNOB} binding, found {len(hits)}")
    return hits[0]


def _gin_binds_fwd() -> Optional[str]:
    """The value item (105) binds DLRM_HSTU_FP8_LAB_FWD_SPF to in yambda_5b.gin (None when unbound)."""
    for ln in _gin_lines():
        m = re.match(rf"^apply_env_bootstrap\.{_FWD}\s*=\s*(.+?)\s*$", ln)
        if m:
            v = ast.literal_eval(m.group(1))
            return {True: "1", False: "0"}.get(v, str(v)) if isinstance(v, bool) else str(v)
    return None


@unittest.skipIf(gin is None, "gin-config not installed (_env_bootstrap imports gin)")
class EnvEffectsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.eb = importlib.import_module("generative_recommenders.dlrm_v4.train._env_bootstrap")

    def _boot(self, preset: Dict[str, str], bound: Optional[bool]) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
        with mock.patch.dict(os.environ, preset, clear=True):
            with self.assertLogs(self.eb.logger, level="INFO") as cm:
                self.eb.logger.info("probe")  # assertLogs needs at least one record
                self.eb.apply_env_bootstrap(**{_KNOB: bound})
            env = dict(os.environ)
        return env, [(r.levelname, r.getMessage()) for r in cm.records]

    @staticmethod
    def _lines(recs: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        acted = [r for r in recs if r[1] == _ACTED]
        declined = [r for r in recs if r[1].startswith(_DECLINED)]
        return acted, declined

    def test_on_with_forward_snap_or_1(self) -> None:
        # (105) v2 validates the forward value at bootstrap (spf_mode: exact
        # "snap" / "1"; " snap " raises there), and its fold needs the pv=k128
        # forward env (FWD_PIPE and FWD_PV "1", gin-bound in production).
        pv = {"DLRM_HSTU_FP8_FWD_PIPE": "1", "DLRM_HSTU_FP8_FWD_PV": "1"}
        for fwd in ("snap", "1"):
            env, recs = self._boot({_FWD: fwd, **pv}, True)
            self.assertEqual(env.get(_KNOB), "1", fwd)
            self.assertEqual(self._lines(recs), ([("INFO", _ACTED)], []), fwd)

    def test_on_without_forward_declines(self) -> None:
        for preset, named in (({}, "<unset>"), ({_FWD: "0"}, "0"), ({_FWD: ""}, "<unset>")):
            env, recs = self._boot(preset, True)
            self.assertNotIn(_KNOB, env, preset)
            acted, declined = self._lines(recs)
            self.assertEqual(acted, [], preset)
            self.assertEqual(len(declined), 1, recs)
            level, msg = declined[0]
            self.assertEqual(level, "WARNING")
            self.assertIn(f"{_FWD}={named} ", msg)
            self.assertIn("item (106) rides on item (105)", msg)

    def test_invalid_forward_value_fails_at_bootstrap(self) -> None:
        # (105) v2: an invalid forward value raises ValueError in
        # apply_env_bootstrap (before RUN_START), before item (106)'s block.
        for bad in ("fold", " snap ", "true"):
            with mock.patch.dict(os.environ, {_FWD: bad}, clear=True):
                with self.assertRaises(ValueError) as cm:
                    self.eb.apply_env_bootstrap(**{_KNOB: True})
                self.assertIn("use unset, '0', 'snap' or '1'", str(cm.exception))
                self.assertNotIn(_KNOB, os.environ)

    def test_kill_switch_and_preset_win(self) -> None:
        for preset_val in ("0", "1"):
            env, recs = self._boot({_KNOB: preset_val, _FWD: "snap"}, True)
            self.assertEqual(env.get(_KNOB), preset_val)
            self.assertEqual(self._lines(recs), ([], []))
            self.assertIn(("INFO", f"env bootstrap: honoring pre-set {_KNOB}={preset_val} (overrides gin binding)"), recs)

    def test_false_and_unbound(self) -> None:
        env, recs = self._boot({_FWD: "snap"}, False)
        self.assertEqual(env.get(_KNOB), "0")
        self.assertIn(("INFO", f"env bootstrap: {_KNOB}=0"), recs)
        env, recs = self._boot({_FWD: "snap"}, None)
        self.assertNotIn(_KNOB, env)
        self.assertEqual([r for r in recs if _KNOB in r[1]], [])

    def test_knob_changes_nothing_else(self) -> None:
        on, _ = self._boot({_FWD: "snap"}, True)
        unbound, _ = self._boot({_FWD: "snap"}, None)
        declined, _ = self._boot({}, True)
        none, _ = self._boot({}, None)
        self.assertEqual({k for k in set(on) | set(unbound) if on.get(k) != unbound.get(k)}, {_KNOB})
        self.assertEqual(declined, none)


class GinItemTest(unittest.TestCase):
    def test_binds_once_true_with_item_text(self) -> None:
        lines = _gin_lines()
        i_bind = _binding_index(lines)
        self.assertEqual(lines[i_bind], f"apply_env_bootstrap.{_KNOB} = True")
        heads = [i for i in range(i_bind) if _HEAD.match(lines[i])]
        self.assertTrue(heads, "no '# (106) SPF fold bwd' item header before the binding")
        i_head = heads[-1]
        self.assertTrue(lines[i_head - 1].startswith("# ----"), lines[i_head - 1])
        block = lines[i_head:i_bind]
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = " ".join(" ".join(ln.lstrip("#").split()) for ln in block)
        for marker in ("DECLARED PRECISION CHANGE", "rides on item (105)", "Code default OFF", "bound True below",
                       "WHAT:", '"tsc"', "IMPACT:", "NUMERICS: PRECISION CLASS", "BIT-IDENTICAL",
                       "user-OKed convergence pair", "COLD START:", "before RUN_START", "ARMED (every rank, once):",
                       f'"{_ACTED}"', f'"{_KNOB}=1 ARMED: bwd_dkdv VALU token tsc"', "Must be ABSENT:",
                       f'"{_KNOB} declined:"', f'"{_KNOB}: VALU token tsc in"', f'"honoring pre-set {_KNOB}"',
                       f"Opt out / kill switch: {_KNOB}=0", "a pre-set env value wins over this binding"):
            self.assertIn(marker, text, marker)
        self.assertIsNone(re.search(r"\d\.\d|%|\bms\b|\bVGPR|\bpJ\b|\bGB\b", text), text)
        self.assertNotIn("/apps/", text)
        self.assertNotIn("/home/", text)

    def test_quoted_launcher_lines_exist(self) -> None:
        src = _SPFB.read_text()
        self.assertIn('f"{BWD_KNOB}=1 ARMED: bwd_dkdv VALU token {TOKEN} (', src)
        self.assertIn('f"{BWD_KNOB}: VALU token {TOKEN} in {source}=', src)
        boot = _BOOT.read_text()
        self.assertIn(f'"{_DECLINED}"', boot)
        self.assertIn(f'logger.info("{_ACTED}")', boot)


_CHILD = textwrap.dedent(
    """
    import json, os, sys
    root, gin_path = sys.argv[1], sys.argv[2]
    import gin
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
    boot.apply_env_bootstrap()
    keep = ("DLRM_HSTU_FP8_LAB_BWD_SPF", "DLRM_HSTU_FP8_LAB_FWD_SPF")
    print("SPFB_GIN_CHILD " + json.dumps({"env": {k: os.environ.get(k) for k in keep},
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
    out = [x for x in r.stdout.splitlines() if x.startswith("SPFB_GIN_CHILD ")]
    got = json.loads(out[0][len("SPFB_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


class BootstrapTest(unittest.TestCase):
    def test_one_parameter_no_torch(self) -> None:
        tree = ast.parse(_BOOT.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(names.count(_KNOB), 1)
        arg = next(a for a in fn.args.args if a.arg == _KNOB)
        self.assertEqual(ast.unparse(arg.annotation) if arg.annotation else None, "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[_KNOB], ast.Constant) and defaults[_KNOB].value is None)
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("torch", imported)

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_gin_phase1(self) -> None:
        fwd_gin = _gin_binds_fwd()
        fwd_on = fwd_gin in ("snap", "1")
        cases = (
            ({}, "1" if fwd_on else None),  # item (105) bound in gin -> armed; absent -> declined
            ({_FWD: "snap"}, "1"),  # a pre-set forward knob arms it as well
            ({_KNOB: "0", _FWD: "snap"}, "0"),  # kill switch
        )
        for preset, want in cases:
            rc, log, got = _run_child(preset)
            self.assertEqual(rc, 0, log)
            assert got is not None, log
            self.assertEqual(got["env"].get(_KNOB), want, (preset, fwd_gin))
            self.assertFalse(got["torch_imported"])


if __name__ == "__main__":
    unittest.main()
