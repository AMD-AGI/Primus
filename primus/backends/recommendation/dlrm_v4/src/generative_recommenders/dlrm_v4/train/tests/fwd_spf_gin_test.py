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
"""Ship17 row (105) SPF fold fwd: DLRM_HSTU_FP8_LAB_FWD_SPF is bound ON in yambda_5b.gin (DECLARED PRECISION
CHANGE; knob rule: an apply_env_bootstrap binding with a documented item block; a pre-set env value wins: "0" is
the kill switch, "snap" the lab's snapped control).

  AST   apply_env_bootstrap declares the knob once (Optional[bool] = None, right after DLRM_HSTU_FP8_LAB_FWD_RIDER)
        and binds it with one env-wins _bind_bool, right after item (100)'s two _bind_bool calls and before any
        other use of the knob in the function (item (106)'s end-of-function block reads it). No parameter is
        declared twice.
  GIN   exactly one "= True" binding over all gin files (yambda_5b.gin), right after item (100)'s bindings, with
        a documented (105) item block of at most 80 columns whose quoted log texts exist in the code.
  RUN   a clean child interpreter runs train_ranker's Phase-1 sequence (gin.parse_config_file(yambda_5b.gin,
        skip_unknown=True), then apply_env_bootstrap()): unset -> "1" (with item (100)'s V8SWZ=1 and RIDER=1);
        a pre-set "0" is kept (kill switch); a pre-set "snap" is kept. The child imports no torch.
CPU only; no torch import."""

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import unittest
from typing import Any, Dict, List, Optional, Tuple

try:
    import gin  # noqa: F401
except ImportError:  # pragma: no cover
    gin = None  # type: ignore[assignment]

KNOB = "DLRM_HSTU_FP8_LAB_FWD_SPF"
RB1 = ("DLRM_HSTU_FP8_LAB_FWD_V8SWZ", "DLRM_HSTU_FP8_LAB_FWD_RIDER")
# Ship17 item (120) LUT-SiLU fwd, bound ON right after this item: its bootstrap
# check needs the pv=k128 forward too, so a run that kills FWD_PV / FWD_PIPE
# sets its kill switch as well.
LUT = "DLRM_HSTU_FP8_LUT_SILU_FWD"
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GR = os.path.dirname(os.path.dirname(_TRAIN))  # generative_recommenders/
_ROOT = os.path.dirname(_GR)  # holds generative_recommenders/
_GIN_DIR = os.path.join(_TRAIN, "gin")
_GIN = os.path.join(_GIN_DIR, "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_HEAD = "# (105) SPF fold fwd: "


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _gin_lines() -> List[str]:
    return _read(_GIN).splitlines()


def _bind_stmt_index(body: List[ast.stmt], name: str) -> List[int]:
    """Indexes of the top-level statements `_bind_bool(<name>, ...)`."""
    out = []
    for i, st in enumerate(body):
        if (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                and getattr(st.value.func, "id", None) == "_bind_bool" and st.value.args
                and isinstance(st.value.args[0], ast.Constant) and st.value.args[0].value == name):
            out.append(i)
    return out


class BootstrapAstTest(unittest.TestCase):
    def setUp(self) -> None:
        tree = ast.parse(_read(_BOOT))
        self.fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")

    def test_one_parameter_after_rider(self) -> None:
        fn = self.fn
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        self.assertEqual(names.count(KNOB), 1)
        self.assertEqual(names.index(KNOB), names.index(RB1[1]) + 1)
        arg = next(a for a in fn.args.args if a.arg == KNOB)
        self.assertEqual(ast.unparse(arg.annotation) if arg.annotation else None, "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[KNOB], ast.Constant) and defaults[KNOB].value is None)

    def test_one_env_wins_bind_right_after_rb1(self) -> None:
        fn = self.fn
        # Bare helper calls naming the knob (_bind_bool / _bind_value / ...);
        # item (106)'s os.environ.get(KNOB, ...) read is an attribute call.
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == KNOB]
        self.assertEqual(len(calls), 1)
        self.assertEqual(getattr(calls[0].func, "id", None), "_bind_bool")
        self.assertEqual([getattr(a, "id", None) for a in calls[0].args[1:]], [KNOB])
        # No direct os.environ[KNOB] = ... write: the env-wins rule is _bind_bool's.
        stores = [n for n in ast.walk(fn) if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
                  and isinstance(n.slice, ast.Constant) and n.slice.value == KNOB]
        self.assertEqual(stores, [])
        i_v8, i_rd, i_spf = (_bind_stmt_index(fn.body, n) for n in RB1 + (KNOB,))
        self.assertEqual((len(i_v8), len(i_rd), len(i_spf)), (1, 1, 1))
        self.assertEqual((i_rd[0], i_spf[0]), (i_v8[0] + 1, i_rd[0] + 1))
        # Before the function's end and before any other use of the knob
        # (item (106)'s block reads os.environ[KNOB] as the last statement).
        self.assertLess(i_spf[0], len(fn.body) - 1)
        for i, st in enumerate(fn.body):
            uses = [n for n in ast.walk(st) if isinstance(n, ast.Constant) and n.value == KNOB]
            if uses and i != i_spf[0]:
                self.assertGreater(i, i_spf[0], ast.unparse(st)[:200])


class GinBindingTest(unittest.TestCase):
    def test_exactly_one_true_binding_in_all_gin_files(self) -> None:
        binds = []
        for fn in sorted(os.listdir(_GIN_DIR)):
            if fn.endswith(".gin"):
                for ln in _read(os.path.join(_GIN_DIR, fn)).splitlines():
                    m = re.match(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % KNOB, ln)
                    if m:
                        binds.append((fn, m.group(1)))
        self.assertEqual(binds, [("yambda_5b.gin", "True")])
        names = [m.group(1) for m in (re.match(r"^\s*apply_env_bootstrap\.(\w+)\s*=", ln)
                                      for ln in _gin_lines()) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])

    def test_item_block_documents_the_row(self) -> None:
        lines = _gin_lines()
        i_bind = next(i for i, ln in enumerate(lines) if ln.startswith(f"apply_env_bootstrap.{KNOB} "))
        self.assertEqual(lines[i_bind], f"apply_env_bootstrap.{KNOB} = True")
        heads = [i for i in range(i_bind) if lines[i].startswith(_HEAD)]
        self.assertTrue(heads, f"no '{_HEAD}' item header before the binding")
        i_head = heads[-1]
        # Placed right after item (100)'s two bindings; the next item follows.
        self.assertTrue(lines[i_head - 1].startswith("# ----"), lines[i_head - 1])
        self.assertEqual(lines[i_head - 2], f"apply_env_bootstrap.{RB1[1]} = True")
        self.assertEqual(lines[i_head - 3], f"apply_env_bootstrap.{RB1[0]} = True")
        self.assertTrue(lines[i_bind + 1].startswith("# ----"), lines[i_bind + 1])
        block = lines[i_head:i_bind]
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in lines[i_head - 1:i_bind + 1] if len(ln) > 80], [])
        text = " ".join(" ".join(ln.lstrip("#").split()) for ln in block)
        for marker in ("DECLARED PRECISION CHANGE", "Code default OFF", "bound True below", "WHAT:", "IMPACT:",
                       "NUMERICS: PRECISION CLASS", "BIT-IDENTICAL", "user-OKed convergence pair",
                       "bundled with item (106)", "Resume only from a checkpoint of the same arm",
                       "COLD START:", "one build serves every j", "before RUN_START",
                       "ARMED (every rank):", f'"env bootstrap: {KNOB}=1"',
                       f'"[hstu-fwd] {KNOB}=1 armed: spf=True (QK MFMA E8M0 A-scale byte',
                       "the once-latch is keyed by the mode and is not in fp8_warmup._MHA_SETS",
                       "no SPF arming line prints in the timed region",
                       f'"[hstu-fp8] {KNOB}: qkv scale snapped onto the SPF grid at publication:',
                       "Must be ABSENT:", f'"honoring pre-set {KNOB}"', f'"{KNOB}=snap armed"',
                       '"is not on the SPF grid"', '"outside the arming range"',
                       f'"{KNOB}=1 needs the pv=k128 forward"', "\"use unset, '0', 'snap' or '1'\"",
                       "Item (106) (SPF fold bwd) rides on this item.",
                       f"Opt out / kill switch: {KNOB}=0",
                       "a pre-set env value wins over this binding"):
            self.assertIn(marker, text, marker)
        self.assertNotIn("/apps/", text)
        self.assertNotIn("/home/", text)

    def test_quoted_log_texts_exist(self) -> None:
        spf = _read(os.path.join(_GR, "ops", "fp8_qkv_spf.py"))
        mha = _read(os.path.join(_GR, "ops", "flydsl", "hstu_flydsl_mha.py"))
        tf8 = _read(os.path.join(_GR, "ops", "triton", "triton_fp8.py"))
        boot = _read(_BOOT)
        self.assertIn('KNOB = "DLRM_HSTU_FP8_LAB_FWD_SPF"', spf)
        # fp8_qkv_spf is stdlib-only: its fail-closed messages, at run time.
        sys.path.insert(0, _ROOT)
        try:
            from generative_recommenders.ops import fp8_qkv_spf as m
        finally:
            sys.path.remove(_ROOT)
        alpha = 1.0 / (128 ** 0.5)
        for s, text in ((0.0146325, "is not on the SPF grid"),
                        (m.grid_scale(m.J_MAX + 1, alpha), "outside the arming range")):
            with self.assertRaises(RuntimeError) as cm:
                m.arm_check(s, alpha)
            self.assertIn(text, str(cm.exception))
            self.assertTrue(str(cm.exception).startswith(KNOB + ": "), str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            m.spf_mode({KNOB: "on"})
        self.assertIn("use unset, '0', 'snap' or '1'", str(cm.exception))
        for text in ('f"[hstu-fwd] {spf.KNOB}=1 armed: spf=True (QK MFMA E8M0 A-scale "',
                     'f"[hstu-fwd] {spf.KNOB}=snap armed: snapped control, kernel "',
                     'f"{spf.KNOB}=1 needs the pv=k128 forward (DLRM_HSTU_FP8_FWD_PV=1 "'):
            self.assertIn(text, mha, text)
        self.assertIn('"[hstu-fp8] %s: qkv scale snapped onto the SPF grid at publication: %s"', tf8)
        self.assertIn('"env bootstrap: honoring pre-set %s=%s (overrides gin binding)"', boot)
        self.assertIn('logger.info("env bootstrap: %s=%s", name, os.environ[name])', boot)


_CHILD = textwrap.dedent(
    """
    import json, os, sys
    root, gin_path = sys.argv[1], sys.argv[2]
    import gin
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
    boot.apply_env_bootstrap()
    print("SPF_GIN_CHILD " + json.dumps({"env": {k: os.environ.get(k) for k in sys.argv[3:]},
                                        "torch_imported": "torch" in sys.modules}, sort_keys=True))
    """
)
_PREFIXES = ("DLRM_", "TRITON_", "HSTU_", "GR_", "PYTORCH_")


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    """A clean child interpreter: only `preset` of the knob prefixes is set when it starts."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_PREFIXES)}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_ROOT, env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, _ROOT, _GIN, *RB1, KNOB], env=env, cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("SPF_GIN_CHILD ")]
    got = json.loads(out[0][len("SPF_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


@unittest.skipIf(gin is None, "gin-config not installed")
class GinRunTest(unittest.TestCase):
    def _check(self, preset: Dict[str, str], want: Dict[str, str]) -> None:
        rc, log, got = _run_child(preset)
        self.assertEqual(rc, 0, log)
        assert got is not None, log
        self.assertEqual(got["env"], want, log)
        self.assertFalse(got["torch_imported"], log)

    def test_unset_binds_fold_with_rb1(self) -> None:
        self._check({}, {RB1[0]: "1", RB1[1]: "1", KNOB: "1"})

    def test_preset_wins(self) -> None:
        self._check({KNOB: "0"}, {RB1[0]: "1", RB1[1]: "1", KNOB: "0"})  # kill switch
        self._check({KNOB: "snap"}, {RB1[0]: "1", RB1[1]: "1", KNOB: "snap"})  # snapped control

    def _check_fails(self, preset: Dict[str, str], text: str) -> None:
        rc, log, got = _run_child(preset)
        self.assertNotEqual(rc, 0, log)
        self.assertIsNone(got, log)
        self.assertIn("ValueError", log)
        self.assertIn(text, log)

    def test_fails_closed_at_bootstrap(self) -> None:
        # (105) v2: an invalid value or the fold without the pv=k128 forward
        # raises in apply_env_bootstrap, i.e. before RUN_START.
        for bad in ("true", "on", " 1", "2"):
            self._check_fails({KNOB: bad}, "use unset, '0', 'snap' or '1'")
        for k in ("DLRM_HSTU_FP8_FWD_PV", "DLRM_HSTU_FP8_FWD_PIPE"):
            self._check_fails({k: "0"}, "needs the pv=k128 forward")
            self._check_fails({k: "0", KNOB: "1"}, "needs the pv=k128 forward")
            self._check({k: "0", KNOB: "0", LUT: "0"}, {RB1[0]: "1", RB1[1]: "1", KNOB: "0"})
            self._check({k: "0", KNOB: "snap", LUT: "0"}, {RB1[0]: "1", RB1[1]: "1", KNOB: "snap"})


if __name__ == "__main__":
    unittest.main()
