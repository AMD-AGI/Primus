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
"""Ship17 item (120) LUT-SiLU fwd: DLRM_HSTU_FP8_LUT_SILU_FWD is bound ON in yambda_5b.gin (DECLARED PRECISION
CHANGE; knob rule: an apply_env_bootstrap binding with a documented item block; a pre-set env value wins: "0" is
the kill switch).

  AST   apply_env_bootstrap declares the knob once (Optional[bool] = None, right after DLRM_HSTU_FP8_LAB_FWD_SPF)
        and binds it with one env-wins _bind_bool, right after item (105)'s _bind_bool and fail-closed block (which
        follow item (100)'s two _bind_bool calls) and before any other use of the knob in the function. No
        parameter is declared twice.
  GIN   exactly one "= True" binding over all gin files (yambda_5b.gin), right after item (105)'s binding, with
        a documented (120) item block of at most 80 columns, no filesystem paths, whose quoted texts exist.
  RUN   a clean child interpreter runs train_ranker's Phase-1 sequence (gin.parse_config_file(yambda_5b.gin,
        skip_unknown=True), then apply_env_bootstrap()): unset -> "1" (with item (100)'s V8SWZ=1 and RIDER=1 and
        item (105)'s SPF=1); a pre-set "0" is kept with the honoring line (kill switch); an invalid pre-set value,
        or "1" without the pv=k128 forward with mask_arith=3 (FWD_PIPE or FWD_PV "0", with item (105) killed: its
        own check runs first; or FWD_MASK_ARITH "0"), raises ValueError at bootstrap. The child imports no torch.
Ladder-chain (v2) variant: the tree carries item (105) SPF fold fwd, bound ON just before this item.
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

KNOB = "DLRM_HSTU_FP8_LUT_SILU_FWD"
RB1 = ("DLRM_HSTU_FP8_LAB_FWD_V8SWZ", "DLRM_HSTU_FP8_LAB_FWD_RIDER")
SPF = "DLRM_HSTU_FP8_LAB_FWD_SPF"  # item (105), bound right before this item
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GR = os.path.dirname(os.path.dirname(_TRAIN))  # generative_recommenders/
_ROOT = os.path.dirname(_GR)  # holds generative_recommenders/
_GIN_DIR = os.path.join(_TRAIN, "gin")
_GIN = os.path.join(_GIN_DIR, "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_HEAD = "# (120) LUT-SiLU fwd: "
_HONOR = "env bootstrap: honoring pre-set %s=0 (overrides gin binding)" % KNOB


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

    def test_one_parameter_after_spf(self) -> None:
        fn = self.fn
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        self.assertEqual(names.count(KNOB), 1)
        self.assertEqual(names.index(SPF), names.index(RB1[1]) + 1)
        self.assertEqual(names.index(KNOB), names.index(SPF) + 1)
        arg = next(a for a in fn.args.args if a.arg == KNOB)
        self.assertEqual(ast.unparse(arg.annotation) if arg.annotation else None, "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[KNOB], ast.Constant) and defaults[KNOB].value is None)

    def test_one_env_wins_bind_right_after_spf(self) -> None:
        fn = self.fn
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == KNOB]
        self.assertEqual(len(calls), 1)
        self.assertEqual(getattr(calls[0].func, "id", None), "_bind_bool")
        self.assertEqual([getattr(a, "id", None) for a in calls[0].args[1:]], [KNOB])
        # No direct os.environ[KNOB] = ... write: the env-wins rule is _bind_bool's.
        stores = [n for n in ast.walk(fn) if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
                  and isinstance(n.slice, ast.Constant) and n.slice.value == KNOB]
        self.assertEqual(stores, [])
        i_v8, i_rd, i_spf, i_lut = (_bind_stmt_index(fn.body, n) for n in RB1 + (SPF, KNOB))
        self.assertEqual((len(i_v8), len(i_rd), len(i_spf), len(i_lut)), (1, 1, 1, 1))
        # RB1's two binds, item (105)'s bind and its fail-closed block, then this item.
        self.assertEqual((i_rd[0], i_spf[0], i_lut[0]), (i_v8[0] + 1, i_rd[0] + 1, i_spf[0] + 2))
        spf_check = fn.body[i_spf[0] + 1]
        self.assertIsInstance(spf_check, ast.If)
        self.assertIn(SPF, ast.unparse(spf_check.test))
        self.assertIn("fp8_qkv_spf", ast.unparse(spf_check))
        # The fail-closed block is the next statement; every other use of the
        # knob comes after the bind.
        nxt = fn.body[i_lut[0] + 1]
        self.assertIsInstance(nxt, ast.If)
        self.assertIn(KNOB, ast.unparse(nxt.test))
        self.assertIn("fp8_lut_silu", ast.unparse(nxt))
        for i, st in enumerate(fn.body):
            uses = [n for n in ast.walk(st) if isinstance(n, ast.Constant) and n.value == KNOB]
            if uses and i != i_lut[0]:
                self.assertGreater(i, i_lut[0], ast.unparse(st)[:200])


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
        # Placed right after item (105)'s binding (which follows item (100)'s); the next item follows.
        self.assertTrue(lines[i_head - 1].startswith("# ----"), lines[i_head - 1])
        self.assertEqual(lines[i_head - 2], f"apply_env_bootstrap.{SPF} = True")
        i_rb1 = lines.index(f"apply_env_bootstrap.{RB1[1]} = True")
        self.assertEqual(lines[i_rb1 - 1], f"apply_env_bootstrap.{RB1[0]} = True")
        self.assertLess(i_rb1, i_head - 2)
        self.assertFalse([ln for ln in lines[i_rb1 + 1:i_head - 2] if ln.startswith("apply_env_bootstrap.")])
        self.assertTrue(lines[i_bind + 1].startswith("# ----"), lines[i_bind + 1])
        block = lines[i_head:i_bind]
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in lines[i_head - 1:i_bind + 1] if len(ln) > 80], [])
        text = " ".join(" ".join(ln.lstrip("#").split()) for ln in block)
        for marker in ("DECLARED PRECISION CHANGE", "Code default OFF", "bound True below", "WHAT:",
                       "E8M0 A-scale byte 127 + j", "never from data", "IMPACT:", "no TRANS",
                       "64 ds_read_u8 per trip", "-1.13 J/call",
                       "to be measured on the ship17 ladder (L21)",
                       "on the pv=k128 forward with mask_arith=3",
                       "(51) FWD_PV and (48)(d) FWD_MASK_ARITH",
                       "packed P8 byte mask stays the diagonal group's causal mask",
                       "NUMERICS: PRECISION CLASS", "1.8%", "+-1 e4m3 ulp", "unbiased",
                       "-0.004 dB mean, worst -0.015 dB", "The backward recomputes P",
                       "COLD START:", "before RUN_START", "no compile",
                       "ARMED (every rank):", f'"env bootstrap: {KNOB}=1"',
                       f'"[hstu-fwd] {KNOB}=1 armed: lut=13 (P8 = 8 KB LDS e4m3 table',
                       f'"[hstu-fwd] {KNOB} table built: score_scale=',
                       "Must be ABSENT:", f'"honoring pre-set {KNOB}"', f'"{KNOB}=1 declined"',
                       '"outside the table contract"',
                       f'"{KNOB}=1 (item (120)) needs the pv=k128 forward"', "\"use unset, '0' or '1'\"",
                       "FAILS CLOSED AT BOOTSTRAP (before RUN_START)",
                       "DLRM_HSTU_FP8_FWD_PIPE, DLRM_HSTU_FP8_FWD_MASK_ARITH or DLRM_HSTU_FP8_FWD_PV not \"1\"",
                       "(needs (26)(a), (48)(d) and (51))",
                       f"Opt out / kill switch: {KNOB}=0",
                       "a pre-set env value wins over this binding",
                       # the ladder chain: item (105) on, r = 1, its kernel under the kill switch
                       "with items (100) and (105)", "on the SPF grid, so r = 1",
                       "ISA 9f537d700151", "the item (105) kernel (ISA 30a3a2ce63b4), bit for bit"):
            self.assertIn(marker, text, marker)
        for path in ("/apps/", "/home/", "/dev/shm", "/tmp/"):
            self.assertNotIn(path, text)

    def test_quoted_log_texts_exist(self) -> None:
        mha = _read(os.path.join(_GR, "ops", "flydsl", "hstu_flydsl_mha.py"))
        lut = _read(os.path.join(_GR, "ops", "fp8_lut_silu.py"))
        boot = _read(_BOOT)
        self.assertIn(f'KNOB = "{KNOB}"', lut)
        for text in ('"[hstu-fwd] DLRM_HSTU_FP8_LUT_SILU_FWD=1 armed: lut=13 (P8 = 8 KB "',
                     '"[hstu-fwd] DLRM_HSTU_FP8_LUT_SILU_FWD=1 declined: needs the "',
                     '"[hstu-fwd] DLRM_HSTU_FP8_LUT_SILU_FWD table built: "',
                     "is outside the table contract"):
            self.assertIn(text, mha, text)
        self.assertIn('"DLRM_HSTU_FP8_LUT_SILU_FWD=1 (item (120)) needs the pv=k128 "', boot)
        self.assertIn('"forward with mask_arith=3: DLRM_HSTU_FP8_FWD_PIPE=1, "', boot)
        self.assertIn('"env bootstrap: honoring pre-set %s=%s (overrides gin binding)"', boot)
        self.assertIn('logger.info("env bootstrap: %s=%s", name, os.environ[name])', boot)
        # fp8_lut_silu is stdlib-only: its parse error, at run time.
        sys.path.insert(0, _ROOT)
        try:
            from generative_recommenders.ops import fp8_lut_silu as m
        finally:
            sys.path.remove(_ROOT)
        with self.assertRaises(ValueError) as cm:
            m.lut_mode({KNOB: "on"})
        self.assertIn("use unset, '0' or '1'", str(cm.exception))


_CHILD = textwrap.dedent(
    """
    import json, logging, os, sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    root, gin_path = sys.argv[1], sys.argv[2]
    import gin
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
    boot.apply_env_bootstrap()
    print("LUT_GIN_CHILD " + json.dumps({"env": {k: os.environ.get(k) for k in sys.argv[3:]},
                                        "torch_imported": "torch" in sys.modules}, sort_keys=True))
    """
)
_PREFIXES = ("DLRM_", "TRITON_", "HSTU_", "GR_", "PYTORCH_")


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, str, Optional[Dict[str, Any]]]:
    """A clean child interpreter: only `preset` of the knob prefixes is set when it starts."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_PREFIXES)}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_ROOT, env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, _ROOT, _GIN, *RB1, SPF, KNOB], env=env, cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("LUT_GIN_CHILD ")]
    got = json.loads(out[0][len("LUT_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], r.stderr, got


@unittest.skipIf(gin is None, "gin-config not installed")
class GinRunTest(unittest.TestCase):
    def _check(self, preset: Dict[str, str], want: Dict[str, str]) -> str:
        rc, log, err, got = _run_child(preset)
        self.assertEqual(rc, 0, log)
        assert got is not None, log
        self.assertEqual(got["env"], want, log)
        self.assertFalse(got["torch_imported"], log)
        return err

    def test_unset_binds_on_with_rb1_and_spf(self) -> None:
        err = self._check({}, {RB1[0]: "1", RB1[1]: "1", SPF: "1", KNOB: "1"})
        self.assertIn(f"env bootstrap: {KNOB}=1", err.splitlines())
        self.assertNotIn(f"honoring pre-set {KNOB}", err)

    def test_preset_zero_wins(self) -> None:
        err = self._check({KNOB: "0"}, {RB1[0]: "1", RB1[1]: "1", SPF: "1", KNOB: "0"})  # kill switch
        self.assertIn(_HONOR, err.splitlines())
        # Item (105)'s kill switch alone leaves this item on (LUT alone, r in [1, 2)).
        self._check({SPF: "0"}, {RB1[0]: "1", RB1[1]: "1", SPF: "0", KNOB: "1"})

    def _check_fails(self, preset: Dict[str, str], text: str) -> None:
        rc, log, _err, got = _run_child(preset)
        self.assertNotEqual(rc, 0, log)
        self.assertIsNone(got, log)
        self.assertIn("ValueError", log)
        self.assertIn(text, log)

    def test_fails_closed_at_bootstrap(self) -> None:
        for bad in ("true", "on", " 1", "2", "13"):
            self._check_fails({KNOB: bad}, "use unset, '0' or '1'")
        for k in ("DLRM_HSTU_FP8_FWD_PV", "DLRM_HSTU_FP8_FWD_PIPE"):
            # Item (105)'s check runs first (its block precedes this item's).
            self._check_fails({k: "0"}, f"{SPF}=1 (item (105)) needs the pv=k128 forward")
            self._check_fails({k: "0", SPF: "0"}, f"{KNOB}=1 (item (120)) needs the pv=k128 forward")
            self._check_fails({k: "0", SPF: "0", KNOB: "1"}, f"{KNOB}=1 (item (120)) needs the pv=k128 forward")
            self._check({k: "0", SPF: "0", KNOB: "0"}, {RB1[0]: "1", RB1[1]: "1", SPF: "0", KNOB: "0"})
        # lut's only causal mask is mask_arith=3's packed P8 byte mask: FWD_MASK_ARITH "0" fails closed too
        # (item (105) does not need mask_arith, so this item's check is the one that fires).
        need = f"{KNOB}=1 (item (120)) needs the pv=k128 forward with mask_arith=3"
        ma = "DLRM_HSTU_FP8_FWD_MASK_ARITH"
        self._check_fails({ma: "0"}, need)
        self._check_fails({ma: "0", KNOB: "1"}, need)
        self._check({ma: "0", KNOB: "0"}, {RB1[0]: "1", RB1[1]: "1", SPF: "1", KNOB: "0"})

    def test_kill_switch_texts_name_item_120(self) -> None:
        # Each item whose kill switch makes (120) fail closed says so in its opt-out text.
        text = " ".join(" ".join(ln.lstrip("#").split()) for ln in _read(_GIN).splitlines() if ln.startswith("#"))
        for opt in ("DLRM_HSTU_FP8_DELTA_FWD_BM16=0. With item (120) bound, FWD_PIPE=0 also needs " + KNOB + "=0",
                    "DLRM_HSTU_FP8_FWD_MASK_ARITH=0 (the f32 select of (26)(a)); with item (120) bound, also "
                    + KNOB + "=0",
                    "DLRM_HSTU_FP8_FWD_PV=0 (bf16 P, the (26)(a)/(48)(d) forward); with item (120) bound, also "
                    + KNOB + "=0"):
            self.assertIn(opt, text, opt)


if __name__ == "__main__":
    unittest.main()
