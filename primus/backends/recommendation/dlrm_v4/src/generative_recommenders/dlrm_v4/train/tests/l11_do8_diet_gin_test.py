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
"""Item (89) DO8 DIET lives in gin (yambda_5b.gin, DLRM_HSTU_FP8_L11_DO8_DIET; knob rule: every feature switch is an
apply_env_bootstrap binding with a documented item block, env is only the kill switch).

  AST   apply_env_bootstrap declares the parameter once (Optional[bool] = None) and binds it with one env-wins
        _bind_bool call.
  GIN   exactly one binding, "= True", right after item (88)'s; the item block documents the knob and its kill switch.
  RUN   clean child interpreters run train_ranker's Phase-1 sequence (gin.parse_config_file(yambda_5b.gin,
        skip_unknown=True), then apply_env_bootstrap()): unset -> "1"; pre-set "0" -> "0" (kill switch honored).
  CU    ops/hip/l11_do8.cu's "// (89) DIET" guards resolve, for L11DO8_DIET = 1, to the ship15 diet source byte for byte
        (sha256 pin) and, for 0, to ship14's (c68395d) kernel body plus item (88)'s DO8-ONLY lines (sha256 pin); the
        .cu default is 1.
  PY    ops/hip/l11_do8.py's knob reader and build flags (the functions themselves, run without torch): only "1" arms
        it, read once; off adds -DL11DO8_DIET=0 to the 'plus' and 'only' flags; on adds nothing.
CPU only; no torch import. The ON vs OFF bitwise GPU check is ops/tests/l11_do8_test.py DO8DietGpuTest."""

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import typing
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

try:
    import gin  # noqa: F401
except ImportError:  # pragma: no cover
    gin = None  # type: ignore[assignment]

KNOB = "DLRM_HSTU_FP8_L11_DO8_DIET"
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GR = os.path.abspath(os.path.join(_TRAIN, "..", ".."))  # generative_recommenders/
_ROOT = os.path.dirname(_GR)
_GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_CU = os.path.join(_GR, "ops", "hip", "l11_do8.cu")
_PY = os.path.join(_GR, "ops", "hip", "l11_do8.py")
_HEAD = "# (89) DO8 DIET "
_PREV = "apply_env_bootstrap.DLRM_HSTU_FP8_L11_DO8_ONLY = True"
# sha256 of the resolved l11_do8.cu text (see _resolve). ON = the ship15 / ship15.1 diet source as shipped before this
# switch existed; OFF = ship14 c68395d's l11_do8.cu + item (88)'s DO8-ONLY lines only. Any edit of l11_do8.cu changes
# these: re-derive them, and re-check OFF against c68395d (git show c68395d:recommendation/.../l11_do8.cu).
# Re-derived for the NIB v2 row (item (94), re-cut on 734246a): ship15.1's pins were ON 48c2667d / OFF 36b32446; the
# row's 64 changed lines all sit outside the 14 (89) guards and are exactly the ON delta and the OFF delta, so ON is
# s15p1b + NIB v2 (the measured NIB source) and OFF is c68395d + item (88)'s DO8-ONLY lines + the NIB lines.
# Re-derived for the r16 MX6 kernel row (symbol-preserving; every MX line outside the 14 (89) guards): NIB-row pins
# ON f656c9e5 / OFF 491153c6 -> ON = s15p1b + NIB v2 + MX6 (= r16 iter57 symfix_inc of iter56 kern 6d41b39a),
# OFF = c68395d + item (88)'s DO8-ONLY lines + the NIB lines + the MX6 lines.
# Re-derived for ROCm 10.2 T14b (Triton 3.9 in-lane row-sum order, -DL11_TLSUM_TREE=1 from l11_outproj_lnmd.tlsum_tree()):
# MX6-row pins ON cdc58733 / OFF 5ae9449e; the 34 added lines (the L11_TLSUM_TREE note + default define and its three
# #if blocks) all sit outside the 14 (89) guards, are the same in ON and OFF, no line is removed, and with the default
# L11_TLSUM_TREE 0 they preprocess to the previous source.
_SHA_ON = "bd7356bfa8d4ff87642d748ef8e9a937a06359c40fcf58b1b0a773c7998d2276"
_SHA_OFF = "4bb70e730c7c5d7ea843cb07448b343d6279886e58e35aa182cab35b34269635"
_N_GUARDS = 14
_GUARD = re.compile(r"#(if !?L11DO8_DIET|else|endif)  // \(89\) DIET")


def _gin_lines() -> List[str]:
    with open(_GIN) as f:
        return f.read().splitlines()


def _binding_index(lines: List[str]) -> int:
    return next(i for i, ln in enumerate(lines) if ln.startswith(f"apply_env_bootstrap.{KNOB} "))


def _item_block(lines: List[str]) -> List[str]:
    i_bind = _binding_index(lines)
    i_head = max(i for i in range(i_bind) if lines[i].startswith(_HEAD))
    return lines[i_head:i_bind]


def _flat(block: List[str]) -> str:
    return " ".join(ln[1:].strip() for ln in block)


def _resolve(text: str, on: bool) -> Tuple[str, int]:
    """l11_do8.cu with every (89) guard resolved for L11DO8_DIET = on and the switch's define region dropped (what the
    preprocessor keeps, comments included). Returns (text, number of guarded regions)."""
    out: List[str] = []
    keep: List[bool] = []
    region, n = False, 0
    for ln in text.split("\n"):
        if ln.startswith("// >>> item (89) DIET switch"):
            region = True
            continue
        if region:
            region = not ln.startswith("// <<< item (89) DIET switch")
            continue
        m = _GUARD.match(ln)
        if m:
            if m.group(1).startswith("if"):
                if keep:
                    raise AssertionError("nested (89) guard")
                keep.append(on if m.group(1) == "if L11DO8_DIET" else not on)
                n += 1
            elif m.group(1) == "else":
                keep[-1] = not keep[-1]
            else:
                keep.pop()
            continue
        if "L11DO8_DIET" in ln and not ln.startswith("//"):
            raise AssertionError(f"unrecognized L11DO8_DIET line: {ln!r}")
        if not keep or keep[-1]:
            out.append(ln)
    if keep or region:
        raise AssertionError("unterminated (89) guard / region")
    return "\n".join(out), n


class BootstrapAstTest(unittest.TestCase):
    def test_one_parameter_one_env_wins_bind(self) -> None:
        with open(_BOOT) as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(names.count(KNOB), 1)
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        arg = next(a for a in fn.args.args if a.arg == KNOB)
        self.assertEqual(ast.unparse(arg.annotation) if arg.annotation else None, "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[KNOB], ast.Constant) and defaults[KNOB].value is None)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == KNOB]
        self.assertEqual(len(calls), 1)
        (call,) = calls
        self.assertEqual(getattr(call.func, "id", None), "_bind_bool")  # env wins; True -> "1"
        self.assertEqual([getattr(a, "id", None) for a in call.args[1:]], [KNOB])


class GinBindingTest(unittest.TestCase):
    def test_exactly_one_binding_true_after_item_88(self) -> None:
        lines = _gin_lines()
        pat = re.compile(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % KNOB)
        self.assertEqual([m.group(1) for m in map(pat.match, lines) if m], ["True"])
        names = [m.group(1) for m in (re.match(r"^\s*apply_env_bootstrap\.(\w+)\s*=", ln) for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        i_bind = _binding_index(lines)
        i_prev = lines.index(_PREV)
        self.assertLess(i_prev, i_bind)
        self.assertTrue(all(ln.startswith("#") for ln in lines[i_prev + 1:i_bind]))  # only (89)'s comment between

    def test_item_block_documents_the_knob(self) -> None:
        block = _item_block(_gin_lines())
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = _flat(block)
        for marker in ("ops/hip/l11_do8.cu", "L11DO8_DIET", "-DL11DO8_DIET=0", "WHAT:", "NUMERICS: bit-identical",
                       "BUILD:", "ARMED:", "Code default OFF", "pre-set env var wins", "Only the value 1 arms it",
                       f"Kill switch: env {KNOB}=0", "c68395d", "'only'"):
            self.assertIn(marker, text)


_CHILD = textwrap.dedent(
    """
    import json, os, sys
    root, gin_path = sys.argv[1], sys.argv[2]
    import gin
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
    boot.apply_env_bootstrap()
    print("I89_GIN_CHILD " + json.dumps({
        "env": {k: v for k, v in os.environ.items() if k.startswith(("DLRM_", "TRITON_", "HSTU_", "GR_"))},
        "torch_imported": "torch" in sys.modules,
    }, sort_keys=True))
    """
)
_PREFIXES = ("DLRM_", "TRITON_", "HSTU_", "GR_", "PYTORCH_TUNABLEOP")


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_PREFIXES)}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_ROOT, env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, _ROOT, _GIN], env=env, cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("I89_GIN_CHILD ")]
    got = json.loads(out[0][len("I89_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


@unittest.skipIf(gin is None, "gin-config not installed")
class GinRunTest(unittest.TestCase):
    def _check(self, preset: Dict[str, str], want: str) -> None:
        rc, log, got = _run_child(preset)
        self.assertEqual(rc, 0, log)
        assert got is not None, log
        self.assertEqual(got["env"].get(KNOB), want, preset)
        self.assertFalse(got["torch_imported"])

    def test_gin_default_is_on(self) -> None:
        self._check({}, "1")

    def test_preset_zero_is_the_kill_switch(self) -> None:
        self._check({KNOB: "0"}, "0")


class CuSwitchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(_CU) as f:
            cls.text = f.read()

    def test_default_define_is_on(self) -> None:
        self.assertEqual(self.text.count("#ifndef L11DO8_DIET\n#define L11DO8_DIET 1\n#endif\n"), 1)
        i_def = self.text.index("#define L11DO8_DIET 1")
        self.assertLess(i_def, self.text.index("#if L11DO8_DIET  // (89) DIET"))
        self.assertLess(i_def, self.text.index("#if !L11DO8_DIET  // (89) DIET"))

    def test_on_is_the_shipped_diet_source(self) -> None:
        on, n = _resolve(self.text, True)
        self.assertEqual(n, _N_GUARDS)
        self.assertEqual(hashlib.sha256(on.encode()).hexdigest(), _SHA_ON)
        for marker in ("pkmax_u16", "int nit = 0;", "O2SL", "combine v6x", "e += (mant <= 0x600000) ? 1 : 0;"):
            self.assertTrue(marker in on, marker)

    def test_off_is_ship14_c68395d_plus_do8_only(self) -> None:
        off, n = _resolve(self.text, False)
        self.assertEqual(n, _N_GUARDS)
        self.assertEqual(hashlib.sha256(off.encode()).hexdigest(), _SHA_OFF)
        for marker in ("pkmax_u16", "nitc", "O2SL", "iter52", "DIET"):
            self.assertFalse(marker in off, marker)
        for line in ("    int e = (mant <= 0x600000) ? 135 - E : 134 - E;\n",  # ship14's grp_exp / row tests
                     "    e = (E == 0) ? 60 : e;        // fp32 subnormal amax\n",
                     "              if (WLIST == 0 || it * 8 + w < rv) {  // DO8 WLIST:",
                     "                if (RET == 2 || r < rv) {  // RET 2: straight-line"):
            self.assertTrue(line in off, line)
        self.assertTrue("#if L11DO8_ONLY" in off)  # item (88) is not part of this switch


def _load_py_subset() -> Dict[str, Any]:
    """The knob reader and the flag builder of ops/hip/l11_do8.py, executed without torch (the module imports it):
    only the module constants and functions they use, with sched() as the default scheduler."""
    with open(_PY) as f:
        tree = ast.parse(f.read())
    want_assign = {"KNOB", "_ARCH", "_BASE_FLAGS", "_SCHED_FLAGS", "_ONLY_DEFINE", "DIET_KNOB", "_DIET_OFF_DEFINE",
                   "_NIB_DEFINE", "_diet", "_diet_override"}
    want_fn = {"diet_requested", "set_diet_for_testing", "_diet_tail", "_flags"}
    body: List[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in want_assign:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in want_assign
                                                    for t in node.targets):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in want_fn:
            body.append(node)
    found = {getattr(getattr(n, "target", None), "id", None) for n in body} | {
        getattr(n, "name", None) for n in body}
    missing = (want_assign | want_fn) - found
    if missing:
        raise AssertionError(f"l11_do8.py lacks {sorted(missing)}")
    # _nib_src: the NIB row's 'only' define needs the NIB knob and a source with its switch; b"x" has neither
    # (ops/tests/l11_do8_nib_test.py tests the real function)
    ns: Dict[str, Any] = {"os": os, "hashlib": hashlib, "sched": lambda: "default", "_nib_src": lambda src: False}
    ns.update({k: getattr(typing, k) for k in ("Any", "Dict", "List", "Optional", "Tuple")})
    exec(compile(ast.Module(body=body, type_ignores=[]), _PY, "exec"), ns)  # noqa: S102 -- our own source
    return ns


class PyPlumbingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ns = _load_py_subset()

    def _requested(self, value: Optional[str]) -> bool:
        env = {k: v for k, v in os.environ.items() if k != KNOB}
        if value is not None:
            env[KNOB] = value
        with mock.patch.dict(os.environ, env, clear=True):
            self.ns["set_diet_for_testing"](None)
            return self.ns["diet_requested"]()

    def test_names(self) -> None:
        self.assertEqual(self.ns["DIET_KNOB"], KNOB)
        self.assertEqual(self.ns["_DIET_OFF_DEFINE"], "-DL11DO8_DIET=0")
        self.assertEqual(self.ns["_ONLY_DEFINE"], "-DL11DO8_ONLY=1")

    def test_only_one_arms_and_absent_is_off(self) -> None:
        for value, want in (("1", True), (" 1 ", True), ("0", False), ("", False), ("true", False),
                            ("True", False), ("2", False), (None, False)):
            self.assertEqual(self._requested(value), want, value)

    def test_read_once(self) -> None:
        self.assertTrue(self._requested("1"))
        with mock.patch.dict(os.environ, {KNOB: "0"}):
            self.assertTrue(self.ns["diet_requested"]())

    def test_flags_follow_the_knob(self) -> None:
        flags, tail = self.ns["_flags"], self.ns["_diet_tail"]
        self.assertTrue(self._requested("1"))
        on, on_only = flags(b"x"), flags(b"x", True)
        self.assertFalse([f for f in on + on_only if "L11DO8_DIET" in f])  # the flags without the switch
        self.assertEqual(on_only, on + ["-DL11DO8_ONLY=1"])
        self.assertEqual(tail(), "")
        self.assertFalse(self._requested("0"))
        off, off_only = flags(b"x"), flags(b"x", True)
        self.assertEqual(off, on + ["-DL11DO8_DIET=0"])
        self.assertEqual(off_only, on + ["-DL11DO8_DIET=0", "-DL11DO8_ONLY=1"])
        self.assertIn(f"diet OFF ({KNOB} != 1: -DL11DO8_DIET=0", tail())
        self.assertIn('-DL11DO8_SRC_SHA256="' + hashlib.sha256(b"x").hexdigest() + '"', off)


if __name__ == "__main__":
    unittest.main()
