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
"""Item (104) lives in gin (yambda_5b.gin; knob rule: every feature switch is an apply_env_bootstrap binding with a
documented item block, env is only the kill switch). Two knobs: DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6 (bool, bound False:
a declared precision change ships ON only with an end-to-end gain, through a separate flip row) and its static site
set DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6_SITES (a value, bound "top").

  AST   apply_env_bootstrap declares each parameter once (Optional[bool] = None / Optional[str] = None) and binds it
        with one env-wins call (_bind_bool / _bind_value), after item (89)'s DIET bind; no parameter is repeated.
  GIN   exactly one binding each ("False", '"top"'), after (94)'s or (111)'s; no binding repeated; the item block
        documents both knobs, the precision change, the site set (with the E-RM rule), fallback, build, armed lines
        (all at the prebuild, before RUN_START; none at an MX6 record) and the absent markers with the loader's real
        strings; at most 80 columns; no internal paths or job ids.
  PY    ops/hip/l11_do8.py's knob readers and site predicate, run without torch (AST subset): only "1" arms the knob,
        read once; the site set "top" / "bottom" / "all" (default "top"), anything else None (INERT); the per-layer
        predicate (top = layers >= 1, bottom = 0, all; untagged never).
  RUN   clean child interpreters run train_ranker's Phase-1 sequence (gin.parse_config_file(yambda_5b.gin,
        skip_unknown=True), then apply_env_bootstrap()): unset -> "0" and "top"; pre-set values win ("1", "0",
        "all"); torch never imported. Skipped where gin is not importable.
CPU only; no torch import."""

import ast
import importlib.util
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

_HAS_GIN = importlib.util.find_spec("gin") is not None

KNOB = "DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6"
SITES = KNOB + "_SITES"
_BOUND = "True"  # the code row binds MX6 OFF (the flip row: True)
_BOUND_SITES = '"top"'
_RUN_DEFAULT = "1"  # what the binding exports when nothing is pre-set (the flip row: "1")
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GR = os.path.abspath(os.path.join(_TRAIN, "..", ".."))  # generative_recommenders/
_ROOT = os.path.dirname(_GR)
_GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_PY = os.path.join(_GR, "ops", "hip", "l11_do8.py")
_THL = os.path.join(_GR, "ops", "triton", "triton_hstu_linear.py")
_HEAD = "# (104) "
_PREV = "apply_env_bootstrap.DLRM_HSTU_FP8_LNMD_MASK_NIB "
_PREV_111 = "apply_env_bootstrap.DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3 "  # item (111), where that row is in the tree


def _gin_lines() -> List[str]:
    with open(_GIN) as f:
        return f.read().splitlines()


def _binding_index(lines: List[str], knob: str) -> int:
    return next(i for i, ln in enumerate(lines) if ln.startswith(f"apply_env_bootstrap.{knob} "))


def _item_block(lines: List[str]) -> List[str]:
    i_bind = _binding_index(lines, KNOB)
    i_head = max(i for i in range(i_bind) if lines[i].startswith(_HEAD))
    return lines[i_head:i_bind]


def _flat(block: List[str]) -> str:
    return " ".join(ln[1:].strip() for ln in block)


def _bootstrap_fn() -> ast.FunctionDef:
    with open(_BOOT) as f:
        tree = ast.parse(f.read())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")


def _py_constants() -> Dict[str, Any]:
    """Module-level string / tuple constants of ops/hip/l11_do8.py (literal_eval of their AST; no import)."""
    with open(_PY) as f:
        tree = ast.parse(f.read())
    out: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass  # not a literal (an expression of other names)
    return out


class BootstrapAstTest(unittest.TestCase):
    def test_one_parameter_and_one_env_wins_bind_each(self) -> None:
        fn = _bootstrap_fn()
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        lines = {}
        for knob, ann, binder in ((KNOB, "Optional[bool]", "_bind_bool"), (SITES, "Optional[str]", "_bind_value")):
            self.assertEqual(names.count(knob), 1, knob)
            arg = next(a for a in fn.args.args if a.arg == knob)
            self.assertEqual(ast.unparse(arg.annotation) if arg.annotation else None, ann, knob)
            self.assertTrue(isinstance(defaults[knob], ast.Constant) and defaults[knob].value is None, knob)
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                     and isinstance(n.args[0], ast.Constant) and n.args[0].value == knob]
            self.assertEqual(len(calls), 1, knob)
            (call,) = calls
            self.assertEqual(getattr(call.func, "id", None), binder, knob)  # env wins
            self.assertEqual([getattr(a, "id", None) for a in call.args[1:]], [knob], knob)
            lines[knob] = call.lineno
        diet = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == "DLRM_HSTU_FP8_L11_DO8_DIET"]
        self.assertEqual(len(diet), 1)
        self.assertLess(diet[0].lineno, lines[KNOB])  # after the DO8 / DIET binds (read at the same prebuild)
        self.assertLess(lines[KNOB], lines[SITES])


class GinBindingTest(unittest.TestCase):
    def test_exactly_one_binding_each(self) -> None:
        lines = _gin_lines()
        for knob, want in ((KNOB, _BOUND), (SITES, _BOUND_SITES)):
            pat = re.compile(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % knob)
            self.assertEqual([m.group(1) for m in map(pat.match, lines) if m], [want], knob)
        names = [m.group(1) for m in (re.match(r"^\s*([\w./]+)\s*=", ln) for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])

    def test_item_follows_nib_and_the_bindings_close_it(self) -> None:
        lines = _gin_lines()
        i_bind, i_sites = _binding_index(lines, KNOB), _binding_index(lines, SITES)
        self.assertEqual(i_sites, i_bind + 1)
        i_prev = next(i for i, ln in enumerate(lines) if ln.startswith(_PREV))
        i_head = i_bind - len(_item_block(lines))
        self.assertLess(i_prev, i_head)
        # only comments between, apart from item (111)'s binding where that row is in the tree (ship17 cand_v2b:
        # its block is bound right after (94)'s and this item's block follows it)
        between = [ln for ln in lines[i_prev + 1:i_bind] if not ln.startswith("#")]
        self.assertTrue(len(between) <= 1 and all(ln.startswith(_PREV_111) for ln in between), between)

    def test_item_block_documents_both_knobs(self) -> None:
        block = _item_block(_gin_lines())
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = _flat(block)
        for marker in ("DECLARED PRECISION CHANGE", "end-to-end time-to-train gain", "WHAT:", "WHY:", "NUMERICS:",
                       "E2M3 x E2M3 with E8M0 per-32 block scales", "SITES:", SITES, "top", "bottom", "all",
                       "STATIC per-layer set", "FALLBACK:", "RESCUED", "captured scale", "BUILD", "-DL11DO8_MX6=1",
                       "ARMED", "Must be ABSENT:", "STATE:", "INERT", "Code default OFF", "pre-set env var wins",
                       "Only the value 1 arms it", f"Opt out / kill switch: env {KNOB}=0",
                       "output_projection_dgrad_operands_mx6", "output_projection_g8_rescue",
                       "output_projection_mx6_encode",
                       # the E-RM rule (l11_do8._mx6_inert_reason; the seam's E-RM decline)
                       "MX6 never takes an E-RM site", "item (97) requested",
                       "a set holding layer 0 (bottom, all) leaves MX6 INERT too (fail closed)",
                       "E-RM's forward tagged is declined",
                       "With sites=all and item (97) off",
                       # the seam's E-RM decline line is a must-be-ABSENT marker of an MX6 arm
                       '"... MX6 NOT armed: E-RM owns this call"'):
            self.assertIn(marker, text)
        absent = text[text.index("Must be ABSENT:"):text.index("Knob off:")]
        self.assertIn("MX6 NOT armed: E-RM owns this call", absent)
        # re-cut v2: every armed line prints at the prebuild, before RUN_START (the ACTIVE line right after the
        # prebuild line); an MX6 record prints none
        armed = text[text.index("ARMED ("):text.index("Must be ABSENT:")]
        self.assertNotIn("at the first MX6 record", armed)
        for marker in ("all at the prebuild, before RUN_START", "right after it", '"sites=<set> poison=<0|1>"',
                       "An MX6 record prints no line"):
            self.assertIn(marker, armed)
        # no internal paths, job ids or session references in the gin comments
        for bad in (r"/apps/", r"/home/", r"probes/", r"\bjob \d", r"\biter\d", r"arbor_sessions"):
            self.assertIsNone(re.search(bad, text), bad)

    def test_documented_strings_are_the_loaders(self) -> None:
        text = _flat(_item_block(_gin_lines()))
        c = _py_constants()
        self.assertEqual((c["MX6_KNOB"], c["MX6_SITES"], c["MX6_SITES_DEFAULT"]), (KNOB, ("top", "bottom", "all"),
                                                                                 "top"))
        self.assertEqual(c["_MX6_DEFINE"], "-DL11DO8_MX6=1")
        family = c["FAMILY"] + " MX6"
        self.assertEqual(family, "L11 DO8 dO8/E8 emission MX6")
        for line in (f"armed OK: {family} prebuild before run_start", f"armed OK: {family} ACTIVE mode=e5 only",
                     f"{family} DISABLED", "[l11-do8] mx6 counters: decided= launched= published= rescued= released= "
                     "latched="):
            self.assertIn(line, text)
        self.assertIn(c["KERNEL_NAME"] + "_mx6<1, 2, 0, 0, 1, 2, 1, 0, 1, 2, 2, 0, 2, 0>", text)
        self.assertEqual(c["MX6_PRODUCER_API"], ("output_projection_dgrad_operands_mx6", "output_projection_g8_rescue",
                                                 "output_projection_mx6_encode"))
        self.assertIn(f"MX6_NIB_GATED switch ({c['MX6_NIB_GATED']})", text)  # the block states the loader's value
        with open(_PY) as f:
            src = f.read()
        for code in ('f"[l11-do8] mx6 counters: decided={c[\'decided\']} launched={c[\'launched\']} published=',
                     'f"armed OK: {FAMILY_MX6} prebuild before run_start: built+loaded+self-check OK',
                     'f"armed OK: {FAMILY_MX6} ACTIVE mode=e5 only ({MX6_KNOB}, item {ITEM_MX6}',
                     'f"{FAMILY_MX6} DISABLED: {reason} -> FP8 dual cast + {ITEM_ONLY} for this process"',
                     'f"{FAMILY_MX6} INERT: {reason}; nothing built',
                     'f"{FAMILY_MX6} rescue: ', 'f"{FAMILY_MX6} release: ',
                     'f"{FAMILY_MX6} NOT armed: {reason} -> FP8 dual cast + {ITEM_ONLY} for this call"'):
            self.assertIn(code, src)
        with open(_THL) as f:  # the seam's E-RM decline reason (the must-be-ABSENT marker above)
            self.assertIn("\"E-RM owns this call (the forward's E-RM site tag)", f.read())


def _load_py_subset() -> Dict[str, Any]:
    """The MX6 knob readers and site predicate of ops/hip/l11_do8.py, executed without torch (the module imports
    it): only the module constants and functions they use."""
    with open(_PY) as f:
        tree = ast.parse(f.read())
    want_assign = {"MX6_KNOB", "MX6_SITES_KNOB", "MX6_SITES", "MX6_SITES_DEFAULT", "_mx6_requested", "_mx6_override",
                   "_mx6_sites_read", "_mx6_sites_raw", "_mx6_sites_override"}
    want_fn = {"mx6_requested", "set_mx6_for_testing", "_mx6_sites_value", "set_mx6_sites_for_testing", "mx6_sites",
               "mx6_site_selected"}
    body: List[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in want_assign:
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in want_fn:
            body.append(node)
    found = {getattr(getattr(n, "target", None), "id", None) for n in body} | {getattr(n, "name", None) for n in body}
    missing = (want_assign | want_fn) - found
    if missing:
        raise AssertionError(f"l11_do8.py lacks {sorted(missing)}")
    ns: Dict[str, Any] = {"os": os}
    ns.update({k: getattr(typing, k) for k in ("Any", "Dict", "List", "Optional", "Tuple")})
    exec(compile(ast.Module(body=body, type_ignores=[]), _PY, "exec"), ns)  # noqa: S102 -- our own source
    return ns


class PyKnobTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ns = _load_py_subset()

    def _env(self, **kv: Optional[str]) -> "mock._patch[Dict[str, str]]":
        env = {k: v for k, v in os.environ.items() if k not in (KNOB, SITES)}
        env.update({k: v for k, v in kv.items() if v is not None})
        return mock.patch.dict(os.environ, env, clear=True)

    def test_only_one_arms_and_is_read_once(self) -> None:
        for value, want in (("1", True), (" 1 ", True), ("0", False), ("", False), ("true", False), ("True", False),
                            ("2", False), (None, False)):
            with self._env(**{KNOB: value}):
                self.ns["set_mx6_for_testing"](None)
                self.assertEqual(self.ns["mx6_requested"](), want, value)
        with self._env(**{KNOB: "1"}):
            self.ns["set_mx6_for_testing"](None)
            self.assertTrue(self.ns["mx6_requested"]())
            os.environ[KNOB] = "0"
            self.assertTrue(self.ns["mx6_requested"]())  # read once

    def test_site_set_values(self) -> None:
        for value, want in ((None, "top"), ("top", "top"), (" TOP ", "top"), ("bottom", "bottom"), ("all", "all"),
                            ("", None), ("0", None), ("1", None), ("top,bottom", None), ("layer1", None)):
            with self._env(**{SITES: value}):
                self.ns["set_mx6_sites_for_testing"](None)
                self.assertEqual(self.ns["mx6_sites"](), want, value)
        with self._env(**{SITES: "all"}):
            self.ns["set_mx6_sites_for_testing"](None)
            self.assertEqual(self.ns["mx6_sites"](), "all")
            os.environ[SITES] = "bottom"
            self.assertEqual(self.ns["mx6_sites"](), "all")  # read once

    def test_static_site_predicate(self) -> None:
        sel = self.ns["mx6_site_selected"]
        for sites, want in (("top", (False, False, True, True)), ("bottom", (False, True, False, False)),
                            ("all", (False, True, True, True)), ("bogus", (False, False, False, False))):
            self.ns["set_mx6_sites_for_testing"](sites)
            self.assertEqual(tuple(sel(layer) for layer in (-1, 0, 1, 2)), want, sites)
        self.ns["set_mx6_sites_for_testing"](None)


_CHILD = textwrap.dedent(
    """
    import json, os, sys
    root, gin_path = sys.argv[1], sys.argv[2]
    import gin
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
    boot.apply_env_bootstrap()
    print("MX6_GIN_CHILD " + json.dumps({
        "env": {k: v for k, v in os.environ.items() if k.startswith(("DLRM_", "TRITON_", "HSTU_", "GR_"))},
        "torch_imported": "torch" in sys.modules,
    }, sort_keys=True))
    """
)
_PREFIXES = ("DLRM_", "TRITON_", "HSTU_", "GR_", "PYTORCH_TUNABLEOP")


def _child_env(preset: Dict[str, str]) -> Dict[str, str]:
    """A clean child environment: only `preset` of the knob prefixes set; the parent's import path (so a gin found
    by the parent is found by the child)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_PREFIXES)}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join([_ROOT] + [p for p in sys.path if p])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, _ROOT, _GIN], env=_child_env(preset), cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("MX6_GIN_CHILD ")]
    got = json.loads(out[0][len("MX6_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


@unittest.skipUnless(_HAS_GIN, "gin-config not installed")
class GinRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        probe = subprocess.run([sys.executable, "-B", "-c", "import gin"], env=_child_env({}), cwd=_ROOT,
                               capture_output=True, text=True, timeout=120)
        if probe.returncode != 0:
            raise unittest.SkipTest("gin-config not importable in a child interpreter")

    def _check(self, preset: Dict[str, str], want: str, want_sites: str) -> None:
        rc, log, got = _run_child(preset)
        self.assertEqual(rc, 0, log)
        assert got is not None, log
        self.assertEqual((got["env"].get(KNOB), got["env"].get(SITES)), (want, want_sites), preset)
        self.assertFalse(got["torch_imported"])

    def test_gin_default(self) -> None:
        self._check({}, _RUN_DEFAULT, "top")

    def test_preset_values_win(self) -> None:
        self._check({KNOB: "1"}, "1", "top")  # the A/B arm
        self._check({KNOB: "0"}, "0", "top")  # the kill switch
        self._check({SITES: "all"}, _RUN_DEFAULT, "all")


if __name__ == "__main__":
    unittest.main()
