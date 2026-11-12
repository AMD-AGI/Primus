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
"""Ship17 row (100) RB1: the fwd_kernel_0 riders (DLRM_HSTU_FP8_LAB_FWD_V8SWZ, DLRM_HSTU_FP8_LAB_FWD_RIDER) are
bound ON in yambda_5b.gin (knob rule: an apply_env_bootstrap binding with a documented item block; a pre-set env
value wins and "0" is the kill switch).

  AST   apply_env_bootstrap declares each knob once (Optional[bool] = None) and binds it with one env-wins _bind_bool.
  GIN   exactly one "= True" binding per knob, no name bound twice, a documented (100) item block.
  RUN   a clean child interpreter runs train_ranker's Phase-1 sequence (gin.parse_config_file(yambda_5b.gin,
        skip_unknown=True), then apply_env_bootstrap()): unset -> "1"; a pre-set "0" is kept (kill switch); the
        two knobs are independent.
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

KNOBS = ("DLRM_HSTU_FP8_LAB_FWD_V8SWZ", "DLRM_HSTU_FP8_LAB_FWD_RIDER")
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(_TRAIN, "..", "..", ".."))  # holds generative_recommenders/
_GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_HEAD = "# (100) RB1: "


def _gin_lines() -> List[str]:
    with open(_GIN) as f:
        return f.read().splitlines()


class BootstrapAstTest(unittest.TestCase):
    def test_one_parameter_one_env_wins_bind_per_knob(self) -> None:
        with open(_BOOT) as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        args = fn.args.args + fn.args.kwonlyargs
        names = [a.arg for a in args]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        for knob in KNOBS:
            self.assertEqual(names.count(knob), 1, knob)
            arg = next(a for a in fn.args.args if a.arg == knob)
            self.assertEqual(ast.unparse(arg.annotation) if arg.annotation else None, "Optional[bool]")
            self.assertTrue(isinstance(defaults[knob], ast.Constant) and defaults[knob].value is None, knob)
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                     and isinstance(n.args[0], ast.Constant) and n.args[0].value == knob]
            self.assertEqual(len(calls), 1, knob)
            self.assertEqual(getattr(calls[0].func, "id", None), "_bind_bool", knob)
            self.assertEqual([getattr(a, "id", None) for a in calls[0].args[1:]], [knob])


class GinBindingTest(unittest.TestCase):
    def test_exactly_one_true_binding_per_knob_and_no_repeats(self) -> None:
        lines = _gin_lines()
        for knob in KNOBS:
            pat = re.compile(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % knob)
            self.assertEqual([m.group(1) for m in map(pat.match, lines) if m], ["True"], knob)
        names = [m.group(1) for m in (re.match(r"^\s*apply_env_bootstrap\.(\w+)\s*=", ln) for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])

    def test_item_block_documents_the_row(self) -> None:
        lines = _gin_lines()
        i_bind = next(i for i, ln in enumerate(lines) if ln.startswith(f"apply_env_bootstrap.{KNOBS[0]} "))
        i_head = max(i for i in range(i_bind) if lines[i].startswith(_HEAD))
        block = lines[i_head:i_bind]
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        self.assertTrue(lines[i_bind + 1].startswith(f"apply_env_bootstrap.{KNOBS[1]} "))
        text = " ".join(ln[1:].strip() for ln in block)
        for marker in ("WHAT:", "NUMERICS: bit-identical", "ARMED", "rider=3", "v8_swz=1",
                       f"Opt out (kill switch): {KNOBS[0]}=0", f"{KNOBS[1]}=0", "9efc775fbca5"):
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
    print("RB1_GIN_CHILD " + json.dumps({k: os.environ.get(k) for k in sys.argv[3:]}, sort_keys=True))
    """
)
_PREFIXES = ("DLRM_", "TRITON_", "HSTU_", "GR_", "PYTORCH_TUNABLEOP")


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    """A clean child interpreter: only `preset` of the knob prefixes is set when it starts."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_PREFIXES)}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_ROOT, env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, _ROOT, _GIN, *KNOBS], env=env, cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("RB1_GIN_CHILD ")]
    got = json.loads(out[0][len("RB1_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


@unittest.skipIf(gin is None, "gin-config not installed")
class GinRunTest(unittest.TestCase):
    def _check(self, preset: Dict[str, str], want: Dict[str, str]) -> None:
        rc, log, got = _run_child(preset)
        self.assertEqual(rc, 0, log)
        self.assertEqual(got, want, log)

    def test_unset_binds_on(self) -> None:
        self._check({}, {KNOBS[0]: "1", KNOBS[1]: "1"})

    def test_kill_switches_are_honoured_and_independent(self) -> None:
        self._check({KNOBS[0]: "0"}, {KNOBS[0]: "0", KNOBS[1]: "1"})
        self._check({KNOBS[1]: "0"}, {KNOBS[0]: "1", KNOBS[1]: "0"})
        self._check({KNOBS[0]: "0", KNOBS[1]: "0"}, {KNOBS[0]: "0", KNOBS[1]: "0"})


if __name__ == "__main__":
    unittest.main()
