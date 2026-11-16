#!/usr/bin/env python3
# pyre-strict
"""Item (109): DL_PRESTART_RANK_TIMELINE lives in gin (yambda_5b.gin), OFF by default.

  AST   apply_env_bootstrap declares the parameter once (Optional[bool] = None) and binds it with one env-wins
        _bind_bool call; the name is the one dl_prestart.report() reads.
  GIN   exactly one binding, "= False", right after its "# (109) " item block (comment lines <= 80 columns that
        document the knob, its default, its kill switch and the arming line).
  RUN   a clean child interpreter runs the bootstrap with the gin file's bound value: unset -> "0" (off);
        pre-set "1" -> "1" (on); pre-set "0" -> "0"; dl_prestart.rank_timeline_enabled() follows. Uses the real
        gin parse (train_ranker Phase 1) when gin-config is installed, else a stub gin.configurable and the
        value read from the binding line.
  LAUNCH launch_slurm.sh forwards a non-empty value by name (DL_* knobs are not prefix-forwarded).
Standard library only; no torch import."""

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import unittest
from typing import Any, Dict, List, Optional, Tuple

KNOB = "DL_PRESTART_RANK_TIMELINE"
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(_TRAIN, "..", "..", ".."))  # holds generative_recommenders/ and scripts/
_GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_DLP = os.path.join(_TRAIN, "dl_prestart.py")
_LAUNCH = os.path.join(_ROOT, "scripts", "launch_slurm.sh")
_HEAD = "# (109) "
_BIND = re.compile(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % KNOB)


def _gin_lines() -> List[str]:
    with open(_GIN) as f:
        return f.read().splitlines()


def _binding_index(lines: List[str]) -> int:
    return next(i for i, ln in enumerate(lines) if _BIND.match(ln))


def _item_block(lines: List[str]) -> List[str]:
    i_bind = _binding_index(lines)
    i_head = max(i for i in range(i_bind) if lines[i].startswith(_HEAD))
    return lines[i_head:i_bind]


def _flat(block: List[str]) -> str:
    return " ".join(ln[1:].strip() for ln in block)


class BootstrapAstTest(unittest.TestCase):
    def test_one_parameter_one_env_wins_bind(self) -> None:
        with open(_BOOT) as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(names.count(KNOB), 1)
        arg = next(a for a in fn.args.args if a.arg == KNOB)
        self.assertEqual(ast.unparse(arg.annotation) if arg.annotation else None, "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[KNOB], ast.Constant) and defaults[KNOB].value is None)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == KNOB]
        self.assertEqual(len(calls), 1)
        (call,) = calls
        self.assertEqual(getattr(call.func, "id", None), "_bind_bool")  # env wins; False -> "0"
        self.assertEqual([getattr(a, "id", None) for a in call.args[1:]], [KNOB])

    def test_reader_uses_the_same_name(self) -> None:
        with open(_DLP) as f:
            tree = ast.parse(f.read())
        consts = [n.value for n in tree.body if isinstance(n, ast.AnnAssign)
                  and getattr(n.target, "id", None) == "RANK_TIMELINE_ENV"]
        self.assertEqual([getattr(c, "value", None) for c in consts], [KNOB])


class GinBindingTest(unittest.TestCase):
    def test_exactly_one_binding_false(self) -> None:
        lines = _gin_lines()
        self.assertEqual([m.group(1) for m in map(_BIND.match, lines) if m], ["False"])

    def test_item_block_documents_the_knob(self) -> None:
        block = _item_block(_gin_lines())
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = _flat(block)
        for marker in ("Per-rank startup timeline", "Code default OFF; the binding below keeps it OFF.",
                       "WHAT:", "WHY:", "NUMERICS: none", "OFF (default): report() is byte-identical to 544e691",
                       "MLPERF:", "COST (on):", '"[dlstart] rank-timeline rank="',
                       "%s=1 turns it on, =0 is the kill switch" % KNOB, "pre-set env var wins",
                       'only "1" arms it', "launch_slurm.sh forwards"):
            self.assertIn(marker, text)


_CHILD = textwrap.dedent(
    """
    import importlib.util, json, os, sys, types
    root, gin_path, boot_path, dlp_path, bound = sys.argv[1:6]
    real_gin = True
    try:
        import gin
    except ImportError:
        real_gin = False
        gin = types.ModuleType("gin")
        gin.configurable = lambda fn: fn
        sys.modules["gin"] = gin
    if real_gin:
        sys.path.insert(0, root)
        from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
        assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
        gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
        boot.apply_env_bootstrap()
    else:
        spec = importlib.util.spec_from_file_location("boot_under_test", boot_path)
        boot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(boot)
        boot.apply_env_bootstrap(**{"DL_PRESTART_RANK_TIMELINE": {"True": True, "False": False}[bound]})
    spec = importlib.util.spec_from_file_location("dlp_under_test", dlp_path)
    dlp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dlp)
    print("R109_CHILD " + json.dumps({
        "value": os.environ.get("DL_PRESTART_RANK_TIMELINE"),
        "enabled": dlp.rank_timeline_enabled(),
        "real_gin": real_gin,
        "torch_imported": "torch" in sys.modules,
    }, sort_keys=True))
    """
)


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    """A clean child interpreter (gin config and the environment are process-global)."""
    lines = _gin_lines()
    m = _BIND.match(lines[_binding_index(lines)])
    bound = m.group(1) if m else "?"
    env = {k: v for k, v in os.environ.items() if k != KNOB}
    env.update(preset)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, _ROOT, _GIN, _BOOT, _DLP, bound], env=env, cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("R109_CHILD ")]
    got = json.loads(out[0][len("R109_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


class GinRunTest(unittest.TestCase):
    def _check(self, preset: Dict[str, str], want: str, enabled: bool) -> None:
        rc, log, got = _run_child(preset)
        self.assertEqual(rc, 0, log)
        self.assertIsNotNone(got, log)
        assert got is not None
        self.assertEqual((got["value"], got["enabled"]), (want, enabled), preset)
        self.assertFalse(got["torch_imported"])

    def test_gin_default_is_off(self) -> None:
        self._check({}, "0", False)

    def test_preset_one_arms(self) -> None:
        self._check({KNOB: "1"}, "1", True)

    def test_preset_zero_is_the_kill_switch(self) -> None:
        self._check({KNOB: "0"}, "0", False)


class LauncherTest(unittest.TestCase):
    def test_launcher_forwards_the_knob_by_name(self) -> None:
        if not os.path.exists(_LAUNCH):
            self.skipTest("no scripts/launch_slurm.sh in this tree")
        with open(_LAUNCH) as f:
            src = f.read()
        self.assertEqual(src.count("${%s:+-e %s=$%s} \\\n" % (KNOB, KNOB, KNOB)), 1)


if __name__ == "__main__":
    unittest.main()
