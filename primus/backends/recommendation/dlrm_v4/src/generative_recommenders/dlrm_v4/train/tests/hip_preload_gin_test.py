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
"""S14-d v2: DLRM_HSTU_FP8_HIP_PRELOAD lives in gin (yambda_5b.gin item (70) S14-d; knob rule: every feature switch is
an apply_env_bootstrap binding with a documented item block, env is only the kill switch, diagnostic knobs are listed
in a "Diagnostic env knobs (not set in production)" block).

  AST   apply_env_bootstrap declares the parameter once (Optional[bool] = None) and binds it with one env-wins
        _bind_bool call; no parameter is declared twice; the diagnostic _REPORT knob is not a parameter.
  GIN   exactly one binding, "= True"; no binding is repeated; the item block documents the knob, its row counts and
        cost model are the table header's, and its quoted rank-0 arming line has the format preload_from_env really
        logs; the diagnostic block lists the env-only values.
  RUN   clean child interpreters run train_ranker's Phase-1 sequence (gin.parse_config_file(yambda_5b.gin,
        skip_unknown=True), then apply_env_bootstrap()): unset -> "1" (on); pre-set "0" -> "0" (kill switch honored);
        pre-set "" -> "" (kept: off); pre-set "all" -> "all"; other bindings unchanged.
  ORDER every rank process runs the bootstrap before the preload hook reads the env, and launch_slurm.sh forwards
        non-empty DLRM_HSTU_FP8_* values by prefix, so the env kill switch reaches the ranks.
CPU only; no torch import."""

import ast
import ctypes
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

from generative_recommenders.dlrm_v4.train import hip_preload as hp

try:
    import gin  # noqa: F401
except ImportError:  # pragma: no cover
    gin = None  # type: ignore[assignment]

KNOB = "DLRM_HSTU_FP8_HIP_PRELOAD"
REPORT = "DLRM_HSTU_FP8_HIP_PRELOAD_REPORT"
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(_TRAIN, "..", "..", ".."))  # holds generative_recommenders/ and scripts/
_GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_RANKER = os.path.join(_TRAIN, "train_ranker.py")
_LAUNCH = os.path.join(_ROOT, "scripts", "launch_slurm.sh")
_HEAD = "# (70) S14-d: "
_DIAG = "# Diagnostic env knobs (not set in production) -- item (70) S14-d:"


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


class BootstrapAstTest(unittest.TestCase):
    def test_one_parameter_one_env_wins_bind(self) -> None:
        with open(_BOOT) as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        args = fn.args.args + fn.args.kwonlyargs
        names = [a.arg for a in args]
        self.assertEqual(names.count(KNOB), 1)
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        self.assertNotIn(REPORT, names)  # diagnostic knob: env only
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
        # the bootstrap binds exactly the names the runtime reads
        self.assertEqual((hp.ENV, hp.REPORT_ENV), (KNOB, REPORT))


class GinBindingTest(unittest.TestCase):
    def test_exactly_one_binding_true_and_no_repeated_binding(self) -> None:
        lines = _gin_lines()
        pat = re.compile(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % KNOB)
        self.assertEqual([m.group(1) for m in map(pat.match, lines) if m], ["True"])
        self.assertEqual([ln for ln in lines if REPORT in ln and not ln.lstrip().startswith("#")], [])
        names = [m.group(1) for m in (re.match(r"^\s*apply_env_bootstrap\.(\w+)\s*=", ln) for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])

    def test_item_block_documents_the_knob(self) -> None:
        block = _item_block(_gin_lines())
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = _flat(block)
        for marker in ("WHAT:", "WHEN:", "TABLE (v2):", "NUMERICS: BIT-IDENTICAL", "MLPERF:", "COST:", "RESUME:",
                       "Code default (no binding): OFF", "ARMED (rank 0 only, once)",
                       "shape=1n entries=46 (other_shape=2 bad_lines=0) ok=46", "failed=0",
                       "entries=48 (other_shape=0 bad_lines=0) ok=48", "Must be ABSENT:",
                       '"[hip-preload] NOT armed"', '"[hip-preload] rank N:"', "a rank-0 line with failed > 0",
                       f"Kill switch: {KNOB}=0", "pre-set env var wins"):
            self.assertIn(marker, text)
        # the block's row counts and cost model are the checked-in table's header (n_*, entries_by_set)
        header, entries, _bad = hp.load_table(hp.DEFAULT_TABLE)
        by = header["entries_by_set"]
        self.assertEqual(len(entries), header["n_entries"])
        n = (header["n_entries"], header["n_dlsym"], header["n_symtab"])
        self.assertIn("TABLE (v2): %d rows (%d dlsym, %d symtab)." % n, text)
        rows = tuple(by[s]["bundles"] for s in ("1n", "2n", "4n"))
        self.assertIn("header entries_by_set): 1n %d, 2n %d, 4n %d." % rows, text)
        self.assertEqual(by["2n"]["est_s"], by["4n"]["est_s"])  # the block names one value for 2n and 4n
        cost = (by["1n"]["est_s"], by["2n"]["est_s"])
        self.assertIn("(header entries_by_set): 1n %.3f s, 2n and 4n %.3f s." % cost, text)

    def test_diagnostic_block_lists_the_env_only_knobs(self) -> None:
        lines = _gin_lines()
        i = lines.index(_DIAG)
        self.assertEqual(i, _binding_index(lines) + 1)
        j = i + 1
        while j < len(lines) and lines[j].startswith("#   "):
            j += 1
        text = _flat(lines[i:j])
        for marker in (f"{REPORT}=<dir> (env only", f"{KNOB}=all (env only", f"{KNOB}=<table path> (env only"):
            self.assertIn(marker, text)

    def test_documented_arming_line_matches_the_code(self) -> None:
        text = _flat(_item_block(_gin_lines()))
        m = re.search(r'"(armed OK: \[hip-preload\] table=.*?no kernel launch, no tensor/RNG state)"', text)
        self.assertIsNotNone(m)
        doc = re.escape(m.group(1) if m else "")
        for placeholder, rx in (("already_loaded=K", r"already_loaded=\d+"), ("S.SSs", r"\d+\.\d\ds"),
                                ("dev=D", r"dev=-?\d+")):
            doc = doc.replace(re.escape(placeholder), rx)
        logged = _dry_run_arming_line(world_size=8)
        self.assertRegex(logged, "^" + doc + "$")


class _Handle:
    def __getitem__(self, sym: str) -> ctypes.c_void_p:
        return ctypes.c_void_p(0x100000 + 16 * (hash(sym) & 0xFFFF))


class _Hip:
    path = "/fake/libtorch_hip.so"

    def __init__(self, maps: Dict[str, Dict[str, int]]) -> None:
        pass

    def device(self) -> int:
        return 0

    def func_attributes(self, addr: int) -> int:
        return 0

    def clear_error(self) -> int:
        return 0


def _dry_run_arming_line(world_size: int) -> str:
    """preload_from_env with the checked-in table, every library 'loaded' and a fake HIP: rank 0's log message."""
    real = hp.preload_entries

    def opener(name: str, meta: Dict[str, Any], maps: Dict[str, Dict[str, int]]) -> Any:
        return hp._Lib(_Handle(), "/fake/" + name, 0x7F0000000000, "")  # type: ignore[arg-type]

    def entries(sel: List[Dict[str, Any]], libs: Dict[str, Any], fga: Any, clr: Any,
                maps: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, Any]:
        return real(sel, libs, fga, clr, maps, opener=opener)

    records: List[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    keep = _Keep(level=logging.INFO)
    hp.logger.addHandler(keep)
    old = hp.logger.level
    hp.logger.setLevel(logging.INFO)
    try:
        with mock.patch.object(hp, "mapped_libraries", return_value={}), mock.patch.object(hp, "_Hip", _Hip), \
                mock.patch.object(hp, "preload_entries", entries):
            hp.preload_from_env(rank=0, world_size=world_size, local_world_size=8, env={KNOB: "1"})
    finally:
        hp.logger.removeHandler(keep)
        hp.logger.setLevel(old)
    msgs = [r.getMessage() for r in records if r.getMessage().startswith(hp.ARMED)]
    return msgs[0] if len(msgs) == 1 else repr(msgs)


_CHILD = textwrap.dedent(
    """
    import json, os, sys
    root, gin_path = sys.argv[1], sys.argv[2]
    import gin
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    from generative_recommenders.dlrm_v4.train import hip_preload as hp
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
    boot.apply_env_bootstrap()
    path, use_all = hp.resolve_table(os.environ.get(hp.ENV))
    print("S14D_GIN_CHILD " + json.dumps({
        "env": {k: v for k, v in os.environ.items() if k.startswith(("DLRM_", "TRITON_", "HSTU_", "GR_"))},
        "table": os.path.basename(path) if path else None,
        "use_all": use_all,
        "torch_imported": "torch" in sys.modules,
    }, sort_keys=True))
    """
)
_PREFIXES = ("DLRM_", "TRITON_", "HSTU_", "GR_", "PYTORCH_TUNABLEOP")


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    """A clean child interpreter (gin.clear_config and the environment are process-global): only `preset` of the
    knob prefixes is set when it starts."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_PREFIXES)}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_ROOT, env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, _ROOT, _GIN], env=env, cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("S14D_GIN_CHILD ")]
    got = json.loads(out[0][len("S14D_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


@unittest.skipIf(gin is None, "gin-config not installed")
class GinRunTest(unittest.TestCase):
    clean: Optional[Dict[str, Any]] = None
    clean_log: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        _rc, cls.clean_log, cls.clean = _run_child({})

    def _check(self, preset: Dict[str, str], want: str, table: Optional[str], use_all: bool) -> None:
        self.assertIsNotNone(self.clean, self.clean_log)
        if preset:
            rc, log, got = _run_child(preset)
            self.assertEqual(rc, 0, log)
            self.assertIsNotNone(got, log)
        else:
            got = self.clean
        assert got is not None and self.clean is not None
        self.assertEqual(got["env"].get(KNOB), want, preset)
        self.assertEqual((got["table"], got["use_all"]), (table, use_all), preset)
        self.assertEqual(got["env"].get(REPORT), preset.get(REPORT), preset)  # gin never sets it
        self.assertFalse(got["torch_imported"])
        # every other binding of the gin file is the same as in the clean case
        others = {k: v for k, v in got["env"].items() if k != KNOB and k not in preset}
        self.assertEqual(others, {k: v for k, v in self.clean["env"].items() if k != KNOB and k not in preset})

    def test_gin_default_is_on(self) -> None:
        self._check({}, "1", "hip_preload_table.jsonl", False)

    def test_preset_zero_is_the_kill_switch(self) -> None:
        self._check({KNOB: "0"}, "0", None, False)
        self._check({KNOB: ""}, "", None, False)  # an empty value already in the env is kept and disarms (gin note)

    def test_preset_all_wins(self) -> None:
        self._check({KNOB: "all"}, "all", "hip_preload_table.jsonl", True)

    def test_preset_report_dir_is_kept_and_the_knob_stays_on(self) -> None:
        self._check({REPORT: "/tmp/hip_preload_report"}, "1", "hip_preload_table.jsonl", False)


class OrderTest(unittest.TestCase):
    def test_bootstrap_runs_before_the_hook_in_every_rank_process(self) -> None:
        with open(_RANKER) as f:
            tree = ast.parse(f.read())
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

        def calls(fn: ast.FunctionDef, name: str) -> List[Tuple[int, ast.Call]]:
            return [(n.lineno, n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                    and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name)]

        boot = calls(fns["_main_func"], "apply_env_bootstrap")
        hook = calls(fns["_main_func"], "preload_from_env")
        self.assertEqual((len(boot), len(hook)), (1, 1))
        self.assertLess(boot[0][0], hook[0][0])
        parse = [ln for ln, c in calls(fns["_main_func"], "parse_config_file")]
        self.assertTrue(parse and min(parse) < boot[0][0])  # gin bindings parsed before the bootstrap
        # every rank process is a _main_func: launch() (main()'s body, which Primus calls)
        # spawns it once per local GPU
        spawn = calls(fns.get("launch", fns["main"]), "start_processes")
        self.assertEqual(len(spawn), 1)
        self.assertEqual(getattr(spawn[0][1].args[0], "id", None), "_main_func")

    def test_launcher_forwards_the_knob_prefix(self) -> None:
        if not os.path.exists(_LAUNCH):
            self.skipTest("no scripts/launch_slurm.sh in this tree")
        with open(_LAUNCH) as f:
            src = f.read()
        self.assertIn('for v in "${!DLRM_HSTU_FP8_@}"; do', src)
        self.assertIn('FP8_ENV_ARGS="$FP8_ENV_ARGS -e $v=$val"', src)
        # only non-empty values are forwarded: an empty value set for sbatch leaves the gin "1" (gin item note)
        self.assertIn('    if [ -n "$val" ]; then FP8_ENV_ARGS="$FP8_ENV_ARGS -e $v=$val"; fi\n', src)
        self.assertRegex(src, r"\n\s*\$FP8_ENV_ARGS \\\n")  # passed to the worker container


if __name__ == "__main__":
    unittest.main()
