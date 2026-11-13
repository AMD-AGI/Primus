# pyre-strict
"""Item (83) IX wgrad S_L: DLRM_HSTU_FP8_EC_IX_WGRAD_IDX_AFTER_DOT lives in gin (yambda_5b.gin item (83); knob rule:
every feature switch is an apply_env_bootstrap binding with a documented item block, env is only the kill switch).

  AST     apply_env_bootstrap declares the parameter once (Optional[bool] = None) and binds it with one env-wins
          _bind_bool call.
  GIN     exactly one binding, "= True", in an item (83) block placed after item (78) IX's block.
  RUN     a clean child interpreter runs train_ranker's Phase-1 sequence (gin.parse_config_file(yambda_5b.gin,
          skip_unknown=True), then apply_env_bootstrap(); without gin-config installed: a stub gin and the file's
          literal apply_env_bootstrap bindings): unset -> "1"; pre-set "0" -> "0" (kill switch); pre-set "1" -> "1".
  FLAG    ops/triton/triton_ec_indexed_l1.wgrad_idx_after_dot() (executed from its source, no torch): absent / "0"
          -> False (ship14's IDXPF kernel), "1" -> True; read once per process (a later env change is ignored).
  KERNELS _ecix_wgrad (ON) is the current tree's kernel; _ecix_wgrad_idxpf (OFF) has its parameter list and
          decorator, and the two differ only in the docstring and the token-loop statement order: ON
          a, b, dot, ii, ip, dh_ptrs; OFF (ship14 c68395d) a, ii_n, b, ii = ii_n, ip, dot, dh_ptrs. ecix_wgrad
          selects between them with wgrad_idx_after_dot().
CPU only; no torch import (the bitwise ON-vs-OFF kernel check is GPU: ops/tests/ec_indexed_l1_test.py)."""

import ast
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import unittest
from typing import Any, Dict, List, Optional, Tuple

KNOB = "DLRM_HSTU_FP8_EC_IX_WGRAD_IDX_AFTER_DOT"
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(_TRAIN, "..", "..", ".."))  # holds generative_recommenders/ and scripts/
_GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_KMOD = os.path.join(_ROOT, "generative_recommenders", "ops", "triton", "triton_ec_indexed_l1.py")
_HEAD = "# (83) IX wgrad S_L: "


def _gin_lines() -> List[str]:
    with open(_GIN) as f:
        return f.read().splitlines()


def _kernel_module() -> ast.Module:
    with open(_KMOD) as f:
        return ast.parse(f.read())


def _find_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


class BootstrapAstTest(unittest.TestCase):
    def test_one_parameter_one_env_wins_bind(self) -> None:
        with open(_BOOT) as f:
            fn = _find_def(ast.parse(f.read()), "apply_env_bootstrap")
        args = fn.args.args + fn.args.kwonlyargs
        names = [a.arg for a in args]
        self.assertEqual(names.count(KNOB), 1)
        arg = next(a for a in args if a.arg == KNOB)
        self.assertEqual(ast.unparse(arg.annotation) if arg.annotation else None, "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        defaults.update({a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)})
        self.assertTrue(isinstance(defaults[KNOB], ast.Constant) and defaults[KNOB].value is None)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == KNOB]
        self.assertEqual(len(calls), 1)
        (call,) = calls
        self.assertEqual(getattr(call.func, "id", None), "_bind_bool")  # env wins; True -> "1"
        self.assertEqual([getattr(a, "id", None) for a in call.args[1:]], [KNOB])


class GinBindingTest(unittest.TestCase):
    def test_exactly_one_binding_true(self) -> None:
        lines = _gin_lines()
        pat = re.compile(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % KNOB)
        self.assertEqual([m.group(1) for m in map(pat.match, lines) if m], ["True"])
        names = [m.group(1) for m in (re.match(r"^\s*apply_env_bootstrap\.(\w+)\s*=", ln) for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])

    def test_item_block(self) -> None:
        lines = _gin_lines()
        i_bind = next(i for i, ln in enumerate(lines) if ln.startswith(f"apply_env_bootstrap.{KNOB} "))
        i_head = max(i for i in range(i_bind) if lines[i].startswith(_HEAD))
        block = lines[i_head:i_bind]
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = " ".join(ln[1:].strip() for ln in block)
        for marker in ("Code default OFF", "pre-set env var wins", "NUMERICS: BIT-IDENTICAL", "`_ecix_wgrad`",
                       "`_ecix_wgrad_idxpf`", "c68395d", f"Opt out (kill switch): {KNOB}=0",
                       "TSPEC row identity (module, qualname, key) are unchanged", "starting_line_number"):
            self.assertIn(marker, text)
        # next to item (78) IX's block (its binding precedes this one, IX-FP8F (79) follows)
        i_ix = next(i for i, ln in enumerate(lines)
                    if ln.startswith("apply_env_bootstrap.DLRM_HSTU_FP8_EC_INDEXED_L1 "))
        i_f8 = next(i for i, ln in enumerate(lines)
                    if ln.startswith("apply_env_bootstrap.DLRM_HSTU_FP8_EC_INDEXED_L1_FP8 "))
        self.assertLess(i_ix, i_head)
        self.assertLess(i_bind, i_f8)


_CHILD = textwrap.dedent(
    """
    import ast, json, os, re, sys, types
    root, gin_path = sys.argv[1], sys.argv[2]
    try:
        import gin
        real = True
    except ImportError:  # host without gin-config: a stub decorator + the file's literal bindings
        gin = types.ModuleType("gin")
        gin.configurable = lambda f=None, **k: f if f is not None else (lambda g: g)
        sys.modules["gin"] = gin
        real = False
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    if real:
        gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
        boot.apply_env_bootstrap()
    else:
        kw = {}
        with open(gin_path) as f:
            for ln in f:
                m = re.match(r"^apply_env_bootstrap\\.(\\w+)\\s*=\\s*(.+?)\\s*$", ln)
                if m:
                    try:
                        kw[m.group(1)] = ast.literal_eval(m.group(2))
                    except (ValueError, SyntaxError):
                        pass
        boot.apply_env_bootstrap(**kw)
    print("IX83_GIN_CHILD " + json.dumps({"knob": os.environ.get(sys.argv[3]), "real_gin": real,
                                          "torch_imported": "torch" in sys.modules}))
    """
)


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DLRM_", "TRITON_", "HSTU_", "GR_"))}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_ROOT, env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, _ROOT, _GIN, KNOB], env=env, cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("IX83_GIN_CHILD ")]
    got = json.loads(out[0][len("IX83_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


class GinRunTest(unittest.TestCase):
    def _check(self, preset: Dict[str, str], want: str) -> None:
        rc, log, got = _run_child(preset)
        self.assertEqual(rc, 0, log)
        self.assertIsNotNone(got, log)
        assert got is not None
        self.assertEqual(got["knob"], want, preset)
        self.assertFalse(got["torch_imported"])

    def test_gin_default_is_on(self) -> None:
        self._check({}, "1")

    def test_preset_zero_is_the_kill_switch(self) -> None:
        self._check({KNOB: "0"}, "0")

    def test_preset_one_kept(self) -> None:
        self._check({KNOB: "1"}, "1")


def _reader() -> Tuple[Dict[str, Any], Any]:
    """wgrad_idx_after_dot() executed from its own source with the module constants (no torch import)."""
    tree = _kernel_module()
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in (
                "WGRAD_IDX_AFTER_DOT_ENV", "TAG") and isinstance(n.value, ast.Constant):
            consts[n.target.id] = n.value.value
    fn = _find_def(tree, "wgrad_idx_after_dot")
    ns: Dict[str, Any] = dict(consts, os=os, Optional=Optional, logger=logging.getLogger("ix83_test"),
                              _WGRAD_IDX_AFTER_DOT=None)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), _KMOD, "exec"), ns)
    return ns, ns["wgrad_idx_after_dot"]


class FlagTest(unittest.TestCase):
    def test_env_name(self) -> None:
        ns, _ = _reader()
        self.assertEqual(ns["WGRAD_IDX_AFTER_DOT_ENV"], KNOB)

    def test_env_to_flag_read_once(self) -> None:
        for val, want in ((None, False), ("0", False), ("", False), ("1", True), ("true", True), (" On ", True)):
            ns, fn = _reader()
            old = os.environ.pop(KNOB, None)
            try:
                if val is not None:
                    os.environ[KNOB] = val
                self.assertIs(fn(), want, val)
                os.environ[KNOB] = "0" if want else "1"  # read once: a later change is ignored
                self.assertIs(fn(), want, val)
                self.assertIs(ns["_WGRAD_IDX_AFTER_DOT"], want)
            finally:
                os.environ.pop(KNOB, None)
                if old is not None:
                    os.environ[KNOB] = old

    def test_module_default_unset(self) -> None:
        tree = _kernel_module()
        (a,) = [n for n in tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                and n.target.id == "_WGRAD_IDX_AFTER_DOT"]
        self.assertTrue(isinstance(a.value, ast.Constant) and a.value.value is None)


def _loop_stmts(fn: ast.FunctionDef) -> List[str]:
    (loop,) = [n for n in ast.walk(fn) if isinstance(n, ast.For)]
    return [ast.unparse(s) for s in loop.body]


def _strip_doc(fn: ast.FunctionDef) -> List[ast.stmt]:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


class KernelSourceTest(unittest.TestCase):
    def test_two_loop_orders(self) -> None:
        tree = _kernel_module()
        on, off = _find_def(tree, "_ecix_wgrad"), _find_def(tree, "_ecix_wgrad_idxpf")
        nxt = "tl.load(ip + BM * NH, mask=(it + 1) * BM + offs_bm < nrow, other=0)"
        b = "b = tl.load(U + (ii.to(tl.int32) * D)[:, None] + col[None, :])"
        self.assertEqual(_loop_stmts(on), ["a = tl.load(dh_ptrs)", b, "acc = tl.dot(a, b, acc)", f"ii = {nxt}",
                                           "ip += BM * NH", "dh_ptrs += BM * NOUT"])
        self.assertEqual(_loop_stmts(off), ["a = tl.load(dh_ptrs)", f"ii_n = {nxt}", b, "ii = ii_n",
                                            "ip += BM * NH", "acc = tl.dot(a, b, acc)", "dh_ptrs += BM * NOUT"])
        # same decorator, same parameters, everything outside the token loop identical
        self.assertEqual([ast.dump(d) for d in on.decorator_list], [ast.dump(d) for d in off.decorator_list])
        self.assertEqual(ast.dump(on.args), ast.dump(off.args))
        b_on, b_off = _strip_doc(on), _strip_doc(off)
        self.assertEqual(len(b_on), len(b_off))
        diff = [i for i, (x, y) in enumerate(zip(b_on, b_off)) if ast.dump(x) != ast.dump(y)]
        self.assertEqual(len(diff), 1)
        self.assertIsInstance(b_on[diff[0]], ast.For)

    def test_launch_selects_by_flag(self) -> None:
        fn = _find_def(_kernel_module(), "ecix_wgrad")
        src = ast.unparse(fn)
        self.assertIn("wgrad_kernel = _ecix_wgrad if wgrad_idx_after_dot() else _ecix_wgrad_idxpf", src)
        self.assertIn("LAST_KERNELS['wgrad%d' % nh] = wgrad_kernel[nt * s,](", src)
        self.assertNotIn("_ecix_wgrad[", src)


if __name__ == "__main__":
    unittest.main()
