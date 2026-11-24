# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""Tests of MX6's seam (S2; gin item (104), DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6) in ops/triton/triton_hstu_linear.py.

STATIC (no import of the kernels' code paths): every Triton decorator of the module keeps the base's line and def
(critique C6: Triton / TSPEC keys carry the starting line); the three hooked functions (the forward, the backward,
prebuild_l11_do8) are AST-identical to the base (ship16 544e691, E-RM's seams included) once the hooks are
taken out (all three compared by name; the harness interpreter's pin entry must exist and hold all three, so the pin
cannot be skipped or narrowed by a table edit); the backward's dispatch is `l11_go and getattr(ctx, "l11_mx6_layer",
None) is not None and
(l11_mx6 := _l11_mx6_backward(...)) is not None`,
then the 8-name unpack, then the base block under `elif l11_go:`; the MX6 helpers sit after the last kernel; the
keep-mask census of the MX6 helpers (every call that receives the mask, raw or as lnmd_mask_nib.as_int8: the census
lnmd_mask_nib_writers_test's backward census defers to this module); the FP8 fallback's cast is the shipped cast
(same arguments, erm=_erm_take(ctx) included); the MX6 helpers take no other E-RM record and _l11_mx6_backward
declines an E-RM call first (one NOT armed line while MX6 is armed, quiet otherwise).

FORWARD TAG: knob off -> one l11_do8.mx6_enabled() call, nothing set; the static site set decides per layer tag.

BACKWARD, the REAL HSTUComputeOutputFunction.backward with the REAL dO8 registry (plan / cancel / publish / bind) and
the REAL l11_do8 rows (DO8, DO8-ONLY, MX6: preflight, decide, produce, launch, release, rescue accounting, publish).
Stubbed: the ctypes entries of the MX6 .so (the MX6 'only' launch) and the DO8 launch, the CUDA device / stream calls,
L11's own launches, B1a's plan and emitter, the FP8 casts, the out-proj e5m2 scale site (a host stand-in whose `calls`
counts every bookkeeping), r14's producer API (attached to bwd_fp8 under the contract's names) and the W3 wgrad.
  OFF       the knob off, and a call the site set excludes: exactly the base block's calls in the base order and no
            MX6 call at all.
  ON        success (one MX6 launch, an 'only' record, W3 on the producer's dout_t, one bookkeeping; the record is
            counted and prints no MX6 line: the ACTIVE and counters lines print at the prebuild); the registry's
            'plus' (asked once); B1a declined (bf16-du mode, no emission site); preflight declines (the base block);
            the producer declining / raising (plan released, the FP8 cast's own bookkeeping, today's chain); contract
            breaks (raised, plan closed; a W3 dout_t that is not 16-B aligned included, before W3 reads it); a
            failure after the producer at every stage (operand check, launch rc > 0,
            the shared work list, host work after the launch): plan closed, the right latches, the rescue called ONCE
            with the producer's result (the captured inv_scale), no second bookkeeping, L11's own E5 with the SAME B1a
            plan (else the dgrad GEMM and B1a's emitter with that plan), never a second DO8 attempt; FAIL_CLOSED.
  NIB       MX6 + NIB, the state MX6_NIB_GATED True allows (these tests arm MX6 directly and never read the switch:
            the loader's tests in l11_do8_mx6_test.py cover it): the 'only' .so and its MX6 build read NIB and the
            forward saved a NIB keep mask (the writers' layout, lnmd_mask_nib.pack_ref): the MX6 path with the SAME mask
            tensor at the MX6 launch (never unpacked), published and counted; a rescue (the MX6 launch rc, the shared
            work list, host work after the launch) joins L11's own E5 with the same B1a plan and the exact int8 unpack
            (lnmd_mask_nib.as_int8, counted once), which the dgrad GEMM's B1a emitter shares when L11 declines; the
            registry's 'plus' answer: run_e5_plus declines the NIB mask (mask_nib_plus) and L11's own E5 takes the
            unpack; a release: DO8-ONLY's real 'only' launch path gets the saved NIB tensor itself; an int8 mask with
            these builds: the preflight declines (its line names the 'plus' route) and DO8 serves the call as 'plus'
            (run_e5_plus's int8_plus route); a malformed NIB mask: a per-call preflight decline, never a rescue.
  TRANSACTION  every other exit: R2's host checks (a rescue that moves the scale state or returns other operands:
            Mx6ContractError); S1 declining after its bookkeeping; exceptions (incl. BaseException) out of the
            producer, the MX6 launch, the checks after the producer, the decision's counter and the post-launch line:
            mx6_decide's guard / the seam's mx6_abandon close the plan; a registry raise inside mx6_decide (DO8
            latched, the plan cancelled); the hand-over last; cancel before latch before line / count (a raise in the
            rescue line leaves the latches set; FAIL_CLOSED prints and counts no rescue or release); and the ON
            branches: a tag outside the set, an E5 decline or no emission site inside mx6_decide, the bottom / all
            sets on layer 0, the keep-mask format, and the (unreachable) FP8 cast declining after a release.
  E-RM      MX6 beside r14's E-RM row, with the REAL forward tags of both rows: E-RM requested + sites=top (top MX6
            with no _erm_take, bottom E-RM's record taken once by the shipped cast); E-RM requested + latched (bottom
            the FP8 dual cast, top still MX6); sites=bottom / all + E-RM requested, also latched (one INERT line,
            nothing built or tagged); a forced E-RM site on an MX6-tagged call (declined before the preflight, counted,
            one NOT armed line, then one _erm_take); the same pair on a ctx while MX6 is off or latched (declined
            quietly: no MX6 line); the FP8 fallback passes erm=_erm_take(ctx) (None) exactly once.
After every call: no 'only' plan is left open (bind)."""

import ast
import contextlib
import ctypes
import hashlib
import io
import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

import torch

from generative_recommenders.ops import lnmd_mask_nib
from generative_recommenders.ops.hip import l11_do8 as do8
from generative_recommenders.ops.hip import l11_outproj_lnmd as l11
from generative_recommenders.ops.triton import bwd_fp8, hstu_du_e5m2

E4, E5, BF, F32, U8, I8 = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float32, torch.uint8, torch.int8
D, NOUT, H, HD, DU_ROW, T = 512, 1536, 4, 128, 2048, 1000
_THL = Path(do8.__file__).resolve().parents[1] / "triton" / "triton_hstu_linear.py"
IMPL_MX6 = ("l11_outproj_lnmd_bwd_do8 abi=1 variant=e5only(E5 1,WLIST 2 device list,EM 2 only no bf16 DX store,RET 0 "
            "LDS staging,EMV 2 bf16 convert,packed running max)+vd(C1,C2,C3,C4,LANEFIX,PIN)+l11_do8_wl_build "
            "src_sha256=x+mx6(e2m3xe2m3,row400,opsel_k)")
# MX6 + NIB: the NIB MX6 build's impl ("+mkb=nib" before src_sha256, as kitLB m_nib) and its NIB 'only' .so's
IMPL_MX6_NIB = IMPL_MX6.replace(" src_sha256=", "+mkb=nib src_sha256=")
IMPL_ONLY_NIB = IMPL_MX6_NIB.replace("+mx6(e2m3xe2m3,row400,opsel_k)", "")
_PRODUCER = ("output_projection_dgrad_operands_mx6", "output_projection_g8_rescue", "output_projection_mx6_encode")
_MX6_API = ("mx6_preflight", "mx6_decide", "mx6_produce", "run_e5_only_mx6", "mx6_rescue", "mx6_release",
            "mx6_note_decline", "mx6_abandon", "launch_e5_only_mx6", "mx6_site_selected")
_LAUNCH_E5_PLUS = do8.launch_e5_plus  # the real raw DO8 launch (the harness stubs it; real_do8_launch restores it)


def _thl():  # pyre-ignore[3]
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    return thl


class _Capture(contextlib.AbstractContextManager):
    def __enter__(self):
        self.buf = io.StringIO()
        self._cm = contextlib.redirect_stdout(self.buf)
        self._cm.__enter__()
        return self

    def __exit__(self, *a):
        self._cm.__exit__(*a)
        return False

    @property
    def text(self) -> str:
        return self.buf.getvalue()


# ------------------------------------------------------------------------------------------------ STATIC
# The base module's Triton decorators: (decorator line, decorator, def name). Every one must keep its line (its Triton
# cache key, disk-cache key and TSPEC fn_cache_key cover the starting line); kernels may only be appended after them.
_BASE_DECORATORS: Tuple[Tuple[int, str, str], ...] = (
    (278, "triton.jit", "rand3x"), (287, "triton.jit", "_philox4x32_10_mad64"), (318, "triton.jit", "_mask_keep"),
    (330, "triton.jit", "_generate_random_mask"), (379, "triton.jit", "_generate_indexed_random_mask"),
    (417, "triton.jit", "_keep16"), (427, "triton.jit", "_generate_random_mask16"),
    (485, "triton.jit", "_generate_indexed_random_mask16"), (524, "triton_autotune", "_ln_mul_dropout_fwd_rng"),
    (528, "triton.jit", "_ln_mul_dropout_fwd_rng"), (751, "triton.jit", "_amd_v_exp_f32"),
    (758, "triton.jit", "_amd_v_rcp_f32"), (765, "triton.jit", "_sigmoid_ieee_fast"), (788, "triton.jit", "_keep_bit_and"),
    (807, "triton.jit", "_ln_mul_dropout_fwd_rng_v2"), (939, "triton.jit", "_ln_mul_dropout_fwd_rng_v3"),
    (1189, "triton.jit", "_pair_keep_bits"), (1204, "triton.jit", "_xor3s"), (1213, "triton.jit", "_philox4x32_10_b3"),
    (1250, "triton.jit", "_pk_word"), (1258, "triton.jit", "_load_row"), (1268, "triton.jit", "_row_c32"),
    (1325, "triton.jit", "_ln_mul_dropout_fwd_maskgen"), (1430, "triton.jit", "_ln_mul_dropout_fwd"),
    (1568, "triton.jit", "_ln_mul_dropout_bwd_dx_du_rng"), (1788, "triton.jit", "_ln_mul_dropout_bwd_dx_du_rng_v2"),
    (1948, "triton.jit", "_ln_mul_dropout_bwd_dx_du_rng_v3"), (2074, "triton.jit", "_ln_mul_dropout_bwd_dx_du"),
    (2306, "triton_autotune", "_ln_mul_dropout_bwd_dwdb"), (2310, "triton.jit", "_ln_mul_dropout_bwd_dwdb"),
    (3154, "triton.jit", "_ln_mul_dropout_fwd_fp8_yt_kernel"), (4067, "triton.jit", "_group_norm_mul_dropout_fwd"),
    (4158, "triton.jit", "_group_norm_mul_dropout_bwd_dx_du"), (4504, "triton_autotune", "_group_norm_bwd_dwdb"),
    (4508, "triton.jit", "_group_norm_bwd_dwdb"), (5909, "triton.jit", "_helion_ln_mul_dropout_fwd"),
    (6071, "triton.jit", "_helion_ln_mul_dropout_bwd_dx_du"), (6228, "triton_autotune", "_helion_ln_mul_dropout_bwd_dwdb"),
    (6232, "triton.jit", "_helion_ln_mul_dropout_bwd_dwdb"), (6592, "triton.jit", "_mg_sp"),
    (6598, "triton.jit", "_mg_p2"), (6604, "triton.jit", "_mg_bc"), (6610, "triton.jit", "_mg_lo"),
    (6615, "triton.jit", "_mg_hi"), (6620, "triton.jit", "_mg_pk_mul"), (6627, "triton.jit", "_mg_pk_mul_b"),
    (6634, "triton.jit", "_mg_pk_nmul_b"), (6641, "triton.jit", "_mg_pk_fma"), (6647, "triton.jit", "_mg_pk_add1"),
    (6654, "triton.jit", "_mg_pk_nfma1"), (6664, "triton.jit", "_mg_perm"), (6670, "triton.jit", "_mg_or_and_s"),
    (6677, "triton.jit", "_mg_bytemask"), (6696, "triton.jit", "_mg_cvt4"), (6704, "triton.jit", "_mg_fp8_sat_mode"),
    (6712, "triton.jit", "_mg_pk_word"), (6722, "triton.jit", "_row_c32_pk"),
    (6810, "triton.jit", "_ln_mul_dropout_fwd_maskgen_pk"), (7222, "triton.jit", "_mg_nib"),
    (7231, "triton.jit", "_row_c32_pk_nib"), (7311, "triton.jit", "_ln_mul_dropout_fwd_maskgen_pk_nib"),
    (7397, "triton.jit", "_c32_nib"), (7428, "triton.jit", "_row_c32_nib"),
    (7485, "triton.jit", "_ln_mul_dropout_fwd_maskgen_nib"),
)
_DECORATOR = re.compile(r"^\s*@(triton\.jit|triton_autotune\(|triton\.autotune\(|triton\.heuristics)")
# sha256 of ast.dump (docstrings dropped) of the base's three functions the MX6 seam hooks (CPython 3.12's AST;
# re-derive with the base tree for another interpreter): the hooks taken out, the current functions must match.
# The base is the ship16 final tree V2 (tree digest 8a4ff18e): its triton_hstu_linear.py is CM 11ff2e1c's, i.e. NIB v2
# plus r14's E-RM seams (the forward returns _erm_forward_tag(...); the backward's two casts pass erm=_erm_take(ctx)
# and its four wgrad dout reads go through the _erm_* wrappers). Re-derived on V2's own file with _function_digests
# (the MX6 hooks are absent there, so the unhook is the identity). The 734246a + NIB testbed's pins were forward
# b179526d / backward 6ebfc7dd; prebuild_l11_do8 is the same in both (a868fe5d).
# ship16 544e691 (V2 plus main's item renumbering, all outside these three functions) gives the same three digests.
_BASE_DIGESTS: Dict[Tuple[int, int], Dict[str, str]] = {
    (3, 12): {
        "prebuild_l11_do8": "a868fe5d55960626fc49ccf18362bf09f7c51ac2054cea77395fb4564336a372",
        "HSTUComputeOutputFunction.forward": "e54d74ab2b96255725c6235c93e99a7dbbfdb05eb802966d4cca8a17fecba64d",
        "HSTUComputeOutputFunction.backward": "ebae83ecb2c57e81f66e83f886a83496f19561f8fc198fc9b12c0a02a0bddc11",
    },
}
# The three functions the MX6 seam hooks: the pin compares exactly these (never the table's own keys, so a table
# entry that goes missing fails instead of dropping out of the comparison), and the harness interpreter's entry must
# hold all three (CPython 3.12: the venv and the container), so re-keying or emptying the table cannot skip the pin.
_HOOKED: Tuple[str, ...] = ("prebuild_l11_do8", "HSTUComputeOutputFunction.forward",
                            "HSTUComputeOutputFunction.backward")
_PIN_PYTHON: Tuple[int, int] = (3, 12)
_MX6_DEFS = ("_l11_mx6_forward_tag", "_l11_mx6_backward", "_l11_mx6_fp8_block", "_l11_mx6_do8_or_none")
# The keep-mask census of the MX6 helpers: (helper, callee, how the mask is passed). With MX6 + NIB (MX6_NIB_GATED) a
# NIB mask reaches them, so every raw consumer must read the NIB layout or route it: mx6_preflight (the MX6 .so's
# format, then its layout), _du_e5m2_plan (B1a's layout check accepts NIB), run_e5_only_mx6 (the MX6 launch checks the
# MX6 .so's format), _l11_du_e5m2_or_none (lnmd_mask_nib.as_int8 before L11's run_e5), _l11_do8_e5_or_none and
# run_e5_plus (NIB routing: the NIB 'only' .so, else a counted decline), and the helpers themselves (via kw). A new
# consumer must be classified here.
_MX6_MASK_CENSUS: Tuple[Tuple[str, str, str], ...] = (
    ("_l11_mx6_backward", "l11_do8.mx6_preflight", "raw"),
    ("_l11_mx6_backward", "_du_e5m2_plan", "raw"),
    ("_l11_mx6_backward", "l11_do8.run_e5_only_mx6", "**kw"),
    ("_l11_mx6_backward", "_l11_mx6_fp8_block", "kw"),
    ("_l11_mx6_backward", "_l11_du_e5m2_or_none", "**kw"),
    ("_l11_mx6_fp8_block", "l11_outproj_lnmd.run", "as_int8"),
    ("_l11_mx6_fp8_block", "_l11_mx6_do8_or_none", "kw"),
    ("_l11_mx6_fp8_block", "_l11_du_e5m2_or_none", "**kw"),
    ("_l11_mx6_do8_or_none", "_l11_do8_e5_or_none", "**kw"),
    ("_l11_mx6_do8_or_none", "l11_do8.run_e5_plus", "raw"),
)


def _decorators(src: str) -> List[Tuple[int, str, str]]:
    lines = src.split("\n")
    out = []
    for i, ln in enumerate(lines):
        m = _DECORATOR.match(ln)
        if not m:
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"^\s*def \w+", lines[j]):
            j += 1
        out.append((i + 1, m.group(1).rstrip("("), re.match(r"^\s*def (\w+)", lines[j]).group(1)))
    return out


class _Unhook(ast.NodeTransformer):
    """The module's AST with the MX6 hooks taken out (what the base had)."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "_l11_mx6_forward_tag" and len(node.args) == 2:
            return node.args[1]
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        t = node.test
        if (isinstance(t, ast.BoolOp) and isinstance(t.op, ast.And) and len(t.values) == 3
                and isinstance(t.values[0], ast.Name) and t.values[0].id == "l11_go"
                and "l11_mx6_layer" in ast.unparse(t.values[1]) and len(node.orelse) == 1
                and isinstance(node.orelse[0], ast.If)):
            return node.orelse[0]
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if node.value is not None and ast.unparse(node.value) == "l11_do8.prebuild_mx6()":
            return ast.Return(value=ast.Constant(value=False))
        return node


def _function_digests(src: str) -> Dict[str, str]:
    tree = _Unhook().visit(ast.parse(src))
    ast.fix_missing_locations(tree)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value,
                                                                                                        str):
                n.body = b[1:] or [ast.Pass()]
    out = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "prebuild_l11_do8":
            out[n.name] = hashlib.sha256(ast.dump(n).encode()).hexdigest()
        if isinstance(n, ast.ClassDef) and n.name == "HSTUComputeOutputFunction":
            for m in n.body:
                if isinstance(m, ast.FunctionDef):
                    out[f"{n.name}.{m.name}"] = hashlib.sha256(ast.dump(m).encode()).hexdigest()
    return out


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)


class StaticSeamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _THL.read_text()
        cls.tree = ast.parse(cls.src)

    def test_every_triton_decorator_keeps_its_base_line(self) -> None:
        got = _decorators(self.src)
        last = _BASE_DECORATORS[-1][0]
        self.assertEqual([d for d in got if d[0] <= last], list(_BASE_DECORATORS))  # nothing moved, nothing inserted

    def test_the_pin_table_holds_the_harness_interpreter_with_every_hooked_function(self) -> None:
        """Interpreter-independent: _HOOKED is exactly the module's functions that carry an MX6 hook (found in the
        source, the MX6 helpers excluded), the harness interpreter's entry exists, and every entry pins exactly those
        functions with a sha256 each (a re-keyed, emptied or partial table, or a narrowed _HOOKED, fails here)."""
        hooks = ("_l11_mx6_forward_tag(", "_l11_mx6_backward(", "l11_do8.prebuild_mx6()")
        carriers = []
        for node in self.tree.body:
            members = [(node.name, node)] if isinstance(node, ast.FunctionDef) else (
                [(f"{node.name}.{m.name}", m) for m in node.body if isinstance(m, ast.FunctionDef)]
                if isinstance(node, ast.ClassDef) else [])
            for name, fn in members:
                if name not in _MX6_DEFS and any(h in ast.unparse(fn) for h in hooks):
                    carriers.append(name)
        self.assertEqual(sorted(carriers), sorted(_HOOKED))
        self.assertIn(_PIN_PYTHON, _BASE_DIGESTS, f"no pin for the harness interpreter {_PIN_PYTHON}")
        for version, pins in _BASE_DIGESTS.items():
            self.assertEqual(sorted(pins), sorted(_HOOKED), version)
            for name, digest in pins.items():
                self.assertRegex(digest, r"^[0-9a-f]{64}$", (version, name))

    def test_hooked_functions_are_the_base_without_the_hooks(self) -> None:
        version = sys.version_info[:2]
        pins = _BASE_DIGESTS.get(version)
        if pins is None:
            if version == _PIN_PYTHON:
                self.fail(f"no base digest pinned for the harness interpreter {version}")
            self.skipTest(f"no base digest pinned for Python {version} (re-derive with the base tree)")
        digests = _function_digests(self.src)
        got = {k: digests.get(k) for k in _HOOKED}
        want = {k: pins.get(k) for k in _HOOKED}
        moved = sorted(k for k in _HOOKED if got[k] is None or got[k] != want[k])
        self.assertEqual(moved, [], f"not the base (544e691) without the MX6 hooks: {moved}")
        self.assertEqual(got, want)

    def test_backward_dispatch_shape(self) -> None:
        bwd = _method(self.tree, "HSTUComputeOutputFunction", "backward")
        ifs = [n for n in ast.walk(bwd) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
               and "_l11_mx6_backward" in ast.unparse(n.test)]
        self.assertEqual(len(ifs), 1)
        (node,) = ifs
        values = node.test.values
        self.assertTrue(isinstance(node.test.op, ast.And) and len(values) == 3)
        self.assertEqual(ast.unparse(values[0]), "l11_go")
        self.assertEqual(ast.unparse(values[1]), "getattr(ctx, 'l11_mx6_layer', None) is not None")  # the one check
        self.assertEqual(ast.unparse(values[2]),
                         "(l11_mx6 := _l11_mx6_backward(ctx, dout, y_fp8, attn, u, mean, rstd, norm_weight, norm_bias, "
                         "output_weight, random_mask)) is not None")
        self.assertEqual(len(node.body), 1)
        self.assertEqual(ast.unparse(node.body[0]), "dy, dout_fp8, dout_fp8_scale, l11_out, l11_e5, l11_do8_pair, "
                                                    "du_plan, l11_du_planned = l11_mx6")
        self.assertEqual(len(node.orelse), 1)
        self.assertEqual(ast.unparse(node.orelse[0].test), "l11_go")
        self.assertIn("output_projection_dgrad_operands(", ast.unparse(node.orelse[0].body[0]))  # the base block
        fwd = _method(self.tree, "HSTUComputeOutputFunction", "forward")
        tags = [n for n in ast.walk(fwd) if isinstance(n, ast.Assign) and "_l11_mx6_forward_tag" in ast.unparse(n)]
        self.assertEqual([ast.unparse(n) for n in tags],
                         ["ctx.du_e5m2_site = _l11_mx6_forward_tag(ctx, getattr(u, HSTU_DUVQK_SITE_ATTR, None))"])
        pre = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "prebuild_l11_do8")
        self.assertEqual(ast.unparse(pre).count("l11_do8.prebuild_mx6()"), 1)

    def test_mx6_helpers_follow_the_last_kernel(self) -> None:
        defs = {n.name: n.lineno for n in self.tree.body if isinstance(n, ast.FunctionDef)}
        last = _decorators(self.src)[-1][0]
        for name in _MX6_DEFS:
            self.assertGreater(defs[name], last, name)
        body = "".join(ast.get_source_segment(self.src, n) for n in self.tree.body
                       if isinstance(n, ast.FunctionDef) and n.name in _MX6_DEFS)
        for word in ("bf16_written", "_ONLY", ".plan(", "cancel(", "poison", "atexit", "triton.jit"):
            self.assertNotIn(word, body)  # the 'only' / MX6 logic lives in ops/hip/l11_do8.py
        fp8_block = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "_l11_mx6_fp8_block")
        runs = [n for n in ast.walk(fp8_block) if isinstance(n, ast.Call) and ast.unparse(n.func) == "l11_outproj_lnmd.run"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(ast.unparse(runs[0].args[6]), "lnmd_mask_nib.as_int8(random_mask, 'L11 run (bf16-du mode)')")

    def test_mx6_helpers_keep_mask_census(self) -> None:
        """Every call in the MX6 helpers that receives the keep mask (random_mask, kw['random_mask'], kw or **kw) and
        how (raw / as_int8) is exactly _MX6_MASK_CENSUS; kw carries the raw mask and _l11_mx6_fp8_block reads it from
        kw. lnmd_mask_nib_writers_test.test_backward_consumer_census defers these helpers to this census."""
        fns = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name in _MX6_DEFS}
        rows: List[Tuple[str, str, str]] = []
        for name in _MX6_DEFS:
            for node in ast.walk(fns[name]):
                if not isinstance(node, ast.Call):
                    continue
                callee = ast.unparse(node.func)
                if callee in ("dict", "lnmd_mask_nib.as_int8"):
                    continue  # kw's construction (checked below) and the unpack itself
                for a in list(node.args) + [k.value for k in node.keywords if k.arg is not None]:
                    text = ast.unparse(a)
                    if text in ("random_mask", "kw['random_mask']"):
                        rows.append((name, callee, "raw"))
                    elif text.startswith("lnmd_mask_nib.as_int8(") and "random_mask" in text:
                        rows.append((name, callee, "as_int8"))
                    elif text == "kw":
                        rows.append((name, callee, "kw"))
                    elif "random_mask" in text or re.search(r"\bkw\b(?!\[)", text):
                        rows.append((name, callee, "other: " + text))  # an unclassified use: fails below
                for k in node.keywords:
                    text = ast.unparse(k.value)
                    if k.arg is None and ("random_mask" in text or re.search(r"\bkw\b(?!\[)", text)):
                        rows.append((name, callee, "**kw" if text == "kw" else "other: **" + text))
        self.assertEqual(sorted(rows), sorted(_MX6_MASK_CENSUS))  # a new consumer must be classified there
        kw = [n for n in ast.walk(fns["_l11_mx6_backward"]) if isinstance(n, ast.AnnAssign)
              and ast.unparse(n.target) == "kw"]
        self.assertEqual(len(kw), 1)
        self.assertEqual([ast.unparse(k.value) for k in kw[0].value.keywords if k.arg == "random_mask"],
                         ["random_mask"])  # the raw mask (the helpers' consumers above receive it through kw)
        reads = [ast.unparse(n) for n in ast.walk(fns["_l11_mx6_fp8_block"]) if isinstance(n, ast.Assign)
                 and "random_mask" in [ast.unparse(t) for t in n.targets]]
        self.assertEqual(reads, ["random_mask = kw['random_mask']"])

    def test_fp8_fallback_cast_is_the_shipped_cast(self) -> None:
        """Critique C10's mirror, E-RM included: _l11_mx6_fp8_block's FP8 dual cast is the shipped block's call (the
        `elif l11_go:` branch) with the same positional arguments and the same keywords in the same order,
        erm=_erm_take(ctx) included (token_major / t_tiled are the values the seam evaluated once)."""
        cast = "bwd_fp8.output_projection_dgrad_operands"
        bwd = _method(self.tree, "HSTUComputeOutputFunction", "backward")
        shipped = [n for n in ast.walk(bwd) if isinstance(n, ast.Call) and ast.unparse(n.func) == cast]
        fb = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "_l11_mx6_fp8_block")
        mirror = [n for n in ast.walk(fb) if isinstance(n, ast.Call) and ast.unparse(n.func) == cast]
        self.assertEqual((len(shipped), len(mirror)), (1, 1))
        (s,), (m,) = shipped, mirror
        self.assertEqual([ast.unparse(a) for a in m.args], [ast.unparse(a) for a in s.args])
        self.assertEqual([k.arg for k in s.keywords], ["token_major", "t_tiled", "erm"])
        self.assertEqual([k.arg for k in m.keywords], [k.arg for k in s.keywords])
        self.assertEqual([ast.unparse(k.value) for k in m.keywords], ["token_major", "t_tiled", "_erm_take(ctx)"])
        self.assertEqual(ast.unparse(s.keywords[2].value), "_erm_take(ctx)")

    def test_mx6_takes_no_erm_record_and_declines_an_erm_call_first(self) -> None:
        """E-RM (r14's block): the only _erm_take of the MX6 helpers is the FP8 mirror's (above); _l11_mx6_backward's
        first statement declines a call E-RM's forward tagged (ctx.erm_site), counted with one NOT armed line, before
        the preflight or anything else is evaluated."""
        fns = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name in _MX6_DEFS}
        takes = {name: [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                        and ast.unparse(n.func) == "_erm_take"] for name, fn in fns.items()}
        self.assertEqual(takes, {"_l11_mx6_forward_tag": [], "_l11_mx6_backward": [],
                                 "_l11_mx6_fp8_block": ["_erm_take(ctx)"], "_l11_mx6_do8_or_none": []})
        body = fns["_l11_mx6_backward"].body
        first = body[1] if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) else body[0]
        self.assertIsInstance(first, ast.If)
        self.assertEqual(ast.unparse(first.test), "getattr(ctx, 'erm_site', None) is not None")
        self.assertEqual(len(first.body), 2)
        call = first.body[0].value
        self.assertEqual(ast.unparse(call.func), "l11_do8.mx6_note_decline")
        self.assertTrue(call.args[0].value.startswith("E-RM owns this call"))
        # one NOT armed line while MX6 is armed; quiet otherwise (a stale tag with MX6 off or latched prints nothing)
        self.assertEqual([ast.unparse(k) for k in call.keywords], ["quiet=not l11_do8.mx6_enabled()"])
        self.assertEqual(ast.unparse(first.body[1]), "return None")
        self.assertEqual(first.orelse, [])


# ------------------------------------------------------------------------------------------------ FORWARD TAG
class ForwardTagTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.thl = _thl()
        except Exception as error:  # noqa: BLE001
            self.skipTest(f"triton_hstu_linear not importable: {error}")
        do8.reset_for_testing()
        do8.set_mx6_sites_for_testing(None)

    def tearDown(self) -> None:
        do8.set_mx6_sites_for_testing(None)
        do8.reset_for_testing()

    def tag(self, layer: int, site: Any = "B1A-SITE") -> SimpleNamespace:
        ctx = SimpleNamespace()
        with mock.patch.object(self.thl, "_LNMD_LAYER_TAG", layer):
            self.assertIs(self.thl._l11_mx6_forward_tag(ctx, site), site)
        return ctx

    def test_knob_off_one_cheap_call_nothing_set(self) -> None:
        calls = []
        real = do8.mx6_enabled

        def spy() -> bool:
            calls.append(1)
            return real()

        with mock.patch.object(do8, "mx6_enabled", side_effect=spy), \
                mock.patch.object(do8, "mx6_site_selected", side_effect=AssertionError("asked")):
            ctx = self.tag(1)
        self.assertEqual((len(calls), vars(ctx)), (1, {}))

    def test_static_site_set(self) -> None:
        do8.set_mx6_prebuilt_for_testing(True)
        for sites, tagged in (("top", (1, 2)), ("bottom", (0,)), ("all", (0, 1, 2)), ("Top ", (1, 2)), ("x", ())):
            do8.set_mx6_sites_for_testing(sites)
            for layer in (-1, 0, 1, 2):
                ctx = self.tag(layer)
                self.assertEqual(getattr(ctx, "l11_mx6_layer", None), layer if layer in tagged else None, (sites, layer))
        do8.set_mx6_sites_for_testing("all")
        self.assertEqual(vars(self.tag(1, site=None)), {})  # no B1a site: no e5m2-du mode, never MX6
        do8._mx6_broken = "latched"
        self.assertEqual(vars(self.tag(1)), {})


# ------------------------------------------------------------------------------------------------ BACKWARD
class _Facts:
    """Stand-in of the attention forward's C8 plan inputs: reason() None = the C8 do_blk branch (plan "only")."""

    def __init__(self, why: Optional[str] = None) -> None:
        self.why = why

    def reason(self) -> Optional[str]:
        return self.why


class _Leaf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):  # pyre-ignore[2,3]
        return x.detach().view(-1, H, HD).clone()

    @staticmethod
    def backward(ctx, g):  # pyre-ignore[2,3]
        return g.reshape(-1, D)


def _ops6(m: int = T, **over: Any) -> do8.Mx6Operands:
    f = {
        "g6": torch.zeros((m, do8.G6_ROW), dtype=U8),
        "w6": torch.zeros(do8.W6_BYTES, dtype=U8),
        "w6s": torch.zeros(do8.W6S_BYTES, dtype=U8),
        "dout_fp8": torch.zeros(((m + 127) // 128, D, 128), dtype=E4),
        "dout_fp8_scale": torch.full((1,), 2.0**-17),
        "g_decode_scale": torch.full((1,), 2.0**-24),
        "inv_scale_used": torch.full((1,), 2.0**24),
        "rescue_args": do8.Mx6RescueArgs(True, True, E4, 2.0**-7, 448.0, 7, True),
    }
    f.update(over)
    return do8.Mx6Operands(**f)


def _nib_keep_mask(m: int = T, seed: int = 20261013) -> Tuple[torch.Tensor, torch.Tensor]:
    """(an int8 [m, 512] keep mask holding every 3-bit pattern, its NIB packing): uint8 [m, 256], row stride 256,
    16-B aligned, i.e. what the NIB writers save (lnmd_mask_nib.pack_ref is their exact reference)."""
    g = torch.Generator().manual_seed(seed)
    m8 = torch.randint(0, 8, (m, D), generator=g, dtype=I8)
    return m8, lnmd_mask_nib.pack_ref(m8)


class _BackwardBase(unittest.TestCase):
    """HSTUComputeOutputFunction.backward on CPU through the real seam (module docstring)."""

    def setUp(self) -> None:
        try:
            self.thl = _thl()
            from generative_recommenders.ops.flydsl import hstu_do8_registry as reg
        except Exception as error:  # noqa: BLE001
            self.skipTest(f"triton_hstu_linear / hstu_do8_registry not importable: {error}")
        if not hasattr(reg, "plan"):
            self.skipTest("R15's hook has no contract v2")
        self.reg = reg
        self._keys = (reg.KNOB_ENV, reg.TAKE_ENV, reg.ONLY_ENV, reg.POISON_ENV, l11.VD_KNOB, do8.KNOB + "_SRC",
                      do8.KNOB + "_FAIL_CLOSED", do8.MX6_KNOB, do8.MX6_SITES_KNOB, lnmd_mask_nib.KNOB)
        self._env = {k: os.environ.pop(k, None) for k in self._keys}  # once per test (restored by tearDown)
        self._fresh()

    def _fresh(self) -> None:
        """Per case: the knob variables and every registry / DO8 / MX6 / L11 state reset, MX6 armed, fresh stand-ins."""
        reg = self.reg
        for k in self._keys:
            os.environ.pop(k, None)
        os.environ.update({reg.KNOB_ENV: "1", reg.ONLY_ENV: "1", l11.VD_KNOB: "1"})
        reg.reset_for_testing()
        do8.reset_for_testing(unload=True)
        do8.set_requested_for_testing(True)
        do8.set_prebuilt_for_testing(True)
        do8.set_only_prebuilt_for_testing(True)
        do8.set_mx6_sites_for_testing(None)
        l11.reset_for_testing()
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        self.arm_mx6(True)
        self.events: List[str] = []
        self.calls: Dict[str, List[Tuple[Any, ...]]] = {}
        self.state = SimpleNamespace(calls=50, published=True)
        # per-test knobs of the stand-ins
        # "ok" | "none" | "raise" | "none_after_commit" | "raise_after_commit" | "no_bookkeeping" | "double" | an object
        self.s1: Any = "ok"
        self.ops6 = _ops6()
        self.mx6_rc = 0
        self.wl_fail: Optional[BaseException] = None
        self.finalize_fail: Optional[BaseException] = None
        self.l11_e5_ok = True
        self.plan: Any = (torch.full((1,), 2.0**20), torch.full((1,), 2.0**-20), ("GEN", 3))
        self.tiled = True
        self.facts: Optional[_Facts] = None
        self.keep_mask: Optional[torch.Tensor] = None  # the saved keep mask (None: int8 all-ones; NIB: nib_builds)
        self.extra: List[Tuple[Any, str, Dict[str, Any]]] = []  # (target, attr, mock.patch.object kwargs) last

    def tearDown(self) -> None:
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        do8.set_requested_for_testing(None)
        do8.set_mx6_for_testing(None)
        do8.set_mx6_sites_for_testing(None)
        do8.reset_for_testing(unload=True)
        l11.set_requested_for_testing(None)
        l11.reset_for_testing()
        self.reg.reset_for_testing()
        lnmd_mask_nib.reset_for_testing()

    def arm_mx6(self, on: bool) -> None:
        do8.set_mx6_for_testing(on)
        do8.set_mx6_prebuilt_for_testing(on)

    def nib_builds(self, mask: torch.Tensor) -> None:
        """MX6 + NIB (the state MX6_NIB_GATED True allows; MX6 is armed directly, the switch is not read): the NIB knob
        on, the loaded 'only' .so and its MX6 build reading NIB ("+mkb=nib"; _load_mx6 requires the two to agree), and
        `mask` as the keep mask the forward saved. Unpack counter at 0."""
        os.environ[lnmd_mask_nib.KNOB] = "1"
        self.keep_mask = mask
        self.extra += [(do8, "_only_lib", {"new": object()}), (do8, "_impl_only", {"new": IMPL_ONLY_NIB}),
                       (do8, "_impl_mx6", {"new": IMPL_MX6_NIB})]
        lnmd_mask_nib.reset_for_testing()

    def spy_mx6_mask(self) -> List[Tuple[Any, str, str]]:
        """(keep mask, 'only' .so format, MX6 .so format) at every MX6 .so launch (_launch_mx6_so; the real path)."""
        seen: List[Tuple[Any, str, str]] = []
        real = do8._launch_mx6_so

        def spy(*a: Any, **k: Any) -> Any:
            seen.append((a[5], do8.only_mask_format(), do8.mx6_mask_format()))
            return real(*a, **k)

        self.extra.append((do8, "_launch_mx6_so", {"new": spy}))
        return seen

    def real_do8_launch(self) -> List[Tuple[Any, ...]]:
        """The REAL l11_do8.launch_e5_plus (its layout checks and the keep-mask format check of the .so it launches)
        down to the ctypes entries, which record (kind, *args) (MK = the keep mask's pointer at index 8) and the
        "do8_launch:<kind>" event the stub would have recorded."""
        seen: List[Tuple[Any, ...]] = []

        def entry(kind: str) -> Callable[..., int]:
            def f(*args: Any) -> int:
                self.events.append(f"do8_launch:{kind}")
                seen.append((kind,) + args)
                return 0
            return f

        self.extra += [(do8, "launch_e5_plus", {"new": _LAUNCH_E5_PLUS}), (do8, "_fn_launch", {"new": entry("plus")}),
                       (do8, "_fn_launch_only", {"new": entry("only")})]
        return seen

    # ---------------------------------------------------------------- the call
    def graph(self) -> Tuple[Any, torch.Tensor]:
        x = torch.randn(T, D).to(BF).requires_grad_()
        out = _Leaf.apply(x)
        offs = torch.tensor([0, 400, T], dtype=torch.int64)
        self.reg.stash_plan(out, self.facts or _Facts())
        site = self.reg.new_site(out, offs, 4096)
        attn = self.reg.carry_site(out, out.view(-1, D))
        self._keep = (x, out, attn)
        return site, attn

    def make_ctx(self, site: Any, attn: torch.Tensor, layer: Optional[int]) -> SimpleNamespace:
        self.W = torch.nn.Parameter(torch.randn(NOUT, D) * 0.02)
        self.mask = torch.ones(T, D, dtype=I8) if self.keep_mask is None else self.keep_mask
        saved = (attn, torch.randn(T, D).to(BF), torch.ones(D, dtype=BF), torch.zeros(D, dtype=BF), torch.zeros(T),
                 torch.ones(T), self.W, torch.zeros(T, NOUT, dtype=E4), torch.ones(1), self.mask)
        ctx = SimpleNamespace(
            saved_tensors=saved, recompute_y_in_backward=True, fp8_y_saved=True, has_random_mask=True, BLOCK_D=D,
            num_warps=2, eps=1e-6, seed=0, training=True, concat_u=True, concat_x=True, dropout_ratio=0.1, num_heads=1,
            linear_dim=-1, group_norm=False, silu_u=True, mul_u_activation_type="silu", fp8_forward=True,
            du_row_width=DU_ROW, du_e5m2_site=hstu_du_e5m2.DuE5m2Site(4242), do8_site=site, fp8_y_token_major=False,
            fp8_dout_token_major=True,
        )
        if layer is not None:
            ctx.l11_mx6_layer = layer
        return ctx

    def ev(self, name: str, ret: Any = None) -> Callable[..., Any]:
        def f(*a: Any, **k: Any) -> Any:
            self.events.append(name)
            self.calls.setdefault(name, []).append((a, k))
            return ret(*a, **k) if callable(ret) else ret
        return f

    def wrap(self, st: contextlib.ExitStack, obj: Any, name: str, event: str) -> None:
        real = getattr(obj, name)
        st.enter_context(mock.patch.object(obj, name, new=self.ev(event, real)))

    def _fp8_cast(self, *a: Any, **k: Any) -> Tuple[torch.Tensor, ...]:
        self.state.calls += 1  # the FP8 dual cast's own scale bookkeeping
        self.fp8_ops = (torch.zeros(T, D, dtype=E5), torch.full((1,), 2.0**-24),
                        torch.zeros((T + 127) // 128, D, 128, dtype=E4), torch.full((1,), 2.0**-17),
                        torch.zeros(NOUT, D, dtype=E4), torch.full((1,), 2.0**-12))
        return self.fp8_ops

    def _s1(self, dout: torch.Tensor, weight: torch.Tensor, *, token_major: bool, t_tiled: bool) -> Any:
        self.assertTrue(token_major and t_tiled)  # the accepted configuration only
        mode = self.s1
        if mode == "none":
            return None
        if mode == "raise":
            raise RuntimeError("injected: producer launch failed before the bookkeeping")
        if mode == "none_after_commit":
            self.state.calls += 1  # a non-conforming decline: the bookkeeping ran, then None
            return None
        self.state.calls += 1 if mode != "no_bookkeeping" else 0
        self.state.calls += 1 if mode == "double" else 0
        if mode == "raise_after_commit":
            raise RuntimeError("injected: producer raised after its bookkeeping")
        return self.ops6 if mode in ("ok", "no_bookkeeping", "double") else mode

    def _rescue(self, dout: torch.Tensor, weight: torch.Tensor, ops6: Any) -> Tuple[torch.Tensor, ...]:
        self.rescued_with = ops6
        self.rescue_ops = (torch.zeros(T, D, dtype=E5), ops6.g_decode_scale, torch.zeros(NOUT, D, dtype=E4),
                           torch.full((1,), 2.0**-12))
        return self.rescue_ops

    def _do8_launch(self, g8, w8, sa, sb, x, u, mask, mean, rstd, wn, bn, p, inv, dz8, wl, cnt, cap, d8, e8,
                    only=False, dx=None):  # pyre-ignore[2,3]
        self.events.append(f"do8_launch:{'only' if only else 'plus'}")
        dx_ = torch.empty_like(x) if dx is None else dx
        if not only:
            dx_.fill_(1.0)
        return dx_, torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8), 8

    def _mx6_entry(self, *args: Any) -> int:
        self.events.append("mx6_launch")
        self.mx6_args = args
        return self.mx6_rc

    @staticmethod
    def _fn_plan(cap: int, addr: int) -> int:
        out = ctypes.cast(addr, ctypes.POINTER(ctypes.c_longlong))
        for i, v in enumerate((1, 0, 1, 8)):  # G, fl, nf, P
            out[i] = v
        return 0

    def _l11_e5(self, *a: Any, **k: Any) -> Any:
        if not self.l11_e5_ok:
            return None
        x, u = a[4], a[5]
        emit = hstu_du_e5m2.DuE5m2Emit(dz8=torch.zeros(1024, DU_ROW, dtype=E5), u_cols=(0, D), u_amax=torch.zeros(()),
                                       u_colsum=torch.zeros(D), inv_scale=a[17], cast_scale=a[18], rows=T, width=DU_ROW,
                                       site_key=a[16], generation=a[19], buffer_ptr=0)
        du = hstu_du_e5m2.placeholder_du(u, T, D)
        return torch.zeros_like(x), du, torch.zeros(D, dtype=BF), torch.zeros(D, dtype=BF), emit

    def _b1a_emit(self, **k: Any) -> Tuple[Any, ...]:
        emit = hstu_du_e5m2.DuE5m2Emit(dz8=torch.zeros(1024, DU_ROW, dtype=E5), u_cols=(0, D), u_amax=torch.zeros(()),
                                       u_colsum=torch.zeros(D), inv_scale=k["inv_scale"], cast_scale=k["cast_scale"],
                                       rows=T, width=DU_ROW, site_key=k["site_key"], generation=k["generation"],
                                       buffer_ptr=0)
        return torch.zeros_like(k["x"]), hstu_du_e5m2.placeholder_du(k["u"], T, D), torch.zeros(D, dtype=BF), \
            torch.zeros(D, dtype=BF), emit

    def _w3(self, y_fp8, dout_fp8, y_scale, dout_scale, fn):  # pyre-ignore[2,3]
        self.events.append("w3")
        self.w3_operand = (dout_fp8, dout_scale)
        return torch.zeros(D, NOUT), "nn_ck"

    def run_backward(self, layer: Optional[int] = 1, mx6_api_forbidden: bool = False) -> Tuple[Any, str, Any, Any]:
        """One backward of a fresh call; returns (result or the exception, stdout, site, ctx)."""
        thl = self.thl
        site, attn = self.graph()
        ctx = self.make_ctx(site, attn, layer)
        dout = torch.randn(T, D).to(BF)
        self.dout = dout
        wl = (torch.zeros(20, 2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), 20)

        def work_list(*a: Any, **k: Any) -> Any:
            self.events.append("work_list")
            if self.wl_fail is not None:
                raise self.wl_fail
            return wl

        def finalize(*a: Any, **k: Any) -> None:
            self.events.append("dwdb_finalize")
            if self.finalize_fail is not None:
                raise self.finalize_fail

        with contextlib.ExitStack() as st:
            for target, attr, kw in (
                (l11, "decline_reason", {"new": self.ev("decline_reason", None)}),
                (l11, "run_e5", {"new": self.ev("l11_run_e5", self._l11_e5)}),
                (l11, "run", {"new": self.ev("l11_run", None)}),
                (l11, "dwdb_finalize", {"new": finalize}),
                (thl, "_out_wgrad_dout_tiled", {"new": self.ev("dout_tiled", lambda *a: self.tiled)}),
                (thl, "_du_e5m2_plan", {"new": self.ev("b1a_plan", lambda *a, **k: self.plan)}),
                (thl, "_triton_layer_norm_mul_dropout_bwd_du_e5m2", {"new": self.ev("b1a_emit", self._b1a_emit)}),
                (thl, "triton_layer_norm_mul_dropout_bwd",
                 {"new": self.ev("ln_bwd", lambda **k: (torch.zeros_like(k["x"]), torch.zeros(T, D, dtype=BF),
                                                        torch.zeros(D, dtype=BF), torch.zeros(D, dtype=BF), None))}),
                (thl, "_out_wgrad_nn_ck_w3", {"new": self._w3}),
                (thl.wgrad_ck_w3, "enabled", {"return_value": True}),
                (thl, "_fp8_out_wgrad_logged", {"new": True}),
                (thl.fp8_wgrad_snr, "SNR_ENABLED", {"new": False}),
                (bwd_fp8, "output_projection_dgrad_operands", {"new": self.ev("fp8_cast", self._fp8_cast)}),
                (bwd_fp8, "output_projection_dgrad_gemm",
                 {"new": self.ev("gemm", lambda *a: torch.zeros(T, NOUT, dtype=BF))}),
                (bwd_fp8, "scale_controller_enabled", {"return_value": True}),
                (bwd_fp8, "grad_dtype_and_limit", {"return_value": (E5, 57344.0)}),
                (bwd_fp8, "uvqk_tm_pad", {"return_value": 128}),
                (bwd_fp8, "_OUT_WGRAD_T_E4M3", {"new": True}),
                (bwd_fp8, "_helpers_fused_ok", {"return_value": True}),
                (bwd_fp8, "_has_stable_identity", {"return_value": True}),
                (bwd_fp8, "_peek_site_state", {"new": lambda site, disc: self.state}),
                (bwd_fp8, "output_projection_dgrad_operands_mx6", {"create": True, "new": self.ev("s1", self._s1)}),
                (bwd_fp8, "output_projection_g8_rescue", {"create": True, "new": self.ev("rescue", self._rescue)}),
                (bwd_fp8, "output_projection_mx6_encode",
                 {"create": True, "new": mock.Mock(side_effect=AssertionError("encode in a training call"))}),
                (do8, "_lib", {"new": object()}),
                (do8, "launch_e5_plus", {"new": self._do8_launch}),
                (do8, "work_list", {"new": work_list}),
                (do8, "_mx6_lib", {"new": object()}),
                (do8, "_impl_mx6", {"new": IMPL_MX6}),
                (do8, "_fn_launch_mx6", {"new": self._mx6_entry}),
                (do8, "_fn_drain", {"new": lambda: 0}),
                (do8, "_fn_plan", {"new": self._fn_plan}),
                (torch.cuda, "current_device", {"return_value": 0}),
                (torch.cuda, "device", {"new": lambda i: contextlib.nullcontext()}),
                (torch.cuda, "current_stream", {"new": lambda d=None: SimpleNamespace(cuda_stream=0)}),
            ):
                st.enter_context(mock.patch.object(target, attr, **kw))
            for target, attr, kw in self.extra:
                st.enter_context(mock.patch.object(target, attr, **kw))
            st.enter_context(mock.patch.dict(do8._plan_cache, clear=True))
            self.wrap(st, do8, "emission_site", "emission_site")
            self.wrap(st, do8, "publish", "publish")
            if mx6_api_forbidden:
                for name in _MX6_API:
                    st.enter_context(mock.patch.object(do8, name, side_effect=AssertionError(f"MX6 {name} called")))
                for name in ("_l11_mx6_backward", "_l11_mx6_fp8_block", "_l11_mx6_do8_or_none"):
                    st.enter_context(mock.patch.object(thl, name, side_effect=AssertionError(f"{name} called")))
            cap = st.enter_context(_Capture())
            st.enter_context(torch.no_grad())
            try:
                out: Any = thl.HSTUComputeOutputFunction.backward(ctx, dout)
            except BaseException as error:  # noqa: BLE001 -- the test inspects it
                out = error
        return out, cap.text, site, ctx

    # ---------------------------------------------------------------- shared assertions
    def assert_no_open_plan(self, site: Any) -> None:
        self.assertIsNone(site.only_plan)
        if site.record is None or site.record.state != "pending":
            self.assertFalse(self.reg.bind(site, torch.empty(T, H, HD, dtype=BF)))

    def only_stats(self) -> Tuple[int, int, int]:
        s = self.reg.stats()["only"]
        return s["planned"], s["published"], s["cancelled"]


BASE_ORDER = ["decline_reason", "dout_tiled", "fp8_cast", "b1a_plan", "emission_site", "work_list", "do8_launch:only",
              "dwdb_finalize", "publish", "w3"]
MX6_ORDER = ["decline_reason", "dout_tiled", "b1a_plan", "emission_site", "s1", "work_list", "mx6_launch",
             "dwdb_finalize", "publish", "w3"]
# an MX6 decision without an 'only' plan (E5 decline / no emission site inside mx6_decide; a registry failure)
L11_E5_ORDER = ["decline_reason", "dout_tiled", "b1a_plan", "emission_site", "fp8_cast", "l11_run_e5", "w3"]


class _Abort(BaseException):
    """Not an Exception: the seam's own `except Exception` handlers let it through; only mx6_abandon closes the plan."""


class BackwardOffTest(_BackwardBase):
    def test_knob_off_is_the_base_block_with_no_mx6_call(self) -> None:
        self.arm_mx6(False)
        out, text, site, ctx = self.run_backward(layer=None, mx6_api_forbidden=True)
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, BASE_ORDER)
        self.assertNotIn("MX6", text)
        self.assertEqual(self.state.calls, 51)
        self.assertIs(self.w3_operand[0], self.fp8_ops[2])
        self.assertEqual(self.only_stats(), (1, 1, 0))
        self.assertFalse(getattr(out[0], self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid)  # today's 'only' record
        self.assertEqual(do8.mx6_counters()["decided"], 0)

    def test_armed_but_an_excluded_site_is_the_base_block(self) -> None:
        do8.set_mx6_sites_for_testing("top")
        out, text, site, ctx = self.run_backward(layer=None, mx6_api_forbidden=True)  # layer 0: the forward set nothing
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, BASE_ORDER)
        self.assertNotIn("MX6", text)


class BackwardMx6Test(_BackwardBase):
    def test_success_one_mx6_launch_one_bookkeeping_only_record(self) -> None:
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, ["decline_reason", "dout_tiled", "b1a_plan", "emission_site", "s1", "work_list",
                                       "mx6_launch", "dwdb_finalize", "publish", "w3"])
        self.assertEqual(self.state.calls, 51)  # the producer's bookkeeping, once; no FP8 cast
        self.assertIs(self.w3_operand[0], self.ops6.dout_fp8)
        self.assertIs(self.w3_operand[1], self.ops6.dout_fp8_scale)
        head = self.mx6_args[:6]
        self.assertEqual(head[:3], (self.ops6.g6.data_ptr(), self.ops6.w6.data_ptr(), self.ops6.w6s.data_ptr()))
        one = do8._MX6_ONE[str(torch.device("cpu"))]
        self.assertEqual(head[3:], (one.data_ptr(), one.data_ptr(), T))
        self.assertEqual(float(one), 1.0)
        self.assertEqual(self.mx6_args[16], self.plan[0].data_ptr())  # INVS = B1a's inv_scale
        rec = getattr(out[0], self.reg.HSTU_DO8_RECORD_ATTR)
        self.assertFalse(rec.bf16_valid)
        self.assertEqual(self.only_stats(), (1, 1, 0))
        self.assert_no_open_plan(site)
        self.assertTrue(self.reg.bind(site, out[0].view(-1, H, HD)))  # the record binds as any 'only' record
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], c["launched"], c["published"], c["rescued"], c["released"], c["latched"]),
                         (1, 1, 1, 0, 0, 0))
        o = do8.only_counters()
        self.assertEqual((o["planned"], o["only_launched"], o["published"], o["cancelled"]), (1, 1, 1, 0))
        self.assertEqual((do8.counters()["calls"], do8.counters()["emitted"], do8.counters()["published"]), (1, 1, 1))
        # re-cut v2: MX6's ACTIVE line and its counters line print at prebuild_mx6 (before RUN_START); the record is
        # counted (above) and prints neither
        self.assertEqual(text.count("armed OK: " + do8.FAMILY_MX6 + " ACTIVE mode=e5 only"), 0)
        self.assertNotIn("[l11-do8] mx6 counters", text)
        self.assertNotIn("armed OK: " + do8.FAMILY_ONLY + " ACTIVE", text)  # names the FP8 'only' .so: not this record
        self.assertIn("launched as l11_outproj_lnmd_bwd_do8_mx6 (item (104)", text)  # L11's census line, this build
        for absent in ("DISABLED", "INERT", "MX6 rescue:", "MX6 release:", "NOT armed", "DEVICE FAULT", "Do8OnlyError"):
            self.assertNotIn(absent, text)

    def test_nib_mask_takes_the_mx6_path_with_the_same_mask(self) -> None:
        """MX6 + NIB (the state MX6_NIB_GATED allows): with the 'only' .so and its MX6 build reading NIB and the NIB
        keep mask the writers save, MX6 takes the call, the SAME mask tensor reaches the MX6 launch (never unpacked),
        and the record is published and counted exactly as with an int8 mask."""
        m8, nib = _nib_keep_mask()
        self.assertTrue(lnmd_mask_nib.layout_ok(nib, T) and torch.equal(lnmd_mask_nib.unpack(nib), m8))
        self.nib_builds(nib)
        seen = self.spy_mx6_mask()
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertIs(ctx.saved_tensors[-1], nib)  # fixture check: the forward saved the NIB mask (make_ctx)
        self.assertEqual(self.events, MX6_ORDER)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], nib)  # the saved NIB tensor itself ...
        self.assertEqual(seen[0][1:], ("nib", "nib"))  # ... (fixture check: nib_builds' formats) at the MX6 launch
        self.assertEqual(self.mx6_args[8], nib.data_ptr())  # MK of l11do8_e5_only_mx6_launch: the packed bytes
        self.assertEqual(self.mx6_args[16], self.plan[0].data_ptr())  # INVS = B1a's inv_scale
        self.assertEqual(lnmd_mask_nib.counters()["unpack"], 0)  # no consumer unpacked it
        self.assertNotIn("[lnmd-mask-nib]", text)
        self.assertEqual(self.state.calls, 51)  # the producer's bookkeeping, once; no FP8 cast
        self.assertIs(self.w3_operand[0], self.ops6.dout_fp8)
        self.assertFalse(getattr(out[0], self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid)
        self.assertEqual(self.only_stats(), (1, 1, 0))
        self.assert_no_open_plan(site)
        self.assertTrue(self.reg.bind(site, out[0].view(-1, H, HD)))
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], c["launched"], c["published"], c["rescued"], c["released"], c["latched"]),
                         (1, 1, 1, 0, 0, 0))
        self.assertEqual(c["declined"], {})
        o = do8.only_counters()
        self.assertEqual((o["planned"], o["only_launched"], o["published"], o["cancelled"]), (1, 1, 1, 0))
        n = do8.counters()
        self.assertEqual((n["calls"], n["emitted"], n["published"], n["mask_int8_plus"], n["mask_nib_plus"],
                          n["mask_declined"]), (1, 1, 1, 0, 0, 0))
        # re-cut v2: no MX6 ACTIVE / counters line on the record (prebuild_mx6 prints them before RUN_START); item
        # (39)'s census line still names the loaded (fixture) impl that ran
        self.assertEqual(text.count("armed OK: " + do8.FAMILY_MX6 + " ACTIVE mode=e5 only"), 0)
        self.assertNotIn("[l11-do8] mx6 counters", text)
        self.assertIn(f"impl={IMPL_MX6_NIB} M={T} Mp=", text)
        for absent in ("DISABLED", "INERT", "MX6 rescue:", "MX6 release:", "NOT armed", "DEVICE FAULT", "Do8OnlyError"):
            self.assertNotIn(absent, text)

    def test_nib_mask_registry_plus_declines_do8_then_l11s_e5_takes_the_unpack(self) -> None:
        """MX6 + NIB, the dO8 registry plans 'plus' (calibration, debug readers): mx6_decide hands the answer on
        (only_plan False, asked once), run_e5_plus declines the NIB mask (the 'plus' .so reads int8: counted
        mask_nib_plus, one NOT armed line, no DO8 launch) and L11's own E5 launch takes the call with the exact unpack
        (counted once)."""
        self.facts = _Facts("calibration step: BF16 q/k")
        m8, nib = _nib_keep_mask()
        self.nib_builds(nib)
        launches = self.real_do8_launch()
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        # (run_e5_plus builds the work list before it declines a NIB mask on a 'plus' call, as at any DO8 site)
        self.assertEqual(self.events, ["decline_reason", "dout_tiled", "b1a_plan", "emission_site", "fp8_cast",
                                       "work_list", "l11_run_e5", "w3"])
        self.assertEqual(launches, [])  # DO8 never launched: its 'plus' .so reads int8
        n = do8.counters()
        self.assertEqual((n["calls"], n["emitted"], n["mask_nib_plus"], n["mask_int8_plus"], n["mask_declined"]),
                         (1, 0, 1, 0, 1))
        self.assertEqual(text.count(do8.FAMILY + " NOT armed: E5 mask format: keep mask NIB on a call DO8-ONLY runs "
                                    "as 'plus'"), 1)
        mask = self.calls["l11_run_e5"][0][0][14]
        self.assertTrue(mask.dtype == I8 and tuple(mask.shape) == (T, D) and torch.equal(mask, m8))
        self.assertEqual(lnmd_mask_nib.counters()["unpack"], 1)
        self.assertEqual(self.reg.stats()["only"]["forced_plus"], {"calibration step: BF16 q/k": 1})  # asked once
        self.assertEqual(do8.only_counters()["plan_plus"], 1)
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], c["declined"]), (0, {}))
        self.assertEqual(self.state.calls, 51)
        self.assert_no_open_plan(site)

    def test_nib_mask_release_then_do8_only_launches_the_same_nib_tensor(self) -> None:
        """MX6 + NIB, the producer declines (scale state untouched): the plan is released, today's chain plans 'only'
        again and the REAL launch_e5_plus hands the 'only' entry the saved NIB tensor itself (its format check passes:
        the NIB 'only' .so; MK = that tensor's pointer); nothing is unpacked."""
        self.s1 = "none"
        m8, nib = _nib_keep_mask()
        self.nib_builds(nib)
        launches = self.real_do8_launch()
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, ["decline_reason", "dout_tiled", "b1a_plan", "emission_site", "s1", "fp8_cast",
                                       "work_list", "do8_launch:only", "dwdb_finalize", "publish", "w3"])
        self.assertEqual([(k[0], k[8]) for k in launches], [("only", nib.data_ptr())])  # MK: the packed bytes
        self.assertEqual(lnmd_mask_nib.counters()["unpack"], 0)
        self.assertNotIn("[lnmd-mask-nib]", text)
        self.assertEqual(self.only_stats(), (2, 1, 1))  # MX6's plan released, DO8's own plan published
        n = do8.counters()
        self.assertEqual((n["mask_nib_plus"], n["mask_int8_plus"], n["mask_declined"]), (0, 0, 0))
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], c["released"], c["rescued"], c["latched"]), (1, 1, 0, 0))
        self.assertEqual(text.count(do8.FAMILY_MX6 + " release: producer declined (scale state untouched)"), 1)
        self.assertFalse(getattr(out[0], self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid)  # today's 'only' record
        self.assertEqual(self.state.calls, 51)
        self.assert_no_open_plan(site)

    def test_registry_plus_is_asked_once_and_runs_the_fp8_plus_path(self) -> None:
        self.facts = _Facts("calibration step: BF16 q/k")
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, ["decline_reason", "dout_tiled", "b1a_plan", "emission_site", "fp8_cast",
                                       "work_list", "do8_launch:plus", "dwdb_finalize", "publish", "w3"])
        self.assertEqual(self.reg.stats()["only"]["forced_plus"], {"calibration step: BF16 q/k": 1})  # once
        self.assertEqual(do8.only_counters()["plan_plus"], 1)
        self.assertTrue(getattr(out[0], self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid)
        self.assertEqual(self.state.calls, 51)
        self.assertEqual((do8.mx6_counters()["decided"], do8.counters()["calls"]), (0, 1))
        self.assert_no_open_plan(site)

    def test_b1a_declined_is_the_bf16_du_mode_without_an_emission_site(self) -> None:
        self.plan = None
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, ["decline_reason", "dout_tiled", "b1a_plan", "fp8_cast", "l11_run", "gemm",
                                       "ln_bwd", "w3"])
        args = self.calls["l11_run"][0][0]
        self.assertIs(args[6], self.mask)  # lnmd_mask_nib.as_int8(int8 mask) is the mask itself
        self.assertEqual(do8.counters()["calls"], 0)  # DO8's emission site never asked (as the base block)
        self.assertEqual(self.only_stats(), (0, 0, 0))
        self.assertEqual(self.state.calls, 51)

    def test_preflight_declines_run_the_base_block_exactly(self) -> None:
        for name, arrange, quiet in (
            ("warm-up", lambda: setattr(self.state, "published", False), True),
            ("SNR", lambda: None, False),
            ("C3 dout_t copy", lambda: None, False),
        ):
            self._fresh()
            arrange()
            if name == "SNR":
                self.extra.append((self.thl.fp8_wgrad_snr, "SNR_ENABLED", {"new": True}))
                self.extra.append((l11, "decline_reason", {"new": self.ev("decline_reason", None)}))  # L11 D5 aside
            if name == "C3 dout_t copy":
                self.extra.append((bwd_fp8, "_OUT_WGRAD_T_E4M3", {"new": False}))
            out, text, site, ctx = self.run_backward()
            self.assertIsInstance(out, tuple, (name, out))
            self.assertEqual(self.events, BASE_ORDER, name)
            self.assertEqual("NOT armed" in text, not quiet, (name, text))
            self.assertEqual(sum(do8.mx6_counters()["declined"].values()), 1, name)

    def test_dout_t_not_step_tiled_declines_then_the_base_block(self) -> None:
        self.tiled = False
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, ["decline_reason", "dout_tiled"] + BASE_ORDER[1:])  # asked again by the base block
        self.assertIn(do8.FAMILY_MX6 + " NOT armed: C3 the W3 dout_t is not step-tiled", text)

    def test_producer_declines_plan_released_fp8_cast_today_chain(self) -> None:
        self.s1 = "none"
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, ["decline_reason", "dout_tiled", "b1a_plan", "emission_site", "s1", "fp8_cast",
                                       "work_list", "do8_launch:only", "dwdb_finalize", "publish", "w3"])
        self.assertEqual(self.state.calls, 51)  # the FP8 cast's bookkeeping only
        self.assertEqual(self.only_stats(), (2, 1, 1))  # MX6's plan released, DO8's own plan published
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], c["released"], c["rescued"], c["latched"]), (1, 1, 0, 0))
        self.assertEqual(text.count(do8.FAMILY_MX6 + " release: producer declined (scale state untouched)"), 1)
        self.assertIs(self.w3_operand[0], self.fp8_ops[2])
        self.assertEqual(len(self.calls["b1a_plan"]), 1)  # B1a's plan once
        self.assertEqual(do8.counters()["calls"], 1)  # DO8's emission site once
        self.assertTrue(do8.mx6_enabled())
        self.assert_no_open_plan(site)

    def test_producer_raises_plan_released_mx6_latched_fp8_path(self) -> None:
        self.s1 = "raise"
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events[4:7], ["s1", "fp8_cast", "work_list"])
        self.assertEqual(self.state.calls, 51)
        self.assertIn(do8.FAMILY_MX6 + " DISABLED: producer raised: RuntimeError: injected", text)
        self.assertIsNotNone(do8.mx6_broken())
        self.assertTrue(do8.enabled() and do8.only_enabled())
        self.assertEqual(self.only_stats(), (2, 1, 1))
        self.assert_no_open_plan(site)

    def test_contract_breaks_raise_with_the_plan_closed(self) -> None:
        kt = (T + 127) // 128
        for mode, want in (("no_bookkeeping", "ran 0 times"), ("double", "ran 2 times"),
                           ("raise_after_commit", "ran 1 times"), (("not", "operands"), "not l11_do8.Mx6Operands"),
                           ("ok-bad-dout", "dout_fp8"), ("ok-misaligned-dout", "a16=0")):
            self._fresh()
            self.s1 = "ok" if mode in ("ok-bad-dout", "ok-misaligned-dout") else mode
            if mode == "ok-bad-dout":
                self.ops6 = _ops6(dout_fp8=torch.zeros(D, 1024, dtype=E4))  # the 2-D layout: not what W3 is told
            if mode == "ok-misaligned-dout":
                # the step-tiled shape, contiguous, but not 16-B aligned: x21 would decline it at W3 and turn the step
                # tiling (and E-RM's row-major entry) off for the process; a contract break instead, before W3 runs
                mis = torch.zeros(kt * D * 128 + 16, dtype=U8)[1:kt * D * 128 + 1].view(E4).view(kt, D, 128)
                self.assertTrue(mis.is_contiguous() and mis.data_ptr() % 16 != 0)
                self.ops6 = _ops6(dout_fp8=mis)
            out, text, site, ctx = self.run_backward()
            self.assertIsInstance(out, do8.Mx6ContractError, (mode, out))
            self.assertIn(want, str(out), mode)
            self.assertIsNotNone(do8.mx6_broken(), mode)
            self.assertEqual(self.only_stats(), (1, 0, 1), mode)
            self.assert_no_open_plan(site)
            self.assertNotIn("rescue", self.events, mode)
            self.assertNotIn("w3", self.events, mode)  # W3 never reads an operand the contract rejected

    # ---------------------------------------------------------------- failures after the producer: the rescue
    def check_rescue(self, out: Any, text: str, site: Any, do8_latched: bool, stage: str) -> None:
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.calls["rescue"][0][0][2], self.ops6)  # ONE rescue, with the producer's result ...
        self.assertEqual(len(self.calls["rescue"]), 1)
        self.assertIs(self.rescued_with.inv_scale_used, self.ops6.inv_scale_used)  # ... its captured inv_scale bits
        self.assertEqual(self.state.calls, 51)  # no second bookkeeping (no FP8 cast)
        self.assertNotIn("fp8_cast", self.events)
        self.assertFalse(any(e.startswith("do8_launch") for e in self.events))  # never a second DO8 attempt
        g8, g8s, w8, w8s = self.rescue_ops
        args = self.calls["l11_run_e5"][0][0]
        self.assertTrue(args[0] is g8 and args[1] is w8 and args[2] is g8s and args[3] is w8s)
        self.assertIs(g8s, self.ops6.g_decode_scale)
        self.assertTrue(args[17] is self.plan[0] and args[18] is self.plan[1] and args[19] is self.plan[2])  # same plan
        self.assertIs(self.w3_operand[0], self.ops6.dout_fp8)  # W3 keeps the producer's operand
        self.assertNotIn("publish", self.events)
        self.assertEqual(self.only_stats(), (1, 0, 1))
        self.assert_no_open_plan(site)
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], c["rescued"], c["released"], c["latched"]), (1, 1, 0, 1))
        self.assertEqual(text.count(do8.FAMILY_MX6 + " rescue: "), 1)
        self.assertIn(stage, text)
        self.assertEqual(do8.broken() is not None, do8_latched)
        self.assertIsNone(do8.only_broken())

    def test_rescue_on_the_mx6_launch_rc(self) -> None:
        self.mx6_rc = 1
        out, text, site, ctx = self.run_backward()
        self.assertEqual(self.events, ["decline_reason", "dout_tiled", "b1a_plan", "emission_site", "s1", "work_list",
                                       "mx6_launch", "rescue", "l11_run_e5", "w3"])
        self.check_rescue(out, text, site, False, "Mx6LaunchError: l11do8_e5_only_mx6_launch rc=1 (hipError) M=1000 "
                                                  "(MX6, before the MX6 EM = 2 launch)")
        self.assertTrue(do8.enabled() and do8.only_enabled())  # MX6 only

    def test_rescue_on_an_operand_check(self) -> None:
        self.ops6 = _ops6(g6=torch.zeros((T + 1, 400), dtype=U8))
        out, text, site, ctx = self.run_backward()
        self.assertNotIn("mx6_launch", self.events)
        self.check_rescue(out, text, site, False, "Mx6OperandError: G6")

    def test_rescue_on_the_shared_work_list_latches_do8_too(self) -> None:
        self.wl_fail = RuntimeError("l11do8_wl_build_launch rc=1 (hipError) M=1000 B=2")
        out, text, site, ctx = self.run_backward()
        self.check_rescue(out, text, site, True, "(shared machinery, before the MX6 EM = 2 launch)")
        self.assertIn(do8.FAMILY + " DISABLED: e5 only call (before the MX6 EM = 2 launch): RuntimeError: "
                      "l11do8_wl_build_launch rc=1", text)

    def test_rescue_after_the_launch_latches_do8_too(self) -> None:
        self.finalize_fail = RuntimeError("injected: dW/dB finalize failed")
        out, text, site, ctx = self.run_backward()
        self.assertIn("mx6_launch", self.events)
        self.check_rescue(out, text, site, True, "(shared machinery, after the MX6 EM = 2 launch)")
        self.assertEqual(do8.mx6_counters()["launched"], 1)

    def test_rescue_then_l11_declines_gemm_and_b1a_with_the_same_plan(self) -> None:
        self.mx6_rc = 2
        self.l11_e5_ok = False
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events[-5:], ["rescue", "l11_run_e5", "gemm", "b1a_emit", "w3"])
        g8, g8s, w8, w8s = self.rescue_ops
        self.assertEqual(self.calls["gemm"][0][0][:4], (g8, w8, g8s, w8s))
        k = self.calls["b1a_emit"][0][1]
        self.assertTrue(k["inv_scale"] is self.plan[0] and k["cast_scale"] is self.plan[1]
                        and k["generation"] is self.plan[2])
        self.assertEqual(self.state.calls, 51)
        self.assert_no_open_plan(site)

    def test_nib_mask_rescue_joins_l11s_own_e5_with_the_same_plan_and_the_exact_unpack(self) -> None:
        """MX6 + NIB, a failure after the producer committed the scale state: the seam rescues as for an int8 mask
        (check_rescue: one R2 rescue at the captured inv_scale, no second bookkeeping, never a second DO8 attempt, L11's
        own E5 with the SAME B1a plan). L11's E5 reads int8, so it gets the exact unpack of the NIB mask
        (lnmd_mask_nib.as_int8: counted once, one line): byte for byte the int8 mask the writers packed."""
        for name, arrange, launched, do8_latched, stage in (
            ("MX6 launch rc", lambda: setattr(self, "mx6_rc", 1), True, False,
             "Mx6LaunchError: l11do8_e5_only_mx6_launch rc=1 (hipError) M=1000 (MX6, before the MX6 EM = 2 launch)"),
            ("shared work list", lambda: setattr(self, "wl_fail", RuntimeError("l11do8_wl_build_launch rc=1")), False,
             True, "(shared machinery, before the MX6 EM = 2 launch)"),
            ("after the launch", lambda: setattr(self, "finalize_fail", RuntimeError("injected: dW/dB finalize")), True,
             True, "(shared machinery, after the MX6 EM = 2 launch)"),
        ):
            self._fresh()
            arrange()
            m8, nib = _nib_keep_mask()
            self.nib_builds(nib)
            seen = self.spy_mx6_mask()
            out, text, site, ctx = self.run_backward()
            self.check_rescue(out, text, site, do8_latched, stage)
            self.assertEqual([(s[0] is nib,) + s[1:] for s in seen], [(True, "nib", "nib")] if launched else [], name)
            mask = self.calls["l11_run_e5"][0][0][14]  # L11's own E5 launch: its int8 [M, 512] keep mask
            self.assertIsNot(mask, nib, name)
            self.assertTrue(mask.dtype == I8 and tuple(mask.shape) == (T, D) and torch.equal(mask, m8), name)
            self.assertEqual(lnmd_mask_nib.counters()["unpack"], 1, name)
            self.assertEqual(text.count("[lnmd-mask-nib] LN*U*dropout keep mask NIB: exact unpack to int8 for L11 "
                                        "run_e5 (_l11_du_e5m2_or_none: DO8 declined, latched or off)"), 1, name)

    def test_nib_mask_rescue_then_l11_declines_gemm_and_b1a_share_the_unpack(self) -> None:
        self.mx6_rc = 2
        self.l11_e5_ok = False
        m8, nib = _nib_keep_mask()
        self.nib_builds(nib)
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events[-5:], ["rescue", "l11_run_e5", "gemm", "b1a_emit", "w3"])
        g8, g8s, w8, w8s = self.rescue_ops
        self.assertEqual(self.calls["gemm"][0][0][:4], (g8, w8, g8s, w8s))
        k = self.calls["b1a_emit"][0][1]
        self.assertTrue(k["inv_scale"] is self.plan[0] and k["cast_scale"] is self.plan[1]
                        and k["generation"] is self.plan[2])
        self.assertIs(k["random_mask"], self.calls["l11_run_e5"][0][0][14])  # ONE unpack, cached on the NIB mask
        self.assertTrue(torch.equal(k["random_mask"], m8))
        self.assertEqual(lnmd_mask_nib.counters()["unpack"], 1)
        self.assertEqual(self.state.calls, 51)
        self.assert_no_open_plan(site)

    def test_fail_closed_cancels_before_raising(self) -> None:
        for name, arrange, want in (
            ("MX6 launch rc", lambda: setattr(self, "mx6_rc", 1), "MX6 failed with DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1"),
            ("shared work list", lambda: setattr(self, "wl_fail", RuntimeError("wl rc=1")),
             do8.FAMILY + " failed with DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1: e5 only call (before the MX6"),
            ("producer raise", lambda: setattr(self, "s1", "raise"),
             "MX6 failed with DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1: producer raised"),
        ):
            self._fresh()
            os.environ[do8.KNOB + "_FAIL_CLOSED"] = "1"
            arrange()
            out, text, site, ctx = self.run_backward()
            self.assertIsInstance(out, RuntimeError, name)
            self.assertIn(want, str(out), name)
            self.assertNotIn("rescue", self.events, name)
            self.assertNotIn("fp8_cast", self.events, name)
            self.assertEqual(self.only_stats(), (1, 0, 1), name)
            self.assert_no_open_plan(site)
            # nothing was rescued or released: no such line, no such count, no latch (FAIL_CLOSED raises instead)
            for absent in (do8.FAMILY_MX6 + " rescue: ", do8.FAMILY_MX6 + " release: ", "DISABLED"):
                self.assertNotIn(absent, text, name)
            c = do8.mx6_counters()
            self.assertEqual((c["rescued"], c["released"], c["latched"]), (0, 0, 0), name)

    def test_later_calls_after_a_latch_take_the_fp8_path(self) -> None:
        self.mx6_rc = 1
        self.run_backward()
        self.assertIsNotNone(do8.mx6_broken())
        self.events.clear()
        self.mx6_rc = 0
        out, text, site, ctx = self.run_backward()  # the forward would not tag it any more; tagged anyway: preflight
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, BASE_ORDER)
        self.assertNotIn("NOT armed", text)  # a latch is an expected state


class BackwardMx6TransactionTest(_BackwardBase):
    """Every other exit of the MX6 transaction (module docstring, TRANSACTION)."""

    def assert_closed(self, site: Any, name: str = "") -> None:
        self.assert_no_open_plan(site)
        self.assertEqual(self.only_stats(), (1, 0, 1), name)  # MX6's plan: planned once, cancelled once
        self.assertEqual(do8._ONLY_CALL, {}, name)  # no stale in-flight 'only' call

    def raising_print(self, marker: str) -> None:
        real = do8.arm_print_all_ranks

        def bad(msg: str, *a: Any, **k: Any) -> Any:
            if marker in msg:
                raise RuntimeError(f"injected: the line with {marker!r} raised")
            return real(msg, *a, **k)

        self.extra.append((do8, "arm_print_all_ranks", {"new": bad}))

    # ---------------------------------------------------------------- R2 (the rescue) checked on the host
    def test_rescue_that_moves_the_scale_state_is_a_contract_break(self) -> None:
        self.mx6_rc = 1

        def rescue(dout: torch.Tensor, weight: torch.Tensor, ops6: Any) -> Tuple[torch.Tensor, ...]:
            self.state.calls += 1  # a non-conforming rescue: it re-runs the gradient site's bookkeeping
            return self._rescue(dout, weight, ops6)

        self.extra.append((bwd_fp8, "output_projection_g8_rescue", {"create": True, "new": self.ev("rescue", rescue)}))
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, do8.Mx6ContractError, out)
        self.assertIn("broke the R2 contract", str(out))
        self.assertIn("ran 1 times (calls 51 -> 52)", str(out))
        self.assertEqual(len(self.calls["rescue"]), 1)
        self.assertNotIn("l11_run_e5", self.events)  # nothing ran on the rescued operands
        self.assertIsNotNone(do8.mx6_broken())
        self.assert_closed(site)

    def test_rescue_must_return_the_fp8_dgrad_operands(self) -> None:
        for name, bad, want in (
            ("3-tuple", lambda r: r[:3], "not (g8, g8_scale, w8, w8_scale)"),
            ("list", list, "not (g8, g8_scale, w8, w8_scale)"),
            ("g8 e4m3", lambda r: (r[0].view(E4),) + r[1:], "g8 torch.float8_e4m3fn"),
            ("g8 rows", lambda r: (r[0][:-1],) + r[1:], "g8 torch.float8_e5m2 (999, 512)"),
            ("w8 rows", lambda r: r[:2] + (r[2][:-1], r[3]), "w8 torch.float8_e4m3fn (1535, 512)"),
            ("w8 scale", lambda r: r[:3] + (torch.ones(2),), "w8_scale"),
            ("g8 scale fp64", lambda r: (r[0], r[1].double()) + r[2:], "g8_scale"),
        ):
            self._fresh()
            self.mx6_rc = 1

            def rescue(dout: torch.Tensor, weight: torch.Tensor, ops6: Any, bad: Any = bad) -> Any:
                return bad(self._rescue(dout, weight, ops6))

            self.extra.append((bwd_fp8, "output_projection_g8_rescue",
                               {"create": True, "new": self.ev("rescue", rescue)}))
            out, text, site, ctx = self.run_backward()
            self.assertIsInstance(out, do8.Mx6ContractError, (name, out))
            self.assertIn("broke the R2 contract", str(out), name)
            self.assertIn(want, str(out), name)
            self.assertNotIn("l11_run_e5", self.events, name)
            self.assertEqual(self.state.calls, 51, name)
            self.assert_closed(site, name)

    def test_rescue_raise_propagates_with_the_plan_closed(self) -> None:
        self.mx6_rc = 1

        def rescue(*a: Any, **k: Any) -> Any:
            raise RuntimeError("injected: rescue cast failed")

        self.extra.append((bwd_fp8, "output_projection_g8_rescue", {"create": True, "new": self.ev("rescue", rescue)}))
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, RuntimeError, out)
        self.assertIn("injected: rescue cast failed", str(out))
        self.assertIsNotNone(do8.mx6_broken())
        self.assert_closed(site)

    # ---------------------------------------------------------------- S1's contract and exceptions
    def test_producer_declining_after_its_bookkeeping_is_a_contract_break(self) -> None:
        self.s1 = "none_after_commit"
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, do8.Mx6ContractError, out)
        self.assertIn("broke the S1 contract", str(out))
        self.assertIn("ran 1 times (calls 50 -> 51) when the producer declined", str(out))
        self.assertNotIn("fp8_cast", self.events)  # never a second bookkeeping
        self.assertEqual(self.state.calls, 51)
        self.assertIsNotNone(do8.mx6_broken())
        self.assertEqual(do8.mx6_counters()["released"], 0)
        self.assert_closed(site)

    def test_base_exception_from_the_producer_abandons_the_plan(self) -> None:
        def s1(*a: Any, **k: Any) -> Any:
            self.events.append("s1")
            raise _Abort("injected BaseException inside the producer")

        self.extra.append((bwd_fp8, "output_projection_dgrad_operands_mx6", {"create": True, "new": s1}))
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, _Abort, out)
        self.assert_closed(site)
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], c["released"], c["latched"]), (1, 1, 0))
        self.assertNotIn("fp8_cast", self.events)

    def test_base_exception_inside_the_mx6_launch_abandons_the_plan(self) -> None:
        self.wl_fail = _Abort("injected BaseException in the work list")
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, _Abort, out)
        self.assertEqual(self.state.calls, 51)
        self.assertNotIn("rescue", self.events)
        self.assert_closed(site)
        self.assertEqual(do8.mx6_counters()["released"], 1)

    def test_exception_in_the_checks_after_the_producer_abandons_the_plan(self) -> None:
        self.extra.append((do8, "_mx6_contract_mismatch", {"side_effect": RuntimeError("injected: check raised")}))
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, RuntimeError, out)
        self.assertIn("injected: check raised", str(out))
        self.assertEqual(self.state.calls, 51)  # S1's bookkeeping only (the call aborts)
        self.assertNotIn("fp8_cast", self.events)
        self.assert_closed(site)

    def test_registry_raise_in_decide_cancels_and_latches_do8(self) -> None:
        real = self.reg.plan

        def plan(site: Any, attn: Any = None) -> str:
            real(site, attn)  # an 'only' plan opened, then the registry fails
            raise RuntimeError("injected registry failure")

        self.extra.append((self.reg, "plan", {"new": plan}))
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, L11_E5_ORDER)  # DO8 latched: L11's own E5 on the FP8 cast's operands
        self.assertIn(do8.FAMILY + " DISABLED: MX6 decide (dO8 registry plan): RuntimeError: injected registry failure",
                      text)
        self.assertIsNotNone(do8.broken())
        self.assertIsNone(do8.mx6_broken())
        self.assertEqual(do8.mx6_counters()["decided"], 0)
        self.assertIs(self.w3_operand[0], self.fp8_ops[2])
        self.assertEqual(self.state.calls, 51)
        self.assert_closed(site)

    def test_a_raise_while_counting_the_decision_closes_its_plan(self) -> None:
        real = do8._mx6_event

        def event(key: str) -> None:
            if key == "decided":
                raise RuntimeError("injected: the decided counter raised")
            real(key)

        self.extra.append((do8, "_mx6_event", {"new": event}))
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, RuntimeError, out)
        self.assertIn("decided counter raised", str(out))
        self.assertNotIn("s1", self.events)
        self.assertEqual(self.state.calls, 50)
        self.assert_closed(site)

    def test_success_hands_the_plan_over_last(self) -> None:
        seen: List[Any] = []
        real = do8.mx6_decide

        def decide(*a: Any, **k: Any) -> Any:
            seen.append(real(*a, **k))
            return seen[-1]

        self.extra.append((do8, "mx6_decide", {"new": decide}))
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].only_plan is True and seen[0].open is False)  # handed to publish(), not cancelled
        self.assertEqual(self.only_stats(), (1, 1, 0))

    def test_a_raise_after_the_launch_before_the_hand_over_closes_the_plan(self) -> None:
        self.raising_print("launched as " + do8.MX6_KERNEL_NAME)  # L11's once-per-process line, after the record
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, RuntimeError, out)
        self.assertIn("mx6_launch", self.events)
        self.assertNotIn("publish", self.events)
        self.assert_closed(site)

    def test_a_raise_in_the_rescue_line_leaves_the_latches_set(self) -> None:
        for name, arrange, do8_latched in (("MX6 launch rc", lambda: setattr(self, "mx6_rc", 1), False),
                                           ("shared work list", lambda: setattr(self, "wl_fail", RuntimeError("rc=1")),
                                            True)):
            self._fresh()
            arrange()
            self.raising_print(do8.FAMILY_MX6 + " rescue: ")
            out, text, site, ctx = self.run_backward()
            self.assertIsInstance(out, RuntimeError, (name, out))
            self.assertIsNotNone(do8.mx6_broken(), name)  # latched before the line
            self.assertEqual(do8.broken() is not None, do8_latched, name)
            self.assertNotIn("rescue", self.events, name)
            self.assert_closed(site, name)

    def test_a_raise_inside_the_cancel_is_retried_by_the_seam_guard(self) -> None:
        # the decision lets go of its plan only once the cancel went through (release and rescue paths)
        for name, arrange in (("producer declines", lambda: setattr(self, "s1", "none")),
                              ("MX6 launch rc", lambda: setattr(self, "mx6_rc", 1))):
            self._fresh()
            arrange()
            real = do8._cancel_only
            raised: List[int] = []

            def cancel(site: Any, real: Any = real, raised: List[int] = raised) -> None:
                if not raised:
                    raised.append(1)
                    raise RuntimeError("injected: the first cancel raised")
                real(site)

            self.extra.append((do8, "_cancel_only", {"new": cancel}))
            out, text, site, ctx = self.run_backward()
            self.assertIsInstance(out, RuntimeError, (name, out))
            self.assertIn("the first cancel raised", str(out), name)
            self.assertEqual(raised, [1], name)
            self.assertNotIn("rescue", self.events, name)
            self.assertNotIn("fp8_cast", self.events, name)
            self.assert_closed(site, name)

    # ---------------------------------------------------------------- ON branches through the real backward
    def test_a_tag_outside_the_site_set_is_the_base_block(self) -> None:
        for sites, layer in (("top", 0), ("bottom", 1), ("bottom", 2)):
            self._fresh()
            do8.set_mx6_sites_for_testing(sites)
            out, text, site, ctx = self.run_backward(layer=layer)  # a tag the set excludes (e.g. a stale ctx)
            self.assertIsInstance(out, tuple, (sites, out))
            self.assertEqual(self.events, BASE_ORDER, sites)
            self.assertNotIn("NOT armed", text)  # quiet
            self.assertEqual(do8.mx6_counters()["declined"], {"not an": 1}, sites)

    def test_bottom_and_all_take_a_layer_0_call(self) -> None:
        for sites in ("bottom", "all"):
            self._fresh()
            do8.set_mx6_sites_for_testing(sites)
            out, text, site, ctx = self.run_backward(layer=0)
            self.assertIsInstance(out, tuple, (sites, out))
            self.assertEqual(self.events, MX6_ORDER, sites)
            # re-cut v2: the layer-0 record is counted and prints no ACTIVE line ("layer=0 sites=<set>" was its
            # content; the prebuild's ACTIVE line names the site set)
            self.assertEqual(text.count("armed OK: " + do8.FAMILY_MX6 + " ACTIVE"), 0, sites)
            self.assertEqual(do8.mx6_counters()["published"], 1, sites)
            self.assertEqual(self.only_stats(), (1, 1, 0), sites)
            self.assertEqual(self.state.calls, 51, sites)
            self.assertIs(self.w3_operand[0], self.ops6.dout_fp8)

    def test_an_e5_decline_in_decide_runs_the_fp8_chain_once(self) -> None:
        self.extra.append((l11, "e5_decline_reason", {"new": lambda *a, **k: "D9 injected E5 decline"}))
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, L11_E5_ORDER)
        self.assertEqual(len(self.calls["emission_site"]), 1)  # evaluated once (the FP8 chain got dec.site)
        self.assertEqual(do8.counters()["calls"], 1)
        self.assertEqual(self.only_stats(), (0, 0, 0))  # the registry is never asked
        self.assertEqual(do8.mx6_counters()["decided"], 0)
        self.assertEqual(len(self.calls["b1a_plan"]), 1)
        self.assertIs(self.w3_operand[0], self.fp8_ops[2])
        self.assertEqual(self.state.calls, 51)

    def test_no_emission_site_in_decide_runs_l11s_own_e5(self) -> None:
        self.extra.append((do8, "emission_site", {"new": lambda ctx, attn: None}))
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, L11_E5_ORDER)
        self.assertEqual(len(self.calls["emission_site"]), 1)
        self.assertEqual(self.only_stats(), (0, 0, 0))
        self.assertEqual(do8.mx6_counters()["decided"], 0)
        self.assertEqual(self.state.calls, 51)

    def test_a_keep_mask_the_mx6_so_does_not_read_is_the_base_block(self) -> None:
        self.extra.append((do8, "_impl_mx6", {"new": IMPL_MX6 + "+mkb=nib"}))  # the MX6 .so reads NIB, the mask int8
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, BASE_ORDER)
        self.assertEqual(text.count(do8.FAMILY_MX6 + " NOT armed: keep mask format (the MX6 .so reads nib) -> FP8 "
                                    "dual cast + (DO8-ONLY) for this call"), 1)  # the 'only' .so int8: no 'plus' route
        self.assertEqual(sum(do8.mx6_counters()["declined"].values()), 1)

    def test_a_malformed_nib_mask_is_a_preflight_decline_not_a_rescue(self) -> None:
        """Defense in depth (in the model L11's D8 gate rejects such a mask before the MX6 dispatch; the harness stubs
        that gate): a NIB mask the NIB MX6 build cannot take (row stride 512) is declined by the preflight's layout
        check for this call (one NOT armed line, counted) before the producer runs: never a rescue or an MX6 latch."""
        m8 = torch.randint(0, 8, (2 * T, D), generator=torch.Generator().manual_seed(7), dtype=I8)
        bad = lnmd_mask_nib.pack_ref(m8)[::2]  # uint8 [T, 256], row stride 512: is_nib, not layout_ok
        self.assertTrue(lnmd_mask_nib.is_nib(bad) and not lnmd_mask_nib.layout_ok(bad, T))
        self.nib_builds(bad)
        seen = self.spy_mx6_mask()
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(seen, [])
        self.assertNotIn("s1", self.events)  # nothing past the preflight: no producer, no scale-state commit
        self.assertNotIn("rescue", self.events)
        self.assertNotIn("MX6 rescue:", text)
        self.assertIsNone(do8.mx6_broken())  # a per-call decline, never an MX6 latch
        self.assertEqual(text.count(do8.FAMILY_MX6 + " NOT armed: keep mask torch.uint8 (1000, 256) stride (512, 1)"),
                         1, text)
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], c["rescued"], c["latched"], c["declined"]), (0, 0, 0, {"keep mask": 1}))
        self.assert_no_open_plan(site)

    def test_an_int8_mask_with_nib_builds_declines_mx6_then_do8_plus(self) -> None:
        """MX6 + NIB gated, an int8 keep mask (a call the NIB writers did not write, or after their latch) on an MX6
        site while the 'only' .so and its MX6 build read NIB: the preflight declines (one NOT armed line naming the
        'plus' route, counted), the base block runs and DO8 serves the call as 'plus' (run_e5_plus's int8_plus route:
        no 'only' plan, dO8 / E8 emitted, the bf16 d(attn) written); the int8 mask is used as is (no unpack)."""
        self.nib_builds(torch.ones(T, D, dtype=I8))
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, ["decline_reason", "dout_tiled", "fp8_cast", "b1a_plan", "emission_site",
                                       "work_list", "do8_launch:plus", "dwdb_finalize", "publish", "w3"])
        self.assertEqual(text.count(do8.FAMILY_MX6 + " NOT armed: keep mask format (the MX6 .so reads nib): with the "
                                    "NIB 'only' .so, DO8 takes an int8 mask as 'plus' (run_e5_plus's int8_plus route) "
                                    "-> FP8 dual cast + (DO8-ONLY) for this call"), 1, text)
        self.assertEqual(text.count("an int8 keep mask reached DO8 while its 'only' .so reads NIB"), 1)
        self.assertEqual(do8.counters()["mask_int8_plus"], 1)
        self.assertEqual(self.only_stats(), (0, 0, 0))  # the registry was never asked: no 'only' plan
        self.assertTrue(getattr(out[0], self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid)  # a 'plus' record
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], sum(c["declined"].values())), (0, 1))
        self.assertEqual(lnmd_mask_nib.counters()["unpack"], 0)
        self.assertEqual(self.state.calls, 51)
        self.assert_no_open_plan(site)

    def test_fp8_cast_declining_after_a_release_is_the_base_warm_up_path(self) -> None:
        """Not reachable after the preflight (a PUBLISHED site, the controller on, a stable identity); pinned here:
        exactly the shipped warm-up path (the bf16 dgrad, B1a's plan asked again on that dy with the base's
        compute_y, as the base asks it after its own cast; host-only), and no open plan."""
        self.s1 = "none"
        self.extra += [
            (bwd_fp8, "output_projection_dgrad_operands", {"new": self.ev("fp8_cast", None)}),
            (self.thl, "_du_e5m2_plan",
             {"new": self.ev("b1a_plan", lambda *a, **k: None if k.get("compute_y") else self.plan)}),
            (self.thl, "triton_layer_norm_mul_dropout_bwd",
             {"new": self.ev("ln_bwd", lambda **k: (torch.zeros_like(k["x"]), torch.zeros(T, D, dtype=BF),
                                                    torch.zeros(D, dtype=BF), torch.zeros(D, dtype=BF),
                                                    torch.zeros(T, NOUT, dtype=BF)))}),
        ]
        out, text, site, ctx = self.run_backward()
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, ["decline_reason", "dout_tiled", "b1a_plan", "emission_site", "s1", "fp8_cast",
                                       "b1a_plan", "ln_bwd"])
        self.assertEqual([k.get("compute_y") for _, k in self.calls["b1a_plan"]], [False, True])
        self.assertEqual(do8.counters()["calls"], 1)
        self.assertEqual(self.state.calls, 50)
        self.assert_closed(site)
        self.assertEqual(tuple(out[5].shape), (NOUT, D))  # the bf16 wgrad of the warm-up path


# ------------------------------------------------------------------------------------------------ E-RM COMPOSITION
_ERM_UNSET = object()  # make_ctx: E-RM not requested (its forward tag leaves no erm_site on ctx)
# the shipped block's order with E-RM's take: the cast's keyword erm=_erm_take(ctx) right after its t_tiled keyword
BASE_ORDER_ERM = BASE_ORDER[:2] + ["erm_take"] + BASE_ORDER[2:]


class ErmCompositionTest(_BackwardBase):
    """MX6 beside r14's E-RM row (ops/triton/dy_erm.py; triton_hstu_linear's E-RM block): E-RM's site is layer 0's
    out-proj, MX6's shipping set "top". The forward tags are the REAL _l11_mx6_forward_tag and _erm_forward_tag
    (E-RM's prerequisite census and the FP8 NN wgrad flags stubbed on); the backward is the REAL dispatch (harness
    above), the E-RM record a stand-in for the producer's (never consumable here: W3 keeps the shipped operand), and
    every _erm_take is counted (event "erm_take"). Cases: E-RM requested + sites=top (top MX6, bottom E-RM's record
    taken once by the shipped cast); E-RM requested + latched (bottom the FP8 dual cast, top still MX6: critique C1's
    U case, CPU form); sites=bottom / all + E-RM requested or latched (one INERT line, nothing built, nothing tagged);
    a forced E-RM site on an MX6-tagged call (MX6 declines first, then one _erm_take), and the same pair with MX6 off
    or latched (a quiet decline); the FP8 fallback's erm=."""

    def setUp(self) -> None:
        super().setUp()
        from generative_recommenders.ops.triton import dy_erm

        self.erm = dy_erm
        self.addCleanup(self._erm_reset)
        self._erm_reset()

    def _erm_reset(self) -> None:
        self.erm.set_requested_for_testing(None)
        self.erm.reset_for_testing()

    def _fresh(self) -> None:
        super()._fresh()
        self.erm_tag: Any = _ERM_UNSET
        self.extra.append((self.thl, "_erm_take", {"new": self.ev("erm_take", self.thl._erm_take)}))

    def make_ctx(self, site: Any, attn: torch.Tensor, layer: Optional[int]) -> SimpleNamespace:
        ctx = super().make_ctx(site, attn, layer)
        if self.erm_tag is not _ERM_UNSET:
            ctx.erm_site = self.erm_tag
        return ctx

    def forward_tags(self, layer: int) -> SimpleNamespace:
        """The REAL forward tags of one compute output at STUStack layer `layer` (MX6's on the B1a site line, then
        E-RM's on the return): the ctx attributes the backward reads."""
        thl, erm = self.thl, self.erm
        ctx = SimpleNamespace(fp8_y_saved=True, fp8_y_token_major=False, fp8_dout_token_major=True)
        out, weight = torch.zeros(4, D), torch.nn.Parameter(torch.zeros(NOUT, D))
        with contextlib.ExitStack() as st:
            for target, attr, kw in ((thl, "_LNMD_LAYER_TAG", {"new": layer}), (thl, "_FP8_OUT_WGRAD", {"new": True}),
                                     (thl, "_FP8_OUT_WGRAD_IMPL", {"new": "nn"}),
                                     (bwd_fp8, "_helpers_fused_ok", {"return_value": True}),
                                     (erm, "inert_reason", {"return_value": None})):
                st.enter_context(mock.patch.object(target, attr, **kw))
            st.enter_context(_Capture())
            self.assertEqual(thl._l11_mx6_forward_tag(ctx, "B1A-SITE"), "B1A-SITE")
            self.assertIs(thl._erm_forward_tag(ctx, out, weight, True), out)
        return ctx

    def test_erm_requested_top_is_mx6_and_the_bottom_cast_takes_the_record_once(self) -> None:
        self.erm.set_requested_for_testing(True)
        top, bottom = self.forward_tags(1), self.forward_tags(0)
        self.assertEqual((top.l11_mx6_layer, top.erm_site), (1, None))  # E-RM D26: no site above layer 0
        self.assertFalse(hasattr(bottom, "l11_mx6_layer"))  # "top" excludes layer 0
        self.assertIsInstance(bottom.erm_site, self.erm.ErmSite)
        self.erm_tag = top.erm_site
        out, text, site, ctx = self.run_backward(layer=top.l11_mx6_layer)
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, MX6_ORDER)  # MX6 takes no E-RM record (no _erm_take at all)
        self.assertEqual((do8.mx6_counters()["published"], self.state.calls), (1, 51))
        self._fresh()
        rec = SimpleNamespace(used=False, layout="rm")  # the producer's record on E-RM's site (stand-in)
        bottom.erm_site.record = rec
        self.erm_tag = bottom.erm_site
        out, text, site, ctx = self.run_backward(layer=None)
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, BASE_ORDER_ERM)
        self.assertIs(self.calls["fp8_cast"][0][1]["erm"], rec)  # taken once, by the shipped cast
        self.assertIs(ctx.erm_rec, rec)
        self.assertTrue(bottom.erm_site.consumed and bottom.erm_site.record is None)
        self.assertEqual((do8.mx6_counters()["decided"], sum(do8.mx6_counters()["declined"].values())), (0, 0))
        self.assertNotIn("MX6", text)

    def test_erm_latched_bottom_is_the_fp8_dual_cast_and_top_stays_mx6(self) -> None:
        self.erm.set_requested_for_testing(True)
        with _Capture():
            self.erm.set_broken("injected: E-RM latched")
        top, bottom = self.forward_tags(1), self.forward_tags(0)
        self.assertEqual((top.l11_mx6_layer, top.erm_site), (1, None))
        self.assertEqual((getattr(bottom, "l11_mx6_layer", None), bottom.erm_site), (None, None))  # E-RM D0: no site
        self.erm_tag = bottom.erm_site
        out, text, site, ctx = self.run_backward(layer=None)
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, BASE_ORDER_ERM)
        self.assertIsNone(self.calls["fp8_cast"][0][1]["erm"])  # the FP8 dual cast with its own bookkeeping
        self.assertEqual(self.state.calls, 51)
        self._fresh()
        self.erm_tag = top.erm_site
        out, text, site, ctx = self.run_backward(layer=top.l11_mx6_layer)
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, MX6_ORDER)  # an E-RM latch does not move MX6
        self.assertTrue(do8.mx6_enabled())

    def test_bottom_or_all_with_erm_requested_is_inert_and_tags_nothing(self) -> None:
        """Through the REAL prebuild_mx6, under E-RM requested and E-RM requested but latched (the rule reads the
        request, never E-RM's runtime state): one INERT line naming the E-RM rule, nothing built, nothing tagged."""
        for sites in ("bottom", "all"):
            for erm_state in ("requested", "latched"):
                case = (sites, erm_state)
                self._erm_reset()
                self.erm.set_requested_for_testing(True)
                if erm_state == "latched":
                    with _Capture():
                        self.erm.set_broken("injected: E-RM latched")
                self._fresh()
                do8.set_mx6_prebuilt_for_testing(False)  # the real prebuild decides
                do8.set_mx6_sites_for_testing(sites)
                load = mock.MagicMock()
                with mock.patch.object(do8, "_load_mx6", load), _Capture() as cap:
                    self.assertFalse(do8.prebuild_mx6(), case)
                load.assert_not_called()
                self.assertEqual(cap.text.count(do8.FAMILY_MX6 + " INERT: "), 1, (case, cap.text))
                self.assertIn(f"{do8.MX6_SITES_KNOB}={sites} holds layer 0, the out-proj site r14's E-RM row owns "
                              f"({self.erm.KNOB}=1 requested)", cap.text, case)
                self.assertFalse(do8.mx6_enabled(), case)
                tags = [self.forward_tags(layer) for layer in (0, 1)]
                self.assertEqual([hasattr(t, "l11_mx6_layer") for t in tags], [False, False], case)
                if erm_state == "requested":
                    self.assertIsInstance(tags[0].erm_site, self.erm.ErmSite)  # E-RM keeps its site
                else:
                    self.assertIsNone(tags[0].erm_site, case)  # a latched E-RM makes no site; MX6 still INERT

    def test_a_forced_erm_site_on_an_mx6_call_declines_then_one_erm_take(self) -> None:
        self.erm.set_requested_for_testing(True)
        erm_site = self.erm.ErmSite(4242, torch.device("cpu"))
        rec = SimpleNamespace(used=False, layout="rm")
        erm_site.record = rec
        self.erm_tag = erm_site
        pre: List[int] = []
        self.extra.append((do8, "mx6_preflight", {"new": lambda *a, **k: pre.append(1)}))
        out, text, site, ctx = self.run_backward(layer=1)  # a forced tag: unreachable under the INERT rule
        self.assertIsInstance(out, tuple, out)
        self.assertEqual(self.events, BASE_ORDER_ERM)
        self.assertEqual(pre, [])  # declined before the preflight
        self.assertEqual(len(self.calls["erm_take"]), 1)
        self.assertIs(self.calls["fp8_cast"][0][1]["erm"], rec)
        self.assertTrue(erm_site.consumed and erm_site.record is None)
        self.assertEqual(text.count(do8.FAMILY_MX6 + " NOT armed: E-RM owns this call (the forward's E-RM site tag)"), 1)
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], c["declined"]), (0, {"E-RM owns": 1}))
        self.assertEqual(self.only_stats(), (1, 1, 0))  # DO8's own 'only' plan, published by the shipped block
        self.assert_no_open_plan(site)

    def test_a_stale_mx6_tag_beside_an_erm_site_with_mx6_off_or_latched_prints_nothing(self) -> None:
        """A synthetic ctx holding both an MX6 layer tag and E-RM's site while MX6 is NOT armed (the knob off, or MX6
        latched; the forward tags MX6 only while armed, so no real call carries both): the seam declines quietly (no
        MX6 line, as the preflight's "latched or not prebuilt"), and the shipped cast takes the record once."""
        self.erm.set_requested_for_testing(True)

        def latch() -> None:
            with _Capture():
                do8._disable_mx6("injected: MX6 latched")

        for name, arrange in (("knob off", lambda: self.arm_mx6(False)), ("latched", latch)):
            self._fresh()
            arrange()
            self.assertFalse(do8.mx6_enabled(), name)
            erm_site = self.erm.ErmSite(4242, torch.device("cpu"))
            rec = SimpleNamespace(used=False, layout="rm")
            erm_site.record = rec
            self.erm_tag = erm_site
            out, text, site, ctx = self.run_backward(layer=0)  # the stale tag
            self.assertIsInstance(out, tuple, (name, out))
            self.assertEqual(self.events, BASE_ORDER_ERM, name)
            self.assertNotIn("MX6", text, name)
            self.assertEqual(len(self.calls["erm_take"]), 1, name)
            self.assertIs(self.calls["fp8_cast"][0][1]["erm"], rec, name)
            self.assertTrue(erm_site.consumed and erm_site.record is None, name)
            c = do8.mx6_counters()
            self.assertEqual((c["decided"], c["declined"]), (0, {"E-RM owns": 1}), name)  # counted, quietly
            self.assert_no_open_plan(site)

    def test_the_fp8_fallback_passes_erm_like_the_shipped_cast(self) -> None:
        self.erm.set_requested_for_testing(True)
        for name, arrange, tail in (
            ("producer declines", lambda: setattr(self, "s1", "none"),
             ["s1", "erm_take", "fp8_cast", "work_list", "do8_launch:only"]),
            ("registry plus", lambda: setattr(self, "facts", _Facts("calibration step: BF16 q/k")),
             ["erm_take", "fp8_cast", "work_list", "do8_launch:plus"]),
        ):
            self._fresh()
            arrange()
            self.erm_tag = None  # E-RM requested: the top layer's forward left ctx.erm_site None (no site above 0)
            out, text, site, ctx = self.run_backward(layer=1)
            self.assertIsInstance(out, tuple, (name, out))
            self.assertEqual(self.events, ["decline_reason", "dout_tiled", "b1a_plan", "emission_site"] + tail
                             + ["dwdb_finalize", "publish", "w3"], name)
            self.assertEqual(len(self.calls["erm_take"]), 1, name)  # the mirror's, once
            k = self.calls["fp8_cast"][0][1]
            self.assertEqual(sorted(k), ["erm", "t_tiled", "token_major"], name)
            self.assertIsNone(k["erm"], name)
            self.assertFalse(hasattr(ctx, "erm_rec"), name)  # no site: nothing kept on ctx
            self.assertEqual(self.state.calls, 51, name)
            self.assert_no_open_plan(site)


if __name__ == "__main__":
    unittest.main()
