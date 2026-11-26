#!/usr/bin/env python3
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

"""NIB writers and wiring (r16 iteration 54, gin item (94), DLRM_HSTU_FP8_LNMD_MASK_NIB, default OFF).

The LN*U*dropout keep mask of the two full-length STU layers is written as uint8 [N, 256] (ops/lnmd_mask_nib.py
layout A) by _ln_mul_dropout_fwd_maskgen_nib / _ln_mul_dropout_fwd_maskgen_pk_nib (ops/triton/
triton_hstu_linear.py, module end) when _lnmd_mask_nib_why_not arms the call, and is read only by the DO8 NIB
build; every other consumer receives lnmd_mask_nib.as_int8 (the exact unpack).

CPU, torch only (always run):
  * the contract module: pack_ref / unpack round trip and layout A against a pure-python reference (every
    (column 8l+k, column 8l+4+k) value pair), bits 3 / 7 clear, is_nib / layout_ok, as_int8 (no-op on int8 and
    None, exact + cached + counted + one line on NIB);
  * structure of triton_hstu_linear.py (AST, no import): the fail-closed census of every random_mask consumer in
    HSTUComputeOutputFunction.backward (raw only into L11's D8 gate, B1a's plan, DO8 and _l11_du_e5m2_or_none,
    which unpacks inside; as_int8 into L11 run, B1a v3 and the Triton backward), the unpack at the top of the
    int8 entry functions, the two forward hook lines, the forward's out-proj context call, and the six NIB
    kernels being the last @triton.jit functions of the module (appended at its end);
  * gin / bootstrap: one Optional[bool] parameter bound by one env-wins _bind_bool, exactly one gin binding
    "= False", a documented item block (lines <= 80 columns), no repeated binding; with gin importable, child
    interpreters run train_ranker's Phase 1 (unset -> "0", pre-set "1" -> "1", pre-set "0" -> "0");
  * fp8_warmup.py lists both NIB kernels in _JIT_KERNELS_S11 and runs item L4n only with the knob on; L4n asks for
    the startup byte check on L4's rows only (not _ROWS_XL), latches NIB after the item on a mismatch and stops;
    _LNMD_MASK_NIB_BROKEN is in its _KEEP_GLOBALS (the latch outlives the warmup's restore);
  * clean_exit.EXIT_HOOKS lists _lnmd_mask_nib_final (review F1: os._exit skips atexit).
CPU, triton_hstu_linear importable (needs triton; no GPU launch; skipped otherwise):
  * aliases, (keep_min16, kk2), the predicate (knob off touches nothing; every decline reason in order, the
    out-proj context consumed once), _lnmd_mask_nib_note_call, decline / armed / latch lines and counters;
  * the hook and the production entry _triton_layer_norm_mul_dropout_fwd_impl with mocked launchers: knob off ->
    exactly today's launch sequence and int8 mask; predicate forced to None -> the NIB launch only (sh / pk by
    MGPK's gate), no int8 mask; a raising NIB launch -> latch, DISABLED line, today's path for that call;
  * the int8 entries of the backward receive the exact unpack (Triton v2 / v1, B1a v3, L11 run_e5) and
    _du_e5m2_plan accepts the NIB layout; the FP8 warmup's launch helper launches both writers, and with compare
    runs the int8 kernel each replaces and reports (never latches) any differing byte (review R1-F5);
  * lnmd_mask_nib_latch (one DISABLED line, idempotent) and the exit counters line (knob on only, once).
GPU (gfx950 + Triton; skipped elsewhere):
  * sh-NIB vs the shipped maskgen and pk-NIB vs MGPK at N in (1, 7, 4097, 65536) x p in (1e-4, 0.1, 0.5) x 2
    seeds into two junk fills: y / mean / rstd byte-identical, packed == pack_ref(int8), unpack(packed) == int8;
  * the hook end to end through the production entry with the predicate forced to None (layer tags 0 / 1 with
    MGPK's list "1": sh-NIB / pk-NIB), and a raising NIB launch falling back to today's bytes.
  (the GPU tests above run the ship17 (111) default, DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3=1: the V3 pk writer.)
Ship17 (111) _pk_nib V3 (DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3, code default OFF, gin ON, env "0" = kill switch):
  * CPU, AST: the V3 body fills exactly the lines of the old body (same decorator / def / parameter lines, the same
    span as the old body kept verbatim as _ln_mul_dropout_fwd_maskgen_pk_nib_r1), so _c32_nib (the next kernel)
    keeps its distance; the (111) block's @triton.jit functions follow the NIB kernels and precede the E-RM block;
    the launch helper's kernel line asks _lnmd_mask_nib_pk_kernel; one Optional[bool] bootstrap parameter with one
    env-wins _bind_bool; one gin binding "= True" in a documented item block (<= 80 columns) right after (94)'s;
    the documented lines are the code's; with gin, a child Phase 1: unset -> "1", pre-set "0" / "1" kept, a gin
    False binding -> "0";
  * CPU, triton_hstu_linear importable: the selector (unset -> r1, no line; "1" -> V3 + one armed line; "0" or any
    other value -> r1 + one OFF line; "1" with D != BLOCK_D -> r1 + one NOT armed line; each mark holds its line)
    and the launch helper launching the picked kernel with the same arguments (sh-NIB unaffected);
  * main's FV-2 ruling, every (111) line once per process per rank BEFORE RUN_START: both marks are in fp8_warmup's
    _KEEP_GLOBALS and _nib_items calls _pk_v3_reprint after every L4n item (AST); with fp8_warmup importable, a
    real _Runner item drops the selector's line and _pk_v3_reprint prints it once, outside the capture (ARMED, OFF,
    NOT armed; knob absent: nothing); end to end on the CPU (mocked kernels) the warmup's state guards around item
    L4n (take, _nib_items, restore, VERIFY digest) print exactly one (111) line, after the first item, for "1", the
    env kill switch "0" and a gin False binding (none when absent), the marks survive the restore, and the first
    training launch through the production entry prints no (111) line (without the _KEEP_GLOBALS entries it would:
    the FV-2 finding, reproduced);
  * GPU: V3 vs MGPK's int8 kernel and vs r1 at every last-group size (N 1..17, 23, 31, 69, 4097, 65536) and the
    production N 2,301,929: y / mean / rstd / NIB bytes identical, 8 guard rows untouched.
"""

import ast
import contextlib
import hashlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import torch

from generative_recommenders.ops import lnmd_mask_nib as nib

_HAS_GIN = importlib.util.find_spec("gin") is not None

D = 512
EPS = 1e-6
KNOB = nib.KNOB
INV = 23.893333029513894
SEEDS2 = (900450186894289455, 1441151880758558720)  # i64; % 16 = 15 and 0
_OPS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GR = os.path.dirname(_OPS)
_ROOT = os.path.dirname(_GR)
_THL = os.path.join(_OPS, "triton", "triton_hstu_linear.py")
_TRAIN = os.path.join(_GR, "dlrm_v4", "train")
_GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_WARM = os.path.join(_TRAIN, "fp8_warmup.py")
_NIB_KERNELS = (
    "_mg_nib",
    "_row_c32_pk_nib",
    "_ln_mul_dropout_fwd_maskgen_pk_nib",
    "_c32_nib",
    "_row_c32_nib",
    "_ln_mul_dropout_fwd_maskgen_nib",
)
# ship17 (111) _pk_nib V3: the knob, and the @triton.jit functions of its block (inserted right after the last NIB
# kernel, before the NIB host side: the last @triton.jit functions of the module)
PK_V3_KNOB = "DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3"
_PK_V3_BLOCK = (
    "_mg_bytemask_n_v3",
    "_mg_j2_v3",
    "_mg_split8_v3",
    "_mg_lso_v3",
    "_tp_word_v3",
    "_tp_nib_v3",
    "_tp_philox_v3",
    "_row_c32_pk_nib_v3",
    "_ln_mul_dropout_fwd_maskgen_pk_nib_r1",
)
_PK_V3_GLOBALS = ("_LNMD_MASK_NIB_PK_V3_ARMED_LOGGED", "_LNMD_MASK_NIB_PK_V3_OFF_LOGGED")
_NIB_GLOBALS = _PK_V3_GLOBALS + (
    "_LNMD_MASK_NIB_BROKEN",
    "_LNMD_MASK_NIB_ARMED_LOGGED",
    "_LNMD_MASK_NIB_SH_LOGGED",
    "_LNMD_MASK_NIB_PK_LOGGED",
    "_LNMD_MASK_NIB_DECLINES_LOGGED",
    "_LNMD_MASK_NIB_N_ARMED",
    "_LNMD_MASK_NIB_N_SH",
    "_LNMD_MASK_NIB_N_PK",
    "_LNMD_MASK_NIB_N_DECLINED",
    "_LNMD_MASK_NIB_N_FAILED",
    "_LNMD_MASK_NIB_ATEXIT",
    "_LNMD_MASK_NIB_FINAL_DONE",
    "_LNMD_MASK_NIB_CALL_ID",
    "_LNMD_MASK_NIB_CALL_WHY",
    "_LNMD_FWD_MASKGEN_PK_ENABLED",
    "_LNMD_FWD_MASKGEN_PK_ARMED_LOGGED",
    "_LNMD_FWD_MASKGEN_PK_DECLINE_LOGGED",
    "_LNMD_FWD_MASKGEN_PK_BROKEN",
    "_LNMD_FWD_MASKGEN_PK_LAYERS",
    "_LNMD_FWD_MASKGEN_PK_GATED_LOGGED",
    "_LNMD_LAYER_TAG",
    "_LNMD_FWD_MASKGEN_ENABLED",
    "_LNMD_FWD_MASKGEN_ARMED_LOGGED",
    "_LNMD_FWD_MASKGEN_ARCH_LOGGED",
    "_LNMD_FWD_V3_ENABLED",
    "_MASK16_REPORTED",
)


def _thl():
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    return thl


def _importable() -> bool:
    try:
        _thl()
        return True
    except Exception:
        return False


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None or not _importable():
        return False
    try:
        return _thl()._gcn_arch_name(torch.cuda.current_device()).startswith("gfx950")
    except Exception:
        return False


def _rand_mask(n: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 8, (n, D), generator=gen, dtype=torch.int8)


def _pack_py(row: List[int]) -> List[int]:
    """Layout A for one row, pure python: lane l's dword at byte 4l; byte k = col 8l+k (bits 0-2) | col 8l+4+k << 4."""
    out = [0] * (D // 2)
    for lane in range(D // 8):
        for k in range(4):
            out[4 * lane + k] = (row[8 * lane + k] & 7) | ((row[8 * lane + 4 + k] & 7) << 4)
    return out


# ------------------------------------------------------------------------------------------------ CPU: contract
class ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        nib.reset_for_testing()

    def tearDown(self) -> None:
        nib.reset_for_testing()

    def test_round_trip_and_layout(self) -> None:
        g = torch.Generator().manual_seed(54)
        for n in (1, 7, 64, 333):
            m = _rand_mask(n, g)
            p = nib.pack_ref(m)
            self.assertEqual((p.dtype, tuple(p.shape), p.is_contiguous()), (torch.uint8, (n, D // 2), True))
            self.assertTrue(torch.equal(nib.unpack(p), m), n)
            self.assertEqual(int((p & 0x88).sum()), 0, n)  # bits 3 and 7 are never set
            for r in sorted({0, n // 2, n - 1}):
                self.assertEqual(p[r].tolist(), _pack_py(m[r].tolist()), (n, r))
            u = nib.unpack(p)
            self.assertEqual((u.dtype, tuple(u.stride())), (torch.int8, (D, 1)))

    def test_every_field_pair_at_every_lane(self) -> None:
        # one row per (a, b): every column 8l+k holds a and every column 8l+4+k holds b, plus the transposed row
        rows = []
        for a in range(8):
            for b in range(8):
                rows.append([a if (c % 8) < 4 else b for c in range(D)])
                rows.append([(a + c) % 8 if (c % 8) < 4 else (b + 3 * c) % 8 for c in range(D)])
        m = torch.tensor(rows, dtype=torch.int8)
        p = nib.pack_ref(m)
        for r in range(m.shape[0]):
            self.assertEqual(p[r].tolist(), _pack_py(rows[r]), r)
        self.assertTrue(torch.equal(nib.unpack(p), m))

    def test_is_nib_and_layout_ok(self) -> None:
        m8 = torch.zeros((4, D), dtype=torch.int8)
        p = nib.pack_ref(m8)
        self.assertFalse(nib.is_nib(m8))
        self.assertTrue(nib.is_nib(p))
        self.assertTrue(nib.layout_ok(p, 4))
        self.assertFalse(nib.layout_ok(p, 5))
        self.assertFalse(nib.is_nib(None))
        self.assertFalse(nib.is_nib(torch.empty(0, dtype=torch.bfloat16)))  # the no-mask placeholder
        view = torch.zeros((4, D), dtype=torch.uint8)[:, : D // 2]  # NIB shape, row stride 512
        self.assertTrue(nib.is_nib(view))
        self.assertFalse(nib.layout_ok(view, 4))

    def test_as_int8(self) -> None:
        g = torch.Generator().manual_seed(7)
        m8 = _rand_mask(9, g)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertIs(nib.as_int8(m8, "unit"), m8)
            self.assertIsNone(nib.as_int8(None, "unit"))
            placeholder = torch.empty(0, dtype=torch.bfloat16)
            self.assertIs(nib.as_int8(placeholder, "unit"), placeholder)
        self.assertEqual(buf.getvalue(), "")
        self.assertEqual(nib.counters()["unpack"], 0)
        p = nib.pack_ref(m8)
        with contextlib.redirect_stdout(buf):
            out = nib.as_int8(p, "unit")
            again = nib.as_int8(p, "unit")
        self.assertTrue(torch.equal(out, m8))
        self.assertIs(again, out)  # cached on the NIB tensor: one unpack per mask
        self.assertEqual(nib.counters()["unpack"], 1)
        self.assertEqual(buf.getvalue().count("exact unpack to int8 for unit"), 1)

    def test_requested_exactly_one(self) -> None:
        for value, want in (("1", True), ("0", False), ("", False), ("true", False), ("2", False)):
            with mock.patch.dict(os.environ, {KNOB: value}):
                self.assertEqual(nib.requested(), want, value)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(KNOB, None)
            self.assertFalse(nib.requested())


# ------------------------------------------------------------------------------------------------ CPU: structure
def _tree() -> ast.Module:
    with open(_THL) as f:
        return ast.parse(f.read())


def _func(tree: ast.AST, name: str, cls: Optional[str] = None) -> ast.FunctionDef:
    scope: Any = tree
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _dotted(node.value) + "." + node.attr
    return "?"


def _mask_args(call: ast.Call) -> List[str]:
    """How `call` passes random_mask: 'raw' (the name) or 'as_int8' (lnmd_mask_nib.as_int8(random_mask, str))."""
    out = []
    for a in list(call.args) + [k.value for k in call.keywords]:
        if isinstance(a, ast.Name) and a.id == "random_mask":
            out.append("raw")
        elif (
            isinstance(a, ast.Call)
            and _dotted(a.func) == "lnmd_mask_nib.as_int8"
            and a.args
            and isinstance(a.args[0], ast.Name)
            and a.args[0].id == "random_mask"
            and len(a.args) == 2
            and isinstance(a.args[1], ast.Constant)
            and isinstance(a.args[1].value, str)
        ):
            out.append("as_int8")
    return out


class WiringStructureTest(unittest.TestCase):
    """AST of ops/triton/triton_hstu_linear.py; no import (runs without triton)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = _tree()

    def test_backward_consumer_census(self) -> None:
        bwd = _func(self.tree, "backward", "HSTUComputeOutputFunction")
        # r16 MX6 (module end): _l11_mx6_backward decides with the raw mask (the MX6 .so reads the 'only' .so's
        # format, NIB included: MX6_NIB_GATED) and hands it on to the consumers classified here (its own census:
        # ops/tests/l11_do8_mx6_seam_test.py, StaticSeamTest.test_mx6_helpers_keep_mask_census)
        raw_ok = {"l11_outproj_lnmd.decline_reason", "_du_e5m2_plan", "_l11_do8_e5_or_none", "_l11_du_e5m2_or_none",
                  "_l11_mx6_backward"}
        unpacked = {"l11_outproj_lnmd.run", "_triton_layer_norm_mul_dropout_bwd_du_e5m2",
                    "triton_layer_norm_mul_dropout_bwd"}
        seen: Dict[str, List[str]] = {}
        for node in ast.walk(bwd):
            if isinstance(node, ast.Call) and _dotted(node.func) != "lnmd_mask_nib.as_int8":
                kinds = _mask_args(node)
                if kinds:
                    seen.setdefault(_dotted(node.func), []).extend(kinds)
        self.assertEqual(set(seen), raw_ok | unpacked, seen)  # a new consumer must be classified here
        for name in raw_ok:
            self.assertEqual(set(seen[name]), {"raw"}, name)
        for name in unpacked:
            self.assertEqual(set(seen[name]), {"as_int8"}, name)
        self.assertEqual(seen["_du_e5m2_plan"], ["raw", "raw"])  # the L11 path's plan and the unfused path's plan

    def test_int8_entry_functions_unpack_first(self) -> None:
        for name in ("_triton_layer_norm_mul_dropout_bwd_impl", "_triton_layer_norm_mul_dropout_bwd_du_e5m2"):
            fn = _func(self.tree, name)
            assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "random_mask" for t in n.targets)]
            self.assertEqual(len(assigns), 1, name)
            (a,) = assigns
            self.assertEqual(_mask_args(ast.Call(func=ast.Name("f"), args=[a.value], keywords=[])), ["as_int8"])
            uses = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "random_mask"
                    and isinstance(n.ctx, ast.Load) and n.lineno != a.lineno]
            self.assertTrue(uses and min(uses) > a.lineno, name)  # converted before any other read
        fn = _func(self.tree, "_l11_du_e5m2_or_none")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _dotted(n.func) == "l11_outproj_lnmd.run_e5"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(_mask_args(calls[0]), ["as_int8"])

    def test_du_e5m2_plan_layout_accepts_nib(self) -> None:
        fn = _func(self.tree, "_du_e5m2_plan")
        names = {_dotted(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
        self.assertIn("lnmd_mask_nib.is_nib", names)
        self.assertIn("lnmd_mask_nib.layout_ok", names)

    def test_forward_hook_lines(self) -> None:
        fn = _func(self.tree, "_triton_layer_norm_mul_dropout_fwd_impl")
        walrus = [n for n in ast.walk(fn) if isinstance(n, ast.NamedExpr) and n.target.id == "_nib"]
        self.assertEqual(len(walrus), 1)
        self.assertEqual(_dotted(walrus[0].value.func), "_lnmd_fwd_maskgen_nib_try")
        hooks = [n for n in ast.walk(fn) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
                 and any(isinstance(v, ast.Call) and _dotted(v.func) == "_lnmd_fwd_maskgen_pk_try"
                         for v in n.test.values)]
        self.assertEqual(len(hooks), 1)
        (hook,) = hooks
        first = hook.test.values[0]
        self.assertTrue(isinstance(first, ast.Compare) and isinstance(first.ops[0], ast.IsNot)
                        and getattr(first.left, "id", None) == "_nib")
        self.assertEqual(hook.lineno, walrus[0].lineno + 5)  # the MGPK hook line, 5 lines below the allocation
        ret = hook.body[0]
        self.assertTrue(isinstance(ret, ast.Return))
        self.assertEqual([getattr(e, "id", None) for e in ret.value.elts], ["y", "mean", "rstd", "random_mask"])
        fwd = _func(self.tree, "forward", "HSTUComputeOutputFunction")
        notes = [n for n in ast.walk(fwd) if isinstance(n, ast.Call) and _dotted(n.func) == "_lnmd_mask_nib_note_call"]
        self.assertEqual(len(notes), 1)
        self.assertEqual([getattr(a, "id", None) for a in notes[0].args], ["attn", "output_weight", "need_bwd"])
        fp8_fwd = [n for n in ast.walk(fwd) if isinstance(n, ast.Call)
                   and _dotted(n.func) == "triton_layer_norm_mul_dropout_fwd"
                   and any(k.arg == "output_fp8" and getattr(k.value, "value", None) is True for k in n.keywords)]
        self.assertEqual(len(fp8_fwd), 1)
        self.assertTrue(0 < fp8_fwd[0].lineno - notes[0].lineno <= 3)

    def test_nib_kernels_are_appended_last(self) -> None:
        jit = []
        for n in self.tree.body:
            if isinstance(n, ast.FunctionDef) and any(_dotted(d) == "triton.jit" for d in n.decorator_list):
                jit.append(n.name)
        # ship17 (111): the NIB kernels are followed only by the (111) block's (right after the last NIB kernel)
        tail = _NIB_KERNELS + _PK_V3_BLOCK
        self.assertEqual(tuple(jit[-len(tail):]), tail)
        self.assertTrue(all(name not in tail for name in jit[: -len(tail)]))
        with open(_THL) as f:
            src = f.read()
        banner = src.index("# DLRM_HSTU_FP8_LNMD_MASK_NIB (\"NIB\"")
        self.assertGreater(banner, src.index("def _lnmd_fwd_maskgen_pk_try("))  # after the MGPK block


# ------------------------------------------------------------------------------------------------ CPU: gin
_HEAD = "# (94) NIB "


def _gin_lines() -> List[str]:
    with open(_GIN) as f:
        return f.read().splitlines()


def _item_block(lines: List[str]) -> List[str]:
    i_bind = next(i for i, ln in enumerate(lines) if ln.startswith(f"apply_env_bootstrap.{KNOB} "))
    i_head = max(i for i in range(i_bind) if lines[i].startswith(_HEAD))
    return lines[i_head:i_bind]


class GinBootstrapTest(unittest.TestCase):
    def test_bootstrap_one_parameter_one_env_wins_bind(self) -> None:
        with open(_BOOT) as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(names.count(KNOB), 1)
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        arg = next(a for a in fn.args.args if a.arg == KNOB)
        self.assertEqual(ast.unparse(arg.annotation), "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[KNOB], ast.Constant) and defaults[KNOB].value is None)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == KNOB]
        self.assertEqual(len(calls), 1)
        self.assertEqual(getattr(calls[0].func, "id", None), "_bind_bool")
        self.assertEqual([getattr(a, "id", None) for a in calls[0].args[1:]], [KNOB])

    def test_gin_exactly_one_binding_true(self) -> None:
        # gin flip (r16 iteration 55: Leg I v3 KEEP, job 70620): the shipped binding is ON; env 0 is the kill switch.
        lines = _gin_lines()
        pat = re.compile(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % KNOB)
        self.assertEqual([m.group(1) for m in map(pat.match, lines) if m], ["True"])
        names = [m.group(1) for m in (re.match(r"^\s*([\w./]+)\s*=", ln) for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])

    def test_item_block_documents_the_knob(self) -> None:
        block = _item_block(_gin_lines())
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = " ".join(ln[1:].strip() for ln in block)
        for marker in ("(94) NIB", "r16", "code default OFF", "WHAT:", "DISPATCH", "WHY:",
                       "NUMERICS: BIT-IDENTICAL", "FALLBACK:", "WARMUP:", "ARMED", "Must be ABSENT:",
                       "armed OK: LN*U*dropout keep mask NIB ACTIVE (DLRM_HSTU_FP8_LNMD_MASK_NIB, item "
                       "(94))", '"LN*U*dropout keep mask NIB DISABLED"',
                       '"LN*U*dropout keep mask NIB NOT armed"', "exact unpack",
                       f"Kill switch: {KNOB}=0", "pre-set value wins", "+mkb=nib", "item L4n",
                       "_ln_mul_dropout_fwd_maskgen_nib", "_ln_mul_dropout_fwd_maskgen_pk_nib"):
            self.assertIn(marker, text)
        # the item sits next to (DO8) / (DO8-ONLY)
        lines = _gin_lines()
        i_bind = next(i for i, ln in enumerate(lines) if ln.startswith(f"apply_env_bootstrap.{KNOB} "))
        only = "apply_env_bootstrap.DLRM_HSTU_FP8_L11_DO8_ONLY "
        i_only = next(i for i, ln in enumerate(lines) if ln.startswith(only))
        self.assertLess(i_only, i_bind)
        # ship15.1 put item (89) DO8 DIET right after (DO8-ONLY): the item follows the nearer of the two bindings
        diet = "apply_env_bootstrap.DLRM_HSTU_FP8_L11_DO8_DIET "
        i_prev = max([i_only] + [i for i, ln in enumerate(lines) if ln.startswith(diet) and i < i_bind])
        self.assertLess(i_bind - i_prev, 100)

    def test_documented_lines_are_the_code_lines(self) -> None:
        with open(_THL) as f:
            src = f.read()
        for piece in ('_LNMD_MASK_NIB_FAMILY = "LN*U*dropout keep mask NIB"', '_LNMD_MASK_NIB_ITEM = "(94)"',
                      'f"armed OK: {_LNMD_MASK_NIB_FAMILY} ACTIVE ({_LNMD_MASK_NIB_KNOB}, item '
                      '{_LNMD_MASK_NIB_ITEM}): "',
                      '"item (28)(e)\'s maskgen writes ', 'f"{_LNMD_MASK_NIB_FAMILY} DISABLED ({_LNMD_MASK_NIB_KNOB}): ',
                      'f"{_LNMD_MASK_NIB_FAMILY} NOT armed for this call ({_LNMD_MASK_NIB_KNOB}): {why}',
                      'f"[lnmd-mask-nib] counters: armed=', 'f"[lnmd-mask-nib] first launch of {kernel} '):
            self.assertIn(piece, src)

    @unittest.skipUnless(_HAS_GIN, "gin not importable")
    def test_phase1_bootstrap_child(self) -> None:
        code = (
            "import os, sys, gin\n"
            "from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot\n"
            "gin.parse_config_file(sys.argv[1], skip_unknown=True)\n"
            "boot.apply_env_bootstrap()\n"
            f"print('VALUE=' + repr(os.environ.get({KNOB!r})))\n"
        )
        # gin flip: the binding exports "1"; a pre-set env value wins (env 0 is the kill switch).
        for pre, want in ((None, "'1'"), ("1", "'1'"), ("0", "'0'")):
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=os.pathsep.join([_ROOT] + [p for p in sys.path if p]))
            env.pop(KNOB, None)
            if pre is not None:
                env[KNOB] = pre
            out = subprocess.run([sys.executable, "-B", "-c", code, _GIN], env=env, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            self.assertIn(f"VALUE={want}", out.stdout, (pre, out.stdout[-500:]))


class WarmupSourceTest(unittest.TestCase):
    def test_warmup_lists_and_gates_l4n(self) -> None:
        with open(_WARM) as f:
            src = f.read()
        tree = ast.parse(src)
        s11 = next(n for n in tree.body if isinstance(n, ast.AnnAssign)
                   and getattr(n.target, "id", None) == "_JIT_KERNELS_S11")
        names = [e.elts[1].value for e in s11.value.elts]
        for k in ("_ln_mul_dropout_fwd_maskgen_nib", "_ln_mul_dropout_fwd_maskgen_pk_nib",
                  "_ln_mul_dropout_fwd_maskgen_pk_nib_r1"):  # ship17 (111): the knob-off pk writer too
            self.assertIn(k, names)
        fn = _func(tree, "_nib_items")
        first = fn.body[1]  # after the docstring: the knob check that returns before anything else
        self.assertTrue(isinstance(first, ast.Try))
        self.assertIn("requested", ast.unparse(first))
        self.assertIn("_nib_items(r, _thl, ol, ln_wb, Do, ratio, training, fuse, inv, big_rows)", src)

    def test_warmup_keeps_the_nib_latch_across_its_restore(self) -> None:
        """review R1-F5: a latch set by L4n's startup byte check must survive the warmup's scalar restore."""
        with open(_WARM) as f:
            tree = ast.parse(f.read())
        keep = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == "_KEEP_GLOBALS" for t in n.targets))
        pairs = set()
        for node in ast.walk(keep.value):
            if isinstance(node, ast.Tuple) and len(node.elts) == 2 and isinstance(node.elts[1], ast.Constant):
                first = node.elts[0]
                if isinstance(first, ast.BinOp) and isinstance(first.right, ast.Constant):
                    pairs.add((first.right.value, node.elts[1].value))
        self.assertIn(("triton_hstu_linear", "_LNMD_MASK_NIB_BROKEN"), pairs)
        # ship17 (111) (main's FV-2 ruling): the pk writer selector's two line marks are kept too
        self.assertEqual(sorted(p for p in pairs if p[0] == "triton_hstu_linear"),
                         sorted([("triton_hstu_linear", "_LNMD_MASK_NIB_BROKEN")]
                                + [("triton_hstu_linear", k) for k in _PK_V3_GLOBALS]))


class ExitHookTest(unittest.TestCase):
    def test_exit_hook_lists_the_final_line(self) -> None:
        """review F1: the default teardown ends in os._exit (skips atexit): the exit counters line must be an
        EXIT_HOOKS entry (clean_exit_test checks every atexit registration against EXIT_HOOKS)."""
        from generative_recommenders.dlrm_v4.train import clean_exit

        self.assertIn(("generative_recommenders.ops.triton.triton_hstu_linear", "_lnmd_mask_nib_final"),
                      clean_exit.EXIT_HOOKS)
        with open(_THL) as f:
            src = f.read()
        self.assertEqual(src.count("atexit.register(_lnmd_mask_nib_final)"), 1)
        self.assertEqual(src.count("atexit.register("), 1)  # the NIB block's only exit-time registration


def _fp8_warmup():
    from generative_recommenders.dlrm_v4.train import fp8_warmup

    return fp8_warmup


def _warmup_importable() -> bool:
    try:
        _fp8_warmup()
        return True
    except Exception:
        return False


class _FakeRunner:
    def __init__(self) -> None:
        self.device = torch.device("cuda", 0)  # never touched: the fake bufs allocate on the CPU
        self.runs: List[Tuple[str, str, str, Any]] = []
        self.skips: List[Tuple[str, str, str]] = []
        self.bufs = types.SimpleNamespace(act=lambda shape, dtype, big=False: torch.zeros(shape, dtype=dtype))

    def run(self, tag: str, name: str, detail: str, fn: Any, all_ranks: bool = False) -> None:
        self.runs.append((tag, name, detail, fn))

    def skip(self, tag: str, name: str, reason: str) -> None:
        self.skips.append((tag, name, reason))


@unittest.skipUnless(_warmup_importable(), "dlrm_v4/train/fp8_warmup.py not importable (gin / torch missing)")
class WarmupItemsTest(unittest.TestCase):
    """Item L4n (_nib_items) with a fake runner and a fake triton_hstu_linear: nothing with the knob off; one skip
    line where item (28)(e) cannot arm; otherwise one item per (large-row size, seed) calling the warmup launch."""

    def _items(self, env: Dict[str, str], thl: Any, big_rows: List[int], training: bool = True,
               fuse: bool = True) -> _FakeRunner:
        r = _FakeRunner()
        ol = types.SimpleNamespace(_output_norm_weight=torch.ones(D, dtype=torch.bfloat16))

        def ln_wb(width: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.ones(width, dtype=torch.bfloat16), torch.zeros(width, dtype=torch.bfloat16)

        with mock.patch.dict(os.environ, env):
            for k in (KNOB, "DLRM_HSTU_FP8_LNMD_FWD_MASKGEN", "DLRM_HSTU_FP8_MASK16"):
                if k not in env:
                    os.environ.pop(k, None)
            _fp8_warmup()._nib_items(r, thl, ol, ln_wb, D, 0.1, training, fuse, 64.0, big_rows)
        return r

    def test_l4n(self) -> None:
        calls: List[Dict[str, Any]] = []

        def warm(**kw: Any) -> str:
            calls.append(kw)
            return "nib=sh+pk"

        thl = types.SimpleNamespace(lnmd_mask_nib_warmup_launch=warm, _gcn_arch_name=lambda i: "gfx950:sramecc+")
        on = {KNOB: "1", "DLRM_HSTU_FP8_LNMD_FWD_MASKGEN": "1", "DLRM_HSTU_FP8_MASK16": "1"}
        rows = [3 << 19, (3 << 19) + 1]
        for env in ({}, {KNOB: "0"}, {KNOB: "true"}, {"DLRM_HSTU_FP8_LNMD_FWD_MASKGEN": "1"}):
            r = self._items(env, thl, rows)
            self.assertEqual((r.runs, r.skips), ([], []), env)  # knob off: nothing runs, logs or counts
        r = self._items(on, types.SimpleNamespace(), rows)
        self.assertEqual([s[0] for s in r.skips], ["L4n"])
        self.assertIn("no NIB writers", r.skips[0][2])
        for kw, why in ((dict(training=False), "cannot arm"), (dict(fuse=False), "cannot arm")):
            r = self._items(on, thl, rows, **kw)
            self.assertEqual(len(r.skips), 1)
            self.assertIn(why, r.skips[0][2])
        r = self._items(on, thl, [])
        self.assertIn("cannot arm", r.skips[0][2])
        r = self._items(dict(on, DLRM_HSTU_FP8_MASK16="0"), thl, rows)
        self.assertIn("maskgen or mask16 off", r.skips[0][2])
        r = self._items(on, types.SimpleNamespace(lnmd_mask_nib_warmup_launch=warm, _gcn_arch_name=lambda i: "gfx942"),
                        rows)
        self.assertIn("is not gfx950", r.skips[0][2])
        r = self._items(on, thl, rows)
        self.assertEqual(r.skips, [])
        fw = _fp8_warmup()
        # L4's large rows plus the > 2 GiB x / u class (_ROWS_XL), whatever FP8_WARMUP_STORES says
        want = rows + [fw._ROWS_XL, fw._ROWS_XL + 1]
        self.assertEqual([(t, n) for t, n, _, _ in r.runs], [("L4n", "lnmd_mask_nib_writers")] * 8)
        self.assertEqual([d for _, _, d, _ in r.runs], [f"T={t} seed={s} store=rows" for t in want for s in fw._SEEDS])
        self.assertEqual({t % 16 == 0 for t in want}, {True, False})  # both N % 16 classes
        self.assertEqual({t * D * 2 > 2**31 - 1 for t in want}, {True, False})  # x / u below and above 2 GiB
        self.assertEqual({s % 16 == 0 for s in fw._SEEDS}, {True, False})  # both seed % 16 classes
        for _, _, _, fn in r.runs:
            self.assertEqual(fn(), "nib=sh+pk")
        self.assertEqual([(tuple(c["x"].shape), c["x"].dtype, c["seed"]) for c in calls],
                         [((t, D), torch.bfloat16, s) for t in want for s in fw._SEEDS])
        r = self._items(on, thl, rows + [fw._ROWS_XL, fw._ROWS_XL + 1])  # STORES=both: no duplicate rows
        self.assertEqual(len(r.runs), 8)
        self.assertTrue(all(c["inv_scale"] == 64.0 and c["silu_u"] and c["dropout_ratio"] == 0.1 for c in calls))
        self.assertTrue(all("compare" not in c for c in calls))  # a thl without lnmd_mask_nib_latch: no check

    def test_l4n_startup_byte_check_and_latch(self) -> None:
        """review R1-F5: with lnmd_mask_nib_latch available, L4n asks the launch helper for the startup byte check on
        L4's rows only (not _ROWS_XL); a reported mismatch latches NIB once, after the item (outside its stdout
        capture), and L4n stops."""
        fw = _fp8_warmup()
        rows = [3 << 19, (3 << 19) + 1]
        xl = [fw._ROWS_XL, fw._ROWS_XL + 1]
        on = {KNOB: "1", "DLRM_HSTU_FP8_LNMD_FWD_MASKGEN": "1", "DLRM_HSTU_FP8_MASK16": "1"}

        class _ExecRunner(_FakeRunner):
            def __init__(self) -> None:
                super().__init__()
                self.bufs = types.SimpleNamespace(
                    act=lambda shape, dtype, big=False: torch.empty(shape, dtype=dtype, device="meta"))

            def run(self, tag: str, name: str, detail: str, fn: Any, all_ranks: bool = False) -> None:
                self.runs.append((tag, name, detail, fn))
                self.results.append(fn())

        def l4n(bad: Optional[Tuple[int, int]]) -> Tuple[_ExecRunner, List[Dict[str, Any]], List[str]]:
            calls: List[Dict[str, Any]] = []
            latched: List[str] = []

            def warm(**kw: Any) -> str:
                calls.append(kw)
                if kw.get("compare") and (int(kw["x"].shape[0]), kw["seed"]) == bad:
                    kw["verdict"]["mismatch"] = "synthetic mismatch"
                    return "nib=sh+pk cmp=MISMATCH(sh)"
                return "nib=sh+pk" + (" cmp=ok" if kw.get("compare") else "")

            thl = types.SimpleNamespace(lnmd_mask_nib_warmup_launch=warm, _gcn_arch_name=lambda i: "gfx950",
                                        lnmd_mask_nib_latch=latched.append)
            r = _ExecRunner()
            r.results = []
            ol = types.SimpleNamespace(_output_norm_weight=torch.ones(D, dtype=torch.bfloat16))
            with mock.patch.dict(os.environ, on):
                fw._nib_items(r, thl, ol, lambda width, dtype: (torch.ones(width), torch.zeros(width)), D, 0.1,
                              True, True, 64.0, rows)
            return r, calls, latched

        r, calls, latched = l4n(None)
        self.assertEqual(latched, [])
        self.assertEqual([d for _, _, d, _ in r.runs],
                         [f"T={t} seed={s} store=rows cmp=int8" for t in rows for s in fw._SEEDS]
                         + [f"T={t} seed={s} store=rows" for t in xl for s in fw._SEEDS])
        self.assertEqual([bool(c.get("compare")) for c in calls], [True] * 4 + [False] * 4)
        self.assertTrue(all(isinstance(c["verdict"], dict) for c in calls[:4]))
        self.assertTrue(all("verdict" not in c for c in calls[4:]))
        self.assertEqual(r.results, ["nib=sh+pk cmp=ok"] * 4 + ["nib=sh+pk"] * 4)
        r, calls, latched = l4n((rows[1], fw._SEEDS[0]))
        self.assertEqual(latched, ["synthetic mismatch"])  # once, then L4n stops: no further item
        self.assertEqual(len(r.runs), 3)
        self.assertEqual(r.results[-1], "nib=sh+pk cmp=MISMATCH(sh)")


# ------------------------------------------------------------------------------------------------ CPU: thl
class _State:
    """Save / restore the NIB, MGPK and maskgen globals of triton_hstu_linear the tests touch."""

    def __init__(self) -> None:
        self.thl = _thl()
        self.saved = {k: getattr(self.thl, k) for k in _NIB_GLOBALS if hasattr(self.thl, k)}

    def reset(self) -> None:
        t = self.thl
        t._LNMD_MASK_NIB_BROKEN = None
        t._LNMD_MASK_NIB_ARMED_LOGGED = False
        t._LNMD_MASK_NIB_SH_LOGGED = False
        t._LNMD_MASK_NIB_PK_LOGGED = False
        t._LNMD_MASK_NIB_DECLINES_LOGGED = ""
        t._LNMD_MASK_NIB_N_ARMED = 0
        t._LNMD_MASK_NIB_N_SH = 0
        t._LNMD_MASK_NIB_N_PK = 0
        t._LNMD_MASK_NIB_N_DECLINED = 0
        t._LNMD_MASK_NIB_N_FAILED = 0
        t._LNMD_MASK_NIB_ATEXIT = True  # never register the exit line from a test
        t._LNMD_MASK_NIB_FINAL_DONE = False
        t._LNMD_MASK_NIB_CALL_ID = 0
        t._LNMD_MASK_NIB_CALL_WHY = ""
        t._LNMD_FWD_MASKGEN_PK_ENABLED = None
        t._LNMD_FWD_MASKGEN_PK_ARMED_LOGGED = False
        t._LNMD_FWD_MASKGEN_PK_DECLINE_LOGGED = False
        t._LNMD_FWD_MASKGEN_PK_BROKEN = None
        t._LNMD_FWD_MASKGEN_PK_LAYERS = None
        t._LNMD_FWD_MASKGEN_PK_GATED_LOGGED = False
        t._LNMD_LAYER_TAG = -1
        t._LNMD_FWD_MASKGEN_ENABLED = None
        t._LNMD_FWD_MASKGEN_ARMED_LOGGED = False
        for k in _PK_V3_GLOBALS:  # ship17 (111) selector line marks (each holds the line it printed)
            if hasattr(t, k):
                setattr(t, k, "")

    def restore(self) -> None:
        for k, v in self.saved.items():
            setattr(self.thl, k, v)


class _Rec:
    """Stands in for a Triton kernel: kernel[grid](*args, **kw) records the call."""

    def __init__(self, name: str, raise_with: Optional[BaseException] = None) -> None:
        self.name = name
        self.calls: List[Tuple[Any, Tuple[Any, ...], Dict[str, Any]]] = []
        self.raise_with = raise_with

    def __getitem__(self, grid: Any):
        def launch(*args: Any, **kw: Any) -> None:
            self.calls.append((grid, args, kw))
            if self.raise_with is not None:
                raise self.raise_with

        return launch


def _cpu_xu(n: int) -> Tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(n)
    x = torch.randn((n, D), generator=g).to(torch.bfloat16)
    u = torch.randn((n, D), generator=g).to(torch.bfloat16)
    w = torch.ones((D,), dtype=torch.bfloat16)
    b = torch.zeros((D,), dtype=torch.bfloat16)
    return x, u, w, b


def _site(n: int, device: torch.device = torch.device("cpu")) -> Any:
    return types.SimpleNamespace(T=n, H=4, D=128, N=64, device=device,
                                 seq_offsets=torch.tensor([0, n], dtype=torch.int64))


@unittest.skipUnless(_importable(), "triton_hstu_linear not importable (torch / triton missing)")
class NibGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.st = _State()
        self.st.reset()
        self.thl = self.st.thl
        self.env = mock.patch.dict(os.environ, {})
        self.env.start()
        for k in (KNOB, "AOTI_LOWER", "DLRM_HSTU_FP8_MASKGEN_PK", "DLRM_HSTU_FP8_MASKGEN_PK_LAYERS", PK_V3_KNOB):
            os.environ.pop(k, None)
        nib.reset_for_testing()
        self.stack = contextlib.ExitStack()

    def tearDown(self) -> None:
        self.stack.close()
        self.env.stop()
        self.st.restore()
        nib.reset_for_testing()

    # -- an environment in which every predicate condition holds
    def _arm_all(self, n: int = 40) -> Tuple[torch.Tensor, ...]:
        thl = self.thl
        x, u, w, b = _cpu_xu(n)
        setattr(u, thl.HSTU_DUVQK_SITE_ATTR, types.SimpleNamespace(key=11, consumed=False))
        setattr(u, thl.HSTU_DUVQK_WIDTH_ATTR, 2048)
        os.environ[KNOB] = "1"
        p = self.stack.enter_context
        p(mock.patch.object(thl.fp8_wgrad_snr, "SNR_ENABLED", False))
        for name in ("requested", "vd_requested", "enabled", "prebuilt"):
            p(mock.patch.object(thl.l11_outproj_lnmd, name, lambda: True))
        p(mock.patch.object(thl.l11_do8, "nib_ready", lambda: True, create=True))
        self.site = _site(n)
        p(mock.patch.object(thl.l11_do8, "site_of", lambda t: self.site))
        p(mock.patch.object(thl.bwd_fp8, "du_producer_requested", lambda: True))
        p(mock.patch.object(thl.bwd_fp8, "du_producer_enabled", lambda: True))
        p(mock.patch.object(thl.bwd_fp8, "producer_generation", lambda s, k: (object(), 1)))
        p(mock.patch.object(thl.bwd_fp8, "grad_dtype_and_limit", lambda: (torch.float8_e5m2, 57344.0)))
        p(mock.patch.object(thl.hstu_du_e5m2, "broken", lambda: None))
        p(mock.patch.object(thl.hstu_du_e5m2, "diagnostics_on", lambda: False))
        p(mock.patch.object(thl, "_use_lnmd_bwd_v2", lambda: True))
        return x, u, w, b

    def _why(self, x, u, w, b, n: int = 40, ctx: bool = True, silu_u: bool = True) -> Optional[str]:
        if ctx:
            self.thl._LNMD_MASK_NIB_CALL_ID = id(x)
            self.thl._LNMD_MASK_NIB_CALL_WHY = ""
        return self.thl._lnmd_mask_nib_why_not(x, u, w, b, n, D, D, silu_u)

    def test_aliases_and_kk2(self) -> None:
        thl = self.thl
        self.assertIs(thl.lnmd_mask_nibpack_ref, nib.pack_ref)
        self.assertIs(thl.lnmd_mask_nibunpack_ref, nib.unpack)
        for p in (1e-4, 0.1, 0.25, 0.5, 0.9, 0.99999):
            keep_min16 = thl._mask_keep_min16(p)
            kk2 = (keep_min16 - 1) * 0x10001  # the launch site's inline formula
            if kk2 >= 2**31:
                kk2 -= 2**32
            self.assertEqual(thl._lnmd_maskgen_kk2(p), (keep_min16, kk2), p)
        keys = set(thl.lnmd_mask_nib_counters())
        self.assertTrue({"armed", "written_sh", "written_pk", "declined"} <= keys, keys)

    def test_knob_off_touches_nothing(self) -> None:
        thl = self.thl
        x, u, w, b = _cpu_xu(16)
        launch = mock.MagicMock()
        before = dict(thl.lnmd_mask_nib_counters())
        for value in (None, "0", "", "true"):
            if value is None:
                os.environ.pop(KNOB, None)
            else:
                os.environ[KNOB] = value
            thl._LNMD_MASK_NIB_CALL_ID, thl._LNMD_MASK_NIB_CALL_WHY = 12345, "sentinel"
            buf = io.StringIO()
            with mock.patch.object(thl, "_lnmd_fwd_maskgen_nib_launch", launch), contextlib.redirect_stdout(buf):
                self.assertEqual(thl._lnmd_mask_nib_why_not(x, u, w, b, 16, D, D, True), thl._LNMD_MASK_NIB_OFF)
                got = thl._lnmd_fwd_maskgen_nib_try(x, u, None, w, b, None, None, 16, D, EPS, 0.1, INV, 5, D,
                                                    True, True, "silu", True, True)
                thl._lnmd_mask_nib_note_call(x, torch.empty((1536, D)), True)
            self.assertIsNone(got)
            self.assertEqual((thl._LNMD_MASK_NIB_CALL_ID, thl._LNMD_MASK_NIB_CALL_WHY), (12345, "sentinel"))
            self.assertEqual(buf.getvalue(), "")
        launch.assert_not_called()
        self.assertEqual(thl.lnmd_mask_nib_counters(), before)
        self.assertIsNone(thl._LNMD_FWD_MASKGEN_PK_ENABLED)  # MGPK's gate not even consulted

    def test_armed_and_each_decline_in_order(self) -> None:
        thl = self.thl
        x, u, w, b = self._arm_all()
        self.assertIsNone(self._why(x, u, w, b))
        self.assertEqual((thl._LNMD_MASK_NIB_CALL_ID, thl._LNMD_MASK_NIB_CALL_WHY), (0, ""))  # consumed
        self.assertTrue(self._why(x, u, w, b, ctx=False).startswith("N2 no-context"))  # consumed once
        cases = []

        def case(code: str, enter, leave=None) -> None:
            cases.append((code, enter, leave))

        def setg(name, value):
            old = getattr(thl, name)
            setattr(thl, name, value)
            return lambda: setattr(thl, name, old)

        def patch(obj, name, value):
            m = mock.patch.object(obj, name, value, create=True)
            m.start()
            return m.stop

        def setenv(k, v):
            os.environ[k] = v
            return lambda: os.environ.pop(k, None)

        def setattr_t(t, k, v):
            old = getattr(t, k, None)
            setattr(t, k, v)
            return lambda: setattr(t, k, old)

        case("N1 latched", lambda: setg("_LNMD_MASK_NIB_BROKEN", "RuntimeError: x"))
        case("N4 snr", lambda: patch(thl.fp8_wgrad_snr, "SNR_ENABLED", True))
        case("N4 compile", lambda: setenv("AOTI_LOWER", "1"))
        case("N5 l11-off", lambda: patch(thl.l11_outproj_lnmd, "requested", lambda: False))
        case("N5 l11-vd", lambda: patch(thl.l11_outproj_lnmd, "vd_requested", lambda: False))
        case("N5 l11-ready", lambda: patch(thl.l11_outproj_lnmd, "prebuilt", lambda: False))
        case("N6 do8-ready", lambda: patch(thl.l11_do8, "nib_ready", lambda: False))
        case("N7 no-site", lambda: patch(thl.l11_do8, "site_of", lambda t: None))
        case("N7 site-shape", lambda: setattr_t(self.site, "T", 41))
        case("N7 site-device", lambda: setattr_t(self.site, "device", torch.device("meta")))
        case("N7 site-offsets", lambda: setattr_t(self.site, "seq_offsets", torch.tensor([0, 40], dtype=torch.int16)))
        case("N8 b1a-off", lambda: patch(thl.bwd_fp8, "du_producer_requested", lambda: False))
        case("N8 b1a-ready", lambda: patch(thl.hstu_du_e5m2, "broken", lambda: "latched"))
        case("N8 b1a-diag", lambda: patch(thl.hstu_du_e5m2, "diagnostics_on", lambda: True))
        case("N8 b1a-site", lambda: setattr_t(u, thl.HSTU_DUVQK_SITE_ATTR, None))
        case("N8 b1a-warmup", lambda: patch(thl.bwd_fp8, "producer_generation", lambda s, k: None))
        case("N9 e5", lambda: patch(thl.bwd_fp8, "grad_dtype_and_limit", lambda: (torch.float8_e4m3fn, 448.0)))
        case("N9 flags", lambda: patch(thl, "_use_lnmd_bwd_v2", lambda: False))
        case("N9 du-width", lambda: setattr_t(u, thl.HSTU_DUVQK_WIDTH_ATTR, 0))
        for code, enter, _ in cases:
            undo = enter()
            try:
                got = self._why(x, u, w, b)
            finally:
                undo()
            self.assertIsNotNone(got, code)
            self.assertTrue(got.startswith(code + ":"), (code, got))
            self.assertIsNone(self._why(x, u, w, b), code)  # undone: armed again
        # the out-proj context recorded by the forward wins over everything below it
        thl._LNMD_MASK_NIB_CALL_ID, thl._LNMD_MASK_NIB_CALL_WHY = id(x), "N3 out-warmup: test"
        self.assertEqual(thl._lnmd_mask_nib_why_not(x, u, w, b, 40, D, D, True), "N3 out-warmup: test")
        self.assertTrue(self._why(x, u, w, b, silu_u=False).startswith("N9 flags"))
        # no DO8 NIB reader in the tree
        with mock.patch.object(thl, "l11_do8", types.SimpleNamespace(site_of=lambda t: None)):
            self.assertTrue(self._why(x, u, w, b).startswith("N6 do8-api"))
        # L11's D8 layout mirrors
        self.assertIsNone(self._why(x[:, :D], u, w, b))  # a view with the same layout: armed
        xs = torch.empty((40, D + 8), dtype=torch.bfloat16)[:, :D]
        self.assertTrue(self._why(xs, u, w, b).startswith("N9 layout"))
        self.assertTrue(self._why(x, u, w.float(), b).startswith("N9 layout"))
        self.assertTrue(thl._lnmd_mask_nib_why_not(x, u, w, b, 40, D, 256, True).startswith("N2"))  # no context
        thl._LNMD_MASK_NIB_CALL_ID = id(x)
        self.assertTrue(thl._lnmd_mask_nib_why_not(x, u, w, b, 40, D, 256, True).startswith("N9 shape"))

    def test_note_call(self) -> None:
        thl = self.thl
        os.environ[KNOB] = "1"
        attn = torch.empty((8, D), dtype=torch.bfloat16)
        ow = torch.empty((1536, D), dtype=torch.bfloat16)
        state = types.SimpleNamespace(published=True)
        p = self.stack.enter_context
        p(mock.patch.object(thl, "_FP8_OUT_WGRAD", True))
        p(mock.patch.object(thl.bwd_fp8, "scale_controller_enabled", lambda: True))
        p(mock.patch.object(thl.bwd_fp8, "_has_stable_identity", lambda t: True, create=True))
        peek = mock.MagicMock(return_value=state)
        p(mock.patch.object(thl.bwd_fp8, "_peek_site_state", peek, create=True))

        def note(a=attn, w=ow, need_bwd=True) -> str:
            thl._LNMD_MASK_NIB_CALL_ID, thl._LNMD_MASK_NIB_CALL_WHY = 0, "x"
            thl._lnmd_mask_nib_note_call(a, w, need_bwd)
            self.assertEqual(thl._LNMD_MASK_NIB_CALL_ID, id(a))
            return thl._LNMD_MASK_NIB_CALL_WHY

        self.assertEqual(note(), "")
        peek.assert_called_with("output_projection", ow.data_ptr())
        self.assertTrue(note(need_bwd=False).startswith("N3 no-backward:"))
        self.assertTrue(note(w=torch.empty((512, D))).startswith("N3 out-weight:"))
        state.published = False
        self.assertTrue(note().startswith("N3 out-warmup:"))
        peek.return_value = None
        self.assertTrue(note().startswith("N3 out-warmup:"))
        peek.side_effect = RuntimeError("boom")
        self.assertTrue(note().startswith("N3 out-error:"))
        peek.side_effect = None
        peek.return_value = types.SimpleNamespace(published=True)
        with mock.patch.object(thl.bwd_fp8, "_has_stable_identity", lambda t: False, create=True):
            self.assertTrue(note().startswith("N3 out-site:"))
        with mock.patch.object(thl, "_FP8_OUT_WGRAD", False):
            self.assertTrue(note().startswith("N3 out-wgrad:"))

    def test_decline_lines(self) -> None:
        thl = self.thl
        os.environ[KNOB] = "1"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for why in ("N2 no-context: a", "N3 out-warmup: b", "N7 no-site: c", "N8 b1a-warmup: d", "N1 latched: e",
                        "N8 b1a-site: f"):  # review R1-F7: B1a's warm-up declines at b1a-site (no site on u yet)
                thl._lnmd_mask_nib_note_decline(why)
        self.assertEqual(buf.getvalue(), "")  # expected states: counted only
        with contextlib.redirect_stdout(buf):
            thl._lnmd_mask_nib_note_decline("N5 l11-off: L11 item (39) off")
            thl._lnmd_mask_nib_note_decline("N5 l11-off: L11 item (39) off")
            thl._lnmd_mask_nib_note_decline("N5 l11-vd: other")
        out = buf.getvalue()
        line = "LN*U*dropout keep mask NIB NOT armed for this call (DLRM_HSTU_FP8_LNMD_MASK_NIB)"
        self.assertEqual(out.count(line), 2)
        self.assertEqual(out.count("[rank "), 2)
        self.assertEqual(thl.lnmd_mask_nib_counters()["declined"], 9)

    def _try(self, x, u, w, b, n: int, inv: float = INV) -> Tuple[Optional[torch.Tensor], str]:
        y = torch.empty((n, 3 * D), dtype=torch.uint8)
        mean = torch.empty((n,), dtype=torch.float32)
        rstd = torch.empty((n,), dtype=torch.float32)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = self.thl._lnmd_fwd_maskgen_nib_try(x, u, y, w, b, mean, rstd, n, D, EPS, 0.1, inv, 99, D, True, True,
                                                     "silu", True, True)
        return got, buf.getvalue()

    def test_hook_armed_sh_pk_and_lines(self) -> None:
        thl = self.thl
        x, u, w, b = _cpu_xu(24)
        launch = mock.MagicMock()
        p = self.stack.enter_context
        p(mock.patch.object(thl, "_lnmd_mask_nib_why_not", lambda *a, **k: None))
        p(mock.patch.object(thl, "_lnmd_fwd_maskgen_nib_launch", launch))
        os.environ["DLRM_HSTU_FP8_MASKGEN_PK"] = "0"
        got, out = self._try(x, u, w, b, 24)
        self.assertEqual((got.dtype, tuple(got.shape)), (torch.uint8, (24, D // 2)))
        args = launch.call_args.args
        self.assertIs(args[7], got)
        self.assertEqual(args[13:16], thl._lnmd_maskgen_kk2(0.1) + (99,))
        self.assertEqual(args[-1], False)  # sh-NIB
        self.assertEqual(out.count("armed OK: LN*U*dropout keep mask NIB ACTIVE (DLRM_HSTU_FP8_LNMD_MASK_NIB, item "
                                   "(94)): item (28)(e)'s"), 1)
        self.assertIn("first launch: _ln_mul_dropout_fwd_maskgen_nib on layer tag -1, N=24", out)
        self.assertIn("[lnmd-mask-nib] first launch of _ln_mul_dropout_fwd_maskgen_nib", out)
        self.assertIn("[lnmd-mask-nib] counters: armed=1 written_sh=1 written_pk=0", out)
        self.assertNotIn("maskgen pk ACTIVE", out)
        # MGPK on: the pk-NIB kernel, MGPK's own ACTIVE line once, no second NIB ACTIVE line
        self.st.reset()
        thl._LNMD_MASK_NIB_ARMED_LOGGED = True
        thl._LNMD_MASK_NIB_SH_LOGGED = True
        os.environ["DLRM_HSTU_FP8_MASKGEN_PK"] = "1"
        thl.set_lnmd_layer_tag(1)
        got, out = self._try(x, u, w, b, 24)
        self.assertEqual(launch.call_args.args[-1], True)
        self.assertNotIn("NIB ACTIVE", out)
        self.assertIn("[lnmd-mask-nib] first launch of _ln_mul_dropout_fwd_maskgen_pk_nib (MGPK + NIB store) on "
                      "layer tag 1", out)
        self.assertEqual(out.count("armed OK: LN*U*dropout fwd maskgen pk ACTIVE"), 1)
        # MGPK declines this call (fp32 inv_scale overflows): sh-NIB, MGPK's own declined line
        got, out = self._try(x, u, w, b, 24, inv=1.0e39)
        self.assertEqual(launch.call_args.args[-1], False)
        self.assertIn("LN*U*dropout fwd maskgen pk declined", out)
        self.assertEqual(thl.lnmd_mask_nib_counters()["armed"], 2)
        # MGPK's layer list: layer 0 runs sh-NIB (MGPK's layer-gated line), layer 1 pk-NIB
        self.st.reset()
        thl._LNMD_MASK_NIB_ARMED_LOGGED = True
        os.environ["DLRM_HSTU_FP8_MASKGEN_PK_LAYERS"] = "1"
        for tag, pk in ((0, False), (1, True)):
            thl.set_lnmd_layer_tag(tag)
            self._try(x, u, w, b, 24)
            self.assertEqual(launch.call_args.args[-1], pk, tag)

    def test_hook_launch_failure_latches(self) -> None:
        thl = self.thl
        x, u, w, b = self._arm_all(24)
        p = self.stack.enter_context
        real_why = thl._lnmd_mask_nib_why_not
        p(mock.patch.object(thl, "_lnmd_fwd_maskgen_nib_launch", mock.MagicMock(side_effect=RuntimeError("boom"))))
        thl._LNMD_MASK_NIB_CALL_ID = id(x)
        got, out = self._try(x, u, w, b, 24)
        self.assertIsNone(got)
        self.assertIn("LN*U*dropout keep mask NIB DISABLED (DLRM_HSTU_FP8_LNMD_MASK_NIB): launch of "
                      "_ln_mul_dropout_fwd_maskgen_nib raised RuntimeError: boom", out)
        self.assertNotIn("NIB ACTIVE", out)
        c = thl.lnmd_mask_nib_counters()
        self.assertEqual((c["failed"], c["latched"], c["armed"]), (1, 1, 0))
        thl._LNMD_MASK_NIB_CALL_ID = id(x)
        self.assertTrue(real_why(x, u, w, b, 24, D, D, True).startswith("N1 latched"))
        thl._LNMD_MASK_NIB_CALL_ID = id(x)
        got, out = self._try(x, u, w, b, 24)
        self.assertIsNone(got)
        self.assertEqual(out, "")  # latched: quiet

    def test_gate_error_declines(self) -> None:
        thl = self.thl
        x, u, w, b = _cpu_xu(24)
        launch = mock.MagicMock()
        boom = mock.MagicMock(side_effect=AttributeError("no T"))
        with mock.patch.object(thl, "_lnmd_mask_nib_why_not", boom), \
                mock.patch.object(thl, "_lnmd_fwd_maskgen_nib_launch", launch):
            got, out = self._try(x, u, w, b, 24)
            got2, out2 = self._try(x, u, w, b, 24)
        self.assertIsNone(got)
        self.assertIsNone(got2)
        launch.assert_not_called()
        self.assertIn("NOT armed for this call (DLRM_HSTU_FP8_LNMD_MASK_NIB): N10 gate-error: AttributeError: no T", out)
        self.assertEqual(out2, "")  # once per reason
        self.assertEqual(thl.lnmd_mask_nib_counters()["declined"], 2)

    # -- the production entry, maskgen branch, with mocked launchers
    def _impl(self, n: int, nib_launch: Any, why: Any) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, Any], str]:
        thl = self.thl
        x, u, w, b = _cpu_xu(n)
        shipped = _Rec("_ln_mul_dropout_fwd_maskgen")
        pk_try = mock.MagicMock(return_value=False)
        with contextlib.ExitStack() as st:
            p = st.enter_context
            p(mock.patch.object(thl, "use_separated_rng_ln_mul_dropout", lambda: True))
            p(mock.patch.object(thl, "_use_lnmd_fwd_maskgen", lambda *a, **k: True))
            p(mock.patch.object(thl, "_lnmd_fwd_maskgen_pk_try", pk_try))
            p(mock.patch.object(thl, "_ln_mul_dropout_fwd_maskgen", shipped))
            p(mock.patch.object(thl, "_lnmd_fwd_maskgen_nib_launch", nib_launch))
            if why is not None:
                p(mock.patch.object(thl, "_lnmd_mask_nib_why_not", why))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = thl._triton_layer_norm_mul_dropout_fwd_impl(
                    x, u, w, b, EPS, 0.1, True, True, True, True, "silu", 77, torch.empty(0, dtype=torch.int64), True,
                    INV)
        return out, dict(shipped=shipped, pk_try=pk_try), buf.getvalue()

    def test_impl_knob_off_is_todays_path(self) -> None:
        n = 40
        nib_launch = mock.MagicMock()
        for knob in (None, "0", "1"):  # "1" without a context: declined (N2, quiet) -> today's path too
            self.st.reset()
            if knob is None:
                os.environ.pop(KNOB, None)
            else:
                os.environ[KNOB] = knob
            out, rec, text = self._impl(n, nib_launch, None)
            mask = out[3]
            self.assertEqual((mask.dtype, tuple(mask.shape)), (torch.int8, (n, D)), knob)
            self.assertEqual(rec["pk_try"].call_count, 1, knob)
            self.assertIs(rec["pk_try"].call_args.args[7], mask, knob)
            self.assertEqual(len(rec["shipped"].calls), 1, knob)
            grid, args, kw = rec["shipped"].calls[0]
            self.assertEqual(grid, ((n + 7) // 8,))
            self.assertIs(args[7], mask)
            self.assertEqual(args[15], D)  # mask row stride
            self.assertEqual(args[17:20], self.thl._lnmd_maskgen_kk2(0.1) + (77,))
            self.assertEqual(kw["NUM_MASKS"], 3)
            self.assertEqual(text, "", knob)
            self.assertEqual(self.thl.lnmd_mask_nib_counters()["declined"], 1 if knob == "1" else 0)
        nib_launch.assert_not_called()

    def test_impl_predicate_forced_none_writes_nib_only(self) -> None:
        n = 40
        os.environ["DLRM_HSTU_FP8_MASKGEN_PK"] = "0"
        nib_launch = mock.MagicMock()
        out, rec, text = self._impl(n, nib_launch, lambda *a, **k: None)
        mask = out[3]
        self.assertEqual((mask.dtype, tuple(mask.shape)), (torch.uint8, (n, D // 2)))
        self.assertEqual(rec["pk_try"].call_count, 0)
        self.assertEqual(rec["shipped"].calls, [])
        self.assertEqual(nib_launch.call_count, 1)
        self.assertIs(nib_launch.call_args.args[7], mask)
        self.assertIs(nib_launch.call_args.args[2], out[0])  # y, mean, rstd are the entry's own outputs
        self.assertIs(nib_launch.call_args.args[5], out[1])
        self.assertIs(nib_launch.call_args.args[6], out[2])
        self.assertIn("NIB ACTIVE", text)
        # a raising NIB launch: latched, today's path for the same call
        self.st.reset()
        out, rec, text = self._impl(n, mock.MagicMock(side_effect=RuntimeError("compile")), lambda *a, **k: None)
        self.assertEqual((out[3].dtype, tuple(out[3].shape)), (torch.int8, (n, D)))
        self.assertEqual(rec["pk_try"].call_count, 1)
        self.assertEqual(len(rec["shipped"].calls), 1)
        self.assertIn("LN*U*dropout keep mask NIB DISABLED", text)

    # -- the backward's int8 entries
    def test_backward_entries_receive_the_exact_unpack(self) -> None:
        thl = self.thl
        n = 40
        x, u, w, b = _cpu_xu(n)
        g = torch.Generator().manual_seed(3)
        m8 = _rand_mask(n, g)
        packed = nib.pack_ref(m8)
        props = types.SimpleNamespace(multi_processor_count=4)
        dy = torch.zeros((n, 3 * D), dtype=torch.bfloat16)
        mean = torch.zeros((n,), dtype=torch.float32)
        rstd = torch.ones((n,), dtype=torch.float32)
        for mask, expect_unpack in ((packed, True), (m8, False)):
            nib.reset_for_testing()
            v2 = _Rec("_ln_mul_dropout_bwd_dx_du_rng_v2")
            with mock.patch.object(thl, "_ln_mul_dropout_bwd_dx_du_rng_v2", v2), \
                    mock.patch.object(thl, "_use_lnmd_bwd_v2", lambda: True), \
                    mock.patch.object(thl, "_lnmd_bwd_finalize", lambda *a, **k: None), \
                    mock.patch("torch.cuda.get_device_properties", lambda *a, **k: props), \
                    contextlib.redirect_stdout(io.StringIO()):
                thl._triton_layer_norm_mul_dropout_bwd_impl(
                    dy, x, u, w, b, mean, rstd, D, 2, EPS, True, 0.1, 0, True, True, True, "silu", False, mask, 0)
            (_, args, _), = v2.calls
            self.assertEqual(args[12].dtype, torch.int8)
            self.assertTrue(torch.equal(args[12], m8))
            self.assertEqual(args[19], D)  # the mask row stride the kernel receives
            self.assertEqual(nib.counters()["unpack"], int(expect_unpack))
            # B1a v3
            v3 = _Rec("_ln_mul_dropout_bwd_dx_du_rng_v3")
            with mock.patch.object(thl, "_ln_mul_dropout_bwd_dx_du_rng_v3", v3), \
                    mock.patch.object(thl, "_ln_mul_dropout_bwd_dwdb", _Rec("dwdb")), \
                    mock.patch.object(thl.bwd_fp8, "grad_dtype_and_limit", lambda: (torch.float8_e5m2, 57344.0)), \
                    mock.patch.object(thl.bwd_fp8, "uvqk_tm_pad", lambda: 128), \
                    mock.patch.object(thl.hstu_du_e5m2, "placeholder_du", lambda *a: torch.empty(0)), \
                    mock.patch.object(thl.hstu_du_e5m2, "DuE5m2Emit", mock.MagicMock()), \
                    mock.patch("torch.cuda.get_device_properties", lambda *a, **k: props), \
                    contextlib.redirect_stdout(io.StringIO()):
                thl._triton_layer_norm_mul_dropout_bwd_du_e5m2(
                    dy=dy, x=x, u=u, weight=w, bias=b, mean=mean, rstd=rstd, BLOCK_D=D, num_warps=2,
                    dropout_ratio=0.1, silu_u=True, random_mask=mask, du_row_width=2048, site_key=1,
                    inv_scale=torch.ones(1), cast_scale=torch.ones(1), generation=None)
            (_, args, _), = v3.calls
            self.assertTrue(args[10].dtype == torch.int8 and torch.equal(args[10], m8))
            self.assertEqual(args[19], D)  # random_mask.stride(0)
            # L11 run_e5 through _l11_du_e5m2_or_none
            run_e5 = mock.MagicMock(return_value=None)
            with mock.patch.object(thl.l11_outproj_lnmd, "enabled", lambda: True), \
                    mock.patch.object(thl.l11_outproj_lnmd, "prebuilt", lambda: True), \
                    mock.patch.object(thl.l11_outproj_lnmd, "run_e5", run_e5), \
                    contextlib.redirect_stdout(io.StringIO()):
                thl._l11_du_e5m2_or_none(
                    None, None, None, None, x=x, u=u, weight=w, bias=b, mean=mean, rstd=rstd, BLOCK_D=D,
                    num_warps=2, dropout_ratio=0.1, silu_u=True, random_mask=mask, du_row_width=2048, site_key=1,
                    inv_scale=torch.ones(1), cast_scale=torch.ones(1), generation=None)
            got = run_e5.call_args.args[14]
            self.assertTrue(got.dtype == torch.int8 and torch.equal(got, m8))
            if not expect_unpack:
                self.assertIs(got, m8)  # an int8 mask passes unchanged

    def test_du_e5m2_plan_accepts_the_nib_layout(self) -> None:
        thl = self.thl
        n = 40
        x, u, w, b = _cpu_xu(n)
        dy = torch.zeros((n, 3 * D), dtype=torch.bfloat16)
        ctx = types.SimpleNamespace(group_norm=False, training=True, concat_u=True, concat_x=True,
                                    mul_u_activation_type="silu", BLOCK_D=D)
        site = types.SimpleNamespace(consumed=False, key=5)
        log = mock.MagicMock()
        g = torch.Generator().manual_seed(4)
        m8 = _rand_mask(n, g)
        with mock.patch.object(thl.hstu_du_e5m2, "broken", lambda: None), \
                mock.patch.object(thl.hstu_du_e5m2, "diagnostics_on", lambda: False), \
                mock.patch.object(thl.hstu_du_e5m2, "log_not_used", log), \
                mock.patch.object(thl.bwd_fp8, "du_producer_enabled", lambda: True), \
                mock.patch.object(thl.bwd_fp8, "producer_inv_scale", lambda s, k: (torch.ones(1), torch.ones(1))), \
                mock.patch.object(thl.bwd_fp8, "producer_generation", lambda s, k: ("g", 1)), \
                mock.patch.object(thl, "_use_lnmd_bwd_v2", lambda: True), \
                mock.patch.object(thl, "_use_mask16", lambda: True):
            plan = thl._du_e5m2_plan(ctx, site, dy, x, u, m8, 2048, compute_y=False)
            self.assertIsNotNone(plan)
            self.assertIsNotNone(thl._du_e5m2_plan(ctx, site, dy, x, u, nib.pack_ref(m8), 2048, compute_y=False))
            log.assert_not_called()
            bad = torch.zeros((n, D), dtype=torch.uint8)[:, : D // 2]  # NIB shape, wrong row stride
            self.assertIsNone(thl._du_e5m2_plan(ctx, site, dy, x, u, bad, 2048, compute_y=False))
            self.assertEqual(log.call_args.args[0], "layout")
            self.assertIsNone(thl._du_e5m2_plan(ctx, site, dy, x, u, m8[: n - 1], 2048, compute_y=False))

    def test_warmup_launch_helper(self) -> None:
        thl = self.thl
        x, u, w, b = _cpu_xu(24)
        launch = mock.MagicMock()
        before = thl.lnmd_mask_nib_counters()
        buf = io.StringIO()
        with mock.patch.object(thl, "_lnmd_fwd_maskgen_nib_launch", launch), contextlib.redirect_stdout(buf):
            self.assertEqual(thl.lnmd_mask_nib_warmup_launch(x, u, w, b, EPS, 0.1, 2**40, 64.0, True), "nib=sh+pk")
            self.assertEqual([c.args[-1] for c in launch.call_args_list], [False, True])
            a = launch.call_args.args
            self.assertEqual((a[8], a[9], a[16], a[17], a[18], a[19], a[20], a[21]),
                             (24, D, D, True, True, "silu", True, True))
            self.assertEqual((a[7].dtype, tuple(a[7].shape)), (torch.uint8, (24, D // 2)))
            launch.reset_mock()
            self.assertEqual(thl.lnmd_mask_nib_warmup_launch(x, u, w, b, EPS, 0.1, 2**40, 1.0e39, True), "nib=sh")
        self.assertEqual(buf.getvalue(), "")
        self.assertEqual(thl.lnmd_mask_nib_counters(), before)

    def test_warmup_launch_startup_byte_check(self) -> None:
        """review R1-F5, mocked kernels on the CPU: compare=True runs the int8 kernel each NIB writer replaces on the
        same inputs (same production arguments) and reports any differing byte in verdict, never latching or
        printing; an unwritten NIB byte (poison) or an int8 byte > 7 is a difference too."""
        thl = self.thl
        n = 24
        x, u, w, b = _cpu_xu(n)
        g = torch.Generator().manual_seed(9)
        m8 = _rand_mask(n, g)
        m8_flip = m8.clone()
        m8_flip[5, 17] ^= 1
        m8_high = m8.clone()
        m8_high[7, 3] = 8 | (int(m8[7, 3]) & 7)

        def writer(fill_y: int = 3, mean_v: float = 0.5, mask: Optional[torch.Tensor] = None, nib_out: bool = False,
                   skip_mask: bool = False):
            def launch(x_, u_, y, w_, b_, mean, rstd, out, *args) -> None:
                y.view(torch.uint8).fill_(fill_y)
                mean.fill_(mean_v)
                rstd.fill_(2.0)
                if skip_mask:
                    return
                src = m8 if mask is None else mask
                out.copy_(nib.pack_ref(src) if nib_out else src)
            return launch

        before = thl.lnmd_mask_nib_counters()
        cases = (
            (writer(nib_out=True), writer(), "nib=sh+pk cmp=ok", None),
            (writer(nib_out=True), writer(mask=m8_flip), "nib=sh+pk cmp=MISMATCH(sh+pk)", "keep mask"),
            (writer(nib_out=True, skip_mask=True), writer(), "nib=sh+pk cmp=MISMATCH(sh+pk)", "keep mask"),
            (writer(nib_out=True), writer(fill_y=4), "nib=sh+pk cmp=MISMATCH(sh+pk)", ": y"),
            (writer(nib_out=True), writer(mean_v=0.25), "nib=sh+pk cmp=MISMATCH(sh+pk)", "mean"),
            (writer(nib_out=True), writer(mask=m8_high), "nib=sh+pk cmp=MISMATCH(sh+pk)", "int8 reference mask "
             "bytes > 7"),
        )
        for nib_fn, int8_fn, tag, why in cases:
            nib_launch, int8_launch = mock.MagicMock(side_effect=nib_fn), mock.MagicMock(side_effect=int8_fn)
            verdict: Dict[str, str] = {}
            buf = io.StringIO()
            with mock.patch.object(thl, "_lnmd_fwd_maskgen_nib_launch", nib_launch), \
                    mock.patch.object(thl, "_lnmd_fwd_maskgen_int8_launch", int8_launch), \
                    contextlib.redirect_stdout(buf):
                got = thl.lnmd_mask_nib_warmup_launch(x, u, w, b, EPS, 0.1, 2**40, 64.0, True, compare=True,
                                                      verdict=verdict)
            self.assertEqual(got, tag)
            self.assertEqual([c.args[-1] for c in int8_launch.call_args_list], [False, True])  # sh then pk
            a, a8 = nib_launch.call_args.args, int8_launch.call_args.args
            for i in (0, 1, 3, 4):  # the same x, u, weight, bias objects
                self.assertIs(a[i], a8[i])
            self.assertEqual(a[8:], a8[8:])  # the same production arguments after the outputs
            self.assertEqual((a8[7].dtype, tuple(a8[7].shape)), (torch.int8, (n, D)))
            if why is None:
                self.assertEqual(verdict, {})
            else:
                self.assertIn(why, verdict["mismatch"])
                self.assertTrue(verdict["mismatch"].startswith("FP8 warmup startup check (item L4n): "
                                                               "_ln_mul_dropout_fwd_maskgen_nib vs "
                                                               "_ln_mul_dropout_fwd_maskgen on the same inputs (N=24"))
            self.assertEqual(buf.getvalue(), "")
        self.assertEqual(thl.lnmd_mask_nib_counters(), before)  # nothing latched or counted
        self.assertIsNone(thl._LNMD_MASK_NIB_BROKEN)
        # without compare: no reference launch; the fp32 Y_INV_SCALE MGPK cannot take: the sh writer only
        int8_launch = mock.MagicMock(side_effect=writer())
        with mock.patch.object(thl, "_lnmd_fwd_maskgen_nib_launch", mock.MagicMock(side_effect=writer(nib_out=True))), \
                mock.patch.object(thl, "_lnmd_fwd_maskgen_int8_launch", int8_launch):
            self.assertEqual(thl.lnmd_mask_nib_warmup_launch(x, u, w, b, EPS, 0.1, 2**40, 64.0, True), "nib=sh+pk")
            int8_launch.assert_not_called()
            got = thl.lnmd_mask_nib_warmup_launch(x, u, w, b, EPS, 0.1, 2**40, 1.0e39, True, compare=True,
                                                  verdict={})
        self.assertEqual(got, "nib=sh cmp=ok")

    def test_latch_api_and_exit_line(self) -> None:
        thl = self.thl
        x, u, w, b = self._arm_all()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            thl.lnmd_mask_nib_latch("synthetic reason")
            thl.lnmd_mask_nib_latch("second reason")  # already latched: no-op
        out = buf.getvalue()
        self.assertEqual(out.count("LN*U*dropout keep mask NIB DISABLED (DLRM_HSTU_FP8_LNMD_MASK_NIB): synthetic "
                                   "reason; the int8 keep mask (MGPK / the shipped _ln_mul_dropout_fwd_maskgen) is "
                                   "written for the rest of the process [rank "), 1)
        self.assertNotIn("second reason", out)
        c = thl.lnmd_mask_nib_counters()
        self.assertEqual((c["latched"], c["failed"]), (1, 0))
        self.assertTrue(self._why(x, u, w, b).startswith("N1 latched: synthetic reason"))
        # the exit counters line: knob "1" only, once per process (EXIT_HOOKS and atexit can both reach it)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            thl._lnmd_mask_nib_final()
            thl._lnmd_mask_nib_final()
        self.assertEqual(buf.getvalue().count("[lnmd-mask-nib] counters: "), 1)
        self.assertIn("latched=1 final=1", buf.getvalue())
        thl._LNMD_MASK_NIB_FINAL_DONE = False
        os.environ[KNOB] = "0"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            thl._lnmd_mask_nib_final()
        self.assertEqual(buf.getvalue(), "")


# ------------------------------------------------------------------------------------------------ GPU
def _gpu_inputs(n: int, gen_seed: int = 1234) -> Tuple[torch.Tensor, ...]:
    dev = torch.device("cuda")
    g = torch.Generator(device=dev)
    g.manual_seed(gen_seed)
    x = (torch.randn(n, D, device=dev, generator=g) * 0.5 + 0.05).to(torch.bfloat16)
    u = (torch.randn(n, D, device=dev, generator=g) * 2.0).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    b = (0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    u[n - 1, 3] = -100.0  # the sigmoid fallback in the last (possibly partial) group
    if n > 8:
        x[1, 5] = float("nan")
        u[2, 7] = float("inf")
        x[3, :] = 0.75  # var = 0
    return x, u, w, b


def _gpu_bufs(n: int, junk: int) -> Dict[str, torch.Tensor]:
    dev = torch.device("cuda")
    out = dict(
        y=torch.empty((n, 3 * D), dtype=torch.float8_e4m3fn, device=dev),
        mean=torch.empty((n,), dtype=torch.float32, device=dev),
        rstd=torch.empty((n,), dtype=torch.float32, device=dev),
        mask=torch.empty((n, D), dtype=torch.int8, device=dev),
        packed=torch.empty((n, D // 2), dtype=torch.uint8, device=dev),
    )
    out["y"].view(torch.uint8).fill_(0x5A + junk)
    out["mask"].fill_(0x7F - junk)
    out["packed"].fill_(0xA5 - junk)
    out["mean"].view(torch.int32).fill_(0x7FC00011 + junk)
    out["rstd"].view(torch.int32).fill_(0x7FC00021 + junk)
    return out


def _same(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


_ENV_ARMED = {
    "DLRM_HSTU_FP8_LNMD_FWD_MASKGEN": "1",
    "DLRM_HSTU_FP8_MASK16": "1",
    "DLRM_HSTU_FP8_LNMD_FWD_V2": "1",
    "DLRM_HSTU_FP8_LNMD_FWD_V3": "1",
    PK_V3_KNOB: "1",  # ship17 (111) as bound by yambda_5b.gin: the pk writer is the V3 body
}


@unittest.skipUnless(_gfx950(), "needs a gfx950 GPU and Triton (the kernels' inline asm)")
class NibWritersGpuTest(unittest.TestCase):
    def setUp(self) -> None:
        self.st = _State()
        self.st.reset()
        self.thl = self.st.thl
        self.env = mock.patch.dict(os.environ, dict(_ENV_ARMED))
        self.env.start()
        nib.reset_for_testing()

    def tearDown(self) -> None:
        self.env.stop()
        self.st.restore()
        nib.reset_for_testing()

    def test_bitwise_vs_shipped_and_mgpk(self) -> None:
        thl = self.thl
        for n in (1, 7, 4097, 65536):
            x, u, w, b = _gpu_inputs(n)
            for p in (1e-4, 0.1, 0.5):
                keep_min16, kk2 = thl._lnmd_maskgen_kk2(p)
                for seed in SEEDS2:
                    for pk in (False, True):
                        ref = _gpu_bufs(n, 0)
                        kernel = thl._ln_mul_dropout_fwd_maskgen_pk if pk else thl._ln_mul_dropout_fwd_maskgen
                        kernel[((n + 7) // 8,)](
                            x, u, ref["y"], w, b, ref["mean"], ref["rstd"], ref["mask"], n, D, EPS, p, x.stride(0),
                            u.stride(0), ref["y"].stride(0), ref["mask"].stride(0), INV, keep_min16, kk2, seed,
                            BLOCK_D=D, NUM_MASKS=3, MUL_U_ACTIVATION_TYPE="silu", OUTPUT_FP8=True, SILU_U=True,
                            num_warps=1)
                        for junk in (1, 2):  # an unwritten byte cannot equal both junk patterns
                            out = _gpu_bufs(n, junk)
                            thl._lnmd_fwd_maskgen_nib_launch(
                                x, u, out["y"], w, b, out["mean"], out["rstd"], out["packed"], n, D, EPS, p, INV,
                                keep_min16, kk2, seed, D, True, True, "silu", True, True, pk)
                            torch.cuda.synchronize()
                            tag = f"n={n} p={p} seed={seed} pk={pk} junk={junk}"
                            self.assertEqual(_same(out["y"], ref["y"]), 0, tag)
                            self.assertEqual(_same(out["mean"], ref["mean"]), 0, tag)
                            self.assertEqual(_same(out["rstd"], ref["rstd"]), 0, tag)
                            self.assertEqual(_same(out["packed"], nib.pack_ref(ref["mask"])), 0, tag)
                            self.assertTrue(torch.equal(nib.unpack(out["packed"]), ref["mask"]), tag)
                        self.assertEqual(int(((ref["mask"].to(torch.int32) & ~7) != 0).sum()), 0)
            del x, u, w, b

    def _impl(self, x, u, w, b, seed: int, tag: int = -1) -> Tuple[Tuple[torch.Tensor, ...], str]:
        thl = self.thl
        thl.set_lnmd_layer_tag(tag)
        idx = torch.empty(0, dtype=torch.int64, device="cuda")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = thl._triton_layer_norm_mul_dropout_fwd_impl(x, u, w, b, EPS, 0.1, True, True, True, True, "silu",
                                                              seed, idx, True, INV)
        torch.cuda.synchronize()
        return out, buf.getvalue()

    def test_hook_end_to_end_predicate_forced(self) -> None:
        thl = self.thl
        n = (1 << 20) + 8 * 7 + 5
        x, u, w, b = _gpu_inputs(n)
        os.environ["DLRM_HSTU_FP8_MASKGEN_PK"] = "1"
        os.environ["DLRM_HSTU_FP8_MASKGEN_PK_LAYERS"] = "1"
        os.environ.pop(KNOB, None)
        ref, text = self._impl(x, u, w, b, SEEDS2[0], tag=0)
        self.assertEqual(ref[3].dtype, torch.int8)
        self.assertNotIn("NIB", text)
        sh, pk = thl._ln_mul_dropout_fwd_maskgen_nib, thl._ln_mul_dropout_fwd_maskgen_pk_nib
        for tag, want in ((0, "sh"), (1, "pk")):
            self.st.reset()
            counts = {"sh": 0, "pk": 0}

            def counting(kernel, key):
                class _C:
                    def __getitem__(self, grid):
                        launch = kernel[grid]

                        def run(*a, **k):
                            counts[key] += 1
                            return launch(*a, **k)

                        return run

                return _C()

            with mock.patch.object(thl, "_lnmd_mask_nib_why_not", lambda *a, **k: None), \
                    mock.patch.object(thl, "_ln_mul_dropout_fwd_maskgen_nib", counting(sh, "sh")), \
                    mock.patch.object(thl, "_ln_mul_dropout_fwd_maskgen_pk_nib", counting(pk, "pk")):
                got, text = self._impl(x, u, w, b, SEEDS2[0], tag=tag)
            self.assertEqual(counts, {"sh": int(want == "sh"), "pk": int(want == "pk")}, tag)
            self.assertEqual((got[3].dtype, tuple(got[3].shape)), (torch.uint8, (n, D // 2)), tag)
            for i in range(3):
                self.assertEqual(_same(got[i], ref[i]), 0, (tag, i))
            self.assertTrue(torch.equal(nib.unpack(got[3]), ref[3]), tag)
            self.assertEqual(text.count("armed OK: LN*U*dropout keep mask NIB ACTIVE"), 1, tag)
        # a raising NIB launch: latched, today's bytes for the same call
        self.st.reset()

        class _Raising:
            def __getitem__(self, grid):
                def launch(*a, **k):
                    raise RuntimeError("synthetic compile failure")

                return launch

        with mock.patch.object(thl, "_lnmd_mask_nib_why_not", lambda *a, **k: None), \
                mock.patch.object(thl, "_ln_mul_dropout_fwd_maskgen_nib", _Raising()):
            got, text = self._impl(x, u, w, b, SEEDS2[0], tag=0)
        self.assertEqual(got[3].dtype, torch.int8)
        for i in range(4):
            self.assertEqual(_same(got[i], ref[i]), 0, i)
        self.assertIn("LN*U*dropout keep mask NIB DISABLED", text)
        del x, u, w, b

    def test_warmup_launch(self) -> None:
        n = (1 << 20) + 1
        x, u, w, b = _gpu_inputs(n)
        self.assertEqual(self.thl.lnmd_mask_nib_warmup_launch(x, u, w, b, EPS, 0.1, 2**40, 64.0, True), "nib=sh+pk")
        torch.cuda.synchronize()
        # the startup byte check (review R1-F5) on the real kernels: both writers equal their int8 kernels
        verdict: Dict[str, str] = {}
        got = self.thl.lnmd_mask_nib_warmup_launch(x, u, w, b, EPS, 0.1, 2**40 + 1, 64.0, True, compare=True,
                                                   verdict=verdict)
        torch.cuda.synchronize()
        self.assertEqual((got, verdict), ("nib=sh+pk cmp=ok", {}))
        self.assertIsNone(self.thl._LNMD_MASK_NIB_BROKEN)


# ------------------------------------------------------------------------------------------------ ship17 (111) V3
_PK_V3_HEAD = "# (111) NIB pk writer V3 "
_PK_V3_ARMED = "armed OK: (111) LNMD NIB pk writer V3 ACTIVE (DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3=1): "
_PK_V3_OFF = "(111) LNMD NIB pk writer V3 OFF (DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3="
_PK_V3_NOT = "(111) LNMD NIB pk writer V3 NOT armed (DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3=1): "
_PK_V3_KERNELS = ("_ln_mul_dropout_fwd_maskgen_pk_nib", "_ln_mul_dropout_fwd_maskgen_pk_nib_r1")


def _jit_spans(tree: ast.Module) -> Dict[str, Tuple[int, int]]:
    """name -> (first line incl. the decorator, last line) of every top-level @triton.jit function."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and any(_dotted(d) == "triton.jit" for d in n.decorator_list):
            out[n.name] = (min([n.lineno] + [d.lineno for d in n.decorator_list]), n.end_lineno)
    return out


def _line_of(src: str, offset: int) -> int:
    return src.count("\n", 0, offset) + 1


class PkV3StructureTest(unittest.TestCase):
    """ship17 (111) line neutrality, AST of ops/triton/triton_hstu_linear.py (no import). Triton's cache key of a
    @triton.jit function is its source plus its starting line (and its callees'); the TSPEC table keys on it too. The
    (111) row may change no line above its block except in place: the V3 body fills the old body's lines and the
    launch helper's kernel line is edited in place. The tree-external pin (every @triton.jit function's starting line
    and source sha vs 544e691 / cand_v1b) is the row's CHECKS/line_neutrality.txt."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = _tree()
        with open(_THL) as f:
            cls.src = f.read()

    def test_v3_body_fills_the_old_lines(self) -> None:
        sp = _jit_spans(self.tree)
        v3, r1 = sp["_ln_mul_dropout_fwd_maskgen_pk_nib"], sp["_ln_mul_dropout_fwd_maskgen_pk_nib_r1"]
        # r1 is the old body verbatim (renamed): the V3 body spans exactly as many lines, so nothing below it moves
        self.assertEqual(v3[1] - v3[0], r1[1] - r1[0])
        # and the next kernel keeps its distance, as in 544e691 (_pk_nib ends 7394, _c32_nib starts 7397)
        self.assertEqual(sp["_c32_nib"][0], v3[1] + 3)
        # the same decorator, def and parameter lines (only the name differs)
        lines = self.src.splitlines()
        f3 = _func(self.tree, "_ln_mul_dropout_fwd_maskgen_pk_nib")
        f1 = _func(self.tree, "_ln_mul_dropout_fwd_maskgen_pk_nib_r1")
        head3 = lines[v3[0] - 1:f3.body[0].lineno - 1]
        head1 = [ln.replace("_ln_mul_dropout_fwd_maskgen_pk_nib_r1(", "_ln_mul_dropout_fwd_maskgen_pk_nib(")
                 for ln in lines[r1[0] - 1:f1.body[0].lineno - 1]]
        self.assertEqual(head3, head1)
        self.assertEqual(len(head3), 13)
        # V3 calls its own helpers, r1 the shipped ones
        calls3 = {_dotted(n.func) for n in ast.walk(f3) if isinstance(n, ast.Call)}
        calls1 = {_dotted(n.func) for n in ast.walk(f1) if isinstance(n, ast.Call)}
        self.assertTrue({"_tp_philox_v3", "_tp_nib_v3", "_row_c32_pk_nib_v3"} <= calls3, calls3)
        self.assertEqual({"_row_c32_pk_nib", "_mg_pk_word", "_load_row", "_philox4x32_10_b3"} & calls3, set())
        self.assertTrue({"_row_c32_pk_nib", "_mg_pk_word", "_load_row", "_philox4x32_10_b3"} <= calls1, calls1)
        self.assertEqual({c for c in calls1 if c.endswith("_v3")}, set())

    def test_block_follows_the_nib_kernels_and_precedes_the_host_side(self) -> None:
        sp = _jit_spans(self.tree)
        nib_banner = self.src.index("# DLRM_HSTU_FP8_LNMD_MASK_NIB (\"NIB\"")
        block = self.src.index("# (111) _pk_nib V3 (")
        host = self.src.index("# NIB host side: ")
        erm = self.src.index("# r14 E-RM (DLRM_HSTU_FP8_BWD_ERM")
        self.assertLess(nib_banner, block)
        self.assertLess(block, host)
        self.assertLess(host, erm)
        b, h = _line_of(self.src, block), _line_of(self.src, host)
        for name in _PK_V3_BLOCK:
            self.assertTrue(b < sp[name][0] < h, name)
        # every other @triton.jit function lies above the block: inserting it moved none of them
        self.assertEqual([n for n, s in sp.items() if n not in _PK_V3_BLOCK and s[1] > b], [])
        # the block's banner follows the last NIB kernel after two blank lines: the V3 helpers' starting lines, which
        # Triton hashes into _ln_mul_dropout_fwd_maskgen_pk_nib's cache key (and the TSPEC table's fn_cache_key /
        # kernel_hash), do not depend on the NIB host code below the block
        self.assertEqual(b - 1, sp["_ln_mul_dropout_fwd_maskgen_nib"][1] + 3)
        self.assertTrue(b < _func(self.tree, "_lnmd_mask_nib_pk_kernel").lineno < h)

    def test_launch_line_in_place(self) -> None:
        fn = _func(self.tree, "_lnmd_fwd_maskgen_nib_launch")
        assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "kernel" for t in n.targets)]
        self.assertEqual(len(assigns), 1)
        (a,) = assigns
        self.assertEqual(a.lineno, a.end_lineno)  # one physical line, edited in place
        self.assertEqual(ast.unparse(a.value),
                         "_lnmd_mask_nib_pk_kernel(N, D, BLOCK_D) if pk else _ln_mul_dropout_fwd_maskgen_nib")
        # no other launch of either pk writer in the module (only the selector returns them)
        for name in _PK_V3_KERNELS:
            subs = [n for n in ast.walk(self.tree) if isinstance(n, ast.Subscript) and _dotted(n.value) == name]
            self.assertEqual(subs, [], name)
        sel = _func(self.tree, "_lnmd_mask_nib_pk_kernel")
        rets = sorted(_dotted(n.value) for n in ast.walk(sel) if isinstance(n, ast.Return))
        self.assertEqual(rets, sorted(_PK_V3_KERNELS))


# sha256 of each (111) function's text (decorator line through last line, as in the module), pinned in the (111) fix
# round to the code kitV tested (r16's iteration-60 lab kit, seal b1d40d93): the 8 helpers are the lab alias
# module's (ops/triton/triton_hstu_linear_pknib_v3_lab.py) functions byte for byte; the V3 kernel's code lines after
# its docstring are the alias kernel _ln_mul_dropout_fwd_maskgen_pk_nib_v3's; r1, with the def name of the kernel it
# replaces, is 544e691's _ln_mul_dropout_fwd_maskgen_pk_nib. An edit to any of them changes the V3 instruction stream
# or its Triton cache key (and so the TSPEC table's fn_cache_key / kernel_hash of _pk_nib): redo the ISA tie and the
# table refresh, then the pins.
_PK_V3_SRC_SHA256 = {
    "_mg_bytemask_n_v3": "bd2c6b95df041d52bb5aa1bdf05704dea6df2938e2c9b1c4cf8ed02dabc9c9cc",
    "_mg_j2_v3": "5f8113591121e49633d2b1ed98425e5b770fd3fd122a816acae1126f6f75d5af",
    "_mg_split8_v3": "cd4aec2916b36190751e2ec38d5c3e9b493bdd83dfbcea4b376f5525383ab5cc",
    "_mg_lso_v3": "5051c9a080dbe6ff4c7cf5882a99814e5a1dea1a2585ddd091d4317ac52d2ab4",
    "_tp_word_v3": "514eb2be52993d5a7e9a6cdcbbeacf135d6b9aaea9588f6aa23a5d45fc574a44",
    "_tp_nib_v3": "7c026ae9d065888a209cd058fdd3fa9ae01f452a8c4d8229ecf5f91d945de563",
    "_tp_philox_v3": "c8f77c2dbc5b762385cb1053f200057e6dbe20f0bafc8b6e1aede72fa42b7070",
    "_row_c32_pk_nib_v3": "a09c5d26d1d0a9a9423e65053d6d837b0555598418aa36bd56b78283d0f44567",
    "_ln_mul_dropout_fwd_maskgen_pk_nib": "2296431b9581905d18607b61bc3a978b9980bf3af45bf2463881a1d7c4a8ef9a",
    "_ln_mul_dropout_fwd_maskgen_pk_nib_r1": "cb7e6d8fa594491bda64e7abee888c252b6d59047890a301fb1626815a3e5e78",
}
_PK_V3_CODE_SHA256 = "83629a0fe04d6491069d06d424c2196724a793e49300b0dc3060b8c75ade6483"  # == the kitV alias kernel's
_PK_R1_AS_544E691_SHA256 = "5b2ea0e8d7297d9427b2ec99e1c881183ed18bea6d598b6914c933a709ebff61"  # == 544e691's _pk_nib


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class PkV3SourcePinTest(unittest.TestCase):
    """ship17 (111): the shipped V3 code is the code kitV tested (CPU, AST of ops/triton/triton_hstu_linear.py, no
    import). The GPU classes check the bytes on gfx950 only; these pins catch an edit or a merge resolution inside the
    V3 kernel, its 8 helpers or r1 on any host."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = _tree()
        with open(_THL) as f:
            cls.lines = f.read().splitlines(keepends=True)
        cls.spans = _jit_spans(cls.tree)

    def _text(self, name: str) -> str:
        start, end = self.spans[name]
        return "".join(self.lines[start - 1:end])

    def test_function_texts_are_pinned(self) -> None:
        self.assertEqual({n: _sha256(self._text(n)) for n in _PK_V3_SRC_SHA256}, _PK_V3_SRC_SHA256)
        self.assertEqual(set(_PK_V3_SRC_SHA256), set(_PK_V3_BLOCK) | {"_ln_mul_dropout_fwd_maskgen_pk_nib"})

    def test_v3_code_lines_are_the_lab_alias_kernels(self) -> None:
        fn = _func(self.tree, "_ln_mul_dropout_fwd_maskgen_pk_nib")
        doc = fn.body[0]
        self.assertTrue(isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)
                        and isinstance(doc.value.value, str))
        self.assertEqual(_sha256("".join(self.lines[doc.end_lineno:fn.end_lineno])), _PK_V3_CODE_SHA256)
        # the docstring carries no item / lab label (a relabel would re-key the kernel): those live in the banner
        self.assertIsNone(re.search(r"\(1\d\d\)|\br1[0-9]\b|ship1\d|iteration|kitV", doc.value.value))

    def test_r1_is_the_544e691_body(self) -> None:
        text = self._text("_ln_mul_dropout_fwd_maskgen_pk_nib_r1")
        old = text.replace("def _ln_mul_dropout_fwd_maskgen_pk_nib_r1(", "def _ln_mul_dropout_fwd_maskgen_pk_nib(", 1)
        self.assertNotEqual(old, text)
        self.assertEqual(_sha256(old), _PK_R1_AS_544E691_SHA256)


def _pk_v3_block(lines: List[str]) -> List[str]:
    i_bind = next(i for i, ln in enumerate(lines) if ln.startswith(f"apply_env_bootstrap.{PK_V3_KNOB} "))
    i_head = max(i for i in range(i_bind) if lines[i].startswith(_PK_V3_HEAD))
    return lines[i_head:i_bind]


class PkV3GinBootstrapTest(unittest.TestCase):
    def test_bootstrap_one_parameter_one_env_wins_bind(self) -> None:
        with open(_BOOT) as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(names.count(PK_V3_KNOB), 1)
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        self.assertEqual(names.index(PK_V3_KNOB), names.index(KNOB) + 1)  # next to the NIB knob
        arg = next(a for a in fn.args.args if a.arg == PK_V3_KNOB)
        self.assertEqual(ast.unparse(arg.annotation), "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[PK_V3_KNOB], ast.Constant) and defaults[PK_V3_KNOB].value is None)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == PK_V3_KNOB]
        self.assertEqual(len(calls), 1)
        self.assertEqual(getattr(calls[0].func, "id", None), "_bind_bool")
        self.assertEqual([getattr(a, "id", None) for a in calls[0].args[1:]], [PK_V3_KNOB])

    def test_gin_exactly_one_binding_true_after_nib(self) -> None:
        lines = _gin_lines()
        pat = re.compile(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % PK_V3_KNOB)
        self.assertEqual([m.group(1) for m in map(pat.match, lines) if m], ["True"])
        names = [m.group(1) for m in (re.match(r"^\s*([\w./]+)\s*=", ln) for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        i_nib = next(i for i, ln in enumerate(lines) if ln.startswith(f"apply_env_bootstrap.{KNOB} "))
        i_bind = next(i for i, ln in enumerate(lines) if ln.startswith(f"apply_env_bootstrap.{PK_V3_KNOB} "))
        block = _pk_v3_block(lines)
        self.assertEqual((lines[i_nib + 1], i_bind - len(block)), ("#", i_nib + 2))  # right after (94)'s binding

    def test_item_block_documents_the_knob(self) -> None:
        block = _pk_v3_block(_gin_lines())
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = " ".join(ln[1:].strip() for ln in block)
        for marker in ("(111) NIB pk writer V3", "r16", "code default OFF", "bound ON below", "kitV", "b1d40d93",
                       "two runs on two nodes", "360/360", "VOID", "two A/A cells each", "report-only", "kernel leg",
                       "dropped from the candidate if that kernel leg does not PASS or the span reads slower",
                       "right after the NIB kernels", "WHAT:", "NUMERICS: BIT-IDENTICAL", "MLPerf", "no data path",
                       "KERNEL:", "hash_same", "src_changed", "table refresh that lands with this item", "ARMED",
                       "armed OK: (111) LNMD NIB pk writer V3 ACTIVE (DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3=1)",
                       "Must be ABSENT:", '"(111) LNMD NIB pk writer V3 OFF"',
                       '"(111) LNMD NIB pk writer V3 NOT armed"', f"Kill switch: {PK_V3_KNOB}=0",
                       "pre-set value wins", "only the value 1 arms it", "_ln_mul_dropout_fwd_maskgen_pk_nib_r1",
                       "TSPEC: r1 has no table rows", "after RUN_START", "0.6-0.7 s",
                       # main's FV-2 ruling: the (111) line prints before RUN_START (FP8 warmup item L4n)
                       "once per process, BEFORE RUN_START", "item L4n", "printed right after the item",
                       "no training launch prints it", "before RUN_START in the same way"):
            self.assertIn(marker, text)
        # no lab job id or node name in the shipped comment (the submission sanitizer would half-mask them; they are
        # in the row's .md)
        self.assertIsNone(re.search(r"\b\d{5,6}\b|\b[a-h]\d{2}-(?:08|18|28|38|40)\b", text), text)

    def test_documented_lines_are_the_code_lines(self) -> None:
        with open(_THL) as f:
            src = f.read()
        for piece in (f'_LNMD_MASK_NIB_PK_V3_KNOB = "{PK_V3_KNOB}"', '_LNMD_MASK_NIB_PK_V3_ITEM = "(111)"',
                      'f"armed OK: {_LNMD_MASK_NIB_PK_V3_ITEM} LNMD NIB pk writer V3 ACTIVE "',
                      'f"({_LNMD_MASK_NIB_PK_V3_KNOB}=1): _ln_mul_dropout_fwd_maskgen_pk_nib runs the V3 body',
                      'f"{_LNMD_MASK_NIB_PK_V3_ITEM} LNMD NIB pk writer V3 {why}: '
                      '_ln_mul_dropout_fwd_maskgen_pk_nib_r1 runs "',
                      'f"OFF ({_LNMD_MASK_NIB_PK_V3_KNOB}=', 'f"NOT armed ({_LNMD_MASK_NIB_PK_V3_KNOB}=1): '):
            self.assertIn(piece, src)

    @unittest.skipUnless(_HAS_GIN, "gin not importable")
    def test_phase1_bootstrap_child(self) -> None:
        code = (
            "import os, sys, gin\n"
            "from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot\n"
            "gin.parse_config_file(sys.argv[1], skip_unknown=True)\n"
            "gin.parse_config(sys.argv[2:], skip_unknown=True)\n"
            "boot.apply_env_bootstrap()\n"
            f"print('VALUE=' + repr(os.environ.get({PK_V3_KNOB!r})))\n"
        )
        # gin binds True ("1"); a pre-set env value wins (env 0 is the kill switch); a gin False binding (a held-off
        # tree) gives "0", the OFF line's other state
        for pre, bind, want in ((None, None, "'1'"), ("1", None, "'1'"), ("0", None, "'0'"), (None, "False", "'0'")):
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=os.pathsep.join([_ROOT] + [p for p in sys.path if p]))
            env.pop(PK_V3_KNOB, None)
            if pre is not None:
                env[PK_V3_KNOB] = pre
            extra = [f"apply_env_bootstrap.{PK_V3_KNOB} = {bind}"] if bind else []
            out = subprocess.run([sys.executable, "-B", "-c", code, _GIN] + extra, env=env, capture_output=True,
                                 text=True)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            self.assertIn(f"VALUE={want}", out.stdout, (pre, bind, out.stdout[-500:]))


@unittest.skipUnless(_importable(), "triton_hstu_linear not importable (torch / triton missing)")
class PkV3SelectorTest(unittest.TestCase):
    """ship17 (111): _lnmd_mask_nib_pk_kernel and the launch helper with mocked kernels (CPU, no launch)."""

    def setUp(self) -> None:
        self.st = _State()
        self.st.reset()
        self.thl = self.st.thl
        self.env = mock.patch.dict(os.environ, {})
        self.env.start()
        os.environ.pop(PK_V3_KNOB, None)

    def tearDown(self) -> None:
        self.env.stop()
        self.st.restore()

    def _pick(self, n: int = 40, d: int = D, block_d: int = D) -> Tuple[Any, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            k = self.thl._lnmd_mask_nib_pk_kernel(n, d, block_d)
        return k, buf.getvalue()

    def test_selector_states_and_lines(self) -> None:
        thl = self.thl
        v3, r1 = thl._ln_mul_dropout_fwd_maskgen_pk_nib, thl._ln_mul_dropout_fwd_maskgen_pk_nib_r1
        self.assertIsNot(v3, r1)
        # knob absent (the code default): r1, no line
        self.assertEqual(self._pick(), (r1, ""))
        # "1" (the gin binding): V3, one armed line per process
        os.environ[PK_V3_KNOB] = "1"
        thl.set_lnmd_layer_tag(1)
        k, out = self._pick(n=24)
        self.assertIs(k, v3)
        self.assertEqual(out.count(_PK_V3_ARMED), 1)
        self.assertIn("first V3 launch on layer tag 1, N=24 [rank ", out)
        # the mark holds the line it printed (the text before the rank tag): what the FP8 warmup prints again
        mark = thl._LNMD_MASK_NIB_PK_V3_ARMED_LOGGED
        self.assertTrue(mark.startswith(_PK_V3_ARMED) and mark.endswith("first V3 launch on layer tag 1, N=24"), mark)
        self.assertEqual(_pk_v3_lines(out), [_rank_tagged(mark, out)])
        self.assertEqual(thl._LNMD_MASK_NIB_PK_V3_OFF_LOGGED, "")
        self.assertEqual(self._pick(), (v3, ""))
        os.environ[PK_V3_KNOB] = " 1 "  # surrounding whitespace ignored, as lnmd_mask_nib.requested()
        self.assertEqual(self._pick(), (v3, ""))
        # "1" with D != BLOCK_D: r1 and one NOT armed line (V3 addresses the x / y planes with BLOCK_D)
        os.environ[PK_V3_KNOB] = "1"
        k, out = self._pick(d=256, block_d=512)
        self.assertIs(k, r1)
        self.assertEqual(out.count(_PK_V3_NOT + "D=256 != BLOCK_D=512"), 1)
        self.assertIn("_ln_mul_dropout_fwd_maskgen_pk_nib_r1 runs", out)
        self.assertNotIn("armed OK", out)
        # the env kill switch "0" (and any other value): r1, one OFF line per process
        for i, (value, shown) in enumerate((("0", "0"), ("true", "true"), ("", "<empty>"), ("2", "2"))):
            self.st.reset()
            os.environ[PK_V3_KNOB] = value
            k, out = self._pick()
            self.assertIs(k, r1, value)
            want = (_PK_V3_OFF + shown + ": not 1; 0 is a gin False binding or the env kill switch; only the value 1 "
                    "arms it): _ln_mul_dropout_fwd_maskgen_pk_nib_r1 runs")
            self.assertEqual(out.count(want), 1, value)
            self.assertTrue(thl._LNMD_MASK_NIB_PK_V3_OFF_LOGGED.startswith(want), value)
            self.assertEqual(_pk_v3_lines(out), [_rank_tagged(thl._LNMD_MASK_NIB_PK_V3_OFF_LOGGED, out)])
            self.assertEqual(thl._LNMD_MASK_NIB_PK_V3_ARMED_LOGGED, "")
            self.assertEqual(self._pick(), (r1, ""))  # once per process
        # the kill-switch line and the armed line are independent marks
        os.environ[PK_V3_KNOB] = "1"
        k, out = self._pick()
        self.assertIs(k, v3)
        self.assertEqual(out.count(_PK_V3_ARMED), 1)

    def test_launch_helper_launches_the_pick_with_the_same_arguments(self) -> None:
        thl = self.thl
        names = _PK_V3_KERNELS + ("_ln_mul_dropout_fwd_maskgen_nib",)
        recs = {name: _Rec(name) for name in names}
        n = 24
        x, u, w, b = _cpu_xu(n)
        y = torch.empty((n, 3 * D), dtype=torch.uint8)
        mean, rstd = torch.empty(n), torch.empty(n)
        packed = torch.empty((n, D // 2), dtype=torch.uint8)
        k16, kk2 = thl._lnmd_maskgen_kk2(0.1)
        cases = ((None, True, names[1]), ("1", True, names[0]), ("0", True, names[1]), ("1", False, names[2]),
                 ("0", False, names[2]), (None, False, names[2]))
        with contextlib.ExitStack() as st:
            for name in names:
                st.enter_context(mock.patch.object(thl, name, recs[name]))
            st.enter_context(contextlib.redirect_stdout(io.StringIO()))
            for value, pk, want in cases:
                if value is None:
                    os.environ.pop(PK_V3_KNOB, None)
                else:
                    os.environ[PK_V3_KNOB] = value
                before = {k: len(r.calls) for k, r in recs.items()}
                thl._lnmd_fwd_maskgen_nib_launch(x, u, y, w, b, mean, rstd, packed, n, D, EPS, 0.1, INV, k16, kk2, 99,
                                                 D, True, True, "silu", True, True, pk)
                after = {k: len(r.calls) for k, r in recs.items()}
                self.assertEqual({k: after[k] - before[k] for k in names}, {k: int(k == want) for k in names},
                                 (value, pk))
        (g3, a3, k3), (g1, a1, k1) = recs[names[0]].calls[0], recs[names[1]].calls[0]
        self.assertEqual(g3, ((n + 7) // 8,))
        self.assertEqual(g3, g1)
        self.assertEqual(k3, k1)
        self.assertEqual(len(a3), len(a1))
        for i, (p, q) in enumerate(zip(a3, a1)):
            self.assertTrue(p is q if isinstance(p, torch.Tensor) else p == q, i)
        self.assertEqual(k3["NUM_MASKS"], 3)
        self.assertEqual(k3["num_warps"], 1)

    def test_line_marks_hold_their_line_and_are_kept_by_the_fp8_warmup(self) -> None:
        """main's FV-2 ruling (the v1 test pinned the opposite: plain bools the warmup restored, so the line printed
        again at the first training launch): the two marks are str (the line printed, "" = none yet) and are in
        fp8_warmup's _KEEP_GLOBALS, so its scalar restore leaves them; triton_hstu_linear's other plain scalars are
        still snapshotted and restored (_SCALAR_MODULES); fp8_warmup reprints exactly these two marks."""
        thl_tree = _tree()
        for k in _PK_V3_GLOBALS:  # the module's own initial value (setUp's reset sets "" as well)
            init = [n.value for n in thl_tree.body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == k for t in n.targets)]
            self.assertEqual([ast.literal_eval(v) for v in init], [""], k)
        with open(_WARM) as f:
            tree = ast.parse(f.read())
        scal = next(n for n in tree.body if isinstance(n, ast.AnnAssign)
                    and getattr(n.target, "id", None) == "_SCALAR_MODULES")
        self.assertIn('_P_TRITON + "triton_hstu_linear"', [ast.unparse(e).replace("'", '"') for e in scal.value.elts])
        pairs = _keep_pairs(tree)
        for k in _PK_V3_GLOBALS:
            self.assertIn(("triton_hstu_linear", k), pairs)
        marks = next(n for n in tree.body if isinstance(n, ast.AnnAssign)
                     and getattr(n.target, "id", None) == "_PK_V3_MARKS")
        self.assertEqual(tuple(e.value for e in marks.value.elts), _PK_V3_GLOBALS)


# ------------------------------------------------------------------------------------------------ ship17 (111) FV-2
# Main's FV-2 ruling (2026-10-15): the ARMED line (and the OFF / NOT armed lines) print once BEFORE RUN_START,
# reprinted after the FP8 warmup's item L4n as (105) does; no (111) line inside the timed region.
_PK_V3_ANY = "(111) LNMD NIB pk writer V3 "  # every (111) selector line (ARMED, OFF, NOT armed) contains it


def _keep_pairs(tree: ast.Module) -> set:
    """(module suffix, global) pairs of fp8_warmup._KEEP_GLOBALS (AST)."""
    keep = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "_KEEP_GLOBALS" for t in n.targets))
    pairs = set()
    for node in ast.walk(keep.value):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2 and isinstance(node.elts[1], ast.Constant):
            first = node.elts[0]
            if isinstance(first, ast.BinOp) and isinstance(first.right, ast.Constant):
                pairs.add((first.right.value, node.elts[1].value))
    return pairs


def _rank_tagged(mark: str, out: str) -> str:
    """mark plus the ' [rank N]' tag arm_print_all_ranks appends (N as printed in out)."""
    m = re.search(r" \[rank (\d+)\]", out)
    return f"{mark} [rank {m.group(1) if m else '?'}]"


def _pk_v3_lines(out: str) -> List[str]:
    """The (111) selector lines in out (other first-use lines, e.g. an import's note, are not counted)."""
    return [ln for ln in out.splitlines() if _PK_V3_ANY in ln]


class PkV3WarmupSourceTest(unittest.TestCase):
    """ship17 (111) FV-2, AST of dlrm_v4/train/fp8_warmup.py (no import): after EVERY item L4n runs (outside the
    item's stdout capture, before the startup byte check's latch), _nib_items prints the (111) line that item latched,
    through triton_hstu_linear's own arm_print_all_ranks."""

    def test_nib_items_reprints_after_each_item(self) -> None:
        with open(_WARM) as f:
            src = f.read()
        tree = ast.parse(src)
        fn = _func(tree, "_nib_items")
        loops = [n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.target) == "seed"]
        self.assertEqual(len(loops), 1)
        body = [ast.unparse(s) for s in loops[0].body]
        i_run = next(i for i, s in enumerate(body) if s.startswith("r.run('L4n'"))
        self.assertEqual(body[i_run - 1], "pk_v3 = _pk_v3_marks(thl)")
        self.assertEqual(body[i_run + 1], "_pk_v3_reprint(thl, pk_v3)")
        self.assertTrue(body[i_run + 2].startswith("if verdict.get('mismatch'):"), body[i_run + 2])
        self.assertEqual(src.count("_pk_v3_reprint("), 2)  # its def and this one call
        self.assertEqual(src.count("_pk_v3_marks("), 2)
        self.assertIn("thl.arm_print_all_ranks(line)", ast.unparse(_func(tree, "_pk_v3_reprint")))


@unittest.skipUnless(_importable() and _warmup_importable(), "triton_hstu_linear / fp8_warmup not importable")
class PkV3ReprintTest(unittest.TestCase):
    """ship17 (111) FV-2: the selector's line, printed inside an FP8 warmup item (the real _Runner, whose stdout
    capture drops it), is printed once by _pk_v3_reprint after the item, outside the capture, with the selector's
    text and rank tag; a later item prints nothing (the mark is kept). Every state: ARMED ("1"), OFF ("0"), NOT armed
    ("1" with D != BLOCK_D, unreachable in production); knob absent: no line at all."""

    def setUp(self) -> None:
        self.st = _State()
        self.st.reset()
        self.thl = self.st.thl
        self.fw = _fp8_warmup()
        self.env = mock.patch.dict(os.environ, {})
        self.env.start()
        for k in (PK_V3_KNOB, "DLRM_HSTU_FP8_ARM_PRINT_ATOMIC"):
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        self.env.stop()
        self.st.restore()

    def _item(self, value: Optional[str], d: int = D) -> Tuple[str, int, int, str]:
        fw, thl = self.fw, self.thl
        self.st.reset()
        if value is None:
            os.environ.pop(PK_V3_KNOB, None)
        else:
            os.environ[PK_V3_KNOB] = value

        def pick() -> None:
            thl._lnmd_mask_nib_pk_kernel(24, d, D)

        r = fw._Runner(torch.device("cpu"), _WarmBufs(), True, set(), False)
        out = io.StringIO()
        with mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None), contextlib.redirect_stdout(out):
            before = fw._pk_v3_marks(thl)
            r.run("L4n", "lnmd_mask_nib_writers", "pk_v3_reprint_test", pick)
            inside = out.getvalue()
            n = fw._pk_v3_reprint(thl, before)
            n2 = fw._pk_v3_reprint(thl, fw._pk_v3_marks(thl))  # a later item (the mark kept): nothing
        self.assertEqual((r.ok, r.failed), (1, 0))
        return inside, n, n2, out.getvalue()

    def test_every_state_prints_once_after_the_item(self) -> None:
        thl = self.thl
        for value, d, head, mark in (
                ("1", D, _PK_V3_ARMED, "_LNMD_MASK_NIB_PK_V3_ARMED_LOGGED"),
                ("0", D, _PK_V3_OFF + "0: not 1; 0 is a gin False binding", "_LNMD_MASK_NIB_PK_V3_OFF_LOGGED"),
                ("1", 384, _PK_V3_NOT + "D=384 != BLOCK_D=512", "_LNMD_MASK_NIB_PK_V3_OFF_LOGGED")):
            inside, n, n2, out = self._item(value, d)
            self.assertEqual(_pk_v3_lines(inside), [], value)  # dropped by the item's stdout capture
            self.assertEqual((n, n2), (1, 0), value)
            line = getattr(thl, mark)
            self.assertTrue(line.startswith(head), (value, line))
            self.assertEqual(_pk_v3_lines(out), [_rank_tagged(line, out)], value)  # the selector's text and tag, once
        inside, n, n2, out = self._item(None)
        self.assertEqual((_pk_v3_lines(out), n, n2), ([], 0, 0))  # knob absent (the code default): no line at all


class _WarmBufs:
    """fp8_warmup._Buffers on the CPU: zero tensors (L4n's x / u), nothing to reset."""

    def reset(self) -> None:
        pass

    def act(self, shape: Tuple[int, ...], dtype: torch.dtype, big: bool = False) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)


@unittest.skipUnless(_importable() and _warmup_importable(), "triton_hstu_linear / fp8_warmup not importable")
class PkV3WarmupLineTest(unittest.TestCase):
    """ship17 (111) FV-2 end to end on the CPU, in production order: the FP8 warmup's state guards around item L4n
    (_Guards.take(verify=True); the real _Runner and _nib_items -> triton_hstu_linear.lnmd_mask_nib_warmup_launch ->
    _lnmd_fwd_maskgen_nib_launch -> the selector; _Guards.restore(); the VERIFY globals digest), then, after RUN_START,
    the first training launch through the production entry _lnmd_fwd_maskgen_nib_try (gate forced armed, MGPK on,
    layer tag 1). Mocked: the three NIB writer kernels (no compile, no launch) and the startup byte check (its int8
    reference kernels need the GPU); L4's rows are 24 / 25 and _ROWS_XL 40 / 41, two seeds: 8 items."""

    ROWS = [24, 25]

    def setUp(self) -> None:
        self.st = _State()
        self.st.reset()
        self.thl = self.st.thl
        self.fw = _fp8_warmup()
        try:  # what _Guards.take() imports
            for name in ("ops.flydsl.hstu_dq_split", "ops.flydsl.hstu_flydsl_mha", "ops.triton.bwd_fp8",
                         "ops.triton.triton_fp8"):
                importlib.import_module("generative_recommenders." + name)
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"fp8_warmup._Guards' modules not importable here: {error}")
        self.env = mock.patch.dict(os.environ, {})
        self.env.start()
        for k in (PK_V3_KNOB, "DLRM_HSTU_FP8_MASKGEN_PK_LAYERS", "DLRM_HSTU_FP8_ARM_PRINT_ATOMIC"):
            os.environ.pop(k, None)
        os.environ.update({KNOB: "1", "DLRM_HSTU_FP8_LNMD_FWD_MASKGEN": "1", "DLRM_HSTU_FP8_MASK16": "1",
                           "DLRM_HSTU_FP8_MASKGEN_PK": "1"})
        nib.reset_for_testing()
        self.recs = {name: _Rec(name) for name in _PK_V3_KERNELS + ("_ln_mul_dropout_fwd_maskgen_nib",)}
        self.check: Any = lambda *a, **k: None  # the startup byte check: every compare passes
        self.stack = contextlib.ExitStack()
        p = self.stack.enter_context
        for name, rec in self.recs.items():
            p(mock.patch.object(self.thl, name, rec))
        p(mock.patch.object(self.thl, "_gcn_arch_name", lambda index: "gfx950:sramecc+:xnack-"))
        p(mock.patch.object(self.thl, "_lnmd_mask_nib_check", lambda *a, **k: self.check(*a, **k)))
        p(mock.patch.object(self.fw, "_ROWS_XL", 40))
        p(mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None))

    def tearDown(self) -> None:
        self.stack.close()
        self.env.stop()
        self.st.restore()
        nib.reset_for_testing()

    def _warmup(self, keep: bool = True) -> Dict[str, Any]:
        """Item L4n inside the warmup's state guards; keep=False drops the two marks from _KEEP_GLOBALS (the v1
        design). Returns the stdout outside the items, the (111) arm_print_all_ranks calls as (line, printed outside
        every item capture?), _pk_v3_reprint's count per item, the guards, the runner and the marks after restore."""
        fw, thl = self.fw, self.thl
        outer = io.StringIO()
        prints: List[Tuple[str, bool]] = []
        counts: List[int] = []
        real_print, real_reprint = thl.arm_print_all_ranks, fw._pk_v3_reprint

        def spy_print(msg: str) -> None:
            prints.append((msg, sys.stdout is outer))
            real_print(msg)

        def spy_reprint(t: Any, before: Dict[str, Any]) -> int:
            counts.append(real_reprint(t, before))
            return counts[-1]

        keep_set = fw._KEEP_GLOBALS if keep else frozenset(k for k in fw._KEEP_GLOBALS if k[1] not in _PK_V3_GLOBALS)
        ol = types.SimpleNamespace(_output_norm_weight=torch.ones(D, dtype=torch.bfloat16))

        def ln_wb(width: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.ones(width, dtype=dtype), torch.zeros(width, dtype=dtype)

        guards = fw._Guards()
        r = fw._Runner(torch.device("cpu", 0), _WarmBufs(), True, set(), False)
        with mock.patch.object(thl, "arm_print_all_ranks", spy_print), \
                mock.patch.object(fw, "_pk_v3_reprint", spy_reprint), \
                mock.patch.object(fw, "_KEEP_GLOBALS", keep_set), contextlib.redirect_stdout(outer):
            guards.take(verify=True)
            try:
                fw._nib_items(r, thl, ol, ln_wb, D, 0.1, True, True, 64.0, list(self.ROWS))
            finally:
                guards.restore()
            guards.verify_globals()
        return dict(out=outer.getvalue(), prints=[(m, o) for m, o in prints if _PK_V3_ANY in m], counts=counts,
                    guards=guards, runner=r, marks={k: getattr(thl, k) for k in _PK_V3_GLOBALS})

    def _train_launch(self, n: int = 24) -> Tuple[str, Dict[str, int]]:
        """After RUN_START: the first training forward's pk-NIB write, production entry _lnmd_fwd_maskgen_nib_try ->
        _lnmd_fwd_maskgen_nib_launch -> the selector -> the picked (mocked) kernel. Returns its stdout and the
        launches per kernel."""
        thl = self.thl
        x, u, w, b = _cpu_xu(n)
        y = torch.empty((n, 3 * D), dtype=torch.uint8)
        mean, rstd = torch.empty(n), torch.empty(n)
        before = {k: len(r.calls) for k, r in self.recs.items()}
        thl.set_lnmd_layer_tag(1)
        buf = io.StringIO()
        with mock.patch.object(thl, "_lnmd_mask_nib_why_not", lambda *a, **k: None), contextlib.redirect_stdout(buf):
            got = thl._lnmd_fwd_maskgen_nib_try(x, u, y, w, b, mean, rstd, n, D, EPS, 0.1, INV, 99, D, True, True,
                                                "silu", True, True)
        self.assertEqual((got.dtype, tuple(got.shape)), (torch.uint8, (n, D // 2)))
        return buf.getvalue(), {k: len(r.calls) - before[k] for k, r in self.recs.items()}

    def test_one_line_before_run_start_none_in_the_timed_region(self) -> None:
        from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot

        v3, r1 = _PK_V3_KERNELS
        off0 = _PK_V3_OFF + "0: not 1; 0 is a gin False binding or the env kill switch"
        armed, off = _PK_V3_GLOBALS
        # (state, pre-set env, the gin binding train_ranker's Phase 1 passes to _bind_bool, line, mark, pk kernel)
        cases = (("gin ON (the default)", None, True, _PK_V3_ARMED, armed, v3),
                 ("env kill switch 0", "0", True, off0, off, r1),
                 ("gin False (held off)", None, False, off0, off, r1),
                 ("knob absent (no binding)", None, None, None, None, r1))
        for name, pre, bind, head, mark, kernel in cases:
            with self.subTest(name):
                self.st.reset()
                os.environ.pop(PK_V3_KNOB, None)
                if pre is not None:
                    os.environ[PK_V3_KNOB] = pre
                boot._bind_bool(PK_V3_KNOB, bind)
                n_pk = len(self.recs[kernel].calls)
                w = self._warmup()
                g = w["guards"]
                self.assertEqual((g.changed, g.globals_changed), ([], []))  # VERIFY: nothing left behind
                self.assertEqual((w["runner"].ok, w["runner"].failed), (8, 0))
                self.assertEqual(len(self.recs[kernel].calls) - n_pk, 8)  # every L4n item launched this pk writer
                if head is None:
                    self.assertEqual((_pk_v3_lines(w["out"]), w["prints"], w["counts"]), ([], [], [0] * 8))
                    self.assertEqual(w["marks"], {k: "" for k in _PK_V3_GLOBALS})
                else:
                    line = w["marks"][mark]  # kept across the restore
                    self.assertTrue(line.startswith(head), line)
                    # printed by the selector inside the first item (captured: dropped), then by _pk_v3_reprint right
                    # after that item, outside every capture; no later item printed it
                    self.assertEqual(w["prints"], [(line, False), (line, True)])
                    self.assertEqual(w["counts"], [1] + [0] * 7)
                    self.assertEqual(_pk_v3_lines(w["out"]), [_rank_tagged(line, w["out"])])
                    self.assertEqual([k for k, v in w["marks"].items() if v], [mark])
                if mark == armed:  # L4n's first pk launch: untagged (L4 sets -1), its first row size
                    self.assertTrue(w["marks"][mark].endswith(f"first V3 launch on layer tag -1, N={self.ROWS[0]}"))
                # RUN_START. The first training launch runs the picked kernel and prints no (111) line
                out, launched = self._train_launch()
                self.assertEqual({k: v for k, v in launched.items() if v}, {kernel: 1})
                self.assertEqual(_pk_v3_lines(out), [], out)
                # item (94)'s own first-launch line is unchanged (it names the pk writer's qualname either way)
                self.assertIn("[lnmd-mask-nib] first launch of _ln_mul_dropout_fwd_maskgen_pk_nib (MGPK + NIB store) "
                              "on layer tag 1, N=24", out)

    def test_without_the_keep_entries_the_timed_region_prints_it(self) -> None:
        """The FV-2 finding reproduced: with the two marks dropped from _KEEP_GLOBALS (v1), the scalar restore
        resets the mark and the first training launch prints the (111) line inside the timed region."""
        os.environ[PK_V3_KNOB] = "1"
        w = self._warmup(keep=False)
        self.assertEqual((w["guards"].changed, w["marks"]), ([], {k: "" for k in _PK_V3_GLOBALS}))
        out, launched = self._train_launch()
        self.assertEqual(len(_pk_v3_lines(out)), 1)
        self.assertIn(_PK_V3_ARMED, out)
        self.assertIn("first V3 launch on layer tag 1, N=24 [rank ", out)

    def test_reprint_precedes_the_startup_check_latch(self) -> None:
        """A startup byte check mismatch (review R1-F5) on L4n's first item: the (111) line prints first, then item
        (94)'s DISABLED line (the latch), both after the item and before RUN_START; L4n then stops."""
        os.environ[PK_V3_KNOB] = "1"
        self.check = lambda *a, **k: "bytes differ: y"
        w = self._warmup()
        self.assertEqual(w["runner"].ok, 1)
        out = w["out"]
        self.assertEqual(len(_pk_v3_lines(out)), 1)
        i_disabled = out.index("LN*U*dropout keep mask NIB DISABLED (DLRM_HSTU_FP8_LNMD_MASK_NIB): FP8 warmup startup "
                               "check (item L4n)")
        self.assertLess(out.index(_PK_V3_ARMED), i_disabled)
        self.assertIsNotNone(self.thl._LNMD_MASK_NIB_BROKEN)  # kept (_KEEP_GLOBALS)


@unittest.skipUnless(_gfx950(), "needs a gfx950 GPU and Triton (the kernels' inline asm)")
class PkV3GpuTest(unittest.TestCase):
    """ship17 (111) on the GPU: the V3 pk writer (knob "1") and r1 (knob "0") through the launch helper against
    MGPK's int8 kernel on the same inputs: y / mean / rstd identical, NIB == pack_ref(int8), and 8 junk guard rows
    after row N-1 untouched; every last-group size 1..8 (N 1..17, 23, 31, 69), 4097, 65536 (two junk fills, 3 p,
    2 seeds) and the production N 2,301,929 (a one-row last group; p 0.1, 2 seeds)."""

    G = 8

    def setUp(self) -> None:
        self.st = _State()
        self.st.reset()
        self.thl = self.st.thl
        self.env = mock.patch.dict(os.environ, dict(_ENV_ARMED))
        self.env.start()
        nib.reset_for_testing()

    def tearDown(self) -> None:
        self.env.stop()
        self.st.restore()
        nib.reset_for_testing()

    def _bufs(self, n: int, junk: int, nibm: bool) -> Dict[str, torch.Tensor]:
        dev = torch.device("cuda")
        g = self.G
        full = dict(
            y=torch.empty((n + g, 3 * D), dtype=torch.float8_e4m3fn, device=dev),
            mean=torch.empty((n + g,), dtype=torch.float32, device=dev),
            rstd=torch.empty((n + g,), dtype=torch.float32, device=dev),
            mask=torch.empty((n + g, D // 2) if nibm else (n + g, D), dtype=torch.uint8 if nibm else torch.int8,
                             device=dev),
        )
        full["y"].view(torch.uint8).fill_(0x5A + junk)
        full["mask"].view(torch.uint8).fill_(0x7F - junk)  # bit 3 set: never a NIB / int8 mask byte
        full["mean"].view(torch.int32).fill_(0x7FC00011 + junk)
        full["rstd"].view(torch.int32).fill_(0x7FC00021 + junk)
        return full

    def _guard_touched(self, full: Dict[str, torch.Tensor], n: int, junk: int) -> int:
        return (int((full["y"][n:].view(torch.uint8) != 0x5A + junk).sum())
                + int((full["mask"][n:].view(torch.uint8) != 0x7F - junk).sum())
                + int((full["mean"][n:].view(torch.int32) != 0x7FC00011 + junk).sum())
                + int((full["rstd"][n:].view(torch.int32) != 0x7FC00021 + junk).sum()))

    def _case(self, n: int, x, u, w, b, p: float, seed: int, junks: Tuple[int, ...]) -> None:
        thl = self.thl
        k16, kk2 = thl._lnmd_maskgen_kk2(p)
        ref = self._bufs(n, 0, False)
        thl._ln_mul_dropout_fwd_maskgen_pk[((n + 7) // 8,)](
            x, u, ref["y"][:n], w, b, ref["mean"][:n], ref["rstd"][:n], ref["mask"][:n], n, D, EPS, p, x.stride(0),
            u.stride(0), ref["y"].stride(0), ref["mask"].stride(0), INV, k16, kk2, seed, BLOCK_D=D, NUM_MASKS=3,
            MUL_U_ACTIVATION_TYPE="silu", OUTPUT_FP8=True, SILU_U=True, num_warps=1)
        torch.cuda.synchronize()
        want = nib.pack_ref(ref["mask"][:n])
        for value in ("1", "0"):  # V3, r1
            os.environ[PK_V3_KNOB] = value
            for junk in junks:
                out = self._bufs(n, junk, True)
                with contextlib.redirect_stdout(io.StringIO()):
                    thl._lnmd_fwd_maskgen_nib_launch(
                        x, u, out["y"][:n], w, b, out["mean"][:n], out["rstd"][:n], out["mask"][:n], n, D, EPS, p, INV,
                        k16, kk2, seed, D, True, True, "silu", True, True, True)
                torch.cuda.synchronize()
                tag = f"knob={value} n={n} p={p} seed={seed} junk={junk}"
                for k in ("y", "mean", "rstd"):
                    self.assertEqual(_same(out[k][:n], ref[k][:n]), 0, (tag, k))
                self.assertEqual(_same(out["mask"][:n], want), 0, tag)
                self.assertEqual(self._guard_touched(out, n, junk), 0, tag)
                del out

    def test_v3_and_r1_vs_mgpk_every_last_group_size(self) -> None:
        for n in tuple(range(1, 18)) + (23, 31, 69, 4097, 65536):
            x, u, w, b = _gpu_inputs(n)
            for p in (1e-4, 0.1, 0.5):
                for seed in SEEDS2:
                    self._case(n, x, u, w, b, p, seed, (1, 2))
            del x, u, w, b

    def test_v3_and_r1_vs_mgpk_production_n(self) -> None:
        n = 2301929  # N % 8 == 1: the last group has one row (V3 copies qe into qo there)
        x, u, w, b = _gpu_inputs(n)
        for seed in SEEDS2:
            self._case(n, x, u, w, b, 0.1, seed, (1,))
        del x, u, w, b
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
