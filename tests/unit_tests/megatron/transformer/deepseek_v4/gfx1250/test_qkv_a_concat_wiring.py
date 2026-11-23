###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""wq_a || wkv concat wiring in ``DeepseekV4Attention`` (``PRIMUS_V4_QKV_A_CONCAT=1``).

The joint GEMM itself is Primus-Turbo's ``gemm_fp8_block_concat_n`` (tested there). Here, on CPU tensors:

* ``_qkv_a_concat_ok`` admits the concat only with the switch on, bias-free linears that read the same hidden size,
  and tensor parallel size 1;
* off the block-FP8 path (no Turbo precision context) ``_qkv_a_concat`` returns ``(None, None)`` so the caller keeps
  the two separate linears, and it fixes the weight order once: ``kv_q`` when ``linear_kv``'s main_grad sits right
  before ``linear_q_down_proj``'s in the gradient buffer (one wgrad GEMM into both), else ``q_kv``.
"""

import ast
import functools
import os
import sys
import types
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn

from .helpers import requires_gfx1250

pytestmark = requires_gfx1250

_SRC = (
    Path(__file__).resolve().parents[6] / "primus/backends/megatron/core/transformer/deepseek_v4_attention.py"
)
_FUNCS = {"_qkv_a_concat_enabled", "_no_bias", "_qkv_a_fp8blk"}
_METHODS = {"_qkv_a_concat_ok", "_qkv_a_concat"}


@functools.lru_cache(maxsize=1)
def _attn_defs():
    """The concat helpers are pulled out of the attention module source with ``ast`` and executed in a small
    namespace: importing the full module drags in the whole training stack, which this CPU-side check does not need.
    """
    tree = ast.parse(_SRC.read_text())
    ns = {"os": os, "torch": torch, "nn": nn, "Optional": Optional}
    methods = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _FUNCS:
            exec(compile(ast.Module([node], []), str(_SRC), "exec"), ns)
        if isinstance(node, ast.ClassDef):
            methods += [s for s in node.body if isinstance(s, ast.FunctionDef) and s.name in _METHODS]
    assert len(methods) == len(_METHODS) and _FUNCS <= ns.keys()
    holder = ast.Module(
        body=[
            ast.ClassDef(
                name="Holder", bases=[], keywords=[], body=methods, decorator_list=[], type_params=[]
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(holder)
    exec(compile(holder, str(_SRC), "exec"), ns)
    return ns["Holder"]


def _fake(D=256, nq=128, nkv=72, bias=False, tp=1, d_kv=None):
    cls = _attn_defs()
    f = types.SimpleNamespace(
        linear_q_down_proj=nn.Linear(D, nq, bias=bias),
        linear_kv=nn.Linear(d_kv or D, nkv, bias=False),
        config=types.SimpleNamespace(tensor_model_parallel_size=tp),
    )
    f._qkv_a_concat_ok = types.MethodType(cls._qkv_a_concat_ok, f)
    f._qkv_a_concat = types.MethodType(cls._qkv_a_concat, f)
    return f


@pytest.fixture(autouse=True)
def _no_turbo_extension(monkeypatch):
    """No Turbo precision context on CPU tensors: ``_qkv_a_fp8blk`` must decline."""
    monkeypatch.setitem(sys.modules, "primus.backends.megatron.core.extensions.primus_turbo", None)


def test_qkv_a_concat_gating(monkeypatch):
    h = torch.randn(1, 4, 256)
    monkeypatch.setenv("PRIMUS_V4_QKV_A_CONCAT", "0")
    assert not _fake()._qkv_a_concat_ok(h)
    monkeypatch.setenv("PRIMUS_V4_QKV_A_CONCAT", "1")
    assert _fake()._qkv_a_concat_ok(h)
    assert not _fake(bias=True)._qkv_a_concat_ok(h)
    assert not _fake(tp=2)._qkv_a_concat_ok(h)
    assert not _fake(d_kv=128)._qkv_a_concat_ok(h)


@pytest.mark.parametrize("kv_first", [True, False])
def test_qkv_a_concat_declines_off_the_block_fp8_path(monkeypatch, kv_first):
    monkeypatch.setenv("PRIMUS_V4_QKV_A_CONCAT", "1")
    f = _fake()
    wq, wkv = f.linear_q_down_proj.weight, f.linear_kv.weight
    buf = torch.zeros(wq.numel() + wkv.numel())
    first, second = (wkv, wq) if kv_first else (wq, wkv)
    first.main_grad = buf[: first.numel()].view(first.shape)
    second.main_grad = buf[first.numel() :].view(second.shape)
    h = torch.randn(2, 8, 256)
    assert f._qkv_a_concat_ok(h)
    assert f._qkv_a_concat(h) == (None, None)
    assert f._qkv_a_fp8_order == ("kv_q" if kv_first else "q_kv")
