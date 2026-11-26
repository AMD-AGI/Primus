###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compressor pool + kv_norm + compress-base RoPE as one op (``comp_fused``).

* ``comp_pool_norm_rope`` against the unfused chain it replaces (``fused_window_pool`` with the d(ape) fan-in ->
  ``LocalRMSNorm`` on the glue path -> ``apply_interleaved_partial_rope``): output, d(proj), d(ape) and d(kv_norm
  weight) byte-equal, with plain gradients, ``main_grad`` first-write and accumulate, and a row-strided incoming
  gradient.
* ``comp_fused_ok`` declines the configurations it does not reproduce (HCA windows, the order-changing glue
  switches off).
* The wiring: ``DeepseekV4Attention._build_compressed_pool`` hands the RoPE tables to ``Compressor.forward``; the
  CSA pool comes back rotated from the fused op, the HCA pool from the unfused ops, both byte-equal to the path
  without the tables.
"""

import ast
import functools
import os
import types
from pathlib import Path

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
HD, RD = 512, 64

_ATTN_SRC = (
    Path(__file__).resolve().parents[6] / "primus/backends/megatron/core/transformer/deepseek_v4_attention.py"
)
_ATTN_METHODS = {"_build_compressed_pool", "_compress_rope_cos_sin", "_compress_rope_tables"}


@pytest.fixture(autouse=True)
def _glue_env(monkeypatch):
    for n in ("PRIMUS_GLUE_COMPRESS_POOL", "PRIMUS_FUSE2_WPOOL_DAPE", "PRIMUS_GLUE_RMSNORM"):
        monkeypatch.setenv(n, "1")
    for n in ("PRIMUS_ROPE_TRITON", "PRIMUS_RMSNORM_TRITON", "PRIMUS_FUSE2_COMP_MG"):
        monkeypatch.delenv(n, raising=False)


def _tables(B, P, ratio):
    inv = 1.0 / (160000.0 ** (torch.arange(0, RD, 2, device=DEV).float() / RD))
    ang = (torch.arange(P, device=DEV).float() * ratio)[:, None] * inv[None, :]
    cos, sin = torch.cos(ang).to(torch.bfloat16), torch.sin(ang).to(torch.bfloat16)
    return cos.repeat(B, 1).contiguous(), sin.repeat(B, 1).contiguous()  # [B * P, RD / 2]


def _param(t, mg0):
    p = torch.nn.Parameter(t.clone())
    if mg0 is not None:
        p.main_grad = mg0.clone()
        p.grad_added_to_main_grad = False
    return p


@pytest.mark.parametrize(
    "B,S,seed,mg,first,strided",
    [
        (1, 4096, 0, False, True, False),
        (1, 4096, 1, False, True, False),
        (2, 2048, 2, False, True, False),
        (1, 4096, 3, True, True, False),
        (1, 4096, 4, True, False, True),
    ],
)
def test_comp_fused_matches_unfused_chain(B, S, seed, mg, first, strided):
    from primus_turbo.pytorch.ops import utils as tu

    from primus.backends.megatron.core.transformer.dual_rope import (
        apply_interleaved_partial_rope,
    )
    from primus.backends.megatron.core.transformer.local_rmsnorm import LocalRMSNorm
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        comp_fused as CF,
    )
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.compressor_pool import (
        fused_window_pool,
    )

    ratio, ov = 4, True
    C, R, P = 2 * 2 * HD, 2 * ratio, S // ratio
    torch.manual_seed(seed)
    proj0 = (torch.randn(B, S, C, device=DEV) * 1.5).to(torch.bfloat16)
    ape0 = torch.randn(R, HD, device=DEV) * 0.3
    wn0 = (1 + 0.1 * torch.randn(HD, device=DEV)).to(torch.bfloat16)
    dout0 = (torch.randn(B, P, HD, device=DEV) * 0.05).to(torch.bfloat16)
    ape_mg0 = torch.randn(R, HD, device=DEV) * 0.01 if mg else None
    wn_mg0 = torch.randn(HD, device=DEV) * 0.01 if mg else None
    cos, sin = _tables(B, P, ratio)

    tu.set_main_grad_first_write_policy((lambda p: first) if mg else None)
    res = {}
    try:
        for fused in (False, True):
            proj = proj0.clone().requires_grad_(True)
            ape = _param(ape0, ape_mg0)
            norm = LocalRMSNorm(HD, eps=1e-6).to(DEV).to(torch.bfloat16)
            with torch.no_grad():
                norm.weight.copy_(wn0)
            if mg:
                norm.weight.main_grad = wn_mg0.clone()
                norm.weight.grad_added_to_main_grad = False
            if fused:
                assert CF.comp_fused_ok(proj, ape, norm.weight, ratio, HD, ov, RD)
                out = CF.comp_pool_norm_rope(
                    proj,
                    ape,
                    norm.weight,
                    cos,
                    sin,
                    ratio=ratio,
                    hd=HD,
                    overlap=ov,
                    eps=norm.eps,
                    rotary_dim=RD,
                )
            else:
                y = norm(fused_window_pool(proj, ape, ratio, HD, ov))
                out = apply_interleaved_partial_rope(
                    y.unsqueeze(2), cos.view(B, P, -1), sin.view(B, P, -1), rotary_dim=RD
                ).squeeze(2)
            if strided:
                big = torch.zeros(B, P, HD + 64, device=DEV, dtype=torch.bfloat16)
                big[..., :HD] = dout0
                d = big[..., :HD]
            else:
                d = dout0
            out.backward(d)
            torch.cuda.synchronize()
            res[fused] = {
                "out": out.detach(),
                "dproj": proj.grad,
                "dape": ape.main_grad if mg else ape.grad,
                "dnorm_w": norm.weight.main_grad if mg else norm.weight.grad,
            }
    finally:
        tu.set_main_grad_first_write_policy(None)
    bad = [n for n in res[False] if not bits_equal(res[False][n].contiguous(), res[True][n].contiguous())]
    assert not bad, f"fused compressor tail differs from the unfused chain in: {bad}"


def test_comp_fused_declines_other_configurations(monkeypatch):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        comp_fused as CF,
    )

    wn = torch.ones(HD, device=DEV, dtype=torch.bfloat16)
    csa = (torch.zeros(1, 1024, 4 * HD, device=DEV, dtype=torch.bfloat16), torch.zeros(8, HD, device=DEV))
    hca = (torch.zeros(1, 1024, 2 * HD, device=DEV, dtype=torch.bfloat16), torch.zeros(128, HD, device=DEV))
    for t in (csa[1], hca[1]):
        t.requires_grad_(True)
    assert CF.comp_fused_ok(csa[0], csa[1], wn, 4, HD, True, RD)
    assert not CF.comp_fused_ok(hca[0], hca[1], wn, 128, HD, False, RD), "HCA windows keep the unfused ops"
    assert not CF.comp_fused_ok(csa[0], csa[1], wn.float(), 4, HD, True, RD), "fp32 norm weight"
    for name in ("PRIMUS_GLUE_RMSNORM", "PRIMUS_FUSE2_WPOOL_DAPE", "PRIMUS_GLUE_COMPRESS_POOL"):
        monkeypatch.setenv(name, "0")
        assert not CF.comp_fused_ok(csa[0], csa[1], wn, 4, HD, True, RD), f"{name}=0"
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("PRIMUS_ROPE_TRITON", "0")
    assert not CF.comp_fused_ok(csa[0], csa[1], wn, 4, HD, True, RD)


@functools.lru_cache(maxsize=1)
def _attn_holder():
    """The compressed-pool methods of ``DeepseekV4Attention``, pulled out of the module source with ``ast`` and
    executed in a small namespace (importing the attention module drags in the whole training stack)."""
    from primus.backends.megatron.core.transformer.dual_rope import (
        apply_interleaved_partial_rope,
    )

    tree = ast.parse(_ATTN_SRC.read_text())
    methods = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4Attention":
            methods = [s for s in node.body if isinstance(s, ast.FunctionDef) and s.name in _ATTN_METHODS]
    assert {m.name for m in methods} == _ATTN_METHODS
    holder = ast.Module(
        body=[
            ast.ClassDef(
                name="Holder", bases=[], keywords=[], body=methods, decorator_list=[], type_params=[]
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(holder)
    ns = {
        "torch": torch,
        "os": os,
        "apply_interleaved_partial_rope": apply_interleaved_partial_rope,
        "_v4_get_cp_group": lambda: None,
    }
    exec(compile(holder, str(_ATTN_SRC), "exec"), ns)
    return ns["Holder"]


def _fake_attention(ratio, seed):
    from primus.backends.megatron.core.transformer.compressor import Compressor
    from primus.backends.megatron.core.transformer.dual_rope import DualRoPE

    torch.manual_seed(seed)
    comp = Compressor(hidden_size=1024, head_dim=HD, ratio=ratio).to(DEV)
    comp.wkv_gate.to(torch.bfloat16)
    comp.kv_norm.to(torch.bfloat16)
    with torch.no_grad():
        comp.kv_norm.weight.copy_(1 + 0.1 * torch.randn(HD, device=DEV))
        comp.ape.normal_(std=0.3)
    f = types.SimpleNamespace(
        compressor=comp,
        rope=DualRoPE(rotary_dim=RD, rope_theta=10000.0, compress_rope_theta=160000.0).to(DEV),
        rotary_dim=RD,
        compress_ratio=ratio,
    )
    cls = _attn_holder()
    for name in _ATTN_METHODS:
        setattr(f, name, types.MethodType(getattr(cls, name), f))
    return f


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("rope_tbl", ["1", "0"])
def test_build_compressed_pool_wiring(monkeypatch, ratio, rope_tbl):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        comp_fused as CF,
    )

    monkeypatch.setenv("PRIMUS_MB_ROPE_TBL", rope_tbl)
    calls = []
    orig = CF.comp_pool_norm_rope
    monkeypatch.setattr(CF, "comp_pool_norm_rope", lambda *a, **kw: calls.append(1) or orig(*a, **kw))
    S = 4096
    h0 = torch.randn(1, S, 1024, device=DEV).to(torch.bfloat16)
    g0 = (torch.randn(1, S // ratio, HD, device=DEV) * 0.05).to(torch.bfloat16)
    res = []
    for fused in (False, True):
        f = _fake_attention(ratio, seed=7)
        if not fused:
            f._compress_rope_tables = lambda hidden: None  # the path before the fused op
        h = h0.clone().requires_grad_(True)
        out = f._build_compressed_pool(h)
        out.backward(g0)
        torch.cuda.synchronize()
        c = f.compressor
        res.append(
            {
                "out": out.detach(),
                "dhidden": h.grad,
                "dwkv_gate": c.wkv_gate.weight.grad,
                "dape": c.ape.grad,
                "dkv_norm": c.kv_norm.weight.grad,
            }
        )
    assert len(calls) == (1 if ratio == 4 else 0), "fused op used where it should (not) apply"
    bad = [n for n in res[0] if not bits_equal(res[0][n].contiguous(), res[1][n].contiguous())]
    assert not bad, f"compressed pool differs from the unfused path in: {bad}"


def test_compressor_rope_rejects_packed_input():
    from primus.backends.megatron.core.transformer.compressor import Compressor

    comp = Compressor(hidden_size=256, head_dim=128, ratio=4)
    with pytest.raises(ValueError):
        comp(torch.zeros(1, 16, 256), cu_seqlens=torch.tensor([0, 16]), rope=(None, None, 64))
