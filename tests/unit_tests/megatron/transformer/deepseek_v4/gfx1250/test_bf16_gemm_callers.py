###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""BF16 GEMM routing in ``fusions/bf16tune.py`` and ``fusions/bf16grp.py`` (Primus-side callers of the Gluon GEMMs).

At the training shapes of the DSv4-Flash proxy (T = H = 4096) the routed Gluon GEMMs must reproduce the bytes of the
hipBLASLt kernels they replace (fixed split-K slice order / StaggerU K start), and anything not planned must be left to
the caller (``None`` / ``False``):

* ``bf16tune``: HCA forward, HCA wgrad, q192 forward (split-K); CSA forward / dgrad / wgrad, HCA dgrad and the
  grouped o_a wgrad (StaggerU K order);
* ``bf16grp``: indexer compressor + q-weight projections in one grouped launch.

Both are on with the Gluon kernels on gfx1250 and off with ``PRIMUS_DSV4_FAST=0``.

The reference kernels are hipBLASLt's, selected through the repository's TunableOp file for these shapes (read
without tuning); the byte equality is tied to the hipBLASLt of the image, so rerun after an image change.
"""

import importlib
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from .helpers import bits_equal, requires_gluon

pytestmark = requires_gluon

DEV, BF = "cuda", torch.bfloat16
T, H = 4096, 4096
_TUNABLEOP = os.path.join(
    os.path.dirname(__file__), *[".."] * 6, "examples", "megatron", "models", "deepseek-v4", "mi455x_opt3",
    "tunableop_dsv4_gfx1250_bf16_.csv",
)  # fmt: skip


@pytest.fixture(scope="module", autouse=True)
def _tunableop():
    path = os.path.abspath(_TUNABLEOP)
    if not os.path.exists(path.replace("_.csv", "_0.csv")):
        pytest.skip("TunableOp file for the proxy shapes not found")
    tun = torch.cuda.tunable
    old = (tun.is_enabled(), tun.tuning_is_enabled(), tun.get_filename())
    tun.enable(True)
    tun.tuning_enable(False)
    tun.set_filename(path, insert_device_ordinal=True)
    tun.read_file()
    yield
    tun.enable(old[0])
    tun.tuning_enable(old[1])
    tun.set_filename(old[2], insert_device_ordinal=False)


def _reload(monkeypatch, mod, **env):
    monkeypatch.delenv("PRIMUS_DSV4_FAST", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    m = importlib.import_module(f"primus.backends.megatron.core.fusions.{mod}")
    m = importlib.reload(m)
    m._STATE.clear()
    return m


@pytest.fixture
def restore_modules():
    yield
    for mod in ("bf16tune", "bf16grp"):
        importlib.reload(importlib.import_module(f"primus.backends.megatron.core.fusions.{mod}"))


def rn(*s, scale=1.0):
    return (torch.randn(*s, device=DEV) * scale).to(BF)


def _shape_cases():
    """name -> (bt_mm operands, torch reference callable), training layouts."""
    torch.manual_seed(0)
    x = rn(1, T, H)
    xv = x.view(T, H)
    cases = {}
    for tag, n in (("csa", 2048), ("hca", 1024)):
        w, g = rn(n, H, scale=0.02), rn(T, n, scale=1e-3)
        cases[f"{tag}_fwd"] = ((xv, w.t()), lambda x=x, w=w: F.linear(x, w).view(T, -1))
        cases[f"{tag}_dgrad"] = ((g, w), lambda g=g, w=w: g.mm(w))
        cases[f"{tag}_wgrad"] = ((g.t(), xv), lambda g=g: torch.mm(g.t(), xv))
    w192 = rn(192, H, scale=0.02)
    cases["q192_fwd"] = ((xv, w192.t()), lambda: F.linear(x, w192).view(T, -1))
    G, R = 8, 1024
    x_oa = rn(1, T, G, H).reshape(T, G, H)
    do = rn(1, T, G * R, scale=1e-3).reshape(T, G, R).transpose(0, 1)
    cases["oa_wgrad"] = ((do.transpose(1, 2), x_oa.transpose(0, 1)),
                         lambda: torch.bmm(do.transpose(1, 2), x_oa.transpose(0, 1)))  # fmt: skip
    return cases


@pytest.mark.parametrize(
    "names",
    [["hca_fwd", "hca_wgrad", "q192_fwd"], ["csa_fwd", "csa_dgrad", "csa_wgrad", "hca_dgrad", "oa_wgrad"]],
    ids=["split_k", "stagger_u"],
)
def test_planned_gemms_match_hipblaslt_bytes(monkeypatch, restore_modules, names):
    bt = _reload(monkeypatch, "bf16tune")
    cases = _shape_cases()
    for name in names:
        (a, b), ref = cases[name]
        got = bt.bt_mm(a, b)
        assert got is not None, f"{name} not routed"
        torch.cuda.synchronize()
        assert bits_equal(got.reshape(-1, got.shape[-1]), ref().reshape(-1, got.shape[-1])), name


def test_bf16tune_leaves_unplanned_cases_to_the_caller(monkeypatch, restore_modules):
    bt = _reload(monkeypatch, "bf16tune")
    a, b = rn(T, 1024), rn(1024, 768)  # shape not in the plan
    assert bt.bt_mm(a, b) is None
    x, w = rn(T, H), rn(1024, H)
    assert bt.bt_mm(x, w.t().float().to(BF).requires_grad_(True)) is None  # autograd is never bypassed
    assert bt.bt_mm(x.float(), w.t()) is None  # not BF16
    off = _reload(monkeypatch, "bf16tune", PRIMUS_DSV4_FAST="0")
    assert off.bt_mm(x, w.t()) is None and not off.enabled()


def _fake_indexer(n_comp=512, n_q=192, requires_grad=False):
    torch.manual_seed(1)
    wc = torch.nn.Parameter(rn(n_comp, H, scale=0.02), requires_grad=requires_grad)
    wq = torch.nn.Parameter(rn(n_q, H, scale=0.02), requires_grad=requires_grad)
    comp = SimpleNamespace(_fuse_proj=True, wkv_gate=SimpleNamespace(weight=wc, bias=None))
    return SimpleNamespace(
        indexer_compressor=comp, _fuse_qw_proj=True, w_dq_w=SimpleNamespace(weight=wq, bias=None)
    )


def test_indexer_pair_equals_two_linears(monkeypatch, restore_modules):
    g = _reload(monkeypatch, "bf16grp")
    idx = _fake_indexer()
    hidden = rn(1, T, H)
    with torch.no_grad():
        res = g.indexer_pair(idx, hidden)
        assert res is not None
        yc, yq = res
        rc = F.linear(hidden, idx.indexer_compressor.wkv_gate.weight)
        rq = F.linear(hidden, idx.w_dq_w.weight)
    torch.cuda.synchronize()
    assert bits_equal(yc, rc) and bits_equal(yq, rq)
    # not planned: wrong sequence length, trainable weights, hidden requiring grad
    assert g.indexer_pair(idx, rn(1, 2048, H)) is None
    assert g.indexer_pair(_fake_indexer(requires_grad=True), hidden) is None
    assert g.indexer_pair(idx, hidden.clone().requires_grad_(True)) is None
    assert _reload(monkeypatch, "bf16grp", PRIMUS_DSV4_FAST="0").indexer_pair(idx, hidden) is None
