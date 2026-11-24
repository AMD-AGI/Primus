###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``qkv_post`` on the outputs of the Primus-Turbo concat-N op (wq_a || wkv) with the concat's dY pre-quant slot.

The forward of ``gemm_fp8_block_concat_n`` leaves a ``DyPrequantSlot``; ``qkv_post`` takes it, and its backward
quantizes the concat's dY together with both dgamma reductions in one launch (``dy_prequant_fill``), which the
concat backward then reuses. Checked against the same step without the slot (``_dg_reduce2_kernel`` + the concat's
own quant) and with a Turbo that has no slot API: outputs, dX, both weight main_grads and both dgammas byte-equal,
for both weight orders and with the norm dgammas in ``.grad`` or ``main_grad``.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gluon

pytestmark = requires_gluon

DEV = "cuda"
T, K, HQ, HKV, RD = 1024, 1024, 1024, 512, 64


def _cfg():
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScalingGranularity,
    )

    return Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.BLOCKWISE, block_size=128)


@pytest.fixture
def concat_env(monkeypatch):
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "1")
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_KERNEL", "dsv4up")
    # the slot carries the per-weight FP8 W^T copies, which exist only with a ready FP8 weight cache (written by the
    # optimizer in training, by _fill_weight_cache here); VERIFY re-quantizes every cache hit
    monkeypatch.setenv("PRIMUS_TURBO_FP8_WCACHE", "1")
    monkeypatch.setenv("PRIMUS_TURBO_FP8_WCACHE_VERIFY", "1")


def _data(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def r(*sh):
        return torch.randn(*sh, device=DEV, generator=g)

    return dict(
        x=(r(T, K) * 0.5).bfloat16(),
        wq=(r(HQ, K) * 0.02).bfloat16(),
        wkv=(r(HKV, K) * 0.02).bfloat16(),
        mgq=r(HQ, K) * 0.01,
        mgkv=r(HKV, K) * 0.01,
        gq=(1 + 0.1 * r(HQ)).bfloat16(),
        gkv=(1 + 0.1 * r(HKV)).bfloat16(),
        dyq=(r(T, HQ) * 0.01).bfloat16(),
        dykv=(r(T, HKV) * 0.02).bfloat16(),
        pos=torch.arange(T, device=DEV),
        inv=1.0 / (160000.0 ** (torch.arange(0, RD, 2, device=DEV).float() / RD)),
    )


def _fill_weight_cache(w):
    """Write the FP8 weight cache of ``w`` as the optimizer does (W, W^T, scales) and mark it ready."""
    from primus_turbo.pytorch.core import fp8blk_weight_cache as wcache
    from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
        quant_fp8_blockwise_for_weight_impl,
    )

    f8 = torch.float8_e4m3fn
    q, sc = quant_fp8_blockwise_for_weight_impl(w.detach().contiguous(), f8, block_size=128)
    e = ((sc.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    c = wcache.attach(w, f8)
    c.w.copy_(q)
    c.scale_inv.copy_(sc)
    c.e8m0.copy_(e)
    c.wt.copy_(q.t())
    c.e8m0_t.copy_(e.t())
    c.ready = True


def _step(d, order, glue):
    """concat-N forward -> qkv_post -> backward of both; returns the bytes to compare."""
    from primus_turbo.pytorch.ops.gemm_fp8_concat import (
        gemm_fp8_block_concat_n,
        gemm_fp8_block_concat_n_supported,
    )

    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        qkv_post as QP,
    )

    wq, wkv = torch.nn.Parameter(d["wq"].clone()), torch.nn.Parameter(d["wkv"].clone())
    for w in (wq, wkv):
        _fill_weight_cache(w)
    # one main_grad buffer in weight order, as DDP lays them out (the concat's wgrad is one GEMM into both)
    pairs = ((wkv, d["mgkv"]), (wq, d["mgq"])) if order == "kv_q" else ((wq, d["mgq"]), (wkv, d["mgkv"]))
    buf = torch.cat([m for _, m in pairs], 0)
    off = 0
    for w, m in pairs:
        w.main_grad = buf[off : off + m.shape[0]]
        w.grad_added_to_main_grad = False
        off += m.shape[0]
    gq, gkv = torch.nn.Parameter(d["gq"].clone()), torch.nn.Parameter(d["gkv"].clone())
    if glue:
        for g in (gq, gkv):
            g.main_grad = torch.zeros(g.shape, device=DEV, dtype=torch.bfloat16)
            g.grad_added_to_main_grad = False
    ws = [wkv, wq] if order == "kv_q" else [wq, wkv]
    x = d["x"].clone().requires_grad_(True)
    if not gemm_fp8_block_concat_n_supported(x, ws, _cfg()):
        pytest.skip("concat-N op not enabled for this configuration")
    outs = gemm_fp8_block_concat_n(x, ws, _cfg(), ["megatron", "megatron"])
    q_a, kv_a = (outs[1], outs[0]) if order == "kv_q" else (outs[0], outs[1])
    yq, ykv = QP.qkv_post(
        q_a,
        kv_a,
        gq,
        gkv,
        d["pos"],
        d["inv"],
        eps_q=1e-6,
        eps_kv=1e-6,
        rotary_dim=RD,
        q_first=order == "q_kv",
    )
    torch.autograd.backward([yq, ykv], [d["dyq"], d["dykv"]])
    torch.cuda.synchronize()
    return {
        "yq": yq.detach(),
        "ykv": ykv.detach(),
        "dx": x.grad,
        "main_grad_q": wq.main_grad,
        "main_grad_kv": wkv.main_grad,
        "dgamma_q": gq.main_grad if glue else gq.grad,
        "dgamma_kv": gkv.main_grad if glue else gkv.grad,
    }


@pytest.mark.parametrize("mode", ["slot", "no_slot_api"])
@pytest.mark.parametrize("order,glue,seed", [("q_kv", False, 1), ("kv_q", True, 2), ("kv_q", False, 3)])
def test_qkv_post_prequant_matches_plain_path(monkeypatch, concat_env, mode, order, glue, seed):
    from primus_turbo.pytorch.ops import gemm_fp8_concat as gc
    from primus_turbo.pytorch.ops import utils as tu

    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        qkv_post as QP,
    )

    monkeypatch.setenv("PRIMUS_GLUE_RMSNORM", "1" if glue else "0")
    d = _data(seed)
    fills, cats = [], []
    orig_fill, orig_cat = gc.dy_prequant_fill, gc._quant_cat_cols
    monkeypatch.setattr(gc, "dy_prequant_fill", lambda *a: (lambda r: fills.append(r) or r)(orig_fill(*a)))
    monkeypatch.setattr(gc, "_quant_cat_cols", lambda *a: cats.append(1) or orig_cat(*a))
    tu.set_main_grad_first_write_policy((lambda p: True) if glue else None)
    try:
        with monkeypatch.context() as m:  # reference: no slot taken -> _dg_reduce2_kernel + the concat's own quant
            m.setattr(QP, "_take_dy_prequant", lambda a, b: (None, False))
            ref = _step(d, order, glue)
        assert fills == [] and len(cats) == 1
        fills.clear()
        cats.clear()
        if mode == "no_slot_api":  # a Turbo without take_dy_prequant / dy_prequant_fill
            monkeypatch.delattr(gc, "take_dy_prequant")
            assert QP._take_dy_prequant(ref["yq"], ref["ykv"]) == (None, False)
        got = _step(d, order, glue)
    finally:
        tu.set_main_grad_first_write_policy(None)
    if mode == "slot":
        assert fills == [True], "qkv_post did not quantize the concat dY in its backward"
        assert cats == [], "the concat backward quantized dY again"
    else:
        assert fills == [] and len(cats) == 1
    bad = [n for n in ref if not bits_equal(ref[n].contiguous(), got[n].contiguous())]
    assert not bad, f"{mode}: differs from the plain path in: {bad}"
