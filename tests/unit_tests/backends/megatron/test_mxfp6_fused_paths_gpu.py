###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gate on vs off, at the Flux shapes these kernels exist for (micro-batch 32, sequence 512 = 256 text + 256 image,
hidden 3072, 24 heads of 128, MLP 12288), eager and under torch.compile:

* ``mxfp6_single_linear2_cat``: the single block's gated MLP + out-projection with the residual in one GEMM. Output
  close, every gradient bitwise except the gate's (it reads h, now summed in fp32 before one rounding), the cat kernel
  in the trace only with the gate on, the same number of graph breaks.
* ``mxfp6_attn_q_norm_rope``: q's RMSNorm and RoPE applied inside the attention forward, single and joint blocks. The
  attention output and every gradient bitwise, the q-norm kernel only with the gate on, the Triton norm pass for k
  only (half the launches), the same number of graph breaks.
* ``mxfp6_fwd_fp4_joint_txt_proj``: the joint out-projection pair with the text stream's forward in MXFP4. The image
  stream's output and weight gradient bitwise, the text stream at the MXFP4 error level, one more A4W4 launch.
* ``mxfp6_fwd_fp4_heal_*``: healing a joint MLP / text out-projection (``_mxfp6_fwd_fp4_healed``) mid-process under one
  compiled function gives bitwise what a fresh run with the MXFP4 forward off gives, through exactly one new graph, and
  flipping back is a cache hit that reproduces the unhealed result.

Stochastic rounding is off: its per-call seeds would round two runs in one process differently.
"""

import types

import pytest
import torch

S_TXT, S, B, H, D = 256, 512, 32, 24, 128
K, F = H * D, 4 * H * D


def _gfx950():
    try:
        return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gfx950(), reason="the MXFP6 tilescale kernels need gfx950")


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def _profiled(f, t, *extra):
    """Warm f (compile, kernel loads), then run it once under the profiler: output, input grads, kernel names, breaks."""
    torch._dynamo.utils.counters.clear()
    f(t, *extra)[0].backward()
    for x in t.values():
        x.grad = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        loss, out = f(t, *extra)
        loss.backward()
        torch.cuda.synchronize()
    breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
    return out.detach(), {n: x.grad for n, x in t.items() if x.grad is not None}, [e.name for e in prof.events()], breaks


# -- single-block linear2 with the gated residual in one GEMM --------------------------------------------------------


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compile"])
def test_single_linear2_cat(compiled, monkeypatch):
    import primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local as L
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import (
        Mxfp6Gates,
        reset,
    )

    if not L._CAT_TS:
        pytest.skip("no two-segment tilescale GEMM in this aiter / Primus-Turbo")
    dev, bf16 = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(0)
    rn = lambda *s, scale=1.0: torch.randn(*s, device=dev, dtype=bf16, generator=g) * scale  # noqa: E731
    base = dict(x=rn(S, B, K), o=rn(S, B, K), res=rn(S, B, K), w1=rn(F, K, scale=0.02), b1=rn(F, scale=0.02),
                w2=rn(K, F, scale=0.02), wp=rn(K, K, scale=0.02), b2=rn(K, scale=0.02),
                mod=rn(B, 3 * K))  # the AdaLN output; the gate is its third chunk (a strided view)
    g_out = rn(S, B, K)
    # the stand-in modules are not MXFP6RowParallelLinear
    monkeypatch.setattr(L, "_mlp_proj_eligible", lambda mlp, proj: True)

    def single(t):
        lin = lambda w, b: types.SimpleNamespace(weight=w, bias=b, _fuse_wgrad_accum=False)  # noqa: E731
        mlp = types.SimpleNamespace(linear_fc1=lin(t["w1"], t["b1"]), linear_fc2=lin(t["w2"], t["b2"]))
        proj = types.SimpleNamespace(weight=t["wp"], _weight_is_fp4=False)
        out = L.mlp_proj_gated(mlp, proj, t["x"], t["o"], t["mod"].chunk(3, dim=-1)[2], None, t["res"])
        return (out.float() * g_out.float()).sum(), out

    def run(on):
        reset(Mxfp6Gates(gemm_layout="tilescale", bwd_fp4_dgrad=True, bwd_fp4_wgrad=True, shared_grad_pack=True,
                         gate_mul_pack=True, gate_mul_pack_bias=True, single_linear2_cat=on))
        torch._dynamo.reset()
        t = {n: x.clone().requires_grad_(True) for n, x in base.items()}
        return _profiled(torch.compile(single) if compiled else single, t)

    try:
        o0, g0, n0, br0 = run(False)
        o1, g1, n1, br1 = run(True)
    finally:
        reset(Mxfp6Gates())
    assert _rel(o1, o0) < 1e-2
    assert sum("a6w6_cat" in n for n in n0) == 0 and sum("a6w6_cat" in n for n in n1) > 0
    assert br0 == br1
    for n in g0:
        if n == "mod":
            assert _rel(g1[n], g0[n]) < 1e-2
        else:
            assert torch.equal(g0[n], g1[n]), n


# -- q's norm and RoPE inside the attention forward -------------------------------------------------------------------


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compile"])
@pytest.mark.parametrize("block", ["single", "joint"])
def test_attn_q_norm_rope(block, compiled, monkeypatch):
    import primus_turbo.pytorch as pt

    import primus.backends.megatron.core.models.diffusion.flux.attention as A
    from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
        MXFP6JointQKVFunction,
        MXFP6QKVNormRopeFunction,
    )
    from primus.backends.megatron.core.models.diffusion.common.fused_norm_rope import (
        apply_autotune_pin,
    )
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import (
        Mxfp6Gates,
        reset,
    )

    if not hasattr(torch.ops.primus_turbo, "attention_aiter_qnorm_forward_impl"):
        pytest.skip("Primus-Turbo without the q-norm attention forward")
    monkeypatch.setenv("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32", "0")  # the a16 backward is deterministic: bitwise grads
    dev, bf16, eps = "cuda", torch.bfloat16, 1e-6
    g = torch.Generator(device=dev).manual_seed(1)
    rn = lambda *s, scale=1.0: torch.randn(*s, device=dev, dtype=bf16, generator=g) * scale  # noqa: E731

    def angles(s):  # Flux: both entries of a pair share one angle
        a = torch.randn(s, B, 1, D // 2, device=dev, generator=g) * 3
        return torch.stack([a, a], dim=-1).reshape(s, B, 1, D)

    base = dict(x=rn(S, B, K), x_txt=rn(S_TXT, B, K), w=rn(3 * K, K, scale=0.02), b=rn(3 * K, scale=0.02),
                w_txt=rn(3 * K, K, scale=0.02), b_txt=rn(3 * K, scale=0.02), wq=1 + rn(D, scale=0.1),
                wk=1 + rn(D, scale=0.1), wq_txt=1 + rn(D, scale=0.1), wk_txt=1 + rn(D, scale=0.1))
    freqs, freqs_img, freqs_txt = angles(S), angles(S - S_TXT), angles(S_TXT)
    g_out = rn(S, B, H, D)

    def attend(q, k, v, q_norm):
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))  # bshd views of sbhd storage
        return pt.ops.flash_attn_func(q, k, v, **({} if q_norm is None else {"q_norm": q_norm})).transpose(0, 1)

    def single(t, on):
        cos, sin = A._rope_cos_sin(freqs, bf16)
        out = MXFP6QKVNormRopeFunction.apply(t["x"], t["w"], t["b"], t["wq"], t["wk"], cos, sin, eps, True, False,
                                             True, False, on)
        tab = A._q_norm_tab(cos, sin) if on else None
        o = attend(out[0], out[1], out[2], (t["wq"], tab, t["wq"], tab, S // 256, eps, out[4]) if on else None)
        return (o.float() * g_out.float()).sum(), o

    def joint(t, on):
        ca, sa = A._rope_cos_sin(freqs_txt, bf16)
        cb, sb = A._rope_cos_sin(freqs_img, bf16)
        out = MXFP6JointQKVFunction.apply(t["x_txt"], t["x"][S_TXT:], t["w_txt"], t["w"], t["b_txt"], t["b"],
                                          t["wq_txt"], t["wk_txt"], t["wq"], t["wk"], ca, sa, cb, sb, eps, True,
                                          False, True, False, on)
        q_norm = None
        if on:
            q_norm = (t["wq_txt"], A._q_norm_tab(ca, sa), t["wq"], A._q_norm_tab(cb, sb), S_TXT // 256, eps, out[4])
        o = attend(out[0], out[1], out[2], q_norm)
        return (o.float() * g_out.float()).sum(), o

    fn = single if block == "single" else joint

    def run(on):
        reset(Mxfp6Gates(gemm_layout="tilescale", bwd_fp4_dgrad=True, bwd_fp4_wgrad=True, strided_v=True,
                         joint_qkv=True, fused_qkv="on", norm_rope_pin=True, attn_q_norm_rope=on))
        assert apply_autotune_pin(force=True)  # after the gates: the pin is a no-op unless its gate is set
        torch._dynamo.reset()
        t = {n: x.clone().requires_grad_(True) for n, x in base.items()}
        return _profiled(torch.compile(fn) if compiled else fn, t, on)

    try:
        o0, g0, n0, br0 = run(False)
        o1, g1, n1, br1 = run(True)
    finally:
        reset(Mxfp6Gates())
    assert torch.equal(o0, o1)
    assert sum("qnorm" in n.lower() for n in n0) == 0 and sum("qnorm" in n.lower() for n in n1) > 0
    assert sum(n.startswith("_fwd_kernel") for n in n1) * 2 == sum(n.startswith("_fwd_kernel") for n in n0)
    assert br0 == br1
    for n in g0:
        assert n in g1 and torch.equal(g0[n], g1[n]), n


# -- the joint out-projection pair with the text stream's forward in MXFP4 --------------------------------------------


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compile"])
def test_fwd_fp4_joint_txt_proj(compiled):
    from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
        MXFP6JointProjFunction,
    )
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import (
        Mxfp6Gates,
        reset,
    )

    dev, bf16 = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(2)
    rn = lambda *s, scale=1.0: torch.randn(*s, device=dev, dtype=bf16, generator=g) * scale  # noqa: E731
    base = dict(o=rn(S, B, K), w_img=rn(K, K, scale=0.02), w_txt=rn(K, K, scale=0.02))
    g_img, g_txt = rn(S - S_TXT, B, K), rn(S_TXT, B, K)

    def run(on):
        reset(Mxfp6Gates(gemm_layout="tilescale", bwd_fp4_dgrad=True, bwd_fp4_wgrad=True, joint_proj=True,
                         fwd_fp4_joint_txt_proj=on))

        def fn(t):  # every argument passed: the text forward's format is the last one (joint_proj_pair passes the gate)
            out = MXFP6JointProjFunction.apply(t["o"], S_TXT, t["w_img"], t["w_txt"], False, True, False, None, on)
            return (out[0].float() * g_img.float()).sum() + (out[1].float() * g_txt.float()).sum(), out[:2]

        torch._dynamo.reset()
        t = {n: x.clone().requires_grad_(True) for n, x in base.items()}
        f = torch.compile(fn) if compiled else fn
        f(t)[0].backward()
        for x in t.values():
            x.grad = None
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            loss, (img, txt) = f(t)
            loss.backward()
            torch.cuda.synchronize()
        img, txt = img.detach(), txt.detach()
        a4w4 = sum("a4w4" in e.name and "8192x3072x3072" in e.name for e in prof.events())
        return img, txt, {n: x.grad for n, x in t.items()}, a4w4

    try:
        i0, t0, g0, n0 = run(False)
        i1, t1, g1, n1 = run(True)
    finally:
        reset(Mxfp6Gates())
    ref = base["o"][:S_TXT].float() @ base["w_txt"].float().t()
    e0, e1 = _rel(t0, ref), _rel(t1, ref)
    assert torch.equal(i0, i1) and torch.equal(g0["w_img"], g1["w_img"])
    assert e0 < e1 < 0.25  # MXFP6 -> MXFP4 error level on the text stream
    assert all(torch.isfinite(x).all() for x in g1.values())
    assert n1 == n0 + 1  # the text forward's A4W4 (the A4W4 backward already runs at this shape)


# -- forward healing: the healed flag, flipped in place, under one compiled function ---------------------------------


def _heal_check(fn, base, gates_on, gates_off, compiled):
    """``fn(t, holder)`` reads ``holder._mxfp6_fwd_fp4_healed`` as the model does. Returns nothing; asserts that
    healing in place equals a fresh run with the MXFP4 forward off (bitwise: output and every gradient), costs exactly
    one compiled graph, and that un-healing is a cache hit reproducing the unhealed run."""
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import (
        Mxfp6Gates,
        reset,
    )

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._mxfp6_fwd_fp4_healed = False

    def step(f, holder):
        t = {n: x.clone().requires_grad_(True) for n, x in base.items()}
        loss, out = f(t, holder)
        loss.backward()
        torch.cuda.synchronize()
        return out.detach(), {n: x.grad for n, x in t.items()}

    def graphs():
        return torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)

    try:
        reset(Mxfp6Gates(**gates_off))
        torch._dynamo.reset()
        ref_o, ref_g = step(torch.compile(fn) if compiled else fn, Holder())

        reset(Mxfp6Gates(**gates_on))
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        f, holder = (torch.compile(fn) if compiled else fn), Holder()
        fp4_o, fp4_g = step(f, holder)
        n_fp4 = graphs()
        holder._mxfp6_fwd_fp4_healed = True
        heal_o, heal_g = step(f, holder)
        n_heal = graphs()
        holder._mxfp6_fwd_fp4_healed = False
        back_o, back_g = step(f, holder)
        n_back = graphs()
    finally:
        reset(Mxfp6Gates())
    assert torch.equal(heal_o, ref_o) and not torch.equal(heal_o, fp4_o)
    for n in ref_g:
        assert torch.equal(heal_g[n], ref_g[n]), n
    assert torch.equal(back_o, fp4_o) and all(torch.equal(back_g[n], fp4_g[n]) for n in fp4_g)
    if compiled:
        assert n_heal > n_fp4 and n_back == n_heal  # one new entry for the healed variant; un-healing is a cache hit


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compile"])
def test_fwd_fp4_heal_joint_txt_proj(compiled):
    from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
        MXFP6JointProjFunction,
    )
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import gates

    dev, bf16 = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(3)
    rn = lambda *s, scale=1.0: torch.randn(*s, device=dev, dtype=bf16, generator=g) * scale  # noqa: E731
    base = dict(o=rn(S, B, K), w_img=rn(K, K, scale=0.02), w_txt=rn(K, K, scale=0.02))
    g_img, g_txt = rn(S - S_TXT, B, K), rn(S_TXT, B, K)

    def fn(t, holder):  # as joint_proj_pair: the gate, unless the text projection is healed
        fp4_txt = gates().fwd_fp4_joint_txt_proj and not holder._mxfp6_fwd_fp4_healed
        out = MXFP6JointProjFunction.apply(t["o"], S_TXT, t["w_img"], t["w_txt"], False, True, False, None, fp4_txt)
        loss = (out[0].float() * g_img.float()).sum() + (out[1].float() * g_txt.float()).sum()
        return loss, out[1]

    common = dict(gemm_layout="tilescale", bwd_fp4_dgrad=True, bwd_fp4_wgrad=True, joint_proj=True)
    _heal_check(fn, base, dict(common, fwd_fp4_joint_txt_proj=True), dict(common), compiled)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compile"])
def test_fwd_fp4_heal_joint_mlp(compiled):
    from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
        MXFP6MLPFunction,
    )
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import gates

    dev, bf16 = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(4)
    rn = lambda *s, scale=1.0: torch.randn(*s, device=dev, dtype=bf16, generator=g) * scale  # noqa: E731
    base = dict(x=rn(S_TXT, B, K), w1=rn(F, K, scale=0.02), b1=rn(F, scale=0.02), w2=rn(K, F, scale=0.02))
    g_out = rn(S_TXT, B, K)

    def fn(t, holder):  # as MXFP6FusedMLP.forward for the text stream's MLP
        fp4 = not holder._mxfp6_fwd_fp4_healed
        fp4_fc1 = fp4 and gates().joint_mlp_fp4(False, "fc1")
        fp4_fc2 = fp4 and gates().joint_mlp_fp4(False, "fc2")
        out = MXFP6MLPFunction.apply(t["x"], t["w1"], t["b1"], t["w2"], False, True, False, fp4_fc2, fp4_fc1, False,
                                     False)[0]
        return (out.float() * g_out.float()).sum(), out

    common = dict(gemm_layout="tilescale", bwd_fp4_dgrad=True, bwd_fp4_wgrad=True)
    on = dict(common, fwd_fp4_joint_mlp=True, fwd_fp4_joint_mlp_parts="txt_fc1,txt_fc2")
    _heal_check(fn, base, on, dict(common), compiled)
