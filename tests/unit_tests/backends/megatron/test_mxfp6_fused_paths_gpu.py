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


# -- tensorwise FP8 forward (Mxfp6Gates.fwd_fp8, the forward-healing target "fp8") ------------------------------------

_F8_COMMON = dict(gemm_layout="tilescale", bwd_fp4_dgrad=True, bwd_fp4_wgrad=True)


def _f8_ready():
    import primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local as L

    if not L.f8_available():
        pytest.skip("no FP8 tilescale kernels in this Primus-Turbo / aiter")
    return L


def _f8_attach(*weights):
    import primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local as L

    for w in weights:
        w._mxfp6_f8 = L.F8State(w.device)
    return [w._mxfp6_f8 for w in weights]


def _f8_dq(t, q):
    """The FP8 forward's operand as the GEMM sees it: E4M3 of t * q (saturating), dequantized. q must be the packer's
    fp32 multiplier bit for bit: x * q lands on E4M3 rounding ties often enough that a last-bit change in q moves ~1%
    of the codes by an ulp."""
    return (t.float() * q).clamp(-448, 448).to(torch.float8_e4m3fn).float() / q


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compile"])
def test_fwd_fp8_linear(compiled):
    """FP8 forward of a plain linear: the output is the product of the FP8 operands; the backward reads the same MX
    columns as the MX forward's, so every gradient is bitwise the MX forward's."""
    L = _f8_ready()
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import (
        Mxfp6Gates,
        reset,
    )

    dev, bf16 = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(41)
    x0 = torch.randn(S, B, K, device=dev, dtype=bf16, generator=g)
    w0 = torch.randn(K, K, device=dev, dtype=bf16, generator=g) * 0.02
    g_out = torch.randn(S, B, K, device=dev, dtype=bf16, generator=g)

    def run(fp8):
        reset(Mxfp6Gates(**_F8_COMMON, fwd_fp8=fp8))
        torch._dynamo.reset()
        x, w = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
        (st,) = _f8_attach(w)
        st.q["a"].fill_(448.0 / x0.float().abs().max().item())
        st.q["w"].fill_(448.0 / w0.float().abs().max().item())

        def fn(x, w):
            return L.MXFP6LinearFunction.apply(x, w, None, False, None, 0, 0, False, True, False, False, False)[0]

        out = (torch.compile(fn) if compiled else fn)(x, w)
        (out.float() * g_out.float()).sum().backward()
        return out.detach(), x.grad, w.grad, st

    try:
        o8, gx8, gw8, st = run(True)
        o6, gx6, gw6, _ = run(False)
    finally:
        reset(Mxfp6Gates())
    qa, qw = st.q["a"].item(), st.q["w"].item()
    ref = _f8_dq(x0.reshape(-1, K), qa) @ _f8_dq(w0, qw).t()
    assert _rel(o8.reshape(-1, K), ref) < 2**-7
    assert torch.equal(gx8, gx6) and torch.equal(gw8, gw6)
    assert st.amax["a"].item() == x0.float().abs().max().item()
    assert st.amax["w"].item() == w0.float().abs().max().item()

    # evaluation (no grad): the same rows, and the amax goes to the scratch slot, never the training one
    reset(Mxfp6Gates(**_F8_COMMON, fwd_fp8=True))
    try:
        for t in st.amax.values():
            t.zero_()
        w0._mxfp6_f8 = st
        with torch.no_grad():
            o_ev = L.MXFP6LinearFunction.forward(x0, w0, None, False, None, 0, 0, False, False, False)[0]
    finally:
        reset(Mxfp6Gates())
    assert torch.equal(o_ev, o8)
    assert st.amax["a"].item() == 0 and st.amax["w"].item() == 0 and st.scratch.item() > 0


def test_fwd_fp8_mlp():
    """FP8 forward of the fused MLP: fc1, the bias+GELU prologue inside fc2's FP8 pack, fc2; the columns the backward
    reads of x, w1 and w2 are bitwise the MX forward's."""
    L = _f8_ready()
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import (
        Mxfp6Gates,
        reset,
    )

    dev, bf16 = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(42)
    x = torch.randn(S, B, K, device=dev, dtype=bf16, generator=g)
    w1 = torch.randn(F, K, device=dev, dtype=bf16, generator=g) * 0.02
    b1 = torch.randn(F, device=dev, dtype=bf16, generator=g) * 0.02
    w2 = torch.randn(K, F, device=dev, dtype=bf16, generator=g) * 0.02

    def run(fp8):
        reset(Mxfp6Gates(**_F8_COMMON, fwd_fp8=fp8, fwd_fp8_seed=fp8))  # seed step: scales from the operands
        s1, s2 = _f8_attach(w1, w2)
        out = L.MXFP6MLPFunction.forward(x, w1, b1, w2, False, True, False, False, False, False, False)
        torch.cuda.synchronize()
        return out, s1, s2

    try:
        o8, s1, s2 = run(True)
        o6, _, _ = run(False)
    finally:
        reset(Mxfp6Gates())
    # the FP8 chain emulated: seed scales from each operand's amax, fc1 out in bf16, bias+GELU on it, fc2
    q = lambda t: 448.0 / t.float().abs().amax()  # noqa: E731 -- fp32, as the seed step computes it
    x2 = x.reshape(-1, K)
    y1 = (_f8_dq(x2, q(x)) @ _f8_dq(w1, q(w1)).t()).to(bf16)
    h = torch.nn.functional.gelu((y1 + b1).float(), approximate="tanh").to(bf16)
    assert abs(s2.amax["a"].item() / h.float().abs().max().item() - 1) < 2**-6  # fc1 accumulation order differs
    ref = _f8_dq(h, 448.0 / s2.amax["a"]) @ _f8_dq(w2, q(w2)).t()
    assert _rel(o8[0].reshape(-1, K), ref) < 2**-5  # fc2 requantizes h: a bf16 ulp in y1 moves its FP8 rounding
    exact = torch.nn.functional.gelu(x2.float() @ w1.float().t() + b1.float(), approximate="tanh") @ w2.float().t()
    assert _rel(o8[0].reshape(-1, K), exact) < 0.08  # FP8 forward error level of two chained GEMMs
    for i in (2, 3, 6, 7, 8, 9):  # x, w1, w2 columns (a_col depends on y1, which differs)
        assert torch.equal(o8[i], o6[i]), i
    assert s1.amax["a"].item() == x.float().abs().max().item()
    assert s1.amax["w"].item() == w1.float().abs().max().item() and s2.amax["w"].item() == w2.float().abs().max().item()
    assert s2.amax["a"].item() > 0


def test_fwd_fp8_qkv():
    """FP8 forward of the single-block QKV (bias in the FP8 GEMM's epilogue): mixed_qkv at the FP8 error level against
    the MX forward's; the columns the backward reads unchanged."""
    L = _f8_ready()
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import (
        Mxfp6Gates,
        reset,
    )

    dev, bf16 = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(43)
    n3 = 3 * K
    x = torch.randn(S, B, K, device=dev, dtype=bf16, generator=g)
    w = torch.randn(n3, K, device=dev, dtype=bf16, generator=g) * 0.02
    b = torch.randn(n3, device=dev, dtype=bf16, generator=g) * 0.02
    wq = torch.ones(D, device=dev, dtype=bf16)
    wk = torch.ones(D, device=dev, dtype=bf16)
    cos = torch.ones(S, 1, 1, D, device=dev, dtype=torch.float32)
    sin = torch.zeros(S, 1, 1, D, device=dev, dtype=torch.float32)

    (st,) = _f8_attach(w)

    def single(fp8):
        reset(Mxfp6Gates(**_F8_COMMON, fwd_fp8=fp8, fwd_fp8_seed=fp8))
        return L.MXFP6QKVNormRopeFunction.forward(x, w, b, wq, wk, cos, sin, 1e-6, False, False, True, False)

    try:
        a8, a6 = single(True), single(False)
    finally:
        reset(Mxfp6Gates())
    q = lambda t: 448.0 / t.float().abs().amax()  # noqa: E731 -- fp32, as the seed step computes it
    ref = _f8_dq(x.reshape(-1, K), q(x)) @ _f8_dq(w, q(w)).t() + b.float()
    assert _rel(a8[3], ref) < 2**-7
    assert _rel(a6[3], x.reshape(-1, K).float() @ w.float().t() + b.float()) < 0.05
    assert all(torch.equal(a8[i], a6[i]) for i in (6, 7, 8, 9))
    assert st.amax["a"].item() == x.float().abs().max().item()
    assert st.amax["w"].item() == w.float().abs().max().item()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compile"])
def test_fwd_fp8_joint_qkv(compiled):
    """FP8 forward of the joint QKV: both streams' FP8 GEMMs (bias in the epilogue) into row slices of one mixed_qkv,
    each with its own amax slots; the backward runs on the MX columns and stays at the MX forward's gradients."""
    L = _f8_ready()
    import primus.backends.megatron.core.models.diffusion.flux.attention as A
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import (
        Mxfp6Gates,
        reset,
    )

    dev, bf16, eps = "cuda", torch.bfloat16, 1e-6
    g = torch.Generator(device=dev).manual_seed(44)
    rn = lambda *s, scale=1.0: torch.randn(*s, device=dev, dtype=bf16, generator=g) * scale  # noqa: E731
    s_img = S - S_TXT
    base = dict(x_a=rn(S_TXT, B, K), x_b=rn(s_img, B, K, scale=2.0), w_a=rn(3 * K, K, scale=0.02),
                w_b=rn(3 * K, K, scale=0.03), b_a=rn(3 * K, scale=0.02), b_b=rn(3 * K, scale=0.02),
                wq_a=1 + rn(D, scale=0.1), wk_a=1 + rn(D, scale=0.1), wq_b=1 + rn(D, scale=0.1),
                wk_b=1 + rn(D, scale=0.1))
    ca, sa = A._rope_cos_sin(torch.randn(S_TXT, B, 1, D, device=dev, generator=g), bf16)
    cb, sb = A._rope_cos_sin(torch.randn(s_img, B, 1, D, device=dev, generator=g), bf16)
    g_out = rn(S, B, H, D)

    def fn(t):
        out = L.MXFP6JointQKVFunction.apply(t["x_a"], t["x_b"], t["w_a"], t["w_b"], t["b_a"], t["b_b"], t["wq_a"],
                                            t["wk_a"], t["wq_b"], t["wk_b"], ca, sa, cb, sb, eps, True, False, True,
                                            False)
        loss = sum((o.float() * g_out.float()).sum() for o in out[:3])
        return loss, out[3]

    def run(fp8):
        reset(Mxfp6Gates(**_F8_COMMON, joint_qkv=True, fwd_fp8=fp8, fwd_fp8_seed=fp8))
        torch._dynamo.reset()
        t = {n: x.clone().requires_grad_(True) for n, x in base.items()}
        st = _f8_attach(t["w_a"], t["w_b"])
        loss, mixed = (torch.compile(fn) if compiled else fn)(t)
        mixed = mixed.detach().clone()  # the backward reuses mixed_qkv's storage
        loss.backward()
        return mixed, {n: x.grad for n, x in t.items()}, st

    try:
        m8, g8, (sa_, sb_) = run(True)
        m6, g6, _ = run(False)
    finally:
        reset(Mxfp6Gates())
    q = lambda t: 448.0 / t.float().abs().amax()  # noqa: E731 -- fp32, as the seed step computes it
    xa, xb = base["x_a"].reshape(-1, K), base["x_b"].reshape(-1, K)
    ref_a = _f8_dq(xa, q(xa)) @ _f8_dq(base["w_a"], q(base["w_a"])).t() + base["b_a"].float()
    ref_b = _f8_dq(xb, q(xb)) @ _f8_dq(base["w_b"], q(base["w_b"])).t() + base["b_b"].float()
    ma = xa.shape[0]
    assert _rel(m8[:ma], ref_a) < 2**-7 and _rel(m8[ma:], ref_b) < 2**-7
    for st, x, w in ((sa_, xa, base["w_a"]), (sb_, xb, base["w_b"])):
        assert st.amax["a"].item() == x.float().abs().max().item()
        assert st.amax["w"].item() == w.float().abs().max().item()
    for n in g6:  # FP8 vs MXFP6 forward: the gradients differ only through the forward's values
        assert torch.isfinite(g8[n]).all() and _rel(g8[n], g6[n]) < 0.1, n
