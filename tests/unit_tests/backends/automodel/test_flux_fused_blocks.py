###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Numerics of the FLUX blocks that call Primus-Turbo's fused FLUX ops, against diffusers' blocks.

WHAT THESE ARE DEFENDING:
  The patches re-implement diffusers' block forwards with Primus-Turbo's ops in the middle: the
  MLP's up-projection + GELU (``linear_gelu``, both block types), the q/k RMSNorm + RoPE of the
  attention processor (``dit_qk_norm_rope``, which adds the QKV biases), the single-stream adaLN
  (``adaln_modulate``) and the dual-stream gated residual + second adaLN (``gated_residual_adaln``).
  A wrong chunk, slice, bias or residual in that wiring still produces plausible numbers, so the outputs and every input
  and parameter gradient of a patched block are compared with diffusers' block in FP32, and must be
  as close to it as diffusers' block in BF16 is (twice its distance, plus a small floor). Calls an
  op does not take (token counts off its tiling) must take the fallback path and still match.

The ops are specialized to the training's layout (hidden size 3072, head dim 128, token counts in
multiples of 256), so the tests run one block of it on a small batch. They need a gfx950 GPU,
Primus-Turbo with the ops above and the diffusers forwards the patches re-implement, and skip
elsewhere. The class-level patches are undone after each test.
"""

import copy
import hashlib
import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformer_flux = pytest.importorskip("diffusers.models.transformers.transformer_flux")


def _gfx950() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")


def _turbo_flux_ops() -> bool:
    # Importing Primus-Turbo needs a GPU, so only after _gfx950().
    try:
        import primus_turbo.pytorch.ops as ops
    except ImportError:
        return False
    return all(
        hasattr(ops, name)
        for name in ("adaln_modulate", "gated_residual_adaln", "linear_gelu", "dit_qk_norm_rope")
    )


def _known_diffusers() -> bool:
    from primus.backends.automodel.models.flux import adaln_dual_flydsl, mlp_gelu

    def digest(fn):
        return hashlib.sha256(inspect.getsource(inspect.unwrap(fn)).encode()).hexdigest()

    return (
        digest(transformer_flux.FluxSingleTransformerBlock.forward) == mlp_gelu.SINGLE_FORWARD_SHA256
        and digest(transformer_flux.FluxTransformerBlock.forward) == adaln_dual_flydsl._DUAL_FORWARD_SHA256
    )


pytestmark = [
    pytest.mark.skipif(not _gfx950(), reason="the ops' FlyDSL kernels are written for gfx950"),
    pytest.mark.skipif(
        not (_gfx950() and _turbo_flux_ops()),
        reason="needs Primus-Turbo's adaln_modulate, gated_residual_adaln, linear_gelu and dit_qk_norm_rope",
    ),
    pytest.mark.skipif(not _known_diffusers(), reason="the patches re-implement diffusers 0.40.0's forwards"),
]

BF16, DEV = torch.bfloat16, "cuda"
B, S_IMG, C, H, D = 2, 256, 3072, 24, 128


def _rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def _check(name, got, ref32, ref16):
    err, err16 = _rel(got, ref32), _rel(ref16, ref32)
    assert err <= 2 * err16 + 2e-3, f"{name}: {err:.2e} from FP32, the BF16 reference {err16:.2e}"


def _randn(*shape, scale=1.0, gen):
    return (torch.randn(*shape, generator=gen, device=DEV) * scale).to(BF16)


@pytest.fixture
def gen():
    return torch.Generator(device=DEV).manual_seed(0)


@pytest.fixture
def patches(monkeypatch):
    """The patch modules, with every class attribute their installs set restored afterwards."""
    from diffusers.models.attention import FeedForward

    from primus.backends.automodel.models.flux import (
        adaln_dual_flydsl,
        adaln_flydsl,
        mlp_gelu,
        qk_norm_rope,
    )

    single, dual = transformer_flux.FluxSingleTransformerBlock, transformer_flux.FluxTransformerBlock
    processor = transformer_flux.FluxAttnProcessor
    monkeypatch.setattr(processor, "__call__", processor.__call__)
    for cls in (FeedForward, single, dual, transformer_flux.FluxTransformer2DModel):
        monkeypatch.setattr(cls, "forward", cls.forward)
    for cls, attr in (
        (FeedForward, mlp_gelu._PATCHED_ATTR),
        (processor, qk_norm_rope._PATCHED_ATTR),
        (single, adaln_flydsl._PATCHED_ATTR),
        (dual, adaln_dual_flydsl._PATCHED_ATTR),
    ):
        monkeypatch.setattr(cls, attr, False, raising=False)
    return SimpleNamespace(
        adaln_dual=adaln_dual_flydsl,
        adaln=adaln_flydsl,
        mlp_gelu=mlp_gelu,
        qk_norm_rope=qk_norm_rope,
    )


def _tables(s_txt):
    ids = torch.zeros(s_txt + S_IMG, 3)
    side = int(S_IMG**0.5)
    ids[s_txt:, 1] = torch.arange(S_IMG) // side
    ids[s_txt:, 2] = torch.arange(S_IMG) % side
    cos, sin = transformer_flux.FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])(ids.to(DEV))
    # FSDP2's mixed precision passes the tables to the blocks in BF16.
    return cos.to(BF16), sin.to(BF16)


def _block(cls):
    """A block of the training's shape with fused QKV projections and BF16-representable FP32 weights."""
    torch.manual_seed(0)
    block = cls(dim=C, num_attention_heads=H, attention_head_dim=D).to(DEV)
    block.attn.fuse_projections()
    with torch.no_grad():
        for name, p in block.named_parameters():
            if name.split(".")[-2].startswith("norm_"):
                p.normal_(1.0, 0.1)
    return block.to(BF16).float()


def _inputs(s_txt, gen):
    inputs = (_randn(B, S_IMG, C, gen=gen), _randn(B, s_txt, C, gen=gen), _randn(B, C, gen=gen))
    upstream = (_randn(B, s_txt, C, gen=gen), _randn(B, S_IMG, C, gen=gen))
    return inputs, upstream, _tables(s_txt)


def _run(block, dtype, inputs, upstream, rope):
    """Outputs, input gradients and parameter gradients of a copy of ``block`` run in ``dtype``."""
    block = copy.deepcopy(block).to(dtype)
    leaves = [t.detach().to(dtype).requires_grad_() for t in inputs]
    outs = block(*leaves, image_rotary_emb=tuple(t.to(dtype) for t in rope))
    torch.autograd.backward(outs, [u.to(o.dtype) for u, o in zip(upstream, outs)])
    grads = {n: p.grad for n, p in block.named_parameters() if p.grad is not None}
    return [o.detach() for o in outs], [t.grad for t in leaves], grads


def _compare(got, ref32, ref16):
    for i, (g, r, b) in enumerate(zip(got[0], ref32[0], ref16[0])):
        _check(f"output {i}", g, r, b)
    for name, g, r, b in zip(("hidden_states", "encoder_hidden_states", "temb"), got[1], ref32[1], ref16[1]):
        _check(f"{name}.grad", g, r, b)
    assert got[2].keys() == ref32[2].keys()
    for name, g in got[2].items():
        _check(f"{name}.grad", g, ref32[2][name], ref16[2][name])


@pytest.mark.parametrize("s_txt", [256, 64])
def test_single_stream_block(s_txt, patches, gen):
    """GELU + concat, q/k norm + RoPE and adaLN fused; 64 text tokens leave the adaLN's tiling."""
    block = _block(transformer_flux.FluxSingleTransformerBlock)
    args = _inputs(s_txt, gen)
    ref32, ref16 = _run(block, torch.float32, *args), _run(block, BF16, *args)
    assert patches.mlp_gelu.install() and patches.qk_norm_rope.install() and patches.adaln.install()
    _compare(_run(block, BF16, *args), ref32, ref16)


@pytest.mark.parametrize("s_txt,takes_op", [(256, True), (128, True), (48, False)])
def test_dual_stream_block(s_txt, takes_op, patches, gen, monkeypatch):
    """GELU, q/k norm + RoPE and the gated residual + second adaLN fused. The two streams may differ
    in length (the op takes any multiples of 32); 48 text tokens leave its tiling and keep diffusers'
    forward around the other two ops."""
    import primus_turbo.pytorch.ops as ops

    calls = []
    real = ops.gated_residual_adaln
    monkeypatch.setattr(ops, "gated_residual_adaln", lambda *a, **k: calls.append(1) or real(*a, **k))
    block = _block(transformer_flux.FluxTransformerBlock)
    args = _inputs(s_txt, gen)
    ref32, ref16 = _run(block, torch.float32, *args), _run(block, BF16, *args)
    assert patches.mlp_gelu.install() and patches.qk_norm_rope.install() and patches.adaln_dual.install()
    _compare(_run(block, BF16, *args), ref32, ref16)
    assert bool(calls) == takes_op
