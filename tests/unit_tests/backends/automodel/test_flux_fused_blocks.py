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
  op does not take (token counts off its tiling) must take the fallback path and still match. The
  fallbacks match the references too, so each test also counts the calls of every op: a guard that
  always declined would otherwise pass. An adaLN norm the op cannot compute (a LayerNorm with weights)
  must take the fallback too.

  A QKV projection that a low-precision swap replaced runs its own forward, its bias included, so the
  q/k op must not add that bias again.

The ops are specialized to the training's layout (hidden size 3072, head dim 128, token counts in
multiples of 256), so the tests run one block of it on a small batch. They need a GPU, Primus-Turbo
with the ops above and the diffusers forwards the patches re-implement, and skip elsewhere. The ops
run on every GPU (FlyDSL kernels on gfx950, Triton and eager PyTorch elsewhere); the adaLN patches
decline off gfx950 only because the ops are no faster there, so the tests lift that gate and check
the wiring on any GPU. The class-level patches are undone after each test.
"""

import copy

import pytest

torch = pytest.importorskip("torch")
transformer_flux = pytest.importorskip("diffusers.models.transformers.transformer_flux")

_OPS = ("adaln_modulate", "gated_residual_adaln", "linear_gelu", "dit_qk_norm_rope")


def _turbo_flux_ops() -> bool:
    # Importing Primus-Turbo needs a GPU, so only after torch.cuda.is_available().
    try:
        import primus_turbo.pytorch.ops as ops
    except ImportError:
        return False
    return all(hasattr(ops, name) for name in _OPS)


def _known_diffusers() -> bool:
    from diffusers.models.activations import GELU
    from diffusers.models.attention import FeedForward
    from diffusers.models.normalization import AdaLayerNormZeroSingle

    from primus.backends.automodel.models.flux import (
        adaln,
        adaln_dual,
        mlp_gelu,
        pinned_source,
    )

    return all(
        pinned_source.digest(fn) == expected
        for fn, expected in (
            (transformer_flux.FluxSingleTransformerBlock.forward, mlp_gelu.SINGLE_FORWARD_DIGEST),
            (FeedForward.forward, mlp_gelu.FEED_FORWARD_DIGEST),
            (GELU.forward, mlp_gelu.GELU_FORWARD_DIGEST),
            (AdaLayerNormZeroSingle.forward, adaln.ADA_NORM_FORWARD_DIGEST),
            (transformer_flux.FluxTransformerBlock.forward, adaln_dual.DUAL_FORWARD_DIGEST),
        )
    )


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the ops run on a GPU"),
    pytest.mark.skipif(
        not (torch.cuda.is_available() and _turbo_flux_ops()),
        reason="needs Primus-Turbo's adaln_modulate, gated_residual_adaln, linear_gelu and dit_qk_norm_rope",
    ),
    pytest.mark.skipif(
        not _known_diffusers(), reason="the patches re-implement diffusers 0.38.0's to 0.43.0's forwards"
    ),
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
def patches(flux_fused_patches, monkeypatch):
    """The patch modules, with the adaLN patches' gfx950-only gate lifted."""
    from primus.backends.automodel.models.flux.turbo_ops import turbo_ops

    def on_any_gpu(tag, *names, gain_on_gfx950_only=False):
        return turbo_ops(tag, *names)

    monkeypatch.setattr(flux_fused_patches.adaln, "turbo_ops", on_any_gpu)
    monkeypatch.setattr(flux_fused_patches.adaln_dual, "turbo_ops", on_any_gpu)
    return flux_fused_patches


@pytest.fixture
def calls(monkeypatch):
    """How often each op ran. Set before the installs, which may keep a reference to an op."""
    import primus_turbo.pytorch.ops as ops

    counts = dict.fromkeys(_OPS, 0)
    for name in _OPS:

        def counted(*args, _name=name, _real=getattr(ops, name), **kwargs):
            counts[_name] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(ops, name, counted)
    return counts


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


class SwappedLinear(torch.nn.Linear):
    """What a low-precision swap puts in place of a linear: a subclass with its own forward."""


def _swap_qkv(block):
    """``block`` with its fused QKV projections replaced by ``SwappedLinear`` modules of the same weights."""
    for name in ("to_qkv", "to_added_qkv"):
        linear = getattr(block.attn, name, None)
        if linear is None:
            continue
        swapped = SwappedLinear(
            linear.in_features, linear.out_features, device=linear.weight.device, dtype=linear.weight.dtype
        )
        swapped.load_state_dict(linear.state_dict())
        setattr(block.attn, name, swapped)
    return block


def _installed_processor(block):
    """``block`` with the processor its attention is now built with. The blocks are built before the
    installs, so that the references run diffusers' processor."""
    block.attn.set_processor(transformer_flux.FluxAttention._default_processor_cls())
    return block


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


@pytest.mark.parametrize("s_txt,takes_adaln", [(256, True), (64, False)])
def test_single_stream_block(s_txt, takes_adaln, patches, calls, gen):
    """GELU + concat, q/k norm + RoPE and adaLN fused; 64 text tokens leave the adaLN's tiling and keep
    the fused GELU forward around the other two ops."""
    block = _block(transformer_flux.FluxSingleTransformerBlock)
    args = _inputs(s_txt, gen)
    ref32, ref16 = _run(block, torch.float32, *args), _run(block, BF16, *args)
    assert patches.mlp_gelu.install() and patches.qk_norm_rope.install() and patches.adaln.install()
    _compare(_run(_installed_processor(block), BF16, *args), ref32, ref16)
    assert calls == {
        "adaln_modulate": int(takes_adaln),
        "gated_residual_adaln": 0,
        "linear_gelu": 1,
        "dit_qk_norm_rope": 1,
    }


def test_single_stream_block_with_an_affine_ada_norm(patches, calls, gen):
    """The adaLN forward computes only an affine-free LayerNorm from its eps; one with weights keeps the
    fused GELU forward around the other two ops."""
    block = _block(transformer_flux.FluxSingleTransformerBlock)
    block.norm.norm = torch.nn.LayerNorm(C, eps=1e-6, device=DEV)
    with torch.no_grad():
        block.norm.norm.weight.normal_(1.0, 0.1)
        block.norm.norm.bias.normal_(0.0, 0.1)
    block = block.to(BF16).float()
    args = _inputs(256, gen)
    ref32, ref16 = _run(block, torch.float32, *args), _run(block, BF16, *args)
    assert patches.mlp_gelu.install() and patches.qk_norm_rope.install() and patches.adaln.install()
    _compare(_run(_installed_processor(block), BF16, *args), ref32, ref16)
    assert calls == {
        "adaln_modulate": 0,
        "gated_residual_adaln": 0,
        "linear_gelu": 1,
        "dit_qk_norm_rope": 1,
    }


@pytest.mark.parametrize("s_txt,takes_op", [(256, True), (128, True), (48, False)])
def test_dual_stream_block(s_txt, takes_op, patches, calls, gen):
    """GELU, q/k norm + RoPE and the gated residual + second adaLN fused. The two streams may differ
    in length (the op takes any multiples of 32); 48 text tokens leave its tiling and keep diffusers'
    forward around the other two ops."""
    block = _block(transformer_flux.FluxTransformerBlock)
    args = _inputs(s_txt, gen)
    ref32, ref16 = _run(block, torch.float32, *args), _run(block, BF16, *args)
    assert patches.mlp_gelu.install() and patches.qk_norm_rope.install() and patches.adaln_dual.install()
    _compare(_run(_installed_processor(block), BF16, *args), ref32, ref16)
    assert calls == {
        "adaln_modulate": 0,
        "gated_residual_adaln": int(takes_op),
        "linear_gelu": 2,
        "dit_qk_norm_rope": 1,
    }


@pytest.mark.parametrize("block_cls", ["FluxSingleTransformerBlock", "FluxTransformerBlock"])
def test_swapped_qkv_projections(block_cls, patches, calls, gen):
    """QKV projections a low-precision swap replaced: the q/k op still runs, without adding their biases."""
    block = _swap_qkv(_block(getattr(transformer_flux, block_cls)))
    args = _inputs(256, gen)
    ref32, ref16 = _run(block, torch.float32, *args), _run(block, BF16, *args)
    assert patches.qk_norm_rope.install()
    _compare(_run(_installed_processor(block), BF16, *args), ref32, ref16)
    assert calls["dit_qk_norm_rope"] == 1
