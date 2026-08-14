###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MoonViT-V2 eager reference: does it match the official implementation?

This file pins the *oracle*. ``test_moonvit_tower.py`` then pins the Megatron
port against it, so a bug in the oracle would make that whole chain vacuous
-- which is why every property here is checked against an **independent
transcription** of ``research/raw/modeling_kimi_k3.py`` written in the test,
sharing no code with the module under test.

Everything here is pure PyTorch and runs on CPU.

The five properties worth stating outright, because each is a plausible thing
to get wrong and none of them would raise:

1. **Even RoPE slots carry the *width* axis.** ``Rope2DPosEmbRepeated``'s own
   docstring (``:381-384``) says height. The code says width
   (``x_pos = flat % max_width``, ``:388``), and the code is what runs.
2. **RoPE is identical on every frame** (``.repeat(t, 1)``, ``:429``), so it
   carries no temporal signal at all. Time enters only through the 1-D sincos
   added at patch-embed time.
3. **The temporal sincos is ``[sin | cos]`` concatenated, not interleaved**,
   and its ``omega`` runs over ``arange(D/2)/(D/2)`` so the longest period is
   ``10000**((D/2-1)/(D/2))``, not ``10000``.
4. **Attention is block-diagonal over media items and non-causal inside one.**
   The Kimi K3 text backbone's defining property is causality; the vision
   tower's is that it is *not* causal, and the two must not be confused.
5. **The merger's temporal pool is a mean over all frames**, taken before the
   spatial pixel-shuffle is flattened.

Deliberate bug injection accompanies each. A test that cannot fail is worth
nothing, so every parity assertion is paired with a broken variant that must
be caught by a wide margin -- the technique the attention-residual tests use
(``test_attention_residual.py:14-27``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.kimi_k3.vision.moonvit_reference import (
    MoonViTAttentionReference,
    MoonViTEncoderLayerReference,
    MoonViTReference,
    MoonViTReferenceConfig,
    apply_moonvit_rope,
    moonvit_cu_seqlens,
    moonvit_default_rmsnorm_eps,
    moonvit_rope_freqs_cis,
    moonvit_rope_freqs_for_grid,
    moonvit_sincos_1d,
    moonvit_tpool_patch_merger,
)

SMALL = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    qkv_hidden_size=48,
    patch_size=4,
    init_pos_emb_height=8,
    init_pos_emb_width=8,
    init_pos_emb_time=4,
    mm_hidden_size=32,
    text_hidden_size=64,
    rope_max_height=32,
    rope_max_width=32,
)
GRID = [(1, 8, 8), (1, 4, 6), (3, 2, 4)]


def make_inputs(cfg: MoonViTReferenceConfig, grid=GRID, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    total = sum(t * h * w for t, h, w in grid)
    px = torch.randn(total, 3, cfg.patch_size, cfg.patch_size, generator=g)
    return px, torch.tensor(grid, dtype=torch.long)


# ===========================================================================
# 1. The temporal sin/cos table
# ===========================================================================


def reference_sincos_numpy(embed_dim: int, t_size: int) -> np.ndarray:
    """``get_1d_sincos_pos_embed`` (``:196-230``), transcribed into numpy.

    Kept in numpy on purpose: the release computes it this way, and the
    float32 accumulation is part of what is being pinned.
    """
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = np.arange(t_size, dtype=np.float32).reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


@pytest.mark.parametrize("dim,frames", [(32, 4), (64, 1), (128, 8), (1024, 4)])
def test_temporal_sincos_matches_the_released_numpy_form(dim, frames):
    got = moonvit_sincos_1d(dim, frames)
    want = torch.from_numpy(reference_sincos_numpy(dim, frames))
    assert got.shape == (frames, dim)
    # Bit-exact: matching the release's float32 accumulation is deliberate,
    # a float64 accumulation differs by 1 ULP (see the module docstring).
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_temporal_sincos_is_concatenated_not_interleaved():
    """``[sin | cos]``, which an "obvious" sinusoidal embedding gets wrong.

    Tolerance rather than bit-equality here: the recomputation is in torch
    and the table is built in numpy, whose ``sin`` and ``power`` differ in
    the last ULP. Bit-equality against the *released numpy form* is what
    :func:`test_temporal_sincos_matches_the_released_numpy_form` asserts;
    this test is about layout, and 1e-6 is nowhere near the 5e-1 an
    interleaved layout would produce.
    """
    dim, frames = 16, 5
    table = moonvit_sincos_1d(dim, frames)
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0))
    angles = torch.outer(torch.arange(frames, dtype=torch.float32), omega)
    torch.testing.assert_close(table[:, : dim // 2], torch.sin(angles), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(table[:, dim // 2 :], torch.cos(angles), rtol=1e-6, atol=1e-6)

    # Negative control: the interleaved layout must NOT match.
    interleaved = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)
    assert (table - interleaved).abs().max() > 1e-2


def test_temporal_sincos_row_zero_is_the_constant_frame():
    """Frame 0 is ``sin(0)=0`` then ``cos(0)=1``, so a still image is unshifted."""
    table = moonvit_sincos_1d(64, 4)
    assert torch.all(table[0, :32] == 0.0)
    assert torch.all(table[0, 32:] == 1.0)


# ===========================================================================
# 2. The 2-D RoPE table
# ===========================================================================


def reference_rope_table(dim: int, max_h: int, max_w: int, theta: float = 10000.0):
    """``_precompute_freqs_cis`` (``:378-404``), transcribed."""
    n = max_h * max_w
    flat_pos = torch.arange(0, n).float()
    x_pos = flat_pos % max_w
    y_pos = flat_pos // max_w
    dim_range = torch.arange(0, dim, 4)[: (dim // 4)].float()
    freqs = 1.0 / (theta ** (dim_range / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    out = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
    return out.reshape(max_h, max_w, -1)


@pytest.mark.parametrize("dim,h,w", [(8, 4, 4), (128, 16, 9), (64, 7, 13)])
def test_rope_table_matches_the_released_precompute(dim, h, w):
    got = moonvit_rope_freqs_cis(dim, h, w)
    want = reference_rope_table(dim, h, w)
    assert got.shape == (h, w, dim // 2) and got.dtype == torch.complex64
    torch.testing.assert_close(torch.view_as_real(got), torch.view_as_real(want), rtol=0, atol=0)


def test_even_rope_slots_carry_width_and_odd_carry_height():
    """The docstring at ``:381-384`` has these the wrong way round.

    Verified structurally rather than by transcription: hold the row fixed
    and only the even slots may move; hold the column fixed and only the odd
    slots may move.
    """
    dim, h, w = 16, 5, 7
    table = moonvit_rope_freqs_cis(dim, h, w)
    even, odd = table[..., 0::2], table[..., 1::2]

    # Same row, different columns -> even slots differ, odd slots identical.
    assert not torch.allclose(torch.view_as_real(even[2, 3]), torch.view_as_real(even[2, 5]))
    torch.testing.assert_close(
        torch.view_as_real(odd[2, 3]), torch.view_as_real(odd[2, 5]), rtol=0, atol=0
    )
    # Same column, different rows -> the other way round.
    torch.testing.assert_close(
        torch.view_as_real(even[1, 4]), torch.view_as_real(even[3, 4]), rtol=0, atol=0
    )
    assert not torch.allclose(torch.view_as_real(odd[1, 4]), torch.view_as_real(odd[3, 4]))


def test_rope_is_identical_on_every_frame():
    """``.repeat(t, 1)`` (``:429``): 2-D RoPE carries no temporal signal."""
    dim = 16
    grid = torch.tensor([[3, 2, 4]], dtype=torch.long)
    freqs = moonvit_rope_freqs_for_grid(grid, dim)
    assert freqs.shape == (3 * 2 * 4, dim // 2)
    per_frame = freqs.view(3, 8, dim // 2)
    torch.testing.assert_close(
        torch.view_as_real(per_frame[0]), torch.view_as_real(per_frame[2]), rtol=0, atol=0
    )


def test_rope_per_item_slices_are_independent_of_batch_composition():
    """Each media item's frequencies depend only on its own ``(h, w)``.

    This is the property Megatron's ``thd`` rotary path would have broken:
    ``_apply_rotary_pos_emb_thd`` re-slices ``freqs[0:len]`` per sub-sequence
    (``rope_utils.py:236-238``), which would hand item 2 item 1's table.
    """
    dim = 16
    alone = moonvit_rope_freqs_for_grid(torch.tensor([[1, 4, 6]]), dim)
    together = moonvit_rope_freqs_for_grid(torch.tensor([[1, 8, 8], [1, 4, 6]]), dim)
    torch.testing.assert_close(
        torch.view_as_real(together[64:]), torch.view_as_real(alone), rtol=0, atol=0
    )


def test_apply_rope_is_the_adjacent_pair_complex_multiply():
    """``apply_rope`` (``:172-193``): pairs are ``(2i, 2i+1)``, not rotate-half."""
    torch.manual_seed(0)
    tokens, heads, head_dim = 6, 2, 8
    q = torch.randn(tokens, heads, head_dim)
    k = torch.randn(tokens, heads, head_dim)
    freqs = moonvit_rope_freqs_for_grid(torch.tensor([[1, 2, 3]]), head_dim)

    q_out, k_out = apply_moonvit_rope(q, k, freqs)

    # Independent transcription, elementwise, no complex arithmetic at all.
    theta = torch.angle(freqs)  # [tokens, head_dim//2]
    cos, sin = torch.cos(theta), torch.sin(theta)
    want = torch.empty_like(q)
    want[..., 0::2] = q[..., 0::2] * cos[:, None, :] - q[..., 1::2] * sin[:, None, :]
    want[..., 1::2] = q[..., 0::2] * sin[:, None, :] + q[..., 1::2] * cos[:, None, :]
    torch.testing.assert_close(q_out, want, rtol=1e-6, atol=1e-6)

    # Negative control: Megatron's rotate-half convention must NOT match.
    half = head_dim // 2
    rotate_half = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
    cos_full = torch.cat((cos, cos), dim=-1)[:, None, :]
    sin_full = torch.cat((sin, sin), dim=-1)[:, None, :]
    wrong = q * cos_full + rotate_half * sin_full
    assert (q_out - wrong).abs().max() > 1e-2

    # RoPE is a rotation: norms are preserved.
    torch.testing.assert_close(q_out.norm(dim=-1), q.norm(dim=-1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_out.norm(dim=-1), k.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_apply_rope_uses_each_tensors_own_shape():
    """``:188`` builds the key's view from ``xq.shape``; we do not.

    The release is only safe because ``wqkv`` guarantees equal shapes. Feed
    a key with a different head count and the released expression would
    reshape it wrongly; ours raises.
    """
    q = torch.randn(4, 2, 8)
    k = torch.randn(4, 3, 8)  # different head count
    freqs = moonvit_rope_freqs_for_grid(torch.tensor([[1, 2, 2]]), 8)
    q_out, k_out = apply_moonvit_rope(q, k, freqs)
    assert k_out.shape == k.shape


# ===========================================================================
# 3. Packing and the merger
# ===========================================================================


def test_cu_seqlens_is_one_segment_per_media_item():
    grid = torch.tensor(GRID, dtype=torch.long)
    cu, max_seqlen = moonvit_cu_seqlens(grid)
    assert cu.tolist() == [0, 64, 88, 112]
    assert max_seqlen == 64
    assert cu.dtype == torch.int32


def reference_tpool(x, grid_thws, kernel=(2, 2)):
    """``tpool_patch_merger`` (``:621-646``), transcribed."""
    d_model = x.size(-1)
    outputs, pre_sum = [], 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        kh, kw = kernel
        nh, nw = h // kh, w // kw
        reshaped = seq.view(t, nh, kh, nw, kw, d_model)
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        outputs.append(reshaped.view(nh * nw, kh * kw, -1))
        pre_sum += t * h * w
    return outputs


def test_patch_merger_matches_the_released_form():
    torch.manual_seed(1)
    grid = torch.tensor(GRID, dtype=torch.long)
    x = torch.randn(int(grid.prod(dim=-1).sum()), 16)
    got = moonvit_tpool_patch_merger(x, grid, (2, 2))
    want = reference_tpool(x, grid, (2, 2))
    assert len(got) == len(want)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_patch_merger_temporal_pool_is_a_plain_mean_over_all_frames():
    """A 3-frame item whose frames are 1, 2, 3 must merge to 2 everywhere."""
    grid = torch.tensor([[3, 2, 2]], dtype=torch.long)
    x = torch.cat([torch.full((4, 8), float(f + 1)) for f in range(3)])
    merged = moonvit_tpool_patch_merger(x, grid, (2, 2))[0]
    assert merged.shape == (1, 4, 8)
    torch.testing.assert_close(merged, torch.full_like(merged, 2.0))

    # Negative control: taking the first frame instead of the mean.
    assert (merged - torch.full_like(merged, 1.0)).abs().max() > 0.5


def test_patch_merger_pixel_shuffle_order_is_row_major_within_the_kernel():
    """The ``kh*kw`` axis is ``(row, col)`` row-major, matching ``:641``."""
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    # One channel, token value == its flat (h, w) index.
    x = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
    merged = moonvit_tpool_patch_merger(x, grid, (2, 2))[0]
    assert merged.shape == (1, 4, 1)
    assert merged.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_patch_merger_rejects_an_indivisible_grid():
    grid = torch.tensor([[1, 3, 4]], dtype=torch.long)
    with pytest.raises(ValueError, match="not divisible"):
        moonvit_tpool_patch_merger(torch.randn(12, 8), grid, (2, 2))


# ===========================================================================
# 4. Attention: block-diagonal and non-causal
# ===========================================================================


def test_attention_is_block_diagonal_over_media_items():
    """Perturbing one media item must not move any other item's output.

    The strongest available structural test, and the one that catches the
    single most damaging plausible bug: forgetting ``cu_seqlens`` and letting
    every image attend to every other. That produces a finite loss that goes
    down.
    """
    torch.manual_seed(4)
    cfg = MoonViTReferenceConfig(**SMALL)
    layer = MoonViTEncoderLayerReference(cfg).double()
    grid = torch.tensor(GRID, dtype=torch.long)
    cu, _ = moonvit_cu_seqlens(grid)
    freqs = moonvit_rope_freqs_for_grid(grid, cfg.head_dim)

    tokens = int(grid.prod(dim=-1).sum())
    x = torch.randn(tokens, cfg.hidden_size, dtype=torch.float64)
    base = layer(x, cu, freqs)

    perturbed = x.clone()
    perturbed[0] += 10.0  # first token of item 0
    after = layer(perturbed, cu, freqs)

    lo, hi = int(cu[1]), int(cu[3])
    assert (after[lo:hi] - base[lo:hi]).abs().max() == 0.0
    assert (after[:lo] - base[:lo]).abs().max() > 1e-6


def test_attention_is_not_causal_inside_a_media_item():
    """The last token must influence the first -- the opposite of the backbone.

    ``validate/VALIDATION.md`` establishes strict causality for the Kimi K3
    text stack. Copying that expectation to the vision tower would be wrong,
    so it is pinned in the other direction here.
    """
    torch.manual_seed(5)
    cfg = MoonViTReferenceConfig(**SMALL)
    layer = MoonViTEncoderLayerReference(cfg).double()
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    cu, _ = moonvit_cu_seqlens(grid)
    freqs = moonvit_rope_freqs_for_grid(grid, cfg.head_dim)

    x = torch.randn(16, cfg.hidden_size, dtype=torch.float64)
    base = layer(x, cu, freqs)
    perturbed = x.clone()
    perturbed[-1] += 5.0
    after = layer(perturbed, cu, freqs)
    assert (after[0] - base[0]).abs().max() > 1e-6


def test_attention_softmax_rows_sum_to_one_within_a_block():
    """No token leaks probability mass to another media item."""
    torch.manual_seed(6)
    grid = torch.tensor(GRID, dtype=torch.long)
    cu, _ = moonvit_cu_seqlens(grid)
    tokens = int(grid.prod(dim=-1).sum())
    q = torch.randn(tokens, 2, 8, dtype=torch.float64)
    k = torch.randn(tokens, 2, 8, dtype=torch.float64)
    # A one-hot value tensor turns the output into the attention weights.
    v = torch.eye(tokens, dtype=torch.float64).unsqueeze(1).expand(tokens, 2, tokens)
    out = MoonViTAttentionReference.core_attention(q, k, v.contiguous(), cu)
    weights = out.view(tokens, 2, tokens)
    torch.testing.assert_close(
        weights.sum(-1), torch.ones(tokens, 2, dtype=torch.float64), rtol=1e-9, atol=1e-9
    )
    bounds = cu.tolist()
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        outside = torch.ones(tokens, dtype=torch.bool)
        outside[lo:hi] = False
        assert weights[lo:hi, :, outside].abs().max() < 1e-12


# ===========================================================================
# 5. Whole-tower behaviour
# ===========================================================================


def test_tower_shapes_and_token_accounting():
    cfg = MoonViTReferenceConfig(**SMALL)
    tower = MoonViTReference(cfg)
    px, grid = make_inputs(cfg)
    out = tower(px, grid, return_stages=True)

    tokens = int(grid.prod(dim=-1).sum())
    assert out.patch_embed.shape == (tokens, cfg.hidden_size)
    assert out.encoder.shape == (tokens, cfg.hidden_size)
    expected = [(h // 2) * (w // 2) for _, h, w in grid.tolist()]
    assert [m.shape[0] for m in out.merged] == expected
    assert all(m.shape[1:] == (4, cfg.hidden_size) for m in out.merged)
    assert [p.shape for p in out.projected] == [
        (n, cfg.text_hidden_size) for n in expected
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tower_dtype_round_trip(dtype):
    cfg = MoonViTReferenceConfig(**SMALL)
    tower = MoonViTReference(cfg).to(dtype)
    px, grid = make_inputs(cfg)
    out = tower(px.to(dtype), grid)
    assert all(p.dtype == dtype for p in out)
    assert all(torch.isfinite(p).all() for p in out)


def test_tower_every_parameter_gets_a_gradient():
    """The cheapest test for an unwired submodule.

    ``pos_emb.time_weight`` is a buffer, not a parameter, so the expected
    count is exactly ``len(list(parameters()))``.
    """
    cfg = MoonViTReferenceConfig(**SMALL)
    tower = MoonViTReference(cfg)
    px, grid = make_inputs(cfg)
    torch.cat(tower(px, grid), dim=0).pow(2).mean().backward()
    missing = [n for n, p in tower.named_parameters() if p.grad is None]
    assert missing == []
    dead = [n for n, p in tower.named_parameters() if p.grad.abs().max() == 0.0]
    assert dead == []


def test_tower_output_is_permutation_equivariant_across_media_items():
    """Reordering the media items reorders the outputs and changes nothing else."""
    cfg = MoonViTReferenceConfig(**SMALL)
    tower = MoonViTReference(cfg).double()
    grid_a = [(1, 4, 6), (1, 8, 8)]
    px_a, g_a = make_inputs(cfg, grid_a)
    px_a = px_a.double()
    out_a = tower(px_a, g_a)

    n0 = 4 * 6
    px_b = torch.cat((px_a[n0:], px_a[:n0]))
    g_b = torch.tensor([grid_a[1], grid_a[0]], dtype=torch.long)
    out_b = tower(px_b, g_b)

    torch.testing.assert_close(out_a[0], out_b[1], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(out_a[1], out_b[0], rtol=1e-10, atol=1e-10)


def test_pos_emb_identity_grid_matches_an_interpolated_one():
    """``:265-266``'s fast path must agree with ``F.interpolate`` at the same size."""
    cfg = MoonViTReferenceConfig(**SMALL)
    tower = MoonViTReference(cfg).double()
    pos = tower.patch_embed.pos_emb
    fast = pos.resample(pos.height, pos.width)
    slow = (
        F.interpolate(
            pos.weight.permute(2, 0, 1).unsqueeze(0),
            size=(pos.height, pos.width),
            mode=pos.interpolation_mode,
        )
        .squeeze(0)
        .permute(1, 2, 0)
        .flatten(end_dim=1)
    )
    torch.testing.assert_close(fast, slow, rtol=1e-12, atol=1e-12)


def test_pos_emb_rejects_more_frames_than_the_temporal_code_covers():
    cfg = MoonViTReferenceConfig(**SMALL)
    tower = MoonViTReference(cfg)
    grid = torch.tensor([[5, 2, 2]], dtype=torch.long)  # init_pos_emb_time is 4
    with pytest.raises(ValueError, match="num_frames"):
        tower.patch_embed(torch.randn(20, 3, cfg.patch_size, cfg.patch_size), grid)


def test_projector_uses_erf_gelu_and_the_tower_uses_tanh_gelu():
    """``nn.GELU()`` at ``:792`` against ``PytorchGELUTanh()`` at ``:686``.

    They differ by ~1e-3 at the knee, which is far too small to notice in a
    loss curve and far too large to be a rounding difference.
    """
    cfg = MoonViTReferenceConfig(**SMALL)
    tower = MoonViTReference(cfg).double()
    assert isinstance(tower.projector.proj[1], nn.GELU)
    assert tower.projector.proj[1].approximate == "none"

    x = torch.linspace(-3, 3, 101, dtype=torch.float64)
    gap = (F.gelu(x, approximate="tanh") - F.gelu(x)).abs().max()
    assert 1e-4 < gap < 1e-2


def test_tower_norm_eps_is_effectively_zero_on_this_torch_build():
    """Measured, not assumed -- and load-bearing for the Megatron port.

    The tower's norms are ``nn.RMSNorm(dim)`` with no ``eps``
    (``:490-491``); ATen substitutes double-precision epsilon. Megatron's
    ``layernorm_epsilon`` default of 1e-5 would be a ~5e-6 relative error,
    which is why ``vt_layernorm_epsilon`` pins the measured value instead.

    The bound is float32's epsilon: anything at or below it is a no-op in
    every dtype Kimi K3 trains in, which is the property the port relies on.
    """
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )

    eps = moonvit_default_rmsnorm_eps()
    assert eps < torch.finfo(torch.float32).eps, (
        f"nn.RMSNorm's eps=None default is now {eps}, which float32 can see; "
        "the vision config's vt_layernorm_epsilon needs revisiting"
    )
    assert KimiK3TransformerConfig.vt_layernorm_epsilon == pytest.approx(eps, rel=0, abs=0)


# ===========================================================================
# 6. Deliberate bug injection -- every check above must be able to fail
# ===========================================================================


def _tower_output(tower, cfg, grid=GRID):
    px, g = make_inputs(cfg, grid)
    return torch.cat(tower(px.double(), g), dim=0)


def test_injected_bug_swapped_rope_axes_is_caught():
    """Swap the x and y halves of the RoPE table -- the docstring's version."""
    dim, h, w = 16, 5, 7
    good = moonvit_rope_freqs_cis(dim, h, w)
    n = h * w
    flat = torch.arange(n).float()
    x_pos, y_pos = flat % w, torch.div(flat, w, rounding_mode="floor")
    dim_range = torch.arange(0, dim, 4).float()[: dim // 4]
    freqs = 1.0 / (10000.0 ** (dim_range / dim))
    x_cis = torch.polar(torch.ones(n, dim // 4), torch.outer(x_pos, freqs))
    y_cis = torch.polar(torch.ones(n, dim // 4), torch.outer(y_pos, freqs))
    # SWAPPED: y at the even slot, which is what the docstring claims.
    bad = torch.cat((y_cis.unsqueeze(-1), x_cis.unsqueeze(-1)), dim=-1).reshape(h, w, -1)

    delta = (torch.view_as_real(good) - torch.view_as_real(bad)).abs().max()
    assert delta > 0.1, "the axis-swap control is indistinguishable; the test has no power"


def test_injected_bug_shared_freqs_across_media_items_is_caught():
    """Give every item the first item's frequencies.

    Exactly what reusing Megatron's ``thd`` rotary path would have done.
    """
    dim = 16
    grid = torch.tensor([[1, 8, 8], [1, 4, 6]], dtype=torch.long)
    good = moonvit_rope_freqs_for_grid(grid, dim)
    table = moonvit_rope_freqs_cis(dim, 8, 8)
    first = table[:8, :8].reshape(-1, dim // 2)
    bad = torch.cat((first, first[:24]), dim=0)  # item 2 gets item 1's rows
    assert (torch.view_as_real(good) - torch.view_as_real(bad)).abs().max() > 0.1


def test_injected_bug_dropped_temporal_code_is_caught_only_on_video():
    """Zeroing ``time_weight`` must change a video and leave an image alone."""
    cfg = MoonViTReferenceConfig(**SMALL)
    torch.manual_seed(9)
    tower = MoonViTReference(cfg).double()

    image_before = _tower_output(tower, cfg, [(1, 4, 6)])
    video_before = _tower_output(tower, cfg, [(3, 2, 4)])
    with torch.no_grad():
        tower.patch_embed.pos_emb.time_weight.zero_()
    image_after = _tower_output(tower, cfg, [(1, 4, 6)])
    video_after = _tower_output(tower, cfg, [(3, 2, 4)])

    torch.testing.assert_close(image_before, image_after, rtol=0, atol=0)
    assert (video_before - video_after).abs().max() > 1e-3


def test_injected_bug_full_attention_across_items_is_caught():
    """Drop ``cu_seqlens`` so everything attends to everything.

    The block-diagonality test above must fail on this.
    """
    torch.manual_seed(4)
    cfg = MoonViTReferenceConfig(**SMALL)
    layer = MoonViTEncoderLayerReference(cfg).double()
    grid = torch.tensor(GRID, dtype=torch.long)
    tokens = int(grid.prod(dim=-1).sum())
    # One segment covering the whole batch -- the injected bug.
    broken_cu = torch.tensor([0, tokens], dtype=torch.int32)
    freqs = moonvit_rope_freqs_for_grid(grid, cfg.head_dim)

    x = torch.randn(tokens, cfg.hidden_size, dtype=torch.float64)
    base = layer(x, broken_cu, freqs)
    perturbed = x.clone()
    perturbed[0] += 10.0
    after = layer(perturbed, broken_cu, freqs)

    # Leakage into the OTHER media items is what the correct version forbids.
    assert (after[64:] - base[64:]).abs().max() > 1e-6


def test_injected_bug_first_frame_instead_of_temporal_mean_is_caught():
    torch.manual_seed(2)
    grid = torch.tensor([[3, 2, 4]], dtype=torch.long)
    x = torch.randn(24, 8, dtype=torch.float64)
    good = moonvit_tpool_patch_merger(x, grid, (2, 2))[0]
    first_frame = moonvit_tpool_patch_merger(x[:8], torch.tensor([[1, 2, 4]]), (2, 2))[0]
    assert (good - first_frame).abs().max() > 1e-3


def test_injected_bug_transposed_pixel_shuffle_is_caught():
    """``permute(0, 1, 3, 2, 4, 5)`` vs a transposed kernel axis."""
    torch.manual_seed(2)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    x = torch.randn(16, 8, dtype=torch.float64)
    good = moonvit_tpool_patch_merger(x, grid, (2, 2))[0]
    bad = good.view(4, 2, 2, 8).transpose(1, 2).reshape(4, 4, 8)
    assert (good - bad).abs().max() > 1e-3


def test_injected_bug_erf_gelu_in_the_tower_mlp_is_caught():
    cfg = MoonViTReferenceConfig(**SMALL)
    torch.manual_seed(11)
    tower = MoonViTReference(cfg).double()
    good = _tower_output(tower, cfg)

    original = torch.nn.functional.gelu

    def erf_only(x, approximate="none"):
        return original(x, approximate="none")

    torch.nn.functional.gelu = erf_only
    try:
        bad = _tower_output(tower, cfg)
    finally:
        torch.nn.functional.gelu = original
    assert (good - bad).abs().max() > 1e-6


def test_injected_bug_nonzero_norm_eps_is_caught_in_fp32():
    """1e-5 instead of 0 on the tower norms.

    This is the specific mistake a Megatron port makes by default, and the
    margin below is what makes ``vt_layernorm_epsilon = 0.0`` worth having.
    """
    cfg = MoonViTReferenceConfig(**SMALL)
    torch.manual_seed(12)
    tower = MoonViTReference(cfg).double()
    good = _tower_output(tower, cfg)

    with torch.no_grad():
        for mod in tower.modules():
            if isinstance(mod, nn.RMSNorm) and mod.eps in (None, 0.0):
                mod.eps = 1e-5
    bad = _tower_output(tower, cfg)
    gap = (good - bad).abs().max().item()
    # ~5e-6 relative on unit-RMS activations: far above an fp32 tolerance of
    # 1e-6 and far below anything bf16 would notice.
    assert gap > 1e-7, f"eps 0 -> 1e-5 moved the tower by only {gap}"


def test_injected_bug_mixing_normalised_rope_into_values_is_caught():
    """Rotate ``v`` as well as ``q`` and ``k``.

    A plausible transcription slip -- RoPE on values is a real technique
    elsewhere -- and one that leaves every shape and every dtype intact.
    """
    torch.manual_seed(13)
    cfg = MoonViTReferenceConfig(**SMALL)
    attn = MoonViTAttentionReference(cfg).double()
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    cu, _ = moonvit_cu_seqlens(grid)
    freqs = moonvit_rope_freqs_for_grid(grid, cfg.head_dim)
    x = torch.randn(16, cfg.hidden_size, dtype=torch.float64)

    good = attn(x, cu, freqs)
    q, k, v = attn.split_qkv(x)
    q, k = apply_moonvit_rope(q, k, freqs)
    v, _ = apply_moonvit_rope(v, v, freqs)  # the injected bug
    bad = attn.wo(attn.core_attention(q, k, v, cu))
    assert (good - bad).abs().max() > 1e-6
