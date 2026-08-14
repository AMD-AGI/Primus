###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The primary numerical acceptance gate for Kimi Delta Attention.

KDA differs from Gated DeltaNet in exactly one respect: its forget gate
is per-channel (``g: [B, T, H, K]``) rather than per-head scalar
(``g: [B, T, H]``). Collapse the per-channel gate to a channel-constant
value and the two must therefore be *the same operator*.

Megatron already ships a faithful eager Gated DeltaNet
(``megatron.core.ssm.gated_delta_net.torch_chunk_gated_delta_rule``,
itself derived from HF's Qwen3-Next implementation). Asserting that
:func:`eager_chunk_kda` reproduces it under that collapse retires most of
the project's numerical risk in one test: it validates the chunk
decomposition, the WY/UT transform, the diagonal convention of the
intra-chunk mask, the inter-chunk state carry, the ``q`` scaling and the
optional L2 norms all at once, against code written by someone else.

Both implementations use ``chunk_size=64`` and ``scale = K ** -0.5``.
The comparison runs in fp32 (both upcast internally), so the tolerance
here measures nothing but accumulated fp32 rounding over different — but
algebraically identical — orderings. It is stated as
``max|Δ| / max|reference|`` (see
:func:`kda_reference_impls.assert_close_scaled` for why an elementwise
``rtol`` is the wrong criterion here).

The optional ``q``/``k`` L2 normalisation is applied **outside** both
implementations. That keeps the test about the recurrence rather than
about a shared helper, and it side-steps a pre-existing incompatibility:
``torch_chunk_gated_delta_rule(use_qk_l2norm_in_kernel=True)`` calls
``fla.modules.l2norm.l2norm(x, dim=-1, eps=1e-6)``
(``gated_delta_net.py:605-607``), but ``fla`` 0.4.2's ``l2norm`` signature
is ``(x, eps=1e-6, output_dtype=None)`` — no ``dim`` — so that path raises
``TypeError`` in this image.
"""

from __future__ import annotations

import pytest
import torch

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
    eager_chunk_kda,
    eager_recurrent_kda,
    kda_l2norm,
)
from tests.unit_tests.megatron.transformer.kimi_k3.kda_reference_impls import (
    assert_close_scaled,
)

# Tight enough to be meaningful in fp32 (machine epsilon ~1.2e-7) after a
# few hundred sequential accumulation steps, loose enough to survive the
# two implementations summing in different orders.
COLLAPSE_TOL = 1e-5

# ``megatron`` is put on sys.path by tests/unit_tests/conftest.py.
torch_chunk_gated_delta_rule = pytest.importorskip(
    "megatron.core.ssm.gated_delta_net", reason="Megatron-LM not importable"
).torch_chunk_gated_delta_rule


def _inputs(batch, seq_len, num_heads, k_dim, v_dim, *, device, dtype=torch.float32, seed=0):
    """Random KDA inputs with a *per-head scalar* log-decay.

    ``g_head`` is what Gated DeltaNet consumes; ``g_head`` broadcast over
    the channel axis is what KDA consumes. ``g`` is forced negative (as
    the real gate guarantees) so the chunked form's decay factors stay
    ``<= 1``.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    q = rnd(batch, seq_len, num_heads, k_dim)
    k = rnd(batch, seq_len, num_heads, k_dim)
    v = rnd(batch, seq_len, num_heads, v_dim)
    # bounded exactly as the real gate is: g = -5 * sigmoid(.) in (-5, 0)
    g_head = -5.0 * torch.sigmoid(rnd(batch, seq_len, num_heads).float())
    beta = torch.sigmoid(rnd(batch, seq_len, num_heads).float())
    return q, k, v, g_head, beta


@pytest.mark.parametrize("use_qk_l2norm", [False, True], ids=["raw_qk", "l2norm_qk"])
@pytest.mark.parametrize(
    "batch,seq_len,num_heads,k_dim,v_dim",
    [
        (2, 64, 3, 32, 32),  # exactly one chunk
        (2, 256, 4, 64, 64),  # several full chunks, production head dim
        (1, 137, 2, 32, 48),  # ragged length + K != V, exercises padding
    ],
    ids=["one_chunk", "multi_chunk", "ragged_kv_mismatch"],
)
def test_collapsing_per_channel_gate_reproduces_gated_delta_rule(
    batch, seq_len, num_heads, k_dim, v_dim, use_qk_l2norm, kda_device
):
    """**THE** acceptance test: per-channel KDA with a channel-constant gate == GDN."""
    q, k, v, g_head, beta = _inputs(batch, seq_len, num_heads, k_dim, v_dim, device=kda_device, seed=1234)
    if use_qk_l2norm:
        q, k = kda_l2norm(q), kda_l2norm(k)
    g_channel = g_head.unsqueeze(-1).expand(batch, seq_len, num_heads, k_dim).contiguous()

    kda_out, kda_state = eager_chunk_kda(q, k, v, g_channel, beta, output_final_state=True, chunk_size=64)
    gdn_out, gdn_state = torch_chunk_gated_delta_rule(
        q, k, v, g=g_head, beta=beta, chunk_size=64, initial_state=None, output_final_state=True
    )

    assert kda_out.shape == gdn_out.shape == (batch, seq_len, num_heads, v_dim)
    tag = f"collapse {batch}x{seq_len}x{num_heads}x{k_dim}x{v_dim} l2norm={use_qk_l2norm}"
    assert_close_scaled(kda_out, gdn_out, COLLAPSE_TOL, f"{tag} out")
    assert_close_scaled(kda_state, gdn_state, COLLAPSE_TOL, f"{tag} state")


def test_collapse_also_holds_for_the_sequential_reference(kda_device):
    """The ``O(T)`` recurrence collapses to GDN too — pins the chunked form's oracle."""
    batch, seq_len, num_heads, k_dim, v_dim = 2, 128, 3, 32, 32
    q, k, v, g_head, beta = _inputs(batch, seq_len, num_heads, k_dim, v_dim, device=kda_device, seed=99)
    q, k = kda_l2norm(q), kda_l2norm(k)
    g_channel = g_head.unsqueeze(-1).expand(batch, seq_len, num_heads, k_dim).contiguous()

    seq_out, _ = eager_recurrent_kda(q, k, v, g_channel, beta)
    gdn_out, _ = torch_chunk_gated_delta_rule(q, k, v, g=g_head, beta=beta, chunk_size=64)
    assert_close_scaled(seq_out, gdn_out, COLLAPSE_TOL, "collapse sequential out")


def test_a_genuinely_per_channel_gate_does_not_match_gated_delta_rule(kda_device):
    """Guard against a vacuous pass.

    If :func:`eager_chunk_kda` silently ignored the channel axis of ``g``
    (e.g. by reducing it), the collapse test above would still pass. Feed
    a gate that varies across channels and require a *large* disagreement
    with the per-head formulation, whose ``g`` is the channel mean.
    """
    batch, seq_len, num_heads, k_dim = 2, 128, 3, 32
    q, k, v, _, beta = _inputs(batch, seq_len, num_heads, k_dim, k_dim, device=kda_device, seed=7)
    q, k = kda_l2norm(q), kda_l2norm(k)
    gen = torch.Generator(device="cpu").manual_seed(7)
    g_channel = -5.0 * torch.sigmoid(torch.randn(batch, seq_len, num_heads, k_dim, generator=gen)).to(
        kda_device
    )

    kda_out, _ = eager_chunk_kda(q, k, v, g_channel, beta)
    gdn_out, _ = torch_chunk_gated_delta_rule(q, k, v, g=g_channel.mean(-1), beta=beta, chunk_size=64)
    rel = (kda_out - gdn_out).abs().max().item() / gdn_out.abs().max().item()
    print(f"[per-channel is not per-head] rel max|dout|={rel:.3e}")
    assert rel > 1e-2, (
        "A per-channel gate must NOT reduce to the per-head formulation with the mean gate; "
        f"got only {rel:.3e} relative difference, which suggests the channel axis of `g` "
        "is being collapsed somewhere."
    )
