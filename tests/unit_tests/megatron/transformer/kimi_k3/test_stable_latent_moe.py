###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the Kimi K3 Stable Latent MoE FFN (WP5).

What each group of tests pins down, and why:

``routed_expert_hidden_size=None`` equivalence
    :class:`StableLatentMoE` subclasses upstream :class:`MoELayer`, so with
    the latent width unset it must be *bit*-identical to a stock
    ``MoELayer``, not merely close. This is what proves the latent path is
    a clean opt-in.

HF parity
    Output matches a line-by-line transcription of
    ``KimiSparseMoeBlock.forward`` (``modeling_kimi_linear.py:815-838``),
    including the norm placement. The transcription is also evaluated with
    the norm moved *after* the up-projection and that variant is asserted
    to disagree, so the test cannot pass by accident on a symmetric case.

Router width
    ``[num_experts, hidden_size]``, never ``[num_experts, latent]``: HF
    calls the gate at ``:818``, before the down-projection at ``:822``.

Shared-expert bypass
    Zeroing the down-projection must leave the output exactly equal to the
    shared-expert branch. If the shared experts read the post-projection
    hidden state their input would be zero and the output would collapse to
    zero too, so this discriminates.

Training path
    The HF release is inference-only (``assert not self.training`` at
    ``:721``, ``NotImplementedError`` at ``:827``, ``@torch.no_grad()`` at
    ``:840``). Ours trains: gradients must reach the router, both latent
    projections, the norm, the routed experts and the shared experts.

The activation is stock SwiGLU throughout. Kimi K3's ``situ`` is WP4's, and
wiring it into these experts (via ``config.activation_func``, which is how
Megatron takes activations anyway) is a WP6 integration concern.
"""

from __future__ import annotations

import math
import os
import socket
from copy import deepcopy
from typing import Optional

import pytest
import torch  # must precede any transformer_engine import
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "Stable Latent MoE runs on Megatron's MoELayer, whose router allocates CUDA "
        "buffers in __init__ (router.py:172-189) and whose latent projections are "
        "TELinear (moe_layer.py:200)."
    ),
)

HIDDEN = 128
LATENT = 64
MOE_FFN = 96
NUM_EXPERTS = 4
TOPK = 2
SEQ = 8
BATCH = 2


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


@pytest.fixture()
def moe_parallel_state():
    """TP=PP=EP=CP=1 parallel state, one fresh state per test.

    Same shape as ``tests/unit_tests/backends/megatron/conftest.py``'s
    ``init_parallel_state``; duplicated here rather than shared because the
    Kimi K3 ``conftest.py`` belongs to the KDA work package and the KDA
    tests deliberately need no distributed init.
    """
    from megatron.core import parallel_state as ps

    if not torch.distributed.is_initialized():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        try:
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=1,
                rank=0,
            )
        except Exception as exc:  # pragma: no cover - environment guard
            pytest.skip(f"could not initialize torch.distributed: {exc}")

    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )

    from megatron.core.tensor_parallel import random as tp_random

    try:
        tp_random.initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    except (ImportError, AssertionError):  # pragma: no cover - environment guard
        tp_random.initialize_rng_tracker(use_cudagraphable_rng=True, force_reset=True)
    tp_random.model_parallel_cuda_manual_seed(42)

    yield

    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()


def make_config(
    *,
    latent: Optional[int] = LATENT,
    use_norm: bool = True,
    grouped: bool = False,
    dtype: torch.dtype = torch.float32,
    shared_expert_overlap: bool = False,
    aux_loss: bool = True,
):
    """A single-layer Kimi K3 config sized for CPU-readable numerics.

    Uses the real :class:`KimiK3TransformerConfig` (WP1) so the fields the
    MoE reads — ``routed_expert_hidden_size`` and ``latent_moe_use_norm`` —
    are the ones production sets.
    """
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )

    return KimiK3TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        kv_channels=16,
        ffn_hidden_size=MOE_FFN,
        moe_ffn_hidden_size=MOE_FFN,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        # KimiMoEGate: sigmoid scores + noaux_tc selection bias
        # (modeling_kimi_linear.py:711-712, 723).
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=1.0,
        # num_expert_group == topk_group == 1 in the release: group-limited
        # routing is inert (:724-744), so both stay None.
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="seq_aux_loss" if aux_loss else "none",
        moe_aux_loss_coeff=1e-3 if aux_loss else 0.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=grouped,
        moe_permute_fusion=False,
        moe_shared_expert_intermediate_size=MOE_FFN,
        moe_shared_expert_overlap=shared_expert_overlap,
        routed_expert_hidden_size=latent,
        latent_moe_use_norm=use_norm,
        gated_linear_unit=True,
        # Stock SwiGLU. Production comes here from ``swiglu: true`` in the YAML;
        # constructing the dataclass directly would otherwise inherit
        # ``activation_func = F.gelu`` (transformer_config.py:188). Kimi K3's
        # ``situ`` is WP4's and gets wired through this same field by WP6.
        activation_func=F.silu,
        bias_activation_fusion=False,
        add_bias_linear=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        params_dtype=dtype,
        bf16=dtype is torch.bfloat16,
        sequence_parallel=False,
    )


def build_k3_moe(config, *, layer_number: int = 1):
    """Build a :class:`StableLatentMoE` through the production spec factory."""
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.transformer.kimi_k3.moe import (
        build_stable_latent_moe_spec,
    )

    spec = build_stable_latent_moe_spec(config=config)
    moe = build_module(spec, config=config)
    moe.set_layer_number(layer_number)
    return moe.cuda()


def build_stock_moe(config, *, layer_number: int = 1):
    """Build an upstream :class:`MoELayer` with the same expert wiring."""
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
    from megatron.core.transformer.spec_utils import ModuleSpec

    from primus.backends.megatron.core.models.kimi_k3.build_context import (
        resolve_k3_provider,
    )

    provider = resolve_k3_provider(config)
    experts_module, experts_submodules = provider.k3_grouped_mlp_modules(
        moe_use_grouped_gemm=bool(config.moe_grouped_gemm),
        moe_use_legacy_grouped_gemm=False,
    )
    submodules = MoESubmodules(
        experts=(
            ModuleSpec(module=experts_module)
            if experts_submodules is None
            else ModuleSpec(module=experts_module, submodules=experts_submodules)
        ),
        shared_experts=ModuleSpec(
            module=SharedExpertMLP,
            submodules=MLPSubmodules(
                linear_fc1=provider.column_parallel_linear(),
                linear_fc2=provider.row_parallel_linear(),
                activation_func=provider.k3_mlp_activation_func(),
            ),
        ),
    )
    moe = MoELayer(config=config, submodules=submodules)
    moe.set_layer_number(layer_number)
    return moe.cuda()


def randomize_(module, *, seed: int = 0, scale: float = 0.05):
    """Fill every parameter with reproducible noise.

    Megatron's default init leaves the router weight and the expert weights
    at very different scales; a uniform small-noise fill keeps the reference
    comparison well-conditioned in fp32 without changing what is under test.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, param in sorted(module.named_parameters()):
            noise = torch.randn(param.shape, generator=generator, dtype=torch.float32)
            if "norm" in name:
                # RMSNorm gamma: centre on 1 so the norm is non-trivial but
                # does not dominate.
                noise = 1.0 + 0.3 * noise
            else:
                noise = scale * noise
            param.copy_(noise.to(device=param.device, dtype=param.dtype))


def make_input(config, *, seed: int = 1234, requires_grad: bool = False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(SEQ, BATCH, HIDDEN, generator=generator, dtype=torch.float32)
    x = x.to(device="cuda", dtype=config.params_dtype)
    x.requires_grad_(requires_grad)
    return x


# ---------------------------------------------------------------------------
# reference transcription of KimiSparseMoeBlock (modeling_kimi_linear.py)
# ---------------------------------------------------------------------------


def kimi_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``KimiRMSNorm.forward`` (``modeling_kimi_linear.py:232-236``)."""
    dtype = hidden_states.dtype
    x = hidden_states.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def kimi_gate(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """``KimiMoEGate.forward`` (``modeling_kimi_linear.py:703-759``).

    The group-limited branch (``:724-744``) is omitted: the release has
    ``num_expert_group == topk_group == 1``, which makes its guard false.
    """
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = F.linear(flat.float(), weight.float(), None)  # :707-710
    scores = logits.sigmoid()  # :712
    scores_for_choice = scores + expert_bias.float().unsqueeze(0)  # :723
    _, topk_idx = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)  # :747-749
    topk_weight = scores.gather(1, topk_idx)  # :750  from the UN-shifted scores
    if topk > 1 and renormalize:  # :753-755
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_idx, topk_weight * routed_scaling_factor  # :757


def kimi_block_sparse_mlp(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, act
) -> torch.Tensor:
    """``KimiBlockSparseMLP.forward`` (``:262-269``).

    Megatron stores the gate and up projections fused as
    ``linear_fc1.weight = cat([w1, w3])`` and chunks them back apart in
    ``mlp.py:313``, so ``w_gate`` / ``w_up`` here are the two halves.
    ``act`` comes from ``config.activation_func``, which is the field Kimi
    K3's ``situ`` will occupy.
    """
    return F.linear(act(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


def kimi_moe_infer(
    latent_x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_weights,
    act,
    *,
    per_expert_norm=None,
) -> torch.Tensor:
    """``KimiSparseMoeBlock.moe_infer`` (``:840-874``), written densely.

    The HF version is an argsort/gather permutation whose only purpose is
    batching; the weighted sum it computes is what matters here.

    Args:
        per_expert_norm: negative control. When given, the norm is applied to
            each expert's output *before* the weighted sum instead of to the
            combined result -- the "not per-expert" mistake the module must
            not make.
    """
    out = torch.zeros_like(latent_x)
    for slot in range(topk_idx.shape[1]):
        idx = topk_idx[:, slot]
        weight = topk_weight[:, slot].unsqueeze(-1).to(latent_x.dtype)
        for expert_id in range(len(expert_weights)):
            mask = idx == expert_id
            if not bool(mask.any()):
                continue
            rows = mask.nonzero(as_tuple=True)[0]
            expert_out = kimi_block_sparse_mlp(latent_x[rows], *expert_weights[expert_id], act)
            if per_expert_norm is not None:
                expert_out = per_expert_norm(expert_out)
            out[rows] = out[rows] + weight[rows] * expert_out
    return out


def read_expert_weights(moe):
    """``[(w_gate, w_up, w_down)]`` per local expert of a ``SequentialMLP``."""
    weights = []
    for expert in moe.experts.local_experts:
        fc1 = expert.linear_fc1.weight
        w_gate, w_up = torch.chunk(fc1, 2, dim=0)
        weights.append((w_gate, w_up, expert.linear_fc2.weight))
    return weights


def kimi_sparse_moe_reference(moe, x: torch.Tensor, *, norm_placement: str = "combined") -> torch.Tensor:
    """``KimiSparseMoeBlock.forward`` (``:815-838``), transcribed.

    Args:
        norm_placement: ``"combined"`` is the real thing — the norm on the
            already-combined, top-k-weighted latent sum, before the
            up-projection (``:829-832``). ``"per_expert"`` and ``"none"``
            are negative controls for the two ways to get the placement
            wrong; the module must disagree with both.
    """
    assert norm_placement in ("combined", "per_expert", "none")
    config = moe.config
    act = config.activation_func
    identity = x  # :816
    orig_shape = x.shape  # :817

    topk_idx, topk_weight = kimi_gate(  # :818
        x,
        moe.router.weight,
        moe.router.expert_bias,
        topk=int(config.moe_router_topk),
        renormalize=True,
        routed_scaling_factor=float(config.moe_router_topk_scaling_factor or 1.0),
    )
    flat = x.reshape(-1, x.shape[-1])  # :819

    latent = getattr(moe, "latent_size", None)
    if latent is not None:
        flat = F.linear(flat, moe.fc1_latent_proj.weight)  # :822

    norm = None
    if moe.routed_expert_norm is not None:
        eps = float(config.layernorm_epsilon)
        weight = moe.routed_expert_norm.weight

        def norm(t):  # noqa: F811
            return kimi_rms_norm(t, weight, eps)

    y = kimi_moe_infer(  # :825
        flat,
        topk_idx,
        topk_weight.to(flat.dtype),
        read_expert_weights(moe),
        act,
        per_expert_norm=norm if norm_placement == "per_expert" else None,
    )

    if latent is not None:  # :829-832
        if norm is not None and norm_placement == "combined":
            y = norm(y)
        y = F.linear(y, moe.fc2_latent_proj.weight)

    y = y.view(*orig_shape)  # :834

    shared_fc1 = moe.shared_experts.linear_fc1.weight  # :836-837
    shared_gate, shared_up = torch.chunk(shared_fc1, 2, dim=0)
    y = y + kimi_block_sparse_mlp(identity, shared_gate, shared_up, moe.shared_experts.linear_fc2.weight, act)
    return y


# ---------------------------------------------------------------------------
# 1. routed_expert_hidden_size=None is exactly a stock MoELayer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("training", [False, True])
def test_no_latent_is_bit_identical_to_stock_moe_layer(moe_parallel_state, training):
    config = make_config(latent=None, use_norm=False)
    k3 = build_k3_moe(config)
    stock = build_stock_moe(deepcopy(config))

    randomize_(k3, seed=7)
    stock.load_state_dict(k3.state_dict())
    k3.train(training)
    stock.train(training)

    x = make_input(config)
    k3_out, k3_bias = k3(x)
    stock_out, stock_bias = stock(x)

    assert k3_bias is None and stock_bias is None
    assert torch.equal(k3_out, stock_out), (
        "StableLatentMoE must run the parent's code path untouched when "
        f"routed_expert_hidden_size is None; max abs diff "
        f"{(k3_out - stock_out).abs().max().item():.3e}"
    )


def test_no_latent_builds_no_latent_machinery(moe_parallel_state):
    config = make_config(latent=None, use_norm=True)
    moe = build_k3_moe(config)

    assert moe.latent_size is None
    # latent_moe_use_norm without a bottleneck is a no-op in HF too
    # (modeling_kimi_linear.py:829-831).
    assert moe.routed_expert_norm is None
    assert not hasattr(moe, "fc1_latent_proj")
    assert not hasattr(moe, "fc2_latent_proj")
    assert moe.config.moe_latent_size is None


# ---------------------------------------------------------------------------
# 2. HF parity, including the norm placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_norm", [True, False])
def test_matches_kimi_sparse_moe_block_forward(moe_parallel_state, use_norm):
    config = make_config(latent=LATENT, use_norm=use_norm, grouped=False)
    moe = build_k3_moe(config)
    randomize_(moe, seed=11)
    moe.eval()

    x = make_input(config)
    got, bias = moe(x)
    assert bias is None
    want = kimi_sparse_moe_reference(moe, x)

    max_diff = (got - want).abs().max().item()
    scale = want.abs().max().item()
    assert max_diff <= 2e-5 * max(scale, 1.0), (
        f"output disagrees with the KimiSparseMoeBlock.forward transcription: "
        f"max abs diff {max_diff:.3e} (output scale {scale:.3e})"
    )


def test_norm_is_on_the_combined_routed_output(moe_parallel_state):
    """Negative controls for the norm placement.

    The aggregated RMSNorm sits on the already-combined, top-k-weighted sum,
    inside the bottleneck, before the up-projection
    (``modeling_kimi_linear.py:829-832``). The two plausible wrong answers
    are "per-expert, before the sum" and "not there at all"; the module must
    match the right ordering and disagree with both of the others, otherwise
    the parity test above would pass on a degenerate case.
    """
    config = make_config(latent=LATENT, use_norm=True, grouped=False)
    moe = build_k3_moe(config)
    randomize_(moe, seed=13)
    moe.eval()

    x = make_input(config)
    got, _ = moe(x)
    right = kimi_sparse_moe_reference(moe, x, norm_placement="combined")
    tolerance = 2e-5 * max(right.abs().max().item(), 1.0)
    assert (got - right).abs().max().item() <= tolerance

    for wrong_placement in ("per_expert", "none"):
        wrong = kimi_sparse_moe_reference(moe, x, norm_placement=wrong_placement)
        wrong_diff = (got - wrong).abs().max().item()
        assert wrong_diff > 1e-3, (
            f"norm placement {wrong_placement!r} is indistinguishable from the "
            f"module (diff {wrong_diff:.3e}); the placement test is not discriminating."
        )


def test_latent_norm_matches_kimi_rmsnorm(moe_parallel_state):
    """The norm module itself is ``KimiRMSNorm`` (``:226-236``)."""
    config = make_config(latent=LATENT, use_norm=True)
    moe = build_k3_moe(config)
    randomize_(moe, seed=17)

    norm = moe.routed_expert_norm
    assert norm is not None
    assert tuple(norm.weight.shape) == (LATENT,), "the norm lives in the latent space"

    y = torch.randn(SEQ * BATCH, LATENT, device="cuda", dtype=config.params_dtype)
    got = norm(y)
    want = kimi_rms_norm(y, norm.weight, float(config.layernorm_epsilon))
    assert torch.allclose(
        got, want, atol=1e-6, rtol=1e-5
    ), f"max abs diff {(got - want).abs().max().item():.3e}"


# ---------------------------------------------------------------------------
# 3. router width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent", [None, LATENT])
def test_router_weight_is_full_width(moe_parallel_state, latent):
    """``self.gate(hidden_states)`` at ``:818`` precedes the down-proj at ``:822``."""
    config = make_config(latent=latent, use_norm=latent is not None)
    moe = build_k3_moe(config)

    assert tuple(moe.router.weight.shape) == (NUM_EXPERTS, HIDDEN)
    if latent is not None:
        assert HIDDEN != LATENT, "the fixture must not make the two widths equal"
        assert tuple(moe.router.weight.shape) != (NUM_EXPERTS, LATENT)
        # The down/up projections are per-layer and shared by every expert
        # (modeling_kimi_linear.py:803-809).
        assert tuple(moe.fc1_latent_proj.weight.shape) == (LATENT, HIDDEN)
        assert tuple(moe.fc2_latent_proj.weight.shape) == (HIDDEN, LATENT)


def test_router_scores_the_unprojected_hidden_state(moe_parallel_state):
    """The routing decision must not move when the down-projection changes."""
    config = make_config(latent=LATENT, use_norm=True)
    moe = build_k3_moe(config)
    randomize_(moe, seed=19)
    moe.eval()

    x = make_input(config)
    probs_before, map_before = moe.route(x)
    with torch.no_grad():
        moe.fc1_latent_proj.weight.mul_(-3.0)
    probs_after, map_after = moe.route(x)

    assert torch.equal(map_before, map_after)
    assert torch.equal(probs_before, probs_after)


# ---------------------------------------------------------------------------
# 4. shared experts bypass the latent space
# ---------------------------------------------------------------------------


def test_shared_experts_see_the_pre_down_projection_hidden(moe_parallel_state):
    """Zeroing the down-projection must leave only the shared-expert branch.

    HF passes ``identity`` — the *original* hidden state — to the shared
    experts at ``:837``, after the routed path has already been projected
    at ``:822``. With ``W_down = 0`` the routed contribution is
    ``up(norm(0)) == 0`` (no biases anywhere), so the output must equal
    ``shared_experts(x)`` computed from the full-width input. Had the
    shared experts been fed the projected hidden they would see zeros and
    contribute zero as well.
    """
    config = make_config(latent=LATENT, use_norm=True)
    moe = build_k3_moe(config)
    randomize_(moe, seed=23)
    moe.eval()

    x = make_input(config)
    with torch.no_grad():
        moe.fc1_latent_proj.weight.zero_()

    got, _ = moe(x)
    want = moe.shared_experts(x)

    assert want.abs().max().item() > 1e-3, "the shared branch must be non-trivial"
    assert torch.allclose(
        got, want, atol=1e-6, rtol=1e-5
    ), f"max abs diff {(got - want).abs().max().item():.3e}"


def test_shared_expert_widths_are_model_space(moe_parallel_state):
    """Shared experts stay at ``hidden_size`` and ``moe_intermediate_size``.

    ``mlp.py:210`` only substitutes the latent width when ``is_expert`` is
    true, and ``SharedExpertMLP`` is built with ``is_expert=False``. Their
    intermediate width is ``moe_intermediate_size * num_shared_experts``
    (``modeling_kimi_linear.py:798``) — not ``intermediate_size``.
    """
    config = make_config(latent=LATENT, use_norm=True)
    moe = build_k3_moe(config)

    fc1 = moe.shared_experts.linear_fc1.weight
    fc2 = moe.shared_experts.linear_fc2.weight
    assert tuple(fc1.shape) == (2 * MOE_FFN, HIDDEN)
    assert tuple(fc2.shape) == (HIDDEN, MOE_FFN)


# ---------------------------------------------------------------------------
# 5. expert parameter accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("latent", [None, LATENT])
def test_expert_param_count_per_expert(moe_parallel_state, grouped, latent):
    """``3 * latent * moe_intermediate_size`` per expert.

    Three matrices per expert (gate / up / down, ``w1`` / ``w3`` / ``w2`` at
    ``modeling_kimi_linear.py:249-251``); Megatron fuses gate+up into
    ``linear_fc1``, so the total is unchanged. The latent bottleneck halves
    this against model space, which is the whole point of the design.
    """
    config = make_config(latent=latent, use_norm=latent is not None, grouped=grouped)
    moe = build_k3_moe(config)

    width = latent if latent is not None else HIDDEN
    total = sum(p.numel() for p in moe.experts.parameters())
    assert total % moe.num_local_experts == 0
    per_expert = total // moe.num_local_experts
    assert per_expert == 3 * width * MOE_FFN

    if latent is not None:
        assert per_expert == 3 * HIDDEN * MOE_FFN * LATENT // HIDDEN
        assert per_expert * 2 == 3 * HIDDEN * MOE_FFN, "LATENT is half of HIDDEN in this fixture"


def test_production_expert_param_accounting():
    """The released shape: 33.0 M per expert instead of 66.1 M. Pure arithmetic."""
    hidden, latent, moe_intermediate = 7168, 3584, 3072
    per_expert_latent = 3 * latent * moe_intermediate
    per_expert_model = 3 * hidden * moe_intermediate
    assert math.isclose(per_expert_latent / 1e6, 33.03, abs_tol=0.01)
    assert math.isclose(per_expert_model / 1e6, 66.06, abs_tol=0.01)
    assert per_expert_latent * 2 == per_expert_model


# ---------------------------------------------------------------------------
# 6. training-mode forward + backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grouped", [False, True])
def test_training_forward_and_backward(moe_parallel_state, grouped):
    """Unlike the HF reference, the training path works and gradients flow.

    HF raises ``NotImplementedError`` at ``:827`` and asserts
    ``not self.training`` at ``:721``; ``moe_infer`` is ``@torch.no_grad()``
    at ``:840``. Everything below is ours.
    """
    config = make_config(latent=LATENT, use_norm=True, grouped=grouped)
    moe = build_k3_moe(config)
    randomize_(moe, seed=29)
    moe.train()

    x = make_input(config, requires_grad=True)
    out, bias = moe(x)
    assert bias is None
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    out.float().pow(2).mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()

    required = {
        "router.weight": moe.router.weight,
        "fc1_latent_proj.weight": moe.fc1_latent_proj.weight,
        "fc2_latent_proj.weight": moe.fc2_latent_proj.weight,
        "routed_expert_norm.weight": moe.routed_expert_norm.weight,
        "shared_experts.linear_fc1.weight": moe.shared_experts.linear_fc1.weight,
        "shared_experts.linear_fc2.weight": moe.shared_experts.linear_fc2.weight,
    }
    for name, param in required.items():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"
        assert param.grad.abs().max().item() > 0.0, f"{name} gradient is identically zero"

    expert_grads = [p.grad for p in moe.experts.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in expert_grads)
    assert any(g.abs().max().item() > 0.0 for g in expert_grads), "no expert received a gradient"

    missing = [n for n, p in moe.named_parameters() if p.grad is None]
    assert not missing, f"parameters left without a gradient: {missing}"


def test_aux_loss_reaches_the_router(moe_parallel_state):
    """``seq_aux_loss`` at 1e-3 is the phase-1 load-balancing recipe.

    ``DECISIONS.md`` §2: mirror DeepSeek-V4's configuration rather than
    implementing Kimi K3's Quantile Balancing, which has no reference
    implementation (deferred to WP10).
    """
    from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker

    config = make_config(latent=LATENT, use_norm=True, aux_loss=True)
    moe = build_k3_moe(config)
    randomize_(moe, seed=31)
    moe.train()

    clear_aux_losses_tracker()
    x = make_input(config, requires_grad=True)
    out, _ = moe(x)
    out.float().pow(2).mean().backward()

    assert moe.router.expert_bias is not None, "noaux_tc bias buffer must exist"
    assert moe.router.local_tokens_per_expert is not None
    assert (
        moe.router.local_tokens_per_expert.sum().item() == SEQ * BATCH * TOPK
    ), "the expert-bias update accumulator must see every routed token"
    assert moe.router.weight.grad is not None


# ---------------------------------------------------------------------------
# config / spec plumbing
# ---------------------------------------------------------------------------


def test_spec_factory_shape(moe_parallel_state):
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.spec_utils import ModuleSpec

    from primus.backends.megatron.core.transformer.kimi_k3.moe import (
        StableLatentMoE,
        build_stable_latent_moe_spec,
    )

    config = make_config(latent=LATENT, use_norm=True)
    spec = build_stable_latent_moe_spec(config=config)

    assert spec.module is StableLatentMoE
    # MoELayer takes `layer_number`, threaded via set_layer_number after the
    # build (transformer_layer.py:395-397) -- never through spec params.
    assert spec.params == {}
    assert spec.submodules.router is TopKRouter, "no bespoke router: TopKRouter is exact"
    assert isinstance(spec.submodules.latent_norm, ModuleSpec)
    assert not hasattr(spec.submodules, "token_dispatcher"), (
        "the dispatcher comes from config.moe_token_dispatcher_type "
        "(moe_layer.py:224-248), not from the spec tree"
    )

    no_norm_spec = build_stable_latent_moe_spec(config=make_config(latent=LATENT, use_norm=False))
    assert no_norm_spec.submodules.latent_norm is None


def test_resolve_latent_size_alias_agreement():
    from primus.backends.megatron.core.transformer.kimi_k3.moe import (
        resolve_latent_size,
    )

    config = make_config(latent=LATENT, use_norm=True)
    assert resolve_latent_size(config) == LATENT

    config.moe_latent_size = LATENT
    assert resolve_latent_size(config) == LATENT

    config.moe_latent_size = LATENT + 1
    with pytest.raises(ValueError, match="disagrees with moe_latent_size"):
        resolve_latent_size(config)

    none_config = make_config(latent=None, use_norm=False)
    assert resolve_latent_size(none_config) is None


def test_missing_latent_norm_spec_is_rejected(moe_parallel_state):
    """A spec that forgets the norm must fail loudly, not silently drop it."""
    from primus.backends.megatron.core.transformer.kimi_k3.moe import (
        StableLatentMoE,
        StableLatentMoESubmodules,
        build_stable_latent_moe_spec,
    )

    config = make_config(latent=LATENT, use_norm=True)
    good = build_stable_latent_moe_spec(config=config)
    broken = StableLatentMoESubmodules(
        experts=good.submodules.experts,
        shared_experts=good.submodules.shared_experts,
        latent_norm=None,
    )
    with pytest.raises(AssertionError, match="latent_norm"):
        StableLatentMoE(config=config, submodules=broken)


def test_shared_expert_overlap_is_disabled_for_the_latent_path(moe_parallel_state):
    """The family YAML enables the overlap, which upstream rejects here.

    ``moe_layer.py:360-362`` asserts ``not shared_expert_overlap`` whenever
    the latent projections are live, and it does so at *forward* time. A YAML
    that enables the overlap would therefore build a model fine and crash on
    step 1, so the module turns it off on its own config copy rather than
    mutating the caller's.

    Note the copy is now needed *only* for that flag: the latent width itself
    reaches upstream through
    ``KimiK3TransformerConfig.__post_init__``'s ``moe_latent_size`` mapping.
    """
    config = make_config(latent=LATENT, use_norm=True, shared_expert_overlap=True)
    moe = build_k3_moe(config)
    randomize_(moe, seed=37)
    moe.eval()

    assert moe.shared_expert_overlap is False
    assert moe.config.moe_shared_expert_overlap is False
    # The caller's config is not mutated: the copy exists only to clear the
    # overlap flag.
    assert config.moe_shared_expert_overlap is True
    assert moe.config is not config
    # The latent width is *not* something the copy has to carry any more.
    # KimiK3TransformerConfig.__post_init__ already mirrored
    # routed_expert_hidden_size onto moe_latent_size, so both configs agree and
    # the shallow copy is left doing exactly one job.
    assert config.moe_latent_size == LATENT
    assert moe.config.moe_latent_size == LATENT

    out, _ = moe(make_input(config))
    assert torch.isfinite(out).all()


def test_bf16_forward_and_backward(moe_parallel_state):
    """The production dtype path runs too, at bf16 tolerance."""
    config = make_config(latent=LATENT, use_norm=True, dtype=torch.bfloat16, grouped=True)
    moe = build_k3_moe(config)
    randomize_(moe, seed=41)
    moe.train()

    x = make_input(config, requires_grad=True)
    out, _ = moe(x)
    assert out.dtype is torch.bfloat16
    assert torch.isfinite(out).all()
    out.float().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
