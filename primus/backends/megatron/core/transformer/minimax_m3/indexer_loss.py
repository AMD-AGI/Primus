###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""What actually trains MiniMax-M3's indexer.

``topk`` is not differentiable, so the indexer's projections receive exactly
zero gradient from the main attention path: without this loss they stay at
their init for the whole run and MSA selects blocks at random. The fix is the
same one DeepSeek-V3.2 / DSA use -- distil the real attention distribution into
the indexer's scores.

This is the block-level analogue of ``compute_dsa_indexer_loss`` in Megatron's
``transformer/experimental_attention_variant/dsa.py``: M3 selects blocks per
GQA group rather than tokens globally, so the target is aggregated the same way
the selection is.

The gradient flows one way, into the indexer only: the target is detached
here, and ``MinimaxSparseAttention`` feeds the indexer a detached
``input_layernorm(x)``, so the KL can neither reshape the attention it imitates
nor leak into the layers below.
"""

from typing import Optional

import torch

# Key the loss is reported under. It shares Megatron's MoE aux-loss tracker, so
# it is reduced across pipeline stages and printed next to the MoE losses; the
# ``megatron.minimax_m3.indexer_loss_logging`` patch adds it to the key list
# ``training_log`` reduces.
MSA_INDEXER_LOSS_NAME = "msa_indexer_loss"


def _moe_aux_loss_scale() -> Optional[torch.Tensor]:
    """The per-microbatch scale the pipeline schedule installs for the MoE aux loss.

    ``forward_step_calc_loss`` sets it to ``grad_scale * cp_size / num_microbatches``
    (``grad_scale`` alone under ``calculate_per_token_loss``) whenever
    ``num_moe_experts`` is set, which M3 always has. That is exactly the scale
    the indexer loss needs, so it is read rather than duplicated. ``None`` when
    Megatron is not importable.
    """
    try:
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    except Exception:
        return None
    return getattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)


class MSAIndexerLossAutoScaler(torch.autograd.Function):
    """Attach the indexer loss to an activation without changing its value.

    Same device as ``DSAIndexerLossAutoScaler``: forward is the identity, and
    backward seeds the loss's gradient so it participates in the same backward
    pass as the main loss.

    Seeding a gradient of one would make the loss ``num_microbatches`` times too
    strong under gradient accumulation and ignore the grad scaler, so by default
    the seed follows :func:`_moe_aux_loss_scale`. :meth:`set_loss_scale`
    installs an explicit override.
    """

    # None means "follow the MoE aux-loss scale".
    main_loss_backward_scale: Optional[torch.Tensor] = None

    @staticmethod
    def current_loss_scale(reference: torch.Tensor) -> torch.Tensor:
        """The scale to seed, on ``reference``'s device and dtype."""
        scale = MSAIndexerLossAutoScaler.main_loss_backward_scale
        if scale is None:
            scale = _moe_aux_loss_scale()
        if scale is None:
            return torch.ones((), device=reference.device, dtype=reference.dtype)
        return scale.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indexer_loss,) = ctx.saved_tensors
        scale = MSAIndexerLossAutoScaler.current_loss_scale(indexer_loss)
        return grad_output, torch.ones_like(indexer_loss) * scale

    @staticmethod
    def set_loss_scale(scale: Optional[torch.Tensor]) -> None:
        """Override the seeded gradient; ``None`` goes back to following the MoE aux loss."""
        MSAIndexerLossAutoScaler.main_loss_backward_scale = scale


def record_indexer_loss(loss: torch.Tensor, layer_number: Optional[int], num_layers: int) -> None:
    """Write one layer's indexer loss into Megatron's MoE aux-loss tracker.

    ``training_log`` divides each tracked key by the number of MoE layers, which
    is the number of MSA layers as long as the two per-layer patterns match (they
    do in the released config), so the reported value is the mean over MSA
    layers. It is averaged over data-parallel ranks as well; every member of a
    DP group holds the same layers, so they agree on joining that collective.
    """
    # 1-based: 0 or None would write to the wrong slot.
    if not layer_number:
        return
    try:
        from megatron.core import parallel_state
        from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker
    except Exception:
        return

    avg_group = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        avg_group = parallel_state.get_data_parallel_group(with_context_parallel=False)

    save_to_aux_losses_tracker(
        MSA_INDEXER_LOSS_NAME,
        loss.detach().float(),
        layer_number,
        num_layers,
        avg_group=avg_group,
    )


def compute_indexer_loss(
    dense_scores: torch.Tensor,
    block_scores: torch.Tensor,
    block_size: int,
    loss_coeff: float,
) -> torch.Tensor:
    """KL(real attention over blocks || indexer's block distribution).

    Args:
        dense_scores: ``[b, n_q, sq, sk]`` causal attention logits, before the
            block mask -- the indexer is taught what dense attention *would*
            attend to, which is the whole point of the distillation.
        block_scores: ``[b, n_index, sq, n_blocks]`` fp32 indexer block scores,
            as returned by the indexer (``-inf`` on unreachable blocks,
            ``+inf`` on the always-visible ones).
        block_size: keys per block.
        loss_coeff: scaling applied to the KL.

    Returns:
        Scalar loss.
    """
    b, n_q, sq, sk = dense_scores.shape
    n_index = block_scores.shape[1]
    n_blocks = block_scores.shape[-1]
    n_rep = n_q // n_index

    # Target: real attention probabilities, summed over each GQA group's heads
    # (the group is what shares one block selection), then over each block's
    # keys, then L1-normalised. Detached -- this trains the indexer, not the
    # main attention. A padding row the mask leaves with no key softmaxes to
    # NaN; zero it so it yields an all-zero target and no loss.
    probs = torch.softmax(dense_scores.float(), dim=-1).detach().nan_to_num(nan=0.0)
    probs = probs.view(b, n_index, n_rep, sq, sk).sum(dim=2)

    pad = n_blocks * block_size - sk
    if pad:
        probs = torch.nn.functional.pad(probs, (0, pad), value=0.0)
    target = probs.view(b, n_index, sq, n_blocks, block_size).sum(dim=-1)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    # Prediction: the indexer's own distribution over the same blocks. The
    # always-visible boost sets +inf, which would make softmax NaN, so clamp it
    # back to the largest finite score before normalising.
    finite = torch.isfinite(block_scores)
    finite_max = torch.where(finite, block_scores, torch.full_like(block_scores, -torch.inf)).amax(
        dim=-1, keepdim=True
    )
    # A row whose reachable blocks are all forced (every query in the first
    # block) has no finite score, so finite_max is -inf and the softmax would be
    # NaN. Its selection is fixed and carries no signal: it contributes 0.
    has_free = finite.any(dim=-1, keepdim=True)
    logits = torch.minimum(block_scores, finite_max).masked_fill(~has_free, 0.0)
    prediction = torch.softmax(logits, dim=-1)

    kl = target * (torch.log(target + 1e-10) - torch.log(prediction + 1e-10))
    kl = kl.masked_fill(~has_free, 0.0)
    return kl.sum(dim=-1).mean() * loss_coeff
