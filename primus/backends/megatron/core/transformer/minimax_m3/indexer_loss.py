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
"""

from typing import Optional

import torch

from primus.core.utils.module_utils import log_rank_0


class MSAIndexerLossAutoScaler(torch.autograd.Function):
    """Attach the indexer loss to an activation without changing its value.

    Same device as ``DSAIndexerLossAutoScaler``: forward is the identity, and
    backward seeds the loss's gradient so it participates in the same backward
    pass as the main loss (and inherits its scale).
    """

    main_loss_backward_scale: Optional[torch.Tensor] = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indexer_loss,) = ctx.saved_tensors
        if MSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            MSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=indexer_loss.device)
        scale = MSAIndexerLossAutoScaler.main_loss_backward_scale
        return grad_output, torch.ones_like(indexer_loss) * scale

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        """Match the main loss's backward scale."""
        if MSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            MSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            MSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


class MSAIndexerLossTracker:
    """Per-layer indexer losses, for logging.

    Mirrors ``DSAIndexerLossLoggingHelper``: layers accumulate into one buffer
    during the forward pass and the trainer drains it when it logs.
    """

    values: Optional[torch.Tensor] = None

    @classmethod
    def record(cls, loss: torch.Tensor, layer_number: Optional[int], num_layers: int) -> None:
        if layer_number is None:
            return
        if cls.values is None or cls.values.shape[0] != num_layers:
            cls.values = torch.zeros(num_layers, device=loss.device)
        cls.values[layer_number - 1] += loss.detach()

    @classmethod
    def reduce(cls) -> Optional[float]:
        """Mean loss across layers, then clear. None when nothing was recorded."""
        if cls.values is None:
            return None
        mean = (cls.values.sum() / cls.values.shape[0]).item()
        cls.values = None
        return mean

    @classmethod
    def log(cls) -> None:
        mean = cls.reduce()
        if mean is not None:
            log_rank_0(f"[MSA] indexer distillation loss (mean over layers): {mean:.6f}")


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
    # main attention.
    probs = torch.softmax(dense_scores.float(), dim=-1).detach()
    probs = probs.view(b, n_index, n_rep, sq, sk).sum(dim=2)

    pad = n_blocks * block_size - sk
    if pad:
        probs = torch.nn.functional.pad(probs, (0, pad), value=0.0)
    target = probs.view(b, n_index, sq, n_blocks, block_size).sum(dim=-1)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    # Prediction: the indexer's own distribution over the same blocks. The
    # always-visible boost sets +inf, which would make softmax NaN, so clamp it
    # back to the largest finite score before normalising.
    finite_max = torch.where(
        torch.isfinite(block_scores), block_scores, torch.full_like(block_scores, -torch.inf)
    ).amax(dim=-1, keepdim=True)
    prediction = torch.softmax(torch.minimum(block_scores, finite_max), dim=-1)

    kl = target * (torch.log(target + 1e-10) - torch.log(prediction + 1e-10))
    return kl.sum(dim=-1).mean() * loss_coeff
