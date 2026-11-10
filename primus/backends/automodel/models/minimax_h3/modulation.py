###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MiniMax-H3's per-row AdaLN modulation, with its backward reduction accumulated in FP32.

WHY:
  Every MiniMax-H3 block projects the timestep embedding to six modulation tables (shift,
  scale and gate for attention and for the MLP) with one row per (timestep, modality), about
  six rows, and expands them to the packed sequence with ``table.index_select(0, rows)``. The
  output norm does the same with two tables. In BF16 training the backward of that gather,
  ``index_add``, sums the gradients of every sequence row into those few table rows in BF16.
  Measured on MI355X with the 4-block proxy at 7.3K rows, against an FP32 reference: the
  AdaLN projection gradients were 9.7% off and the output norm's 3.6%, against about 1% for
  every other parameter. The AdaLN projections hold 13B of the model's 33B parameters, and
  the error grows with the number of rows (38K at 5.2 s, 109K at 15 s).

WHAT:
  ``gather_rows`` is ``index_select`` in the forward. Its backward is a one-hot matrix
  product, ``one_hot(rows).T @ grad``: the GEMM accumulates in FP32 whatever its input dtype,
  so the reduction is as accurate as an FP32 one, deterministic (no atomics), and needs no
  FP32 copy of the gradient. The forward and every value it produces are unchanged.

  ``install`` replaces ``MiniMaxH3TransformerBlock.forward`` and
  ``MiniMaxH3AdaLayerNormOut.forward`` with copies of diffusers' code that call
  ``gather_rows`` where diffusers calls ``index_select``, and nothing else. It checks a
  fingerprint of diffusers' source first: if diffusers changed either function, it leaves
  them alone and says so, rather than run a copy of code that no longer matches.
"""
from __future__ import annotations

import hashlib
import inspect
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusMiniMaxH3]"
_MARKER = "_primus_fp32_modulation_grads"
# sha256 of inspect.getsource() of the two upstream forwards this module copies (diffusers 0.40.0).
UPSTREAM_FINGERPRINTS = {
    "MiniMaxH3TransformerBlock.forward": "ca6d9ca44871d0ed10fceaa4eecd184a7ac907cbd3e3b8fd9f761b91598a944d",
    "MiniMaxH3AdaLayerNormOut.forward": "dda2027623d7adece2e644cae537ccaf404f5460d3d2fda0f66ca018399c93b1",
}


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(rows)
        ctx.num_rows = table.shape[0]
        return table.index_select(0, rows)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (rows,) = ctx.saved_tensors
        one_hot = nn.functional.one_hot(rows, ctx.num_rows).to(grad.dtype)
        # [num_rows, seq] @ [seq, hidden]: the GEMM accumulates in FP32.
        return one_hot.transpose(0, 1) @ grad.reshape(rows.shape[0], -1), None


def gather_rows(table: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``table.index_select(0, rows)``, whose backward reduces into ``table`` with FP32 accumulation."""
    if not table.requires_grad or table.dtype == torch.float32:
        return table.index_select(0, rows)
    return _GatherRows.apply(table, rows)


def fingerprint(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def _block_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    adaln_indices: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # diffusers' MiniMaxH3TransformerBlock.forward, with gather_rows for index_select.
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

    residual = hidden_states
    norm_hidden_states = self.norm1(hidden_states)
    norm_hidden_states = norm_hidden_states * (1.0 + gather_rows(scale_msa, adaln_indices)) + gather_rows(
        shift_msa, adaln_indices
    )
    attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
    hidden_states = residual + gather_rows(gate_msa, adaln_indices) * attn_output

    residual = hidden_states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1.0 + gather_rows(scale_mlp, adaln_indices)) + gather_rows(
        shift_mlp, adaln_indices
    )
    ff_output = self.ff(norm_hidden_states)
    hidden_states = residual + gather_rows(gate_mlp, adaln_indices) * ff_output

    return hidden_states


def _norm_out_forward(
    self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor
) -> torch.Tensor:
    # diffusers' MiniMaxH3AdaLayerNormOut.forward, with gather_rows for index_select.
    from diffusers.models.modeling_utils import get_parameter_dtype

    shift, scale = self.linear(nn.functional.silu(temb).to(get_parameter_dtype(self.linear))).chunk(2, dim=-1)
    hidden_states = self.norm(hidden_states)
    return hidden_states * (1.0 + gather_rows(scale, timestep_indices)) + gather_rows(shift, timestep_indices)


def _targets():
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    return {
        "MiniMaxH3TransformerBlock.forward": (h3.MiniMaxH3TransformerBlock, _block_forward),
        "MiniMaxH3AdaLayerNormOut.forward": (h3.MiniMaxH3AdaLayerNormOut, _norm_out_forward),
    }


def install() -> bool:
    """Swap in the FP32-accumulating forwards. Idempotent; returns whether they are in place."""
    targets = _targets()
    if all(getattr(cls.forward, _MARKER, False) for cls, _ in targets.values()):
        return True
    changed = [
        name
        for name, (cls, _) in targets.items()
        if fingerprint(cls.forward) != UPSTREAM_FINGERPRINTS[name]
    ]
    if changed:
        logger.warning(
            "%s diffusers' %s changed since this patch was written; leaving them as they are, so the "
            "AdaLN modulation gradients accumulate in BF16 (about 10%% error at 7K rows, more at longer "
            "clips). Update primus/backends/automodel/models/minimax_h3/modulation.py.",
            _LOG_PREFIX,
            " and ".join(changed),
        )
        return False
    for cls, replacement in targets.values():
        setattr(replacement, _MARKER, True)
        cls.forward = replacement
    logger.info("%s AdaLN modulation gradients accumulate in FP32 (blocks and output norm).", _LOG_PREFIX)
    return True
