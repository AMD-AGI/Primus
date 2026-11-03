###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling multi-token prediction (HF ``mtp_config``; SGLang ``InklingMTPLayer``) on Megatron's
MTP path (:class:`MultiTokenPredictionBlock`, ``process_mtp_loss``, the MTP loss scaler and logger).

Depth ``d`` (``d = 0 .. mtp_num_layers - 1``) at position ``i``::

    e = embed_norm(main_embed_norm(embed(tok[i + d + 1])))   # shared embedding + trunk embed_norm
    x = input_proj(cat(hidden_norm(h_prev[i]), e))           # hidden FIRST (HF input_proj [h, 2h])
    h = InklingDecoderLayer(x)   # dense SwiGLU MLP; sliding (MTP window) if d in mtp local ids
    logits_d = (h / logits_mup_width_multiplier) @ unembed   # the shared output layer

``h_prev`` is the trunk's last-layer output *before* its final norm for ``d = 0`` (SGLang's
``hidden_states_before_norm``), else the previous depth's raw ``h`` (``chain_hidden_post_norm``
is False: no norm between depths). Megatron's ``MultiTokenPredictionLayer`` supplies the rolled
ids (``roll_tensor``), the chaining, the per-depth loss (labels rolled ``d + 1`` times) and the
checkpoint layout (``mtp.layers.{d}.{enorm,hnorm,eh_proj,transformer_layer}.*``). Inkling only
changes the concat order (hidden first), the norms (``enorm`` = the MTP ``embed_norm`` on the
already trunk-normed embedding, ``hnorm`` = ``hidden_norm``), the block and the final norm (none).
"""

import torch
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

from primus.backends.megatron.core.transformer.inkling.inkling_layers import (
    InklingAttention,
    InklingRMSNorm,
)


class InklingMTPLayer(MultiTokenPredictionLayer):
    """One Inkling MTP depth (see the module docstring).

    Full recompute checkpoints the whole depth (projection + block) for ``uniform`` and ``block``
    with any ``recompute_num_layers`` >= 1: a depth is a single layer. (Upstream asserts
    ``recompute_num_layers == 1`` and skips ``block``.)"""

    def _concat_embeddings(self, hidden_states, decoder_input):
        e = make_viewless_tensor(inp=self.enorm(decoder_input), requires_grad=True, keep_graph=True)
        h = make_viewless_tensor(inp=self.hnorm(hidden_states), requires_grad=True, keep_graph=True)
        # HF / SGLang: input_proj(cat(hidden_norm(h), embed_norm(e))), hidden first, so the HF
        # input_proj weight is eh_proj.weight as is (Megatron's own order is embedding first).
        x, _ = self.eh_proj(torch.cat((h, e), -1))
        return gather_from_tensor_model_parallel_region(x)

    def _checkpointed_forward(self, forward_func, *args, **kwargs):
        names = list(kwargs)

        def run(*values):
            return forward_func(*args, **dict(zip(names, values)))

        return tensor_parallel.checkpoint(run, self.config.distribute_saved_activations, *kwargs.values())


def _inkling_mtp_model_layer_spec():
    from primus.backends.megatron.core.models.inkling.inkling_block import (
        InklingDecoderLayer,
    )

    # The submodules only carry the causal mask type that MultiTokenPredictionLayer validates;
    # InklingDecoderLayer builds its own modules (InklingAttention: causal, with its window).
    return ModuleSpec(
        module=InklingDecoderLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(module=InklingAttention, params={"attn_mask_type": AttnMaskType.causal})
        ),
    )


def get_inkling_mtp_block_spec(config):
    """The ``MultiTokenPredictionBlock`` spec: one :class:`InklingMTPLayer` per depth, or None
    without MTP."""
    if not config.mtp_num_layers:
        return None
    layer = ModuleSpec(
        module=InklingMTPLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=InklingRMSNorm,
            hnorm=InklingRMSNorm,
            eh_proj=tensor_parallel.ColumnParallelLinear,
            mtp_model_layer=_inkling_mtp_model_layer_spec(),
            layer_norm=IdentityOp,  # chain_hidden_post_norm=False: no norm on a depth output
        ),
    )
    return ModuleSpec(
        module=MultiTokenPredictionBlock,
        submodules=MultiTokenPredictionBlockSubmodules(layer_specs=[layer] * config.mtp_num_layers),
    )
