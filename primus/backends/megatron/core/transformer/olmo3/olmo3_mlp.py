###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 MLP: SwiGLU MLP followed by ``post_feedforward_layernorm`` (post-norm).

The sharded state dict is upstream ``MLP.sharded_state_dict`` (fc1 / fc2 with the SwiGLU factory
and the TP / DP groups); only the post-norm's entries are redone with this MLP's TP group, so the
replicated norm weight gets one replica per TP rank (upstream passes no TP group for it).
"""

from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    sharded_state_dict_default,
)

__all__ = ["Olmo3MLP"]


class Olmo3MLP(MLP):
    def __init__(
        self, config, submodules, *args, post_norm=None, pg_collection=None, is_mtp_layer=False, **kwargs
    ):
        if post_norm is None:
            raise ValueError("OLMo-3 MLP needs post_norm (its post_feedforward_layernorm)")
        # TransformerLayer (Megatron-LM core_v0.19) builds its mlp with pg_collection / is_mtp_layer,
        # as MLP.as_mlp_submodule takes them; MLP itself takes tp_group.
        del is_mtp_layer
        if pg_collection is not None and kwargs.get("tp_group") is None:
            kwargs["tp_group"] = pg_collection.tp
        super().__init__(config, submodules, *args, **kwargs)
        self.post_feedforward_layernorm = post_norm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )

    def forward(self, hidden_states, *args, **kwargs):
        out, bias = super().forward(hidden_states, *args, **kwargs)
        if bias is not None:
            out = out + bias
            bias = None
        return self.post_feedforward_layernorm(out), bias

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Upstream ``MLP.sharded_state_dict``, with the post-norm made TP-replicated on ``self.tp_group``."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        norm_prefix = f"{prefix}post_feedforward_layernorm."
        for key in [k for k in sd if k.startswith(norm_prefix)]:
            del sd[key]
        sd.update(
            sharded_state_dict_default(
                self.post_feedforward_layernorm,
                norm_prefix,
                sharded_offsets,
                metadata,
                tp_group=self.tp_group,
            )
        )
        return sd
