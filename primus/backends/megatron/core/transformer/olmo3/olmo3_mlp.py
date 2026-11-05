###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 MLP: SwiGLU MLP followed by ``post_feedforward_layernorm`` (post-norm)."""

from megatron.core.transformer.mlp import MLP, apply_swiglu_sharded_factory
from megatron.core.transformer.utils import sharded_state_dict_default

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
        """Upstream ``MLP.sharded_state_dict`` with the post-norm handled too.

        Upstream calls ``sharded_state_dict`` on every child, which a TE norm lacks. This
        override walks the children itself (no change to the module tree): fc1 / fc2 the
        upstream way (fc1 with the SwiGLU split factory), the post-norm through
        ``sharded_state_dict_default`` as a TP-replicated tensor.
        """
        singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)
        tp_group = getattr(self, "tp_group", None)
        sd = {}
        for name, module in self._modules.items():
            if module is None:
                continue
            sub_prefix = f"{prefix}{name}."
            if name == "post_feedforward_layernorm":
                sub_sd = sharded_state_dict_default(
                    module, sub_prefix, sharded_offsets, metadata, tp_group=tp_group
                )
            else:
                sub_sd = module.sharded_state_dict(sub_prefix, sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == "linear_fc1":
                for k in (f"{sub_prefix}weight", f"{sub_prefix}bias"):
                    if k in sub_sd:
                        sub_sd[k] = apply_swiglu_sharded_factory(
                            sub_sd[k], sharded_offsets, singleton_local_shards
                        )
            sd.update(sub_sd)
        return sd
