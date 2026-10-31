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
    def __init__(self, config, submodules, *args, post_norm=None, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        self.post_feedforward_layernorm = (
            post_norm(config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon)
            if post_norm is not None
            else None
        )

    def forward(self, hidden_states, *args, **kwargs):
        out, bias = super().forward(hidden_states, *args, **kwargs)
        if self.post_feedforward_layernorm is not None:
            if bias is not None:
                out = out + bias
                bias = None
            out = self.post_feedforward_layernorm(out)
        return out, bias

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
                for k, v in sub_sd.items():
                    if k in (f"{sub_prefix}weight", f"{sub_prefix}bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets, singleton_local_shards)
            sd.update(sub_sd)
        return sd
