###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig

# Import Transformer Engine
try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear,
    )
    from megatron.core.post_training.modelopt.layers import Linear

    HAVE_TE = True
except ImportError:
    TEColumnParallelLinear, TELinear, Linear = None, None, None
    HAVE_TE = False

# Import Primus-Turbo
try:
    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboColumnParallelLinear,
        PrimusTurboLinear,
    )

    HAVE_TE = True
except ImportError:
    PrimusTurboColumnParallelLinear, PrimusTurboLinear = None, None
    HAVE_TE = False


class PrimusMLASelfAttention(MLASelfAttention):
    """MLA Self-attention layer class wrapper for Primus

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        if self.config.q_lora_rank is None:
            # Not projecting query
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_proj",
            )

        else:
            q_down_proj_kwargs = {}
            # NOTE: Add support for Primus-Turbo
            if submodules.linear_q_down_proj in [TELinear, PrimusTurboLinear]:
                q_down_proj_kwargs["parallel_mode"] = "duplicated"
            elif submodules.linear_q_down_proj in [
                Linear,
                TEColumnParallelLinear,
                ColumnParallelLinear,
                PrimusTurboColumnParallelLinear,
            ]:
                q_down_proj_kwargs["gather_output"] = False
            else:
                raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

            self.linear_q_down_proj = build_module(
                submodules.linear_q_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_down_proj",
                skip_weight_param_allocation=False,
                **q_down_proj_kwargs,
            )

            self.linear_q_up_proj = build_module(
                submodules.linear_q_up_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_up_proj",
            )

        kv_down_proj_kwargs = {}
        # NOTE: Add support for Primus-Turbo
        if submodules.linear_kv_down_proj in [TELinear, PrimusTurboLinear]:
            kv_down_proj_kwargs["parallel_mode"] = "duplicated"
        elif submodules.linear_kv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
            PrimusTurboColumnParallelLinear,
        ]:
            kv_down_proj_kwargs["gather_output"] = False
        else:
            raise ValueError(f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}")

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_down_proj",
            skip_weight_param_allocation=False,
            **kv_down_proj_kwargs,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_up_proj",
        )

        if self.config.q_lora_rank is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
