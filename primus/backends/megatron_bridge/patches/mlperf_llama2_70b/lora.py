###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
NeMo-stable LoRA for MLPerf Llama2-70B.

Default ``use_te_fused_lora=False`` keeps unfused :class:`LoRALinear` adapters so
loss matches the MLPerf reference path.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from megatron.core import parallel_state
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import unwrap_model

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora_layers import (
    LinearAdapter,
    LoRALinear,
    LoRATopKRouter,
    TEFusedLoRALinear,
    TELinearAdapter,
    patch_linear_module,
)
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.utils import (
    ParallelLinearAdapter,
    get_adapter_attributes_from_linear,
    is_expert_linear,
    wildcard_match,
)

logger = logging.getLogger(__name__)

try:
    import bitsandbytes

    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False


def _te_fused_lora_allowed_for_module(
    full_name: str,
    module_name: Optional[str],
    include_modules: Optional[List[str]],
    exclude_modules: List[str],
) -> bool:
    """Return True if this FQN may use :class:`TEFusedLoRALinear`."""
    for pattern in exclude_modules:
        if module_name == pattern or wildcard_match(pattern, full_name):
            return False
    if include_modules is None:
        return True
    for pattern in include_modules:
        if module_name == pattern or wildcard_match(pattern, full_name):
            return True
    return False


@dataclass
class LoRA(PEFT, ModuleMatcher):
    """LoRA with explicit control over TE fused adapters (MLPerf defaults to unfused)."""

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: torch.dtype = None
    use_te_fused_lora: bool = False
    te_fused_lora_include_modules: Optional[List[str]] = None
    te_fused_lora_exclude_modules: List[str] = field(default_factory=list)

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        adapter_types = (LinearAdapter, LoRALinear, LoRATopKRouter, TELinearAdapter)
        if isinstance(module, adapter_types):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            (match, full_name) = ans
            if isinstance(module, nn.Linear) or (module.__class__ == te.Linear):
                if hasattr(module.weight.data, "_local_tensor") or (
                    HAVE_BNB
                    and getattr(module, "quant_state", None) is not None
                    and module.quant_state.__class__ == bitsandbytes.functional.QuantState
                ):
                    lora_cls = patch_linear_module
                elif module.__class__ == te.Linear:
                    lora_cls = TELinearAdapter
                else:
                    lora_cls = LinearAdapter

                return lora_cls(
                    module,
                    dim=self.dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    lora_A_init_method=self.lora_A_init_method,
                    lora_dtype=self.lora_dtype,
                )

            is_expert = is_expert_linear(full_name)
            attrs = get_adapter_attributes_from_linear(module, is_expert=is_expert)

            enable_op_fuser = (
                self.use_te_fused_lora
                and hasattr(module, "config")
                and getattr(module.config, "use_transformer_engine_op_fuser", False)
                and parallel_state.get_tensor_model_parallel_world_size() == 1
                and _te_fused_lora_allowed_for_module(
                    full_name,
                    name,
                    self.te_fused_lora_include_modules,
                    self.te_fused_lora_exclude_modules,
                )
            )

            logging.info(f"Adding lora to: {full_name}")
            adapter = ParallelLinearAdapter(
                attrs.in_features,
                attrs.out_features,
                self.dim,
                base_linear_name=full_name,
                activation="identity",
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=attrs.input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(module, "config", None),
                alpha=self.alpha,
                is_expert=is_expert,
                a2a_experimental=self.a2a_experimental,
                disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                base_linear_is_parallel=attrs.base_linear_is_parallel,
            )
            if isinstance(module, TopKRouter):
                return LoRATopKRouter(module, adapter)
            if enable_op_fuser:
                return TEFusedLoRALinear(module, adapter)
            return LoRALinear(module, adapter)
        return module


@dataclass
class VLMLoRA(LoRA):
    """VLM LoRA variant (re-exported for parity with upstream Megatron-Bridge)."""

    freeze_vision_model: bool = True
    freeze_vision_projection: bool = True
    freeze_language_model: bool = True

    def freeze_model(self, model: nn.Module, training: bool = True) -> None:
        modules_to_freeze = []
        model = unwrap_model(model)[0]
        if hasattr(model, "llava_model"):
            model = model.llava_model

        if self.freeze_vision_model and hasattr(model, "vision_model"):
            modules_to_freeze.append(model.vision_model)
        if self.freeze_vision_projection and hasattr(model, "vision_projection"):
            modules_to_freeze.append(model.vision_projection)
        if self.freeze_language_model and hasattr(model, "language_model"):
            modules_to_freeze.append(model.language_model)

        for module in modules_to_freeze:
            module.eval()
            for param in module.parameters():
                param.requires_grad = training is False
