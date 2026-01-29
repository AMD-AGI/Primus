###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
LoRA (Low-Rank Adaptation) utilities for Megatron-LM SFT.

This module provides LoRA implementation for Megatron-LM models:
    - LoraConfig: Configuration dataclass for LoRA parameters
    - LoraLinear: LoRA wrapper for linear layers
    - apply_lora_to_model: Apply LoRA to a Megatron model
    - get_lora_state_dict: Extract LoRA parameters for saving
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from primus.modules.module_utils import log_rank_0


@dataclass
class LoraConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation).

    Attributes:
        enabled: Whether LoRA is enabled
        r: LoRA rank (dimension of low-rank matrices)
        alpha: LoRA scaling factor (alpha/r is the actual scaling)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module name patterns to apply LoRA
        modules_to_save: Modules to save in full (not just LoRA weights)
        fan_in_fan_out: Whether the layer stores weights as (fan_in, fan_out)
        bias: Bias configuration ('none', 'all', 'lora_only')
    """

    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "query",
            "key",
            "value",
            "dense",
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
        ]
    )
    modules_to_save: List[str] = field(default_factory=list)
    fan_in_fan_out: bool = False
    bias: str = "none"  # 'none', 'all', 'lora_only'

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoraConfig":
        """Create LoraConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "modules_to_save": self.modules_to_save,
            "fan_in_fan_out": self.fan_in_fan_out,
            "bias": self.bias,
        }


class LoraLayer(nn.Module):
    """
    Base class for LoRA layers.

    LoRA decomposes weight updates as: W' = W + (alpha/r) * BA
    where B is (out_features, r) and A is (r, in_features)
    """

    def __init__(
        self,
        r: int,
        alpha: int,
        dropout: float = 0.0,
        fan_in_fan_out: bool = False,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fan_in_fan_out = fan_in_fan_out
        self.merged = False


class LoraLinear(LoraLayer):
    """
    LoRA implementation for linear layers.

    Wraps a linear layer and adds low-rank adaptation matrices.
    During training, computes: output = original_output + scaling * dropout(x @ A.T) @ B.T
    """

    def __init__(
        self,
        original_layer: nn.Module,
        r: int,
        alpha: int,
        dropout: float = 0.0,
        fan_in_fan_out: bool = False,
    ):
        super().__init__(r=r, alpha=alpha, dropout=dropout, fan_in_fan_out=fan_in_fan_out)

        self.original_layer = original_layer

        # Get dimensions from original layer
        if hasattr(original_layer, "weight"):
            weight = original_layer.weight
            if fan_in_fan_out:
                self.in_features = weight.shape[0]
                self.out_features = weight.shape[1]
            else:
                self.out_features = weight.shape[0]
                self.in_features = weight.shape[1]
        else:
            # For Megatron parallel layers
            self.in_features = getattr(original_layer, "input_size", None)
            self.out_features = getattr(original_layer, "output_size", None)

            if self.in_features is None or self.out_features is None:
                raise ValueError(
                    f"Cannot determine dimensions for layer {type(original_layer).__name__}"
                )

        # Initialize LoRA matrices
        # A: (r, in_features), initialized with Kaiming uniform
        # B: (out_features, r), initialized with zeros
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original layer
        self._freeze_original_layer()

    def _freeze_original_layer(self):
        """Freeze the original layer parameters."""
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with LoRA.

        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments passed to original layer

        Returns:
            Output tensor with LoRA adaptation applied
        """
        # Original layer output
        result = self.original_layer(x, *args, **kwargs)

        # Handle tuple outputs (some Megatron layers return (output, bias))
        if isinstance(result, tuple):
            original_output, *rest = result
        else:
            original_output = result
            rest = []

        # LoRA adaptation: x @ A.T @ B.T * scaling
        # A: (r, in_features), B: (out_features, r)
        # x: (..., in_features)
        # lora_output: (..., out_features)
        lora_output = self.dropout(x)
        lora_output = F.linear(lora_output, self.lora_A)  # (..., r)
        lora_output = F.linear(lora_output, self.lora_B)  # (..., out_features)
        lora_output = lora_output * self.scaling

        # Combine outputs
        combined_output = original_output + lora_output

        if rest:
            return (combined_output, *rest)
        return combined_output

    def merge_weights(self):
        """Merge LoRA weights into original layer (for inference)."""
        if self.merged:
            return

        if hasattr(self.original_layer, "weight"):
            # Standard linear layer
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            if self.fan_in_fan_out:
                delta_w = delta_w.T
            self.original_layer.weight.data += delta_w
            self.merged = True

    def unmerge_weights(self):
        """Unmerge LoRA weights from original layer."""
        if not self.merged:
            return

        if hasattr(self.original_layer, "weight"):
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            if self.fan_in_fan_out:
                delta_w = delta_w.T
            self.original_layer.weight.data -= delta_w
            self.merged = False


def _find_target_modules(
    model: nn.Module,
    target_patterns: List[str],
) -> List[Tuple[nn.Module, str, nn.Module]]:
    """
    Find modules matching target patterns.

    Args:
        model: Model to search
        target_patterns: List of regex patterns for module names

    Returns:
        List of (parent_module, attr_name, target_module) tuples
    """
    targets = []

    for name, module in model.named_modules():
        # Check if module name matches any target pattern
        for pattern in target_patterns:
            if re.search(pattern, name):
                # Find parent module
                parts = name.rsplit(".", 1)
                if len(parts) == 1:
                    parent = model
                    attr_name = parts[0]
                else:
                    parent_name, attr_name = parts
                    parent = model.get_submodule(parent_name)

                # Only wrap linear-like layers
                if _is_linear_layer(module):
                    targets.append((parent, attr_name, module))
                    break

    return targets


def _is_linear_layer(module: nn.Module) -> bool:
    """Check if module is a linear layer that can be wrapped with LoRA."""
    # Standard PyTorch linear
    if isinstance(module, nn.Linear):
        return True

    # Megatron parallel linear layers
    module_type = type(module).__name__
    if module_type in [
        "ColumnParallelLinear",
        "RowParallelLinear",
        "Linear",
        "TEColumnParallelLinear",
        "TERowParallelLinear",
    ]:
        return True

    return False


def apply_lora_to_model(
    model: nn.Module,
    config: LoraConfig,
) -> Tuple[nn.Module, Set[str]]:
    """
    Apply LoRA to a model.

    Args:
        model: Model to apply LoRA to
        config: LoRA configuration

    Returns:
        Tuple of (modified model, set of wrapped module names)
    """
    if not config.enabled:
        log_rank_0("LoRA is disabled, returning original model")
        return model, set()

    log_rank_0(f"Applying LoRA with r={config.r}, alpha={config.alpha}")
    log_rank_0(f"Target modules: {config.target_modules}")

    # Find target modules
    targets = _find_target_modules(model, config.target_modules)

    wrapped_names = set()
    for parent, attr_name, target_module in targets:
        # Create LoRA wrapper
        lora_layer = LoraLinear(
            original_layer=target_module,
            r=config.r,
            alpha=config.alpha,
            dropout=config.dropout,
            fan_in_fan_out=config.fan_in_fan_out,
        )

        # Replace module
        setattr(parent, attr_name, lora_layer)

        # Track wrapped module name
        full_name = f"{type(parent).__name__}.{attr_name}"
        wrapped_names.add(full_name)
        log_rank_0(f"  Applied LoRA to: {full_name}")

    log_rank_0(f"LoRA applied to {len(wrapped_names)} modules")

    # Log trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_rank_0(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model, wrapped_names


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA parameters from model.

    Args:
        model: Model with LoRA layers

    Returns:
        Dictionary of LoRA parameters
    """
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data

    return lora_state_dict


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Load LoRA parameters into model.

    Args:
        model: Model with LoRA layers
        state_dict: Dictionary of LoRA parameters
        strict: Whether to require all keys to match

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    missing_keys = []
    unexpected_keys = list(state_dict.keys())

    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            lora_a_key = f"{name}.lora_A"
            lora_b_key = f"{name}.lora_B"

            if lora_a_key in state_dict:
                module.lora_A.data.copy_(state_dict[lora_a_key])
                unexpected_keys.remove(lora_a_key)
            elif strict:
                missing_keys.append(lora_a_key)

            if lora_b_key in state_dict:
                module.lora_B.data.copy_(state_dict[lora_b_key])
                unexpected_keys.remove(lora_b_key)
            elif strict:
                missing_keys.append(lora_b_key)

    return missing_keys, unexpected_keys


def merge_lora_weights(model: nn.Module) -> None:
    """Merge all LoRA weights into base model (for inference)."""
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module.merge_weights()


def unmerge_lora_weights(model: nn.Module) -> None:
    """Unmerge all LoRA weights from base model."""
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module.unmerge_weights()
