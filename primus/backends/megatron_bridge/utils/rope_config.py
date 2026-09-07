###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Transformers RoPE config compatibility for Megatron-Bridge."""

from __future__ import annotations

from typing import Any

_ORIG_PRETRAINED_CONFIG_GETATTRIBUTE = None


def resolve_rope_theta(config: Any) -> float:
    """Return RoPE base (theta) from a HuggingFace config.

    transformers 4.x exposes ``rope_theta`` on the config directly.
    transformers 5.x moved the value into ``default_theta`` / ``rope_parameters``.
    """
    default_theta = getattr(config, "default_theta", None)
    if default_theta is not None:
        return default_theta

    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        if isinstance(rope_parameters, dict):
            theta = rope_parameters.get("rope_theta")
            if theta is not None:
                return theta
        else:
            theta = getattr(rope_parameters, "rope_theta", None)
            if theta is not None:
                return theta

    raise AttributeError(
        f"{type(config).__name__} has no rope_theta, default_theta, " f"or rope_parameters['rope_theta']"
    )


def install_transformers_rope_theta_shim() -> bool:
    """Shim ``PretrainedConfig.rope_theta`` for transformers 5.x configs.

    Megatron-Bridge reads ``hf_config.rope_theta`` when mapping HF configs to
    Megatron providers. transformers 5.x removed the flat field, so bridge
    conversion fails before training starts. This patch restores the old
    attribute transparently for any config subclass.
    """
    global _ORIG_PRETRAINED_CONFIG_GETATTRIBUTE

    from transformers.configuration_utils import PretrainedConfig

    if getattr(PretrainedConfig, "_primus_rope_theta_shim_installed", False):
        return False

    _ORIG_PRETRAINED_CONFIG_GETATTRIBUTE = PretrainedConfig.__getattribute__

    def _getattribute(self, key: str) -> Any:
        if key == "rope_theta":
            try:
                return _ORIG_PRETRAINED_CONFIG_GETATTRIBUTE(self, key)
            except AttributeError:
                return resolve_rope_theta(self)
        return _ORIG_PRETRAINED_CONFIG_GETATTRIBUTE(self, key)

    PretrainedConfig.__getattribute__ = _getattribute
    PretrainedConfig._primus_rope_theta_shim_installed = True
    return True
