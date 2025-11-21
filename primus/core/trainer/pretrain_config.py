###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PretrainConfig:
    """
    Pure Primus-side configuration for a pre-training module.
    Backend-agnostic (NOT tied to Megatron / Titan).
    """

    # core fields
    name: str
    module: str = "pre_trainer"
    framework: str = "megatron"  # or 'torchtitan' / 'light-megatron' / etc
    model: Optional[str] = None

    # High-level arguments relevant to backend-specific builders
    # e.g. { global_batch_size: 16, lr: 1e-5 }
    arguments: Dict[str, Any] = field(default_factory=dict)

    # preserve unknown keys for forward-compatibility
    _extra_fields: Dict[str, Any] = field(default_factory=dict, init=False)

    # ----------------------------------------------------------------------
    # Debug / logging
    # ----------------------------------------------------------------------
    def describe(self) -> str:
        return f"[PretrainConfig] name={self.name}, model={self.model}, " f"{len(self.arguments)} arguments"

    # ----------------------------------------------------------------------
    # Build from SimpleNamespace
    # ----------------------------------------------------------------------
    @classmethod
    def from_simple_namespace(cls, ns) -> "PretrainConfig":

        if not hasattr(ns, "name"):
            raise ValueError("[PretrainConfig] module entry missing 'name' field")

        name = getattr(ns, "name", None)
        module = getattr(ns, "module", "pre_trainer")
        framework = getattr(ns, "framework", "megatron")
        model = getattr(ns, "model", None)

        raw_args = getattr(ns, "arguments", {})
        if raw_args is None:
            arguments = {}
        elif not isinstance(raw_args, dict):
            raise TypeError(f"[PretrainConfig] 'arguments' must be dict, got {type(raw_args)}")
        else:
            arguments = raw_args

        cfg = cls(
            name=name,
            module=module,
            framework=framework,
            model=model,
            arguments=arguments,
        )

        # preserve unknown YAML keys
        for key, val in ns.__dict__.items():
            if key in ["name", "module", "framework", "model", "arguments"]:
                continue

            cfg._extra_fields[key] = val
            setattr(cfg, key, val)

        return cfg
