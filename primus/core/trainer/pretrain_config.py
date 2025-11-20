from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PretrainConfig:
    """
    Configuration object for the 'pre_trainer' module.

    This class represents the Primus-side configuration specifically for a
    Megatron-based pre-training workflow.

    YAML example:
        - name: pretrain_stage
          module: pre_trainer
          framework: megatron
          model: llama2_7B
          overrides:
            global_batch_size: 16
            micro_batch_size: 1
            lr: 1e-5

    Fields
    -------
    name:
        The name of this module instance within the workflow.
    module:
        Module type, fixed to "pre_trainer".
    framework:
        Training backend framework (e.g., "megatron").
    model:
        Predefined model preset name (e.g., "llama2_7B"). Primus resolves this
        to a corresponding model YAML or built-in preset.
    overrides:
        A flat dictionary of Megatron argument overrides, merged into the final
        Megatron arg namespace. These override both model defaults and
        Megatron-LM defaults.
    """

    name: str
    module: str = "pre_trainer"
    framework: str = "megatron"

    # Model preset name, e.g. "llama2_7B".
    model: Optional[str] = None

    # Flat dictionary of Megatron arg overrides:
    #   {"global_batch_size": 16, "lr": 1e-5, ...}
    arguments: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------
    # Methods for Primus runtime
    # -----------------------------------------------
    def to_megatron_args(self) -> Dict[str, Any]:
        """
        Convert this config into a flat Megatron argument dict.
        Only includes values provided via `arguments`.

        Model presets are resolved outside this class.
        """
        return dict(self.arguments)

    def describe(self) -> str:
        """Human-readable summary for logging/debugging."""
        return f"[PretrainConfig] model={self.model}, " f"arguments={len(self.arguments)} keys"

    # ----------------------------------------------------------------------
    # Construction from YAML SimpleNamespace
    # ----------------------------------------------------------------------
    @classmethod
    def from_simple_namespace(self, cls, ns) -> "PretrainConfig":
        """
        Construct a PretrainConfig object from a SimpleNamespace produced by
        the Primus YAML parser.

        Expected YAML structure:
            - name: pretrain_stage
              module: pre_trainer
              framework: megatron
              model: llama2_7B
              arguments:
                global_batch_size: 16
                micro_batch_size: 1

        This method:
            - Extracts core fields (name/module/framework/model)
            - Validates and loads 'arguments' as a dict
            - Preserves all additional YAML keys inside _extra_fields
              (for debugging or future extensions)
        """
        # --- Core fields (strict) --------------------------------------------------
        name = getattr(ns, "name", None)
        module = getattr(ns, "module", None)
        framework = getattr(ns, "framework", None)
        model = getattr(ns, "model", None)

        # --- overrides must be a dict --------------------------------------------
        arguments = getattr(ns, "arguments", {})
        if overrides is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            raise TypeError(f"[PretrainConfig] 'arguments' must be a dict, " f"got {type(arguments)!r}")

        # --- Create instance -------------------------------------------------------
        cfg = cls(
            name=name,
            module=module,
            framework=framework,
            model=model,
            overrides=overrides,
        )

        # --- Preserve all additional YAML fields ----------------------------------
        for key, value in ns.__dict__.items():
            if key in ["name", "module", "framework", "model", "overrides"]:
                continue
            cfg._extra_fields[key] = value
            setattr(cfg, key, value)

        return cfg
