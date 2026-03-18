###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus extension of MaxText's Pydantic configuration.

``PrimusMaxTextConfig`` directly inherits from ``MaxTextConfig`` and adds a
small number of Primus-specific fields.  Any future upstream additions to
``MaxTextConfig`` (new mixin base classes, new fields, etc.) are automatically
inherited — **no manual synchronization required**.

For **legacy MaxText versions** (date-versioned, e.g. ``2025.07.24``) that
pre-date the Pydantic config system, ``PrimusMaxTextConfig`` is ``None`` and
all config handling falls back to the legacy ``_HyperParameters`` dict path
via the ``maxtext.pyconfig_compat`` patch.

Changes relative to ``MaxTextConfig`` (new-version only):

* **New fields** (MoE):        ``expert_balance``
* **New fields** (Debug):      ``jax_distributed_heartbeat_timeout_seconds``
* **New fields** (Turbo):      ``enable_primus_turbo``, ``use_turbo_grouped_gemm``
* **New fields** (WandB):      ``enable_wandb``, ``wandb_project``, ``wandb_exp_name``, ``wandb_save_dir``
"""

import os
from typing import Any

# ---------------------------------------------------------------------------
# Version detection: new MaxText (>= 0.1.1) has configs.types with Pydantic,
# old MaxText (2025.x.x) does not.
# ---------------------------------------------------------------------------
try:
    from MaxText.configs.types import MaxTextConfig  # noqa: F401

    MAXTEXT_HAS_PYDANTIC_CONFIG = True
except (ImportError, ModuleNotFoundError):
    MaxTextConfig = None  # type: ignore[assignment,misc]
    MAXTEXT_HAS_PYDANTIC_CONFIG = False


# ---------------------------------------------------------------------------
# Primus-specific field names and defaults (shared by both code paths)
# ---------------------------------------------------------------------------
PRIMUS_EXTRA_FIELDS: dict[str, Any] = {
    "expert_balance": False,
    "jax_distributed_heartbeat_timeout_seconds": 100,
    "enable_primus_turbo": False,
    "use_turbo_grouped_gemm": False,
    "enable_wandb": False,
    "wandb_project": None,
    "wandb_exp_name": None,
    "wandb_save_dir": None,
}


def apply_primus_validations(keys: dict[str, Any]) -> None:
    """
    Apply Primus-specific cross-field validations on a raw config dict.

    This is used by **both** the Pydantic path (inside
    ``PrimusMaxTextConfig.set_derived_and_validate_values``) and the legacy
    dict path (inside ``post_initialize``).
    """
    if (not keys.get("wandb_save_dir")) and keys.get("base_output_directory"):
        keys["wandb_save_dir"] = os.path.join(keys["base_output_directory"], "wandb")

    if not keys.get("wandb_project"):
        keys["wandb_project"] = os.getenv("WANDB_PROJECT", "Primus-MaxText-Pretrain")

    if (not keys.get("wandb_exp_name")) and keys.get("run_name"):
        keys["wandb_exp_name"] = keys["run_name"]

    if keys.get("enable_wandb") and "WANDB_API_KEY" not in os.environ:
        raise ValueError("WANDB_API_KEY is not set. Please set it or login wandb before proceeding")

    if not keys.get("enable_primus_turbo"):
        keys["use_turbo_grouped_gemm"] = False


# ---------------------------------------------------------------------------
# PrimusMaxTextConfig — only defined when the Pydantic config system exists
# ---------------------------------------------------------------------------
if MAXTEXT_HAS_PYDANTIC_CONFIG:
    from pydantic import ConfigDict, model_validator
    from pydantic.fields import Field

    class PrimusMaxTextConfig(MaxTextConfig):  # type: ignore[misc]
        """
        The main configuration object for Primus MaxText.

        Directly inherits from ``MaxTextConfig`` so that all upstream fields,
        validators, and future additions are automatically available.

        Only Primus-specific fields and validations are declared here.
        """

        # ---- MoE extension ----
        expert_balance: bool = Field(False, description="Whether to use expert balancing.")

        # ---- Debug extension ----
        jax_distributed_heartbeat_timeout_seconds: int = Field(
            100,
            description=(
                "How long before a missing heartbeat marks a task as dead. "
                "Increase for slow NFS checkpoint restores."
            ),
        )

        # ---- Primus Turbo ----
        enable_primus_turbo: bool = Field(False, description="Whether to enable Primus Turbo.")
        use_turbo_grouped_gemm: bool = Field(False, description="Whether to use turbo grouped gemm.")

        # ---- WandB integration ----
        enable_wandb: bool = Field(False, description="Whether to enable WandB.")
        wandb_project: None | str = Field(None, description="The name of the WandB project.")
        wandb_exp_name: None | str = Field(
            None, description="The name of the WandB experiment, derived from the run_name if not set."
        )
        wandb_save_dir: None | str = Field(None, description="The directory to save the WandB logs.")

        # Keep extra="forbid" to catch typos in YAML.
        model_config = ConfigDict(extra="forbid", protected_namespaces=())

        @model_validator(mode="before")
        @classmethod
        def load_model_specific_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
            """No-op: ``pyconfig.initialize`` handles model-specific config loading."""
            return values

        @model_validator(mode="after")
        def set_derived_and_validate_values(self) -> "PrimusMaxTextConfig":
            """
            Run MaxTextConfig's derived-value computation, then apply Primus-specific validations.
            """
            # Invoke upstream MaxTextConfig validator (sets run_name, paths, derived dims, etc.)
            super().set_derived_and_validate_values()

            # ---- Primus-specific validations (reuse shared logic) ----
            # model_dump() returns a mutable dict view; apply_primus_validations mutates it
            # but since we're inside a model_validator(mode="after"), we mutate self directly.
            _d: dict[str, Any] = {}
            for k in PRIMUS_EXTRA_FIELDS:
                _d[k] = getattr(self, k)
            _d["base_output_directory"] = self.base_output_directory
            _d["run_name"] = self.run_name
            apply_primus_validations(_d)
            for k in PRIMUS_EXTRA_FIELDS:
                setattr(self, k, _d[k])

            return self

else:
    # Legacy MaxText — PrimusMaxTextConfig not available.
    # Config building falls back to the dict-based path; Primus-specific
    # fields are injected by ``apply_primus_validations`` in ``post_initialize``.
    PrimusMaxTextConfig = None  # type: ignore[assignment,misc]
