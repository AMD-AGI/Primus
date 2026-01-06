###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Model Override Patch
================================

This patch enables dynamic model parameter overrides by monkey-patching
``torchtitan.protocols.train_spec.get_train_spec()``.

Purpose:
--------
Allow Primus to override TorchTitan model configuration parameters (e.g.,
``n_layers``, ``dim``, ``n_heads``) at runtime without modifying TorchTitan's
train_spec registry.

Behavior:
---------
1. When enabled, intercepts calls to ``get_train_spec()``
2. Extracts model args for the specified flavor from the returned spec
3. Applies overrides from ``params.model_overrides`` (all keys must start with "model.")
4. Returns the modified spec with updated model_args

Configuration:
--------------
Enable via config:
    params:
      model_overrides:
        model.n_layers: 8
        model.dim: 4096

Or nested form (automatically flattened):
    params:
      model_overrides:
        model:
          n_layers: 8
          dim: 4096

Usage:
------
This patch is automatically applied during the "setup" phase when
``params.model_overrides`` is present in the configuration.
"""

import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from primus.core.patches import PatchContext, get_args, get_param, register_patch
from primus.modules.module_utils import log_rank_0


def _flatten_model_overrides(model_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested model_overrides dict to dot notation.

    Example:
        {"model": {"n_layers": 8}} → {"model.n_layers": 8}
    """
    flat_overrides = {}
    for k, v in model_overrides.items():
        if k == "model" and isinstance(v, dict):
            for subk, subv in v.items():
                flat_overrides[f"model.{subk}"] = subv
        else:
            flat_overrides[k] = v
    return flat_overrides


@register_patch(
    patch_id="torchtitan.model_override",
    backend="torchtitan",
    phase="before_train",
    description="Override TorchTitan model args dynamically via config",
    condition=lambda ctx: get_param(ctx, "model_overrides", None) is not None,
)
def patch_torchtitan_model_override(ctx: PatchContext) -> None:
    """
    Monkey patch torchtitan.train_spec.get_train_spec to override model args dynamically.
    All override keys MUST start with "model." (e.g., {"model.n_layers": 8}).
    """
    get_args(ctx)
    model_overrides = get_param(ctx, "model", {})

    if not model_overrides:
        log_rank_0("[PrimusPatch][ModelOverride] No model_overrides provided, skip patch.")
        return

    # Flatten nested form {"model": {"n_layers": 4}} → {"model.n_layers": 4}
    model_overrides = _flatten_model_overrides(model_overrides)

    # Get model name and flavor from config
    model_name = get_param(ctx, "model.name", None)
    flavor = get_param(ctx, "model.flavor", None)

    if not model_name:
        raise ValueError("[PrimusPatch][ModelOverride] model.name is required for model override patch")
    if not flavor:
        raise ValueError("[PrimusPatch][ModelOverride] model.flavor is required for model override patch")

    log_rank_0(
        f"[PrimusPatch][ModelOverride] model_overrides provided for '{model_name}' "
        f"(flavor={flavor}): {model_overrides}"
    )

    # Import dynamically to allow mocking in tests
    train_spec_module = sys.modules.get("torchtitan.protocols.train_spec")
    if train_spec_module is None:
        import torchtitan.protocols.train_spec as train_spec_module

    orig_get_train_spec = train_spec_module.get_train_spec

    def patched_get_train_spec(name: str):
        spec = orig_get_train_spec(name)
        if name != model_name:
            return spec  # only patch targeted model

        assert hasattr(
            spec, "model_args"
        ), f"[PrimusPatch][ModelOverride] train_spec for '{name}' missing model_args"
        model_args_root = spec.model_args
        assert isinstance(
            model_args_root, dict
        ), f"[PrimusPatch][ModelOverride] train_spec.model_args must be dict, got {type(model_args_root)}"

        if flavor not in model_args_root:
            raise KeyError(
                f"[PrimusPatch][ModelOverride] flavor '{flavor}' not found in model_args for '{name}'. "
                f"Available flavors: {list(model_args_root.keys())}"
            )

        target_args = model_args_root[flavor]
        assert is_dataclass(
            target_args
        ), f"[PrimusPatch][ModelOverride] Expected dataclass model_args, got {type(target_args)}"

        before = asdict(target_args)
        for k, v in model_overrides.items():
            field_name = k[len("model.") :]
            if not hasattr(target_args, field_name):
                raise AttributeError(
                    f"[PrimusPatch][ModelOverride] '{type(target_args).__name__}' has no field '{field_name}'"
                )
            setattr(target_args, field_name, v)

        log_rank_0(
            f"[PrimusPatch][ModelOverride] Patched dataclass model_args['{flavor}'] "
            f"for '{name}' with {model_overrides} (before={before})"
        )
        return spec

    # Apply the patch globally
    train_spec_module.get_train_spec = patched_get_train_spec
    log_rank_0(
        f"[PrimusPatch][ModelOverride] get_train_spec for '{model_name}' successfully monkey patched (flavor={flavor})."
    )
