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
from typing import Any

from primus.core.patches import PatchContext, get_param, register_patch
from primus.modules.module_utils import log_rank_0


def get_standard_model_config_fields() -> set:
    """
    Dynamically get standard model config fields from TorchTitan's JobConfig.

    These are configuration fields (name, flavor, tokenizer_path, etc.) that
    should NOT be treated as model architecture override parameters.

    Returns:
        Set of standard field names from JobConfig's model field definition
    """
    try:
        from dataclasses import fields as dataclass_fields

        from torchtitan.config.job_config import JobConfig

        # Get the type annotation of JobConfig.model field
        job_config_fields = {f.name: f for f in dataclass_fields(JobConfig)}

        if "model" not in job_config_fields:
            # Fallback: if JobConfig doesn't have a model field, use minimal set
            return {"name", "flavor"}

        model_field_type = job_config_fields["model"].type

        # If model field type is a dataclass, get its fields
        if hasattr(model_field_type, "__dataclass_fields__"):
            standard_fields = {f.name for f in dataclass_fields(model_field_type)}
            return standard_fields
        else:
            # Fallback: if model is not a dataclass, use minimal set
            return {"name", "flavor"}

    except Exception as e:
        # Fallback: if we can't inspect JobConfig, use minimal standard fields
        log_rank_0(f"[PrimusPatch][ModelOverride] Warning: Could not inspect JobConfig.model fields: {e}")
        return {"name", "flavor"}


def convert_dict_to_dataclass_recursive(data: Any, target_type: type) -> Any:
    """
    Recursively convert dict to dataclass, handling nested dataclass fields.

    Args:
        data: The data to convert (could be dict, list, or primitive)
        target_type: The target dataclass type (if applicable)

    Returns:
        Converted data with proper types
    """
    from dataclasses import fields as dataclass_fields

    # If data is not a dict or target is not a dataclass, return as-is
    if not isinstance(data, dict) or not is_dataclass(target_type):
        return data

    # Get all fields of the target dataclass
    field_types = {f.name: f.type for f in dataclass_fields(target_type)}

    # Convert each field recursively
    converted_data = {}
    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]

            # Handle nested dataclass
            if is_dataclass(field_type) and isinstance(value, dict):
                converted_data[key] = convert_dict_to_dataclass_recursive(value, field_type)
            # Handle list of dataclasses
            elif (
                isinstance(value, list)
                and hasattr(field_type, "__origin__")
                and field_type.__origin__ is list
            ):
                # Try to get the list element type
                if hasattr(field_type, "__args__") and len(field_type.__args__) > 0:
                    element_type = field_type.__args__[0]
                    if is_dataclass(element_type):
                        converted_data[key] = [
                            (
                                convert_dict_to_dataclass_recursive(item, element_type)
                                if isinstance(item, dict)
                                else item
                            )
                            for item in value
                        ]
                    else:
                        converted_data[key] = value
                else:
                    converted_data[key] = value
            else:
                converted_data[key] = value
        else:
            # Field not in dataclass definition, include it anyway
            converted_data[key] = value

    # Create the dataclass instance
    try:
        return target_type(**converted_data)
    except Exception as e:
        raise ValueError(f"Failed to convert dict to {target_type.__name__}: {e}. " f"Data: {converted_data}")


@register_patch(
    patch_id="torchtitan.model_override",
    backend="torchtitan",
    phase="before_train",
    description="Override TorchTitan model args dynamically via config",
)
def patch_torchtitan_model_override(ctx: PatchContext) -> None:
    """
    Monkey patch torchtitan.train_spec.get_train_spec to override model args dynamically.
    All override keys MUST start with "model." (e.g., {"model.n_layers": 8}).
    """
    # Get params.model which contains both model config (name, flavor) and overrides
    model_params_raw = get_param(ctx, "model", None)

    if not model_params_raw:
        log_rank_0("[PrimusPatch][ModelOverride] No model params provided, skip patch.")
        return

    # Convert SimpleNamespace to dict if needed
    if hasattr(model_params_raw, "__dict__") and not isinstance(model_params_raw, dict):
        from primus.core.utils.yaml_utils import nested_namespace_to_dict

        model_params = nested_namespace_to_dict(model_params_raw)
    else:
        model_params = model_params_raw

    # Extract model name and flavor from params.model
    model_name = model_params.get("name", None)
    flavor = model_params.get("flavor", None)

    if not model_name:
        raise ValueError(
            "[PrimusPatch][ModelOverride] params.model.name is required for model override patch"
        )
    if not flavor:
        raise ValueError(
            "[PrimusPatch][ModelOverride] params.model.flavor is required for model override patch"
        )

    # Get standard config fields dynamically from JobConfig.model definition
    standard_fields = get_standard_model_config_fields()
    log_rank_0(f"[PrimusPatch][ModelOverride] Standard model config fields: {standard_fields}")

    # Extract override parameters (exclude standard config fields)
    # Only parameters NOT in standard_fields are treated as overrides
    override_params = {k: v for k, v in model_params.items() if k not in standard_fields}

    if not override_params:
        log_rank_0(
            f"[PrimusPatch][ModelOverride] No model override params provided "
            f"(only standard config fields: {list(standard_fields)}), skip patch."
        )
        return

    # Add "model." prefix to all override keys
    model_overrides = {f"model.{k}": v for k, v in override_params.items()}

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
            old_value = getattr(target_args, field_name)

            # If the original value is a dataclass and new value is a dict,
            # recursively convert the dict to the dataclass type
            if is_dataclass(old_value) and isinstance(v, dict):
                dataclass_type = type(old_value)
                try:
                    # Recursively convert dict to dataclass (handles nested dataclasses)
                    v = convert_dict_to_dataclass_recursive(v, dataclass_type)
                    log_rank_0(
                        f"[PrimusPatch][ModelOverride] Converted dict to {dataclass_type.__name__} for field '{field_name}'"
                    )
                except Exception as e:
                    raise ValueError(
                        f"[PrimusPatch][ModelOverride] Failed to convert dict to {dataclass_type.__name__} "
                        f"for field '{field_name}': {e}"
                    )

            setattr(target_args, field_name, v)
            log_rank_0(f"[PrimusPatch][ModelOverride] Override {field_name}: {old_value} -> {v}")

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
