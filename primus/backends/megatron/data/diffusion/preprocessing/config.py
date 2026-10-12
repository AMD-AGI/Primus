###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Configuration resolution for Megatron diffusion preprocessing commands."""

from collections.abc import Mapping
from typing import Any

from primus.core.config.merge_utils import deep_merge
from primus.core.utils import yaml_utils

# Defaults use the public YAML shape so configuration can be resolved through
# the same nested deep-merge path as the rest of Primus.
ENCODED_CONFIG_DEFAULTS = {
    "source": {
        "type": None,
        "hf_dataset": None,
        "hf_split": "train",
        "hf_data_files": None,
        "input_dir": None,
        "input_path": None,
    },
    "data_format": {
        "image_key": None,
        "caption_key": None,
        "image_keys": None,
        "caption_keys": None,
    },
    "output": {
        "output_dir": None,
        "shard_size": 1000,
        "max_samples": None,
        "compress": False,
    },
    "model": {
        "model_path": "black-forest-labs/FLUX.1-dev",
        "vae_path": None,
        "t5_path": None,
        "clip_path": None,
        "precision": "bf16",
        "device": "cuda",
        "batch_size": 8,
        "t5_max_length": 512,
        "vae_latent_mode": "presampled",
    },
    "image": {
        "image_size": 1024,
        "variable_size": False,
        "center_crop": True,
        "max_size": 1024,
    },
    "auth": {"hf_token_file": None},
}

PREPROCESSING_CONFIG_FIELDS = {
    "source": {
        "type": "source_type",
        "hf_dataset": "hf_dataset",
        "hf_split": "hf_split",
        "hf_data_files": "hf_data_files",
        "input_dir": "input_dir",
        "input_path": "input_path",
    },
    "data_format": {
        "image_key": "image_key",
        "caption_key": "caption_key",
        "image_keys": "image_keys",
        "caption_keys": "caption_keys",
    },
    "output": {
        "output_dir": "output_dir",
        "shard_size": "shard_size",
        "max_samples": "max_samples",
        "compress": "compress",
    },
    "model": {
        "model_path": "model_path",
        "vae_path": "vae_path",
        "t5_path": "t5_path",
        "clip_path": "clip_path",
        "precision": "precision",
        "device": "device",
        "batch_size": "batch_size",
        "t5_max_length": "t5_max_length",
        "vae_latent_mode": "vae_latent_mode",
    },
    "image": {
        "image_size": "image_size",
        "variable_size": "variable_size",
        "center_crop": "center_crop",
        "max_size": "max_size",
    },
    "auth": {"hf_token_file": "hf_token_file"},
}

CONFIG_PATH_BY_DEST = {
    dest: (section, key)
    for section, fields in PREPROCESSING_CONFIG_FIELDS.items()
    for key, dest in fields.items()
}


def flatten_preprocessing_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt nested preprocessing configuration to the legacy flat namespace."""
    flat = {}
    for section, fields in PREPROCESSING_CONFIG_FIELDS.items():
        if section not in config:
            continue

        section_config = config[section]
        if not isinstance(section_config, Mapping):
            raise ValueError(f"Data preprocessing config section '{section}' must be a mapping.")

        for config_key, dest in fields.items():
            if config_key in section_config:
                flat[dest] = section_config[config_key]

    return flat


def get_encoded_config_defaults() -> dict[str, Any]:
    """Return encoded preprocessing defaults in the legacy flat namespace."""
    return flatten_preprocessing_config(ENCODED_CONFIG_DEFAULTS)


def config_overrides_from_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert explicitly supplied flat CLI values back to nested config form."""
    overrides = {}
    for dest, (section, config_key) in CONFIG_PATH_BY_DEST.items():
        if dest not in values:
            continue
        overrides.setdefault(section, {})[config_key] = values[dest]
    return overrides


def resolve_encoded_preprocessing_config(
    config_path: str | None, explicit_cli: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Resolve defaults, YAML, and explicit CLI values using Primus merge rules."""
    user_config = yaml_utils.parse_yaml(config_path) if config_path else {}
    if not isinstance(user_config, Mapping):
        raise ValueError("Data preprocessing config must be a mapping.")

    cli_config = config_overrides_from_mapping(explicit_cli)
    resolved = deep_merge(ENCODED_CONFIG_DEFAULTS, user_config)
    resolved = deep_merge(resolved, cli_config)

    explicit_keys = set(flatten_preprocessing_config(cli_config))
    yaml_keys = set(flatten_preprocessing_config(user_config))
    overridden = sorted(explicit_keys & yaml_keys)
    return flatten_preprocessing_config(resolved), overridden
