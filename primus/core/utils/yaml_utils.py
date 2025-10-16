###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os
import re
from types import SimpleNamespace

import yaml


def parse_yaml(yaml_file: str):
    def replace_env_variables(config):
        """Recursively replace environment variable placeholders in the config."""

        def try_convert_numeric(value: str):
            """Try to convert a string to int or float, else return original string."""
            try:
                if re.fullmatch(r"-?\d+", value):
                    return int(value)
                return float(value)  # handles 1.0, -1.5, 1e-5, etc.
            except ValueError:
                return value

        def replace_match(m):
            """
            Replace matched environment variable patterns.
            - If no default is provided: require the environment variable to be set.
            - If a default is provided: use it when the environment variable is not set.
            """
            var_name = m.group(1)
            default = m.group(2)

            if default is None:
                # ${VAR} → must be set in environment
                if var_name not in os.environ:
                    raise ValueError(
                        f"Environment variable '{var_name}' is required but not set, "
                        f"and no default value is provided."
                    )
                return os.environ[var_name]
            else:
                # ${VAR:default} → use default if VAR is not set
                return os.environ.get(var_name, default)

        if isinstance(config, dict):
            return {replace_env_variables(key): replace_env_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [replace_env_variables(item) for item in config]
        elif isinstance(config, str):
            pattern = re.compile(r"\${([^:{}]+)(?::([^}]*))?}")
            replaced = pattern.sub(replace_match, config)

            if replaced == config:
                return replaced

            if replaced != config and replaced == config.strip():
                return replaced

            return try_convert_numeric(replaced)

        return config

    with open(yaml_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config = replace_env_variables(config)

        if config is None:
            return {}

        if "bases" in config:
            for base_file in config["bases"]:
                full_base_file = os.path.join(os.path.dirname(yaml_file), base_file)
                base_config = parse_yaml(full_base_file)
                # key must in base config
                for key in config:
                    if key != "bases":
                        assert key in base_config, (
                            f"The argument '{key}' in the a config file '{yaml_file}' "
                            f"cannot be found in the base config file '{base_file}'."
                        )
                for key, value in base_config.items():
                    if key != "bases" and key not in config:
                        config[key] = value
            # remove bases config
            del config["bases"]

        if "includes" in config:
            for include_file in config["includes"]:
                full_include_file = os.path.join(os.path.dirname(yaml_file), include_file)
                include_config = parse_yaml(full_include_file)
                # overrides if exist
                for key, value in include_config.items():
                    if key == "includes":
                        continue
                    if key not in config:
                        config[key] = value
            # remove includes config
            del config["includes"]

        return config


def dict_to_nested_namespace(d: dict):
    """Recursively convert dictionary to a nested SimpleNamespace."""
    return SimpleNamespace(
        **{k: dict_to_nested_namespace(v) if isinstance(v, dict) else v for k, v in d.items()}
    )


def nested_namespace_to_dict(obj):
    """Recursively convert nested SimpleNamespace to a dictionary."""
    if isinstance(obj, SimpleNamespace):
        return {key: nested_namespace_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        return [nested_namespace_to_dict(item) for item in obj]
    return obj


def parse_yaml_to_namespace(yaml_file: str):
    return dict_to_nested_namespace(parse_yaml(yaml_file))


def parse_nested_namespace_to_str(namespace: SimpleNamespace, indent=4):
    return json.dumps(nested_namespace_to_dict(namespace), indent=indent)


def delete_namespace_key(namespace: SimpleNamespace, key: str):
    if hasattr(namespace, key):
        delattr(namespace, key)


def has_key_in_namespace(namespace: SimpleNamespace, key: str):
    return hasattr(namespace, key)


def check_key_in_namespace(namespace: SimpleNamespace, key: str):
    # WARN: namespace should have name attr
    assert has_key_in_namespace(namespace, key), f"Failed to find key({key}) in namespace({namespace.name})"


def get_value_by_key(namespace: SimpleNamespace, key: str):
    check_key_in_namespace(namespace, key)
    return getattr(namespace, key)


def set_value_by_key(namespace: SimpleNamespace, key: str, value, allow_override=False):
    if not allow_override:
        assert not hasattr(namespace, key), f"Not allowed to override key({key}) in namespace({namespace})"
    if value == "null":
        value = None
    return setattr(namespace, key, value)


def override_namespace(original_ns: SimpleNamespace, overrides_ns: SimpleNamespace):
    if overrides_ns is None:
        return

    for key in vars(overrides_ns):
        if not has_key_in_namespace(original_ns, key):
            raise Exception(f"Override namespace failed: can't find key({key}) in namespace {original_ns}")
        new_value = get_value_by_key(overrides_ns, key)
        if isinstance(new_value, SimpleNamespace):
            override_namespace(get_value_by_key(original_ns, key), new_value)
        else:
            set_value_by_key(original_ns, key, new_value, allow_override=True)


def merge_namespace(dst: SimpleNamespace, src: SimpleNamespace, allow_override=False, excepts: list = None):
    src_dict = vars(src)
    dst_dict = vars(dst)
    for key, value in src_dict.items():
        if key in excepts:
            continue
        if key in dst_dict and not allow_override:
            raise ValueError(f"Key '{key}' from {src.name} already exists in {dst.name}.")
        else:
            setattr(dst, key, value)


def dump_namespace_to_yaml(ns: SimpleNamespace, file_path: str):
    """
    Recursively convert a SimpleNamespace (or nested namespaces) into a Python dict
    and dump it to a YAML file.

    Args:
        ns (SimpleNamespace): The namespace object to serialize.
        file_path (str): The output path for the YAML file.

    Example:
        >>> ns = SimpleNamespace(a=1, b=SimpleNamespace(c=2))
        >>> dump_namespace_to_yaml(ns, "config.yaml")
    """

    def ns_to_dict(obj):
        if isinstance(obj, SimpleNamespace):
            return {k: ns_to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: ns_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ns_to_dict(v) for v in obj]
        else:
            return obj

    with open(file_path, "w") as f:
        yaml.dump(ns_to_dict(ns), f, default_flow_style=False, sort_keys=False)
