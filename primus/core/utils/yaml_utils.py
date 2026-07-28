###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
from types import SimpleNamespace
from typing import Any, Mapping

from primus.core.config.merge_utils import deep_merge
from primus.core.config.yaml_loader import parse_yaml as _parse_yaml_core


def parse_yaml(yaml_file: str):
    return _parse_yaml_core(yaml_file)


def dict_to_nested_namespace(obj: Any) -> Any:
    """
    Recursively convert a mapping (and nested mappings) into SimpleNamespace.

    This also walks lists/tuples and converts any nested mappings they contain.
    """
    if isinstance(obj, Mapping):
        return SimpleNamespace(**{k: dict_to_nested_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_nested_namespace(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(dict_to_nested_namespace(v) for v in obj)
    return obj


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


def _assign_namespace_from_dict_inplace(namespace: SimpleNamespace, src_dict: dict):
    """
    Assign nested dict values into a namespace in place.

    Existing nested SimpleNamespace objects are updated recursively to keep
    references stable where possible.
    """
    for key, value in src_dict.items():
        if isinstance(value, dict):
            current = getattr(namespace, key, None)
            if isinstance(current, SimpleNamespace):
                _assign_namespace_from_dict_inplace(current, value)
            else:
                setattr(namespace, key, dict_to_nested_namespace(value))
        else:
            set_value_by_key(namespace, key, dict_to_nested_namespace(value), allow_override=True)


def deep_merge_namespace(namespace: SimpleNamespace, override_dict: dict):
    """
    Apply dict-style deep merge into namespace in place.
    """
    base_dict = nested_namespace_to_dict(namespace)
    merged_dict = deep_merge(base_dict, override_dict)
    _assign_namespace_from_dict_inplace(namespace, merged_dict)
    return namespace


def override_namespace(original_ns: SimpleNamespace, overrides_ns: SimpleNamespace):
    if overrides_ns is None:
        return
    deep_merge_namespace(original_ns, nested_namespace_to_dict(overrides_ns))


def merge_namespace(dst: SimpleNamespace, src: SimpleNamespace, allow_override=False, excepts: list = None):
    src_dict = nested_namespace_to_dict(src)
    dst_dict = nested_namespace_to_dict(dst)
    excepts = set(excepts or [])

    effective_override = {}
    for key, value in src_dict.items():
        if key in excepts:
            continue
        if key in dst_dict and not allow_override:
            continue  # Skip duplicate keys, keep dst value
        effective_override[key] = value

    deep_merge_namespace(dst, effective_override)


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
    import yaml

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
