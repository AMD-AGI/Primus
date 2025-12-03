###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Environment setup utilities for Primus training.

This module provides functions to configure environment variables for various
third-party libraries and frameworks used in training.
"""

import os
from pathlib import Path
from typing import Optional


def _ensure_directory(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_huggingface_cache(data_path: str, force: bool = False) -> str:
    """
    Setup HuggingFace cache directory.

    Args:
        data_path: Base data directory path
        force: If True, override existing HF_HOME setting

    Returns:
        The HF_HOME path that was set or already existed
    """
    if "HF_HOME" in os.environ and not force:
        hf_home = os.environ["HF_HOME"]
        print(f"[Primus:Env] HF_HOME already set: {hf_home}")
        return hf_home

    hf_home = os.path.join(data_path, "huggingface")
    os.environ["HF_HOME"] = hf_home
    print(f"[Primus:Env] Set HF_HOME={hf_home}")

    _ensure_directory(hf_home)

    return hf_home


def setup_torch_cache(data_path: str, force: bool = False) -> str:
    """
    Setup PyTorch cache directory.

    Args:
        data_path: Base data directory path
        force: If True, override existing TORCH_HOME setting

    Returns:
        The TORCH_HOME path that was set or already existed
    """
    if "TORCH_HOME" in os.environ and not force:
        torch_home = os.environ["TORCH_HOME"]
        print(f"[Primus:Env] TORCH_HOME already set: {torch_home}")
        return torch_home

    torch_home = os.path.join(data_path, "torch")
    os.environ["TORCH_HOME"] = torch_home
    print(f"[Primus:Env] Set TORCH_HOME={torch_home}")

    _ensure_directory(torch_home)

    return torch_home


def setup_transformers_cache(data_path: str, force: bool = False) -> str:
    """
    Setup Transformers cache directory.

    Args:
        data_path: Base data directory path
        force: If True, override existing TRANSFORMERS_CACHE setting

    Returns:
        The TRANSFORMERS_CACHE path that was set or already existed
    """
    if "TRANSFORMERS_CACHE" in os.environ and not force:
        transformers_cache = os.environ["TRANSFORMERS_CACHE"]
        print(f"[Primus:Env] TRANSFORMERS_CACHE already set: {transformers_cache}")
        return transformers_cache

    transformers_cache = os.path.join(data_path, "transformers")
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    print(f"[Primus:Env] Set TRANSFORMERS_CACHE={transformers_cache}")

    _ensure_directory(transformers_cache)

    return transformers_cache


def setup_training_env(
    data_path: str,
    setup_hf: bool = True,
    setup_torch: bool = False,
    setup_transformers: bool = False,
    force: bool = False,
) -> dict:
    """
    Setup all training environment variables.

    Args:
        data_path: Base data directory path
        setup_hf: Whether to setup HuggingFace cache
        setup_torch: Whether to setup PyTorch cache
        setup_transformers: Whether to setup Transformers cache
        force: If True, override existing settings

    Returns:
        Dictionary of environment variables that were set
    """
    env_vars = {}

    if setup_hf:
        env_vars["HF_HOME"] = setup_huggingface_cache(data_path, force)

    if setup_torch:
        env_vars["TORCH_HOME"] = setup_torch_cache(data_path, force)

    if setup_transformers:
        env_vars["TRANSFORMERS_CACHE"] = setup_transformers_cache(data_path, force)

    return env_vars


def set_env_var(
    key: str,
    value: str,
    force: bool = False,
    verbose: bool = True,
) -> Optional[str]:
    """
    Set an environment variable with optional override.

    Args:
        key: Environment variable name
        value: Environment variable value
        force: If True, override existing value
        verbose: If True, print status messages

    Returns:
        The value that was set, or None if not set
    """
    if key in os.environ and not force:
        if verbose:
            print(f"[Primus:Env] {key} already set: {os.environ[key]}")
        return os.environ[key]

    os.environ[key] = value
    if verbose:
        print(f"[Primus:Env] Set {key}={value}")

    return value
