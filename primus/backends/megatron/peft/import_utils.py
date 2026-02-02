###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Import utilities for PEFT module."""

from typing import Any, Tuple
import importlib


def safe_import_from(module_path: str, attr_name: str) -> Tuple[Any, bool]:
    """
    Safely import an attribute from a module.
    
    Args:
        module_path: Full path to the module (e.g., "megatron.core.extensions.transformer_engine")
        attr_name: Name of the attribute to import
        
    Returns:
        Tuple of (attribute, success_flag). If import fails, returns (None, False).
    """
    try:
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        return attr, True
    except (ImportError, AttributeError):
        return None, False


def safe_import(module_name: str) -> Tuple[Any, bool]:
    """
    Safely import a module.
    
    Args:
        module_name: Name of the module to import
        
    Returns:
        Tuple of (module, success_flag). If import fails, returns (None, False).
    """
    try:
        module = importlib.import_module(module_name)
        return module, True
    except ImportError:
        return None, False
