###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Patch Collection

This module defines the public entrypoint for applying MaxText-specific patches.

All ``*_patches.py`` files under this package are auto-discovered and imported
at package-import time, which triggers their ``@register_patch`` side effects.
This follows the same convention used by ``primus.backends.megatron.patches``.
"""

_ALLOWED_PATCH_MODULES = [
    "primus.backends.maxtext.patches.checkpoint_patches",
    "primus.backends.maxtext.patches.config_patches",
    "primus.backends.maxtext.patches.data_patches",
    "primus.backends.maxtext.patches.model_patches",
    "primus.backends.maxtext.patches.optimizer_patches",
    "primus.backends.maxtext.patches.tokenizer_patches",
]


def _auto_import_patch_modules() -> None:
    """
    Import explicitly allowed patch modules under this package.

    Only whitelisted modules are imported, which triggers their
    ``@register_patch`` side effects. This prevents arbitrary code
    execution from untrusted files in the package directory.
    """
    import importlib

    for mod_name in _ALLOWED_PATCH_MODULES:
        importlib.import_module(mod_name)


# Eagerly import all patch modules on package import so patches are registered
# before any backend-specific logic runs.
_auto_import_patch_modules()
