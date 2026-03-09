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

import importlib
import pkgutil


def _auto_import_patch_modules() -> None:
    """
    Automatically import all patch modules under this package.

    Any module whose fully-qualified name ends with ``"_patches"`` will be
    imported, which in turn triggers its ``@register_patch`` side effects.

    This allows adding new ``*_patches.py`` files without having to update
    this ``__init__`` module.
    """
    package_name = __name__

    for module_info in pkgutil.walk_packages(__path__, prefix=package_name + "."):
        mod_name = module_info.name

        if not (mod_name.endswith("_patches") or mod_name.endswith("_patch")):
            continue

        importlib.import_module(mod_name)


# Eagerly import all patch modules on package import so patches are registered
# before any backend-specific logic runs.
_auto_import_patch_modules()
