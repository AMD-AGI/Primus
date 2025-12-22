###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Patch Collection

This module defines the public entrypoint for applying Megatron-specific patches.
"""

import importlib
import pkgutil

from primus.core.patches import run_patches


def _auto_import_patch_modules() -> None:
    """
    Automatically import all patch modules under this package.

    Any module whose fully-qualified name ends with ``"_patches"`` will be
    imported, which in turn triggers its ``@register_patch`` side effects.

    This allows adding new ``*_patches.py`` files (including in subpackages
    like ``moe_patches`` or ``parallelism``) without having to update this
    ``__init__`` module.
    """
    package_name = __name__

    # Walk all submodules / subpackages under this package's filesystem path.
    for module_info in pkgutil.walk_packages(__path__, prefix=package_name + "."):
        mod_name = module_info.name

        # Only import modules whose names clearly identify them as patch modules.
        # Support both "*_patches.py" and "*_patch.py" naming conventions.
        if not (mod_name.endswith("_patches") or mod_name.endswith("_patch")):
            continue

        importlib.import_module(mod_name)


# Eagerly import all patch modules on package import so patches are registered
# before any backend-specific logic runs.
_auto_import_patch_modules()

