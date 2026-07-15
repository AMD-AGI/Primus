###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-Bridge Patch Collection

This module defines the public entrypoint for applying Megatron-Bridge-specific
patches.  It mirrors the auto-import mechanism used by
``primus.backends.megatron.patches``.
"""

import importlib
import pkgutil


def _auto_import_patch_modules() -> None:
    """
    Automatically import all patch modules under this package.

    Any module whose fully-qualified name ends with ``"_patches"`` or
    ``"_patch"`` will be imported, which triggers ``@register_patch``
    side effects.
    """
    package_name = __name__

    for module_info in pkgutil.walk_packages(__path__, prefix=package_name + "."):
        mod_name = module_info.name
        if not (mod_name.endswith("_patches") or mod_name.endswith("_patch")):
            continue
        importlib.import_module(mod_name)


_auto_import_patch_modules()
