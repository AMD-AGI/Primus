###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
NeMo AutoModel patch collection.

Patches are how this backend adds Primus-side behaviour (numerics, sharding,
profiling, per-model wiring) without modifying AutoModel or diffusers source.
This package is the entrypoint: importing it imports every ``*_patches.py``
module beneath it, which runs their ``@register_patch`` side effects.

Adding a patch
--------------
Create ``<something>_patches.py`` anywhere under this package and decorate a
function with ``@register_patch``. Discovery is automatic -- **this module does
not need to be edited**, and neither does the trainer::

    from primus.core.patches import PatchContext, get_param, register_patch

    def _enabled(ctx: PatchContext) -> bool:
        return os.getenv("PRIMUS_MY_FEATURE", "0").lower() in {"1", "true", "yes", "on"}

    @register_patch(
        "nemo_automodel.my_feature",
        backend="nemo_automodel",
        phase="before_train",
        description="one line, shown in the run log",
        condition=_enabled,
        priority=50,
    )
    def _apply(ctx: PatchContext) -> None:
        ...

Registration vs. application
----------------------------
Importing this package only *registers* patches; it does not apply them.
``run_patches`` evaluates each patch's ``condition`` at call time, so a patch
whose condition is False is a no-op for that job. Make the condition precise:
registration is global to the process, so a model-specific patch is registered
even for jobs training a different model and must decline to apply itself.

Ordering is by ``priority`` (lower runs first, ties keep registration order),
which is the only supported way to express "this must run before that". Do not
rely on module import order -- ``pkgutil`` walks the tree alphabetically and
that is not a contract.

Failure policy
--------------
``run_patches`` catches and logs a failing patch and continues, so a broken
optional feature degrades the run rather than ending it. A patch that must not
fail silently should validate loudly inside its own body.
"""

import importlib
import pkgutil


def _auto_import_patch_modules() -> None:
    """Import every ``*_patches.py`` / ``*_patch.py`` module under this package.

    Walking the tree (rather than listing modules here) is what lets a new patch
    file register itself without touching shared code. That matters more than it
    looks: a hand-maintained list is a single file every feature branch has to
    edit, which turns independent changes into merge conflicts.
    """
    for module_info in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
        mod_name = module_info.name
        if not (mod_name.endswith("_patches") or mod_name.endswith("_patch")):
            continue
        importlib.import_module(mod_name)


_auto_import_patch_modules()
