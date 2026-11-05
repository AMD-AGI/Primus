###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE TopKRouter Patches

Patches for replacing Megatron's TopKRouter with PrimusTopKRouter.

After commit 7857383d, MoESubmodules has a router field with default TopKRouter:
    @dataclass
    class MoESubmodules:
        router: RouterBuilder = TopKRouter

When MoELayer.__init__ is called (line 163 in moe_layer.py):
    self.router = submodules.router(config=self.config, pg_collection=pg_collection)

If submodules.router is the default TopKRouter, we need to ensure it's PrimusTopKRouter.
"""

import dataclasses
import functools
import inspect
import sys

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _use_primus_topk_router_in_moe_spec(spec):
    """Swap Megatron's ``TopKRouter`` in an MoE spec's ``MoESubmodules.router`` for ``PrimusTopKRouter``.

    The spec is a ``ModuleSpec`` before Megatron-core 0.19 and ``functools.partial(MoELayer,
    submodules=...)`` from 0.19 on; either way its ``MoESubmodules`` is edited in place. Any other
    router (e.g. one the backend chose) is kept.
    """
    from primus.backends.megatron.core.transformer.moe.router import (
        primus_topk_router_for,
    )

    if isinstance(spec, functools.partial):
        submodules = spec.keywords.get("submodules")
    else:
        submodules = getattr(spec, "submodules", None)
    if submodules is not None and hasattr(submodules, "router"):
        submodules.router = primus_topk_router_for(submodules.router)
    return spec


@register_patch(
    "megatron.moe.primus_topk_router",
    backend="megatron",
    phase="build_args",  # Execute early to patch before model building and MoESubmodules instances are created
    description="Replace TopKRouter with PrimusTopKRouter",
    condition=lambda ctx: not getattr(get_args(ctx), "disable_primus_topk_router", False),
)
def patch_primus_topk_router(ctx: PatchContext):
    """
    Patch Megatron to use Primus TopKRouter.

    This patch addresses the issue introduced in commit 7857383d where MoESubmodules
    has a router field with default TopKRouter. We need to:
    1. Replace TopKRouter class references
    2. Update MoESubmodules.router default value in dataclass fields
    3. Ensure all imports use the patched version

    Behavior:
        - Replace TopKRouter with PrimusTopKRouter in sys.modules and imported modules
        - Patch MoESubmodules.router default value to use PrimusTopKRouter
        - Patch moe_module_specs.MoESubmodules if imported
        - If deprecated MoE is enabled, also patch deprecated_20251209.moe_layer
    """
    from primus.backends.megatron.core.transformer.moe.router import (
        PrimusTopKRouter,
        primus_topk_router_for,
    )

    # Step 1: Patch sys.modules for core megatron router module
    # This ensures any future imports will get PrimusTopKRouter
    router_module = sys.modules.get("megatron.core.transformer.moe.router")
    if router_module is not None:
        router_module.TopKRouter = PrimusTopKRouter
        log_rank_0(
            f"[Patch:megatron.moe.primus_topk_router]   Patched sys.modules['megatron.core.transformer.moe.router'].TopKRouter "
            f"-> {PrimusTopKRouter.__name__}"
        )

    # Step 2: Import and patch moe_layer module
    # This is where MoESubmodules is defined (line 69-75 in moe_layer.py)
    from megatron.core.transformer.moe import moe_layer

    # Replace TopKRouter in moe_layer module
    # This is imported at line 20: from megatron.core.transformer.moe.router import TopKRouter
    moe_layer.TopKRouter = PrimusTopKRouter
    log_rank_0(
        f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.moe_layer.TopKRouter "
        f"-> {PrimusTopKRouter.__name__}"
    )

    # Step 3: Patch MoESubmodules.router default value
    # The dataclass field default is bound at class definition time (line 75: router: RouterBuilder = TopKRouter)
    # We need to update the field's default attribute to use PrimusTopKRouter
    # This moves only what dataclasses.fields() reports (K3's quantile-balancing router reads it):
    # MoESubmodules() keeps the default its generated __init__ was built with, so a spec that leaves
    # the router unset still gets that class. Steps 4 and 5 swap it where specs and MoELayer use it.
    if hasattr(moe_layer, "MoESubmodules"):
        fields_dict = moe_layer.MoESubmodules.__dataclass_fields__
        if "router" in fields_dict:
            field = fields_dict["router"]
            old_default = field.default

            # Update the default value
            field.default = PrimusTopKRouter
            # Clear default_factory if it exists (dataclass uses either default or default_factory, not both)
            if hasattr(field, "default_factory") and field.default_factory is not dataclasses.MISSING:
                field.default_factory = dataclasses.MISSING

            # What MoESubmodules() itself defaults to, which the edit above does not change
            init_router = inspect.signature(moe_layer.MoESubmodules).parameters.get("router")
            init_note = (
                f" (MoESubmodules() still defaults to {getattr(init_router.default, '__name__', init_router.default)})"
                if init_router is not None and init_router.default is not PrimusTopKRouter
                else ""
            )
            log_rank_0(
                f"[Patch:megatron.moe.primus_topk_router]   Patched MoESubmodules.router field default: "
                f"{old_default.__name__ if hasattr(old_default, '__name__') else old_default} "
                f"-> {PrimusTopKRouter.__name__}{init_note}"
            )
        else:
            log_rank_0(
                f"[Patch:megatron.moe.primus_topk_router]   WARNING: MoESubmodules has no 'router' field. "
                f"Available fields: {list(fields_dict.keys())}"
            )

    # Step 4: Patch get_moe_module_spec_for_backend to explicitly use PrimusTopKRouter
    # This is the most reliable method: directly patch the function that creates MoESubmodules
    # to ensure router=PrimusTopKRouter is explicitly passed, bypassing the default value issue
    try:
        from megatron.core.models.gpt import moe_module_specs

        # Store original function
        original_get_moe_module_spec_for_backend = moe_module_specs.get_moe_module_spec_for_backend

        def patched_get_moe_module_spec_for_backend(*args, **kwargs):
            """Wrapped version that ensures MoESubmodules uses PrimusTopKRouter."""
            result = original_get_moe_module_spec_for_backend(*args, **kwargs)
            # result is a ModuleSpec with submodules=MoESubmodules(...), or a functools.partial on 0.19+
            return _use_primus_topk_router_in_moe_spec(result)

        moe_module_specs.get_moe_module_spec_for_backend = patched_get_moe_module_spec_for_backend
        log_rank_0(
            f"[Patch:megatron.moe.primus_topk_router]   Patched get_moe_module_spec_for_backend "
            f"to explicitly set router={PrimusTopKRouter.__name__}"
        )

        # Also patch get_moe_module_spec if it exists
        if hasattr(moe_module_specs, "get_moe_module_spec"):
            original_get_moe_module_spec = moe_module_specs.get_moe_module_spec

            def patched_get_moe_module_spec(*args, **kwargs):
                """Wrapped version that ensures MoESubmodules uses PrimusTopKRouter."""
                return _use_primus_topk_router_in_moe_spec(original_get_moe_module_spec(*args, **kwargs))

            moe_module_specs.get_moe_module_spec = patched_get_moe_module_spec
            log_rank_0(
                f"[Patch:megatron.moe.primus_topk_router]   Patched get_moe_module_spec "
                f"to explicitly set router={PrimusTopKRouter.__name__}"
            )

        # Also verify/update MoESubmodules field default
        if hasattr(moe_module_specs, "MoESubmodules"):
            fields_dict = moe_module_specs.MoESubmodules.__dataclass_fields__
            if "router" in fields_dict:
                field = fields_dict["router"]
                if field.default is PrimusTopKRouter:
                    log_rank_0(
                        f"[Patch:megatron.moe.primus_topk_router]   Verified moe_module_specs.MoESubmodules.router "
                        f"field default is {PrimusTopKRouter.__name__}"
                    )
                else:
                    field.default = PrimusTopKRouter
                    if hasattr(field, "default_factory") and field.default_factory is not dataclasses.MISSING:
                        field.default_factory = dataclasses.MISSING
                    log_rank_0(
                        f"[Patch:megatron.moe.primus_topk_router]   Patched moe_module_specs.MoESubmodules.router field default "
                        f"-> {PrimusTopKRouter.__name__}"
                    )
    except ImportError:
        # moe_module_specs may not be imported yet, which is fine
        # It will be imported later and will use the patched moe_layer.MoESubmodules
        pass

    # Step 5: Patch MoELayer.__init__ to ensure submodules.router is PrimusTopKRouter
    # This is the most reliable method: directly patch MoELayer.__init__ to check and fix
    # submodules.router before it's used, regardless of when MoESubmodules was created
    # (PrimusTurboMegaMoELayer replaces MoELayer and makes the same swap itself)
    original_moelayer_init = moe_layer.MoELayer.__init__

    def patched_moelayer_init(self, config, submodules=None, layer_number=None, pg_collection=None, **kwargs):
        """Wrapped MoELayer.__init__ that ensures submodules.router is PrimusTopKRouter."""
        # Fix submodules.router before calling original __init__
        if submodules is not None and hasattr(submodules, "router"):
            router = submodules.router
            # Only Megatron's TopKRouter class is replaced, not other router classes or builders
            primus_router = primus_topk_router_for(router)
            if primus_router is not router:
                submodules.router = primus_router
                log_rank_0(
                    f"[Patch:megatron.moe.primus_topk_router]   Fixed submodules.router in MoELayer.__init__: "
                    f"{router.__module__}.{router.__name__} -> {primus_router.__name__}"
                )

        # Call original __init__ (forward all kwargs like is_mtp_layer, etc.)
        return original_moelayer_init(self, config, submodules, layer_number, pg_collection, **kwargs)

    moe_layer.MoELayer.__init__ = patched_moelayer_init
    log_rank_0(
        f"[Patch:megatron.moe.primus_topk_router]   Patched MoELayer.__init__ to ensure submodules.router is {PrimusTopKRouter.__name__}"
    )

    # Step 6: If deprecated MoE is enabled, also patch the deprecated module
    use_deprecated_moe = getattr(get_args(ctx), "use_deprecated_20241209_moe_layer", False)
    if use_deprecated_moe:
        from primus.backends.megatron.core.transformer.moe import deprecated_20251209

        deprecated_20251209.moe_layer.TopKRouter = PrimusTopKRouter
        log_rank_0(
            f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.deprecated_20251209.moe_layer.TopKRouter "
            f"-> {PrimusTopKRouter.__name__}"
        )
