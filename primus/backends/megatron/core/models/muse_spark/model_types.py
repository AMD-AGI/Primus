###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark ``model_type`` values and the one condition shared by the Muse Spark patches."""

from primus.backends.megatron.core.models.muse_spark.layer_patterns import MODEL_TYPES
from primus.core.patches import get_args

__all__ = ["MODEL_TYPES", "MergedArgs", "backend_args_of", "is_muse_spark_model_type", "merged_args_of"]


def backend_args_of(ctx):
    """The ``build_args`` phase's backend args (``ctx.extra['backend_args']``); RuntimeError
    when the context has none (the Muse Spark ``build_args`` patches need them)."""
    args = ctx.extra.get("backend_args")
    if args is None:
        raise RuntimeError("Muse Spark build_args patch: ctx.extra has no backend_args")
    return args


class MergedArgs:
    """View of several args namespaces.

    Reads take the first source that has the key. Writes go to every source that has
    the key (to the first source when none has it), so a patch that clears a flag
    through this view clears it in both namespaces.

    The ``build_args`` patches see two namespaces: ``backend_args`` (the Megatron args,
    built from ``module_config.params`` by ``adapter.convert_config``, then changed by CLI
    overrides and earlier patches) and ``module_config.params`` (which also holds the
    Primus-only keys, e.g. the Turbo flags). ``backend_args`` is the newer copy of a key
    both carry, so it comes first; a Primus-only key is read from ``params``.
    """

    def __init__(self, *sources):
        object.__setattr__(self, "_sources", tuple(s for s in sources if s is not None))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        for s in object.__getattribute__(self, "_sources"):
            if hasattr(s, name):
                return getattr(s, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        """Set ``name`` on every source that has it (on the first source when none has it)."""
        sources = object.__getattribute__(self, "_sources")
        holders = [s for s in sources if hasattr(s, name)] or list(sources[:1])
        for s in holders:
            setattr(s, name, value)


def merged_args_of(ctx) -> MergedArgs:
    """``backend_args`` first, then ``module_config.params`` (see :class:`MergedArgs`)."""
    params = getattr(ctx.extra.get("module_config"), "params", None)
    return MergedArgs(ctx.extra.get("backend_args"), params)


def is_muse_spark_model_type(ctx) -> bool:
    """Patch condition: the run's ``model_type`` is a Muse Spark one.

    Reads the Megatron args when they exist, else the ``build_args`` phase's backend_args.
    """
    try:
        model_type = getattr(get_args(ctx), "model_type", None)
    except AssertionError:
        model_type = None
    if model_type is None:
        model_type = getattr(ctx.extra.get("backend_args"), "model_type", None)
    return model_type in MODEL_TYPES
