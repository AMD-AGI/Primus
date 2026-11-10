###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers shared by the Muse Spark patches: the model-type condition, flag parsing
(``bool_flag`` / ``config_flag``) and merged-args access."""

import numbers

from primus.backends.megatron.core.models.muse_spark.layer_patterns import MODEL_TYPE
from primus.core.patches import get_args

__all__ = [
    "MODEL_TYPE",
    "MergedArgs",
    "backend_args_of",
    "bool_flag",
    "config_flag",
    "is_muse_spark_model_type",
    "merged_args_of",
]

TRUE_TEXT = ("1", "true", "yes", "on")
FALSE_TEXT = ("0", "false", "no", "off")


def bool_flag(value, key, default=False) -> bool:
    """A Muse Spark boolean config key (``key``) as a bool. None (unset) or a blank string
    (an unset env-substituted value) is ``default``, a bool is
    used as is, the ints 0 / 1 are False / True, and a string (an env-driven yaml value) is
    true for 1/true/yes/on and false for 0/false/no/off (any case; surrounding whitespace
    is stripped); an empty value means ``default``. A numpy bool (``np.bool_``) is used as a bool.
    Anything else (e.g. a typo such as ``"ture"``, ``2``, ``1.0``) raises ValueError naming
    ``key``: a typo must not silently turn a feature on or off."""
    if value is None or (isinstance(value, str) and not value.strip()):
        # the default is parsed by the same rules (a string default "false" is False)
        return bool_flag(default, key, False)
    if isinstance(value, bool):
        return value
    if getattr(getattr(value, "dtype", None), "kind", None) == "b" and getattr(value, "ndim", None) == 0:
        return bool(value)  # np.bool_ (not a numbers.Integral)
    if isinstance(value, numbers.Integral) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in TRUE_TEXT:
            return True
        if text in FALSE_TEXT:
            return False
    raise ValueError(f"{key}={value!r} is not a boolean: use true / false (also yes / no, on / off, 1 / 0)")


def config_flag(config, key, default) -> bool:
    """The boolean ``key`` of a model config, parsed with :func:`bool_flag`: ``default`` when
    the config has no such attribute or it is unset (None / a blank string, e.g. YAML ``~`` or
    an unset env value), so a default-on kernel is not turned off by an empty value;
    ``"false"`` / ``"0"`` are False and a typo raises ValueError."""
    return bool_flag(getattr(config, key, default), key, default)


def backend_args_of(ctx):
    """The ``build_args`` phase's backend args (``ctx.extra['backend_args']``); RuntimeError
    when the context has none (the Muse Spark ``build_args`` patches need them)."""
    args = ctx.extra.get("backend_args")
    if args is None:
        raise RuntimeError("Muse Spark build_args patch: ctx.extra has no backend_args")
    return args


class MergedArgs:
    """View of several args namespaces.

    Reads take the first source that has the key with a value other than None (an
    argparse default of None in ``backend_args`` must not hide a value configured in
    ``params``); None when every source that has the key holds None. Writes go to every source that has
    the key (to the first source when none has it), so a patch that clears a flag
    through this view clears it in both namespaces.

    The ``build_args`` patches see two namespaces: ``backend_args`` (the Megatron args,
    built from ``module_config.params`` by ``adapter.convert_config``, then changed by CLI
    overrides and earlier patches) and ``module_config.params`` (which also holds the
    Primus-only keys, e.g. the Turbo flags). ``backend_args`` is the newer copy of a key
    both carry, so it comes first; a Primus-only key is read from ``params``.
    """

    def __init__(self, *sources):
        """``sources`` in read order; None entries are dropped (e.g. a context without
        ``module_config``). ValueError when none is left: a write would have nowhere to go."""
        kept = tuple(s for s in sources if s is not None)
        if not kept:
            raise ValueError("MergedArgs needs at least one args namespace (all sources are None)")
        object.__setattr__(self, "_sources", kept)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        found = False
        for s in object.__getattribute__(self, "_sources"):
            if hasattr(s, name):
                value = getattr(s, name)
                if value is not None:
                    return value
                found = True
        if found:
            return None
        raise AttributeError(name)

    def __setattr__(self, name, value):
        """Set ``name`` on every source that has it (on the first source when none has it)."""
        sources = object.__getattribute__(self, "_sources")
        holders = [s for s in sources if hasattr(s, name)] or list(sources[:1])
        for s in holders:
            setattr(s, name, value)


def merged_args_of(ctx) -> MergedArgs:
    """``backend_args`` first, then ``module_config.params`` (see :class:`MergedArgs`).

    RuntimeError when the context has no ``backend_args`` (:func:`backend_args_of`): writes
    through the view must reach the Megatron args. A context without ``module_config``
    gives a view of ``backend_args`` alone."""
    params = getattr(ctx.extra.get("module_config"), "params", None)
    return MergedArgs(backend_args_of(ctx), params)


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
    return model_type == MODEL_TYPE
