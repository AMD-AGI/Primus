###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Training-framework config adapters for the projection tool.

The projection describes a workload in one vocabulary: the flat argument names
that ``ModelConfig`` / ``RuntimeConfig`` / ``ModelParallelConfig`` declare.  That
vocabulary is spelled the way Megatron spells it for historical reasons, but it
is not a claim that the workload must be a Megatron job -- a TorchTitan Llama 3
and a MaxText Llama 3 run the same GEMMs over the same tensors, and the profiler
tree only ever asked for shapes.

A *config adapter* is what closes the spelling gap: given the merged
``pre_trainer`` namespace for one backend, it writes the projection's field names
onto that same namespace.  ``convert_primus_config_to_projection_config`` then
proceeds identically for every backend.

Two properties are required of an adapter, and both come from how the
performance driver uses the result:

* **In place.**  The driver holds the namespace across the whole run and hands
  the same object to the profilers, so an adapter that returned a fresh object
  would strand the driver's later edits.
* **Idempotent.**  The driver mutates the *normalized* namespace between
  conversions -- it limits the stack to a couple of representative layers,
  rescales EP to fit the bench node, flattens PP -- and then converts again.  A
  second pass that re-derived the architecture from the backend-native keys would
  silently undo all of it and project the full model instead of the benched one.
  Adapters therefore mark the namespace once (:func:`mark_normalized`) and, on
  any later call, re-derive only from the flat fields already present.

Selection is registry-driven so no framework name is hardcoded at the call site.
Out-of-tree code can add a backend by calling :func:`register_config_adapter`.
"""

from typing import Callable, Optional, Tuple

# A config adapter normalizes one backend's merged trainer namespace in place and
# returns it.
ConfigAdapter = Callable[..., object]

_CONFIG_ADAPTER_REGISTRY: "dict[str, ConfigAdapter]" = {}

# Set on a namespace once its backend-native keys have been translated, so a
# second conversion re-derives from the (possibly driver-mutated) flat fields
# instead of from the backend's model registry.
NORMALIZED_FLAG = "_primus_projection_normalized"


def mark_normalized(args) -> None:
    """Record that *args* now carries the projection's flat field names."""
    setattr(args, NORMALIZED_FLAG, True)


def is_normalized(args) -> bool:
    """Whether *args* has already been translated by a config adapter."""
    return bool(getattr(args, NORMALIZED_FLAG, False))


def register_config_adapter(name: str, adapter: ConfigAdapter) -> None:
    """Register *adapter* under framework *name* (case-insensitive).

    Safe to call multiple times; the last registration for a name wins.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Framework name must be a non-empty string, got {name!r}")
    if not callable(adapter):
        raise TypeError(f"Config adapter for '{name}' must be callable, got {adapter!r}")
    _CONFIG_ADAPTER_REGISTRY[name.lower().strip()] = adapter


def get_config_adapter(name: str) -> Optional[ConfigAdapter]:
    """Return the adapter registered for *name*, or ``None``."""
    if not name:
        return None
    _ensure_builtins_registered()
    return _CONFIG_ADAPTER_REGISTRY.get(name.lower().strip())


def available_frameworks() -> Tuple[str, ...]:
    """Return the sorted names of every framework the projection can read."""
    _ensure_builtins_registered()
    return tuple(sorted(_CONFIG_ADAPTER_REGISTRY))


def _ensure_builtins_registered() -> None:
    """Register the in-tree adapters on first use (idempotent).

    Imports are deferred to here so that importing this registry costs nothing;
    the Megatron and DLRM adapters live in ``training_config``, which imports
    this module, and would otherwise be a cycle.
    """
    if getattr(_ensure_builtins_registered, "_done", False):
        return
    _ensure_builtins_registered._done = True

    from primus.core.projection.training_config import (
        _DLRM_FRAMEWORKS,
        dlrm_derive_default_args,
        megatron_derive_default_args,
    )

    register_config_adapter("megatron", megatron_derive_default_args)
    for alias in _DLRM_FRAMEWORKS:
        register_config_adapter(alias, dlrm_derive_default_args)

    from primus.core.projection.frameworks.torchtitan import (
        torchtitan_apply_bench_overrides,
        torchtitan_derive_default_args,
    )

    register_config_adapter("torchtitan", torchtitan_derive_default_args)
    register_bench_override("torchtitan", torchtitan_apply_bench_overrides)

    from primus.core.projection.frameworks.jax import maxtext_derive_default_args

    # MaxText is the JAX pretraining backend Primus ships; ``jax`` is accepted as
    # the name users reach for when they mean "the JAX one".
    for alias in ("maxtext", "jax"):
        register_config_adapter(alias, maxtext_derive_default_args)


def framework_of(args) -> str:
    """Return the training framework *args* describes, defaulting to Megatron."""
    return (getattr(args, "framework", "") or "").lower().strip() or "megatron"


_BENCH_OVERRIDE_REGISTRY: "dict[str, ConfigAdapter]" = {}


def register_bench_override(name: str, fn) -> None:
    """Register the flat-to-native write-back for framework *name*.

    Normalization translates a backend's config into the projection's flat
    fields; the performance driver then edits those fields to shrink the model
    onto the bench node.  A backend that will be *benchmarked* needs the reverse
    translation too, so the model it builds is the one those edits describe.
    Megatron needs no entry here -- the flat fields are its own.
    """
    _BENCH_OVERRIDE_REGISTRY[name.lower().strip()] = fn


def apply_bench_overrides(primus_config, module_name: str = "pre_trainer") -> None:
    """Write the driver's bench edits back into the backend's own config.

    A no-op for backends whose config the projection already reads natively.
    """
    args = primus_config.get_module_config(module_name)
    _ensure_builtins_registered()
    writeback = _BENCH_OVERRIDE_REGISTRY.get(framework_of(args))
    if writeback is not None:
        writeback(args)


def normalize_primus_config(primus_config, module_name: str = "pre_trainer") -> str:
    """Translate a loaded Primus config into the projection's field names.

    The projection drivers edit the trainer namespace directly -- capping the
    layer stack, rescaling EP onto the bench node, flattening PP -- and only then
    build a :class:`TrainingConfig` from it.  Those edits are written in the
    projection's vocabulary, so the namespace has to already be speaking it:
    normalizing at load time is what keeps a backend-native config from being
    read as an empty Megatron one (every parallel degree defaulting to 1) and
    then re-expanded to the full model on conversion.

    Returns the framework name.
    """
    args = primus_config.get_module_config(module_name)
    framework = framework_of(args)
    resolve_config_adapter(framework)(args)
    return framework


def resolve_config_adapter(name: str) -> ConfigAdapter:
    """Return the adapter for framework *name*.

    Raises:
        NotImplementedError: if no adapter is registered for *name*.
    """
    adapter = get_config_adapter(name)
    if adapter is None:
        supported = ", ".join(available_frameworks())
        raise NotImplementedError(
            f"Unsupported framework for projection: {name!r}. "
            f"Supported frameworks: {supported}. Register another by calling "
            "primus.core.projection.frameworks.register_config_adapter(name, adapter)."
        )
    return adapter
