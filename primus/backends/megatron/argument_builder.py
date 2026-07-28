###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import argparse
import enum
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Union

from primus.core.utils.env import get_torchrun_env
from primus.core.utils.module_utils import warning_rank_0


# ------------------------------------------------------------
# Build the original Megatron argument parser
# ------------------------------------------------------------
def _build_megatron_parser() -> argparse.ArgumentParser:
    """
    Construct Megatron-LM's official argparse parser without touching sys.argv.

    This function directly calls:
        megatron.training.arguments.add_megatron_arguments(parser)

    We do NOT parse command-line arguments here; we simply want Megatron's
    argument *definitions* (names, defaults, types).
    """
    from megatron.training.arguments import add_megatron_arguments

    parser = argparse.ArgumentParser(
        description="Primus Megatron arguments",
        allow_abbrev=False,  # Disable abbreviation to avoid unexpected behaviour
    )
    return add_megatron_arguments(parser)


# ------------------------------------------------------------
# Load Megatron's default values (cached)
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_megatron_defaults() -> Dict[str, Any]:
    """
    Load all default values defined by Megatron-LM's argparse.

    We call parser.parse_args([]), which returns a Namespace containing ONLY
    default values (because no CLI arguments are provided).
    """
    parser = _build_megatron_parser()
    args = parser.parse_args([])  # Parse an empty list → only defaults
    return vars(args).copy()  # Convert Namespace → dict and cache


# ------------------------------------------------------------
# Load Megatron's enum argparse type converters (cached)
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_megatron_enum_types() -> Dict[str, Callable[[str], Any]]:
    """Map each *enum-typed* Megatron arg ``dest`` to its argparse converter.

    Primus feeds config/CLI values straight into the namespace instead of
    through Megatron's argparse, so an enum arg like ``attention_backend``
    arrives as a plain ``str`` and silently loses against every downstream
    ``== SomeEnum.member`` comparison. We only touch enum args (the class of
    bug this addresses); int/float/str args already arrive well-typed from
    YAML. Reuse the parser's own converters so nothing is hand-maintained.
    Returns ``{}`` if the parser cannot be built.
    """
    try:
        parser = _build_megatron_parser()
    except Exception:  # noqa: BLE001 - megatron may be unavailable (e.g. unit tests)
        return {}
    types: Dict[str, Callable[[str], Any]] = {}
    for action in parser._actions:
        convert = getattr(action, "type", None)
        choices = getattr(action, "choices", None)
        if callable(convert) and choices and all(isinstance(c, enum.Enum) for c in choices):
            types[action.dest] = convert
    return types


def _coerce_value(convert: Callable[[str], Any], value: Any) -> Any:
    """Apply an argparse ``type`` converter to a raw override value.

    Mirrors what argparse does for a CLI string. Non-string values (already the
    right type, e.g. from YAML) pass through untouched; list values are
    converted element-wise to support ``nargs='+'`` args.
    """
    if isinstance(value, str):
        return convert(value)
    if isinstance(value, list):
        return [convert(v) if isinstance(v, str) else v for v in value]
    return value


# ------------------------------------------------------------
# MegatronArgBuilder: merge Primus → Megatron
# ------------------------------------------------------------
class MegatronArgBuilder:
    """
    A lightweight utility to build final Megatron-LM arguments for Primus.

    It merges:
        1. Primus CLI arguments
        2. Primus config arguments
        3. Megatron-LM's default values

    WITHOUT defining any mapping and WITHOUT maintaining version compatibility
    manually — because we rely entirely on Megatron's own argparse.

    Usage:
        builder = MegatronArgBuilder()
        builder.update(cli_args)
        builder.update(config_args)
        megatron_ns = builder.finalize()

    'megatron_ns' is a SimpleNamespace containing all fields Megatron expects.
    """

    def __init__(self):
        # Stores override values coming from CLI + config
        self.overrides: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Add values to the override dictionary
    # ------------------------------------------------------------------
    def update(self, values: Union[Mapping[str, Any], SimpleNamespace]) -> "MegatronArgBuilder":
        """
        Merge a collection of values (e.g., CLI args or config) into the
        current override set.

        - Supports both Mapping (e.g., dict) and SimpleNamespace inputs.
        - Only parameters that exist in Megatron's default arguments are accepted.
        - None values are allowed and will override Megatron defaults.
        - Non-Megatron parameters are silently ignored.
        """
        # Get Megatron's supported parameters and enum argparse converters
        megatron_defaults = _load_megatron_defaults()
        megatron_keys = set(megatron_defaults.keys())
        enum_types = _load_megatron_enum_types()

        # Normalize input to a (key, value) iterable
        if isinstance(values, SimpleNamespace):
            items = vars(values).items()
        else:
            items = values.items()

        for key, value in items:
            # Only accept parameters that Megatron recognizes (including None overrides)
            if key not in megatron_keys:
                continue

            self.overrides[key] = self._coerce_enum(key, value, enum_types)

        return self

    @staticmethod
    def _coerce_enum(key: str, value: Any, enum_types: Mapping[str, Callable[[str], Any]]) -> Any:
        """Coerce an enum override to its enum type (see _load_megatron_enum_types)."""
        convert = enum_types.get(key)
        if convert is None or value is None:
            return value

        try:
            return _coerce_value(convert, value)
        except Exception as exc:  # noqa: BLE001 - keep raw value, don't abort the build
            warning_rank_0(f"[MegatronArgBuilder] could not coerce '{key}'={value!r}: {exc}")
            return value

    # ------------------------------------------------------------------
    # Produce the final Megatron Namespace
    # ------------------------------------------------------------------
    def to_namespace(self) -> SimpleNamespace:
        """
        Combine:
            - Megatron default arguments
            - Primus overrides (CLI + config)
            - Distributed environment variables (runtime)
        and produce a final SimpleNamespace that can be passed directly
        to Megatron's training entrypoints.

        Fields not provided by Primus are automatically filled with Megatron's defaults.
        Distributed environment (world_size, rank, local_rank) is injected automatically.
        """
        # Start with Megatron's defaults
        final = dict(_load_megatron_defaults())

        # Apply overrides (Primus CLI / config)
        for key, value in self.overrides.items():
            final[key] = value

        # Inject distributed environment variables
        # This ensures Megatron uses the correct distributed settings
        dist_env = get_torchrun_env()
        final["world_size"] = dist_env["world_size"]
        final["rank"] = dist_env["rank"]
        final["local_rank"] = dist_env["local_rank"]

        # Convert to namespace (Megatron-LM expects argparse Namespace-like object)
        return SimpleNamespace(**final)

    # Alias for usage style:
    # builder.finalize()
    finalize = to_namespace
