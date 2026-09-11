###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Translate Primus module params into a SpecForge command line.

SpecForge is configured by its own Hydra-style YAML plus dotted ``key=value``
overrides on the command line. Primus therefore does not rebuild SpecForge's
config tree; it only points at a SpecForge config and forwards overrides.
Offline capture is the same shape: Primus builds argv for SpecForge's
``scripts/prepare_hidden_states.py`` and does not reimplement capture.

Params consumed here:

    specforge_config      Path to the SpecForge YAML (required for train)
    specforge_overrides   Nested mapping flattened to dotted Hydra overrides
    specforge_entrypoint  argv[0] for the SpecForge CLI (default ``specforge``)
    specforge_root        SpecForge checkout used as cwd (see resolve_specforge_root)
    specforge_mode        ``train`` (default) or ``capture``
    specforge_capture     Nested mapping of ``prepare_hidden_states.py`` flags
    output_dir            Convenience alias for ``specforge_overrides.output_dir``
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

from primus.core.utils.yaml_utils import nested_namespace_to_dict

DEFAULT_ENTRYPOINT = "specforge"
DEFAULT_CAPTURE_SCRIPT = "scripts/prepare_hidden_states.py"
CAPTURE_STORE_TRUE = frozenset(
    {
        "trust_remote_code",
        "is_preformatted",
        "compress",
        "sglang_enable_nccl_nvls",
        "sglang_enable_symm_mem",
        "sglang_enable_torch_compile",
        "sglang_enable_dp_attention",
        "sglang_enable_dp_lm_head",
        "sglang_disable_radix_cache",
    }
)
CAPTURE_SKIP_KEYS = frozenset(
    {
        "nproc_per_node",
        "filter_output_path",
        "filter_block_size",
        "torchrun",
        "script",
    }
)


def specforge_mode(params: Any) -> str:
    raw = getattr(params, "specforge_mode", None) or "train"
    return str(raw).strip().lower()


def _as_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_as_override_value(v) for v in value) + "]"
    return str(value)


def flatten_overrides(obj: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a nested mapping into dotted Hydra keys.

    Keys that already contain dots are kept verbatim, so a config may mix
    ``training: {max_steps: 20}`` and ``training.max_steps: 20``.
    """

    if obj is None:
        return {}
    if isinstance(obj, SimpleNamespace):
        obj = nested_namespace_to_dict(obj)
    if not isinstance(obj, dict):
        return {prefix: _as_override_value(obj)} if prefix else {}

    flat: dict[str, str] = {}
    for key, value in obj.items():
        next_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, (dict, SimpleNamespace)):
            flat.update(flatten_overrides(value, next_prefix))
        else:
            flat[next_prefix] = _as_override_value(value)
    return flat


def build_specforge_argv(params: Any, extra_overrides: Optional[list[str]] = None) -> list[str]:
    """Build the ``specforge train`` argv for a Primus pre_trainer module."""

    specforge_config = getattr(params, "specforge_config", None)
    if not specforge_config:
        raise ValueError(
            "[Primus:specforge] 'specforge_config' is required; point it at a SpecForge YAML "
            "(e.g. a SpecForge examples/configs YAML)."
        )

    overrides = flatten_overrides(getattr(params, "specforge_overrides", None))

    output_dir = getattr(params, "output_dir", None)
    if output_dir and "output_dir" not in overrides:
        overrides["output_dir"] = str(output_dir)

    entrypoint = getattr(params, "specforge_entrypoint", None) or DEFAULT_ENTRYPOINT

    argv = [str(entrypoint), "train", "--config", str(specforge_config)]
    argv.extend(f"{key}={value}" for key, value in sorted(overrides.items()))
    if extra_overrides:
        argv.extend(extra_overrides)
    return argv


def _capture_flag_name(key: str) -> str:
    return "--" + str(key).replace("_", "-")


def _is_true_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_capture_argv(params: Any, extra_args: Optional[list[str]] = None) -> list[str]:
    """Build ``torchrun … scripts/prepare_hidden_states.py`` for offline capture."""

    capture = flatten_overrides(getattr(params, "specforge_capture", None))
    nproc = capture.get("nproc_per_node") or os.environ.get("NPROC_PER_NODE") or "1"
    torchrun = capture.get("torchrun") or "torchrun"
    script = capture.get("script") or DEFAULT_CAPTURE_SCRIPT

    argv = [
        str(torchrun),
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        str(script),
    ]
    for key, value in sorted(capture.items()):
        if key in CAPTURE_SKIP_KEYS:
            continue
        flag = _capture_flag_name(key)
        if key in CAPTURE_STORE_TRUE:
            if _is_true_flag(value):
                argv.append(flag)
            continue
        if value is None or str(value).lower() in {"none", "null", ""}:
            continue
        argv.extend([flag, str(value)])
    if extra_args:
        argv.extend(extra_args)
    return argv


def resolve_specforge_root(params: Any, env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """Directory to chdir into before exec'ing SpecForge.

    SpecForge resolves draft/model config paths relative to its own checkout,
    so running from an arbitrary cwd silently breaks config resolution.
    Resolution order: explicit param, ``SPECFORGE_ROOT``, then the nearest
    ancestor of the config that looks like a SpecForge checkout.
    """

    environ = os.environ if env is None else env

    explicit = getattr(params, "specforge_root", None)
    if explicit and Path(explicit).is_dir():
        return Path(explicit)

    from_env = environ.get("SPECFORGE_ROOT")
    if from_env and Path(from_env).is_dir():
        return Path(from_env)

    specforge_config = getattr(params, "specforge_config", None)
    if not specforge_config:
        return None
    config_path = Path(specforge_config)
    if not config_path.is_absolute():
        return None
    for parent in config_path.parents:
        looks_like_checkout = (parent / "configs").is_dir() and (parent / "pyproject.toml").is_file()
        if parent.name == "SpecForge" or looks_like_checkout:
            return parent
    return None
