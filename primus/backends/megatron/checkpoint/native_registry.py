###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Registry of the model families Primus' native (bridge-free) HF -> Megatron
converter supports.

The post-train conversion hook (``01_convert_checkpoints.py``) and
``native_convert_common`` read everything family-specific from here: which HF
``config.json`` names a family, which module converts it, the family's native
options and their defaults, the dtype fallback, the ``model_type`` its mcore
model is built with, and optional hook-side steps. Error and help messages list
the registered families.

Adding a family
---------------
A model port adds two files and edits none of the shared ones:

1. ``native_loader_<family>.py``: the converter. It defines
   ``convert(hf_dir, save_dir, *, dtype, tensor_parallel_size,
   pipeline_parallel_size, expert_parallel_size, ckpt_format, **options)``
   like ``native_loader_llama.convert``.
2. ``native_families/<family>.py``: its registration, one call::

       from primus.backends.megatron.checkpoint.native_registry import (
           NativeFamily, NativeOption, register_native_family,
       )

       register_native_family(
           NativeFamily(
               name="olmo3",
               label="OLMo-3",
               converter="primus.backends.megatron.checkpoint.native_loader_olmo3",
               architectures=("Olmo3ForCausalLM",),
               model_types=("olmo3",),
               model_type="olmo3",
               prepare="primus.backends.megatron.checkpoint.native_loader_olmo3:hook_prepare",
           )
       )

The hook imports every module of the ``native_families`` package
(:func:`load_family_modules`) before it looks a model up, so the new file is
found without a list to extend. A registration module must stay light: it
imports this module only, and names the converter and its hooks as
``"module"`` / ``"module:function"`` strings that are imported only when that
family converts. (Loader modules import torch and Primus model code, which must
not run for every fine-tune or before the convert-phase patches.)

The built-in families (Qwen3, DeepSeek, Llama) are registered below, in this
module.
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union

FAMILY_PACKAGE = "primus.backends.megatron.checkpoint.native_families"

# Raw HF dtype strings -> the converter's dtype names. Matched by substring, in
# this order ("bfloat16" before "float16"), as the hook always did.
_DTYPE_NAMES = (("bfloat16", "bf16"), ("float16", "fp16"), ("float32", "fp32"))
DEFAULT_DTYPE = "bf16"


class NativeConvertRefused(Exception):
    """Raised by a family's hook-side step with a reason for the user.

    The conversion hook prints the message and exits with status 1, before
    anything is converted or reused.
    """


@dataclass(frozen=True)
class NativeOption:
    """A per-family conversion option read from the experiment config.

    ``key`` is looked up in the ``sft_trainer`` / ``post_trainer`` module's
    params, then on the module itself (as the hook reads ``native_ckpt_*``).
    The value, or ``default`` when the config does not set it, is passed to the
    converter's ``convert()`` as ``kwarg``. With ``default=None`` and the key
    unset, ``kwarg`` is not passed, so ``convert()``'s own default applies. An
    explicit ``0`` / ``False`` in the config is a value, not "unset".
    """

    key: str
    kwarg: str
    default: Any = None
    help: str = ""


@dataclass
class ConvertContext:
    """What a family's ``prepare`` step sees.

    ``experiment`` is the fine-tune's layout and checkpoint format
    (``{"tp", "pp", "ep", "ckpt_format"}``). ``options`` are the family's
    resolved :class:`NativeOption` values, by ``kwarg``. ``argv`` is the hook's
    command line after ``--config`` (the training overrides), ``[]`` when not
    known. ``config_value(name)`` reads a key of the experiment's
    ``sft_trainer`` / ``post_trainer`` module (params first), ``None`` when unset
    or when no config file is known.
    """

    hf_path: str
    hf_config: dict
    family: "NativeFamily"
    experiment: dict
    options: dict
    argv: list = field(default_factory=list)
    config_value: Callable[[str], Any] = field(default=lambda name: None)


@dataclass
class ConvertPlan:
    """What a family's ``prepare`` step returns (every field optional).

    * ``convert_kwargs``: extra keywords for ``convert()``.
    * ``experiment``: replaces the experiment settings the hook checks the
      checkpoint format against (e.g. after reading command-line overrides).
    * ``path_suffix``: appended to the default output directory name (not to
      ``native_ckpt_out``), so conversions that differ get their own directory.
    * ``check_existing(path) -> str | None``: called before an existing
      checkpoint of the right format is reused; a string is the reason to refuse
      it (the hook exits with it).
    * ``before_convert(path)``: called on the converting node before the
      conversion (e.g. to record the settings the checkpoint is converted with).
    """

    convert_kwargs: dict = field(default_factory=dict)
    experiment: dict | None = None
    path_suffix: str = ""
    check_existing: Callable[[Path], str | None] | None = None
    before_convert: Callable[[Path], None] | None = None


Hook = Union[str, Callable, None]


@dataclass(frozen=True)
class NativeFamily:
    """One family of the native converter.

    * ``name``: the family id (``opts["family"]``, log lines).
    * ``label``: how messages and help list it, e.g. ``"Llama 2/3/3.1"``.
    * ``converter``: module that defines ``convert()``, imported when this
      family converts.
    * ``architectures`` / ``model_types``: HF ``config.json`` names that select
      the family, matched case-insensitively. Architectures are checked first.
    * ``model_type_needs_no_architectures``: the ``model_types`` select the
      family only when ``config.json`` lists no architectures (a model_type
      shared with task heads the converter does not build).
    * ``options``: :class:`NativeOption` values passed to ``convert()``.
    * ``model_type``: the provider ``build_mcore_gpt_model(margs, family=name)``
      builds (``get_model_provider(model_type=...)``).
    * ``dtype_from_config(cfg) -> str | None``: the raw HF dtype string when
      ``torch_dtype`` is absent (e.g. a top-level ``dtype``); default: none.
    * ``prepare(ctx: ConvertContext) -> ConvertPlan | None``: hook-side step
      before the output path is chosen, on every node; raise
      :class:`NativeConvertRefused` to stop with a reason.
    * ``wait(megatron_path, done_file, lock_file, node_rank)``: how a node
      other than node 0 waits for the conversion; default: the hook's
      ``wait_for_conversion``.

    Hooks are callables or ``"module:function"`` strings (imported on use).
    """

    name: str
    label: str
    converter: str
    architectures: tuple = ()
    model_types: tuple = ()
    model_type_needs_no_architectures: bool = False
    options: tuple = ()
    model_type: str = "gpt"
    dtype_from_config: Hook = None
    prepare: Hook = None
    wait: Hook = None

    def converter_module(self):
        return importlib.import_module(self.converter)

    def hook(self, which: str) -> Callable | None:
        """The ``which`` hook (``prepare`` / ``wait`` / ``dtype_from_config``) as a callable."""
        return resolve_hook(getattr(self, which))


# name -> NativeFamily, in registration order (messages list them in this order)
_FAMILIES: dict = {}
# lower-cased architecture or model_type -> family name. One table for both, as
# the hook's ``_NATIVE_FAMILIES`` was: an architecture is looked up among the
# model types too, and the other way round.
_NAMES: dict = {}
# family modules that failed to import: module name -> error text
_LOAD_ERRORS: dict = {}
_loaded = False


def resolve_hook(hook: Hook) -> Callable | None:
    """A hook given as a callable, or as ``"module:function"`` (imported now)."""
    if hook is None or callable(hook):
        return hook
    module, sep, attr = str(hook).partition(":")
    if not sep or not module or not attr:
        raise ValueError(f"native family hook {hook!r} is not 'module:function'")
    return getattr(importlib.import_module(module), attr)


def register_native_family(family: NativeFamily) -> NativeFamily:
    """Register ``family``; returns it.

    Registering the same family again (an equal entry, e.g. a module imported
    twice) is a no-op. A different family under a registered name, or an
    architecture / model_type another family already claims, raises
    ``ValueError``: one HF config must select one converter.
    """
    if not family.name or not family.converter:
        raise ValueError("a native family needs a name and a converter module")
    existing = _FAMILIES.get(family.name)
    if existing is not None:
        if existing == family:
            return existing
        raise ValueError(f"native family {family.name!r} is already registered")
    names = [str(n).lower() for n in tuple(family.architectures) + tuple(family.model_types)]
    for n in names:
        if n in _NAMES and _NAMES[n] != family.name:
            raise ValueError(f"native family {family.name!r}: {n!r} already selects family {_NAMES[n]!r}")
    for opt in family.options:
        if not isinstance(opt, NativeOption):
            raise TypeError(f"native family {family.name!r}: options must be NativeOption, got {opt!r}")
    _FAMILIES[family.name] = family
    for n in names:
        _NAMES[n] = family.name
    return family


def unregister_native_family(name: str) -> None:
    """Remove a family (for tests)."""
    family = _FAMILIES.pop(name, None)
    if family is None:
        return
    for key in [k for k, v in _NAMES.items() if v == name]:
        del _NAMES[key]


def load_family_modules() -> dict:
    """Import every module of the ``native_families`` package (once).

    Each module registers its family when imported. A module that fails to
    import is reported on stderr and skipped, so one broken registration does
    not stop the other families; its error is listed in the unknown-family
    message. Returns the import errors by module name.
    """
    global _loaded
    if _loaded:
        return dict(_LOAD_ERRORS)
    _loaded = True
    try:
        package = importlib.import_module(FAMILY_PACKAGE)
    except ImportError as e:  # the package itself is missing: built-ins only
        _LOAD_ERRORS[FAMILY_PACKAGE] = repr(e)
        return dict(_LOAD_ERRORS)
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module = f"{FAMILY_PACKAGE}.{info.name}"
        try:
            importlib.import_module(module)
        except Exception as e:  # noqa: BLE001 - report and keep the other families
            _LOAD_ERRORS[module] = f"{type(e).__name__}: {e}"
            print(f"[WARNING] native converter family module {module} failed to import: {e}", file=sys.stderr)
    return dict(_LOAD_ERRORS)


def families() -> list:
    """The registered families, built-ins first, then in discovery order."""
    load_family_modules()
    return list(_FAMILIES.values())


def get_family(name: str) -> NativeFamily:
    """The family registered as ``name``; ``ValueError`` naming the registered ones otherwise."""
    load_family_modules()
    family = _FAMILIES.get(name)
    if family is None:
        raise ValueError(f"no native converter family {name!r} ({supported_text()})")
    return family


def supported_text() -> str:
    """``"supported: Qwen3 dense/MoE, DeepSeek-V2/V3, ..."`` from the registry."""
    return "supported: " + ", ".join(f.label for f in families())


def unknown_model_text(hf_path: str) -> str:
    """Why no native converter matches the model at ``hf_path``, with the supported families."""
    text = f"no native converter matches the HF model at {hf_path} ({supported_text()})"
    if _LOAD_ERRORS:
        text += "; family modules that failed to import: " + ", ".join(sorted(_LOAD_ERRORS))
    return text


def read_config(hf_path: str) -> dict | None:
    cfg_file = os.path.join(hf_path, "config.json")
    if not os.path.isfile(cfg_file):
        return None
    with open(cfg_file) as f:
        return json.load(f)


def family_for_config(cfg: dict) -> NativeFamily | None:
    """The family an HF ``config.json`` dict selects, or None.

    An architecture of a registered family wins (first match in the list);
    else the ``model_type``, unless that family's model types count only for
    configs without architectures. A single architecture written as a string is
    one name.
    """
    load_family_modules()
    architectures = cfg.get("architectures", []) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    for arch in architectures:
        name = _NAMES.get(str(arch).lower())
        if name:
            return _FAMILIES[name]
    name = _NAMES.get(str(cfg.get("model_type", "")).lower())
    if not name:
        return None
    family = _FAMILIES[name]
    if architectures and family.model_type_needs_no_architectures:
        return None
    return family


def detect_family(hf_path: str) -> NativeFamily | None:
    """The family of the HF model directory ``hf_path`` (its ``config.json``), or None."""
    cfg = read_config(hf_path)
    return None if cfg is None else family_for_config(cfg)


def dtype_name(raw: str) -> str | None:
    """``"bf16"`` / ``"fp16"`` / ``"fp32"`` for an HF dtype string, None when it names none."""
    raw = str(raw or "").lower()
    for needle, name in _DTYPE_NAMES:
        if needle in raw:
            return name
    return None


def dtype_from_hf_config(hf_path: str, override: str | None, family: NativeFamily | None = None) -> str:
    """The dtype to convert in: ``override``, else the HF config's ``torch_dtype``,
    else the family's ``dtype_from_config`` (only when ``torch_dtype`` is absent),
    else ``bf16``."""
    if override:
        return override
    cfg = read_config(hf_path)
    if cfg is None:
        return DEFAULT_DTYPE
    raw = str(cfg.get("torch_dtype", "")).lower()
    if (not raw or raw == "none") and family is not None:
        fallback = family.hook("dtype_from_config")
        if fallback is not None:
            raw = str(fallback(cfg) or "").lower()
    return dtype_name(raw) or DEFAULT_DTYPE


def read_options(family: NativeFamily, config_value: Callable[[str], Any]) -> dict:
    """``family``'s option values by ``convert()`` keyword, from ``config_value(key)``."""
    values = {}
    for opt in family.options:
        value = config_value(opt.key)
        if value is None or value == "":
            value = opt.default
        if value is not None:
            values[opt.kwarg] = value
    return values


def model_type_for(family_name: str | None) -> str:
    """The ``get_model_provider`` model_type of ``family_name`` (``"gpt"`` when None)."""
    if family_name is None:
        return "gpt"
    return get_family(family_name).model_type


# ---------------------------------------------------------------------------
# Built-in families. Same names, labels and dispatch as before the registry.
# ---------------------------------------------------------------------------
_CHECKPOINT = "primus.backends.megatron.checkpoint"

register_native_family(
    NativeFamily(
        name="qwen3",
        label="Qwen3 dense/MoE",
        converter=f"{_CHECKPOINT}.native_loader_qwen3",
        architectures=("Qwen3ForCausalLM", "Qwen3MoeForCausalLM"),
        model_types=("qwen3", "qwen3_moe"),
    )
)
register_native_family(
    NativeFamily(
        name="deepseek",
        label="DeepSeek-V2/V3",
        converter=f"{_CHECKPOINT}.native_loader_deepseek",
        architectures=("DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"),
        model_types=("deepseek_v2", "deepseek_v3", "deepseek"),
    )
)
register_native_family(
    NativeFamily(
        name="llama",
        label="Llama 2/3/3.1",
        converter=f"{_CHECKPOINT}.native_loader_llama",
        architectures=("LlamaForCausalLM",),
        model_types=("llama",),
    )
)


def print_families(out=sys.stdout) -> None:
    """Print the registered families, their HF names and options (the module's CLI)."""
    for fam in families():
        names = ", ".join(tuple(fam.architectures) + tuple(fam.model_types))
        print(f"{fam.name:12s} {fam.label}  ({names})", file=out)
        for opt in fam.options:
            print(
                f"{'':12s}   {opt.key} -> convert({opt.kwarg}=...), default {opt.default!r}  {opt.help}",
                file=out,
            )
    for module, error in sorted(_LOAD_ERRORS.items()):
        print(f"failed to import {module}: {error}", file=sys.stderr)


if __name__ == "__main__":  # python -m primus.backends.megatron.checkpoint.native_registry
    # Family modules register into the imported module, not into this __main__ copy.
    from primus.backends.megatron.checkpoint import native_registry

    native_registry.print_families()
