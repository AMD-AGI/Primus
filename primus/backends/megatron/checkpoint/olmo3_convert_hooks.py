###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Posttrain-hook steps of the OLMo-3 native converter family (``native_families/olmo3.py``).

Torch-free: the conversion hook calls :func:`hook_prepare` on every node before it picks the
output path. It reads the fine-tune's own settings, so the converted weights (their padded
vocabulary) fit the model the run builds. ``seq_length`` and the RoPE layout are passed too: they
change no weight, only the args saved with the checkpoint, which a fine-tune (``finetune: true``)
does not read; the checkpoint is therefore keyed on the padded vocabulary alone, and a later run
with another ``seq_length`` or layout reuses it with its own args. A bad setting is refused
(``NativeConvertRefused``, naming the key) before anything is converted:

* ``convert_kwargs``: ``seq_length`` and ``rope_layout`` (``olmo3_rope_layout``) when the
  experiment sets them, and ``make_vocab_size_divisible_by`` times the run's TP (training pads
  the vocabulary to that multiple; the converter runs at TP = 1), plus the padded size the
  converter must reproduce (``expected_padded_vocab_size``);
* ``path_suffix``: ``_vocab<N>``, one default directory per padded vocabulary size (the only
  converted tensor shape that depends on the run; a fine-tune does not read the saved args);
* ``check_existing``: an existing checkpoint converted for another padding (or without the
  settings file) is refused instead of reused;
* ``before_convert``: the settings file is written before the conversion, so the tracker file
  (written last by the save) never exists without it.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from primus.backends.megatron.checkpoint.native_registry import (
    ConvertPlan,
    NativeConvertRefused,
)
from primus.core.utils.arg_utils import parse_cli_overrides

# experiment key -> native_loader_olmo3.convert keyword
CONVERT_KEYS = {
    "seq_length": "seq_length",
    "make_vocab_size_divisible_by": "make_vocab_size_divisible_by",
    "olmo3_rope_layout": "rope_layout",
}
# Megatron's default make_vocab_size_divisible_by
DEFAULT_VOCAB_DIVISOR = 128
# written next to the converted checkpoint, compared before it is reused
CONVERT_SETTINGS_FILE = "primus_native_convert_settings.json"


# the RoPE layouts a converted checkpoint may record (``auto`` is resolved by the converter only
# for tools that do not save; convert() refuses it)
ROPE_LAYOUTS = ("per_type", "yarn_all")


def _positive_int(name: str, value) -> int:
    """``value`` as a positive int, read as Megatron's ``type=int`` arguments are: an int, a
    decimal integer string (``"8192"``), or a whole-number float (YAML may load ``8192.0``; the
    CLI override parser turns ``"8192.0"`` into one). Anything else (``8192.5``, ``"1e4"``, a
    bool, inf) is refused with ``NativeConvertRefused`` naming ``name``."""
    number = None
    if isinstance(value, bool):
        number = None
    elif isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            number = int(value)
    elif isinstance(value, str) and re.fullmatch(r"\s*\+?[0-9]+\s*", value):
        number = int(value)
    if number is None or number <= 0:
        raise NativeConvertRefused(
            f"OLMo-3 conversion: {name} must be a positive whole number, got {value!r}"
        )
    return number


def convert_settings(config_value, tp=None) -> dict:
    """``native_loader_olmo3.convert`` keywords from the experiment (``config_value(name)``).

    Raises ``NativeConvertRefused`` naming the key for a value that is not a positive whole number
    (``seq_length``, ``make_vocab_size_divisible_by``, TP) or a RoPE layout other than
    ``per_type`` / ``yarn_all``. ``tp`` None (unset) is 1; any other value is checked (0 is refused).
    """
    kwargs = {}
    for name, kw in CONVERT_KEYS.items():
        value = config_value(name)
        if value is None or value == "":
            continue
        if kw == "rope_layout":
            if str(value) not in ROPE_LAYOUTS:
                raise NativeConvertRefused(
                    f"OLMo-3 conversion: olmo3_rope_layout must be one of {', '.join(ROPE_LAYOUTS)} "
                    f"for a converted checkpoint, got {value!r}"
                )
            kwargs[kw] = str(value)
        else:
            kwargs[kw] = _positive_int(name, value)
    tp = _positive_int("tensor_model_parallel_size", 1 if tp is None else tp)
    kwargs["make_vocab_size_divisible_by"] = (
        kwargs.get("make_vocab_size_divisible_by", DEFAULT_VOCAB_DIVISOR) * tp
    )
    return kwargs


def cache_key(vocab_size: int, settings: dict) -> dict:
    """Cache key ``{"padded_vocab_size": ceil(vocab / m) * m}`` with ``m`` =
    make_vocab_size_divisible_by * TP (already multiplied in ``settings``)."""
    m = int(settings.get("make_vocab_size_divisible_by", DEFAULT_VOCAB_DIVISOR))
    return {"padded_vocab_size": int(math.ceil(int(vocab_size) / m) * m)}


def path_suffix(key: dict) -> str:
    """The default output directory suffix, e.g. ``_vocab100352``."""
    return f"_vocab{key['padded_vocab_size']}"


class _BadSettingsFile(Exception):
    pass


def read_settings(megatron_path) -> dict | None:
    """The settings file of the checkpoint at ``megatron_path``: None when there is none;
    ``_BadSettingsFile`` when it is not a JSON object (e.g. truncated by a crashed conversion)."""
    path = Path(megatron_path) / CONVERT_SETTINGS_FILE
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            saved = json.load(f)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise _BadSettingsFile(f"has an unreadable converter settings file {path} ({e})") from e
    if not isinstance(saved, dict):
        raise _BadSettingsFile(f"has a converter settings file {path} that is not a JSON object")
    return saved


def write_settings(megatron_path, key: dict) -> None:
    path = Path(megatron_path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / CONVERT_SETTINGS_FILE, "w") as f:
        json.dump(key, f, sort_keys=True)


def check_existing(megatron_path, key: dict, multiple: int | None = None) -> str | None:
    """Why the existing checkpoint at ``megatron_path`` cannot be reused for ``key``, or None.
    ``multiple`` is the padding multiple (make_vocab_size_divisible_by * TP) behind ``key``."""
    try:
        saved = read_settings(megatron_path)
    except _BadSettingsFile as e:
        state = str(e)
    else:
        if saved == key:
            return None
        state = (
            f"was converted with settings {saved}"
            if saved is not None
            else "has no converter settings file (converted by an older Primus)"
        )
    why = "the vocabulary padded to a multiple of make_vocab_size_divisible_by * TP"
    if multiple is not None:
        why += f" = {multiple}"
    return (
        f"{megatron_path} {state}, but this run needs {key} ({why}). "
        "Remove it, or set native_ckpt_out to another directory."
    )


def dtype_from_config(cfg: dict) -> str | None:
    """The family's ``dtype_from_config`` step: transformers >= 4.56 saves ``dtype``, not
    ``torch_dtype`` (the released config.json has ``"dtype": "bfloat16"``)."""
    return cfg.get("dtype")


# experiment keys the hook's command line may override (``--key value`` / ``key=value``), the TP
# among them (``ctx.experiment["tp"]`` is read from the config file only)
_TP_KEY = "tensor_model_parallel_size"


def _argv_overrides(argv) -> dict:
    """The flat ``key -> value`` overrides on the hook's command line (the training overrides)."""
    try:
        parsed = parse_cli_overrides(list(argv or []))
    except Exception as e:  # noqa: BLE001 - a malformed command line is the user's to fix
        raise NativeConvertRefused(f"OLMo-3 conversion: cannot read the command-line overrides {argv}: {e}")
    # ``--tensor-model-parallel-size`` is Megatron's spelling of ``tensor_model_parallel_size``
    return {k.replace("-", "_"): v for k, v in parsed.items() if not isinstance(v, dict)}


def hook_prepare(ctx):
    """The family's ``prepare`` step (``native_registry.ConvertContext`` -> ``ConvertPlan``).

    The experiment's settings are read from its config file (``ctx.config_value``,
    ``ctx.experiment["tp"]``), overridden by the hook's command line (``ctx.argv``, e.g.
    ``--tensor_model_parallel_size 8 --make_vocab_size_divisible_by 256``), as training reads
    them. A TP given on the command line also replaces the experiment's ``tp``
    (``ConvertPlan.experiment``).
    """
    vocab = ctx.hf_config.get("vocab_size")
    if vocab is None:
        raise NativeConvertRefused(f"OLMo-3: {ctx.hf_path}/config.json has no vocab_size")
    vocab = _positive_int("config.json vocab_size", vocab)
    overrides = _argv_overrides(ctx.argv)

    def value(name):
        return overrides[name] if name in overrides else ctx.config_value(name)

    experiment = None
    tp = ctx.experiment.get("tp")
    if _TP_KEY in overrides:
        tp = _positive_int(_TP_KEY, overrides[_TP_KEY])
        experiment = {**ctx.experiment, "tp": tp}
    settings = convert_settings(value, tp=tp)
    multiple = settings["make_vocab_size_divisible_by"]
    key = cache_key(vocab, settings)
    return ConvertPlan(
        convert_kwargs={**settings, "expected_padded_vocab_size": key["padded_vocab_size"]},
        experiment=experiment,
        path_suffix=path_suffix(key),
        check_existing=lambda path: check_existing(path, key, multiple),
        before_convert=lambda path: write_settings(path, key),
    )
