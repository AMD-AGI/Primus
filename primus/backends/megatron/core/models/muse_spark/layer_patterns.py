###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark per-layer 0/1 patterns (``window_attn_skip_freq`` / ``no_rope_freq``).

One parser for the config and the ``build_args`` patch: ``parse_layer_pattern``
(``muse_spark/layer_freq.py``, shared with the FLOPs patches). It reads an int, a list, or a
Python-list expression string such as ``"([1,0,1,1]*15+[1,0])"`` without ``eval``. An
int means "every N-th layer" (Megatron's meaning) and must be >= 1.

A longer list is cut to ``num_layers`` on purpose: a depth-reduced build (proxy,
trainer test) keeps the first layers of the released model, so it keeps the first
pattern entries. The cut is reported through the ``warn`` callback (the ``build_args``
patch logs it on rank 0), so a wrong ``num_layers`` (e.g. 60 for the 62-layer release)
is visible in the log; the checkpoint converter derives ``num_layers`` from the HF
config, so its patterns always match exactly. A shorter list is an error, and an int
must be in ``[1, num_layers]`` (a larger N skips no layer). Every error names the
argument. Pure Python (no torch / Megatron).
"""

import math
import re

# re-exported: the one eval-free pattern parser lives in the shared light helper
from primus.backends.megatron.core.models.muse_spark.layer_freq import (
    parse_layer_pattern,
)
from primus.backends.megatron.core.models.muse_spark.number_utils import (
    parse_positive_int,
    require_positive_int,
)

__all__ = [
    "MODEL_TYPE",
    "parse_layer_pattern",
    "normalize_layer_pattern",
    "check_all_moe_layers",
    "parse_window_size",
    "hf_sliding_window",
    "MAKE_VOCAB_SIZE_DIVISIBLE_BY",
    "HF_OUTPUT_MULTIPLIER",
    "vocab_size_with_padding",
]

# The Primus model_type (one value: the trainer, get_model_provider and the patch condition
# compare against it; ``muse_spark_text`` is only the HF text_config's model_type, which the
# checkpoint hook maps to this family).
MODEL_TYPE = "muse_spark"

# Muse Spark pads the vocab to a multiple of 64 (202048 = 64 * 3157) at every TP;
# args.make_vocab_size_divisible_by is not read (Megatron's default is 128).
MAKE_VOCAB_SIZE_DIVISIBLE_BY = 64

# HF ``output_multiplier`` (1/sqrt(32)): HF applies the one value to the router logits and to
# the lm-head logits. The config field defaults and the converter's defaults both use it.
HF_OUTPUT_MULTIPLIER = 1 / math.sqrt(32)


def _vocab_padding_int(value, name, default=None):
    """For :func:`vocab_size_with_padding` only (its name is in the error): ``value`` as an
    int >= 1, parsed by the shared config-number parser
    (:func:`number_utils.require_positive_int` / ``parse_positive_int``); with a ``default``, None or a
    blank string is that default. A bool, a non-integral number (e.g. 202047.9), None
    (without a default), text or a value < 1 raises ValueError naming ``name``."""
    error = f"vocab_size_with_padding: {name}={value!r} must be an integer >= 1"
    if default is not None:
        return parse_positive_int(value, default, error)
    return require_positive_int(value, error)


def vocab_size_with_padding(vocab_size, divisible_by=MAKE_VOCAB_SIZE_DIVISIBLE_BY, tp=1):
    """The embedding / output-layer rows: ``vocab_size`` rounded up to ``divisible_by``.

    The training ``build_args`` patch and the checkpoint converter both use this, so a
    converted checkpoint has the training shape. Unlike Megatron's own padding
    (``divisible_by * TP``), TP is not a factor: the rows depend only on the model config,
    so one checkpoint loads at every TP. The vocab-parallel layers need the rows to divide
    by TP; that is checked here (202048 rows divide by every TP that 128 heads / 16 query
    groups allow) and the error asks for a TP that divides the rows.
    """
    multiple = _vocab_padding_int(divisible_by, "divisible_by")
    vocab = _vocab_padding_int(vocab_size, "vocab_size")
    tp = _vocab_padding_int(tp, "tensor_model_parallel_size", default=1)
    rows = -(-vocab // multiple) * multiple
    if rows % tp:
        # the multiple is the model's fixed constant (args.make_vocab_size_divisible_by is
        # not read), so the only knob the user has is the TP size
        raise ValueError(
            f"padded vocab {rows} (vocab_size {vocab_size} rounded up to a multiple of {multiple}) "
            f"does not divide by tensor_model_parallel_size={tp}: choose a TP that divides {rows}"
        )
    return rows


def normalize_layer_pattern(value, num_layers, name="layer pattern", warn=None):
    """Parse ``value``, validate the whole pattern, then cut a list to ``num_layers`` entries.

    An int (every N-th layer) must be in ``[1, num_layers]`` (``num_layers`` unknown: >= 1).
    A list must be non-empty, every entry (also those past ``num_layers``) must be 0 or 1,
    and it must have at least ``num_layers`` entries; ``num_layers`` must be a positive int
    when ``value`` is a list. A cut calls ``warn(message)`` when ``warn`` is given. Every
    ValueError names ``name``.
    """
    source = value
    try:
        value = parse_layer_pattern(value)
    except ValueError as e:
        raise ValueError(f"{name}: {e}") from None
    layers_ok = not isinstance(num_layers, bool) and isinstance(num_layers, int) and num_layers >= 1
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{name}: an int means every N-th layer and must be >= 1, got {value}")
        if layers_ok and value > num_layers:
            raise ValueError(
                f"{name}: every {value}-th layer is outside num_layers={num_layers} "
                f"(the int must be in [1, {num_layers}])"
            )
    if isinstance(value, list):
        if not value:
            shown = f" ({source!r} has 0 entries)" if isinstance(source, str) else ""
            raise ValueError(f"{name} is an empty list{shown}; it needs num_layers={num_layers} entries")
        if any(v not in (0, 1) for v in value):
            raise ValueError(f"{name} entries must be 0 or 1, got {sorted(set(value))}")
        if not layers_ok:
            raise ValueError(f"{name}: a per-layer list needs num_layers >= 1, got {num_layers!r}")
        if len(value) < num_layers:
            raise ValueError(f"{name} has {len(value)} entries, fewer than num_layers={num_layers}")
        if len(value) > num_layers:
            if warn is not None:
                warn(
                    f"{name}: the pattern has {len(value)} entries; num_layers={num_layers} keeps "
                    f"the first {num_layers} (a depth-reduced build). Check num_layers if this "
                    f"is not intended."
                )
            value = value[:num_layers]
    return value


def check_all_moe_layers(moe_layer_freq, num_layers):
    """Muse Spark has a MoE MLP in every layer: ``moe_layer_freq`` must mean "all layers".

    The layer spec builds every layer as MoE (the released model has no dense layer), so
    any other value would be ignored by the model while FLOPs / memory estimates used it.
    Accepts None (unset), 1, or a list (or list string) of all ones. Raises ValueError otherwise.
    """
    value = normalize_layer_pattern(moe_layer_freq, num_layers, "moe_layer_freq")
    if value is None or value == 1 or (isinstance(value, list) and all(v == 1 for v in value)):
        return
    raise ValueError(
        f"Muse Spark builds a MoE MLP in every layer: moe_layer_freq must be 1 "
        f"(or all ones), got {moe_layer_freq!r}"
    )


_WINDOW_RE = re.compile(
    r"\s*(?:\(\s*({p})\s*\)|\[\s*({p})\s*\]|({p}))\s*".format(p=r"[-+]?\d+\s*,\s*[-+]?\d+")
)


def parse_window_size(value):
    """``window_size`` as a causal pair ``(left, 0)`` of ints, or None.

    Accepts None, a 2-element list / tuple of ints, or a string of exactly one pair:
    ``"(2048, 0)"``, ``"[2048,0]"`` or ``"2048,0"`` (yaml / CLI). Checked: ``left`` is
    -1 (no limit) or >= 0, and ``right`` is 0 (Muse Spark attention is causal). Anything
    else (nesting, unbalanced brackets, empty or extra items, floats, bools) raises a
    ValueError that names ``window_size``. The config and the FLOPs patch both use it.
    """
    if value is None:
        return None
    if isinstance(value, str):
        m = _WINDOW_RE.fullmatch(value)
        if not m:
            raise ValueError(f"window_size must be one pair '(left, right)' of ints, got {value!r}")
        parts = [int(p) for p in next(g for g in m.groups() if g).split(",")]
    elif (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(p, int) and not isinstance(p, bool) for p in value)
    ):
        parts = list(value)
    else:
        raise ValueError(f"window_size must be a pair (left, right) of ints, got {value!r}")
    left, right = parts
    if left < -1:
        raise ValueError(f"window_size[0] must be -1 (no limit) or >= 0, got {left}")
    if right != 0:
        raise ValueError(f"Muse Spark attention is causal: window_size[1] must be 0, got {right}")
    return left, right


def hf_sliding_window(value):
    """HF ``sliding_window`` (keys incl. the query) from a ``window_size`` in the presets'
    HF convention: the parsed ``left``, or None for no window (None / ``left == -1``).
    ``left == 0`` raises (an HF window counts the query, so it is >= 1)."""
    ws = parse_window_size(value)
    if ws is None or ws[0] == -1:
        return None
    if ws[0] < 1:
        raise ValueError(f"window_size[0] is an HF sliding_window and must be >= 1 (or -1), got {ws[0]}")
    return ws[0]
