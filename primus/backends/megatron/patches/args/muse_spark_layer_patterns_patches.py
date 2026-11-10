###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: per-layer pattern args as lists of ``num_layers`` entries.

The Muse Spark presets give ``no_rope_freq`` / ``window_attn_skip_freq`` as the
full 62-entry Python-list expression (e.g. ``"([0,1,0,0]*15+[0,1])"``). Megatron builds
a plain ``TransformerConfig`` from args in places outside the model builder
(e.g. ``is_dataset_built_on_rank``), and that config checks
``len(no_rope_freq) == num_layers``. Parse the strings (``ast``, no ``eval``; the same
parser as the config) and cut the lists to ``num_layers`` here, so a depth-reduced run (proxy, trainer test) works.
A cut is logged as a warning, and an int must be in ``[1, num_layers]``.

The same patch also checks two args the model reads:

* ``moe_layer_freq`` must make every layer MoE (``check_all_moe_layers``); the spec
  builds every layer as MoE, so another value would make the FLOPs / memory estimates wrong;
* ``final_logit_softcapping`` is None / 0 or a finite number > 0, written back as a float
  (``normalize_softcap``).
"""

import functools
from pathlib import Path

from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
    MAKE_VOCAB_SIZE_DIVISIBLE_BY,
    check_all_moe_layers,
    normalize_layer_pattern,
    vocab_size_with_padding,
)
from primus.backends.megatron.core.models.muse_spark.model_types import (
    backend_args_of,
    is_muse_spark_model_type,
)
from primus.backends.megatron.core.models.muse_spark.number_utils import parse_softcap
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_kv_rank_0, warning_rank_0


def padded_vocab_of(args):
    """The embedding / head rows for ``args`` (``vocab_size_with_padding``: a multiple of
    ``MAKE_VOCAB_SIZE_DIVISIBLE_BY`` (64), the same at every TP), or None when the args set
    no ``vocab_size``.

    The multiple is a Muse Spark model rule, the converter's: ``make_vocab_size_divisible_by``
    in the args is not read (the module config sets it to 128 for every model, so an
    experiment would have to repeat 64 to match the converted checkpoint). Set
    ``padded_vocab_size`` for other rows."""
    vocab = getattr(args, "vocab_size", None)
    if not vocab:
        return None
    return vocab_size_with_padding(
        vocab,
        MAKE_VOCAB_SIZE_DIVISIBLE_BY,
        getattr(args, "tensor_model_parallel_size", None) or 1,
    )


_LAYER_PATTERNS_PATCH = "megatron.args.muse_spark_layer_patterns"
_LAYER_PATTERNS_TAG = f"[Patch:{_LAYER_PATTERNS_PATCH}]"
_VOCAB_PATCH = "megatron.args.muse_spark_vocab"
_VOCAB_TAG = f"[Patch:{_VOCAB_PATCH}]"
_KEYS = ("no_rope_freq", "window_attn_skip_freq")
_TRAINER_BASE_YAML = Path(__file__).resolve().parents[4] / "configs/modules/megatron/trainer_base.yaml"


@functools.lru_cache(maxsize=1)
def module_vocab_divisible_by():
    """``make_vocab_size_divisible_by`` of the Megatron module config (trainer_base.yaml,
    which every run has): read from the file, so it cannot go stale; None when unreadable."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        return (yaml.safe_load(_TRAINER_BASE_YAML.read_text()) or {}).get("make_vocab_size_divisible_by")
    except (OSError, ValueError, AttributeError, yaml.YAMLError):  # YAMLError: malformed file
        return None


def ignored_vocab_multiple(args):
    """The ``make_vocab_size_divisible_by`` the user set that ``padded_vocab_of`` ignores,
    or None: the module config's default (not a user choice) and the model's multiple (64)
    are not reported."""
    value = getattr(args, "make_vocab_size_divisible_by", None)
    if value in (None, module_vocab_divisible_by(), MAKE_VOCAB_SIZE_DIVISIBLE_BY):
        return None
    return value


@register_patch(
    _LAYER_PATTERNS_PATCH,
    backend="megatron",
    phase="build_args",
    condition=is_muse_spark_model_type,
    description=(
        "Muse Spark: parse no_rope_freq / window_attn_skip_freq and cut them to num_layers; "
        "require an all-MoE moe_layer_freq; check final_logit_softcapping"
    ),
)
def patch_muse_spark_layer_patterns(ctx: PatchContext):
    args = backend_args_of(ctx)
    num_layers = getattr(args, "num_layers", None)

    def warn(msg):
        warning_rank_0(f"{_LAYER_PATTERNS_TAG} {msg}")

    for key in _KEYS:
        if not hasattr(args, key):
            continue
        value = normalize_layer_pattern(getattr(args, key), num_layers, key, warn=warn)
        setattr(args, key, value)
        log_kv_rank_0(f"{_LAYER_PATTERNS_TAG} -{key}", f"{value}")
    # every layer is MoE (the spec ignores moe_layer_freq); reject a value that would make
    # the FLOPs / memory estimates count a different number of MoE layers
    check_all_moe_layers(getattr(args, "moe_layer_freq", 1), num_layers)
    if hasattr(args, "final_logit_softcapping"):
        # write back the float (a yaml / CLI string such as "20" would reach the model as str)
        args.final_logit_softcapping = normalize_softcap(args.final_logit_softcapping)


def normalize_softcap(cap):
    """``final_logit_softcapping`` (a key every Megatron preset carries, null by default) is
    None / 0 (no cap) or a finite number > 0 (:func:`parse_softcap`, the check the model
    config's ``__post_init__`` runs too). Returns the cap as a float (None stays None)."""
    return parse_softcap(cap, _LAYER_PATTERNS_TAG)


@register_patch(
    _VOCAB_PATCH,
    backend="megatron",
    phase="build_args",
    condition=is_muse_spark_model_type,
    priority=60,  # after megatron.args.mock_data (50), which may switch to the NullTokenizer
    description="Muse Spark: embedding / head rows from the preset vocab_size (the checkpoint's 202048)",
)
def patch_muse_spark_vocab(ctx: PatchContext):
    """The presets set ``vocab_size: 202048`` (the HF text_config). Megatron sizes the
    embedding and head from the tokenizer instead (``len(tokenizer)`` for the HF tokenizer,
    ``vocab_size + 1`` for the mock-data NullTokenizer), padded to
    ``make_vocab_size_divisible_by * TP``, unless ``padded_vocab_size`` is already set. Set
    it here from ``vocab_size`` with ``vocab_size_with_padding`` and the model's multiple of
    64 (the converter's rule; ``make_vocab_size_divisible_by`` is not read): the rows depend
    only on the config, not on TP, so they match the converted checkpoint at every TP. An
    explicit ``padded_vocab_size`` is kept. When the rows are set here, a user-set
    ``make_vocab_size_divisible_by`` other than 64 or the module default 128 gets a warning
    that it has no effect.

    With mock data the NullTokenizer adds an EOD id equal to ``vocab_size``; ``vocab_size``
    is lowered (as ``megatron.args.mock_data`` does for a vocab read from the HF config)
    until that id fits in the rows, also when ``padded_vocab_size`` was set explicitly."""
    args = backend_args_of(ctx)
    padded = getattr(args, "padded_vocab_size", None)
    if padded is None:
        padded = padded_vocab_of(args)
        if padded is None:
            return
        args.padded_vocab_size = padded
        log_kv_rank_0(_VOCAB_TAG + " -padded_vocab_size", f"{padded}")
        ignored = ignored_vocab_multiple(args)
        if ignored is not None:
            warning_rank_0(
                f"{_VOCAB_TAG} make_vocab_size_divisible_by={ignored!r} is ignored: Muse Spark pads "
                f"vocab_size to a multiple of {MAKE_VOCAB_SIZE_DIVISIBLE_BY} (the converted "
                f"checkpoint's rows, {padded}); set padded_vocab_size for other rows"
            )
    vocab = getattr(args, "vocab_size", None)
    if getattr(args, "tokenizer_type", None) == "NullTokenizer" and vocab and int(vocab) >= int(padded):
        # ids 0..vocab_size (EOD) must stay below padded rows; the check makes a rerun a no-op
        args.vocab_size = int(padded) - 1
        log_kv_rank_0(_VOCAB_TAG + " -vocab_size (NullTokenizer)", f"{args.vocab_size}")
