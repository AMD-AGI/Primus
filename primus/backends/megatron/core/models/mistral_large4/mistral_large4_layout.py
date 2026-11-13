###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 per-layer layout rules, shared by the config, the layer specs, the converter
and the patches. Pure Python (no Megatron import): the patches load before Megatron is on the
path."""

from __future__ import annotations

import ast
import importlib.util
import os
from typing import List

__all__ = [
    "DEFAULT_FIRST_K_DENSE_REPLACE",
    "experts_on_turbo_grouped_gemm",
    "is_mistral_large4_args",
    "kv_carrier_rows",
    "turbo_enabled",
    "MODEL_TYPE",
    "LAYER_LISTS",
    "LIST_KINDS",
    "check_kv_sources",
    "check_layer_list",
    "hf_layer_lists",
    "kv_carrier_needed",
    "kv_sharing_active",
    "moe_layer_freq_list",
    "normalize_layer_list",
    "text_config",
]

# the Primus model_type of this model (presets, patches, model provider)
MODEL_TYPE = "mistral_large4"


def is_mistral_large4_args(args) -> bool:
    """This model's run, from an args / params namespace (the Primus ``model_type`` key). One check
    for the setup / before_train patches, which read the module params (``get_args(ctx)``); the
    build_args patches pass those params through ``_param`` the same way."""
    return getattr(args, "model_type", None) == MODEL_TYPE


# Megatron's tracker file: a ``load`` / ``pretrained_checkpoint`` directory without it holds no
# checkpoint (Megatron reads either only through it; a path to ``release/`` or ``iter_N/`` has none)
MEGATRON_TRACKER = "latest_checkpointed_iteration.txt"


def holds_megatron_checkpoint(path) -> bool:
    """``path`` is a directory with Megatron's tracker file. Without it Megatron only warns and
    starts from random weights: the load == save auto-resume pattern on its first run, a mistyped
    or stale path, a ``release/`` or ``iter_N/`` subdirectory, or the empty output directory of a
    failed conversion."""
    return bool(path) and os.path.isfile(os.path.join(str(path), MEGATRON_TRACKER))


def turbo_enabled(cfg) -> bool:
    """Primus-Turbo modules may be used: ``enable_primus_turbo`` set and ``primus_turbo``
    installed, the gate of the shared ``megatron.turbo.*`` patches (``is_primus_turbo_can_patch``).
    Without it the layer specs read every ``use_turbo_*`` switch as off."""
    return (
        bool(getattr(cfg, "enable_primus_turbo", False))
        and importlib.util.find_spec("primus_turbo") is not None
    )


def experts_on_turbo_grouped_gemm(cfg) -> bool:
    """Whether the layer specs build the routed experts on Primus-Turbo's grouped GEMM
    (``PrimusGroupedMLP``): Primus-Turbo enabled (``turbo_enabled``), ``use_turbo_grouped_gemm`` and
    ``moe_grouped_gemm``. Otherwise they run on TE GEMMs (``SequentialMLP``, or TE's grouped GEMM)."""
    return (
        turbo_enabled(cfg)
        and bool(getattr(cfg, "use_turbo_grouped_gemm", False))
        and bool(getattr(cfg, "moe_grouped_gemm", False))
    )


# dense MLP on the first layers, then MoE (the release: 1)
DEFAULT_FIRST_K_DENSE_REPLACE = 1


def kv_carrier_rows(seq_len: int, kv_lora_rank: int, hidden_size: int) -> int:
    """Rows the shared KV latent adds to the pipeline P2P tensor: the hidden states ``[s, b, h]``
    are followed by each sample's latent ``[s, kv_lora_rank]`` packed densely into
    ``ceil(s * kv_lora_rank / h)`` rows of width ``h`` (the last one zero-padded). The decoder
    block (sender and receiver) and the PP shape patch (buffer size) both use this one rule."""
    s, r, h = int(seq_len), int(kv_lora_rank), int(hidden_size)
    return -(-s * r // h)


# the per-layer lists of a release config.json
LAYER_LISTS = ("no_rope_layers", "indexer_types", "kv_source_layers")
# a config.json that omits a list means the HF config's default for it
_HF_LIST_DEFAULTS = {
    "no_rope_layers": lambda n: [1] * n,  # every layer applies rope
    "kv_source_layers": lambda n: list(range(n)),  # every layer its own KV
    "indexer_types": lambda n: ["full"] * n,  # every layer its own indexer
}


# entry type and allowed values of each per-layer list (None: any value of that type)
LIST_KINDS = {
    "no_rope_layers": (int, (0, 1)),
    "indexer_types": (str, ("full", "shared")),
    "kv_source_layers": (int, None),
}


def check_layer_list(name: str, value: List) -> List:
    """``value`` (already a list of the right type) has only the values ``name`` allows."""
    allowed = LIST_KINDS[name][1]
    if allowed is not None:
        for i, v in enumerate(value):
            if v not in allowed:
                raise ValueError(f"{name}[{i}] = {v!r}; expected one of {list(allowed)}")
    return value


def check_kv_sources(kv_source_layers: List[int], no_rope_layers: List[int] = None) -> List[int]:
    """``kv_source_layers`` follows the KV-sharing rule: layer ``i`` reuses the KV of itself or of an
    earlier layer ``s`` (``0 <= s <= i``) that owns its KV (``kv_source_layers[s] == s``), and, with
    ``no_rope_layers``, of a layer of the same rope / NoPE kind. Raises ValueError otherwise. One
    rule for the model config, the converter and the patches (the PP wire size depends on it)."""
    for i, s in enumerate(kv_source_layers):
        if s < 0 or s > i or kv_source_layers[s] != s:
            owner = kv_source_layers[s] if 0 <= s <= i else "out of range"
            raise ValueError(
                f"kv_source_layers[{i}] = {s} breaks the rule 0 <= s <= {i} and kv_source_layers[s] == s "
                f"(a layer reuses only the KV of itself or of an earlier layer that owns its KV); "
                f"kv_source_layers[{s}] = {owner}"
            )
        if no_rope_layers is not None and s != i and no_rope_layers[s] != no_rope_layers[i]:
            raise ValueError(f"layer {i} reuses layer {s}'s KV but their rope/NoPE kinds differ")
    return kv_source_layers


def hf_layer_lists(t: dict):
    """``(lists, defaulted)``: the per-layer lists of the text config ``t``, each the config's own
    or, when it omits it, the HF default for ``num_hidden_layers`` layers (no NoPE layers, no
    sharing); ``defaulted`` names the lists that took the default. One policy for the converter
    and the training-side patch."""
    n = int(t["num_hidden_layers"])
    lists, defaulted = {}, []
    for k in LAYER_LISTS:
        if t.get(k) is not None:
            # the same parsing and checks as the model config (type, length >= num_hidden_layers,
            # allowed values): a string is a list literal, not a sequence of characters
            lists[k] = check_layer_list(k, normalize_layer_list(t[k], n, k, LIST_KINDS[k][0]))
        else:
            lists[k] = _HF_LIST_DEFAULTS[k](n)
            defaulted.append(k)
    check_kv_sources(lists["kv_source_layers"], lists["no_rope_layers"])
    return lists, defaulted


def text_config(cfg: dict) -> dict:
    """The text decoder's settings of a release ``config.json``: ``text_config`` or the top level."""
    return cfg.get("text_config") or cfg


def normalize_layer_list(value, num_layers: int, name: str, kind=int, default=None) -> List:
    """A per-layer list from a list, a python-literal string, or None (``default`` per layer).

    A longer list is cut to ``num_layers`` (a depth-reduced run takes the release's first
    layers); a shorter one, or anything that is not a list, raises ValueError naming ``name``."""
    if value is None:
        return [default(i) if callable(default) else default for i in range(num_layers)]
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"{name}: {value!r} is not a list literal") from e
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name}: expected a per-layer list, got {value!r}")
    if not all(isinstance(v, kind) and not isinstance(v, bool) for v in value):
        raise ValueError(f"{name}: entries of {value!r} are not all {kind.__name__}")
    value = list(value)
    if len(value) < num_layers:
        raise ValueError(f"{name} has {len(value)} entries, fewer than num_layers={num_layers}")
    return value[:num_layers]


def moe_layer_freq_list(first_k_dense_replace, num_layers: int, num_experts) -> List[int]:
    """Per-layer MoE pattern (1 = MoE, 0 = dense): the first ``first_k_dense_replace`` layers
    dense (None: the release's 1), the rest MoE; all dense without experts (``num_experts`` None
    or 0). The one rule the preset patch, the layer specs and the converter share."""
    n = int(num_layers)
    if not int(num_experts or 0):
        return [0] * n
    k = DEFAULT_FIRST_K_DENSE_REPLACE if first_k_dense_replace is None else int(first_k_dense_replace)
    if k < 0:
        raise ValueError(f"first_k_dense_replace must be >= 0, got {k}")
    k = min(k, n)
    return [0] * k + [1] * (n - k)


def kv_sharing_active(kv_source_layers, num_layers: int, no_rope_layers=None) -> bool:
    """Whether any of the first ``num_layers`` layers reuses another layer's latent KV. The list is
    checked by ``check_kv_sources``, with the rope / NoPE kind rule when ``no_rope_layers`` is given
    (as the model config checks it)."""
    if kv_source_layers is None:
        return False
    n = int(num_layers)
    nope = None if no_rope_layers is None else normalize_layer_list(no_rope_layers, n, "no_rope_layers", int)
    srcs = check_kv_sources(normalize_layer_list(kv_source_layers, n, "kv_source_layers", int), nope)
    return any(s != i for i, s in enumerate(srcs))


def kv_carrier_needed(
    kv_source_layers, num_layers: int, pipeline_model_parallel_size, no_rope_layers=None
) -> bool:
    """Whether the pipeline P2P tensor carries the shared KV latent: ``[s + kv_carrier_rows(s, r, h), b, h]``.

    The decoder block (sender and receiver of the latent) and the PP shape patch (the size of
    the receive buffer) both decide with this one function, both passing ``no_rope_layers`` so the
    lists are checked by the model config's rule.
    """
    return int(pipeline_model_parallel_size or 1) > 1 and kv_sharing_active(
        kv_source_layers, num_layers, no_rope_layers
    )
