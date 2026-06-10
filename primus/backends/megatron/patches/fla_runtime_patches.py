###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
FLA Runtime Knob Patch
======================

Resolves FLA runtime toggles onto the Megatron ``args`` namespace so that
every consumer — Primus-owned code AND Megatron-LM patches — can read a
single, typed ``args.*`` attribute instead of parsing env-var strings.

PRECEDENCE
----------
For every knob the resolved value is, in priority order:

    1. Pre-existing env var (highest — for ad-hoc overrides at launch time)
    2. YAML field in the EXP overrides block
    3. Documented default (matches pre-cleanup behaviour)

After resolution the value is written back onto ``args`` as a properly
typed attribute (bool / int / str).  Consumers never touch ``os.environ``.

CONFIGURATION
-------------
Add any of the following fields to the model YAML (e.g. ``mamba_base.yaml``
or the experiment ``overrides:`` block).  All are optional; unspecified
fields keep their documented default and the patch is a no-op for that
knob.

    # --- FLA Triton kernel toggles -------------------------------------------
    use_fla_fused_swiglu: true          # default true
    use_fla_fused_rmsnorm: false        # default false
    use_fla_fused_gated_norm: false     # default false  (same semantic scope
                                        #   as use_fla_fused_rmsnorm but kept
                                        #   as a separate knob for clarity)
    use_fla_short_conv: false           # default false

    # --- FLA dataset shim (deterministic FLA-order data) ---------------------
    use_fla_data: false                 # default false
    fla_cache_dir: ""                   # default ""

    # --- Fused cross-entropy from FLA ----------------------------------------
    fused_ce_mode: 1                    # 0 = vanilla Megatron CE
                                        # 1 = chunked FLA fused CE (default)
                                        # 2 = single-shot FLA fused CE
    fused_ce_chunks: 32                 # default 32

    # --- FLA MLA attention backend -------------------------------------------
    fla_mla_attn: ""                    # default unset / ""

TIMING
------
The patch runs at ``phase="before_train"`` — after
``train_runtime.py:merge_namespace`` has merged the Primus-only YAML keys
into ``args``, but before ``wrapped_pretrain()`` triggers model module
imports.  Consumers in model code read ``args.*`` in ``__init__`` or later,
which is always after this patch has run.
"""

from __future__ import annotations

import os
from typing import Any

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


# ─── Knob definitions ────────────────────────────────────────────────────────
#
# (yaml_field, legacy_env_var, typed_default)
#
# typed_default carries the type: bool → consumers see True/False,
# int → consumers see an int, str → consumers see a str.
#
# Legacy env var is checked FIRST (backward compat); if unset the YAML
# field value is used; if also null/missing the default applies.
# ──────────────────────────────────────────────────────────────────────────────

_FLA_RUNTIME_KNOBS: tuple = (
    ("use_fla_fused_swiglu",     "PRIMUS_FLA_SWIGLU",      True),
    ("use_fla_fused_rmsnorm",    "PRIMUS_FLA_NORM",        False),
    ("use_fla_fused_gated_norm", "PRIMUS_FLA_NORM",        False),
    ("use_fla_short_conv",       "PRIMUS_FLA_CONV",        False),
    ("use_fla_data",             "PRIMUS_FLA_DATA",        False),
    ("fla_cache_dir",            "PRIMUS_FLA_CACHE_DIR",   ""),
    ("fused_ce_mode",            "PRIMUS_FUSED_CE",        1),
    ("fused_ce_chunks",          "PRIMUS_FUSED_CE_CHUNKS", 32),
    ("fla_mla_attn",             "PRIMUS_FLA_MLA_ATTN",    ""),
)


def _env_to_typed(env_string: str, default: Any) -> Any:
    """Convert an env-var string to the type implied by *default*."""
    if isinstance(default, bool):
        return env_string not in ("0", "", "false", "False")
    if isinstance(default, int):
        return int(env_string)
    return env_string


def _has_any_fla_runtime_field(args) -> bool:
    """Cheap probe — skip the patch entirely when no FLA knob is configured
    AND no legacy env var is set."""
    for name, env_name, _default in _FLA_RUNTIME_KNOBS:
        if getattr(args, name, None) is not None:
            return True
        if env_name in os.environ:
            return True
    return False


@register_patch(
    "megatron.fla_runtime_knobs",
    backend="megatron",
    # Phase MUST be "before_train" (not "build_args").  At build_args time
    # the Primus-only YAML keys have not yet been merged into args
    # (MegatronArgBuilder.convert_config strips unknown keys;
    # train_runtime.py:294 merge_namespace re-adds them BEFORE
    # before_train runs).  Model module imports happen even later — inside
    # wrapped_pretrain() — so args.* values are available to every
    # consumer that reads them in __init__ or forward.
    phase="before_train",
    priority=-100,
    description=(
        "Resolve FLA runtime knobs (env var > YAML > default) onto args.* "
        "attributes.  Consumers read args directly; no os.environ access."
    ),
    condition=lambda ctx: _has_any_fla_runtime_field(get_args(ctx)),
)
def patch_fla_runtime_knobs(ctx: PatchContext):
    args = get_args(ctx)

    for field_name, env_name, default in _FLA_RUNTIME_KNOBS:
        env_raw = os.environ.get(env_name)
        yaml_value = getattr(args, field_name, None)

        if env_raw is not None:
            resolved = _env_to_typed(env_raw, default)
            source = f"env {env_name}={env_raw!r}"
        elif yaml_value is not None:
            resolved = yaml_value
            source = f"YAML {field_name}={yaml_value!r}"
        else:
            resolved = default
            source = f"default"

        setattr(args, field_name, resolved)
        log_rank_0(
            f"[Patch:megatron.fla_runtime_knobs] "
            f"args.{field_name} = {resolved!r}  ({source})"
        )
