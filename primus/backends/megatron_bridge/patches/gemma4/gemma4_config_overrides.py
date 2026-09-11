###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Set arbitrary ``ConfigContainer`` fields from the environment.

The Gemma 4 recipes expose a fixed keyword list (``tensor_model_parallel_size``,
``expert_model_parallel_size``, ``context_parallel_size``, ``sequence_parallel``,
``use_megatron_fsdp``, ...), and Primus' flat YAML ``overrides`` block can only
reach top-level ``ConfigContainer`` attributes. That leaves the knobs that matter
most for tuning Gemma 4 unreachable, among them:

  * ``model.recompute_granularity`` / ``recompute_method`` / ``recompute_num_layers``
    / ``recompute_modules`` -- the only real lever on activation memory, and so
    the gate on micro-batch size and on running at a lower TP degree
  * ``model.moe_token_dispatcher_type`` (``allgather`` / ``alltoall`` / ``flex``)
    and ``model.moe_enable_deepep``
  * ``model.expert_tensor_parallel_size``, ``model.moe_permute_fusion``,
    ``model.moe_shared_expert_overlap``, ``model.moe_router_dtype``

Rather than widen every recipe signature for an experiment, this patch applies
dotted-path assignments after the recipe has produced its ``ConfigContainer``::

    PRIMUS_GEMMA4_SET="model.recompute_granularity=full;model.recompute_method=uniform;model.recompute_num_layers=1"

Entries are separated by ``;`` (so list values keep their commas) and parsed as
Python literals, falling back to the raw string. This is a tuning hook, not a
supported configuration surface: anything that proves worth keeping belongs in
the recipe's keyword list.
"""

from __future__ import annotations

import ast
import os
from enum import Enum
from typing import Any, List, Tuple

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_ENV = "PRIMUS_GEMMA4_SET"
_PATCHED_ATTR = "_primus_gemma4_overrides_patched"


def _parse_value(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("none", "null"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _parse_spec(spec: str) -> List[Tuple[str, Any]]:
    entries = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            log_rank_0(
                f"[Patch:gemma4.config.overrides] Ignoring malformed entry {chunk!r} (expected path=value)"
            )
            continue
        path, _, value = chunk.partition("=")
        entries.append((path.strip(), _parse_value(value)))
    return entries


def _coerce_to_field(before: Any, value: Any, path: str) -> Any:
    """Coerce a parsed string to the existing field's type where it matters.

    Several of the fields worth overriding are enums, not strings -- notably
    ``model.attention_backend``, which is an ``AttnBackend``. megatron-core
    compares it by enum identity, so ``setattr``-ing the plain string
    ``"unfused"`` does not raise, it silently fails to match and the run
    proceeds on the default backend. That is the worst possible outcome for a
    tuning hook: a config knob that reports success and changes nothing.
    """
    if isinstance(before, Enum) and not isinstance(value, Enum):
        cls = type(before)
        try:
            return cls[str(value)]
        except KeyError:
            pass
        try:
            return cls(value)
        except ValueError:
            log_rank_0(
                f"[Patch:gemma4.config.overrides] {path}: {value!r} is not a valid "
                f"{cls.__name__} (valid: {[m.name for m in cls]}); leaving {before!r}"
            )
            return before
    return value


def _apply(container: Any, entries: List[Tuple[str, Any]]) -> None:
    for path, value in entries:
        parts = path.split(".")
        target = container
        for part in parts[:-1]:
            target = getattr(target, part, None)
            if target is None:
                break
        leaf = parts[-1]
        if target is None:
            log_rank_0(f"[Patch:gemma4.config.overrides] Skipped {path}: no such config section")
            continue
        if not hasattr(target, leaf):
            log_rank_0(f"[Patch:gemma4.config.overrides] Skipped {path}: field does not exist")
            continue
        before = getattr(target, leaf)
        value = _coerce_to_field(before, value, path)
        setattr(target, leaf, value)
        log_rank_0(f"[Patch:gemma4.config.overrides] {path}: {before!r} -> {value!r}")


@register_patch(
    "gemma4.config.overrides",
    backend="megatron_bridge",
    phase="setup",
    description="Opt-in: set arbitrary dotted ConfigContainer fields from PRIMUS_GEMMA4_SET",
)
def patch_gemma4_config_overrides(ctx: PatchContext) -> None:
    spec = os.environ.get(_ENV, "").strip()
    if not spec:
        return

    entries = _parse_spec(spec)
    if not entries:
        return

    # The trainers import load_recipe_config by value, and they call it from
    # setup() -- i.e. after this patch phase -- so rebinding the name in each
    # trainer module is enough to post-process the ConfigContainer.
    try:
        from primus.backends.megatron_bridge import (
            megatron_bridge_posttrain_trainer,
            megatron_bridge_pretrain_trainer,
        )
    except Exception:
        return

    for module in (megatron_bridge_pretrain_trainer, megatron_bridge_posttrain_trainer):
        original = getattr(module, "load_recipe_config", None)
        if original is None or getattr(original, _PATCHED_ATTR, False):
            continue

        def load_recipe_config(backend_args, _original=original):
            container = _original(backend_args)
            _apply(container, entries)
            return container

        setattr(load_recipe_config, _PATCHED_ATTR, True)
        module.load_recipe_config = load_recipe_config

    log_rank_0(f"[Patch:gemma4.config.overrides] Will apply {len(entries)} override(s) after recipe load")
