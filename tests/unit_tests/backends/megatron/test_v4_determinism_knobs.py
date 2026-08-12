###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Pin the V4 fusion knobs a bitwise-reproducible run has to turn off.

``PRIMUS_DETERMINISTIC`` and Megatron's ``deterministic_mode`` between them
cover rocBLAS, TE, the collectives and ``torch.use_deterministic_algorithms``.
Neither reaches DeepSeek-V4's own Triton kernels: each is gated by its own env
knob that is *on by default*, so a V4 run with both switches set still executes
Triton for RMSNorm, RoPE, Sinkhorn, hyper-connections, the compressor, the
indexer and the router. Measured on MI355X, at least one of them
(``PRIMUS_HC_TRITON``) breaks bitwise reproducibility, so the recipe in
``docs/04-technical-guides/determinism-and-reproducibility.md`` has to name
every one of them.

That recipe is only correct while the list is complete, and nothing stops a new
fusion from landing with the same default-on pattern. These tests fail when
that happens, so the new knob gets triaged for determinism instead of silently
weakening the recipe.

Deliberately cheap: the inventory is read off the source tree and the toggles
are pure env lookups, so nothing here trains, allocates a GPU or compiles a
kernel.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TRANSFORMER = _REPO_ROOT / "primus" / "backends" / "megatron" / "core" / "transformer"
_DETERMINISM_DOC = _REPO_ROOT / "docs" / "04-technical-guides" / "determinism-and-reproducibility.md"

# Where a V4 fusion knob can live. The attention-backend implementations
# (_triton_v1 / _triton_v2 / gluon / flydsl) are deliberately excluded: the
# recipe already pins the backend to `eager`, and their dozens of internal
# tuning knobs would swamp the inventory without adding signal.
_SCAN_GLOBS = (
    "*.py",
    "v4_attention_kernels/_triton_common/*.py",
    "moe/_triton/*.py",
)

# Every knob in scope that defaults to ON, i.e. every knob the deterministic
# recipe has to switch off. Keep in lockstep with the doc.
_DEFAULT_ON_KNOBS: Set[str] = {
    "PRIMUS_COMPRESS_FUSE_PROJ",
    "PRIMUS_COMPRESS_MASK_CACHE",
    "PRIMUS_COMPRESS_POOL_TRITON",
    "PRIMUS_COMPRESS_POOL_TRITON_BWD",
    "PRIMUS_COMPRESS_ROPE_CACHE",
    "PRIMUS_HC_COLLAPSE_TRITON",
    "PRIMUS_HC_EXPAND_TRITON",
    "PRIMUS_HC_TRITON",
    "PRIMUS_INDEXER_FUSE_PROJ",
    "PRIMUS_INDEXER_MASK_CACHE",
    "PRIMUS_INDEXER_TRITON",
    "PRIMUS_RMSNORM_TRITON",
    "PRIMUS_ROPE_TRITON",
    "PRIMUS_SINKHORN_TRITON",
    "PRIMUS_V4_ROUTER_TRITON",
}

# Modules exposing ``is_triton_path_enabled`` -> (knob, on by default?).
# These are the dispatchers that pick Triton over the eager body.
_DISPATCHER_MODULES: Dict[str, Tuple[str, bool]] = {
    "v4_attention_kernels._triton_common.hc_glue": ("PRIMUS_HC_TRITON", True),
    "v4_attention_kernels._triton_common.hc_collapse": ("PRIMUS_HC_COLLAPSE_TRITON", True),
    "v4_attention_kernels._triton_common.hc_expand": ("PRIMUS_HC_EXPAND_TRITON", True),
    "v4_attention_kernels._triton_common.rmsnorm": ("PRIMUS_RMSNORM_TRITON", True),
    "v4_attention_kernels._triton_common.rope_interleaved_partial": ("PRIMUS_ROPE_TRITON", True),
    "v4_attention_kernels._triton_common.sinkhorn": ("PRIMUS_SINKHORN_TRITON", True),
    "v4_attention_kernels._triton_common.indexer_score_post": ("PRIMUS_INDEXER_TRITON", True),
    # Legacy full-fuse indexer path; off by default, so it costs the recipe nothing.
    "v4_attention_kernels._triton_common.indexer_score": ("PRIMUS_INDEXER_TRITON_FULL", False),
    "moe._triton.v4_router_post": ("PRIMUS_V4_ROUTER_TRITON", True),
}

_MODULE_PREFIX = "primus.backends.megatron.core.transformer."

_KNOB_RE = re.compile(r'os\.environ\.get\(\s*"(PRIMUS_[A-Z0-9_]+)"\s*,\s*"([01])"\s*\)')


def _scan_knobs() -> List[Tuple[str, str, Path]]:
    """``(knob, default_literal, path)`` for every env knob in scope."""
    out = []
    for pattern in _SCAN_GLOBS:
        for path in sorted(_TRANSFORMER.glob(pattern)):
            text = path.read_text(encoding="utf-8", errors="replace")
            for knob, default in _KNOB_RE.findall(text):
                out.append((knob, default, path))
    return out


def test_default_on_fusion_inventory_is_pinned():
    """A newly default-on V4 knob must be triaged, not silently ignored.

    The deterministic recipe works by naming every knob that is on by default.
    If one is added (or an existing one flips its default) the recipe silently
    stops being sufficient, which is exactly the failure this catches.
    """
    scanned = {knob for knob, default, _ in _scan_knobs() if default == "1"}

    added = sorted(scanned - _DEFAULT_ON_KNOBS)
    removed = sorted(_DEFAULT_ON_KNOBS - scanned)
    assert not added and not removed, (
        "The set of default-on V4 fusion knobs changed.\n"
        f"  newly default-on: {added}\n"
        f"  no longer present/default-on: {removed}\n"
        "Decide whether the path is bitwise-deterministic, then update "
        "_DEFAULT_ON_KNOBS here and the recipe in "
        f"{_DETERMINISM_DOC.relative_to(_REPO_ROOT).as_posix()}."
    )


def test_no_knob_has_conflicting_defaults_across_call_sites():
    """The same knob read with two different defaults would make the recipe ambiguous."""
    defaults: Dict[str, Set[str]] = {}
    for knob, default, _ in _scan_knobs():
        defaults.setdefault(knob, set()).add(default)

    conflicting = {k: sorted(v) for k, v in defaults.items() if len(v) > 1}
    assert not conflicting, f"knobs read with inconsistent defaults: {conflicting}"


@pytest.mark.parametrize("suffix", sorted(_DISPATCHER_MODULES))
def test_dispatcher_knob_can_be_switched_off(monkeypatch, suffix):
    """``is_triton_path_enabled`` must honour the knob in both directions."""
    try:
        module = importlib.import_module(_MODULE_PREFIX + suffix)
    except Exception as exc:  # pragma: no cover - dependency-guarded
        pytest.skip(f"{suffix} not importable in this env: {exc!r}")

    knob, default_on = _DISPATCHER_MODULES[suffix]

    monkeypatch.delenv(knob, raising=False)
    assert module.is_triton_path_enabled() is default_on, f"{knob} default changed"

    monkeypatch.setenv(knob, "0")
    assert module.is_triton_path_enabled() is False, f"{knob}=0 did not reach the eager path"

    monkeypatch.setenv(knob, "1")
    assert module.is_triton_path_enabled() is True, f"{knob}=1 did not enable the Triton path"


def test_every_dispatcher_module_is_listed():
    """Catch a new ``is_triton_path_enabled`` module missing from the table above."""
    found = set()
    for pattern in _SCAN_GLOBS:
        for path in sorted(_TRANSFORMER.glob(pattern)):
            if "def is_triton_path_enabled" in path.read_text(encoding="utf-8", errors="replace"):
                rel = path.relative_to(_TRANSFORMER).with_suffix("")
                found.add(rel.as_posix().replace("/", "."))

    assert found == set(_DISPATCHER_MODULES), (
        "The set of Triton dispatch modules changed.\n"
        f"  added  : {sorted(found - set(_DISPATCHER_MODULES))}\n"
        f"  removed: {sorted(set(_DISPATCHER_MODULES) - found)}"
    )


def test_determinism_doc_disables_every_default_on_knob():
    """The recipe is only usable if it switches off every default-on knob."""
    doc = _DETERMINISM_DOC.read_text(encoding="utf-8", errors="replace")
    disabled = set(re.findall(r"\b(PRIMUS_[A-Z0-9_]+)=0", doc))

    missing = sorted(_DEFAULT_ON_KNOBS - disabled)
    assert not missing, (
        f"{_DETERMINISM_DOC.name} does not set {missing} to 0. A default-on V4 "
        "fusion the recipe omits silently breaks bitwise reproducibility."
    )


def test_determinism_doc_does_not_advertise_knobs_nothing_reads():
    """A knob in the recipe that no code reads gives false confidence."""
    doc = _DETERMINISM_DOC.read_text(encoding="utf-8", errors="replace")
    advertised = set(re.findall(r"\b(PRIMUS_[A-Z0-9_]+)=0", doc))

    # Read from the whole package: some knobs the recipe sets (e.g. the turbo
    # gate) are consumed outside the transformer scan scope.
    sources = "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in (_REPO_ROOT / "primus").rglob("*.py")
    )
    dead = sorted(knob for knob in advertised if f'"{knob}"' not in sources)
    assert not dead, (
        f"{_DETERMINISM_DOC.name} tells users to set {dead}, but nothing reads "
        "them, so setting them has no effect."
    )
