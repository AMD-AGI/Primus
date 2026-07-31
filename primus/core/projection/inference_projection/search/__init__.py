###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Sub-scale search / reconstruction layer for the inference projector.

This package turns a *handful of sub-scale benchmark anchors* into projections
across the whole (very large) serving recipe space, following the design in the
"Fast, Cheap Inference Config Search" note:

  * :mod:`regime`  — the canonical split of a recipe into *regime-defining*
    axes (each needs its own measurement) and *transportable* axes (moved
    analytically from an anchor).  Provides the regime signature + config key
    used by both the benchmark cache and the anchor store.
  * :mod:`anchor_store` — a directory of benchmark artifacts indexed by regime
    signature, with nearest-anchor lookup in regime space.
  * :mod:`reconstruct` — drives the existing ``InferencePerformanceProjector``
    from the nearest in-regime anchor, transporting it to a target recipe.

NOTE: this ``__init__`` intentionally imports **nothing** so that
``regime`` (stdlib-only) can be imported by the dependency-light
``benchmark_vllm.py`` running inside a bare vLLM container.  Import the
submodules explicitly (``from ...search import anchor_store``).
"""
