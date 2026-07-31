###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Reconstruction driver — project a target recipe from the nearest anchor.

This is the payoff of the sub-scale search: given a target ``InferenceConfig``
and an :class:`~anchor_store.AnchorStore`, pick the nearest in-regime anchor and
*transport* it to the target by driving the existing
``InferencePerformanceProjector``.  All of the physics (TP/EP/PP + depth restore,
batch/seqlen interpolation) already lives in the projector's
``set_benchmark_calibration`` / ``_restore_whole`` — reconstruction simply loads
the right anchor as the calibration source and calls ``project()``.

The result carries the regime distance to the anchor and a coarse **confidence**
tag so callers (e.g. the tuner) can decide whether to trust the reconstruction
or escalate the recipe to a real run:

* ``high``    — same regime (distance 0) and the target's transport axes are
  within the anchor's measured coverage (interpolation).
* ``interp``  — same regime but extrapolating beyond the measured batch/depth.
* ``escalate``— no in-regime anchor (distance > 0); a new measurement is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import regime
from .anchor_store import AnchorStore


@dataclass
class ReconstructionResult:
    perf: Any                       # InferencePerfResult
    anchor: Dict[str, Any]          # the anchor store entry used
    regime_distance: int            # 0 == same regime (fully transportable)
    confidence: str                 # "high" | "interp" | "escalate"
    reason: str = ""
    projector: Any = field(default=None, repr=False)


def _confidence(recipe: Dict[str, Any], entry: Dict[str, Any], distance: int) -> tuple:
    if distance > 0:
        return "escalate", f"nearest anchor differs on {distance} regime axis(es)"
    t = entry.get("transport", {})
    batches = t.get("batches") or []
    tgt_batch = recipe.get("batch")
    if tgt_batch is not None and batches and not (min(batches) <= tgt_batch <= max(batches)):
        return "interp", (
            f"batch {tgt_batch} outside measured range [{min(batches)}, {max(batches)}]"
        )
    # Depth extrapolation only matters when the anchor did not restore to full.
    full = t.get("full_layers")
    nl = t.get("num_layers")
    tgt_layers = recipe.get("num_layers")
    if full is None and nl and tgt_layers and int(tgt_layers) != int(nl):
        return "interp", f"target depth {tgt_layers} != anchor depth {nl} (no depth restore in anchor)"
    return "high", "same regime, within measured coverage"


def reconstruct(
    inference_config: Any,
    store: AnchorStore,
    *,
    args: Any = None,
    model: Optional[str] = None,
) -> Optional[ReconstructionResult]:
    """Reconstruct the target ``inference_config``'s performance from the nearest
    anchor in ``store``.  Returns ``None`` when the store has no anchors.

    The nearest anchor is loaded as the benchmark calibration source; the
    projector then restores it from the anchor's (reduced) parallelism/depth to
    the target defined by ``inference_config`` — so a TP=1, few-layer anchor is
    transported to, say, TP=8 at full depth analytically, with no new run."""
    from ..performance import InferencePerformanceProjector

    recipe = regime.recipe_from_inference_config(inference_config)
    entry, distance = store.nearest(recipe, model=model)
    if entry is None:
        return None

    artifact = store.load_artifact(entry)
    projector = InferencePerformanceProjector(
        inference_config, args=args, benchmark_layer_times=artifact
    )
    perf = projector.project()
    conf, reason = _confidence(recipe, entry, int(distance or 0))
    return ReconstructionResult(
        perf=perf,
        anchor=entry,
        regime_distance=int(distance or 0),
        confidence=conf,
        reason=reason,
        projector=projector,
    )
