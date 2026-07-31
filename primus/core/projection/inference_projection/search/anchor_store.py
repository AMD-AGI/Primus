###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Anchor store — a directory of benchmark artifacts indexed by regime.

The store is the backbone of the sub-scale search: a handful of measured
anchors (the JSON artifacts ``benchmark_vllm.py`` emits) indexed by their
:func:`regime signature <regime.regime_signature>`, with nearest-anchor lookup
in *regime space*.  Reconstruction (see :mod:`reconstruct`) then transports the
nearest in-regime anchor across the continuous axes to any target recipe.

Design choices (v1, deliberately simple and greppable):

* The index is a plain ``index.json`` manifest — no database.  Each entry
  records the artifact path, its regime signature + axes, and the *transport
  coverage* it measured (parallelism, layer counts, batches) so a lookup can
  tell whether a target is interpolated or extrapolated.
* Lookup distance is Hamming over the regime axes; ties are broken by
  closeness on the transport axes (prefer the anchor whose measured
  parallelism / depth is nearest the target, i.e. the least restore).
* The store is typically **per model** — you harvest anchors for the model you
  are tuning — so the ``model`` axis is only used to *filter* when both sides
  name a model, never to inflate distance otherwise.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from . import regime


class AnchorStore:
    """A directory of benchmark artifacts with nearest-in-regime lookup."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.index_path = os.path.join(self.root, "index.json")
        self._entries: List[Dict[str, Any]] = []
        self._load_index()

    # -- persistence -----------------------------------------------------------

    def _load_index(self) -> None:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path) as f:
                    self._entries = json.load(f).get("anchors", [])
            except (OSError, ValueError):
                self._entries = []

    def _save_index(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(self.index_path, "w") as f:
            json.dump({"anchors": self._entries}, f, indent=2)

    # -- ingestion -------------------------------------------------------------

    def add_artifact(self, artifact_path: str) -> Dict[str, Any]:
        """Index a benchmark artifact JSON already on disk.  Idempotent by path
        (re-adding refreshes the entry)."""
        path = os.path.abspath(artifact_path)
        with open(path) as f:
            art = json.load(f)
        entry = self._make_entry(path, art)
        self._entries = [e for e in self._entries if e.get("path") != path]
        self._entries.append(entry)
        self._save_index()
        return entry

    def add_result(self, result: Dict[str, Any], artifact_path: str) -> Dict[str, Any]:
        """Write a benchmark result dict to ``artifact_path`` and index it."""
        path = os.path.abspath(artifact_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f)
        return self.add_artifact(path)

    def _make_entry(self, path: str, art: Dict[str, Any]) -> Dict[str, Any]:
        meta = art.get("meta", {})
        recipe = regime.recipe_from_meta(meta)
        sig = meta.get("regime_signature") or regime.regime_signature(recipe)
        sweep = art.get("sweep") or []
        batches = sorted({int(e["batch"]) for e in sweep if e.get("batch") is not None})
        return {
            "path": path,
            "regime_signature": sig,
            "regime": {k: recipe.get(k) for k in regime.REGIME_AXES},
            "model": meta.get("model"),
            "transport": {
                "tp": meta.get("benchmark_tp") or meta.get("tp"),
                "ep": meta.get("benchmark_ep") or meta.get("ep"),
                "pp": meta.get("benchmark_pp") or meta.get("pp"),
                "target_tp": meta.get("tp"),
                "target_ep": meta.get("ep"),
                "target_pp": meta.get("pp"),
                "num_layers": meta.get("num_hidden_layers"),
                "full_layers": (meta.get("restore") or {}).get("full_layers")
                if meta.get("restore")
                else None,
                "batches": batches,
                "input_len": meta.get("input_len"),
            },
        }

    # -- query -----------------------------------------------------------------

    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def load_artifact(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        with open(entry["path"]) as f:
            return json.load(f)

    def nearest(
        self, recipe: Dict[str, Any], *, model: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """Return ``(entry, regime_distance)`` for the best anchor, or
        ``(None, None)`` if the store is empty.  Distance 0 => same regime
        (fully transportable).  Candidates are optionally filtered to a model;
        ties on regime distance are broken by transport closeness (least
        restore/extrapolation)."""
        cands = self._entries
        if model:
            named = [e for e in cands if e.get("model") in (None, model)]
            cands = named or cands
        if not cands:
            return None, None

        def transport_gap(e: Dict[str, Any]) -> float:
            t = e.get("transport", {})
            gap = 0.0
            # Prefer anchors whose measured parallelism matches the target
            # (smaller restore) and whose depth is closer (less extrapolation).
            for axis in ("tp", "ep", "pp"):
                tv, rv = t.get(axis), recipe.get(axis)
                if tv and rv:
                    gap += abs(float(tv) - float(rv))
            nl, rnl = t.get("num_layers"), recipe.get("num_layers")
            if nl and rnl:
                gap += abs(float(nl) - float(rnl)) / max(1.0, float(rnl))
            return gap

        scored = [
            (regime.regime_distance(recipe, {**e["regime"]}), transport_gap(e), e)
            for e in cands
        ]
        scored.sort(key=lambda s: (s[0], s[1]))
        best = scored[0]
        return best[2], best[0]
