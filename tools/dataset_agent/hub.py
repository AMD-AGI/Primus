# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Look up a dataset on the Hugging Face Hub before any agent runs."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import Counter
from typing import Any


def summarize_dataset(payload: dict[str, Any], *, dataset: str, error: str = "") -> dict[str, Any]:
    """Reduce a Hub dataset API payload to the facts the planner needs."""
    siblings = payload.get("siblings") or []
    names = [str(item.get("rfilename", "")) for item in siblings if isinstance(item, dict)]
    suffixes = Counter(os.path.splitext(name)[1].lower() or "(none)" for name in names if name)
    features: list[str] = []
    card = payload.get("cardData") or payload.get("card_data") or {}
    info = card.get("dataset_info") if isinstance(card, dict) else None
    if isinstance(info, dict):
        raw_features = info.get("features") or []
        if isinstance(raw_features, dict):
            features = [str(name) for name in raw_features]
        elif isinstance(raw_features, list):
            for feature in raw_features:
                if isinstance(feature, dict) and feature.get("name"):
                    features.append(str(feature["name"]))
                elif isinstance(feature, str):
                    features.append(feature)
    return {
        "dataset": dataset,
        "hub_id": str(payload.get("id") or dataset),
        "private": bool(payload.get("private", False)),
        "tags": [str(tag) for tag in (payload.get("tags") or [])[:20]],
        "file_count": len(names),
        "suffix_counts": dict(suffixes.most_common(12)),
        "sample_files": names[:20],
        "columns": features,
        "error": error,
    }


def fetch_dataset(dataset: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch dataset metadata. A missing dataset is returned as an error summary."""
    repo_id = dataset.strip().strip("/")
    url = "https://huggingface.co/api/datasets/" + repo_id
    headers = {"User-Agent": "primus-dataset-agent", "Accept": "application/json"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        return summarize_dataset({}, dataset=dataset, error=f"HTTP {exc.code}: {detail}")
    except Exception as exc:  # noqa: BLE001 — the planner must see lookup failures
        return summarize_dataset({}, dataset=dataset, error=f"{type(exc).__name__}: {exc}")
    if not isinstance(payload, dict):
        return summarize_dataset({}, dataset=dataset, error="Hub response was not a JSON object")
    return summarize_dataset(payload, dataset=dataset)


def format_hub_facts(facts: dict[str, Any]) -> str:
    lines = [
        f"dataset: {facts.get('dataset', '')}",
        f"hub_id: {facts.get('hub_id', '')}",
        f"private: {facts.get('private', False)}",
        f"columns: {', '.join(facts.get('columns') or []) or '(not in the dataset card)'}",
        f"file_count: {facts.get('file_count', 0)}",
        f"suffix_counts: {facts.get('suffix_counts') or {}}",
        "sample_files:",
    ]
    for name in facts.get("sample_files") or []:
        lines.append(f"  - {name}")
    if not facts.get("sample_files"):
        lines.append("  - (none)")
    if facts.get("error"):
        lines.append(f"lookup_error: {facts['error']}")
    return "\n".join(lines)
