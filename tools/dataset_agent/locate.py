# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Resolve the Primus repo, model preset, and the files this run will write."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Primus checkout that contains this package."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "primus").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not find the Primus repo root from tools/dataset_agent.")


def slugify(*parts: str) -> str:
    raw = "-".join(part.strip().lower() for part in parts if part.strip())
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw)
    slug = re.sub(r"-{2,}", "-", slug).strip("-.")
    return slug[:120] or "dataset"


def module_stem(slug: str) -> str:
    """A Python-importable module name. pytest cannot import a file with dots in its name."""
    stem = re.sub(r"[^a-z0-9_]+", "_", slug.lower())
    stem = re.sub(r"_{2,}", "_", stem).strip("_")
    return stem[:100] or "dataset"


@dataclass(frozen=True)
class Layout:
    repo: Path
    slug: str
    branch: str
    model_preset: Path | None
    template_example: Path | None
    example_yaml: Path
    test_file: Path
    run_dir: Path


def _model_filename(model: str) -> str:
    if model.endswith(".yaml") or model.endswith(".yml"):
        return model
    return f"{model}.yaml"


def find_model_preset(repo: Path, backend: str, model: str) -> Path | None:
    filename = _model_filename(model)
    stem = Path(filename).stem
    preferred = repo / "primus" / "configs" / "models" / backend / filename
    if preferred.is_file():
        return preferred
    root = repo / "primus" / "configs" / "models"
    if not root.is_dir():
        return None
    exact = [path for path in root.rglob(filename) if path.is_file()]
    if exact:
        in_backend = [path for path in exact if backend in path.parts]
        return in_backend[0] if in_backend else exact[0]
    loose = [path for path in root.rglob("*.yaml") if stem in path.stem]
    if not loose:
        return None
    in_backend = [path for path in loose if backend in path.parts]
    return in_backend[0] if in_backend else loose[0]


def find_template_example(repo: Path, backend: str, model: str, stage: str = "") -> Path | None:
    root = repo / "examples"
    if not root.is_dir():
        return None
    stem = Path(_model_filename(model)).stem
    matches = [path for path in root.rglob("*.yaml") if stem in path.name]
    if not matches:
        return None
    in_backend = [path for path in matches if backend in path.parts]
    pool = in_backend or matches

    def rank(path: Path) -> tuple[int, int, int]:
        name = path.name.lower()
        if stage == "posttrain":
            stage_rank = 0 if "sft" in name or "posttrain" in name else 1
        elif stage == "pretrain":
            stage_rank = 0 if "pretrain" in name and "sft" not in name else 1
        else:
            stage_rank = 0
        raw = 0 if "raw" in name else 1
        return (stage_rank, raw, len(path.parts))

    return sorted(pool, key=rank)[0]


def layout_for(repo: Path, backend: str, model: str, dataset: str, stage: str = "") -> Layout:
    slug = slugify(backend, stage, model, dataset)
    return Layout(
        repo=repo,
        slug=slug,
        branch=f"dataset-agent/{slug}",
        model_preset=find_model_preset(repo, backend, model),
        template_example=find_template_example(repo, backend, model, stage),
        example_yaml=repo / "examples" / backend / "configs" / "dataset_agent" / f"{slug}.yaml",
        test_file=repo / "tests" / "unit_tests" / "tools" / "dataset_agent" / f"test_{module_stem(slug)}.py",
        run_dir=repo / "output" / "dataset_agent" / slug,
    )
