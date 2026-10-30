# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Python checks that run after the tester. The model does not grade itself."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from tools.dataset_agent.integration import run_integration, use_writable_hf_home

_MOCK_PATTERNS = (
    re.compile(r"patch\(\s*['\"][^'\"]*(load_dataset|snapshot_download|load_from_disk)"),
    re.compile(r"(load_dataset|snapshot_download|load_from_disk)\s*="),
    re.compile(r"MagicMock\(\)"),
)
_CUDA_PATTERN = re.compile(r"torch\.cuda|hipLaunch|init_process_group")
_SKIP_PATTERN = re.compile(r"skipTest\(|pytest\.skip\(|unittest\.skip|pytest\.importorskip")
_TORCH_IMPORT = re.compile(r"import torch|from torch|sft\.preprocessing|sft\.dataset import|SFTDataset")


def test_file_violations(source: str, *, dataset: str) -> list[str]:
    """Return reasons a generated pytest is not an acceptable gate."""
    reasons: list[str] = []
    if dataset not in source:
        reasons.append(f"test file does not contain the dataset name {dataset!r}")
    for pattern in _MOCK_PATTERNS:
        if pattern.search(source):
            reasons.append(f"test file mocks the dataset loader ({pattern.pattern})")
    if _CUDA_PATTERN.search(source):
        reasons.append("test file references CUDA or a distributed process group")
    if _SKIP_PATTERN.search(source):
        reasons.append("test file skips instead of reading the dataset. Do not skip when torch is missing.")
    if _TORCH_IMPORT.search(source):
        reasons.append(
            "test file imports torch or Megatron preprocessing. "
            "Call tools.dataset_agent.integration.check_megatron_rows on the streamed rows. "
            "That loads the SFT schema and formatter without torch."
        )
    if "load_dataset" not in source and "snapshot_download" not in source and "load_from_disk" not in source:
        reasons.append("test file does not call load_dataset, snapshot_download, or load_from_disk")
    return reasons


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return loaded


def _trainer_module(document: dict[str, Any]) -> dict[str, Any]:
    modules = document.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("example YAML has no modules: block")
    for name in ("pre_trainer", "post_trainer"):
        module = modules.get(name)
        if isinstance(module, dict):
            return module
    first = next(iter(modules.values()))
    if not isinstance(first, dict):
        raise ValueError("example YAML module is not a mapping")
    return first


def _diffusion_params(repo: Path, module: dict[str, Any], preset: dict[str, Any] | None) -> dict[str, Any]:
    overrides = module.get("overrides") if isinstance(module.get("overrides"), dict) else {}
    params = deep_merge(preset or {}, overrides)
    if "model" not in params:
        model_name = str(module.get("model") or "")
        params["model"] = {"name": Path(model_name).stem}
    return params


def check_example_yaml(
    repo: Path, example: Path, *, backend: str, dataset: str, model_preset: Path | None
) -> str:
    """Return an empty string when the example names the dataset and the builder accepts it."""
    if not example.is_file():
        return f"example YAML is missing: {example}"
    text = example.read_text(encoding="utf-8")
    if dataset not in text:
        return f"example YAML does not contain the dataset name {dataset!r}"
    document = _load_yaml(example)
    module = _trainer_module(document)
    preset = _load_yaml(model_preset) if model_preset and model_preset.is_file() else None
    if backend != "diffusion":
        return ""
    params = _diffusion_params(repo, module, preset)
    try:
        from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
    except Exception as exc:  # noqa: BLE001 — CPU hosts may lack trainer deps
        return f"DiffusionArgBuilder could not be imported: {type(exc).__name__}: {exc}"
    builder = DiffusionArgBuilder()
    builder.update(params)
    try:
        finalized = builder.finalize()
    except Exception as exc:  # noqa: BLE001 — the integrator needs the builder's message
        return f"DiffusionArgBuilder rejected the example YAML: {exc}"
    blob = json.dumps(finalized.dataset, default=str)
    if dataset not in blob:
        return "DiffusionArgBuilder dataset config does not contain the dataset name"
    return ""


def run_pytest(repo: Path, test_file: Path) -> tuple[int, str]:
    use_writable_hf_home()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    command = [os.environ.get("PYTHON", "python3"), "-m", "pytest", str(test_file), "-q", "--tb=short"]
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return 1, f"pytest timed out after 1800s: {exc}"
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return completed.returncode, output[-12000:]


def run_gate(
    repo: Path,
    test_file: Path,
    example: Path,
    *,
    backend: str,
    dataset: str,
    samples: int,
    model_preset: Path | None,
    split: str = "train",
    stage: str = "",
) -> tuple[bool, str]:
    """YAML binding, a real Hub stream through the backend parser, then pytest. No GPU."""
    if not test_file.is_file():
        return False, f"pytest file is missing: {test_file}"
    source = test_file.read_text(encoding="utf-8")
    violations = test_file_violations(source, dataset=dataset)
    yaml_error = check_example_yaml(
        repo, example, backend=backend, dataset=dataset, model_preset=model_preset
    )
    integration_error = ""
    if example.is_file() and not yaml_error:
        try:
            document = _load_yaml(example)
        except Exception as exc:  # noqa: BLE001 — the integrator needs the parse error
            integration_error = f"example YAML could not be parsed: {type(exc).__name__}: {exc}"
        else:
            integration_error = run_integration(
                document,
                backend=backend,
                dataset=dataset,
                split=split,
                samples=samples,
                stage=stage,
            )
    if violations or yaml_error or integration_error:
        parts = list(violations)
        if yaml_error:
            parts.append(yaml_error)
        if integration_error:
            parts.append(integration_error)
        return False, "\n".join(parts)
    code, output = run_pytest(repo, test_file)
    if code != 0:
        return False, f"pytest exit {code}\n{output}"
    return True, output
