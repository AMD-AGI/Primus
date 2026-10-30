# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Host-side proof that the example YAML points at a real dataset and the records parse.

No GPU, no container, and no trainer process. The check reads the dataset id from the
YAML keys the backend actually uses, streams a few Hub rows, and runs those rows through
the backend field parser that does not import torch.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

_DATASET_KEYS = {"sft_dataset_name", "dataset_path", "dataset_name"}
_PROMPT_KEYS = ("txt", "caption", "prompt", "text")
_IMAGE_KEYS = ("image", "jpg", "png")


def bound_dataset_fields(node: Any) -> list[tuple[str, str]]:
    """Return dataset-id fields stored in the parsed example YAML."""
    found: list[tuple[str, str]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in _DATASET_KEYS and isinstance(child, str) and child.strip():
                    found.append((str(key), child.strip()))
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return found


def first_field(node: Any, key: str, default: str = "") -> str:
    found = ""

    def walk(value: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(value, dict):
            child = value.get(key)
            if isinstance(child, str) and child.strip():
                found = child.strip()
                return
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return found or default


_INFRA_EXCEPTIONS = (
    "IsADirectoryError",
    "PermissionError",
    "ConnectionError",
    "ConnectionResetError",
    "TimeoutError",
    "gaierror",
    "SSLError",
    "ProxyError",
)


def is_infra_error(log: str) -> bool:
    """An error that editing Primus or the example YAML cannot fix.

    ``run_integration`` wraps the stream failure as
    ``could not stream {dataset!r} split {split!r}: {ExceptionType}: {detail}``.
    A host-level exception there means the agent never reached the dataset's rows.
    """
    if "could not stream" not in log:
        return False
    return any(name in log for name in _INFRA_EXCEPTIONS)


def binding_error(document: Any, dataset: str) -> str:
    fields = bound_dataset_fields(document)
    if any(value == dataset for _, value in fields):
        return ""
    shown = ", ".join(f"{key}={value}" for key, value in fields) or "(none)"
    return (
        f"example YAML does not set sft_dataset_name, dataset_path, or dataset_name to {dataset!r}. "
        f"Found: {shown}"
    )


def check_megatron_rows(rows: list[dict[str, Any]], formatter_name: str) -> str:
    """Map rows with the Megatron SFT schema and formatter. Those files do not import torch."""
    sft_sample, create_formatter = _sft_api()
    try:
        formatter = create_formatter(formatter_name)
    except ValueError as exc:
        return str(exc)
    for index, row in enumerate(rows):
        sample = sft_sample.from_mapping(row)
        text = formatter.format_sample(sample).text.strip()
        if not text:
            return f"row {index} formatted to empty training text with formatter {formatter_name!r}"
        if not sample.is_multi_turn and not (sample.response or "").strip():
            return f"row {index} has no response after Megatron SFT field mapping"
        if not sample.is_multi_turn and not (sample.instruction or "").strip():
            return f"row {index} has no instruction after Megatron SFT field mapping"
    return ""


def check_diffusion_rows(rows: list[dict[str, Any]]) -> str:
    """Use the same caption and image keys as FluxRawImageTextDataset, without importing torch."""
    for index, row in enumerate(rows):
        if not any(_present(row, key) for key in _PROMPT_KEYS):
            return f"row {index} has no caption field ({', '.join(_PROMPT_KEYS)})"
        if not any(_present(row, key) for key in _IMAGE_KEYS):
            return f"row {index} has no image field ({', '.join(_IMAGE_KEYS)})"
    return ""


def new_format_error(document: Any, *, stage: str, backend: str) -> str:
    """The example must name a format and use the trainer module for this stage.

    An existing format is valid. A missing format name is not. A name that
    ``create_formatter`` does not know yet fails later, when the rows are parsed.
    """
    del backend
    modules = document.get("modules") if isinstance(document, dict) else None
    modules = modules if isinstance(modules, dict) else {}
    if stage == "posttrain" and "post_trainer" not in modules:
        return "post-train example must use modules.post_trainer"
    if stage == "pretrain" and "pre_trainer" not in modules:
        return "pre-train example must use modules.pre_trainer"
    if stage == "posttrain":
        name = first_field(document, "sft_conversation_format") or first_field(document, "dataset_format")
        if not name:
            return "post-train example must set sft_conversation_format or dataset_format"
        return ""
    if (
        stage == "pretrain"
        and not first_field(document, "dataset_format")
        and not first_field(document, "data_path")
    ):
        return "pre-train example must set dataset_format or data_path"
    return ""


def check_rows(backend: str, rows: list[dict[str, Any]], document: Any, stage: str = "") -> str:
    if not rows:
        return "the Hub stream returned no records"
    if backend == "diffusion":
        return check_diffusion_rows(rows)
    if stage == "posttrain" and backend in {"megatron", "megatron_bridge"}:
        formatter_name = first_field(document, "sft_conversation_format")
        return check_megatron_rows(rows, formatter_name)
    if backend in {"megatron", "megatron_bridge"} and first_field(document, "sft_conversation_format"):
        formatter_name = first_field(document, "sft_conversation_format")
        return check_megatron_rows(rows, formatter_name)
    for index, row in enumerate(rows):
        if not row:
            return f"row {index} is empty"
    return ""


def use_writable_hf_home(token_path: Path | None = None, home: Path | None = None) -> str:
    """Avoid a root-owned ``~/.cache/huggingface/token`` directory.

    ``huggingface_hub`` reads that path as a token file before every Hub call.
    On this host it is an empty directory, so the read raises ``IsADirectoryError``
    before any dataset row is fetched. A set ``HF_HOME`` is left alone.
    """
    if os.environ.get("HF_HOME", "").strip():
        return ""
    token = token_path or (Path.home() / ".cache" / "huggingface" / "token")
    if not token.is_dir():
        return ""
    target = home or (Path.home() / ".cache" / "primus-dataset-agent" / "hf-home")
    target.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(target)
    return f"Hugging Face token path {token} is a directory, not a token file. " f"Using HF_HOME={target}."


def stream_rows(dataset: str, split: str, samples: int) -> list[dict[str, Any]]:
    note = use_writable_hf_home()
    if note:
        print(note)
    from datasets import load_dataset

    streamed = load_dataset(dataset, split=split, streaming=True)
    rows: list[dict[str, Any]] = []
    for row in streamed:
        if not isinstance(row, dict):
            row = dict(row)
        rows.append(row)
        if len(rows) >= samples:
            break
    return rows


def run_integration(
    document: Any,
    *,
    backend: str,
    dataset: str,
    split: str,
    samples: int,
    stage: str = "",
) -> str:
    """Return an empty string when the YAML binding and the parsed rows both hold."""
    if stage:
        format_error = new_format_error(document, stage=stage, backend=backend)
        if format_error:
            return format_error
    error = binding_error(document, dataset)
    if error:
        return error
    try:
        rows = stream_rows(dataset, split, samples)
    except Exception as exc:  # noqa: BLE001 — the integrator needs the Hub error
        return f"could not stream {dataset!r} split {split!r}: {type(exc).__name__}: {exc}"
    if len(rows) < samples:
        return f"streamed {len(rows)} records from {dataset!r}, need {samples}"
    parsed = check_rows(backend, rows, document, stage)
    if parsed:
        return parsed
    return ""


def _sft_api():
    """Load schema.py and formatters.py without Megatron's package init, which imports torch."""
    schema_name = "primus.backends.megatron.sft.schema"
    formatter_name = "primus.backends.megatron.sft.formatters"
    if schema_name in sys.modules and formatter_name in sys.modules:
        schema = sys.modules[schema_name]
        formatters = importlib.reload(sys.modules[formatter_name])
        return schema.SFTSample, formatters.create_formatter
    repo = Path(__file__).resolve().parents[2]
    import primus.backends  # noqa: F401 — primus itself imports here; megatron's init does not

    for name, path in (
        ("primus.backends.megatron", repo / "primus" / "backends" / "megatron"),
        ("primus.backends.megatron.sft", repo / "primus" / "backends" / "megatron" / "sft"),
    ):
        if name in sys.modules:
            continue
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package
    schema = _load_module(schema_name, repo / "primus" / "backends" / "megatron" / "sft" / "schema.py")
    formatters = _load_module(
        formatter_name, repo / "primus" / "backends" / "megatron" / "sft" / "formatters.py"
    )
    return schema.SFTSample, formatters.create_formatter


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _present(row: dict[str, Any], key: str) -> bool:
    if key not in row:
        return False
    value = row[key]
    return value is not None and value != ""
