###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PRIMUS_ROOT = Path(__file__).resolve().parents[6]
if str(PRIMUS_ROOT) not in sys.path:
    sys.path.insert(0, str(PRIMUS_ROOT))

from primus.backends.diffusion.diffusion_adapter import DiffusionAdapter
from primus.core.config.primus_config import get_module_config, load_primus_config
from primus.core.utils.yaml_utils import nested_namespace_to_dict


def _log(message: str) -> None:
    print(f"[INFO] diffusion prepare: {message}", file=sys.stderr)


def _fail(message: str) -> None:
    print(f"[ERROR] diffusion prepare: {message}", file=sys.stderr)
    raise SystemExit(1)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return nested_namespace_to_dict(value)


def _select_module_name(cfg: Any, requested: str | None) -> str:
    if requested:
        get_module_config(cfg, requested)
        return requested

    for module_name in ("pre_trainer", "post_trainer"):
        try:
            get_module_config(cfg, module_name)
            return module_name
        except Exception:
            continue

    _fail("config must contain either modules.pre_trainer or modules.post_trainer")
    raise AssertionError("unreachable")


def _is_placeholder(path: str | None) -> bool:
    return not path or path.startswith("/path/to/")


def _require_path(path: str | None, description: str, *, kind: str = "any") -> None:
    if _is_placeholder(path):
        _fail(f"{description} is not configured: {path!r}")

    resolved = Path(path).expanduser()
    if kind == "file" and not resolved.is_file():
        _fail(f"{description} file not found: {resolved}")
    if kind == "dir" and not resolved.is_dir():
        _fail(f"{description} directory not found: {resolved}")
    if kind == "any" and not resolved.exists():
        _fail(f"{description} path not found: {resolved}")

    _log(f"{description}: {resolved}")


def _require_optional_path(path: str | None, description: str, *, kind: str = "any") -> None:
    if _is_placeholder(path):
        _log(f"{description}: not configured; skipping optional initialization")
        return
    _require_path(path, description, kind=kind)


def _looks_like_local_path(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~")):
        return True
    parts = value.split("/")
    return len(parts) == 2 and parts[-1].endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt"))


def _looks_like_hf_reference(value: str) -> bool:
    if _looks_like_local_path(value):
        return False
    parts = value.split("/")
    return len(parts) >= 2 and all(part for part in parts[:2])


def _validate_local_or_hf_id(value: str | None, description: str) -> None:
    if _is_placeholder(value):
        _fail(f"{description} is not configured: {value!r}")
    value = str(value)
    path = Path(value).expanduser()
    if path.exists():
        _log(f"{description}: {path}")
    elif _looks_like_local_path(value):
        _fail(f"{description} path not found: {path}")
    elif _looks_like_hf_reference(value):
        _log(f"{description}: {value} (assuming Hugging Face reference)")
    else:
        _fail(f"{description} is neither an existing path nor a Hugging Face reference: {value!r}")


def validate_diffusion_config(config_path: Path, module_name: str | None = None) -> None:
    cfg = load_primus_config(config_path)
    selected_module = _select_module_name(cfg, module_name)
    module_cfg = get_module_config(cfg, selected_module)

    if getattr(module_cfg, "framework", None) != "diffusion":
        _log(f"module {selected_module} framework is not diffusion; skipping")
        return

    backend_args = DiffusionAdapter().convert_config(module_cfg.params)
    model = _as_dict(backend_args.model)
    dataset = _as_dict(backend_args.dataset)
    trainer = _as_dict(backend_args.trainer)

    dataset_cfg = dataset.get("config", {})
    processor_cfg = dataset_cfg.get("processor_config", {})
    model_cfg = model.get("config", {})
    model_name = model.get("name")

    if model_name == "wan":
        encoder_cfg = model_cfg.get("encoder", {}) or model.get("encoder", {})
        _require_path(dataset_cfg.get("dataset_path"), "dataset metadata", kind="file")
        _require_path(dataset_cfg.get("data_folder"), "dataset media folder", kind="dir")
        _require_path(processor_cfg.get("text_tokenizer"), "text tokenizer", kind="dir")
        _require_path(model_cfg.get("load_from_pretrained_path"), "DiT initialization checkpoint")
        _require_path(encoder_cfg.get("t5_encoder"), "text encoder checkpoint", kind="file")
        _require_path(encoder_cfg.get("autoencoder"), "VAE checkpoint", kind="file")
    elif model_name == "flux":
        dataset_type = str(dataset_cfg.get("dataset_type", "precomputed")).lower()
        if dataset_type == "raw":
            dataset_name = dataset_cfg.get("dataset")
            dataset_format = str(dataset_cfg.get("dataset_format", "webdataset")).lower()
            if dataset_name == "cc12m-test" and _is_placeholder(dataset_cfg.get("dataset_path")):
                _fail(
                    "FLUX raw dataset `cc12m-test` requires `dataset_path` to point at local webdataset tars"
                )
            elif dataset_name == "cc12m-wds":
                if _is_placeholder(dataset_cfg.get("dataset_path")):
                    _log("FLUX raw image-text dataset: pixparse/cc12m-wds (Hugging Face dataset)")
                else:
                    _validate_local_or_hf_id(dataset_cfg.get("dataset_path"), "FLUX raw Hugging Face dataset")
            elif dataset_format == "hf_repo":
                _validate_local_or_hf_id(dataset_cfg.get("dataset_path"), "FLUX raw Hugging Face dataset")
            else:
                kind = "file" if dataset_format == "jsonl" else "dir"
                _require_path(dataset_cfg.get("dataset_path"), "FLUX raw image-text dataset", kind=kind)
            encoder_cfg = model_cfg.get("encoder", {}) or model.get("encoder", {})
            _validate_local_or_hf_id(encoder_cfg.get("t5_encoder"), "FLUX T5 encoder")
            _validate_local_or_hf_id(encoder_cfg.get("clip_encoder"), "FLUX CLIP encoder")
            _validate_local_or_hf_id(encoder_cfg.get("autoencoder"), "FLUX autoencoder checkpoint")
        elif dataset_type == "precomputed":
            _require_path(dataset_cfg.get("dataset_path"), "FLUX precomputed dataset", kind="dir")
            prompt_dropout_prob = float(processor_cfg.get("prompt_dropout_prob", 0.0) or 0.0)
            if prompt_dropout_prob > 0.0:
                empty_dir = processor_cfg.get("empty_encodings_path")
                _require_path(empty_dir, "FLUX empty encodings directory", kind="dir")
                _require_path(str(Path(empty_dir) / "t5_empty.npy"), "FLUX empty T5 encoding", kind="file")
                _require_path(
                    str(Path(empty_dir) / "clip_empty.npy"), "FLUX empty CLIP encoding", kind="file"
                )
        else:
            _fail(f"unsupported FLUX dataset_type: {dataset_type!r}")
        _require_optional_path(
            model_cfg.get("load_from_pretrained_path"),
            "FLUX DiT initialization checkpoint",
        )
    else:
        _fail(f"unsupported diffusion model for prepare hook: {model_name!r}")

    _log(
        "validated "
        f"module={selected_module} stage={getattr(backend_args, 'stage', None)} "
        f"trainer={trainer.get('name')} model={model.get('name')}"
    )
    print("env.PREPARED=1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Primus diffusion training environment")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config")
    parser.add_argument("--data_path", type=Path, required=False, help="Reserved for hook API compatibility")
    parser.add_argument(
        "--primus_path", type=Path, required=False, help="Reserved for hook API compatibility"
    )
    parser.add_argument("--patch_args", type=Path, required=False, help="Reserved for hook API compatibility")
    parser.add_argument("--backend_path", type=str, default=None, help="Unused; diffusion is in-tree")
    parser.add_argument("--module_name", type=str, default=None, help="Override module name to validate")
    args, _unknown = parser.parse_known_args()

    if args.backend_path:
        _fail("diffusion is an in-tree backend and does not support --backend_path")

    validate_diffusion_config(args.config, module_name=args.module_name)


if __name__ == "__main__":
    main()
