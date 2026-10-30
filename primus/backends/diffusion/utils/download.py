###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Download AMDiffusionBenchmark models and datasets into the Hugging Face cache.

``primus-cli`` runs this from the diffusion prepare hook, before the training
processes start. Model repos land in ``<cache>/hub``. Dataset repos land in
``<cache>/datasets/<org>/<name>``. The cache root is ``PRIMUS_HF_HOME`` when
that is set. Otherwise an explicit ``HF_HOME`` is used, except when it equals
the runner's automatic ``$DATA_PATH/huggingface``. The fallback is
``~/cache/huggingface``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MODEL_FILES: dict[str, tuple[str, str]] = {
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
    "stable-diffusion-xl": ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
}
_DATASETS: dict[str, tuple[str, str | None]] = {
    "sdxl": ("frank-chieng/chinese_architecture_siheyuan", None),
    "stable-diffusion-xl": ("frank-chieng/chinese_architecture_siheyuan", None),
}


def _log(message: str) -> None:
    print(f"[INFO] diffusion assets: {message}", file=sys.stderr)


def huggingface_home() -> Path:
    """Cache directory for benchmark downloads."""
    requested = os.environ.get("PRIMUS_HF_HOME")
    if requested:
        return Path(requested).expanduser()
    current = os.environ.get("HF_HOME")
    data_path = os.environ.get("DATA_PATH")
    automatic = str(Path(data_path) / "huggingface") if data_path else None
    if current and current != automatic:
        return Path(current).expanduser()
    return Path.home() / "cache" / "huggingface"


def dataset_root() -> Path:
    """Folder that holds dataset repos, separate from the model hub cache."""
    root = huggingface_home() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_folder(repo_id: str) -> Path:
    """Stable directory for one dataset repo: ``datasets/<org>/<name>``."""
    return dataset_root() / repo_id


def configure_huggingface_cache() -> Path:
    """Point Hugging Face clients at the benchmark cache for this process."""
    home = huggingface_home()
    hub = home / "hub"
    processed = home / "datasets-cache"
    home.mkdir(parents=True, exist_ok=True)
    hub.mkdir(parents=True, exist_ok=True)
    dataset_root()
    processed.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(home)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["HF_DATASETS_CACHE"] = str(processed)
    _link_existing_token(home)
    return home


def cache_env() -> dict[str, str]:
    """Environment lines the prepare hook exports to the training processes."""
    home = configure_huggingface_cache()
    return {
        "HF_HOME": str(home),
        "HF_HUB_CACHE": str(home / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(home / "hub"),
        "HF_DATASETS_CACHE": str(home / "datasets-cache"),
    }


def _link_existing_token(home: Path) -> None:
    """Use an existing ``~/.hf_token`` without copying or printing it."""
    destination = home / "token"
    source = Path.home() / ".hf_token"
    if destination.exists() or destination.is_symlink() or not source.is_file():
        return
    destination.symlink_to(source)


def _is_hf_repo(value: str) -> bool:
    path = Path(value).expanduser()
    if path.exists() or value.startswith(("/", "./", "../", "~")):
        return False
    parts = value.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        return False
    return not parts[-1].endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt"))


def _distributed_rank() -> int | None:
    """Return the process rank, or None when distributed is not initialized."""
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return None


def _on_rank0_then_others(download):
    """Run ``download`` on rank 0 first so other ranks hit a warm cache."""
    rank = _distributed_rank()
    result = download() if rank in (None, 0) else None
    if rank is None:
        return result
    import torch.distributed as dist

    dist.barrier()
    if rank != 0:
        result = download()
    return result


def _snapshot(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    from huggingface_hub import snapshot_download

    def download() -> str:
        home = configure_huggingface_cache()
        _log(f"Downloading {repo_id} (model) into {home / 'hub'}")
        kwargs: dict[str, object] = {"repo_id": repo_id, "cache_dir": str(home / "hub")}
        if revision:
            kwargs["revision"] = revision
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        return snapshot_download(**kwargs)

    return _on_rank0_then_others(download)


def _snapshot_dataset(repo_id: str, *, revision: str | None = None) -> str:
    """Download a dataset repo into ``datasets/<org>/<name>``, not the model hub."""
    from huggingface_hub import snapshot_download

    def download() -> str:
        configure_huggingface_cache()
        local_dir = dataset_folder(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        blob_cache = dataset_root() / ".hub"
        blob_cache.mkdir(parents=True, exist_ok=True)
        _log(f"Downloading {repo_id} (dataset) into {local_dir}")
        kwargs: dict[str, str] = {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "cache_dir": str(blob_cache),
            "local_dir": str(local_dir),
        }
        if revision:
            kwargs["revision"] = revision
        snapshot_download(**kwargs)
        return str(local_dir)

    return _on_rank0_then_others(download)


def materialize_pretrained(model_name: str, pretrained: str | None) -> str:
    """Return a local checkpoint path, downloading the official repo when needed."""
    spec = _MODEL_FILES.get(model_name)
    if spec is None:
        return str(pretrained or "")
    default_repo, relative = spec
    text = str(pretrained or "").strip()
    # YAML templates ship ``/path/to/...`` as "not set"; download the official file.
    if not text or text.startswith("/path/to/"):
        text = ""
    local = Path(text).expanduser() if text else None
    if local is not None and local.exists():
        return str(local)
    repo_id = default_repo
    filename = relative
    remote_file = text.count("/") >= 2 and not text.startswith(("/", ".", "~"))
    if text and _is_hf_repo(text):
        repo_id = text
    elif remote_file:
        org, _, remainder = text.partition("/")
        name, _, filename = remainder.partition("/")
        repo_id = f"{org}/{name}"
        filename = filename or relative
    elif text:
        raise FileNotFoundError(f"{model_name} checkpoint does not exist: {text}")
    snapshot = _snapshot(repo_id, allow_patterns=[filename] if filename else None)
    target = Path(snapshot) / filename
    if not target.exists():
        raise FileNotFoundError(f"{repo_id} has no {filename} in {snapshot}")
    _log(f"{model_name} checkpoint: {target}")
    return str(target)


def materialize_dataset(model_name: str, dataset_path: str | None, revision: str | None = None) -> str:
    """Return a local dataset directory, downloading the repo when needed."""
    default = _DATASETS.get(model_name)
    text = str(dataset_path or "").strip()
    if text.startswith("synthetic:") or text.endswith(".pt"):
        return text
    local = Path(text).expanduser() if text else None
    if local is not None and local.exists():
        return str(local)
    if text and _is_hf_repo(text):
        repo_id = text
        rev = revision
    elif not text and default is not None:
        repo_id, rev = default
    else:
        return text
    if repo_id == "Wild-Heart/Disney-VideoGeneration-Dataset" and not rev:
        rev = "refs/pr/4"
    return _snapshot_dataset(repo_id, revision=rev)


def prefetch_huggingface_repo(value: str | None) -> None:
    text = str(value or "").strip()
    if text and _is_hf_repo(text):
        # Root weight and tokenizer files only. Nested fp16/onnx/openvino copies
        # are not used by the SDXL CLIP encoders.
        _snapshot(
            text,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model", "*.tiktoken"],
        )


def _blank(value: object) -> bool:
    text = str(value or "").strip()
    return not text or text.startswith("/path/to/")


def _checkpoint_root(weights: str) -> Path:
    path = Path(weights)
    if path.is_file() or path.name == "transformer":
        return path.parent
    return path


def tokenizer_dir(model_path: str) -> str:
    """Tokenizer folder for a downloaded text encoder.

    Diffusers snapshots keep ``tokenizer/`` next to ``text_encoder/``.
    """
    path = Path(model_path)
    if (path / "tokenizer_config.json").is_file():
        return str(path)
    sibling = path.parent / "tokenizer"
    if (sibling / "tokenizer_config.json").is_file():
        return str(sibling)
    return str(path)


def _require_file(path: Path, description: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not in the downloaded snapshot: {path}")
    return str(path)


def _companion_paths(model_name: str, weights: str, encoder: dict[str, object]) -> dict[str, str]:
    """Local encoder files that belong to the downloaded checkpoint."""
    if not weights:
        return {}
    root = _checkpoint_root(weights)
    found: dict[str, str] = {}
    if model_name in {"sdxl", "stable-diffusion-xl"} and _blank(encoder.get("autoencoder")):
        vae = Path(weights) if Path(weights).is_file() else root / "sd_xl_base_1.0.safetensors"
        found["VAE_CHECKPOINT"] = _require_file(vae, "SDXL autoencoder")
    return found


ENCODER_ENV_KEYS = {
    "VAE_CHECKPOINT": "autoencoder",
}


def _replace_with_download(value: str | None) -> bool:
    text = str(value or "").strip()
    return not text or text.startswith("/path/to/") or _is_hf_repo(text)


def stage_benchmark_assets(
    model_name: str,
    *,
    dataset_path: str | None,
    pretrained: str | None,
    revision: str | None = None,
    encoder: dict[str, object] | None = None,
) -> dict[str, str]:
    """Download benchmark weights and the dataset, and return env exports."""
    env = cache_env()
    encoder = dict(encoder or {})
    weights = materialize_pretrained(model_name, pretrained)
    if weights and _replace_with_download(pretrained):
        env["PRETRAINED_PATH"] = weights
    data = materialize_dataset(model_name, dataset_path, revision=revision)
    if data and data != str(dataset_path or "").strip() and _replace_with_download(dataset_path):
        env["DATASET_PATH"] = data
    if os.environ.get("BENCHMARK_DENOISER_ONLY") != "1":
        env.update(_companion_paths(model_name, weights, encoder))
        for value in encoder.values():
            if isinstance(value, str):
                prefetch_huggingface_repo(value)
    return env
