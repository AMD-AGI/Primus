###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Hugging Face datasets used by the AMDiffusionBenchmark workloads."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import torch

from primus.backends.diffusion.data.collator import RawBatchCollator

_SIHEYUAN = "chinese_architecture_siheyuan"
_PROMPT_KEYS = ("prompt", "text", "caption", "txt", "caption_0")


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def is_siheyuan_dataset(dataset_path: str) -> bool:
    text = str(dataset_path).replace("\\", "/")
    local = Path(text).expanduser()
    return _SIHEYUAN in text or (local.is_dir() and (local / "siheyuan.zip").is_file())


def siheyuan_image_members(names: list[str]) -> list[str]:
    return sorted(
        name for name in names if not name.endswith("/") and Path(name).suffix.lower() in _IMAGE_SUFFIXES
    )


def read_siheyuan_images(dataset_path: str) -> list[tuple[str, bytes]] | None:
    """The 156-image train split: JPEGs from both archives, then the log PNGs.

    ``load_dataset`` on this repo also yields npz, txt, and caption members, so
    the decoded split is larger than the real training images.
    """
    local = Path(str(dataset_path)).expanduser()
    if not local.is_dir() or not is_siheyuan_dataset(str(local)):
        return None
    rows: list[tuple[str, bytes]] = []
    for archive_path in sorted(local.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            for name in siheyuan_image_members(archive.namelist()):
                rows.append((name, archive.read(name)))
    for path in sorted(local.rglob("*.png")):
        rows.append((path.name, path.read_bytes()))
    if not rows:
        raise ValueError(f"{local} has no siheyuan train images")
    return rows


def load_benchmark_image_dataset(dataset_path: str):
    """Load an image dataset. Siheyuan uses its 156-image train split."""
    from datasets import Dataset, Image, load_dataset

    images = read_siheyuan_images(dataset_path)
    if images is not None:
        records = {
            "image": [{"path": name, "bytes": payload} for name, payload in images],
            "caption": [Path(name).stem.replace("_", " ") for name, _payload in images],
        }
        return Dataset.from_dict(records).cast_column("image", Image())
    return load_dataset(str(dataset_path), **image_dataset_load_kwargs(dataset_path))


def image_dataset_load_kwargs(dataset_path: str) -> dict[str, Any]:
    """Load arguments for a Hugging Face image dataset."""
    del dataset_path
    return {"split": "train", "verification_mode": "no_checks"}


def caption_from_image_filename(image: Any) -> str:
    """Caption used by AMDiffusionBenchmark: the filename stem, underscores as spaces."""
    filename = getattr(image, "filename", "") or ""
    stem = Path(str(filename)).stem
    return stem.replace("_", " ") if stem else ""


class AMDBDataset:
    """One image sample plus an optional caption."""

    def __init__(self, records: Any, *, modality: str):
        if modality != "image":
            raise ValueError(f"amdb modality must be 'image', got {modality!r}")
        self.records = records
        self.modality = modality
        self._logged_skips: set[int] = set()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        count = len(self.records)
        last_error: Exception | None = None
        for offset in range(count):
            row_index = (index + offset) % count
            try:
                return self._sample_from_row(self.records[row_index])
            except (OSError, RuntimeError, ValueError) as exc:
                last_error = exc
                if row_index not in self._logged_skips:
                    self._logged_skips.add(row_index)
                    from primus.backends.diffusion.utils.log import logger

                    logger.warning(
                        "Skipping undecodable amdb %s sample %s: %s",
                        self.modality,
                        row_index,
                        exc,
                    )
        raise RuntimeError(f"amdb dataset has no decodable {self.modality} sample") from last_error

    def _sample_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        sample: dict[str, Any] = {}
        image = row["image"]
        if hasattr(image, "load"):
            image.load()
        sample["image"] = image
        for key in _PROMPT_KEYS:
            if key in row and row[key] is not None:
                value = row[key]
                sample["prompt"] = value[0] if isinstance(value, list) else str(value)
                break
        if "prompt" not in sample:
            caption = caption_from_image_filename(sample.get("image"))
            if caption:
                sample["prompt"] = caption
        return sample

    def get_collator(self):
        return RawBatchCollator()


class AMDBProcessor:
    """Group raw samples. The workload adapter tokenizes and transforms them."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def build(self):
        return None

    def prepare_batch(self, batch, device=None, dtype=None):
        del device, dtype
        if isinstance(batch, dict):
            return batch
        prompts = [sample.get("prompt", "") for sample in batch]
        prepared: dict[str, Any] = {"prompts": prompts}
        if batch and "image" in batch[0]:
            prepared["images"] = [sample["image"] for sample in batch]
        if batch:
            for key, value in batch[0].items():
                if key in {"prompt", "image"} or not torch.is_tensor(value):
                    continue
                prepared[key] = torch.stack([sample[key] for sample in batch])
        return prepared


def _load_saved_dataset(dataset_path: Any) -> list[dict[str, Any]] | None:
    path = Path(str(dataset_path))
    # state.json marks a datasets.save_to_disk directory. A downloaded dataset
    # repo can contain dataset_info.json and must stay on the load_dataset path.
    if not path.is_dir() or not (path / "state.json").is_file():
        return None
    from datasets import load_from_disk

    stored = load_from_disk(str(path))
    if hasattr(stored, "keys") and "train" in getattr(stored, "keys")():
        stored = stored["train"]
    stored = stored.with_format("torch")
    columns = list(stored.column_names)
    return [{column: stored[index][column] for column in columns} for index in range(len(stored))]


def build_amdb_dataset(dataset_config: dict[str, Any]):
    from datasets import load_dataset

    from primus.backends.diffusion.utils.log import logger

    modality = str(dataset_config.get("modality", "image"))
    if modality != "image":
        raise ValueError(f"amdb modality must be 'image', got {modality!r}")

    from primus.backends.diffusion.utils.download import materialize_dataset

    dataset_path = materialize_dataset(
        str(dataset_config.get("model_name") or ""),
        dataset_config.get("dataset_path"),
        revision=dataset_config.get("revision"),
    )
    if not dataset_path:
        raise ValueError("amdb dataset requires data.dataset_path")
    if str(dataset_path).endswith(".pt"):
        samples = torch.load(str(dataset_path), map_location="cpu", weights_only=True)
        dataset = _CachedTensorDataset(samples)
        processor = AMDBProcessor(dataset_config)
        processor.build()
        logger.info(f"Built cached {modality} dataset with {len(dataset)} samples from {dataset_path}")
        return dataset, processor
    stored = _load_saved_dataset(dataset_path)
    if stored is not None:
        dataset = _CachedTensorDataset(stored)
        processor = AMDBProcessor(dataset_config)
        processor.build()
        logger.info(f"Built saved {modality} dataset with {len(dataset)} samples from {dataset_path}")
        return dataset, processor
    local_dir = Path(str(dataset_path)).expanduser()
    revision = None if local_dir.is_dir() else dataset_config.get("revision")
    if not local_dir.is_dir() and not revision:
        revision = "main"
    if is_siheyuan_dataset(str(dataset_path)) and Path(str(dataset_path)).expanduser().is_dir():
        records = load_benchmark_image_dataset(str(dataset_path))
    else:
        load_kwargs = image_dataset_load_kwargs(str(dataset_path))
        load_kwargs["split"] = dataset_config.get("split", load_kwargs["split"])
        if revision:
            load_kwargs["revision"] = revision
        records = load_dataset(str(dataset_path), **load_kwargs)
    max_samples = int(dataset_config.get("max_train_samples") or -1)
    if max_samples > 0:
        records = records.select(range(min(max_samples, len(records))))
    dataset = AMDBDataset(records, modality=modality)
    processor = AMDBProcessor(dataset_config)
    processor.build()
    logger.info(f"Built amdb {modality} dataset with {len(dataset)} samples from {dataset_path}")
    return dataset, processor


class _CachedTensorDataset:
    """Samples encoded from the AMDiffusionBenchmark source datasets."""

    def __init__(self, samples: list[dict[str, Any]]):
        if not samples:
            raise ValueError("Cached benchmark dataset is empty")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.samples[index])
        if not sample.get("prompt"):
            for key in _PROMPT_KEYS:
                if sample.get(key):
                    value = sample[key]
                    sample["prompt"] = value[0] if isinstance(value, list) else str(value)
                    break
        return sample

    def get_collator(self):
        return RawBatchCollator()
