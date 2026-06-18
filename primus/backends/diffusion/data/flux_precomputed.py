from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from primus.backends.diffusion.data.collator import RawBatchCollator

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _tensor_from_serialized_numpy(value: bytes) -> torch.Tensor:
    array = np.load(io.BytesIO(value))
    tensor = torch.from_numpy(array)
    if tensor.dtype == torch.uint16:
        tensor = tensor.view(torch.bfloat16)
    return tensor


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (bytes, bytearray)):
        return _tensor_from_serialized_numpy(bytes(value))
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        if tensor.dtype == torch.uint16:
            tensor = tensor.view(torch.bfloat16)
        return tensor
    return torch.as_tensor(value)


class FluxPrecomputedDataset(Dataset):
    """Map-style dataset for precomputed FLUX text and VAE encodings."""

    required_fields = ("t5_encodings", "clip_encodings", "mean", "logvar")

    def __init__(self, dataset_path: str):
        if not dataset_path:
            raise ValueError("FLUX precomputed dataset requires `dataset_path`.")
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"FLUX precomputed dataset directory not found: {dataset_path}")
        from datasets import load_from_disk

        self.dataset_path = dataset_path
        self.dataset = load_from_disk(dataset_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        missing = [field for field in self.required_fields if field not in sample]
        if missing:
            raise KeyError(f"FLUX precomputed sample missing fields: {missing}")
        return {field: _to_tensor(sample[field]) for field in self.required_fields}

    def get_collator(self):
        return RawBatchCollator()


class FluxRawImageTextDataset(Dataset):
    """Map-style raw image-text dataset for online FLUX encoding."""

    def __init__(
        self,
        *,
        dataset_path: str | None,
        dataset_format: str = "webdataset",
        dataset_name: str | None = None,
        data_folder: str | None = None,
    ):
        dataset_name = dataset_name or None
        dataset_path, dataset_format = self._resolve_dataset(dataset_name, dataset_path, dataset_format)
        self.dataset_path = dataset_path
        self.dataset_format = dataset_format
        self.dataset_name = dataset_name
        self.data_folder = data_folder
        self._records: list[dict[str, Any]] | None = None

        if dataset_format == "jsonl":
            path = Path(dataset_path)
            if not path.is_file():
                raise FileNotFoundError(f"FLUX raw jsonl metadata not found: {dataset_path}")
            self._records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            self.dataset = None
        elif dataset_format == "hf_dataset":
            if not os.path.isdir(dataset_path):
                raise FileNotFoundError(f"FLUX raw HF dataset directory not found: {dataset_path}")
            from datasets import load_from_disk

            self.dataset = load_from_disk(dataset_path)
        elif dataset_format == "hf_repo":
            from datasets import load_dataset

            self.dataset = load_dataset(dataset_path, split="train")
        elif dataset_format == "webdataset":
            if not os.path.isdir(dataset_path):
                raise FileNotFoundError(f"FLUX raw webdataset directory not found: {dataset_path}")
            from datasets import load_dataset

            self.dataset = load_dataset(
                "webdataset",
                split="train",
                data_dir=dataset_path,
                data_files={"train": "*.tar"},
            )
        else:
            raise ValueError("FLUX raw dataset_format must be one of: jsonl, hf_dataset, hf_repo, webdataset")

    @staticmethod
    def _resolve_dataset(
        dataset_name: str | None,
        dataset_path: str | None,
        dataset_format: str,
    ) -> tuple[str, str]:
        if dataset_name == "cc12m-test":
            return (
                dataset_path or "/mnt/shared/zirui/code/torchtitan-main/tests/assets/cc12m_test",
                "webdataset",
            )
        if dataset_name == "cc12m-wds":
            return dataset_path or "pixparse/cc12m-wds", "hf_repo"
        if not dataset_path:
            raise ValueError("FLUX raw dataset requires either `dataset` or `dataset_path`.")
        return dataset_path, dataset_format

    def __len__(self) -> int:
        if self._records is not None:
            return len(self._records)
        return len(self.dataset)

    @staticmethod
    def _prompt_from_sample(sample: dict[str, Any]) -> str:
        for key in ("txt", "caption", "prompt", "text"):
            if key in sample:
                value = sample[key]
                if isinstance(value, list):
                    value = value[0]
                return str(value)
        raise KeyError("FLUX raw sample requires one of: txt, caption, prompt, text")

    def _image_from_jsonl_sample(self, sample: dict[str, Any]) -> Image.Image:
        image_key = "image" if "image" in sample else "jpg" if "jpg" in sample else "png"
        image_path = Path(str(sample[image_key]))
        if self.data_folder and not image_path.is_absolute():
            image_path = Path(self.data_folder) / image_path
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _image_from_dataset_sample(sample: dict[str, Any]) -> Image.Image:
        if "image" in sample:
            image = sample["image"]
        elif "jpg" in sample:
            image = sample["jpg"]
        elif "png" in sample:
            image = sample["png"]
        else:
            raise KeyError("FLUX raw dataset sample requires image, jpg, or png")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        return image.convert("RGB")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._records is not None:
            sample = self._records[idx]
            image = self._image_from_jsonl_sample(sample)
        else:
            sample = self.dataset[idx]
            image = self._image_from_dataset_sample(sample)
        return {"image": image, "prompt": self._prompt_from_sample(sample)}

    def get_collator(self):
        return RawBatchCollator()


@dataclass
class FluxPrecomputedProcessor:
    config: dict[str, Any]

    def __post_init__(self):
        self.prompt_dropout_prob = float(self.config.get("prompt_dropout_prob", 0.0) or 0.0)
        self.empty_encodings_path = self.config.get("empty_encodings_path")
        self.img_size = int(self.config.get("img_size", 256) or 256)
        self._empty_t5: torch.Tensor | None = None
        self._empty_clip: torch.Tensor | None = None

    def build(self):
        if self.prompt_dropout_prob > 0.0:
            self._load_empty_encodings()

    @staticmethod
    def _normalize_empty_encoding(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim >= 1 and tensor.shape[0] == 1:
            return tensor[0]
        return tensor

    def _load_empty_encodings(self):
        if self._empty_t5 is not None and self._empty_clip is not None:
            return
        if not self.empty_encodings_path:
            raise ValueError("FLUX prompt dropout requires `data.empty_encodings_path`.")
        t5_path = os.path.join(self.empty_encodings_path, "t5_empty.npy")
        clip_path = os.path.join(self.empty_encodings_path, "clip_empty.npy")
        if not os.path.isfile(t5_path) or not os.path.isfile(clip_path):
            raise FileNotFoundError(
                "FLUX empty encodings must contain t5_empty.npy and clip_empty.npy "
                f"under {self.empty_encodings_path}"
            )
        self._empty_t5 = self._normalize_empty_encoding(torch.from_numpy(np.load(t5_path)))
        self._empty_clip = self._normalize_empty_encoding(torch.from_numpy(np.load(clip_path)))

    @staticmethod
    def _collate_raw(batch: Any) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            return {key: _to_tensor(value) for key, value in batch.items()}
        if not isinstance(batch, Sequence) or not batch:
            raise ValueError(f"FLUX prepare_batch expected non-empty sequence, got {type(batch).__name__}")

        keys = ("t5_encodings", "clip_encodings", "mean", "logvar")
        collated: dict[str, torch.Tensor] = {}
        for key in keys:
            collated[key] = torch.stack([_to_tensor(sample[key]) for sample in batch], dim=0)
        return collated

    def prepare_batch(self, *, batch: Any, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        tensors = self._collate_raw(batch)
        for key, value in tensors.items():
            tensors[key] = value.to(device=device, dtype=dtype, non_blocking=True)

        if self.prompt_dropout_prob > 0.0:
            self._load_empty_encodings()
            assert self._empty_t5 is not None and self._empty_clip is not None
            bsz = tensors["t5_encodings"].shape[0]
            drop_mask = torch.rand((bsz,), device=device) < self.prompt_dropout_prob
            if drop_mask.any():
                tensors["t5_encodings"][drop_mask] = self._empty_t5.to(device=device, dtype=dtype)
                tensors["clip_encodings"][drop_mask] = self._empty_clip.to(device=device, dtype=dtype)

        return tensors


@dataclass
class FluxRawImageTextProcessor:
    config: dict[str, Any]

    def __post_init__(self):
        self.prompt_dropout_prob = float(self.config.get("prompt_dropout_prob", 0.0) or 0.0)
        self.img_size = int(self.config.get("img_size", 256) or 256)
        self.skip_low_resolution = bool(self.config.get("skip_low_resolution", True))

    def build(self):
        return

    def _process_image(self, image: Image.Image) -> torch.Tensor | None:
        width, height = image.size
        if self.skip_low_resolution and (width < self.img_size or height < self.img_size):
            return None

        if width == self.img_size and height == self.img_size:
            resized = image
        elif width >= height:
            new_width, new_height = math.ceil(self.img_size / height * width), self.img_size
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
            left = int(torch.randint(0, new_width - self.img_size + 1, (1,)).item())
            resized = image.crop((left, 0, left + self.img_size, self.img_size))
        else:
            new_width, new_height = self.img_size, math.ceil(self.img_size / width * height)
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
            lower = int(torch.randint(0, new_height - self.img_size + 1, (1,)).item())
            resized = image.crop((0, lower, self.img_size, lower + self.img_size))

        if resized.mode != "RGB":
            resized = resized.convert("RGB")
        np_img = np.array(resized).transpose((2, 0, 1))
        return torch.tensor(np_img).float() / 255.0 * 2.0 - 1.0

    def prepare_batch(self, *, batch: Any, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        if isinstance(batch, dict):
            batch = [batch]
        if not isinstance(batch, Sequence) or not batch:
            raise ValueError(f"FLUX raw prepare_batch expected non-empty sequence, got {type(batch).__name__}")

        images: list[torch.Tensor] = []
        prompts: list[str] = []
        for sample in batch:
            image = self._process_image(sample["image"])
            if image is None:
                continue
            prompt = str(sample.get("prompt", ""))
            if self.prompt_dropout_prob > 0.0 and torch.rand(1).item() < self.prompt_dropout_prob:
                prompt = ""
            images.append(image)
            prompts.append(prompt)

        if not images:
            raise ValueError("FLUX raw batch contained no usable images after preprocessing")
        return {
            "image": torch.stack(images, dim=0).to(device=device, dtype=dtype, non_blocking=True),
            "prompts": prompts,
        }
