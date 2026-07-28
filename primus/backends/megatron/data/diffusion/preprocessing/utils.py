# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Common utilities for dataset preprocessing.

This module provides shared functionality for creating WebDataset shards
from various input sources (HuggingFace Hub, directories, existing WebDatasets).
"""

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import webdataset as wds
from PIL import Image

logger = logging.getLogger(__name__)


def get_distributed_info() -> Tuple[int, int]:
    """
    Get distributed processing information (rank, world_size).

    Returns:
        Tuple of (rank, world_size). Returns (0, 1) if not in distributed mode.

    Example:
        >>> rank, world_size = get_distributed_info()
        >>> if rank == 0:
        ...     print("This is the master process")
    """
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            return rank, world_size
    except (ImportError, AttributeError):
        pass

    # Fallback to environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def split_work_for_rank(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    """
    Calculate start/end indices for this rank in distributed processing.

    Args:
        total: Total number of items to process
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes

    Returns:
        Tuple of (start_idx, end_idx) for this rank

    Example:
        >>> start, end = split_work_for_rank(1000, 0, 4)
        >>> print(f"Process 0 handles items {start} to {end}")
        Process 0 handles items 0 to 250
    """
    items_per_rank = total // world_size
    start_idx = rank * items_per_rank

    if rank == world_size - 1:
        # Last rank gets all remaining items (including remainder)
        end_idx = total
    else:
        end_idx = start_idx + items_per_rank

    return start_idx, end_idx


def save_to_webdataset(
    samples: List[Dict[str, Any]],
    output_dir: str,
    shard_size: int,
    shard_offset: int = 0,
    compress: bool = False,
) -> int:
    """
    Save samples to WebDataset tar shards.

    Args:
        samples: List of sample dictionaries with keys as extensions
            Example: {'images': image_bytes, 'txt': caption_text}
        output_dir: Directory to save shards
        shard_size: Number of samples per shard
        shard_offset: Starting shard number (for distributed processing)
        compress: Whether to compress tar files (gzip)

    Returns:
        Number of shards created

    Example:
        >>> samples = [
        ...     {'images': image_bytes, 'txt': 'a cat'},
        ...     {'images': image_bytes2, 'txt': 'a dog'},
        ... ]
        >>> num_shards = save_to_webdataset(samples, '/path/to/output', shard_size=1000)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate number of shards needed
    num_shards = (len(samples) + shard_size - 1) // shard_size

    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, len(samples))
        shard_samples = samples[start_idx:end_idx]

        # Create shard filename
        shard_num = shard_offset + shard_idx
        ext = ".tar.gz" if compress else ".tar"
        shard_path = os.path.join(output_dir, f"{shard_num:06d}{ext}")

        # Write shard
        with wds.TarWriter(shard_path) as sink:
            for sample_idx, sample in enumerate(shard_samples):
                # Generate unique key for this sample
                key = f"{shard_num:06d}_{sample_idx:06d}"

                # Create WebDataset sample dict
                wds_sample = {"__key__": key}

                for ext, data in sample.items():
                    if ext.startswith("__"):
                        continue  # Skip special keys

                    # Handle different data types
                    if isinstance(data, bytes):
                        wds_sample[ext] = data
                    elif isinstance(data, str):
                        wds_sample[ext] = data.encode("utf-8")
                    elif isinstance(data, torch.Tensor):
                        # Save tensor to bytes
                        buffer = io.BytesIO()
                        torch.save(data, buffer)
                        wds_sample[ext] = buffer.getvalue()
                    elif isinstance(data, Image.Image):
                        # Save PIL image to bytes
                        buffer = io.BytesIO()
                        data.save(buffer, format="JPEG")
                        wds_sample[ext] = buffer.getvalue()
                    else:
                        logger.warning(f"Unknown data type for key {ext}: {type(data)}")
                        continue

                sink.write(wds_sample)

        logger.info(f"Created shard {shard_path} with {len(shard_samples)} samples")

    return num_shards


# Default keys to try when loading from HuggingFace if not specified in config
DEFAULT_IMAGE_KEYS = ["image", "jpg", "jpeg", "png", "img", "photo"]
DEFAULT_CAPTION_KEYS = ["caption", "text", "txt", "description", "prompt"]


def _extract_field(item: dict, field_keys: List[str], idx: int, field_type: str) -> Any:
    """
    Extract field from item using list of possible keys.
    Supports dot notation for JSON paths (e.g., 'json.caption').

    Args:
        item: Dataset sample dictionary
        field_keys: List of keys to try (e.g., ['caption', 'json.caption'])
        idx: Sample index (for logging)
        field_type: 'image' or 'caption' (currently unused)

    Returns:
        Field value if found, None otherwise

    Example:
        >>> item = {'jpg': image_bytes, 'json': {'caption': 'a cat'}}
        >>> _extract_field(item, ['caption', 'json.caption'], 0, 'caption')
        'a cat'
    """
    for key in field_keys:
        if "." in key:
            # Handle JSON path (e.g., 'json.caption')
            value = _extract_json_path(item, key, idx)
            if value is not None:
                return value
        elif key in item:
            return item[key]
    return None


def _extract_json_path(item: dict, path: str, idx: int) -> Any:
    """
    Extract value from nested JSON using dot notation.

    Args:
        item: Dataset sample dictionary
        path: Dot-separated path (e.g., 'json.metadata.caption')
        idx: Sample index (for logging)

    Returns:
        Extracted value if found, None otherwise

    Example:
        >>> item = {'json': b'{"caption": "a cat"}'}
        >>> _extract_json_path(item, 'json.caption', 0)
        'a cat'
    """
    import json as json_module

    parts = path.split(".")
    current = item

    for i, part in enumerate(parts):
        if part not in current:
            return None

        current = current[part]

        # If we hit a JSON string/bytes at any level, parse it
        if isinstance(current, (bytes, str)):
            try:
                if isinstance(current, bytes):
                    current = json_module.loads(current.decode("utf-8"))
                elif isinstance(current, str) and (current.startswith("{") or current.startswith("[")):
                    current = json_module.loads(current)
            except Exception as e:
                logger.debug(f"Sample {idx} failed to parse JSON at '{part}': {e}")
                return None

    return current if isinstance(current, (str, int, float)) else None


def load_from_huggingface(
    dataset_name: str,
    split: str = "train",
    streaming: bool = True,
    data_files: Optional[Union[str, List[str]]] = None,
    image_key: Optional[str] = None,
    caption_key: Optional[str] = None,
    image_keys: Optional[List[str]] = None,
    caption_keys: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Load dataset from HuggingFace Hub with configurable field mappings.

    Args:
        dataset_name: HF dataset identifier (e.g., 'laion/laion400m')
        split: Dataset split to load
        streaming: Whether to stream the dataset (recommended for large datasets)
        data_files: Specific files/paths to load from the dataset (e.g., 'data_1024_10K/*.tar')
        image_key: Single image field name (e.g., 'jpg', 'image')
        caption_key: Single caption field or JSON path (e.g., 'caption', 'json.caption')
        image_keys: List of image field names to try (fallback to defaults if None)
        caption_keys: List of caption fields/paths to try (fallback to defaults if None)

    Yields:
        Dicts with 'image' (PIL Image) and 'caption' (str) keys

    Example:
        >>> # Load with auto-detection (uses defaults)
        >>> for item in load_from_huggingface('diffusers/pokemon-gpt4-captions'):
        ...     print(f"Caption: {item['caption']}")

        >>> # Load with specific field mappings
        >>> for item in load_from_huggingface(
        ...     'jackyhate/text-to-image-2M',
        ...     image_key='jpg',
        ...     caption_key='json.caption',
        ...     data_files='data_1024_10K/*.tar'
        ... ):
        ...     print(f"Caption: {item['caption']}")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace dataset loading. "
            "Install with: pip install datasets"
        )

    if data_files:
        logger.info(f"Loading HuggingFace dataset: {dataset_name} (split: {split}, data_files: {data_files})")
    else:
        logger.info(f"Loading HuggingFace dataset: {dataset_name} (split: {split})")

    dataset = load_dataset(dataset_name, split=split, streaming=streaming, data_files=data_files)

    # Determine which keys to try
    if image_keys is None:
        image_keys = [image_key] if image_key else DEFAULT_IMAGE_KEYS
    if caption_keys is None:
        caption_keys = [caption_key] if caption_key else DEFAULT_CAPTION_KEYS

    for idx, item in enumerate(dataset):
        # Try to extract image using configured keys
        image = _extract_field(item, image_keys, idx, "image")

        # Try to extract caption using configured keys (supports JSON paths)
        caption = _extract_field(item, caption_keys, idx, "caption")

        if isinstance(caption, list):
            caption = caption[0] if caption else None

        if image is None or caption is None:
            logger.warning(f"Sample {idx} missing image or caption, skipping. Keys: {item.keys()}")
            continue

        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            try:
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image))
                else:
                    logger.warning(f"Sample {idx} has unexpected image type: {type(image)}")
                    continue
            except Exception as e:
                logger.warning(f"Sample {idx} failed to load image: {e}")
                continue

        yield {
            "image": image,
            "caption": caption,
        }


def load_from_directory(input_dir: str) -> Iterator[Dict[str, Any]]:
    """
    Load dataset from directory structure.

    Expected structure:
        input_dir/
            images/
                0000000.jpg
                0000001.jpg
            captions/
                0000000.txt
                0000001.txt

    Args:
        input_dir: Root directory containing 'images' and 'captions' subdirectories

    Yields:
        Dicts with 'image' (PIL Image) and 'caption' (str) keys

    Example:
        >>> for item in load_from_directory('/path/to/dataset'):
        ...     print(f"Caption: {item['caption']}")
    """
    input_path = Path(input_dir)
    images_dir = input_path / "images"
    captions_dir = input_path / "captions"

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    if not captions_dir.exists():
        raise ValueError(f"Captions directory not found: {captions_dir}")

    logger.info(f"Loading dataset from directory: {input_dir}")

    # Find all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(sorted(images_dir.glob(f"*{ext}")))

    logger.info(f"Found {len(image_files)} image files")

    for img_path in image_files:
        # Find corresponding caption file
        caption_path = captions_dir / f"{img_path.stem}.txt"

        if not caption_path.exists():
            logger.warning(f"Caption not found for {img_path.name}, skipping")
            continue

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Load caption
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()

            yield {
                "image": image,
                "caption": caption,
            }

        except Exception as e:
            logger.warning(f"Failed to load {img_path.name}: {e}")
            continue


def load_from_webdataset(input_path: str) -> Iterator[Dict[str, Any]]:
    """
    Load existing WebDataset with non-standard keys.

    Converts to standard format {'image': PIL Image, 'caption': str}

    Args:
        input_path: Path or glob pattern to WebDataset tar files
            Examples: '/path/to/shards/*.tar', '/path/to/shard_000000.tar'

    Yields:
        Dicts with 'image' (PIL Image) and 'caption' (str) keys

    Example:
        >>> for item in load_from_webdataset('/path/to/dataset/*.tar'):
        ...     print(f"Caption: {item['caption']}")
    """
    logger.info(f"Loading WebDataset from: {input_path}")

    dataset = wds.WebDataset(input_path)

    for sample in dataset:
        try:
            # Try to find image with various keys
            image = None
            for img_key in ["jpg", "png", "jpeg", "webp", "image"]:
                if img_key in sample:
                    img_data = sample[img_key]
                    if isinstance(img_data, bytes):
                        image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    elif isinstance(img_data, Image.Image):
                        image = img_data.convert("RGB")
                    else:
                        continue
                    break

            # Try to find caption with various keys
            caption = None
            for cap_key in ["txt", "caption.txt", "text", "caption"]:
                if cap_key in sample:
                    cap_data = sample[cap_key]
                    if isinstance(cap_data, bytes):
                        caption = cap_data.decode("utf-8")
                    elif isinstance(cap_data, str):
                        caption = cap_data
                    else:
                        continue
                    break

            if image is None or caption is None:
                logger.warning(f"Sample missing image or caption, skipping. Keys: {sample.keys()}")
                continue

            yield {
                "image": image,
                "caption": caption,
            }

        except Exception as e:
            logger.warning(f"Failed to load sample: {e}")
            continue


def preprocess_image(
    image: Image.Image,
    variable_size: bool = True,  # if True, then image is resized to the nearest multiple of 16, otherwise it is resized to the given size
    size: int = 1024,  # only used if variable_size is False, then this is applied
    center_crop: bool = False,  # only used if variable_size is False, then this is applied
    max_size: int = 1024,  # maximum dimension when variable_size is True
) -> Image.Image:
    """
    Preprocess image (resize, crop).

    Args:
        image: PIL Image
        variable_size: If True, resize to nearest multiple of 16 up to max_size
        size: Target size (square) - only used if variable_size is False
        center_crop: Whether to center crop before resize - only used if variable_size is False
        max_size: Maximum dimension when variable_size is True (default: 1024)

    Returns:
        Preprocessed PIL Image

    Example:
        >>> image = Image.open('photo.jpg')
        >>> processed = preprocess_image(image, variable_size=True, max_size=2048)
    """
    # Center crop if requested
    if center_crop and variable_size is False:
        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        image = image.crop((left, top, right, bottom))

    # Resize
    if variable_size is False:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    else:
        width, height = image.size
        if max(width, height) > max_size:
            scale = max_size / float(max(width, height))
            new_w = int(round(width * scale))
            new_h = int(round(height * scale))
            # High-quality downsampling
            image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        width, height = image.size
        new_size = (round(width / 16) * 16, round(height / 16) * 16)
        image = image.resize(new_size, Image.LANCZOS)

    return image


def encode_image_to_bytes(image: Image.Image, format: str = "JPEG", quality: int = 95) -> bytes:
    """
    Encode PIL Image to bytes.

    Args:
        image: PIL Image
        format: Image format ('JPEG', 'PNG', 'WEBP')
        quality: JPEG/WEBP quality (1-100)

    Returns:
        Image bytes

    Example:
        >>> image = Image.open('photo.jpg')
        >>> image_bytes = encode_image_to_bytes(image, format='JPEG', quality=95)
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


__all__ = [
    "get_distributed_info",
    "split_work_for_rank",
    "save_to_webdataset",
    "load_from_huggingface",
    "load_from_directory",
    "load_from_webdataset",
    "preprocess_image",
    "encode_image_to_bytes",
]
