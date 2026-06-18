###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from typing import Any, Literal

from pydantic import BaseModel, field_validator


class Args(BaseModel):
    extra_kwargs: dict[str, Any] = {}

    def to_dict(self):
        return self.model_dump()

    def to_json(self):
        return self.model_dump_json()


class ProcessorConfig(Args):
    processor_name: str
    processor_type: str


class DatasetConfig(Args):
    dataset_type: str
    data_folder: str
    dataset_format: Literal["json", "jsonl", "csv", "yaml", "hf_dataset", "arrow"]
    processor_config: dict | ProcessorConfig

    # Dataset configuration
    dataset_path: str | None = None  # Optional - used for external files
    datasets: list[dict] | None = None  # Optional - used for inline YAML definitions
    shuffle: bool = True
    data_seed: int | None = 42
    eval_dataset_path: str | None = None

    # Object storage configuration
    object_storage: Literal["azure", "gcs", "none"] | None = "none"
    bucket_name: str | None = None

    # Packing configuration
    packing: bool | None = False
    packing_strategy: str | None = None
    packing_length: int | None = 32000
    filter_overlong: bool | None = True
    filter_overlong_workers: int | None = 8
    max_length: int | None = None

    # Video configuration
    video_sampling_strategy: Literal["fps", "frame_num"] | None = "fps"
    video_max_pixels: int | None = 768 * 28 * 28
    video_max_frames: int | None = 768
    video_min_pixels: int | None = 3136
    frame_num: int | None = 64
    fps: int | None = 1
    video_backend: Literal["decord", "qwen_vl_utils", "qwen_omni_utils", "imageio"] | None = "qwen_vl_utils"

    @field_validator(
        "video_max_pixels",
        "video_max_frames",
        "frame_num",
        "fps",
        "packing_length",
        "max_length",
        "filter_overlong_workers",
    )
    @classmethod
    def validate_positive_values(cls, v, info):
        """Validate that numeric video and packing parameters are positive."""
        if v is not None and v <= 0:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v

    @field_validator("video_backend")
    @classmethod
    def validate_video_backend_migration(cls, v):
        """Provide migration warning for deprecated torchvision backend."""
        if v == "torchvision":
            raise ValueError(
                "The 'torchvision' video backend has been removed. "
                "Please use 'decord', 'qwen_vl_utils', or 'qwen_omni_utils' instead. "
                "Migration guide: If you were using torchvision, 'decord' provides "
                "similar functionality with better performance."
            )
        return v