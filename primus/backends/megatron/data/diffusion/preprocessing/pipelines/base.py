# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Abstract base class for dataset preparation pipelines."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..utils import load_from_directory, load_from_huggingface, load_from_webdataset

logger = logging.getLogger(__name__)


class DatasetPipeline(ABC):
    """Abstract base for all dataset preparation pipelines.

    Subclasses implement run() which executes the full pipeline
    and returns a dict of statistics (samples_processed, shards_written, etc.).
    """

    # Whether HuggingFace sources are loaded in streaming mode. Subclasses that
    # need the full dataset materialized (e.g. for distributed total-count
    # splitting) set this to False.
    HF_STREAMING: bool = True

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]: ...

    def load_data(self, **source_kwargs):
        """Load data based on ``self.source_type``.

        Args:
            **source_kwargs: Source-specific arguments
                - directory: input_dir
                - huggingface: hf_dataset, hf_split, hf_data_files, image/caption keys
                - webdataset: input_path

        Returns:
            Iterator over samples with 'image' and 'caption' keys.
        """
        if self.source_type == "directory":
            logger.info(f"Loading from directory: {source_kwargs['input_dir']}")
            return load_from_directory(source_kwargs["input_dir"])
        elif self.source_type == "huggingface":
            hf_split = source_kwargs.get("hf_split", "train")
            hf_data_files = source_kwargs.get("hf_data_files")
            if hf_data_files:
                logger.info(
                    f"Loading from HuggingFace: {source_kwargs['hf_dataset']} "
                    f"(split: {hf_split}, data_files: {hf_data_files})"
                )
            else:
                logger.info(f"Loading from HuggingFace: {source_kwargs['hf_dataset']} (split: {hf_split})")

            return load_from_huggingface(
                source_kwargs["hf_dataset"],
                split=hf_split,
                streaming=self.HF_STREAMING,
                data_files=hf_data_files,
                image_key=source_kwargs.get("image_key"),
                caption_key=source_kwargs.get("caption_key"),
                image_keys=source_kwargs.get("image_keys"),
                caption_keys=source_kwargs.get("caption_keys"),
            )
        elif self.source_type == "webdataset":
            logger.info(f"Loading from WebDataset: {source_kwargs['input_path']}")
            return load_from_webdataset(source_kwargs["input_path"])
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")
