###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Processes video and text data for training.
"""

from collections.abc import Sequence
from typing import Any

import torch


class WanVideoDataProcessor:
    """Standalone data processor for WanVideo training."""

    def __init__(self, config, model_id=None):
        self.config = config
        self.model_id = model_id
        self.processor = None
        self.tokenizer = None

    def apply_prompt_template(self, hf_messages: str) -> str:
        """Apply prompt template for WanVideo."""
        # WanVideo uses direct prompts without special formatting.
        # Keep backward compatibility:
        # - old path passed `hf_messages` list[dict]
        # - new path passes plain prompt string
        if isinstance(hf_messages, str):
            return hf_messages
        try:
            return hf_messages[0]["content"][1]["text"]
        except Exception:
            # Best-effort fallback
            return str(hf_messages)

    def save_pretrained(self, save_directory: str):
        pass

    def build(self):
        """Initialize the processor and tokenizer."""
        if self.processor is not None and self.tokenizer is not None:
            return

        from transformers import AutoTokenizer

        from primus.backends.diffusion.data.processing_wanvideo import WanVideoProcessor as WanVideoModelProcessor

        wanvideo_kwargs = self.config.get("extra_kwargs", {})
        max_text_length = self.config.get("max_text_length")
        if max_text_length is not None:
            wanvideo_kwargs.setdefault("max_text_length", max_text_length)

        # Load tokenizer if specified
        if self.config.get("text_tokenizer", None) is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("text_tokenizer"))
        else:
            self.tokenizer = None
        self.processor = WanVideoModelProcessor(**wanvideo_kwargs, tokenizer=self.tokenizer)

        if self.tokenizer is None:
            self.tokenizer = self.processor.tokenizer

    def _normalize_raw_batch(self, batch: Any) -> tuple[list[str], list[Any], int | None]:
        """
        Normalize dataloader output into:
          - prompts: list[str]
          - frames_list: list[Any]  (each item is typically np.ndarray (T,H,W,C))
          - num_frames: Optional[int] (only if consistent across samples)
        """
        if isinstance(batch, dict):
            prompts = batch.get("prompt")
            frames_list = batch.get("video_frames")
            num_frames = batch.get("num_frames", None)

            if prompts is None or frames_list is None:
                raise ValueError("prepare_batch(dict) requires keys: 'prompt' and 'video_frames'")
            if isinstance(prompts, str):
                prompts = [prompts]
            if not isinstance(prompts, (list, tuple)):
                raise TypeError(f"'prompt' must be str or list[str], got {type(prompts)}")
            if not isinstance(frames_list, (list, tuple)):
                raise TypeError(f"'video_frames' must be list, got {type(frames_list)}")

            if isinstance(num_frames, (list, tuple)):
                nfs = [int(x) for x in num_frames if x]
                num_frames = nfs[0] if nfs and all(int(x) == int(nfs[0]) for x in nfs) else None
            elif num_frames:
                num_frames = int(num_frames)
            else:
                num_frames = None

            return list(prompts), list(frames_list), num_frames

        # RawBatchCollator returns list[dict]
        if not isinstance(batch, (list, tuple)) or not batch:
            raise ValueError(f"prepare_batch expected non-empty list/tuple, got {type(batch)}")

        prompts = [str(ex.get("prompt", "")) for ex in batch]
        frames_list = [ex.get("video_frames") for ex in batch]

        # Prefer per-example num_frames if present and consistent; else None.
        nfs = [ex.get("num_frames") for ex in batch if ex.get("num_frames")]
        num_frames = nfs[0] if nfs and all(int(x) == int(nfs[0]) for x in nfs) else None
        num_frames = int(num_frames) if num_frames else None

        return prompts, frames_list, num_frames

    def _tokenize_prompts(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        formatted_prompts = [self.apply_prompt_template(p) for p in prompts]
        return self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=self.config.get("padding_strategy", "max_length"),
            truncation=True,
            max_length=self.config.get("max_text_length", 512),
        )

    def _preprocess_videos(self, frames_list: Sequence[Any], num_frames: int | None) -> torch.Tensor:
        if any(v is None for v in frames_list):
            raise ValueError("prepare_batch got None in 'video_frames'")
        video_inputs = self.processor.image_processor.preprocess(
            list(frames_list),
            num_frames=num_frames,
            return_tensors="pt",
        )
        if "pixel_values" not in video_inputs:
            raise KeyError("image_processor.preprocess must return dict with key 'pixel_values'")
        pixel_values = video_inputs["pixel_values"]
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(f"pixel_values must be torch.Tensor, got {type(pixel_values)}")
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected pixel_values shape [B,C,T,H,W], got {tuple(pixel_values.shape)}")
        return pixel_values

    def _assemble_model_batch(
        self, *, pixel_values: torch.Tensor, text_inputs: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        if "input_ids" not in text_inputs or "attention_mask" not in text_inputs:
            raise KeyError("tokenizer output must include 'input_ids' and 'attention_mask'")

        out: dict[str, Any] = {
            "video": pixel_values,  # [B,C,T,H,W]
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

        # Legacy convenience fields (some models expect these keys).
        out.setdefault("num_frames", int(pixel_values.shape[2]))
        out.setdefault("height", int(pixel_values.shape[3]))
        out.setdefault("width", int(pixel_values.shape[4]))
        return out

    def prepare_batch(self, *, batch: Any, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        """
        Convert raw dataloader batch into model inputs.

        Expected raw batch from `RawBatchCollator`:
          - batch: list[dict] with keys:
              - video_frames: np.ndarray (T,H,W,C) or equivalent
              - prompt: str
              - num_frames: int (optional)

        Output matches current training pipelines:
          - video: Tensor [B, C, T, H, W]
          - input_ids: Tensor [B, L]
          - attention_mask: Tensor [B, L]
          - (optional) height/width/num_frames for legacy wan/wan_new paths
        """
        if self.processor is None or self.tokenizer is None:
            # Lazy init to keep behavior robust across call sites.
            self.build()

        prompts, frames_list, num_frames = self._normalize_raw_batch(batch)
        text_inputs = self._tokenize_prompts(prompts)
        pixel_values = self._preprocess_videos(frames_list, num_frames=num_frames)
        return self._assemble_model_batch(pixel_values=pixel_values, text_inputs=text_inputs)