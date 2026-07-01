###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)


class WanVideoImageProcessor(BaseImageProcessor):
    """
    Image/Video processor for WanVideo models.

    Args:
        do_resize: Whether to resize the image/video frames.
        size: Target size for resizing.
        do_center_crop: Whether to center crop.
        crop_size: Size for center cropping.
        do_normalize: Whether to normalize pixel values.
        image_mean: Mean values for normalization.
        image_std: Standard deviation values for normalization.
        do_convert_rgb: Whether to convert to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        do_center_crop: bool = True,
        crop_size: Optional[Dict[str, int]] = None,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        max_pixels: Optional[int] = None,
        height_division_factor: int = 16,
        width_division_factor: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        **kwargs,
    ) -> np.ndarray:
        """Resize image or video frame."""
        from PIL import Image as PILImage

        image = PILImage.fromarray(image.astype(np.uint8))
        # TODO: Here, we align with DiffSynth's resize logic for debugging(may remove in the future)
        if os.getenv("ALIGN_WITH_DIFFSYNTH") == "1":
            # Match DiffSynth logic: scale based on max dimension ratio
            width, height = image.size
            target_width = size["width"]
            target_height = size["height"]

            scale = max(target_width / width, target_height / height)
            new_width = round(width * scale)
            new_height = round(height * scale)

            from torchvision.transforms import InterpolationMode
            from torchvision.transforms import functional as F

            # DiffSynth uses torchvision.transforms.resize with BILINEAR
            # We must use F.resize to match exactly (antialias behavior etc)
            image = F.resize(image, (new_height, new_width), interpolation=InterpolationMode.BILINEAR)
        else:
            image = image.resize((size["width"], size["height"]), PILImage.LANCZOS)
        return np.array(image)

    def _extract_hw(self, image: Union[np.ndarray, Image.Image]) -> Tuple[int, int]:
        if isinstance(image, Image.Image):
            width, height = image.size
            return height, width
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                return image.shape[0], image.shape[1]
            if image.ndim == 2:
                return image.shape[0], image.shape[1]
        raise ValueError("Unsupported image type for size extraction.")

    def _dynamic_target_size(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, int]:
        height, width = self._extract_hw(image)
        if self.max_pixels is not None and width * height > self.max_pixels:
            scale = (width * height / self.max_pixels) ** 0.5
            height = int(height / scale)
            width = int(width / scale)
        height = height // self.height_division_factor * self.height_division_factor
        width = width // self.width_division_factor * self.width_division_factor
        height = max(self.height_division_factor, height)
        width = max(self.width_division_factor, width)
        return {"height": height, "width": width}

    def _resolve_sizes(self, image: Union[np.ndarray, Image.Image]) -> Tuple[Dict[str, int], Dict[str, int]]:
        size = self.size
        crop_size = self.crop_size
        if size is None and crop_size is None:
            size = self._dynamic_target_size(image)
            crop_size = dict(size)
        elif size is None:
            size = dict(crop_size)
        elif crop_size is None:
            crop_size = dict(size)
        size = {
            "height": self._ceil_to_factor(size["height"], self.height_division_factor),
            "width": self._ceil_to_factor(size["width"], self.width_division_factor),
        }
        crop_size = {
            "height": self._ceil_to_factor(crop_size["height"], self.height_division_factor),
            "width": self._ceil_to_factor(crop_size["width"], self.width_division_factor),
        }
        return size, crop_size

    @staticmethod
    def _ceil_to_factor(value: int, factor: int) -> int:
        if value % factor == 0:
            return value
        return (value + factor - 1) // factor * factor

    def center_crop(
        self,
        image: np.ndarray,
        crop_size: Dict[str, int],
        **kwargs,
    ) -> np.ndarray:
        """Center crop image or video frame."""
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size["height"], crop_size["width"]

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        if image.ndim == 3:
            return image[top : top + crop_h, left : left + crop_w, :]
        else:
            return image[top : top + crop_h, left : left + crop_w]

    def normalize(
        self,
        image: np.ndarray,
        mean: List[float],
        std: List[float],
        **kwargs,
    ) -> np.ndarray:
        """Normalize image or video frame."""
        image = image.astype(np.float32) / 255.0
        mean = np.array(mean).reshape(1, 1, -1)
        std = np.array(std).reshape(1, 1, -1)
        return (image - mean) / std

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        num_frames: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Preprocess images or video frames.

        Args:
            images: Input images or video frames.
            return_tensors: Type of tensors to return ("pt" for PyTorch).

        Returns:
            Dictionary with preprocessed pixel values.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        # Handle single image or list of images (video frames)
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.array(image)

            # Handle 4D tensor case (batch of video frames)
            if isinstance(image, np.ndarray) and image.ndim == 4:
                # Process each frame in the batch
                batch_processed_frames = []
                for i in range(image.shape[0]):
                    frame = image[i]  # Get single frame
                    if frame.ndim == 3 and (frame.shape[0] == 3 or frame.shape[0] == 1):
                        frame = np.transpose(frame, (1, 2, 0))  # (C, H, W) -> (H, W, C)

                    # Convert to RGB if needed
                    if do_convert_rgb and frame.shape[-1] != 3:
                        if len(frame.shape) == 2:  # Grayscale
                            frame = np.stack([frame] * 3, axis=-1)
                        elif frame.shape[-1] == 4:  # RGBA
                            frame = frame[..., :3]

                    size_, crop_size_ = self._resolve_sizes(frame)
                    if do_resize:
                        frame = self.resize(frame, size_)
                    if do_center_crop:
                        frame = self.center_crop(frame, crop_size_)

                    # Normalize
                    if do_normalize:
                        frame = self.normalize(frame, image_mean, image_std)

                    batch_processed_frames.append(frame)

                # Stack frames back into 4D tensor
                processed_image = np.stack(batch_processed_frames, axis=0)
            else:
                # Handle single image case (3D or 2D)
                # Convert to RGB if needed
                if do_convert_rgb and image.shape[-1] != 3:
                    if len(image.shape) == 2:  # Grayscale
                        image = np.stack([image] * 3, axis=-1)
                    elif image.shape[-1] == 4:  # RGBA
                        image = image[..., :3]

                size_, crop_size_ = self._resolve_sizes(image)
                if do_resize:
                    image = self.resize(image, size_)
                if do_center_crop:
                    image = self.center_crop(image, crop_size_)

                # Normalize
                if do_normalize:
                    image = self.normalize(image, image_mean, image_std)

                processed_image = image

            processed_images.append(processed_image)

        # Stack frames for video
        processed_images = np.stack(processed_images, axis=0)  # B, T, H, W, C

        # Temporal Handling (Interpolate or Truncate)
        if num_frames is not None:
            current_frames = processed_images.shape[1]
            if current_frames > num_frames:
                logger.info(f"Truncating video frames from {current_frames} to {num_frames}")
                processed_images = processed_images[:, :num_frames, ...]
            elif current_frames < num_frames:
                logger.info(f"Interpolating video frames from {current_frames} to {num_frames}")
                # Interpolate requires (B, C, T, H, W) or (B, C, H, W) - we have (B, T, H, W, C)
                # Permute to (B, C, T, H, W) for interpolate
                vid_tensor = torch.from_numpy(processed_images).permute(0, 4, 1, 2, 3)

                # Interpolate
                vid_tensor = torch.nn.functional.interpolate(
                    vid_tensor,
                    size=(num_frames, vid_tensor.shape[3], vid_tensor.shape[4]),
                    mode="trilinear",
                    align_corners=False,
                )

                # Permute back to (B, T, H, W, C) and convert to numpy
                processed_images = vid_tensor.permute(0, 2, 3, 4, 1).numpy()

        # Convert to tensor if requested
        if return_tensors == "pt":
            processed_images = torch.from_numpy(processed_images)
            # Rearrange to (B, C, T, H, W) for video (since input was B, T, H, W, C)
            if processed_images.ndim == 5:
                processed_images = processed_images.permute(0, 4, 1, 2, 3)
            elif processed_images.ndim == 4:
                # (T, H, W, C) -> (C, T, H, W) if it was list of frames
                processed_images = processed_images.permute(3, 0, 1, 2)
            # Add batch dimension
            # processed_images = processed_images.unsqueeze(0)

        return {"pixel_values": processed_images}


class WanVideoProcessor:
    """
    Processor for WanVideo models, combining image/video processing and text tokenization.

    Args:
        image_processor: Image/video processor instance.
        tokenizer: Text tokenizer instance.
    """

    attributes = ["tokenizer", "image_processor"]
    valid_kwargs = [
        "chat_template",
    ]
    tokenizer_class = "AutoTokenizer"
    image_processor_class = "WanVideoImageProcessor"

    def __init__(self, image_processor=None, tokenizer=None, max_text_length: Optional[int] = None, **kwargs):
        if image_processor is None:
            image_processor = WanVideoImageProcessor(**kwargs)
        if tokenizer is None:
            # Default to T5 tokenizer for text encoding
            try:
                print("[Debug-processor], loading default tokenizer: google/umt5-xxl")
                tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
            except:
                logger.warning("Could not load default tokenizer, using None")
                tokenizer = None

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        images: Optional[ImageInput] = None,
        videos: Optional[ImageInput] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process text and image/video inputs.

        Args:
            text: Text input(s) to tokenize.
            images: Image input(s) to process.
            videos: Video frames to process.
            return_tensors: Type of tensors to return.

        Returns:
            Dictionary with processed inputs.
        """
        if text is None and images is None and videos is None:
            raise ValueError("You must provide either text, images, or videos.")

        data = {}

        # Process text
        if text is not None and self.tokenizer is not None:
            max_length = kwargs.pop("max_length", None)
            if max_length is None:
                max_length = self.max_text_length or getattr(self.tokenizer, "model_max_length", 256)
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=True,
                truncation=True,
                max_length=max_length,
                **kwargs,
            )
            data.update(text_inputs)

        # Process images or video
        if images is not None or videos is not None:
            image_inputs = self.image_processor(
                images or videos,
                return_tensors=return_tensors,
                **kwargs,
            )
            data.update(image_inputs)

        return data

    def batch_decode(self, *args, **kwargs):
        """Delegate to tokenizer's batch_decode method."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer available for decoding.")
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Delegate to tokenizer's decode method."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer available for decoding.")
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """Get model input names from components."""
        tokenizer_input_names = self.tokenizer.model_input_names if self.tokenizer else []
        image_processor_input_names = self.image_processor.model_input_names
        return list(set(tokenizer_input_names + image_processor_input_names))
