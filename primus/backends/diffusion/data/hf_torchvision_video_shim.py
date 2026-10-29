"""Compatibility shim for Hugging Face ``datasets`` + torchvision >= 0.26.

OSS torchvision no longer exports ``torchvision.io.VideoReader`` (see pytorch/vision#9373).
Older ``datasets`` versions do ``from torchvision.io import VideoReader`` inside
``Video.encode_example``, which breaks ``load_dataset`` on video columns.

This module injects ``torchvision.io.VideoReader`` using torchcodec when available,
otherwise decord (already used elsewhere in this repo), so the import succeeds and
decoding yields ``{"data": CHW tensor}`` as expected by ``_preprocess_videos``.

Call :func:`apply_torchvision_io_video_reader_shim` before ``import datasets``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Iterator, Optional, Type

logger = logging.getLogger(__name__)

_applied = False


def apply_torchvision_io_video_reader_shim() -> None:
    """Attach ``VideoReader`` to ``torchvision.io`` if the upstream export is missing."""
    global _applied
    if _applied:
        return

    try:
        import torchvision.io as tv_io
    except ImportError:
        _applied = True
        return

    if getattr(tv_io, "VideoReader", None) is not None:
        _applied = True
        return

    impl: Optional[Type[Any]] = None

    # Prefer torchcodec (matches newer HF datasets).
    try:
        import torch
        from torchcodec.decoders import VideoDecoder

        def _stream_index_for_torchcodec(stream: str) -> Optional[int]:
            if not isinstance(stream, str) or ":" not in stream:
                return None
            try:
                return int(stream.rsplit(":", maxsplit=1)[-1])
            except ValueError:
                return None

        class _TorchCodecVideoReader:
            _hf_shim_video_reader = True

            def __init__(self, src: Any, stream: str = "video") -> None:
                stream_index = _stream_index_for_torchcodec(stream)
                self._tc = VideoDecoder(
                    src,
                    stream_index=stream_index,
                    dimension_order="NCHW",
                    num_ffmpeg_threads=1,
                    device="cpu",
                    seek_mode="exact",
                )

            def __len__(self) -> int:
                tc = getattr(self, "_tc", None)
                if tc is not None:
                    return len(tc)
                raise TypeError("VideoReader length is unavailable for this reader instance")

            def __iter__(self) -> Iterator[dict[str, Any]]:
                tc = getattr(self, "_tc", None)
                if tc is not None:
                    for i in range(len(tc)):
                        frame = tc[i]
                        if frame.dim() == 4:
                            frame = frame[0]
                        yield {"data": frame}
                    return

                if hasattr(self, "_c"):
                    for packet in self._c:
                        if hasattr(packet, "to_ndarray"):
                            arr = packet.to_ndarray(format="rgb24")
                            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                            yield {"data": t}
                        else:
                            yield {"data": packet}
                    return

                raise RuntimeError("VideoReader is not initialized for iteration")

            def get_batch(self, indices: list[int]) -> Any:
                import numpy as np

                tc = getattr(self, "_tc", None)
                if tc is None:
                    raise RuntimeError("VideoReader.get_batch requires torchcodec decoder")
                frames: list[Any] = []
                for i in indices:
                    t = tc[i]
                    if t.dim() == 4:
                        t = t[0]
                    arr = t.detach().cpu().permute(1, 2, 0).numpy()
                    frames.append(np.ascontiguousarray(arr))
                return np.stack(frames, axis=0)

        impl = _TorchCodecVideoReader
    except ImportError:
        pass

    # Fallback: decord (common on ROCm / when torchcodec wheel is missing).
    if impl is None:
        try:
            import numpy as np
            import torch
            from decord import VideoReader as DecordVideoReader
            from decord import cpu

            class _DecordVideoReader:
                _hf_shim_video_reader = True

                def __init__(self, src: Any, stream: str = "video") -> None:
                    self._dr = DecordVideoReader(str(src), ctx=cpu(0))

                def __len__(self) -> int:
                    return len(self._dr)

                def __iter__(self) -> Iterator[dict[str, Any]]:
                    for i in range(len(self._dr)):
                        frame = self._dr[i]
                        if hasattr(frame, "asnumpy"):
                            arr = frame.asnumpy()
                        elif hasattr(frame, "numpy"):
                            arr = frame.numpy()
                        else:
                            arr = frame
                        if not isinstance(arr, np.ndarray):
                            arr = np.asarray(arr)
                        if arr.ndim == 3 and arr.shape[2] == 3:
                            t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
                        else:
                            t = torch.from_numpy(np.ascontiguousarray(arr))
                        yield {"data": t}

                def get_batch(self, indices):  # noqa: ANN001 — decord-compatible
                    return self._dr.get_batch(indices)

            impl = _DecordVideoReader
        except ImportError:
            pass

    if impl is None:
        logger.warning(
            "torchvision.io has no VideoReader and neither torchcodec nor decord could be "
            "imported; HF video datasets will fail. Install torchcodec or decord, or upgrade "
            "datasets to >= 4.8.4."
        )

        class _MissingDepsVideoReader:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "VideoReader was removed from torchvision.io. Install torchcodec or decord, "
                    "or pip install 'datasets>=4.8.4'."
                )

        impl = _MissingDepsVideoReader

    tv_io.VideoReader = impl  # type: ignore[attr-defined]
    sys.modules.setdefault("torchvision.io", tv_io)
    tv_io.__dict__["VideoReader"] = impl

    _applied = True
