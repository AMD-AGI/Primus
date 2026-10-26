###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-step time and throughput for Primus World Mirror logs."""

from __future__ import annotations

import time

from lightning.pytorch.callbacks import Callback

from primus.core.utils.module_utils import log_rank_0


class StepTimeCallback(Callback):
    """Log step wall time and image throughput on rank 0.

    Step time is the GPU-synced gap from the previous step's end to this
    step's end, so it includes the data wait. ``images_per_s`` counts images
    on every GPU.
    """

    def __init__(self):
        super().__init__()
        self._batch_start = None
        self._prev_end = None
        self._images = 0

    def _rank0(self, trainer) -> bool:
        return getattr(trainer, "global_rank", 0) == 0

    def _sync(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_start(self, trainer, pl_module) -> None:
        if not self._rank0(trainer):
            return
        self._sync()
        self._prev_end = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if not self._rank0(trainer):
            return
        self._images = _images_per_gpu(batch)
        self._sync()
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self._rank0(trainer) or self._batch_start is None:
            return
        self._sync()
        now = time.perf_counter()
        wall = now - self._prev_end if self._prev_end is not None else now - self._batch_start
        self._prev_end = now
        world = int(getattr(trainer, "world_size", 1) or 1)
        images_per_s = (self._images * world) / wall if wall > 0 else 0.0
        step = int(trainer.global_step) - 1
        log_rank_0(
            "[Primus:WorldMirror] step=%d step_time_s=%.3f images_per_s=%.2f images_per_gpu=%d"
            % (step, wall, images_per_s, self._images)
        )
        pl_module.log(
            "step_time_s",
            wall,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True,
            rank_zero_only=True,
        )
        pl_module.log(
            "images_per_s",
            images_per_s,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True,
            rank_zero_only=True,
        )


def _images_per_gpu(batch) -> int:
    import torch

    if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict):
        img = batch[0].get("img")
        if torch.is_tensor(img) and img.ndim >= 4:
            return int(img.shape[0]) * len(batch)
    if isinstance(batch, dict) and torch.is_tensor(batch.get("img")):
        img = batch["img"]
        if img.ndim == 5:
            return int(img.shape[0] * img.shape[1])
        if img.ndim == 4:
            return int(img.shape[0])
    return 0
