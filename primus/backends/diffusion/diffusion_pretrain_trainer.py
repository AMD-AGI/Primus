###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Any

from primus.backends.diffusion.runtime import build_trainer, cleanup, prepare_runtime
from primus.core.base_module import BaseModule
from primus.core.trainer.base_trainer import BaseTrainer


class DiffusionPretrainTrainer(BaseTrainer, BaseModule):
    """Thin Primus lifecycle wrapper around the diffusion runtime."""

    def __init__(self, backend_args: Any, *args, **kwargs):
        super().__init__(backend_args=backend_args, *args, **kwargs)
        self.diffusion_trainer = None

    def setup(self):
        prepare_runtime(self.backend_args)

    def init(self):
        self.diffusion_trainer = build_trainer(self.backend_args)

    def train(self):
        if self.diffusion_trainer is None:
            raise RuntimeError("DiffusionPretrainTrainer.init() must be called before train().")
        self.diffusion_trainer.train()
        self.diffusion_trainer.save_model()

    def run(self, *args, **kwargs):
        """Compatibility hook for BaseModule; TrainRuntime drives lifecycle phases."""
        self.train()

    def cleanup(self, on_error: bool = False):
        cleanup(on_error=on_error)
