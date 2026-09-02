###############################################################################
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
NemoAutomodelPretrainTrainer: Primus wrapper for NeMo AutoModel diffusion
pre-training.

Thin-wrapper pattern (same as ``MaxTextPretrainTrainer`` /
``TorchTitanPretrainTrainer``): AutoModel owns FSDP2, the dataloader, the
optimizer and checkpointing internally, so this trainer only

    backend_args (SimpleNamespace)
        -> cleaned dict
        -> temp YAML
        -> AutoModel ``parse_args_and_load_config`` (-> ConfigNode)
        -> patches (``before_train``)
        -> ``TrainDiffusionRecipe``

and then delegates ``setup()`` / ``run_train_validation_loop()`` to the recipe.

By routing through AutoModel's own loader we inherit its config semantics
(``_target_``/``_fn`` resolution, the ``wandb.enable`` toggle, ...) and stay
agnostic to AutoModel internals.

Primus-side behaviour (numerics, sharding repairs, profiling, per-model wiring)
is added as patches under ``primus.backends.nemo_automodel.patches`` rather
than by editing this module, so which models are supported is a property of
that package, not of this trainer. See its docstring for how to add one.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Optional

from primus.core.trainer.base_trainer import BaseTrainer
from primus.core.utils.module_utils import error_rank_0, log_rank_0


class NemoAutomodelPretrainTrainer(BaseTrainer):
    """Trainer class for NeMo AutoModel diffusion pre-training."""

    def __init__(self, backend_args: Any = None, **kwargs):
        # The core runtime instantiates every trainer with BaseModule-style context
        # kwargs (module_name, primus_config, module_rank, ...). Accept and forward
        # them so BaseTrainer can filter them cooperatively; the previous signature
        # raised TypeError on that path.
        super().__init__(backend_args=backend_args, **kwargs)
        self._recipe: Optional[Any] = None
        log_rank_0("Initialized NemoAutomodelPretrainTrainer")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def setup(self):
        """Optional pre-init phase (kept for API symmetry)."""
        log_rank_0("NemoAutomodelPretrainTrainer.setup()")

    def init(self):
        """Build the AutoModel recipe from Primus params and set it up."""
        log_rank_0("NemoAutomodelPretrainTrainer.init() - building AutoModel recipe")

        from nemo_automodel.components.config._arg_parser import (
            parse_args_and_load_config,
        )
        from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

        from primus.backends.nemo_automodel.argument_builder import (
            export_params_to_yaml,
            namespace_to_dict,
            strip_primus_keys,
        )

        params_dict = strip_primus_keys(namespace_to_dict(self.backend_args))

        # Delegate config materialization to AutoModel's own loader (argv=[] so
        # Primus's process argv is never re-parsed).
        yaml_path = export_params_to_yaml(params_dict)
        try:
            cfg = parse_args_and_load_config(yaml_path, argv=[])
        finally:
            try:
                os.unlink(yaml_path)
            except OSError as e:
                error_rank_0(f"NemoAutomodelPretrainTrainer: failed to delete temp YAML {yaml_path}: {e}")

        self._apply_patches()

        self._recipe = TrainDiffusionRecipe(cfg)
        self._recipe.setup()
        log_rank_0("AutoModel recipe initialized successfully")

    def _apply_patches(self):
        """Run this backend's registered patches.

        Timing is the whole point: patches must land after the config is
        materialized (so they can read it) but before ``TrainDiffusionRecipe``
        builds the transformer -- i.e. before ``set_attention_backend`` and the
        first forward. A patch applied after this point would silently miss the
        module it meant to replace.

        ``before_train`` is the phase for that, matching ``MegatronBridgeBaseTrainer``
        which also runs its patches while constructing the trainer. Note this is
        distinct from ``setup()`` above, which is a trainer lifecycle method.
        """
        # Importing the package is what registers the patches (auto-discovery).
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches import run_patches

        run_patches(
            backend="nemo_automodel",
            phase="before_train",
            extra={
                # get_param()/get_args() read module_config.params by attribute,
                # so wrap backend_args the way MegatronBridgeBaseTrainer does.
                "module_config": SimpleNamespace(params=self.backend_args),
                "backend_args": self.backend_args,
            },
        )

    def train(self):
        """Execute the AutoModel train/validation loop."""
        if self._recipe is None:
            raise RuntimeError("NemoAutomodelPretrainTrainer.init() must be called before train().")
        log_rank_0("Executing AutoModel diffusion pretrain...")
        self._recipe.run_train_validation_loop()
        log_rank_0("AutoModel diffusion pretrain completed.")
