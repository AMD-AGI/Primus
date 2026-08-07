###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxDiffusionPretrainTrainer: Primus wrapper for MaxDiffusion (JAX) training.

This trainer bridges Primus's configuration system with MaxDiffusion's
trainers, following the same pattern ``MaxTextPretrainTrainer`` uses for
MaxText.

MaxDiffusion (github.com/AI-Hypercomputer/maxdiffusion) does not expose a
MaxText-style ``initialize()``/``run()`` pair. Its entrypoints
(``src/maxdiffusion/train_wan.py`` and ``train_flux.py``) look like::

    def main(argv):
        pyconfig.initialize(argv, validate_training=True)   # train_wan
        config = pyconfig.config                            # module-level global
        validate_train_config(config)
        WanTrainer(config).start_training()                 # or FluxTrainer

    if __name__ == "__main__":
        with transformer_engine_context():
            app.run(main)

So this trainer mirrors that shape:
    backend_args -> flat dict -> temp YAML  (== the maxdiffusion pyconfig file)
    with transformer_engine_context():
        pyconfig.initialize([module, yaml_path, k=v ...])
        <WanTrainer|FluxTrainer>(pyconfig.config).start_training()

The MaxDiffusion-touching (JAX) work happens inside ``train()`` and within the
``transformer_engine_context()``, matching upstream's import/entry ordering
(jax -> tensorflow -> transformer_engine) that the image patches rely on.
"""

import os
from typing import Any, Dict, Optional

from primus.core.trainer.base_trainer import BaseTrainer
from primus.core.utils.module_utils import (
    error_rank_0,
    log_rank_0,
    set_logging_rank,
    warning_rank_0,
)

# Primus-internal params that are not part of MaxDiffusion's pyconfig schema and
# must be stripped before the config YAML is handed to ``pyconfig.initialize``.
_PRIMUS_ONLY_PARAMS = (
    "file_sink_level",
    "stderr_sink_level",
    "sink_level",
    "trainable",
    "stage",
    "model",
    "maxdiffusion_entrypoint",
    "override_model",
)

# model family -> (train module for argv[0], trainer import path, pyconfig kwargs)
_FAMILY_SPECS = {
    "wan": (
        "src.maxdiffusion.train_wan",
        "maxdiffusion.trainers.wan_trainer:WanTrainer",
        {"validate_training": True},
    ),
    "flux": (
        "src.maxdiffusion.train_flux",
        "maxdiffusion.trainers.flux_trainer:FluxTrainer",
        {},
    ),
}


class MaxDiffusionPretrainTrainer(BaseTrainer):
    """Trainer class for MaxDiffusion pre-training."""

    def __init__(self, backend_args: Any = None, **kwargs):
        super().__init__(backend_args=backend_args, **kwargs)
        self._family: Optional[str] = None
        self._yaml_path: Optional[str] = None
        self._argv: list = []
        log_rank_0("Initialized MaxDiffusionPretrainTrainer")

    # --------------------------------------------------------------------- #
    # Lifecycle hooks
    # --------------------------------------------------------------------- #
    def setup(self):
        log_rank_0("MaxDiffusionPretrainTrainer.setup()")

    def init(self):
        """Prepare the MaxDiffusion pyconfig file and launch argv.

        JAX/MaxDiffusion are intentionally NOT imported here; all backend work
        happens in ``train()`` inside ``transformer_engine_context()`` to match
        MaxDiffusion's upstream import ordering.
        """
        from primus.backends.maxdiffusion.argument_builder import (
            export_params_to_yaml,
            namespace_to_dict,
        )

        self._family = self._resolve_family()
        module_name, _, _ = _FAMILY_SPECS[self._family]

        params_dict = namespace_to_dict(self.backend_args)
        override_argv = self._prepare_overrides(params_dict.get("override_model"))
        for key in _PRIMUS_ONLY_PARAMS:
            params_dict.pop(key, None)

        self._yaml_path = export_params_to_yaml(params_dict)
        # pyconfig.initialize(argv): argv[0]=program, argv[1]=config file,
        # argv[2:]=key=value overrides.
        self._argv = [module_name, self._yaml_path, *override_argv]
        log_rank_0(
            f"MaxDiffusionPretrainTrainer.init() family={self._family} "
            f"config={self._yaml_path} overrides={override_argv}"
        )

    def _resolve_family(self) -> str:
        """Determine the MaxDiffusion model family (wan|flux).

        Prefers an explicit ``maxdiffusion_entrypoint`` param, else infers from
        ``model_name`` / ``pretrained_model_name_or_path``.
        """
        explicit = getattr(self.backend_args, "maxdiffusion_entrypoint", None)
        if explicit:
            fam = str(explicit).strip().lower()
            if fam in _FAMILY_SPECS:
                return fam
            raise ValueError(
                f"MaxDiffusion: unknown maxdiffusion_entrypoint '{explicit}' "
                f"(expected one of {sorted(_FAMILY_SPECS)})"
            )
        name = str(getattr(self.backend_args, "model_name", "")).lower()
        pretrained = str(getattr(self.backend_args, "pretrained_model_name_or_path", "")).lower()
        blob = f"{name} {pretrained}"
        if "wan" in blob:
            return "wan"
        if "flux" in blob:
            return "flux"
        raise ValueError(
            "MaxDiffusion: could not infer model family (wan|flux) from config; "
            "set `maxdiffusion_entrypoint: wan|flux` in the config."
        )

    def _prepare_overrides(self, override_model: Any) -> list:
        """Flatten ``override_model`` into MaxDiffusion ``key=value`` argv entries.

        In the core runtime flow, CLI args like ``--override_model.attention flash``
        are parsed into ``backend_args.override_model.attention = "flash"``.
        MaxDiffusion's ``pyconfig.initialize`` consumes overrides as extra argv
        ``key=value`` strings (append after the config file path).
        """
        if not override_model:
            return []
        if isinstance(override_model, dict):
            override_dict = override_model
        elif hasattr(override_model, "__dict__"):
            override_dict = vars(override_model)
        else:
            return []
        argv = []
        for k, v in override_dict.items():
            if isinstance(v, dict) or (hasattr(v, "__dict__") and not isinstance(v, type)):
                raise ValueError(f"MaxDiffusion: nested override not supported: override_model.{k}={v}")
            argv.append(f"{k}={v}")
        if argv:
            warning_rank_0(f"MaxDiffusion: applying override_model argv: {argv}")
        return argv

    def _update_logger_rank(self):
        """Refresh Primus logger rank/world_size from JAX distributed state."""
        import jax

        rank = jax.process_index()
        world_size = jax.process_count()

        from primus.core.utils.logger import update_rank_info

        update_rank_info(rank, world_size)
        set_logging_rank(rank, world_size)
        log_rank_0(
            f"JAX distributed ready: rank={rank}, world_size={world_size}, "
            f"devices={jax.device_count()}, local_devices={jax.local_device_count()}"
        )

    # --------------------------------------------------------------------- #
    # Training entrypoint
    # --------------------------------------------------------------------- #
    def train(self):
        """Execute MaxDiffusion training.

        Initializes MaxDiffusion's ``pyconfig`` and runs the family-specific
        trainer inside ``transformer_engine_context()``, mirroring the upstream
        ``train_wan`` / ``train_flux`` ``main()`` entrypoints.
        """
        if not self._argv:
            raise RuntimeError("MaxDiffusionPretrainTrainer.init() must be called before train().")

        module_name, trainer_path, init_kwargs = _FAMILY_SPECS[self._family]

        from maxdiffusion import pyconfig
        from maxdiffusion.train_utils import transformer_engine_context, validate_train_config

        log_rank_0(f"Executing MaxDiffusion {self._family} pretrain...")
        try:
            with transformer_engine_context():
                pyconfig.initialize(self._argv, **init_kwargs)
                config = pyconfig.config
                validate_train_config(config)

                if self._family == "wan":
                    self._apply_flax_shard_flag()

                self._update_logger_rank()

                # Import the trainer class (which pulls in the TransformerEngine
                # attention layers) INSIDE the TE mesh-guard, exactly as upstream
                # train_wan.main()/train() does: it imports WanTrainer within the
                # `with transformer_engine_context()` block. Importing it outside
                # the guard leaves TE's fused/sharded attention unconfigured (the
                # global_shard_guard MeshResource is not active), so attention
                # falls back to a full, unsharded S^2 matmul -> ~221 GiB OOM on
                # long WAN video sequences. See train_wan.py / train_flux.py.
                trainer_cls = self._import_trainer_class(trainer_path)
                trainer = trainer_cls(config)
                trainer.start_training()
        finally:
            self._cleanup_yaml()

        log_rank_0("MaxDiffusion pretrain execution completed.")

    @staticmethod
    def _import_trainer_class(trainer_path: str):
        """Import ``module:ClassName`` and return the class."""
        import importlib

        module_path, _, class_name = trainer_path.partition(":")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @staticmethod
    def _apply_flax_shard_flag() -> None:
        """Mirror train_wan.main(): flax.config.update(flax_always_shard_variable=False)."""
        try:
            import flax

            flax.config.update("flax_always_shard_variable", False)
        except Exception:  # noqa: BLE001 - optional parity flag; never abort a run
            pass

    def _cleanup_yaml(self) -> None:
        if self._yaml_path and os.path.exists(self._yaml_path):
            try:
                os.unlink(self._yaml_path)
            except OSError as e:
                error_rank_0(f"MaxDiffusionPretrainTrainer: failed to delete temp YAML {self._yaml_path}: {e}")
