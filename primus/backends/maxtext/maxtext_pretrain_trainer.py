###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxTextPretrainTrainer: Primus wrapper for MaxText pre-training.

This trainer bridges Primus's configuration system with MaxText's training
loop, following the same pattern as ``TorchTitanPretrainTrainer`` does for
TorchTitan.

Flow:
    backend_args → flat dict → temp YAML → ``pyconfig.initialize(argv)``

By delegating config parsing entirely to MaxText's ``pyconfig.initialize``,
this trainer works with both new (Pydantic-based, >= 0.1.1) and old
(dict-based, 2025.x.x) MaxText versions without version-specific patches.

The trainer inherits from ``BaseTrainer`` which provides:
    - Access to ``backend_args`` (a ``SimpleNamespace``)
    - Distributed context (rank, world_size …)
    - Standard lifecycle (setup → init → train → cleanup)
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

# Primus-internal params that are not part of MaxText's config schema. MaxText
# v26.4's pyconfig raises on unknown fields (v26.3 merely warns), so these must
# be stripped before the config is handed to ``pyconfig.initialize``.
_PRIMUS_ONLY_PARAMS = (
    "file_sink_level",
    "stderr_sink_level",
    "sink_level",
    "trainable",
    "model",
)


def _resolve_maxtext_train():
    """Resolve MaxText's train entrypoints across MaxText versions.

    MaxText v26.4+ ships as the ``maxtext`` package with the training loop at
    ``maxtext.trainers.pre_train.train``. MaxText v26.3 and earlier expose it
    as ``MaxText.train``. Prefer the newer layout and fall back to the legacy
    one so a single Primus checkout works against both images.

    Returns:
        Tuple of ``(initialize, run, module_name)`` where ``module_name`` is the
        importable module string to use as ``argv[0]`` for ``initialize``.
    """
    try:
        from maxtext.trainers.pre_train.train import initialize, run

        return initialize, run, "maxtext.trainers.pre_train.train"
    except ImportError:
        from MaxText.train import initialize, run

        return initialize, run, "MaxText.train"


class MaxTextPretrainTrainer(BaseTrainer):
    """
    Trainer class for MaxText pre-training.
    """

    def __init__(self, backend_args: Any = None, **kwargs):
        # The core runtime instantiates every trainer with BaseModule-style
        # context kwargs (module_name, primus_config, module_rank, ...). MaxText
        # does not need them; accept and forward so BaseTrainer can filter them.
        super().__init__(backend_args=backend_args, **kwargs)

        # Training state (populated in init())
        self.train_config: Optional[Any] = None
        self.recorder: Optional[Any] = None
        self.diagnostic_config: Optional[Any] = None
        # Raw tuple returned by MaxText's initialize(); forwarded verbatim to
        # run() so we stay agnostic to its return arity across MaxText versions.
        self._init_result: tuple = ()

        log_rank_0("Initialized MaxTextPretrainTrainer")

    # --------------------------------------------------------------------- #
    # Lifecycle hooks
    # --------------------------------------------------------------------- #

    def setup(self):
        """
        Optional setup phase (kept for API symmetry with other trainers).
        """
        log_rank_0("MaxTextPretrainTrainer.setup()")

    def init(self):
        """
        Construct the underlying MaxText training components.

        Converts ``backend_args`` to a flat dict, writes it to a temporary YAML
        file, and calls ``initialize(argv)`` which delegates config parsing
        entirely to MaxText's ``pyconfig.initialize``.  This works for both
        Pydantic-based (Dec) and dict-based (Aug) MaxText versions.
        """
        log_rank_0("MaxTextPretrainTrainer.init() - initializing MaxText training")

        initialize, _, module_name = _resolve_maxtext_train()

        from primus.backends.maxtext.argument_builder import (
            export_params_to_yaml,
            namespace_to_dict,
        )

        override_model_args = self._prepare_model_overrides()
        params_dict = namespace_to_dict(self.backend_args)
        params_dict.pop("override_model", None)

        # Strip Primus-internal params that MaxText's config schema rejects.
        for key in _PRIMUS_ONLY_PARAMS:
            params_dict.pop(key, None)

        yaml_path = export_params_to_yaml(params_dict)
        try:
            argv = [module_name, yaml_path]
            init_result = initialize(argv, **override_model_args)
        finally:
            try:
                os.unlink(yaml_path)
            except OSError as e:
                error_rank_0(f"MaxTextPretrainTrainer: Failed to delete temporary YAML at {yaml_path}: {e}")

        # MaxText's ``initialize()`` return arity changed across versions:
        #   - v26.4 and earlier: ``(config, recorder, diagnostic_config)``
        #   - v26.5+:            ``(config, recorder)`` (diagnostic_config dropped)
        # Keep the raw tuple so ``train()`` can forward it verbatim to ``run()``,
        # whose parameter count matches ``initialize()``'s arity in the same
        # version. Unpacking a fixed number of names here is what previously
        # crashed on v26.5 with "not enough values to unpack (expected 3, got 2)".
        if not isinstance(init_result, tuple):
            init_result = (init_result,)
        self._init_result = init_result
        self.train_config = init_result[0] if len(init_result) > 0 else None
        self.recorder = init_result[1] if len(init_result) > 1 else None
        self.diagnostic_config = init_result[2] if len(init_result) > 2 else None

        self._update_logger_rank()
        log_rank_0("MaxText training components initialized successfully")

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

    def _prepare_model_overrides(self) -> Dict[str, Any]:
        """
        Prepare model overrides from ``backend_args.override_model``.

        In the core runtime flow, CLI args like
        ``--override_model.base_num_decoder_layers 4`` are parsed into
        ``backend_args.override_model.base_num_decoder_layers = 4``
        (a nested SimpleNamespace attribute).

        This method extracts and flattens those overrides into a plain dict
        suitable for passing as ``**kwargs`` to ``pyconfig.initialize``.

        Returns:
            Dictionary of flat overrides for MaxText, e.g.
            ``{"base_num_decoder_layers": 4}``
        """
        override_model = getattr(self.backend_args, "override_model", None)

        if not override_model:
            warning_rank_0("MaxText Pre-Trainer: No override_model provided, skip patch.")
            return {}

        if isinstance(override_model, dict):
            override_dict = override_model
        else:
            override_dict = vars(override_model) if hasattr(override_model, "__dict__") else {}

        if not override_dict:
            warning_rank_0("MaxText Pre-Trainer: override_model is empty, skip patch.")
            return {}

        warning_rank_0(f"MaxText Pre-Trainer: Applying override_model: {override_dict}")

        flat_overrides: Dict[str, Any] = {}
        for k, v in override_dict.items():
            if isinstance(v, dict) or hasattr(v, "__dict__") and not isinstance(v, type):
                raise ValueError(
                    f"MaxText Pre-Trainer: Nested override not supported: override_model.{k}={v}"
                )
            flat_overrides[k] = v

        return flat_overrides

    # --------------------------------------------------------------------- #
    # Training entrypoint
    # --------------------------------------------------------------------- #

    def train(self):
        """
        Execute MaxText pre-training.

        This method is called by the runtime lifecycle after setup() and init().
        """
        if self.train_config is None:
            raise RuntimeError("MaxTextPretrainTrainer.init() must be called before train().")

        log_rank_0("Executing MaxText pretrain...")

        _, run, _ = _resolve_maxtext_train()

        # Forward the exact tuple returned by initialize(); run()'s signature
        # matches initialize()'s arity within a given MaxText version (3 args on
        # v26.4 with diagnostic_config, 2 args on v26.5+).
        run(*self._init_result)

        log_rank_0("MaxText pretrain execution completed.")
