###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBridgeBaseTrainer: Base class for all Megatron-Bridge trainers.

This mirrors the role of TorchTitanBaseTrainer for TorchTitan:

    - Inherits from the unified BaseTrainer so it participates in the
      common training workflow and patch management (via run_patches)
    - Provides a central place for Megatron-Bridge-specific initialization logic
      and version detection
    - Handles common setup logic shared across all Megatron-Bridge training tasks
"""

from types import SimpleNamespace
from typing import Any

from primus.core.patches import run_patches
from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronBridgeBaseTrainer(BaseTrainer):
    """
    Base trainer class for all Megatron-Bridge training tasks.

    This class provides common functionality for all Megatron-Bridge trainers,
    including version detection, initialization logging, and shared setup logic.

    Responsibilities:
        - Call into the shared BaseTrainer to enable the unified workflow
          (before/after_train patches, lifecycle, logging)
        - Log Megatron-Bridge metadata (version, model, framework, task)
        - Provide a classmethod detect_version used by the patch system
        - Handle Megatron-Bridge specific initialization and setup
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize Megatron-Bridge base trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: Megatron-Bridge configuration as SimpleNamespace
                         (from MegatronBridgeArgBuilder)
        """
        log_rank_0("=" * 80)
        log_rank_0("Initializing MegatronBridgeBaseTrainer...")
        log_rank_0("=" * 80)

        # Initialize BaseTrainer (stores configs, enables patch management)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        # If PrimusTurbo features are enabled, populate Megatron-LM's global
        # args so that PrimusTurbo* classes (which call
        # megatron.training.global_vars.get_args()) can find the config they
        # need.  Megatron-Bridge normally never sets these globals because it
        # uses its own ConfigContainer, but the PrimusTurbo extensions still
        # rely on them.
        self._maybe_init_megatron_global_args()

        import primus.backends.megatron.patches  # noqa: F401

        run_patches(
            backend="megatron",
            phase="before_train",
            backend_version=type(self).detect_megatron_version(),
            model_name=self.model_name,
            extra={
                "backend_args": self.backend_args,
                "primus_config": self.primus_config,
                "module_config": self.module_config,
            },
        )

        log_rank_0("=" * 80)
        log_rank_0("MegatronBridgeBaseTrainer initialized successfully")
        log_rank_0("=" * 80)

    # ------------------------------------------------------------------
    # PrimusTurbo <-> Megatron-LM global-args compatibility
    # ------------------------------------------------------------------

    def _is_primus_turbo_enabled(self) -> bool:
        """Check whether any PrimusTurbo feature is turned on in the config."""
        args = self.backend_args
        if not getattr(args, "enable_primus_turbo", False):
            return False
        return any(
            getattr(args, flag, False)
            for flag in (
                "use_turbo_attention",
                "use_turbo_parallel_linear",
                "use_turbo_grouped_mlp",
                "use_turbo_rms_norm",
                "use_turbo_deepep",
            )
        )

    def _maybe_init_megatron_global_args(self) -> None:
        """Populate ``megatron.training.global_vars._GLOBAL_ARGS`` when PrimusTurbo is active.

        Megatron-Bridge uses its own ``ConfigContainer`` and never calls
        ``megatron.training.global_vars.set_global_variables()``.  However, the
        PrimusTurbo extension classes (``PrimusTurboAttention``, ``PrimusTurboLinear``,
        etc.) call ``megatron.training.global_vars.get_args()`` during ``__init__``.

        This method builds a lightweight compatibility namespace from the current
        Primus / Megatron-Bridge config and registers it with Megatron-LM's
        global state so that PrimusTurbo code can find the values it expects.

        Only ``set_args()`` is called — *not* the full ``set_global_variables()``
        — to avoid re-initialising the microbatch calculator, tokenizer, or
        writers that Megatron-Bridge manages separately.
        """
        if not self._is_primus_turbo_enabled():
            return

        from megatron.training.global_vars import _GLOBAL_ARGS, set_args

        if _GLOBAL_ARGS is not None:
            log_rank_0(
                "[MegatronBridgeBaseTrainer] Megatron-LM global args already initialised; "
                "skipping compatibility shim."
            )
            return

        cfg = self.backend_args

        # Attributes accessed by PrimusTurbo* classes via get_args().
        # Values come from the Primus/Bridge config when present; otherwise
        # safe defaults are used (features disabled / empty).
        compat_args = SimpleNamespace(
            # -- PrimusTurbo attention --
            offload=getattr(cfg, "offload", False),
            offload_ops=getattr(cfg, "offload_ops", []),
            enable_turbo_attention_float8=getattr(cfg, "enable_turbo_attention_float8", False),
            # -- Pipeline / zero-bubble (not applicable in Bridge post-train) --
            patch_primus_pipeline=getattr(cfg, "patch_primus_pipeline", False),
            pp_algorithm=getattr(cfg, "pp_algorithm", None),
            patch_zero_bubble=getattr(cfg, "patch_zero_bubble", False),
            enable_zero_bubble=getattr(cfg, "enable_zero_bubble", False),
            # -- MoE --
            patch_moe_overlap=getattr(cfg, "patch_moe_overlap", False),
            overlap_moe_expert_parallel_comm=getattr(cfg, "overlap_moe_expert_parallel_comm", False),
            use_turbo_fused_act_with_probs=getattr(cfg, "use_turbo_fused_act_with_probs", False),
            use_turbo_grouped_mlp=getattr(cfg, "use_turbo_grouped_mlp", False),
            moe_use_legacy_grouped_gemm=getattr(cfg, "moe_use_legacy_grouped_gemm", False),
            moe_router_force_load_balancing=getattr(cfg, "moe_router_force_load_balancing", False),
            moe_grouped_gemm=getattr(cfg, "moe_grouped_gemm", False),
            # -- DeepEP --
            use_turbo_deepep=getattr(cfg, "use_turbo_deepep", False),
            turbo_deepep_use_comm_stream=getattr(cfg, "turbo_deepep_use_comm_stream", False),
            turbo_deepep_num_cu=getattr(cfg, "turbo_deepep_num_cu", 32),
            turbo_sync_free_moe_stage=getattr(cfg, "turbo_sync_free_moe_stage", 0),
            # -- Training geometry (used by DeepEP dispatcher) --
            sequence_parallel=getattr(cfg, "sequence_parallel", False),
            seq_length=getattr(cfg, "seq_length", 4096),
            context_parallel_size=getattr(cfg, "context_parallel_size", 1),
            micro_batch_size=getattr(cfg, "micro_batch_size", 1),
            # -- Turbo feature flags --
            enable_primus_turbo=getattr(cfg, "enable_primus_turbo", False),
            use_turbo_attention=getattr(cfg, "use_turbo_attention", False),
            use_turbo_parallel_linear=getattr(cfg, "use_turbo_parallel_linear", False),
            use_turbo_rms_norm=getattr(cfg, "use_turbo_rms_norm", False),
        )

        set_args(compat_args)
        log_rank_0(
            "[MegatronBridgeBaseTrainer] Initialised Megatron-LM global args "
            "compatibility shim for PrimusTurbo."
        )

    @classmethod
    def detect_version(cls) -> str:
        """
        Detect Megatron-Bridge version.

        Returns:
            Version string (e.g., "0.3.0rc0") from package_info

        Raises:
            RuntimeError: If version detection fails
        """
        try:
            from megatron.bridge.package_info import __version__

            return __version__
        except ImportError as e:
            raise RuntimeError(
                "Failed to detect Megatron-Bridge version. " "Make sure Megatron-Bridge is installed."
            ) from e

    @classmethod
    def detect_megatron_version(cls) -> str:
        """
        Detect Megatron-LM version using the official method.

        Returns:
            Megatron version string (e.g., "0.15.0rc8")

        Raises:
            RuntimeError: If version cannot be detected (critical requirement)
        """
        try:
            from megatron.core import package_info

            return package_info.__version__
        except Exception as e:
            raise RuntimeError(
                "Failed to detect Megatron-LM version. "
                "Please ensure Megatron-LM is properly installed and "
                "megatron.core.package_info is available."
            ) from e
