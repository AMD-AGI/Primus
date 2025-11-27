###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# Unit tests for Megatron patch registration and basic wiring.
#
# These tests are intentionally lightweight: their main goal is to ensure that
# importing the Megatron patch collection does not raise, and that all core
# patch IDs are registered in the PatchRegistry with the expected metadata.
# This helps us catch broken imports (e.g., moved modules, wrong package paths)
# early in CI instead of failing at runtime when BackendRegistry lazily loads
# `primus.backends.megatron`.
###############################################################################

from types import SimpleNamespace

from primus.backends.megatron import patches as megatron_patches  # noqa: F401
from primus.core.patches import PatchRegistry
from tests.utils import PrimusUT


class TestMegatronPatchesRegistration(PrimusUT):
    def test_core_patch_ids_are_registered(self):
        """All core Megatron patch IDs should be registered with correct backend/phase."""

        expected = {
            # Args / build_args phase
            "megatron.args.profile_tensorboard": ("megatron", "build_args"),
            "megatron.args.checkpoint_path": ("megatron", "build_args"),
            "megatron.args.tensorboard_path": ("megatron", "build_args"),
            "megatron.args.wandb_config": ("megatron", "build_args"),
            "megatron.args.logging_level": ("megatron", "build_args"),
            "megatron.args.data_path_split": ("megatron", "build_args"),
            "megatron.args.mock_data": ("megatron", "build_args"),
            # TE / FP8 / PrimusTurbo
            "megatron.te.disable_fp8_weight_transpose_cache": ("megatron", "before_train"),
            "megatron.fp8.get_fp8_context": ("megatron", "before_train"),
            "megatron.te.primus_turbo_backend": ("megatron", "before_train"),
            "megatron.te.tp_comm_overlap": ("megatron", "before_train"),
            # FSDP / checkpoint / MoE / ZeroBubble / training_log / transformer
            "megatron.fsdp.torch_fsdp2": ("megatron", "before_train"),
            "megatron.checkpoint.filesystem_writer_async": ("megatron", "before_train"),
            "megatron.moe.primus_overrides": ("megatron", "before_train"),
            "megatron.zbpp.enable": ("megatron", "before_train"),
            "megatron.training_log.unified_patch": ("megatron", "before_train"),
        }

        for patch_id, (backend, phase) in expected.items():
            with self.subTest(patch_id=patch_id):
                patch = PatchRegistry.get(patch_id)
                self.assertEqual(
                    patch.backend,
                    backend,
                    msg=f"Patch '{patch_id}' should be registered for backend '{backend}'",
                )
                self.assertEqual(
                    patch.phase,
                    phase,
                    msg=f"Patch '{patch_id}' should be registered for phase '{phase}'",
                )

    def test_apply_megatron_patches_does_not_raise_without_megatron_deps(self):
        """
        Applying Megatron before_train patches should be robust when actual
        Megatron/TE libraries are not installed.

        Most patches are either gated on module_config/args flags or handle
        ImportError/AttributeError internally, so `apply_megatron_patches`
        should not raise when invoked with a minimal context.
        """
        from primus.backends.megatron.patches import apply_megatron_patches

        # Minimal module_config with no special features enabled so that
        # optional patches (FSDP, MoE, ZeroBubble, etc.) short-circuit early.
        module_config = SimpleNamespace(params={})

        extra = {
            # No Megatron args -> training_log patch will SKIP.
            "args": None,
            "config": {},
            "module_config": module_config,
        }

        applied = apply_megatron_patches(
            backend_version="0.0.0-test",
            primus_version="test",
            model_name="dummy_model",
            phase="before_train",
            extra=extra,
        )

        # We don't assert an exact count (it may change as patches evolve),
        # but we do require that the call succeeds and returns an int.
        self.assertIsInstance(applied, int)
