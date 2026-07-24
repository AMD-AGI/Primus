###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for NeMo AutoModel backend registration in __init__.py.

This test ensures that the registration logic in
primus/backends/nemo_automodel/__init__.py correctly registers the backend with
BackendRegistry. Without proper registration, the backend is unavailable to the
runtime. Mirrors tests/unit_tests/backends/megatron/test_megatron_registration.py
(NeMo AutoModel exposes a single pretrain trainer; no SFT stage).

Importing primus.backends.nemo_automodel does not require the `nemo_automodel`
package to be installed: the adapter/trainer modules import it lazily (only inside
the trainer's init()), so these tests run without the backend present.
"""

import pytest

from primus.backends.nemo_automodel.nemo_automodel_adapter import NemoAutomodelAdapter
from primus.backends.nemo_automodel.nemo_automodel_pretrain_trainer import (
    NemoAutomodelPretrainTrainer,
)
from primus.core.backend.backend_registry import BackendRegistry

_SUPPORTS_TRAINER_CLASS_REGISTRY = all(
    hasattr(BackendRegistry, attr)
    for attr in ("_trainer_classes", "register_trainer_class", "get_trainer_class", "has_trainer_class")
)

if not _SUPPORTS_TRAINER_CLASS_REGISTRY:
    pytest.skip(
        "BackendRegistry trainer-class registration API is not available; "
        "skip nemo_automodel registration tests that depend on it.",
        allow_module_level=True,
    )


class TestNemoAutomodelBackendRegistration:
    """Test that the nemo_automodel backend is properly registered via __init__.py."""

    @pytest.fixture(autouse=True)
    def ensure_backend_loaded(self):
        """Ensure the nemo_automodel backend module is loaded before each test."""
        # Import the __init__ module to trigger registration.
        import primus.backends.nemo_automodel  # noqa: F401

    def test_adapter_is_registered(self):
        """Verify that NemoAutomodelAdapter is registered for 'nemo_automodel'."""
        assert BackendRegistry.has_adapter("nemo_automodel"), (
            "NemoAutomodelAdapter not registered. "
            "Check BackendRegistry.register_adapter() call in __init__.py"
        )

        adapter_cls = BackendRegistry._adapters.get("nemo_automodel")
        assert adapter_cls is NemoAutomodelAdapter, (
            f"Expected NemoAutomodelAdapter class, got {adapter_cls}. "
            "Check BackendRegistry.register_adapter('nemo_automodel', NemoAutomodelAdapter) in __init__.py"
        )

    def test_trainer_class_is_registered(self):
        """Verify that NemoAutomodelPretrainTrainer is registered for 'nemo_automodel'."""
        assert BackendRegistry.has_trainer_class("nemo_automodel"), (
            "NemoAutomodelPretrainTrainer not registered. "
            "Check BackendRegistry.register_trainer_class() call in __init__.py"
        )

        trainer_cls = BackendRegistry.get_trainer_class("nemo_automodel")
        assert trainer_cls is NemoAutomodelPretrainTrainer, (
            f"Expected NemoAutomodelPretrainTrainer, got {trainer_cls}. "
            "Check BackendRegistry.register_trainer_class(NemoAutomodelPretrainTrainer, 'nemo_automodel')"
        )

        # Explicit pretrain stage should also resolve.
        assert BackendRegistry.has_trainer_class("nemo_automodel", stage="pretrain")

    def test_adapter_can_be_instantiated_via_registry(self):
        """Verify that get_adapter returns a working NemoAutomodelAdapter instance."""
        from unittest.mock import patch

        # Force the lazy-load path and re-register without relying on module reload
        # side effects (see the megatron registration test for rationale).
        original_adapters = BackendRegistry._adapters.copy()
        original_trainers = BackendRegistry._trainer_classes.copy()
        try:
            BackendRegistry._adapters.pop("nemo_automodel", None)
            BackendRegistry._trainer_classes.pop(("nemo_automodel", "pretrain"), None)

            def _fake_load_backend(_backend: str) -> None:
                BackendRegistry.register_adapter("nemo_automodel", NemoAutomodelAdapter)
                BackendRegistry.register_trainer_class(NemoAutomodelPretrainTrainer, "nemo_automodel")

            with patch.object(BackendRegistry, "_load_backend", side_effect=_fake_load_backend):
                adapter = BackendRegistry.get_adapter("nemo_automodel")
        finally:
            BackendRegistry._adapters = original_adapters
            BackendRegistry._trainer_classes = original_trainers

        assert isinstance(
            adapter, NemoAutomodelAdapter
        ), f"Expected NemoAutomodelAdapter instance, got {type(adapter).__name__}"
        assert (
            adapter.framework == "nemo_automodel"
        ), f"Expected framework='nemo_automodel', got '{adapter.framework}'"

    def test_trainer_class_can_be_retrieved_via_adapter(self):
        """Verify trainer class retrieval through the adapter, including the 'train' alias."""
        from unittest.mock import patch

        original_adapters = BackendRegistry._adapters.copy()
        original_trainers = BackendRegistry._trainer_classes.copy()
        try:
            BackendRegistry._adapters.pop("nemo_automodel", None)
            BackendRegistry._trainer_classes.pop(("nemo_automodel", "pretrain"), None)

            def _fake_load_backend(_backend: str) -> None:
                BackendRegistry.register_adapter("nemo_automodel", NemoAutomodelAdapter)
                BackendRegistry.register_trainer_class(NemoAutomodelPretrainTrainer, "nemo_automodel")

            with patch.object(BackendRegistry, "_load_backend", side_effect=_fake_load_backend):
                adapter = BackendRegistry.get_adapter("nemo_automodel")
        finally:
            BackendRegistry._adapters = original_adapters
            BackendRegistry._trainer_classes = original_trainers

        # The adapter maps the common "train" alias to "pretrain".
        with patch("primus.backends.nemo_automodel.nemo_automodel_adapter.log_rank_0"):
            assert adapter.load_trainer_class() is NemoAutomodelPretrainTrainer
            assert adapter.load_trainer_class(stage="train") is NemoAutomodelPretrainTrainer

    def test_nemo_automodel_in_available_backends_list(self):
        """Verify nemo_automodel appears in the list of available backends."""
        available = BackendRegistry.list_available_backends()
        assert "nemo_automodel" in available, (
            f"'nemo_automodel' not in available backends: {available}. "
            "Registration may have failed in __init__.py"
        )


class TestNemoAutomodelRegistrationOrder:
    """Test that registration happens at import time and is idempotent."""

    def test_registration_is_idempotent(self):
        """Verify that re-importing __init__ doesn't cause errors."""
        import importlib

        import primus.backends.nemo_automodel

        importlib.reload(primus.backends.nemo_automodel)

        assert BackendRegistry.has_adapter("nemo_automodel")
        assert BackendRegistry.has_trainer_class("nemo_automodel")


class TestNemoAutomodelRegistrationFailures:
    """Test error handling when registration is missing."""

    def test_missing_trainer_registration_would_fail(self):
        """Without trainer registration, get_trainer_class should fail."""
        original = BackendRegistry._trainer_classes.pop(("nemo_automodel", "pretrain"), None)
        try:
            with pytest.raises(ValueError, match="No trainer class registered"):
                BackendRegistry.get_trainer_class("nemo_automodel")
        finally:
            if original:
                BackendRegistry._trainer_classes[("nemo_automodel", "pretrain")] = original


class TestStripPrimusKeysGuard:
    """Guard the strip_primus_keys denylist against module_base.yaml drift.

    strip_primus_keys() removes Primus-only top-level keys before handing the
    config to AutoModel. Because it is a denylist, any key later added to
    module_base.yaml would silently leak into the AutoModel ConfigNode. This test
    fails loudly if module_base.yaml grows a top-level key not in the strip set.
    """

    def test_module_base_keys_are_all_stripped(self):
        from pathlib import Path

        import yaml

        import primus
        from primus.backends.nemo_automodel.argument_builder import PRIMUS_ONLY_TOP_KEYS

        module_base = Path(primus.__file__).resolve().parent / "configs" / "modules" / "module_base.yaml"
        assert module_base.is_file(), f"module_base.yaml not found at {module_base}"
        with open(module_base, encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}

        leaked = set(base_cfg) - set(PRIMUS_ONLY_TOP_KEYS)
        assert not leaked, (
            f"module_base.yaml top-level keys {sorted(leaked)} are not in PRIMUS_ONLY_TOP_KEYS, "
            "so they would leak into the AutoModel ConfigNode. Add them to PRIMUS_ONLY_TOP_KEYS "
            "in primus/backends/nemo_automodel/argument_builder.py (or confirm they belong in "
            "the AutoModel recipe)."
        )

    def test_strip_primus_keys_preserves_nested_model_keys(self):
        from primus.backends.nemo_automodel.argument_builder import strip_primus_keys

        cleaned = strip_primus_keys(
            {
                "stage": "pretrain",
                "framework": "nemo_automodel",
                "trainable": False,
                "model": {"stage": "low_noise", "pipeline_spec": {"subfolder": "transformer_2"}},
                "optim": {"learning_rate": 1e-4},
            }
        )
        assert "stage" not in cleaned and "framework" not in cleaned and "trainable" not in cleaned
        # Nested AutoModel keys (incl. model.stage) must survive.
        assert cleaned["model"]["stage"] == "low_noise"
        assert cleaned["optim"]["learning_rate"] == 1e-4


class TestUnsupportedStage:
    """load_trainer_class should name the real cause for an unsupported stage."""

    def test_unsupported_stage_raises_clear_error(self):
        from unittest.mock import patch

        adapter = NemoAutomodelAdapter()
        with patch("primus.backends.nemo_automodel.nemo_automodel_adapter.log_rank_0"):
            with pytest.raises(RuntimeError, match=r"only exposes a 'pretrain' stage"):
                adapter.load_trainer_class(stage="sft")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
