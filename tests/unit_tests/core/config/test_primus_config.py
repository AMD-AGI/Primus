###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from pathlib import Path

from primus.core.config.primus_config import get_module_config, load_primus_config
from tests.utils import PrimusUT

# Top-level attributes that `_normalize_module_for_runtime` keeps in place; every
# other public attribute is moved into `params` on the normalized copy.
_RESERVED_KEYS = {"name", "framework", "config", "model", "params", "trainer_class"}

_EXAMPLE_CONFIG = "examples/megatron/exp_pretrain.yaml"


class TestLoadPrimusConfig(PrimusUT):
    def test_exposes_legacy_primus_config(self):
        """load_primus_config attaches the underlying legacy PrimusConfig as
        `_legacy` so the core runtime can reuse it without re-parsing."""
        cfg = load_primus_config(Path(_EXAMPLE_CONFIG), None)

        self.assertTrue(hasattr(cfg, "_legacy"))
        legacy = cfg._legacy
        # The legacy object must still provide the PrimusConfig interface that
        # BaseModule relies on.
        self.assertTrue(callable(getattr(legacy, "get_module_config", None)))

    def test_legacy_config_is_pristine_after_normalization(self):
        """Regression guard for the Option A deepcopy nuance: normalization must
        operate on deep copies, leaving `_legacy`'s module configs un-mutated.

        With the previous in-place normalization, `delattr` stripped non-reserved
        training params off the legacy module config (they were moved into
        `params`). That made the exposed legacy config unusable for BaseModule.
        Here we assert the legacy module configs still carry their original
        top-level training params.
        """
        cfg = load_primus_config(Path(_EXAMPLE_CONFIG), None)
        legacy = cfg._legacy

        module_keys = list(getattr(legacy, "module_keys", []))
        self.assertTrue(module_keys, "expected at least one module in example config")

        for name in module_keys:
            legacy_mod = legacy.get_module_config(name)
            top_level = {k for k in vars(legacy_mod) if not k.startswith("_")}
            # The legacy module must retain non-reserved top-level params; if the
            # normalization had mutated it in place, these would be gone.
            self.assertTrue(
                top_level - _RESERVED_KEYS,
                f"legacy module '{name}' was mutated/stripped by normalization",
            )

            # The normalized SimpleNamespace copy must still expose those same
            # params under `.params` for the new runtime.
            normalized = get_module_config(cfg, name)
            self.assertIsNotNone(normalized)
            self.assertTrue(hasattr(normalized, "params"))
