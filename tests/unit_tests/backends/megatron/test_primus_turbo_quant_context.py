# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the per-module quantization context the Turbo linears open.

A ``te_precision_config_file`` matcher that pins a module to BF16 has to win over
the experiment-wide autocast. The Turbo linears pick their GEMM from the Turbo
fp8/fp4 flags, so the override has to be decided from those flags and not from
TE's own flag alone: inside an MXFP4 layer a Turbo site can see ``turbo_fp4`` on
while ``te_fp8`` is off, and reading TE's flag alone dropped the override there
and ran a BF16-pinned projection in MXFP4.
"""

import pytest

from tests.utils import PrimusUT, skip_if_no_cuda

skip_if_no_cuda()

from megatron.core.extensions.transformer_engine import (  # noqa: E402  isort:skip
    TEQuantizationRecipe,
)
from transformer_engine.pytorch.fp8 import (  # noqa: E402  isort:skip
    FP8GlobalStateManager,
)

from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402  isort:skip
    PrimusTurboLowPrecisionGlobalStateManager,
    _get_fp8_autocast_for_quant_recipe,
)


class TestTurboQuantContext(PrimusUT):
    """Cover the BF16 pin against every quantization state a Turbo site can see."""

    def setUp(self):
        super().setUp()
        self._saved = PrimusTurboLowPrecisionGlobalStateManager.get_fp8_autocast_state()

    def tearDown(self):
        PrimusTurboLowPrecisionGlobalStateManager.set_fp8_autocast_state(self._saved)
        super().tearDown()

    @staticmethod
    def _bf16_pin():
        """The recipe a ``training_recipe: {}`` matcher entry parses into."""
        return TEQuantizationRecipe.parse_from_config({})

    @staticmethod
    def _set_state(*, te_fp8: bool, turbo_fp4: bool, turbo_fp8: bool = False):
        FP8GlobalStateManager.FP8_ENABLED = te_fp8
        PrimusTurboLowPrecisionGlobalStateManager.PRIMUS_TURBO_FP4_ENABLED = turbo_fp4
        PrimusTurboLowPrecisionGlobalStateManager.PRIMUS_TURBO_FP8_ENABLED = turbo_fp8

    def _assert_pins_bf16(self, context):
        with context:
            self.assertFalse(PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled())
            self.assertFalse(PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled())
            self.assertFalse(FP8GlobalStateManager.is_fp8_enabled())

    def test_no_quantization_leaves_the_autocast_alone(self):
        """override_nonquantized_autocast is off by default, so nothing to do."""
        self._set_state(te_fp8=False, turbo_fp4=False)
        with _get_fp8_autocast_for_quant_recipe(self._bf16_pin()):
            self.assertFalse(PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled())

    def test_te_flag_on_pins_bf16(self):
        self._set_state(te_fp8=True, turbo_fp4=True)
        self._assert_pins_bf16(_get_fp8_autocast_for_quant_recipe(self._bf16_pin()))

    def test_turbo_fp4_without_te_flag_pins_bf16(self):
        """The regression: a BF16-pinned site used to fall through to MXFP4 here."""
        self._set_state(te_fp8=False, turbo_fp4=True)
        self._assert_pins_bf16(_get_fp8_autocast_for_quant_recipe(self._bf16_pin()))

    def test_turbo_fp8_without_te_flag_pins_bf16(self):
        self._set_state(te_fp8=False, turbo_fp4=False, turbo_fp8=True)
        self._assert_pins_bf16(_get_fp8_autocast_for_quant_recipe(self._bf16_pin()))

    def test_state_is_restored_after_the_pin(self):
        """The pin must not leak: the layer's own quantization resumes after it."""
        self._set_state(te_fp8=False, turbo_fp4=True)
        with _get_fp8_autocast_for_quant_recipe(self._bf16_pin()):
            pass
        self.assertTrue(PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
