###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the generic Primus parallelization sidecar."""

import types

import pytest

from primus.backends.nemo_automodel.distributed import sidecar
from tests.unit_tests.backends.nemo_automodel.parallelize._support import (
    FakeConfig,
    FakeMeshContext,
    install_stub_automodel,
)


@pytest.fixture
def automodel(monkeypatch):
    return install_stub_automodel(monkeypatch)


def _block_ac(base, stride=0):
    return sidecar.block_checkpointing(base, block_attrs=("a", "b"), log_prefix="[t]", stride=lambda: stride)


def _model():
    return types.SimpleNamespace(a=[1, 2], b=[3, 4, 5])


class TestInstall:
    def test_default_base_when_upstream_has_none(self, automodel):
        assert sidecar.install("Model", _block_ac, log_prefix="[t]") is True
        installed = automodel.registry["Model"]
        assert isinstance(installed, automodel.ModelParallelizer)

    def test_upstream_sidecar_becomes_the_base(self, automodel):
        automodel.registry["Model"] = automodel.WanModelParallelizer()
        sidecar.install("Model", _block_ac, log_prefix="[t]")
        assert isinstance(automodel.registry["Model"], automodel.WanModelParallelizer)

    def test_install_is_idempotent(self, automodel):
        sidecar.install("Model", _block_ac, log_prefix="[t]")
        first = automodel.registry["Model"]
        sidecar.install("Model", _block_ac, log_prefix="[t]")
        assert automodel.registry["Model"] is first

    def test_missing_registry_declines_without_raising(self, automodel, caplog):
        del automodel.registry_module._PARALLELIZERS
        with caplog.at_level("WARNING"):
            assert sidecar.install("Model", _block_ac, log_prefix="[t]") is False
        assert "not installed" in caplog.text


class TestBlockCheckpointing:
    def test_off_passes_the_context_through(self, automodel):
        ctx = FakeMeshContext()
        _block_ac(automodel.ModelParallelizer)().parallelize(_model(), ctx)
        assert automodel.parallelize == [ctx]
        assert automodel.full == [] and automodel.selective == []

    @pytest.mark.parametrize("raw", ["false", "off", "0"])
    def test_false_like_strings_checkpoint_nothing(self, automodel, raw):
        _block_ac(automodel.ModelParallelizer)().parallelize(
            _model(), FakeMeshContext(activation_checkpointing=raw)
        )
        assert automodel.full == [] and automodel.selective == []

    def test_full_covers_every_list_then_defers_with_ac_off(self, automodel):
        ctx = FakeMeshContext(activation_checkpointing=True)
        _block_ac(automodel.ModelParallelizer)().parallelize(_model(), ctx)
        assert automodel.full == [[1, 2, 3, 4, 5]]
        (passed,) = automodel.parallelize
        assert passed.activation_checkpointing is False
        assert ctx.activation_checkpointing is True, "the caller's context must not be mutated"

    def test_ddp_style_config_flag_is_honoured_and_cleared(self, automodel):
        """DDP reads the flag from the strategy config, not the mesh context."""
        ctx = FakeMeshContext(strategy_config=FakeConfig(activation_checkpointing=True))
        _block_ac(automodel.ModelParallelizer)().parallelize(_model(), ctx)
        assert automodel.full == [[1, 2, 3, 4, 5]]
        (passed,) = automodel.parallelize
        assert passed.strategy_config.activation_checkpointing is False
        assert ctx.strategy_config.activation_checkpointing is True

    def test_selective_uses_the_selective_helper(self, automodel):
        _block_ac(automodel.ModelParallelizer)().parallelize(
            _model(), FakeMeshContext(activation_checkpointing="selective")
        )
        assert automodel.selective == [[1, 2, 3, 4, 5]]
        assert automodel.full == []

    def test_stride_is_read_at_parallelize_time(self, automodel):
        _block_ac(automodel.ModelParallelizer, stride=2)().parallelize(
            _model(), FakeMeshContext(activation_checkpointing=True)
        )
        assert automodel.full == [[1, 3, 5]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
