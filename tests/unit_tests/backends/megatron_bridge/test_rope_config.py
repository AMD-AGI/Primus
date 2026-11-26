###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for Megatron-Bridge transformers rope_theta compatibility."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from primus.backends.megatron_bridge.utils.rope_config import (
    install_transformers_rope_theta_shim,
    resolve_rope_theta,
)


class TestResolveRopeTheta:
    def test_default_theta(self):
        cfg = Mock(spec=[])
        cfg.default_theta = 500000.0
        assert resolve_rope_theta(cfg) == 500000.0

    def test_rope_parameters_dict(self):
        cfg = Mock(spec=[])
        cfg.default_theta = None
        cfg.rope_parameters = {"rope_theta": 750000.0}
        assert resolve_rope_theta(cfg) == 750000.0

    def test_rope_parameters_object(self):
        cfg = Mock(spec=[])
        cfg.rope_parameters = SimpleNamespace(rope_theta=750000.0)
        assert resolve_rope_theta(cfg) == 750000.0

    def test_rope_parameters_dict_wins_over_default_theta(self):
        cfg = Mock(spec=[])
        cfg.default_theta = 10000.0
        cfg.rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0}
        assert resolve_rope_theta(cfg) == 1000000.0

    def test_rope_parameters_object_wins_over_default_theta(self):
        cfg = Mock(spec=[])
        cfg.default_theta = 10000.0
        cfg.rope_parameters = SimpleNamespace(rope_type="default", rope_theta=1000000.0)
        assert resolve_rope_theta(cfg) == 1000000.0

    def test_falls_back_to_default_theta_without_rope_parameters_theta(self):
        cfg = Mock(spec=[])
        cfg.default_theta = 10000.0
        cfg.rope_parameters = {"rope_type": "default"}
        assert resolve_rope_theta(cfg) == 10000.0

    def test_missing_raises(self):
        cfg = Mock(spec=[])
        with pytest.raises(AttributeError, match="rope_theta"):
            resolve_rope_theta(cfg)


class TestTransformersRopeThetaShim:
    def test_shim_exposes_rope_theta_for_transformers5_style_config(self):
        pytest.importorskip("transformers")
        from transformers.configuration_utils import PretrainedConfig

        install_transformers_rope_theta_shim()

        cfg = PretrainedConfig()
        object.__setattr__(cfg, "default_theta", 1000000.0)

        assert cfg.rope_theta == 1000000.0

    def test_shim_prefers_checkpoint_rope_parameters_over_class_default(self):
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "Qwen3Config"):
            pytest.skip("transformers has no Qwen3Config")

        install_transformers_rope_theta_shim()

        cfg = transformers.Qwen3Config(rope_parameters={"rope_type": "default", "rope_theta": 1000000.0})
        if "rope_theta" in vars(cfg):
            pytest.skip("transformers 4.x exposes rope_theta directly")
        assert cfg.rope_theta == 1000000.0
