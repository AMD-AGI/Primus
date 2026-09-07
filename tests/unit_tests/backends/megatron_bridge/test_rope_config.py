###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for Megatron-Bridge transformers rope_theta compatibility."""

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
