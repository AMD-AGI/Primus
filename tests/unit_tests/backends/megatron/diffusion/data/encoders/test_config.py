# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for encoder configuration validation logic.
"""

import pytest

from primus.backends.megatron.data.diffusion.encoders.config import (
    CLIPLConfig,
    EncoderConfig,
    T5XXLConfig,
    VAEConfig,
)
from tests.utils import PrimusUT


class TestConfigValidation(PrimusUT):
    """Tests for config validation across all types."""

    def test_all_configs_validate_precision(self):
        """Test that all config types validate precision."""
        for ConfigClass in [EncoderConfig, VAEConfig, T5XXLConfig, CLIPLConfig]:
            with pytest.raises(ValueError, match="precision must be one of"):
                ConfigClass(type="test", model_path="/path", precision="invalid_precision")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
