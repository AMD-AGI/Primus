# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""reference_weight_decay: no parameter is exempt from weight decay, other overrides are kept."""

from megatron.core.optimizer import OptimizerConfig, get_standard_config_overrides

from primus.backends.megatron.patches.reference_weight_decay_patches import without_wd_skip


def test_standard_overrides_skip_bias_and_1d():
    overrides = get_standard_config_overrides(OptimizerConfig(weight_decay=0.1))
    assert {"wd_mult": 0.0} in [dict(v) for v in overrides.values()]


def test_wd_skip_dropped_others_kept():
    config = OptimizerConfig(weight_decay=0.1, decoupled_lr=1e-3)
    overrides = get_standard_config_overrides(config)
    kept = without_wd_skip(overrides)
    assert all(dict(v).get("wd_mult", 1.0) != 0.0 for v in kept.values())
    assert len(kept) == len(overrides) - 1
    assert any("max_lr" in dict(v) for v in kept.values())
