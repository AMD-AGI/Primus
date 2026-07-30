###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the device_id init_process_group patch gating.

The patch forces eager RCCL communicator creation, which costs ~16 GiB/GPU of
peak VRAM and ~10s of startup with no throughput gain on llama3-70B FSDP2, so
it must stay off unless a config explicitly asks for it. These tests pin the
full truth table of its ``condition`` so the default cannot silently flip back.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from primus.core.patches import PatchContext, PatchRegistry

PATCH_NAME = "megatron.distributed.init_process_group_device_id"
REPO_ROOT = Path(__file__).resolve().parents[5]


def _ctx(**params):
    """Build a PatchContext whose module_config.params carries ``params``."""
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))},
    )


@pytest.fixture
def condition():
    """The patch's real ``condition``, so these tests pin production behaviour.

    Sibling suites clear the process-wide ``PatchRegistry``, so the patch may be
    absent by the time this runs. Reloading its module re-runs ``@register_patch``
    (``register`` overrides a duplicate id), which makes the fixture independent
    of test ordering.
    """
    import importlib

    import primus.backends.megatron.patches.distributed_init_patches as mod

    patch = PatchRegistry.get(PATCH_NAME)
    if patch is None:
        importlib.reload(mod)
        patch = PatchRegistry.get(PATCH_NAME)

    assert patch is not None, f"{PATCH_NAME} is not registered"
    assert patch.condition is not None, "patch must stay conditionally gated"
    return patch.condition


@pytest.mark.parametrize(
    "params, expected, reason",
    [
        ({}, False, "absent flag must not enable the patch"),
        (
            {"use_torch_fsdp2": True},
            False,
            "FSDP2 alone must not enable it -- this is the opt-in change",
        ),
        (
            {"enable_init_process_group_device_id": True, "use_torch_fsdp2": True},
            True,
            "explicit opt-in under FSDP2 applies",
        ),
        (
            {"enable_init_process_group_device_id": True, "use_torch_fsdp2": False},
            False,
            "device_id is FSDP2-only; DDP would pay the memory for nothing",
        ),
        (
            {
                "enable_init_process_group_device_id": True,
                "use_torch_fsdp2": True,
                "enable_odc": True,
            },
            False,
            "ODC carve-out: eager RCCL comms serialize its XGMI copy streams",
        ),
    ],
)
def test_condition_truth_table(condition, params, expected, reason):
    assert bool(condition(_ctx(**params))) is expected, reason


def test_default_config_leaves_patch_disabled():
    """The shipped default must be off, so no recipe pays 16 GiB unknowingly."""
    cfg = REPO_ROOT / "primus/configs/modules/megatron/primus_megatron_module.yaml"
    values = yaml.safe_load(cfg.read_text())
    assert values["enable_init_process_group_device_id"] is False
