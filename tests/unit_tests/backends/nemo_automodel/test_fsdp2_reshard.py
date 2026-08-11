###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the FSDP2 ``reshard_after_forward`` repair hook.

WHY THIS TEST EXISTS:
  The original bug was not a wrong value, it was a value that was never passed. Nothing
  asserted the argument reached its destination, so the setting silently did nothing and
  training ran at ZeRO-3 communication volume while the config said otherwise. Nothing
  raised, and the only symptom was throughput.

  That failure mode is invisible to any test that checks behaviour "works". These tests
  assert the *plumbing* instead: the key is re-applied after the whitelist drops it, the
  DDP path is untouched, re-installing does not nest wrappers, and a missing target
  raises rather than passing silently.

  No GPU is needed. The Automodel module is stubbed, so this also does not depend on the
  submodule being importable.
"""

import sys
import types

import pytest

from primus.backends.nemo_automodel.distributed import fsdp2_reshard

MODULE_PATH = "nemo_automodel._diffusers.auto_diffusion_pipeline"


class FakeManager:
    """Stands in for FSDP2Manager/DDPManager, which set this in __init__ from config."""

    def __init__(self):
        self.reshard_after_forward = "NEVER_SET"


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Install a stub ``auto_diffusion_pipeline`` whose factory records its input.

    The real ``_create_parallel_manager`` copies ``manager_args`` and pops
    ``_manager_type`` from the copy; the stub does the same so the wrapper is exercised
    against the same contract it relies on.
    """
    module = types.ModuleType(MODULE_PATH)
    calls = []

    def _create_parallel_manager(manager_args):
        args = manager_args.copy()
        args.pop("_manager_type", "fsdp2")
        calls.append(args)
        return FakeManager()

    module._create_parallel_manager = _create_parallel_manager
    module.calls = calls

    for name in ("nemo_automodel", "nemo_automodel._diffusers", MODULE_PATH):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, MODULE_PATH, module)

    # The hook records install state in module globals; reset so tests do not leak.
    monkeypatch.setattr(fsdp2_reshard, "patch_installed", False, raising=False)
    monkeypatch.setattr(fsdp2_reshard, "applied_reshard_after_forward", None, raising=False)
    return module


def test_install_reports_success_and_sets_provenance(stub_pipeline):
    assert fsdp2_reshard.install() is True
    assert fsdp2_reshard.patch_installed is True


@pytest.mark.parametrize("value", [False, True, None])
def test_key_is_reapplied_onto_the_manager_for_fsdp2(stub_pipeline, value):
    """The whole point: the whitelist drops this, so the wrapper must put it back."""
    fsdp2_reshard.install()
    manager = stub_pipeline._create_parallel_manager(
        {"_manager_type": "fsdp2", "reshard_after_forward": value, "world_size": 8}
    )
    assert manager.reshard_after_forward == value
    assert fsdp2_reshard.applied_reshard_after_forward == value


def test_fsdp2_is_the_default_manager_type(stub_pipeline):
    """A missing _manager_type means fsdp2 upstream, so the wrapper must still apply."""
    fsdp2_reshard.install()
    manager = stub_pipeline._create_parallel_manager({"reshard_after_forward": False})
    assert manager.reshard_after_forward is False


def test_ddp_path_is_left_untouched(stub_pipeline):
    fsdp2_reshard.install()
    manager = stub_pipeline._create_parallel_manager(
        {"_manager_type": "ddp", "reshard_after_forward": False}
    )
    assert manager.reshard_after_forward == "NEVER_SET"


def test_absent_key_is_not_invented(stub_pipeline):
    """It repairs, it does not decide. No key in YAML means None, i.e. today's heuristic."""
    fsdp2_reshard.install()
    manager = stub_pipeline._create_parallel_manager({"_manager_type": "fsdp2"})
    assert manager.reshard_after_forward is None


def test_the_callee_still_receives_its_arguments(stub_pipeline):
    """The wrapper must not consume or mutate what the real factory needs."""
    fsdp2_reshard.install()
    stub_pipeline._create_parallel_manager(
        {"_manager_type": "fsdp2", "reshard_after_forward": False, "world_size": 8}
    )
    assert stub_pipeline.calls[-1]["world_size"] == 8


def test_reinstalling_does_not_nest_wrappers(stub_pipeline):
    """Without the sentinel, a second install captures the wrapper as its own target."""
    fsdp2_reshard.install()
    first = stub_pipeline._create_parallel_manager
    assert fsdp2_reshard.install() is True
    assert stub_pipeline._create_parallel_manager is first


def test_missing_target_raises_rather_than_silently_skipping(stub_pipeline):
    del stub_pipeline._create_parallel_manager
    with pytest.raises(RuntimeError, match="_create_parallel_manager"):
        fsdp2_reshard.install()


def test_non_callable_target_raises(stub_pipeline):
    stub_pipeline._create_parallel_manager = "not callable"
    with pytest.raises(RuntimeError, match="_create_parallel_manager"):
        fsdp2_reshard.install()
