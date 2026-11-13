###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the preconditions of the copy-engine FSDP2 collectives (``distributed/ce_*.py``).

WHAT THESE ARE DEFENDING:
  The copy-engine all-gather and reduce-scatter must refuse to install without the kernel and the
  environment RCCL reads at start-up, and read their module prefixes as written. Both hook FSDP2's
  internals, so their patches must also decline on runs that shard with DDP or Megatron-FSDP, where
  they would do nothing while logging that they are installed. The collectives themselves need
  several GPUs and are not run here.
"""

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from primus.core.patches import PatchContext  # noqa: E402

CE_PATCHES = [
    ("fsdp_ce_all_gather_patches", "primus_perf.fsdp_ce_all_gather"),
    ("fsdp_ce_reduce_scatter_patches", "primus_perf.fsdp_ce_reduce_scatter"),
]


def _ctx(**sections):
    params = SimpleNamespace(**{name: SimpleNamespace(**keys) for name, keys in sections.items()})
    return PatchContext(
        backend="automodel",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=params)},
    )


def _condition(module_name):
    return importlib.import_module(f"primus.backends.automodel.patches.{module_name}")._enabled


class TestCopyEngineConditions:
    @pytest.mark.parametrize("module_name,flag", CE_PATCHES)
    def test_off_by_default(self, module_name, flag):
        assert _condition(module_name)(_ctx(fsdp={})) is False

    @pytest.mark.parametrize("module_name,flag", CE_PATCHES)
    def test_applies_to_fsdp2(self, module_name, flag, set_option):
        set_option(flag, True)
        assert _condition(module_name)(_ctx(fsdp={})) is True
        assert _condition(module_name)(_ctx()) is True

    @pytest.mark.parametrize("module_name,flag", CE_PATCHES)
    @pytest.mark.parametrize(
        "sections",
        [
            {"fsdp": {"backend": "megatron_fsdp"}},
            {"fsdp": {"backend": "mfsdp"}},
            {"ddp": {"backend": "nccl"}},
        ],
    )
    def test_declines_without_fsdp2(self, module_name, flag, sections, set_option):
        set_option(flag, True)
        assert _condition(module_name)(_ctx(**sections)) is False


class TestCopyEngineHelpers:
    def test_missing_env_names_what_is_wrong(self, monkeypatch):
        from primus.backends.automodel.distributed import ce_all_gather

        monkeypatch.setenv("NCCL_CTA_POLICY", "2")
        monkeypatch.delenv("NCCL_CUMEM_ENABLE", raising=False)
        assert ce_all_gather.missing_env() == ["NCCL_CUMEM_ENABLE=None"]
        monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
        assert ce_all_gather.missing_env() == []

    def test_module_prefixes(self):
        from primus.backends.automodel.distributed import ce_all_gather

        assert ce_all_gather.module_prefixes("") == []
        assert ce_all_gather.module_prefixes(" transformer_blocks, single_transformer_blocks.0 ,") == [
            "transformer_blocks",
            "single_transformer_blocks.0",
        ]

    @pytest.mark.parametrize(
        "release,ok", [("6.8.0-138-generic", True), ("6.11.2", True), ("5.15.0-91", False)]
    )
    def test_kernel_supported(self, monkeypatch, release, ok):
        from primus.backends.automodel.distributed import ce_all_gather

        monkeypatch.setattr(ce_all_gather.platform, "release", lambda: release)
        assert ce_all_gather.kernel_supported() is ok
