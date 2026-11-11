###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the preconditions of the copy-engine FSDP2 collectives (``distributed/ce_*.py``).

WHAT THESE ARE DEFENDING:
  The copy-engine all-gather and reduce-scatter must refuse to install without the kernel and the
  environment RCCL reads at start-up, and read their module prefixes as written. The collectives
  themselves need several GPUs and are not run here.
"""

import pytest

pytest.importorskip("torch")


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
