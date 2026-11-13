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
  they would do nothing while logging that they are installed. Where a symmetric-memory rendezvous
  fails (torch 2.14 on ROCm), they must leave every unit on RCCL's kernels rather than fail at the
  first all-gather. The collectives themselves need several GPUs and are not run here.
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


class _Group:
    group_name = "0"


class _Unit:
    all_gather = reduce_scatter = None

    def set_custom_all_gather(self, comm):
        self.all_gather = comm

    def set_custom_reduce_scatter(self, comm):
        self.reduce_scatter = comm


def _fsdp_tree(n_units, group):
    """A root FSDP2 state over ``n_units`` units that all-gather and reduce-scatter on ``group``."""
    root_module, units = _Unit(), [_Unit() for _ in range(n_units)]
    root_module.named_modules = lambda: [("", root_module)] + [
        (f"blocks.{i}", u) for i, u in enumerate(units)
    ]
    param_group = SimpleNamespace(_all_gather_process_group=group, _reduce_scatter_process_group=group)
    root = SimpleNamespace(_modules=[root_module])
    states = [SimpleNamespace(_modules=[u], _fsdp_param_groups=[param_group]) for u in units]
    root._state_ctx = SimpleNamespace(all_states=[root] + states)
    return root, units


class TestCopyEngineAttach:
    @pytest.mark.parametrize("error", [None, "NCCL host communicator for group 0 not found."])
    @pytest.mark.parametrize("kind", ["all_gather", "reduce_scatter"])
    def test_attaches_only_where_symmetric_memory_rendezvouses(self, kind, error, monkeypatch):
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives

        from primus.backends.automodel.distributed import (
            ce_all_gather,
            ce_reduce_scatter,
        )

        group = _Group()
        monkeypatch.setattr(ce_all_gather, "rendezvous_error", lambda g: error)
        monkeypatch.setattr(_fsdp_collectives, "SymmMemAllGather", lambda g, backend: ("all_gather", g))
        monkeypatch.setattr(ce_reduce_scatter, "_comm_class", lambda checks: lambda g: ("reduce_scatter", g))
        root, units = _fsdp_tree(3, group)
        (ce_all_gather if kind == "all_gather" else ce_reduce_scatter)._attach(root, [])
        assert [getattr(u, kind) for u in units] == [None if error else (kind, group)] * 3
