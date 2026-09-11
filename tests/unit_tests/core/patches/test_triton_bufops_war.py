###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Detection of the buffer-store WAR hazard, and the gate that enables it.

Pure text analysis of AMDGCN, so these run anywhere; no GPU and no Triton.
The patch that uses this only recompiles kernels the detector flags, which
makes a false negative a silently wrong training run and a false positive
merely a slower one.
"""

import inspect
import sys
import types

import primus.core.patches.triton_bufops_war_patches as war


def _asm(*lines: str) -> str:
    return "\n".join(f"\t{line}" for line in lines)


class TestHasHazard:
    def test_store_then_immediate_clobber(self):
        """The bug: v8 is redefined while the store that reads it is in flight."""
        assert war.has_hazard(
            _asm(
                "buffer_store_dwordx4 v[8:11], v4, s[0:3], 0 offen",
                "v_add_f32_e32 v8, v12, v13",
            )
        )

    def test_vmcnt_wait_clears_it(self):
        assert not war.has_hazard(
            _asm(
                "buffer_store_dwordx4 v[8:11], v4, s[0:3], 0 offen",
                "s_waitcnt vmcnt(0)",
                "v_add_f32_e32 v8, v12, v13",
            )
        )

    def test_wait_without_vmcnt_does_not_clear_it(self):
        """Only a vmcnt wait orders the store; lgkmcnt says nothing about it."""
        assert war.has_hazard(
            _asm(
                "buffer_store_dwordx4 v[8:11], v4, s[0:3], 0 offen",
                "s_waitcnt lgkmcnt(0)",
                "v_add_f32_e32 v8, v12, v13",
            )
        )

    def test_clobber_inside_register_range(self):
        """The store reads v[8:11]; redefining any one of them is enough."""
        assert war.has_hazard(
            _asm(
                "buffer_store_dwordx4 v[8:11], v4, s[0:3], 0 offen",
                "v_mul_f32_e32 v10, v12, v13",
            )
        )

    def test_partially_overlapping_write_range(self):
        """A wide write only has to straddle one end of the store's range."""
        assert war.has_hazard(
            _asm(
                "buffer_store_dwordx4 v[8:11], v4, s[0:3], 0 offen",
                "v_lshlrev_b64 v[10:13], 2, v[2:3]",
            )
        )

    def test_adjacent_write_range_is_clean(self):
        assert not war.has_hazard(
            _asm(
                "buffer_store_dwordx4 v[8:11], v4, s[0:3], 0 offen",
                "v_lshlrev_b64 v[12:15], 2, v[2:3]",
            )
        )

    def test_unrelated_register_is_clean(self):
        assert not war.has_hazard(
            _asm(
                "buffer_store_dwordx4 v[8:11], v4, s[0:3], 0 offen",
                "v_add_f32_e32 v20, v12, v13",
            )
        )

    def test_narrow_stores_are_ignored(self):
        """Only dwordx4 has been observed to miscompile, and buffer ops are kept
        wherever possible because dropping them is what costs throughput."""
        assert not war.has_hazard(
            _asm(
                "buffer_store_dwordx2 v[8:9], v4, s[0:3], 0 offen",
                "v_add_f32_e32 v8, v12, v13",
            )
        )

    def test_global_stores_are_ignored(self):
        assert not war.has_hazard(
            _asm(
                "global_store_dwordx4 v4, v[8:11], s[0:1]",
                "v_add_f32_e32 v8, v12, v13",
            )
        )

    def test_end_of_program_terminates_the_scan(self):
        assert not war.has_hazard(
            _asm(
                "buffer_store_dwordx4 v[8:11], v4, s[0:3], 0 offen",
                "s_endpgm",
                "v_add_f32_e32 v8, v12, v13",
            )
        )

    def test_comments_and_directives_are_skipped(self):
        assert war.has_hazard(
            _asm(
                "buffer_store_dwordx4 v[8:11], v4, s[0:3], 0 offen",
                "; %bb.1:",
                ".p2align 6",
                "v_add_f32_e32 v8, v12, v13",
            )
        )

    def test_second_store_is_still_checked(self):
        """A clean first store must not stop the scan."""
        assert war.has_hazard(
            _asm(
                "buffer_store_dwordx4 v[4:7], v0, s[0:3], 0 offen",
                "s_waitcnt vmcnt(0)",
                "buffer_store_dwordx4 v[8:11], v0, s[0:3], 0 offen",
                "v_add_f32_e32 v9, v12, v13",
            )
        )

    def test_kernel_without_buffer_stores(self):
        assert not war.has_hazard(_asm("v_add_f32_e32 v8, v12, v13", "s_endpgm"))

    def test_empty_input(self):
        assert not war.has_hazard("")


class TestEnableGate:
    """The gate must not narrow to an architecture allowlist.

    Wrong values were measured on gfx942; gfx950 emits the same pattern and was
    measured to tolerate it. Since the omitted `s_waitcnt` is a compiler defect
    on both, which chips tolerate it is not something to encode in a gate.
    Whether a kernel is affected is decided by its emitted machine code, so the
    gate only asks whether this is a ROCm GPU at all.
    """

    def _fake_torch(self, hip):
        return types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            version=types.SimpleNamespace(hip=hip),
        )

    def test_enabled_on_any_rocm_gpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self._fake_torch("6.4.0"))
        assert war._on_rocm()

    def test_disabled_on_a_cuda_build(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self._fake_torch(None))
        assert not war._on_rocm()

    def test_disabled_when_torch_is_missing(self, monkeypatch):
        """The module also imports on machines with no torch at all."""
        monkeypatch.setitem(sys.modules, "torch", None)
        assert not war._on_rocm()

    def test_gate_does_not_inspect_the_architecture_name(self):
        """A gate reading gcnArchName is how the gfx950 blind spot happened."""
        assert "gcnArchName" not in inspect.getsource(war._on_rocm)
