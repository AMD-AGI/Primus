###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the inductor precision-cast emulation patch.

Regression focus: the patch has to reach inductor before the first compile and
only for compiled runs, and it must not clobber a value somebody else already
set -- TorchTitan turns the same flag on for float8 rowwise recipes.

The last test pins the reason no cache busting is needed: the flag is part of
the inductor FX graph cache key.  If a torch upgrade ever moves it into one of
the ignore lists, caches populated before the patch would keep serving the
kernels that produce the NaN, silently, and this patch would need the same
``TORCH_COMPILE_CACHE_KEY_TAG`` treatment as the Triton buffer-store one.
"""

from types import SimpleNamespace
from unittest.mock import patch

import primus.backends.torchtitan.patches.inductor_precision_casts_patches as pc_patch
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry

PATCH_ID = "torchtitan.torch.inductor_precision_casts"


def _ctx(compile_enable=True):
    params = SimpleNamespace(compile=SimpleNamespace(enable=compile_enable))
    module_config = SimpleNamespace(params=params)
    return PatchContext(backend="torchtitan", phase="setup", extra={"module_config": module_config})


class TestRegistration:
    def test_patch_registered(self):
        assert PATCH_ID in PatchRegistry.list_ids()
        p = PatchRegistry.get(PATCH_ID)
        assert p is not None
        assert p.backend == "torchtitan"
        # Must land before parallelize_fn calls apply_compile.
        assert p.phase == "setup"


class TestCondition:
    def test_enabled_when_compiling(self):
        import torch

        with patch.object(torch.version, "hip", "7.15"):
            assert pc_patch._compile_enabled(_ctx(compile_enable=True)) is True

    def test_disabled_off_rocm(self):
        import torch

        with patch.object(torch.version, "hip", None):
            assert pc_patch._compile_enabled(_ctx(compile_enable=True)) is False

    def test_disabled_without_compile(self):
        assert pc_patch._compile_enabled(_ctx(compile_enable=False)) is False

    def test_disabled_when_config_absent(self):
        ctx = PatchContext(backend="torchtitan", phase="setup", extra={})
        assert pc_patch._compile_enabled(ctx) is False


class _AlreadyOn:
    def __bool__(self):
        return True


class TestApply:
    # Patch the attribute on the real config module: swapping the module out of
    # sys.modules makes torch re-import torch._inductor.test_operators, which
    # fails on a duplicate TORCH_LIBRARY registration.
    def test_turns_the_flag_on(self):
        import torch._inductor.config as inductor_config

        with patch.object(inductor_config, "emulate_precision_casts", False), patch.object(
            pc_patch, "log_rank_0"
        ):
            pc_patch.patch_inductor_precision_casts(_ctx())
            assert inductor_config.emulate_precision_casts is True

    def test_leaves_an_existing_true_alone(self):
        # TorchTitan sets this for float8 rowwise; re-setting it must stay a no-op
        # rather than racing with whoever owns the value.  A truthy marker instead
        # of True makes the early return observable.
        import torch._inductor.config as inductor_config

        marker = _AlreadyOn()
        with patch.object(inductor_config, "emulate_precision_casts", marker), patch.object(
            pc_patch, "log_rank_0"
        ):
            pc_patch.patch_inductor_precision_casts(_ctx())
            assert inductor_config.emulate_precision_casts is marker


class TestCacheKeyParticipation:
    def test_flag_is_part_of_the_inductor_cache_key(self):
        import torch._inductor.config as inductor_config

        ignore = set(getattr(inductor_config, "_save_config_ignore", ()) or ())
        prefixes = tuple(getattr(inductor_config, "_cache_config_ignore_prefix", ()) or ())
        assert "emulate_precision_casts" not in ignore
        assert not any("emulate_precision_casts".startswith(p) for p in prefixes)
