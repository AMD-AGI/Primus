###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches.patch import FunctionPatch
from primus.core.patches.patch_registry import PatchRegistry, register_patch


class TestPatchRegistry:
    def setup_method(self):
        PatchRegistry.clear()

    def test_register_decorator(self):
        @register_patch("test.patch", backend="megatron")
        def my_patch(ctx):
            pass

        assert "test.patch" in PatchRegistry._patches
        patch = PatchRegistry.get("test.patch")
        assert patch.backend == "megatron"
        # Description falls back to function docstring when not provided
        assert patch.description == ""

    def test_list_ids(self):
        @register_patch("b.patch")
        def p1(ctx):
            pass

        @register_patch("a.patch")
        def p2(ctx):
            pass

        assert PatchRegistry.list_ids() == ["a.patch", "b.patch"]

    def test_iter_by_tag(self):
        @register_patch("p1", tags=["tag1"])
        def p1(ctx):
            pass

        @register_patch("p2", tags=["tag2"])
        def p2(ctx):
            pass

        @register_patch("p3", tags=["tag1", "tag2"])
        def p3(ctx):
            pass

        tag1_patches = list(PatchRegistry.iter_by_tag("tag1"))
        assert len(tag1_patches) == 2
        ids = sorted([p.id for p in tag1_patches])
        assert ids == ["p1", "p3"]

    def test_iter_patches_and_clear(self):
        @register_patch("p1")
        def p1(ctx):
            pass

        @register_patch("p2")
        def p2(ctx):
            pass

        patches = PatchRegistry.iter_patches()
        ids = sorted(p.id for p in patches)
        assert ids == ["p1", "p2"]

        PatchRegistry.clear()
        assert PatchRegistry.list_ids() == []
        assert list(PatchRegistry.iter_patches()) == []

    def test_register_override_logs_and_replaces(self, caplog):
        # Direct use of PatchRegistry.register to test override semantics
        def h1(ctx):
            pass

        def h2(ctx):
            pass

        p1 = FunctionPatch(id="dup.patch", description="first", handler=h1)
        p2 = FunctionPatch(id="dup.patch", description="second", handler=h2)

        PatchRegistry.clear()
        with caplog.at_level("WARNING"):
            PatchRegistry.register(p1)
            PatchRegistry.register(p2)

        # Latest registration should win
        patch = PatchRegistry.get("dup.patch")
        assert patch.description == "second"
        assert patch.handler is h2
        # Warning about overriding should be logged
        assert "overriding" in caplog.text

    def test_register_patch_full_arguments(self):
        @register_patch(
            "full.patch",
            description="explicit description",
            backend="megatron",
            phase="setup",
            backend_versions=[">=0.8.0"],
            primus_versions=["1.0.0~1.0.5"],
            tags=["megatron", "args"],
        )
        def my_patch(ctx):
            """Docstring should be ignored when description is provided."""

        patch = PatchRegistry.get("full.patch")
        assert patch.id == "full.patch"
        assert patch.description == "explicit description"
        assert patch.backend == "megatron"
        assert patch.phase == "setup"
        assert patch.backend_version_patterns == [">=0.8.0"]
        assert patch.primus_version_patterns == ["1.0.0~1.0.5"]
        assert patch.tags == {"megatron", "args"}
