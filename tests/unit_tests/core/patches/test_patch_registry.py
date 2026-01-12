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

        # Check that patch is registered
        patch = PatchRegistry.get("test.patch")
        assert patch is not None
        assert patch.backend == "megatron"
        # Description falls back to function docstring when not provided
        assert patch.description == ""

        # Verify it's in the correct bucket
        assert patch.id in [p.id for p in PatchRegistry._all_patches]

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
        @register_patch("p1", backend="megatron", phase="before_train")
        def p1(ctx):
            pass

        @register_patch("p2", backend="pytorch", phase="setup")
        def p2(ctx):
            pass

        # Test iter_patches with backend filter
        megatron_patches = PatchRegistry.iter_patches(backend="megatron")
        assert len(megatron_patches) == 1
        assert megatron_patches[0].id == "p1"

        pytorch_patches = PatchRegistry.iter_patches(backend="pytorch", phase="setup")
        assert len(pytorch_patches) == 1
        assert pytorch_patches[0].id == "p2"

        PatchRegistry.clear()
        assert PatchRegistry.list_ids() == []
        # After clear, should have no patches for any backend
        assert PatchRegistry.iter_patches(backend="megatron") == []

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

    def test_iter_patches_backend_phase_filtering(self):
        """Test pre-classified patch retrieval by backend and phase."""

        # Create patches in different buckets
        @register_patch("megatron.before_train", backend="megatron", phase="before_train")
        def p1(ctx):
            pass

        @register_patch("megatron.after_train", backend="megatron", phase="after_train")
        def p2(ctx):
            pass

        @register_patch("pytorch.before_train", backend="pytorch", phase="before_train")
        def p3(ctx):
            pass

        @register_patch("generic.before_train", backend=None, phase="before_train")
        def p4(ctx):
            pass

        @register_patch("megatron.generic", backend="megatron", phase=None)
        def p5(ctx):
            pass

        @register_patch("generic.generic", backend=None, phase=None)
        def p6(ctx):
            pass

        # Test 1: Query specific backend and phase
        patches = PatchRegistry.iter_patches(backend="megatron", phase="before_train")
        patch_ids = [p.id for p in patches]
        # Should include: (megatron, before_train) + (None, before_train) + (megatron, None) + (None, None)
        assert "megatron.before_train" in patch_ids
        assert "generic.before_train" in patch_ids
        assert "megatron.generic" in patch_ids
        assert "generic.generic" in patch_ids
        # Should NOT include patches from other backends or phases
        assert "megatron.after_train" not in patch_ids
        assert "pytorch.before_train" not in patch_ids

        # Test 2: Query backend without phase (all phases for that backend)
        patches = PatchRegistry.iter_patches(backend="megatron", phase=None)
        patch_ids = [p.id for p in patches]
        # Should include all megatron patches and generic patches
        assert "megatron.before_train" in patch_ids
        assert "megatron.after_train" in patch_ids
        assert "megatron.generic" in patch_ids
        assert "generic.before_train" in patch_ids
        assert "generic.generic" in patch_ids
        # Should NOT include pytorch patches
        assert "pytorch.before_train" not in patch_ids

        # Test 3: Query different backend
        patches = PatchRegistry.iter_patches(backend="pytorch", phase="before_train")
        patch_ids = [p.id for p in patches]
        assert "pytorch.before_train" in patch_ids
        assert "generic.before_train" in patch_ids
        assert "generic.generic" in patch_ids
        # Should NOT include megatron-specific patches
        assert "megatron.before_train" not in patch_ids
        assert "megatron.after_train" not in patch_ids

    def test_iter_patches_deduplication(self):
        """Test that iter_patches doesn't return duplicates."""

        @register_patch("test.patch", backend="megatron", phase="before_train")
        def p1(ctx):
            pass

        patches = PatchRegistry.iter_patches(backend="megatron", phase="before_train")
        patch_ids = [p.id for p in patches]
        # Each patch should appear only once
        assert patch_ids.count("test.patch") == 1
