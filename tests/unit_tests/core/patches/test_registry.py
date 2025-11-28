from primus.core.patches.registry import PatchRegistry, register_patch


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
