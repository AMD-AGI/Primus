from primus.core.configs.merge_utils import deep_merge, shallow_merge


class TestMergeUtils:

    def test_deep_merge_basic(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        expected = {"a": 1, "b": 3, "c": 4}
        assert deep_merge(base, override) == expected

    def test_deep_merge_recursive(self):
        base = {"nested": {"x": 1, "y": 2}}
        override = {"nested": {"y": 3, "z": 4}}
        expected = {"nested": {"x": 1, "y": 3, "z": 4}}
        assert deep_merge(base, override) == expected

    def test_deep_merge_type_mismatch(self):
        """Test overriding a dict with a primitive and vice-versa."""
        base = {"a": {"x": 1}}
        override = {"a": 2}
        assert deep_merge(base, override) == {"a": 2}

        base = {"a": 1}
        override = {"a": {"x": 2}}
        assert deep_merge(base, override) == {"a": {"x": 2}}

    def test_deep_merge_list(self):
        """Lists should be replaced, not merged."""
        base = {"a": [1, 2]}
        override = {"a": [3]}
        assert deep_merge(base, override) == {"a": [3]}

    def test_deep_merge_immutability(self):
        """Ensure inputs are not modified."""
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}

    def test_shallow_merge(self):
        base = {"a": 1, "nested": {"x": 1}}
        override = {"b": 2, "nested": {"y": 2}}

        # In shallow merge, the "nested" key in override completely replaces the one in base
        expected = {"a": 1, "b": 2, "nested": {"y": 2}}
        assert shallow_merge(base, override) == expected

    def test_shallow_merge_immutability(self):
        base = {"a": 1}
        override = {"b": 2}
        shallow_merge(base, override)
        assert base == {"a": 1}
