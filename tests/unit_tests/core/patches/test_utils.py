###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches.utils import version_in_range, version_matches


class TestVersionMatches:
    def test_exact_match(self):
        assert version_matches("0.8.0", "0.8.0")
        assert not version_matches("0.8.0", "0.8.1")
        assert version_matches("v1.2.3", "v1.2.3")

    def test_wildcard_match(self):
        # Prefix match
        assert version_matches("0.8.0", "0.8.*")
        assert version_matches("0.8.15", "0.8.*")
        assert not version_matches("0.9.0", "0.8.*")

        # Suffix match
        assert version_matches("commit:abc1234", "commit:*")
        assert version_matches("my-branch-v1", "*v1")

        # Infix match
        assert version_matches("0.8.0-rc1", "0.8.*-rc*")
        assert version_matches("1.2.3", "1.*.3")
        assert not version_matches("1.2.4", "1.*.3")

        # Multiple wildcards
        assert version_matches("abc-123-xyz", "*123*")
        assert version_matches("feature/new-algo", "feature/*")

    def test_special_characters(self):
        # Test with semver build metadata and pre-releases
        assert version_matches("1.0.0+20130313144700", "1.0.0+*")
        assert version_matches("1.0.0-beta.1", "1.0.0-beta.*")

        # Regex special chars in version (should be treated literally)
        assert version_matches("1.0+test", "1.0+*")

    def test_edge_cases(self):
        # Empty inputs
        assert not version_matches(None, "0.8.0")
        assert not version_matches("", "0.8.0")

        # Empty pattern (exact match against empty string?)
        # Assuming implementation behavior: regex "^$" matches empty string
        assert not version_matches("0.8.0", "")

        # Match everything
        assert version_matches("any_string", "*")
        assert version_matches("", "*") is False  # Empty version returns False early


class TestVersionInRange:
    def test_exact_and_wildcard_delegation(self):
        # Exact match
        assert version_in_range("0.8.0", "0.8.0")
        assert not version_in_range("0.8.1", "0.8.0")

        # Wildcard should delegate to version_matches
        assert version_in_range("0.8.1", "0.8.*")
        assert not version_in_range("0.9.0", "0.8.*")

        # Non-semver but wildcard pattern should still work
        assert version_in_range("commit:abc123", "commit:*")
        assert version_in_range("feature/patch-v2", "*v2")

    def test_comparators_basic(self):
        # >=
        assert version_in_range("0.8.0", ">=0.8.0")
        assert version_in_range("0.8.1", ">=0.8.0")
        assert not version_in_range("0.7.9", ">=0.8.0")

        # >
        assert version_in_range("0.8.1", ">0.8.0")
        assert not version_in_range("0.8.0", ">0.8.0")

        # <=
        assert version_in_range("0.8.0", "<=0.8.0")
        assert version_in_range("0.7.9", "<=0.8.0")
        assert not version_in_range("0.8.1", "<=0.8.0")

        # <
        assert version_in_range("0.7.9", "<0.8.0")
        assert not version_in_range("0.8.0", "<0.8.0")

    def test_between_range(self):
        # Inclusive on both ends: lo <= v <= hi
        assert version_in_range("0.8.0", "0.8.0~0.8.5")
        assert version_in_range("0.8.3", "0.8.0~0.8.5")
        assert version_in_range("0.8.5", "0.8.0~0.8.5")

        assert not version_in_range("0.7.9", "0.8.0~0.8.5")
        assert not version_in_range("0.8.6", "0.8.0~0.8.5")

    def test_invalid_semver_in_comparators_and_ranges(self):
        # Non-numeric versions with comparator or range should return False
        assert not version_in_range("commit:abc123", ">=0.8.0")
        assert not version_in_range("1.0.0-rc1", ">=1.0.0")
        assert not version_in_range("1.0.0-rc1", "1.0.0~1.0.1")

        # But the same non-semver should still work with wildcard-only pattern
        assert version_in_range("1.0.0-rc1", "1.0.0-*")

    def test_edge_cases(self):
        # Empty or None version should be rejected
        assert not version_in_range(None, ">=0.8.0")
        assert not version_in_range("", ">=0.8.0")

        # Invalid comparator pattern (no digits) falls back to version_matches
        # Here pattern is just "*", so it should match any non-empty version
        assert version_in_range("0.8.0", "*")
        # But still respects the version empty check
        assert not version_in_range("", "*")
