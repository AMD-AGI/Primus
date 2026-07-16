###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import unittest

from primus.core.utils.arg_utils import (
    _coerce_cli_value_legacy,
    _coerce_cli_value_modern,
    parse_cli_overrides,
)


class TestCoerceCliValueModern(unittest.TestCase):
    def test_bool(self):
        self.assertIs(_coerce_cli_value_modern("true"), True)
        self.assertIs(_coerce_cli_value_modern("True"), True)
        self.assertIs(_coerce_cli_value_modern("false"), False)
        self.assertIs(_coerce_cli_value_modern("FALSE"), False)

    def test_int(self):
        self.assertEqual(_coerce_cli_value_modern("0"), 0)
        self.assertEqual(_coerce_cli_value_modern("42"), 42)
        self.assertEqual(_coerce_cli_value_modern("-7"), -7)

    def test_float(self):
        self.assertEqual(_coerce_cli_value_modern("0.001"), 0.001)
        self.assertEqual(_coerce_cli_value_modern("-1.5"), -1.5)

    def test_string_fallback(self):
        self.assertEqual(_coerce_cli_value_modern("hello"), "hello")
        self.assertEqual(_coerce_cli_value_modern("v1.2.3"), "v1.2.3")

    def test_list_literal(self):
        # Regression: `--profile_ranks '[0]'` used to be kept as raw string,
        # which broke Megatron's `int in args.profile_ranks` check.
        self.assertEqual(_coerce_cli_value_modern("[0]"), [0])
        self.assertEqual(_coerce_cli_value_modern("[0, 1, 2]"), [0, 1, 2])
        self.assertEqual(_coerce_cli_value_modern("[]"), [])
        self.assertEqual(_coerce_cli_value_modern('["a", "b"]'), ["a", "b"])

    def test_tuple_literal(self):
        self.assertEqual(_coerce_cli_value_modern("(1, 2)"), (1, 2))

    def test_dict_literal(self):
        self.assertEqual(_coerce_cli_value_modern("{'a': 1}"), {"a": 1})
        self.assertEqual(_coerce_cli_value_modern("{}"), {})

    def test_none_literal(self):
        self.assertIsNone(_coerce_cli_value_modern("None"))

    def test_malformed_container_falls_back_to_string(self):
        # If literal_eval fails, keep the original raw string instead of crashing.
        self.assertEqual(_coerce_cli_value_modern("[0,"), "[0,")
        self.assertEqual(_coerce_cli_value_modern("{not_a_dict}"), "{not_a_dict}")


class TestParseCliOverrides(unittest.TestCase):
    def test_list_override_via_space(self):
        result = parse_cli_overrides(["--profile_ranks", "[0]"])
        self.assertEqual(result, {"profile_ranks": [0]})

    def test_list_override_via_equals(self):
        result = parse_cli_overrides(["--profile_ranks=[0, 4]"])
        self.assertEqual(result, {"profile_ranks": [0, 4]})

    def test_mixed_overrides(self):
        result = parse_cli_overrides(
            [
                "--profile",
                "true",
                "--profile_ranks",
                "[0, 1]",
                "--lr",
                "0.001",
                "--train_iters",
                "10",
            ]
        )
        self.assertEqual(
            result,
            {
                "profile": True,
                "profile_ranks": [0, 1],
                "lr": 0.001,
                "train_iters": 10,
            },
        )

    def test_legacy_mode_unchanged(self):
        # Sanity: legacy eval-based path still works for list literals.
        result = parse_cli_overrides(["--profile_ranks", "[0, 1]"], type_mode="legacy")
        self.assertEqual(result, {"profile_ranks": [0, 1]})


class TestCoerceCliValueLegacyUnchanged(unittest.TestCase):
    """Guard against accidental regression in the legacy coerce path."""

    def test_bool_and_list_and_string(self):
        self.assertIs(_coerce_cli_value_legacy("true"), True)
        self.assertEqual(_coerce_cli_value_legacy("[0, 1]"), [0, 1])
        self.assertEqual(_coerce_cli_value_legacy("hello"), "hello")


if __name__ == "__main__":
    unittest.main()
