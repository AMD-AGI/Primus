###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import unittest

from primus.core.utils.arg_utils import (
    _coerce_cli_value_legacy,
    _coerce_cli_value_modern,
    as_int,
    is_unset,
    parse_cli_overrides,
    parse_optional_switch,
    parse_switch,
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

    def test_scientific_notation_float(self):
        # Regression: `--lr 1e-5` used to stay the string "1e-5" because the
        # float branch only ran when the token contained ".".
        self.assertEqual(_coerce_cli_value_modern("1e-5"), 1e-5)
        self.assertEqual(_coerce_cli_value_modern("2e-4"), 2e-4)
        self.assertEqual(_coerce_cli_value_modern("1E3"), 1000.0)
        self.assertEqual(_coerce_cli_value_modern("-3.5e-2"), -3.5e-2)
        for token in ("1e-5", "2e-4", "1E3"):
            self.assertIsInstance(_coerce_cli_value_modern(token), float)

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

    def test_scientific_notation_override(self):
        result = parse_cli_overrides(["--lr", "1e-5", "--min_lr", "1e-6"])
        self.assertEqual(result, {"lr": 1e-5, "min_lr": 1e-6})
        self.assertIsInstance(result["lr"], float)

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


class TestSettingHelpers(unittest.TestCase):
    """is_unset / parse_switch / as_int (used by the posttrain convert hook and the OLMo-3 code)."""

    def test_is_unset(self):
        for v in (None, "", " ", "null", "NULL", "~"):
            self.assertTrue(is_unset(v), v)
        # "none" is a value: a model or tokenizer path, or an enum value of some options
        for v in ("x", "0", 0, False, "nil", [], "nul", "none", "None"):
            self.assertFalse(is_unset(v), v)

    def test_parse_optional_switch(self):
        for v in (None, "", "null", "~"):
            self.assertIs(parse_optional_switch(v, "k"), False, v)
        with self.assertRaisesRegex(ValueError, "k must be true or false"):
            parse_optional_switch("none", "k")
        self.assertIs(parse_optional_switch("on", "k"), True)
        self.assertIs(parse_optional_switch(False, "k"), False)
        with self.assertRaisesRegex(ValueError, "k must be true or false"):
            parse_optional_switch("maybe", "k")

    def test_parse_switch(self):
        for v in (True, 1, "true", "TRUE", " yes ", "On", "1", " 1 "):
            self.assertIs(parse_switch(v, "k"), True)
        for v in (False, 0, "false", "False", "no", "OFF", "0"):
            self.assertIs(parse_switch(v, "k"), False)
        # only the table: no number forms (a typo such as "+1" or "00" is refused, not read as 1 / 0)
        bad = ("maybe", "", 2, 1.0, 0.0, -0.0, 0.5, "1.0", "0.0", "+1", "-0", "00", " 1.000 ", [], None)
        for v in bad:
            with self.assertRaisesRegex(ValueError, "k must be true or false"):
                parse_switch(v, "k")

    def test_as_int(self):
        self.assertEqual([as_int(v, "k") for v in (4, 4.0, "4", " -2 ", "+3", "4.0")], [4, 4, 4, -2, 3, 4])
        self.assertEqual(as_int("4.00", "k"), 4)
        for bad in (
            True,
            4.5,
            1e20,
            float("inf"),
            "two",
            "4.5",
            "4.",
            "\uff11\uff12",
            "\u0663",
            None,
            "",
            [4],
        ):
            with self.assertRaisesRegex(ValueError, "k must be an integer"):
                as_int(bad, "k")
        with self.assertRaisesRegex(ValueError, "unresolved"):
            as_int("${TP:1}", "k")
        with self.assertRaisesRegex(ValueError, "k must be an integer"):
            as_int("9" * 5000, "k")


if __name__ == "__main__":
    unittest.main()
