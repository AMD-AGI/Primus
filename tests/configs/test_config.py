###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import argparse
import unittest
from types import SimpleNamespace

from primus.core.launcher.parser import (
    PrimusParser,
    _check_keys_exist,
    _deep_merge_namespace,
    _parse_kv_overrides,
)
from primus.core.utils import logger
from tests.utils import PrimusUT


class TestPrimusParser(PrimusUT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        self.config_parser = PrimusParser()
        self.cli_args = argparse.Namespace()

    def tearDown(self):
        pass

    def parse_config(self, cli_args: argparse.Namespace):
        exp_config = self.config_parser.parse(cli_args)
        return exp_config

    def test_exp_configs(self):
        exps = [
            "examples/megatron/exp_pretrain.yaml",
            "examples/torchtitan/configs/llama3.1_8B-BF16-pretrain.yaml",
        ]

        for exp in exps:
            self.cli_args.config = exp
            logger.info(f"test exp config: {exp}")
            logger.debug(f"============================")
            exp_config = self.parse_config(self.cli_args)
            logger.debug(f"exp config: \n{exp_config}")
            logger.debug(f"============================")

    def test_parse_equal_format(self):
        args = ["--a=1", "--b.c=2", "--flag"]
        result = _parse_kv_overrides(args)
        expected = {"a": 1, "b": {"c": 2}, "flag": True}
        self.assertEqual(result, expected)

    def test_parse_space_format(self):
        args = ["--a", "1", "--b.c", "2", "--flag"]
        result = _parse_kv_overrides(args)
        expected = {"a": 1, "b": {"c": 2}, "flag": True}
        self.assertEqual(result, expected)

    def test_override_check_pass(self):
        # Simulate pre_trainer config
        ns = SimpleNamespace(a=1, b=SimpleNamespace(c=2), flag=False)
        overrides = {"a": 123, "b": {"c": 999}, "flag": True}
        # Should not raise
        _check_keys_exist(ns, overrides)

    def test_override_check_fail(self):
        ns = SimpleNamespace(a=1)
        overrides = {"missing": 10}
        with self.assertRaises(AssertionError) as context:
            _check_keys_exist(ns, overrides)
        self.assertIn("Override key 'missing' does not exist", str(context.exception))

    def test_deep_merge_namespace(self):
        ns = SimpleNamespace(a=1, b=SimpleNamespace(c=2), flag=False)
        overrides = {"a": 10, "b": {"c": 20}, "flag": True}
        _deep_merge_namespace(ns, overrides)
        self.assertEqual(ns.a, 10)
        self.assertEqual(ns.b.c, 20)
        self.assertEqual(ns.flag, True)

    def test_export_and_parse_cycle_with_real_yaml(self):
        """
        Test that a config exported by PrimusConfig.export from a real YAML
        can be parsed back by PrimusParser.parse without loss or error.
        """
        import os
        import tempfile

        # Step 1: Load original config using parser
        original_args = argparse.Namespace(config="examples/megatron/exp_pretrain.yaml")
        original_cfg = self.config_parser.parse(original_args)

        # Step 2: Export to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "exported_config.yaml")
            self.config_parser.export(out_path)

            # Step 3: Parse the exported config
            reload_args = argparse.Namespace(config=out_path)
            reloaded_cfg = self.config_parser.parse(reload_args)

            # Step 4: Assert that some key fields match (e.g. model, modules)
            self.assertEqual(
                getattr(reloaded_cfg._exp, "model", None), getattr(original_cfg._exp, "model", None)
            )
            self.assertEqual(hasattr(reloaded_cfg._exp, "modules"), hasattr(original_cfg._exp, "modules"))
            # You can check more fields as needed

            # Optionally, assert all top-level keys are the same
            orig_keys = set(vars(original_cfg._exp).keys())
            reload_keys = set(vars(reloaded_cfg._exp).keys())
            self.assertEqual(orig_keys, reload_keys)


if __name__ == "__main__":
    unittest.main()
