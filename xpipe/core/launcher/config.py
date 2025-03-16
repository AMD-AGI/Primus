###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

import argparse
import os
from types import SimpleNamespace

from xpipe.core.utils import file_utils, yaml_utils


class XPipeConfig(object):
    def __init__(self, cli_args: argparse.Namespace, exp: SimpleNamespace):
        self._cli_args = cli_args
        self._exp = exp
        self._exp_root_path = os.path.join(
            self._exp.platform.workspace, self._exp.work_group, self._exp.user_name, self._exp.exp_name
        )
        file_utils.create_path_if_not_exists(self._exp_root_path)

    def __str__(self):
        return yaml_utils.parse_nested_namespace_to_str(self._exp)

    @property
    def cli_args(self) -> argparse.Namespace:
        return self._cli_args

    @property
    def exp_root_path(self) -> str:
        return self._exp_root_path

    @property
    def exp_meta_info(self) -> dict:
        return {
            "work_group": self._exp.work_group,
            "user_name": self._exp.user_name,
            "exp_name": self._exp.exp_name,
        }

    @property
    def platform_config(self) -> SimpleNamespace:
        return self._exp.platform

    @property
    def num_modules(self) -> int:
        return len(self.module_keys)

    @property
    def module_keys(self) -> list:
        return list(self._exp.modules.__dict__.keys())

    def get_module_config(self, module_name: str) -> SimpleNamespace:
        if not yaml_utils.has_key_in_namespace(self._exp.modules, module_name):
            raise ValueError(f"XPipe config ({self._exp.config_file}) has no module named {module_name}")
        module_config = yaml_utils.get_value_by_key(self._exp.modules, module_name)
        return module_config
