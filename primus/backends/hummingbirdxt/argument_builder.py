###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict

from primus.core.config.merge_utils import deep_merge
from primus.core.utils.yaml_utils import (
    dict_to_nested_namespace,
    nested_namespace_to_dict,
)


def _load_hummingbirdxt_default_args() -> Dict[str, Any]:
    from train import get_arg_parser

    parser = get_arg_parser()

    default_args_dict = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            default_args_dict[action.dest] = action.default

    # used for wandb, will be overwritten if it has been set in yaml config
    default_args_dict["config_name"] = "primus_train_hummingbirdxt_{}".format(
        datetime.now().strftime("%Y%m%d%H%M%S")
    )
    return default_args_dict


class HummingbirdXTArgBuilder:

    def __init__(self) -> None:
        self.config: Dict[str, Any] = _load_hummingbirdxt_default_args()

    def update(self, values: SimpleNamespace) -> "HummingbirdXTArgBuilder":
        # Convert SimpleNamespace to dict
        values_dict = nested_namespace_to_dict(values)
        # Directly merge into the working configuration
        self.config = deep_merge(self.config, values_dict)
        return self

    def to_dict(self) -> Dict[str, Any]:
        import copy

        return copy.deepcopy(self.config)

    def to_namespace(self) -> SimpleNamespace:
        merged = self.to_dict()
        return dict_to_nested_namespace(merged)

    finalize = to_namespace
