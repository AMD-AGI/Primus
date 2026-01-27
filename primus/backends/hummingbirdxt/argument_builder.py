###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from primus.core.config.merge_utils import deep_merge
from primus.core.utils.yaml_utils import (
    dict_to_nested_namespace,
    nested_namespace_to_dict,
)


class HummingbirdXTArgBuilder:

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}

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
