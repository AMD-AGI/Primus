###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.utils.module_utils import log_rank_0
from primus.core.utils.yaml_utils import nested_namespace_to_dict

# model_family (set by the model preset) -> (module, trainer class). Each family
# brings its own trainer code and environment defaults (module-level ENV_DEFAULTS).
MODEL_FAMILIES: Dict[str, Tuple[str, str]] = {
    "dlrm_v4": ("primus.backends.recommendation.dlrm_v4.trainer", "DlrmV4Trainer"),
}


def _log(msg: str) -> None:
    try:
        log_rank_0(msg)
    except Exception:  # the logger is not bound in standalone hook runs
        print(msg, file=sys.stderr)


def _params_dict(params: Any) -> dict:
    if params is None:
        return {}
    return params if isinstance(params, dict) else nested_namespace_to_dict(params)


def model_family(params: Any) -> str:
    family = _params_dict(params).get("model_family")
    if family not in MODEL_FAMILIES:
        raise ValueError(
            f"[Primus:recommendation] model_family={family!r}: set it in the model preset "
            f"(one of {sorted(MODEL_FAMILIES)})"
        )
    return family


class RecommendationAdapter(BackendAdapter):
    """Primus adapter for recommendation models. The model preset's
    `model_family` picks the trainer; the trainer code is in-tree."""

    def __init__(self, framework: str = "recommendation"):
        super().__init__(framework)
        self.family: Optional[str] = None
        self.trainer_src: Optional[str] = None

    def _family_module(self):
        return importlib.import_module(MODEL_FAMILIES[self.family][0])

    def setup_backend_path(self, backend_path=None) -> str:
        # In-tree backend: nothing to put on sys.path here (each family's trainer
        # adds its own source root). --backend_path / BACKEND_PATH name another
        # checkout of the family's trainer source instead.
        override = backend_path or os.getenv("BACKEND_PATH")
        if override:
            if not Path(override).is_dir():
                raise FileNotFoundError(f"[Primus:recommendation] backend path not found: {override}")
            self.trainer_src = str(Path(override).resolve())
            _log(f"[Primus:recommendation] trainer source override -> {self.trainer_src}")
        return str(Path(__file__).resolve().parent)

    def env_defaults(self) -> List:
        if self.family is None:
            return []
        return list(getattr(self._family_module(), "ENV_DEFAULTS", []))

    def prepare_backend(self, config: Any):
        self.family = model_family(getattr(config, "params", None))
        super().prepare_backend(config)

    def convert_config(self, params: Any) -> Any:
        values = dict(_params_dict(params))
        self.family = model_family(values)
        if self.trainer_src and not values.get("trainer_src"):
            values["trainer_src"] = self.trainer_src
        return SimpleNamespace(**values)

    def load_trainer_class(self, stage: str = "pretrain"):
        if stage != "pretrain":
            raise ValueError(f"Invalid stage for the recommendation backend: {stage}")
        module_name, class_name = MODEL_FAMILIES[self.family]
        return getattr(importlib.import_module(module_name), class_name)

    def detect_backend_version(self) -> str:
        return "in-tree"
