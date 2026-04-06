# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FLUX recipe public module.

Imports here stay minimal so ``importlib.import_module(...)`` used by Primus
does not pull in ``megatron.bridge.training.config`` (and thus the full
``megatron.bridge.models`` tree) before training starts. That avoids failures
when an environment has a broken or stub ``modelopt`` package that blows up
during unrelated eager imports.

Heavy dependencies load only when ``flux_12b_*_config()`` is called.
"""

from __future__ import annotations

from typing import Any


def flux_12b_pretrain_config(*_args: Any, **_kwargs: Any) -> Any:
    """Return default pretrain ConfigContainer.

    Primus merges the full module/backend namespace into kwargs when loading the recipe.
    The implementation takes no parameters; flat overrides are applied afterward via
    ``_apply_flat_config_knobs`` on the returned ``ConfigContainer``.
    """
    from primus.diffusion.recipes.flux import _flux_recipe_impl

    return _flux_recipe_impl.flux_12b_pretrain_config()


def flux_12b_sft_config(*_args: Any, **kwargs: Any) -> Any:
    """Same kwargs-swallowing pattern; only ``pretrained_checkpoint`` is passed to the impl."""
    from primus.diffusion.recipes.flux import _flux_recipe_impl

    return _flux_recipe_impl.flux_12b_sft_config(
        pretrained_checkpoint=kwargs.get("pretrained_checkpoint"),
    )
