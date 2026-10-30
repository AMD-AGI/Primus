###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Make Megatron-FSDP's ``average_in_collective`` work without expert parallelism.

WHY:
  With ``fsdp.average_in_collective: true`` Megatron-FSDP reduce-scatters with
  ``ReduceOp.AVG`` instead of first scaling the whole gradient buffer by
  ``1/dp_size``, an elementwise pass that cost 12.7 ms per step on FLUX.1-schnell.
  In megatron-fsdp 0.5.0, ``MegatronFSDP._init_fsdp_param_and_grad_buffer``
  computes the expert gradient scaling factor from
  ``get_dp_group(is_expert_parallel=True).size()`` with no ``None`` check, so a
  model without an expert-parallel group crashes with
  ``AttributeError: 'NoneType' object has no attribute 'size'``.

WHAT:
  Fall back to the regular data-parallel group size when there is no expert
  group, which makes the expert factor 1.0. Nothing reads that factor without
  expert parameters, and the code changes only where it used to crash. If the
  anchor is missing, the installed megatron-fsdp is taken to have fixed this.
"""
from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusMegatronFSDP]"
_PATCH_FLAG = "_primus_average_in_collective_without_expert_dp"
_METHOD = "_init_fsdp_param_and_grad_buffer"

ANCHOR = (
    "                expert_gradient_scaling_factor = (\n"
    "                    self.dist_index.get_dp_group(is_expert_parallel=True).size()\n"
)
REPLACEMENT = (
    "                _expert_dp = self.dist_index.get_dp_group(is_expert_parallel=True)\n"
    "                expert_gradient_scaling_factor = (\n"
    "                    (_expert_dp.size() if _expert_dp is not None else self.dist_index.get_dp_group().size())\n"
)


def install() -> bool:
    """Patch ``megatron_fsdp.MegatronFSDP``. True when the fix is in place."""
    from megatron_fsdp.megatron_fsdp import MegatronFSDP

    return patch_class(MegatronFSDP)


def patch_class(mfsdp_cls) -> bool:
    if getattr(mfsdp_cls, _PATCH_FLAG, False):
        return True

    from primus.backends.automodel._source_patch import patch_method_source

    if ANCHOR not in inspect.getsource(mfsdp_cls.__dict__[_METHOD]):
        logger.warning(
            "%s %s.%s no longer matches the megatron-fsdp 0.5.0 source; assuming a missing expert "
            "group is handled upstream. Not patching.",
            _LOG_PREFIX,
            mfsdp_cls.__name__,
            _METHOD,
        )
        return False

    patch_method_source(mfsdp_cls, _METHOD, [(ANCHOR, REPLACEMENT)])
    setattr(mfsdp_cls, _PATCH_FLAG, True)
    logger.info(
        "%s average_in_collective falls back to the data-parallel group when there is no "
        "expert-parallel group.",
        _LOG_PREFIX,
    )
    return True
