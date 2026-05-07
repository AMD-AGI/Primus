###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Custom WeightGradStore that bridges TE's native delay_wgrad_compute mechanism
with Primus's pipeline-parallel wgrad scheduling (WGradRunningCache / zero-bubble).

Instead of monkey-patching TE's entire _Linear / _GroupedLinear autograd Functions
(~1000+ lines that break on every TE upgrade), we patch only WeightGradStore
(3 methods) to leverage TE's built-in delay_wgrad_compute:

  1. __init__: Force delay_wgrad_compute=True so TE backward splits dgrad/wgrad.
  2. put(): Forward the wgrad closure to Primus's WGradRunningCache or
     zero-bubble WeightGradStore.
  3. pop(): No-op since wgrad execution is driven by the Primus scheduler.

TE's backward handles all FP8 quantization, communication overlap, and GEMM
logic internally; we only intercept the final wgrad closure.
"""

import queue

import torch
from megatron.training import get_args
from transformer_engine.pytorch.module._common import WeightGradStore

_original_init = WeightGradStore.__init__
_original_put = WeightGradStore.put
_original_pop = WeightGradStore.pop


def _primus_init(self, delay_wgrad_compute=False, ub_bulk_wgrad=False):
    """Force delay_wgrad_compute=True when Primus pipeline scheduling is active."""
    self.context = queue.Queue()
    assert not ub_bulk_wgrad, "ub_bulk_wgrad is not supported with Primus wgrad scheduling"
    self.enabled = True


def _snapshot_for_wgrad(item):
    """Take an independent storage snapshot so the wgrad closure outlives
    Megatron fine_grained_callables ``untyped_storage().resize_(0)``.

    TE's call sites pass ``tensor_list`` in a few shapes:
      - ``[inputmat_total, grad_output]``
      - ``[ln_out_total, dact]``
      - ``[inputmats(list), grad_output(list), wgrad_list(list)]``
    Recursively detach+clone Tensor leaves so that none of them keep a
    reference to a storage that the schedule node may release after the
    underlying TE backward returns.
    """
    if item is None:
        return None
    if isinstance(item, torch.Tensor):
        return item.detach().clone()
    if isinstance(item, (list, tuple)):
        cloned = [_snapshot_for_wgrad(t) for t in item]
        return type(item)(cloned)
    # Quantized tensor bases / other opaque objects: pass through. TE
    # owns their lifetime and clones internally where necessary.
    return item


def _primus_put(self, tensor_list, func):
    """Redirect wgrad closure to Primus pipeline scheduling."""
    args = get_args()

    tensor_list = [_snapshot_for_wgrad(t) for t in tensor_list]

    def wgrad_func():
        func(*tensor_list)

    if args.patch_zero_bubble:
        from primus.backends.megatron.core.pipeline_parallel.zerobubble.zbpp_utils import (
            WeightGradStore as ZBWeightGradStore,
        )

        if ZBWeightGradStore.split_bw():

            def pre_func(async_op=True):
                return tensor_list, [], None

            def process_func(_tensors, _unused, _handle=None):
                func(*_tensors)

            ZBWeightGradStore.put(None, pre_func, process_func)
        else:
            wgrad_func()

    elif args.patch_primus_pipeline:
        from primus.core.pipeline_parallel.handler.wgrad_handler import (
            WGradRunningCache,
        )

        WGradRunningCache.append(wgrad_func)
    else:
        wgrad_func()


def _primus_pop(self):
    """No-op: wgrad execution is driven by Primus scheduler, not TE backward_dw."""
    return (None, None), []


def patch_weight_grad_store():
    """Replace WeightGradStore methods with Primus-aware versions.

    This must be called before model construction so that all TE Linear
    modules created afterwards will use the patched WeightGradStore.
    """
    WeightGradStore.__init__ = _primus_init
    WeightGradStore.put = _primus_put
    WeightGradStore.pop = _primus_pop


def unpatch_weight_grad_store():
    """Restore original WeightGradStore methods."""
    WeightGradStore.__init__ = _original_init
    WeightGradStore.put = _original_put
    WeightGradStore.pop = _original_pop
