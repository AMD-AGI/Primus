###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Incremental gradient norm accumulator for overlapping grad norm with FSDP2
reduce-scatter.

Registers a `register_post_accumulate_grad_hook` on each sharded parameter.
FSDP2 fires these hooks inside the post-reduce stream (RS or AR stream)
after the sharded gradient is assigned. Each hook accumulates ||grad||_2^2
into a shared tensor. By the time the default stream synchronises
(`_wait_for_post_backward`), the total squared norm is already computed.
The optimizer's `clip_grad_norm` then only needs a single all-reduce + sqrt
+ clip — eliminating the per-parameter norm recomputation that would
otherwise serialise after the last reduce-scatter.
"""

from typing import Callable

import torch
import torch.distributed as dist


class IncrementalGradNormAccumulator:
    """Accumulates squared L2 norms of sharded gradients in the RS stream.

    Args:
        shard_process_group: The FSDP shard process group used for the final
            all-reduce of the accumulated squared norm. For HSDP this must be
            the shard-only group (not the full DP group) because replicate-group
            ranks hold identical gradients after the HSDP all-reduce.
        device: CUDA device for the accumulator tensor.
    """

    def __init__(
        self,
        shard_process_group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        self._shard_pg = shard_process_group
        self._total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)

    def reset(self) -> None:
        """Zero the accumulator. Called from optimizer.zero_grad()."""
        self._total_norm_sq.zero_()

    def make_hook(self) -> Callable[[torch.Tensor], None]:
        """Return a closure suitable for register_post_accumulate_grad_hook.

        The hook extracts the local shard data (bypassing DTensor overhead)
        and accumulates its squared L2 norm into ``_total_norm_sq``.
        """
        acc = self

        def hook(param: torch.Tensor) -> None:
            grad = param.grad
            if grad is None:
                return
            local_grad = grad._local_tensor if hasattr(grad, "_local_tensor") else grad
            acc._total_norm_sq.add_(local_grad.float().norm(2).pow_(2))

        return hook

    @torch.no_grad()
    def finalize(
        self,
        clip_grad: float,
        params,
    ) -> torch.Tensor:
        """All-reduce the accumulated norm, sqrt, clip, and return total_norm.

        No additional stream synchronisation is needed here: by the time the
        optimizer calls this method the default stream has already waited on
        ``post_reduce_event`` via ``_wait_for_post_backward``, so the
        accumulated value is visible.

        Args:
            clip_grad: Maximum gradient norm for clipping.
            params: Iterable of parameters whose ``.grad`` will be clipped.

        Returns:
            The global L2 gradient norm (scalar tensor).
        """
        dist.all_reduce(self._total_norm_sq, op=dist.ReduceOp.SUM, group=self._shard_pg)
        total_norm = self._total_norm_sq.sqrt_()
        torch.nn.utils.clip_grads_with_norm_(params, clip_grad, total_norm, foreach=True)
        return total_norm
