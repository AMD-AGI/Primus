###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus MoE All-to-All dispatcher D2H patch.

Patches ``MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize`` so that when
``use_turbo_grouped_gemm`` is enabled, ``tokens_per_expert`` is kept on-device
(PrimusTurbo grouped gemm consumes it on the GPU) instead of being copied to the
host. All other splits are still moved to CPU and the stream sync is unchanged.
"""

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _turbo_grouped_gemm_on_device(ctx: PatchContext) -> bool:
    """Whether PrimusTurbo grouped gemm consumes tokens_per_expert on device.

    The authoritative source is the global Primus/Megatron args, not the
    dispatcher's ``TransformerConfig`` (the turbo flags are CLI/args-level and
    are not guaranteed to be mirrored onto ``self.config``).
    ``use_turbo_grouped_gemm`` is the flag users set (also auto-enabled by
    Sync-Free MoE stage >= 2).
    """
    try:
        args = get_args(ctx)
    except Exception:
        return False
    return bool(getattr(args, "use_turbo_grouped_gemm", False))


@register_patch(
    "megatron.moe_alltoall_dtoh_turbo_grouped_gemm",
    backend="megatron",
    phase="before_train",
    description=(
        "Skip tokens_per_expert D2H copy in MoEAlltoAllTokenDispatcher "
        "when PrimusTurbo grouped gemm (use_turbo_grouped_gemm) is enabled"
    ),
)
def patch_moe_alltoall_dtoh(ctx: PatchContext):
    """Replace ``MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize``."""
    from megatron.core.transformer.moe import token_dispatcher as td_mod

    cls = td_mod.MoEAlltoAllTokenDispatcher

    keep_tokens_per_expert_on_device = _turbo_grouped_gemm_on_device(ctx)

    def _maybe_dtoh_and_synchronize(self, point, tokens_per_expert=None):
        """
        Move all possible GPU tensors to CPU and make a synchronization at the expected point.
        """
        maybe_move_tensor_to_cpu = td_mod.maybe_move_tensor_to_cpu

        if not self.drop_and_pad:
            if point == self.cuda_dtoh_point:
                # Move all possible GPU tensors to CPU at self.cuda_dtoh_point.
                on_side_stream = torch.cuda.current_stream() != self.cuda_dtoh_stream
                if on_side_stream:
                    self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.cuda_dtoh_stream):
                    # TODO: use MemcpyBatchAsync instead.
                    # PrimusTurbo grouped gemm consumes tokens_per_expert on device,
                    # so keep it on the GPU and skip the D2H copy when enabled.
                    if not keep_tokens_per_expert_on_device:
                        tokens_per_expert = maybe_move_tensor_to_cpu(
                            tokens_per_expert, record_stream=on_side_stream
                        )
                    self.input_splits = maybe_move_tensor_to_cpu(
                        self.input_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits = maybe_move_tensor_to_cpu(
                        self.output_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits_tp = maybe_move_tensor_to_cpu(
                        self.output_splits_tp, as_numpy=True, record_stream=on_side_stream
                    )
                    self.num_out_tokens = maybe_move_tensor_to_cpu(
                        self.num_out_tokens, record_stream=on_side_stream
                    )
                    if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                        self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                            self.num_global_tokens_per_local_expert, record_stream=on_side_stream
                        )
                self.d2h_event = self.cuda_dtoh_stream.record_event()

            if point == self.cuda_sync_point:
                # Synchronize with the DtoH stream at self.cuda_sync_point.
                self.d2h_event.synchronize()

        return tokens_per_expert

    cls._maybe_dtoh_and_synchronize = _maybe_dtoh_and_synchronize

    log_rank_0(
        "[Patch:megatron.moe_alltoall_dtoh_turbo_grouped_gemm]   Patched "
        "MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize "
        f"(skip tokens_per_expert D2H = {keep_tokens_per_expert_on_device})"
    )
