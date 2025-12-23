###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerModelChunkSchedulePlan as TransformerModelChunkSchedulePlanBase,
)
from megatron.core.pipeline_parallel.utils import get_comm_stream

from primus.backends.megatron.core.pipeline_parallel.zerobubble.zbpp_utils import (
    WeightGradStore,
)


def pop_weight_grad(num=None):
    """Pop the weight gradient from the weight gradient store.

    Args:
        num (int): The number of weight gradients to pop. If None, pop all.
    """
    if WeightGradStore.split_bw():
        WeightGradStore.flush(num=num)
        while WeightGradStore.queue_size() > 0:
            WeightGradStore.pop()


def execute_overlapped_1f1b(f_layer, b_layer, f_input=None, b_grad=None, is_last_layer_in_bwd=False):
    """Schedule one-forward-one-backward operations for a single transformer layer.

    This function interleaves forward and backward operations, overlapping the communications
    (dispatch or combine) of one with the computations (att or mlp) of the other
    to maximize parallelism and efficiency.

    When f_layer and b_layer are not None, forward and backward pass are overlapped as follows:
    comm_stream: combine_bwd            | dispatch_fwd->dispatch_bwd  | combine_fwd
    comp_stream: attn_fwd->post_attn_fwd| mlp_bwd->mlp_bwd_dw->mlp_fwd| post_attn_bwd->attn_bwd
    For MTP, mtp_post_process_fwd is executed after the combine_fwd in the comp_stream,
    and mtp_post_process_bwd is executed before the combine_bwd in the comp_stream.

    Args:
        f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
        b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
        f_input (Tensor): Input for forward computation
        b_grad (Tensor): Gradient for backward computation
        is_last_layer_in_bwd (bool):
            Whether the current layer is the last layer in the backward pass.

    Returns:
        Functions or values for next iteration's computation
    """

    if b_layer is not None:
        b_grad = b_layer.mtp_post_process.backward(b_grad)
        b_grad = b_layer.moe_combine.backward(b_grad)

    if f_layer is not None:
        with f_layer.get_fp8_context():
            f_input = f_layer.attn.forward(f_input)
            f_input = f_layer.post_attn.forward(f_input)

    if b_layer is not None:
        b_grad = b_layer.mlp.backward(b_grad)

    if f_layer is not None:
        with f_layer.get_fp8_context():
            f_input = f_layer.moe_dispatch.forward(f_input)

    if b_layer is not None:
        pop_weight_grad(num=1)
        b_grad = b_layer.moe_dispatch.backward(b_grad)

    if f_layer is not None:
        with f_layer.get_fp8_context():
            f_input = f_layer.mlp.forward(f_input)

    if f_layer is not None:
        with f_layer.get_fp8_context():
            f_input = f_layer.moe_combine.forward(f_input)
            f_input = f_layer.mtp_post_process.forward(f_input)

    if b_layer is not None:
        b_grad = b_layer.post_attn.backward(b_grad)
        b_grad = b_layer.attn.backward(b_grad)

    # Delay the last attn_dw in backward pass (attn_dw of the first layer)
    # for overlapping with the p2p comm
    if b_layer is not None and not is_last_layer_in_bwd:
        pop_weight_grad(num=1)

    return f_input, b_grad


class TransformerModelChunkSchedulePlan(TransformerModelChunkSchedulePlanBase):

    @staticmethod
    def run(
        f_schedule_plan,
        b_schedule_plan,
        b_grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """Model Chunk level 1f1b fine-grained scheduler.

        This function schedules the forward and backward passes for a model chunk,
        which interleaves forward and backward function of multiple Transformer layers
        within a model chunk, and this is needed to overlap the submodules between the individual
        forward and backward functions.

        Assume there are 4 layers in the given model chunk:
        Phase 0: p2p_comm_sync -> forward_preprocess -> p2p_comm_sync -> backward_postprocess
        Phase 1: forward_layer[0] + backward_layer[3], overlapped execution by schedule_layer_1f1b
        Phase 2: forward_layer[1] + backward_layer[2], overlapped execution by schedule_layer_1f1b
        Phase 3: forward_layer[2] + backward_layer[1], overlapped execution by schedule_layer_1f1b
        Phase 4: forward_layer[3] + backward_layer[0], overlapped execution by schedule_layer_1f1b
        Phase 5: send_forward_recv_backward -> send_backward_recv_forward
        Phase 6: backward_dw of the first layer -> forward_postprocess -> backward_preprocess

        Args:
            f_schedule_plan (TransformerModelChunkSchedulePlan): The forward schedule plan
            b_schedule_plan (TransformerModelChunkSchedulePlan): The backward schedule plan
            b_grad (Tensor or None): The gradient of the loss function
            pre_forward (callable or None): The function to call before the forward pass
            pre_backward (callable or None): The function to call before the backward pass
            post_forward (callable or None): The function to call after the forward pass
            post_backward (callable or None): The function to call after the backward pass
        Returns:
            The output of the forward pass.
        """
        f_input = None
        if f_schedule_plan:
            # pp output send/receive sync
            if pre_forward is not None:
                pre_forward(f_schedule_plan.vp_stage)
            f_schedule_plan.record_current_stream()
            f_input = f_schedule_plan.pre_process.forward()

        if b_schedule_plan:
            b_schedule_plan.record_current_stream()
            assert b_grad is not None
            if pre_backward is not None:
                pre_backward(b_schedule_plan.vp_stage)
                b_schedule_plan.record_current_stream()

            if b_schedule_plan.post_process is not None:
                b_grad = b_schedule_plan.post_process.backward(b_grad)

        f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
        b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
        overlapped_layers = min(f_num_layers, b_num_layers)

        # combined forward and backward pass for overlapped layers
        for i in range(overlapped_layers):
            f_layer = f_schedule_plan.get_layer(i)
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
            f_input, b_grad = execute_overlapped_1f1b(
                f_layer,
                b_layer,
                f_input=f_input,
                b_grad=b_grad,
                is_last_layer_in_bwd=(i == b_num_layers - 1),
            )
            torch.cuda.nvtx.range_pop()

        # backward pass for the remaining layers
        for i in range(overlapped_layers, b_num_layers):
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
            _, b_grad = execute_overlapped_1f1b(
                None, b_layer, b_grad=b_grad, is_last_layer_in_bwd=(i == b_num_layers - 1)
            )
            torch.cuda.nvtx.range_pop()

        # forward pass for the remaining layers
        for i in range(overlapped_layers, f_num_layers):
            f_layer = f_schedule_plan.get_layer(i)
            torch.cuda.nvtx.range_push(f"layer_{i}f")
            f_input, _ = execute_overlapped_1f1b(f_layer, None, f_input=f_input)
            torch.cuda.nvtx.range_pop()

        if f_schedule_plan is not None and post_forward is not None:
            # post_forward()/send_forward_recv_forward() is running in the communication stream,
            # so the p2p comm could be overlapped with the attn backward
            with torch.cuda.stream(get_comm_stream()):
                f_schedule_plan.wait_current_stream()
                post_forward(f_input, f_schedule_plan.vp_stage)

        # post_backward()/send_backward_recv_backward() is running in the computation stream,
        # so the p2p comm could be overlapped with the wgrad of attn backward
        if b_schedule_plan is not None and post_backward is not None:
            b_schedule_plan.wait_current_stream()
            post_backward(b_grad, b_schedule_plan.vp_stage)

        # Delay the dw in backward pass for overlapping with the p2p comm
        if b_num_layers > 0:
            # Pop all remaining weight gradients
            pop_weight_grad(num=None)

        # post process forward
        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            f_input = f_schedule_plan.post_process.forward(f_input)
        # pre process backward
        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(b_grad)

        if f_schedule_plan:
            f_schedule_plan.wait_current_stream()
        if b_schedule_plan:
            b_schedule_plan.wait_current_stream()

        # Release reference as early as possible, this helps avoid memory leak.
        if b_schedule_plan is not None:
            b_schedule_plan.release_state()

        return f_input
