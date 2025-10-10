###############################################################################
# Some parts of this code are copied and modified from
# Sea AI Lab's zero-bubble-pipeline-parallelism project
# (https://github.com/sail-sg/zero-bubble-pipeline-parallelism).
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

from typing import List

import amp_C
import torch
from megatron.core import mpu, parallel_state
from megatron.core.optimizer.optimizer import (
    ChainedOptimizer,
    MegatronOptimizer,
    multi_tensor_applier,
)

from primus.modules.module_utils import log_rank_all


class ZeroBubblePPChainedOptimizer(ChainedOptimizer):
    def __init__(self, chained_optimizers: List[MegatronOptimizer]):
        super().__init__(chained_optimizers)

        self.partial_reduced_total_norm = torch.FloatTensor([0])
        self.local_total_norm = None
        self.dummy_overflow_buf = torch.cuda.IntTensor([0])
        self.zero_float_tensor = torch.cuda.FloatTensor([0])
        self.parameters_backup = None
        self.do_prev_step = False
        self.do_this_step = False
        self.send_next_reqs = []
        self.send_prev_reqs = []
        self.grad_norm_no_clip_recorder = 0
        self.post_validation_enabled = False

    def record_grad_norm(self, grad_norm):
        if self.post_validation_enabled:
            return
        if self.config.clip_grad > 0.0:
            if grad_norm is None or grad_norm > self.config.clip_grad:
                self.grad_norm_no_clip_recorder = 0
            else:
                self.grad_norm_no_clip_recorder += 1
            if self.grad_norm_no_clip_recorder >= 10:
                rank = parallel_state.get_pipeline_model_parallel_rank()
                log_rank_all(f"{rank}: enable optimizer post validation")
                self.post_validation_enabled = True
        else:
            if grad_norm is not None:
                # optimizer state update successfully
                rank = parallel_state.get_pipeline_model_parallel_rank()
                log_rank_all(f"{rank}: enable optimizer post validation")
                self.post_validation_enabled = True

    @torch.no_grad()
    def save_parameters_backup(self):
        parameters = self.get_parameters()
        backups = []
        for param in parameters:
            p = param.detach().clone()
            s1 = (
                self.optimizer.state[param]["exp_avg"].detach().clone()
                if "exp_avg" in self.optimizer.state[param]
                else torch.zeros_like(param.data).float()
            )
            s2 = (
                self.optimizer.state[param]["exp_avg_sq"].detach().clone()
                if "exp_avg_sq" in self.optimizer.state[param]
                else torch.zeros_like(param.data).float()
            )
            backups.append((p, s1, s2))
        self.parameters_backup = backups

    @torch.no_grad()
    def rollback_parameters(self):
        parameters = self.get_parameters()
        for param, (backup, s1, s2) in zip(parameters, self.parameters_backup):
            param.copy_(backup)
            self.optimizer.state[param]["exp_avg"] = s1
            self.optimizer.state[param]["exp_avg_sq"] = s2
        self.parameters_backup = None

    def get_mp_group_except_pp_for_bypassing_sync(self):
        """Default returned here, but the distributed optimizer overrides this."""
        # Note: expert parallel are not supported yet
        return mpu.get_tensor_model_parallel_group()

    def calc_local_grad_norm(self):
        grads_for_norm = self.get_main_grads_for_grad_norm()
        return self.do_clac_local_grad_norm(
            grads_for_norm, tensor_parallel_group=self.get_mp_group_except_pp_for_bypassing_sync()
        )

    def get_clip_coeff_and_grad_norm(self, max_norm, norm_type=2):
        _total_norm = self.partial_reduced_total_norm
        if norm_type == torch.inf:
            _total_norm = _total_norm[0].item()
        else:
            _total_norm = _total_norm.item() ** (1.0 / norm_type)
        _clip_coeff = max_norm / (_total_norm + 1.0e-6)
        return _clip_coeff, _total_norm

    def do_clac_local_grad_norm(self, grads_for_norm, norm_type=2, tensor_parallel_group=None):
        if isinstance(grads_for_norm, torch.Tensor):
            grads_for_norm = [grads_for_norm]

        # Norm parameters.
        norm_type = float(norm_type)
        total_norm = 0.0

        # Calculate norm.
        if norm_type == torch.inf:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            # Take max across all model-parallel GPUs.
            torch.distributed.all_reduce(
                total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=tensor_parallel_group
            )
            total_norm = total_norm_cuda
            # total_norm = total_norm_cuda[0].item()

        else:
            if norm_type == 2.0:
                self.dummy_overflow_buf.fill_(0)
                # Use apex's multi-tensor applier for efficiency reasons.
                # Multi-tensor applier takes a function and a list of list
                # and performs the operation on that list all in one kernel.
                if grads_for_norm:
                    grad_norm, _ = multi_tensor_applier(
                        amp_C.multi_tensor_l2norm,
                        self.dummy_overflow_buf,
                        [grads_for_norm],
                        False,  # no per-parameter norm
                    )
                else:
                    self.zero_float_tensor.fill_(0)
                    grad_norm = self.zero_float_tensor
                # Since we will be summing across data parallel groups,
                # we need the pow(norm-type).
                total_norm = grad_norm**norm_type

            else:
                for grad in grads_for_norm:
                    grad_norm = torch.norm(grad, norm_type)
                    total_norm += grad_norm**norm_type

            # Sum across all model-parallel GPUs.
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=tensor_parallel_group
            )
            # total_norm = total_norm.item() ** (1.0 / norm_type)

        self.local_total_norm = total_norm.cpu()
        return total_norm

    def partially_reduce_local_total_norm(self, clip_grad):
        return self.do_partially_reduce_local_total_norm(clip_grad)

    def do_partially_reduce_local_total_norm(self, max_norm, norm_type=2):
        # recv value from prev pipeline stage
        # self.partial_reduced_total_norm = self.recv_one(self.partial_reduced_total_norm)
        prev_clip_coeff, prev_grad_norm = self.get_clip_coeff_and_grad_norm(max_norm, norm_type)

        # reduce
        if norm_type == torch.inf:
            self.partial_reduced_total_norm = torch.maximum(
                self.partial_reduced_total_norm, self.local_total_norm
            )
        else:
            self.partial_reduced_total_norm = self.partial_reduced_total_norm + self.local_total_norm

        this_clip_coeff, grad_norm = self.get_clip_coeff_and_grad_norm(max_norm, norm_type)
        # rank = parallel_state.get_pipeline_model_parallel_rank()
        return prev_clip_coeff, this_clip_coeff, grad_norm

    def downscale_gradient(self, clip_coeff):
        assert clip_coeff < 1.0
        parameters = self.get_parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        # Grads.
        grads = []
        for param in parameters:
            if param.grad is not None:
                assert param.grad.type() == "torch.cuda.FloatTensor"
                grads.append(param.grad.detach())
        self.dummy_overflow_buf.fill_(0)
        multi_tensor_applier(amp_C.multi_tensor_scale, self.dummy_overflow_buf, [grads, grads], clip_coeff)

    def get_reduced_global_states(self):
        return [self.partial_reduced_total_norm]

    def send_all(self, to_next=True):
        need_send = False
        dst = None
        if to_next and not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            need_send = True
            dst = parallel_state.get_pipeline_model_parallel_next_rank()
        if not to_next and not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            need_send = True
            dst = parallel_state.get_pipeline_model_parallel_prev_rank()
        if need_send:
            for global_state in self.get_reduced_global_states():
                send_req = torch.distributed.isend(
                    tensor=global_state,
                    dst=dst,
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                if to_next:
                    self.send_next_reqs.append(send_req)
                else:
                    self.send_prev_reqs.append(send_req)

    def recv_all(self, from_prev=True, init_values=None):
        if from_prev:
            for req in self.send_prev_reqs:
                req.wait()
            self.send_prev_reqs = []
        else:
            for req in self.send_next_reqs:
                req.wait()
            self.send_next_reqs = []
        all_global_states = self.get_reduced_global_states()
        if init_values is None:
            init_values = [0.0] * len(all_global_states)
        for global_state, init_value in zip(all_global_states, init_values):
            self.recv_one(global_state, from_prev=from_prev, init_value=init_value)

    def recv_one(self, global_state, from_prev=True, init_value=0.0):
        if from_prev:
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                global_state.fill_(init_value)
            else:
                req = torch.distributed.irecv(
                    tensor=global_state,
                    src=parallel_state.get_pipeline_model_parallel_prev_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                req.wait()
        else:
            if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                req = torch.distributed.irecv(
                    tensor=global_state,
                    src=parallel_state.get_pipeline_model_parallel_next_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                req.wait()
        return global_state
