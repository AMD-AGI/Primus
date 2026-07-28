###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
FP32 optimizer for FSDP2 mixed precision training.

Uses the TorchTitan approach: model parameters are initialized in FP32,
FSDP2's MixedPrecisionPolicy casts to BF16 for forward/backward, and
the optimizer operates on FP32 parameters with FP32 states.

This eliminates the "stale weights" problem of BF16 optimizer states
while avoiding the master-copy duplication of Float16OptimizerWithFloat16Params.

Memory impact vs BF16 optimizer (Flux 12B, 8 GPUs):
    +3 GB/GPU for FP32 parameters (vs BF16)
    +6 GB/GPU for FP32 optimizer states (vs BF16)
    = +9 GB/GPU total

Gradient clipping uses PyTorch-native DTensor-aware APIs matching TorchTitan.
"""

from typing import TYPE_CHECKING, Callable, List, Optional

import torch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    optim_state_to_sharding_state,
)
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig

from primus.core.utils.module_utils import log_rank_0

if TYPE_CHECKING:
    from megatron.core.process_groups_config import ProcessGroupCollection


def _safe_log_rank_0(msg: str):
    try:
        log_rank_0(msg)
    except (AttributeError, TypeError):
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(msg)


class FSDP2FP32Optimizer(MegatronOptimizer):
    """FP32 optimizer for FSDP2 mixed precision training.

    Extends MegatronOptimizer directly (not MixedPrecisionOptimizer).
    Modeled on Megatron's own FP32Optimizer with two key differences:

    1. prepare_grads is a no-op: FSDP2 writes gradients directly to param.grad
       (no main_grad -> grad copy needed).
    2. clip_grad_norm uses TorchTitan-style DTensor-native APIs:
       torch.nn.utils.get_total_norm + clip_grads_with_norm_ for correct
       norm computation across FSDP2's sharded DTensor parameters.

    Args:
        optimizer: Base PyTorch optimizer (e.g., AdamW with fused=True).
        config: OptimizerConfig from Megatron.
        init_state_fn: Function to initialize optimizer state tensors.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        init_state_fn: Callable,
    ):
        super().__init__(optimizer, config, init_state_fn)
        self._scale = torch.tensor([1.0], dtype=torch.float, device="cuda")
        self.is_stub_optimizer = optimizer is None

        from megatron.training import get_args

        args = get_args()
        self._grad_norm_accumulator = getattr(args, "_grad_norm_accumulator", None)

    def zero_grad(self, set_to_none=True):
        if self.is_stub_optimizer:
            return
        if self._grad_norm_accumulator is not None:
            self._grad_norm_accumulator.reset()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_loss_scale(self):
        return self._scale

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """No-op: FSDP2 writes gradients directly to param.grad."""
        return False

    @torch.no_grad()
    def clip_grad_norm(self, clip_grad: float) -> float | torch.Tensor:
        """DTensor-native gradient clipping matching TorchTitan.

        Uses torch.nn.utils.get_total_norm which natively handles DTensor
        gradients (returns a DTensor with _NormPartial placement that is
        reduced via full_tensor()), then clips with foreach-optimized
        clip_grads_with_norm_.

        When overlap_grad_norm is enabled, the squared norms have already been
        accumulated in the RS stream via post_accumulate_grad_hooks. Only a
        single all-reduce + sqrt + clip is needed.
        """
        params = self.get_parameters()

        if self._grad_norm_accumulator is not None:
            return self._grad_norm_accumulator.finalize(clip_grad, params)

        from torch.distributed.tensor import DTensor

        grads = [p.grad for p in params if p.grad is not None]

        if not grads:
            return 0.0

        total_norm = torch.nn.utils.get_total_norm(grads, norm_type=2.0, foreach=True)
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()
        torch.nn.utils.clip_grads_with_norm_(params, clip_grad, total_norm, foreach=True)
        return total_norm

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        if self.is_stub_optimizer:
            return True
        timers = self.config.timers

        if timers is not None:
            timers("optimizer-inner-step", log_level=1).start(barrier=self.config.barrier_with_L1_time)
        self.optimizer.step()
        if timers is not None:
            timers("optimizer-inner-step").stop()

        return True

    @torch.no_grad()
    def step(self):
        """Clip gradients and step. Always succeeds (no overflow for FP32)."""
        timers = self.config.timers

        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        if timers is not None:
            timers("optimizer-clip-main-grad", log_level=1).start(barrier=self.config.barrier_with_L1_time)
        grad_norm = None
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        if timers is not None:
            timers("optimizer-clip-main-grad").stop()

        if timers is not None:
            timers("optimizer-count-zeros", log_level=1).start(barrier=self.config.barrier_with_L1_time)
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        if timers is not None:
            timers("optimizer-count-zeros").stop()

        success = self.step_with_ready_grads()

        return success, grad_norm, num_zeros_in_grad

    def reload_model_params(self, state_dict=None):
        pass

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if "common_step" in state_dict.get("state", {}):
            common_step = state_dict["state"].pop("common_step")
            self._restore_common_per_param_step(state_dict, common_step)

        state_dict["param_groups"] = self._filter_and_reorder_param_groups(
            self.optimizer.param_groups, state_dict["param_groups"]
        )
        self.optimizer.load_state_dict(state_dict)

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()
        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, self.get_parameters()
        )
        step = self._extract_common_per_param_step(state_dict)

        optim_state_to_sharding_state(state_dict, id_to_sharded_param_map, exclude_keys="step")
        if step:
            state_dict["state"]["common_step"] = step
        return state_dict

    def finalize_dist_ckpt_load(self, iteration):
        """Restore optimizer step counter after dist_checkpointing in-place load.

        When skip_load_to_model_and_opt=True (FSDP2), load_state_dict is not
        called, so common_step is never fanned out to per-parameter step
        entries.  This fills step from the training iteration.
        """
        step_val = float(iteration)
        for p in self.get_parameters():
            if p in self.optimizer.state and "step" in self.optimizer.state[p]:
                self.optimizer.state[p]["step"].fill_(step_val)


def get_fsdp2_fp32_optimizer(
    config: OptimizerConfig,
    model_chunks: List[torch.nn.Module],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
    default_skip_embedding_weight_decay: bool = False,
    pg_collection: Optional["ProcessGroupCollection"] = None,
    base_optimizer_cls=torch.optim.AdamW,
    use_foreach: bool = False,
    **optimizer_kwargs,
) -> FSDP2FP32Optimizer:
    """Factory function to create FSDP2 FP32 param optimizer from model chunks.

    Collects trainable FP32 parameters, builds param groups with weight decay
    and LR scaling, creates AdamW (fused or foreach), and wraps in
    FSDP2FP32Optimizer.

    Args:
        config: OptimizerConfig from Megatron.
        model_chunks: List of model modules (FSDP2-wrapped).
        no_weight_decay_cond: Optional predicate for zero weight decay.
        scale_lr_cond: Optional predicate for scaled learning rate.
        lr_mult: Learning rate multiplier for scaled params.
        use_gloo_process_groups: Unused (kept for API compatibility).
        default_skip_embedding_weight_decay: Skip weight decay for embeddings
            if no_weight_decay_cond not provided.
        pg_collection: Unused (kept for API compatibility).
        base_optimizer_cls: PyTorch optimizer class (default: AdamW).
        use_foreach: If True, use foreach mode; if False (default), use fused.
        **optimizer_kwargs: Additional kwargs for base optimizer.

    Returns:
        FSDP2FP32Optimizer instance.
    """
    all_params = []
    for model_chunk in model_chunks:
        for param in model_chunk.parameters():
            if param.requires_grad:
                all_params.append(param)

    if not all_params:
        raise ValueError("No trainable parameters found in model chunks!")

    weight_decay = config.weight_decay
    lr = config.lr
    param_groups = []

    if default_skip_embedding_weight_decay and no_weight_decay_cond is None:
        embedding_params = []
        non_embedding_params = []
        for param in all_params:
            is_embedding = False
            for model_chunk in model_chunks:
                for name, p in model_chunk.named_parameters():
                    if p is param and "embed" in name.lower():
                        is_embedding = True
                        break
                if is_embedding:
                    break
            if is_embedding:
                embedding_params.append(param)
            else:
                non_embedding_params.append(param)

        if embedding_params:
            param_groups.append({"params": embedding_params, "weight_decay": 0.0, "lr": lr})
        if non_embedding_params:
            param_groups.append({"params": non_embedding_params, "weight_decay": weight_decay, "lr": lr})
    elif no_weight_decay_cond is not None:
        no_wd_params = []
        wd_params = []
        for param in all_params:
            if no_weight_decay_cond(param):
                no_wd_params.append(param)
            else:
                wd_params.append(param)
        if no_wd_params:
            param_groups.append({"params": no_wd_params, "weight_decay": 0.0, "lr": lr})
        if wd_params:
            param_groups.append({"params": wd_params, "weight_decay": weight_decay, "lr": lr})
    else:
        param_groups.append({"params": all_params, "weight_decay": weight_decay, "lr": lr})

    if scale_lr_cond is not None and lr_mult != 1.0:
        new_groups = []
        for param_group in param_groups:
            scaled = [p for p in param_group["params"] if scale_lr_cond(p)]
            normal = [p for p in param_group["params"] if not scale_lr_cond(p)]
            if normal:
                new_groups.append({**param_group, "params": normal})
            if scaled:
                new_groups.append({**param_group, "params": scaled, "lr": param_group["lr"] * lr_mult})
        param_groups = new_groups

    _safe_log_rank_0(
        f"Creating FSDP2 FP32 optimizer with {len(all_params):,} parameters "
        f"in {len(param_groups)} param groups"
    )

    base_optimizer = base_optimizer_cls(
        param_groups,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        fused=not use_foreach,
        foreach=use_foreach,
        **optimizer_kwargs,
    )

    for param_group in base_optimizer.param_groups:
        param_group.setdefault("wd_mult", 1.0)
        param_group.setdefault("lr_mult", 1.0)
        param_group.setdefault("is_expert_parallel", False)
        param_group.setdefault("is_decoupled_lr", False)
        param_group.setdefault("default_config", True)

    def init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                if len(opt.state[p]) == 0:
                    opt.state[p]["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    opt.state[p]["exp_avg"] = torch.zeros_like(p.data)
                    opt.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)

    optimizer = FSDP2FP32Optimizer(
        optimizer=base_optimizer,
        config=config,
        init_state_fn=init_state_fn,
    )

    fp32_count = sum(1 for p in all_params if p.dtype == torch.float32)
    bf16_count = sum(1 for p in all_params if p.dtype == torch.bfloat16)
    other_count = len(all_params) - fp32_count - bf16_count

    _safe_log_rank_0("=" * 80)
    _safe_log_rank_0("[FSDP2FP32ParamOptimizer Initialized]")
    _safe_log_rank_0("  TorchTitan-style: FP32 params + FSDP2 MixedPrecisionPolicy")
    _safe_log_rank_0("  DTensor-native gradient clipping")
    _safe_log_rank_0(f"  AdamW mode: {'foreach' if use_foreach else 'fused'}")
    _safe_log_rank_0(f"  FP32 parameters: {fp32_count:,}")
    _safe_log_rank_0(f"  BF16 parameters: {bf16_count:,}")
    if other_count > 0:
        _safe_log_rank_0(f"  Other dtype parameters: {other_count:,}")
    _safe_log_rank_0(f"  Total trainable parameters: {len(all_params):,}")
    _safe_log_rank_0("=" * 80)

    return optimizer
