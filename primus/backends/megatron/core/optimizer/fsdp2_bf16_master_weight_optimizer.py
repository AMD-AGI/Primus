###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
BF16 Master Weight optimizer for FSDP2 mixed precision training.

Model parameters are stored natively in BF16 (no FP32-to-BF16 cast during
all-gather), while FP32 master weight copies are maintained internally for
optimizer precision. This follows Megatron's Float16OptimizerWithFloat16Params
pattern adapted for FSDP2 DTensor parameters.

Compared to the FP32 optimizer (fsdp2_fp32_optimizer.py), this eliminates
the per-layer CopyFunctor<BFloat16, float> kernel in every forward all-gather,
at the cost of +2 bytes/param for the FP32 master copy.

Memory per parameter per GPU (sharded):
    FP32 optimizer: 4 (FP32 param) + 4 (exp_avg) + 4 (exp_avg_sq) = 12 bytes
    This optimizer: 2 (BF16 param) + 4 (FP32 master) + 4 + 4     = 14 bytes
"""

from itertools import chain
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
    optim_state_to_sharding_state,
)
from megatron.core.optimizer.optimizer import MegatronOptimizer, _zero_grad_group_helper
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


class FSDP2BF16MasterWeightOptimizer(MegatronOptimizer):
    """BF16 optimizer with FP32 master weights for FSDP2.

    Model parameters live in BF16 (eliminating the FP32->BF16 cast in every
    forward all-gather). FP32 master copies are maintained for the optimizer
    step, matching Megatron's Float16OptimizerWithFloat16Params pattern.

    Three parameter groups are tracked:
        bf16_groups: original BF16 model parameters (what FSDP2 shards/all-gathers)
        fp32_from_bf16_groups: FP32 master copies (what the optimizer steps on)
        fp32_from_fp32_groups: natively FP32 params (e.g. LayerNorm), no master copy

    Args:
        optimizer: Base PyTorch optimizer (param_groups already point to FP32 masters).
        config: OptimizerConfig from Megatron.
        init_state_fn: Function to initialize optimizer state tensors.
        bf16_groups: List of lists of original BF16 model parameters.
        fp32_from_bf16_groups: List of lists of FP32 master weight copies.
        fp32_from_fp32_groups: List of lists of natively FP32 parameters.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        init_state_fn: Callable,
        bf16_groups: List[List[torch.nn.Parameter]],
        fp32_from_bf16_groups: List[List[torch.Tensor]],
        fp32_from_fp32_groups: List[List[torch.nn.Parameter]],
        use_foreach: bool = True,
    ):
        super().__init__(optimizer, config, init_state_fn)
        self._scale = torch.tensor([1.0], dtype=torch.float, device="cuda")
        self.is_stub_optimizer = optimizer is None

        self.bf16_groups = bf16_groups
        self.fp32_from_bf16_groups = fp32_from_bf16_groups
        self.fp32_from_fp32_groups = fp32_from_fp32_groups

        self._use_foreach = use_foreach
        self._foreach_ready = False

        from megatron.training import get_args

        args = get_args()
        self._grad_norm_accumulator = getattr(args, "_grad_norm_accumulator", None)
        self._validate_count = 0

    @torch.compiler.disable
    @torch.no_grad()
    def warmup_foreach_cache(self):
        """Eagerly initialize the foreach cache before torch.compile wraps step().

        Creates temporary zero gradients to satisfy _init_foreach_cache's
        requirement that grads exist (including DTensor placements for the
        assertion at line ~164), then clears them.
        Must be called after FSDP wrapping but before torch.compile wraps
        optimizer.step.
        """
        if self._foreach_ready or not self._use_foreach:
            return

        from torch.distributed.tensor import DTensor

        for group in self.bf16_groups:
            for p in group:
                if p.grad is not None:
                    continue
                if isinstance(p.data, DTensor):
                    local = torch.zeros(
                        p.data.to_local().shape,
                        dtype=p.dtype,
                        device=p.device,
                    )
                    p.grad = DTensor.from_local(
                        local,
                        device_mesh=p.data.device_mesh,
                        placements=p.data.placements,
                    )
                else:
                    p.grad = torch.zeros(p.shape, dtype=p.dtype, device=p.device)
        self._init_foreach_cache()
        for group in self.bf16_groups:
            for p in group:
                p.grad = None

    @torch.compiler.disable
    @torch.no_grad()
    def _init_foreach_cache(self):
        """Build cached flat tensor lists for _foreach_copy_ batched operations.

        Called lazily on first prepare_grads() invocation (grads must exist for
        DTensor spec extraction). Uses public DTensor APIs only.
        """
        from torch.distributed.tensor import DTensor

        try:
            from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
                WeightWithFP8AllGatherTensor,
            )

            self._has_fp8_subclass = True
        except ImportError:
            self._has_fp8_subclass = False

        bf16_flat = [p for group in self.bf16_groups for p in group]
        fp32_flat = [p for group in self.fp32_from_bf16_groups for p in group]
        self._bf16_flat = bf16_flat
        self._fp32_flat = fp32_flat

        bf16_inners = []
        for p in bf16_flat:
            t = p.data
            if isinstance(t, DTensor):
                t = t.to_local()
            if self._has_fp8_subclass and isinstance(t, WeightWithFP8AllGatherTensor):
                t = t.inner_data()
            bf16_inners.append(t)
        self._bf16_inners = bf16_inners

        fp32_locals = []
        for m in fp32_flat:
            t = m
            if isinstance(t, DTensor):
                t = t.to_local()
            fp32_locals.append(t)
        self._fp32_locals = fp32_locals

        fp32_grad_locals = [torch.empty_like(fl) for fl in fp32_locals]
        self._fp32_grad_locals = fp32_grad_locals

        fp32_grad_dtensors = []
        for m, gl, bp in zip(fp32_flat, fp32_grad_locals, bf16_flat):
            if isinstance(m, DTensor):
                if m.placements != bp.grad.placements:
                    raise RuntimeError(
                        f"FP32 master placements {m.placements} != "
                        f"BF16 grad placements {bp.grad.placements}"
                    )
                dg = DTensor.from_local(gl, device_mesh=m.device_mesh, placements=m.placements)
            else:
                dg = gl
            fp32_grad_dtensors.append(dg)
            m.grad = dg
        self._fp32_grad_dtensors = fp32_grad_dtensors

        self._foreach_ready = True
        n_params = len(bf16_flat)
        _safe_log_rank_0(
            f"[FSDP2BF16MasterWeightOptimizer] foreach cache initialized: "
            f"{n_params} params, fp8_subclass={self._has_fp8_subclass}"
        )

    def zero_grad(self, set_to_none=True):
        if self.is_stub_optimizer:
            return
        if self._grad_norm_accumulator is not None:
            self._grad_norm_accumulator.reset()
        for group in self.bf16_groups:
            _zero_grad_group_helper(group, set_to_none)
        if not self._foreach_ready:
            for group in self.fp32_from_bf16_groups:
                _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

    def get_loss_scale(self):
        return self._scale

    @torch.compiler.disable
    @torch.no_grad()
    def _prepare_grads_fallback(self) -> bool:
        """Original per-parameter gradient copy (fallback path)."""
        for bf16_group, fp32_group in zip(self.bf16_groups, self.fp32_from_bf16_groups):
            for bf16_param, fp32_master in zip(bf16_group, fp32_group):
                if bf16_param.grad is not None:
                    fp32_master.grad = bf16_param.grad.float()
                    bf16_param.grad = None
        return False

    @torch.compiler.disable
    def _validate_foreach_cache(self):
        """Debug assertion: verify cached tensor refs haven't gone stale."""
        from torch.distributed.tensor import DTensor

        if self._has_fp8_subclass:
            from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
                WeightWithFP8AllGatherTensor,
            )
        for i, p in enumerate(self._bf16_flat):
            live = p.data
            if isinstance(live, DTensor):
                live = live.to_local()
            if self._has_fp8_subclass and isinstance(live, WeightWithFP8AllGatherTensor):
                live = live.inner_data()
            if live.data_ptr() != self._bf16_inners[i].data_ptr():
                raise RuntimeError(f"Cached tensor ref stale for param {i}")
        self._validate_count += 1

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Copy BF16 gradients to FP32 master grads.

        FSDP2 writes gradients to the BF16 model parameter's .grad attribute
        (after FP32 reduce-scatter followed by cast to orig_dtype). We cast
        them to FP32 for the optimizer step on master weights.

        Falls back to per-parameter loop if foreach cache is not ready or
        any grad is missing.
        """
        if not self._foreach_ready:
            any_grad = any(p.grad is not None for group in self.bf16_groups for p in group)
            if any_grad and self._use_foreach:
                self._init_foreach_cache()
            if not self._foreach_ready:
                self._prepare_grads_fallback()
                for fp32_group in self.fp32_from_fp32_groups:
                    for fp32_param in fp32_group:
                        if hasattr(fp32_param, "main_grad"):
                            fp32_param.grad = fp32_param.main_grad
                return False

        from torch.distributed.tensor import DTensor

        bf16_grad_locals = []
        for p in self._bf16_flat:
            if p.grad is None:
                self._prepare_grads_fallback()
                for fp32_group in self.fp32_from_fp32_groups:
                    for fp32_param in fp32_group:
                        if hasattr(fp32_param, "main_grad"):
                            fp32_param.grad = fp32_param.main_grad
                return False
            g = p.grad
            if isinstance(g, DTensor):
                g = g.to_local()
            bf16_grad_locals.append(g)

        if self._validate_count < 3:
            self._validate_foreach_cache()

        torch._foreach_copy_(self._fp32_grad_locals, bf16_grad_locals)

        for p in self._bf16_flat:
            p.grad = None

        for fp32_group in self.fp32_from_fp32_groups:
            for fp32_param in fp32_group:
                if hasattr(fp32_param, "main_grad"):
                    fp32_param.grad = fp32_param.main_grad

        return False

    @torch.no_grad()
    def clip_grad_norm(self, clip_grad: float) -> float | torch.Tensor:
        """DTensor-native gradient clipping matching TorchTitan.

        Operates on FP32 master grads (from prepare_grads) plus any natively
        FP32 parameter grads.

        When overlap_grad_norm is enabled, the squared norms have already been
        accumulated in the RS stream via post_accumulate_grad_hooks on the
        BF16 model params. The pre-computed norm is correct because
        prepare_grads copies BF16->FP32 without scaling. The clipping is
        applied to the FP32 master params.
        """
        all_params = list(chain.from_iterable(self.fp32_from_bf16_groups))
        all_params.extend(chain.from_iterable(self.fp32_from_fp32_groups))

        if self._grad_norm_accumulator is not None:
            return self._grad_norm_accumulator.finalize(clip_grad, all_params)

        from torch.distributed.tensor import DTensor

        all_grads = []
        for fp32_group in self.fp32_from_bf16_groups:
            for p in fp32_group:
                if p.grad is not None:
                    all_grads.append(p.grad)
        for fp32_group in self.fp32_from_fp32_groups:
            for p in fp32_group:
                if p.grad is not None:
                    all_grads.append(p.grad)

        if not all_grads:
            return 0.0

        total_norm = torch.nn.utils.get_total_norm(all_grads, norm_type=2.0, foreach=True)
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(all_params, clip_grad, total_norm, foreach=True)
        return total_norm

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer on FP32 masters, then copy back to BF16 params."""
        if self.is_stub_optimizer:
            return True
        timers = self.config.timers

        if timers is not None:
            timers("optimizer-inner-step", log_level=1).start(barrier=self.config.barrier_with_L1_time)
        self.optimizer.step()
        if timers is not None:
            timers("optimizer-inner-step").stop()

        if timers is not None:
            timers("optimizer-copy-main-to-model", log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self._copy_main_params_to_model_params()
        if timers is not None:
            timers("optimizer-copy-main-to-model").stop()

        return True

    def _copy_main_params_to_model_params(self):
        """Copy FP32 master weights back to BF16 model parameters."""
        if self._foreach_ready:
            self._copy_main_to_model_foreach()
        else:
            self._copy_main_to_model_fallback()

    def _copy_main_to_model_foreach(self):
        torch._foreach_copy_(self._bf16_inners, self._fp32_locals)

    @torch.compiler.disable
    def _copy_main_to_model_fallback(self):
        for bf16_group, fp32_group in zip(self.bf16_groups, self.fp32_from_bf16_groups):
            for bf16_param, fp32_master in zip(bf16_group, fp32_group):
                bf16_param.data.copy_(fp32_master.data)

    @torch.compiler.disable
    def _copy_model_params_to_main_params(self):
        """Copy BF16 model parameters to FP32 masters (for checkpoint reload)."""
        if self._foreach_ready:
            torch._foreach_copy_(self._fp32_locals, self._bf16_inners)
        else:
            self._copy_model_to_main_fallback()

    @torch.compiler.disable
    def _copy_model_to_main_fallback(self):
        for bf16_group, fp32_group in zip(self.bf16_groups, self.fp32_from_bf16_groups):
            for bf16_param, fp32_master in zip(bf16_group, fp32_group):
                fp32_master.data.copy_(bf16_param.data)

    @torch.no_grad()
    def step(self):
        """Clip gradients and step. Always succeeds (no overflow for BF16)."""
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
        """After loading a checkpoint, copy BF16 model params to FP32 masters."""
        self._copy_model_params_to_main_params()

    def state_dict(self, is_loading: bool = False):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = {}
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["fp32_from_fp16_params"] = self.fp32_from_bf16_groups
        return state_dict

    def load_state_dict(self, state_dict):
        optimizer_key = "optimizer"
        if optimizer_key not in state_dict:
            optimizer_key = "optimizer_state_dict"

        if "common_step" in state_dict[optimizer_key].get("state", {}):
            common_step = state_dict[optimizer_key]["state"].pop("common_step")
            self._restore_common_per_param_step(state_dict[optimizer_key], common_step)

        state_dict[optimizer_key]["param_groups"] = self._filter_and_reorder_param_groups(
            self.optimizer.param_groups, state_dict[optimizer_key]["param_groups"]
        )
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Restore FP32 master weights
        if "fp32_from_fp16_params" in state_dict:
            for current_group, saved_group in zip(
                self.fp32_from_bf16_groups, state_dict["fp32_from_fp16_params"]
            ):
                for current_param, saved_param in zip(current_group, saved_group):
                    current_param.data.copy_(saved_param.data)

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()

        # Key on BF16 model params (not FP32 masters) for checkpoint sharding
        # alignment with the model's sharded state dict
        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict,
            chain.from_iterable(g for g in self.bf16_groups),
        )

        # Convert fp32_from_fp16_params to sharded tensors
        if len(state_dict["fp32_from_fp16_params"]) != len(state_dict["optimizer"]["param_groups"]):
            raise ValueError(
                "state_dict fp32_from_fp16_params length does not match optimizer param_groups length"
            )
        state_dict["fp32_from_fp16_params"] = [
            [
                make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id],
                    fp32_param,
                    prefix="optimizer.state.fp32_param",
                )
                for param_id, fp32_param in zip(state_group["params"], fp32_group)
            ]
            for fp32_group, state_group in zip(
                state_dict["fp32_from_fp16_params"],
                state_dict["optimizer"]["param_groups"],
            )
        ]

        step = self._extract_common_per_param_step(state_dict["optimizer"])

        optim_state_to_sharding_state(state_dict["optimizer"], id_to_sharded_param_map, exclude_keys="step")
        if step:
            state_dict["optimizer"]["state"]["common_step"] = step
        return state_dict

    def finalize_dist_ckpt_load(self, iteration):
        """Restore optimizer step counter and sync FP32 masters to BF16 params.

        After dist_checkpointing in-place load (skip_load_to_model_and_opt=True),
        load_state_dict is not called, so we manually set steps and copy masters.
        """
        step_val = float(iteration)
        for fp32_group in self.fp32_from_bf16_groups:
            for p in fp32_group:
                if p in self.optimizer.state and "step" in self.optimizer.state[p]:
                    self.optimizer.state[p]["step"].fill_(step_val)
        for fp32_group in self.fp32_from_fp32_groups:
            for p in fp32_group:
                if p in self.optimizer.state and "step" in self.optimizer.state[p]:
                    self.optimizer.state[p]["step"].fill_(step_val)

        self._copy_main_params_to_model_params()


def get_fsdp2_bf16_master_weight_optimizer(
    config: OptimizerConfig,
    model_chunks: List[torch.nn.Module],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
    default_skip_embedding_weight_decay: bool = False,
    pg_collection: Optional["ProcessGroupCollection"] = None,
    base_optimizer_cls=torch.optim.AdamW,
    use_foreach: bool = True,
    **optimizer_kwargs,
) -> FSDP2BF16MasterWeightOptimizer:
    """Factory function to create FSDP2 BF16 master weight optimizer.

    Collects trainable parameters, creates FP32 master copies for BF16 params,
    replaces param_group entries with FP32 masters, builds a fused AdamW, and
    wraps in FSDP2BF16MasterWeightOptimizer.
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

    # Create FP32 master copies and replace in param_groups
    bf16_groups = []
    fp32_from_bf16_groups = []
    fp32_from_fp32_groups = []

    for param_group in param_groups:
        bf16_params_this_group = []
        fp32_from_bf16_this_group = []
        fp32_from_fp32_this_group = []

        for i, param in enumerate(param_group["params"]):
            if param.dtype in (torch.float16, torch.bfloat16):
                bf16_params_this_group.append(param)
                main_param = param.detach().clone().float()
                param.main_param = main_param
                param_group["params"][i] = main_param
                fp32_from_bf16_this_group.append(main_param)
            elif param.dtype == torch.float32:
                fp32_from_fp32_this_group.append(param)
            else:
                _safe_log_rank_0(f"WARNING: unexpected param dtype {param.dtype}, treating as FP32")
                fp32_from_fp32_this_group.append(param)

        bf16_groups.append(bf16_params_this_group)
        fp32_from_bf16_groups.append(fp32_from_bf16_this_group)
        fp32_from_fp32_groups.append(fp32_from_fp32_this_group)

    bf16_count = sum(len(g) for g in bf16_groups)
    fp32_count = sum(len(g) for g in fp32_from_fp32_groups)
    master_count = sum(len(g) for g in fp32_from_bf16_groups)

    _safe_log_rank_0(
        f"Creating FSDP2 BF16 master weight optimizer with "
        f"{bf16_count + fp32_count:,} parameters in {len(param_groups)} param groups"
    )

    base_optimizer = base_optimizer_cls(
        param_groups,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        fused=True,
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

    optimizer = FSDP2BF16MasterWeightOptimizer(
        optimizer=base_optimizer,
        config=config,
        init_state_fn=init_state_fn,
        bf16_groups=bf16_groups,
        fp32_from_bf16_groups=fp32_from_bf16_groups,
        fp32_from_fp32_groups=fp32_from_fp32_groups,
        use_foreach=use_foreach,
    )

    _safe_log_rank_0("=" * 80)
    _safe_log_rank_0("[FSDP2BF16MasterWeightOptimizer Initialized]")
    _safe_log_rank_0("  BF16 model params + FP32 master weights")
    _safe_log_rank_0("  No FP32->BF16 cast in forward all-gather")
    _safe_log_rank_0("  FP32 optimizer step + copy-back to BF16 per iteration")
    _safe_log_rank_0(f"  BF16 parameters: {bf16_count:,} ({master_count:,} FP32 master copies)")
    _safe_log_rank_0(f"  FP32 parameters: {fp32_count:,} (no master copy needed)")
    _safe_log_rank_0(f"  Total trainable parameters: {bf16_count + fp32_count:,}")
    _safe_log_rank_0("=" * 80)

    optimizer.warmup_foreach_cache()

    return optimizer
