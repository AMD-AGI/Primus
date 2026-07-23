###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Set

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from torch.distributed import ProcessGroup

from primus.core.utils.module_utils import warning_rank_0

try:
    from torch.distributed import DeviceMesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    HAVE_FSDP = True
except ImportError:
    HAVE_FSDP = False


def _validate_mesh_ranks(rank_tensor, shard_group, replicate_group):
    """Validate that mesh tensor slices match the actual process group ranks."""
    current_rank = dist.get_rank()

    shard_group_ranks = dist.get_process_group_ranks(shard_group)
    for row in rank_tensor.tolist():
        if current_rank in row:
            if sorted(row) != sorted(shard_group_ranks):
                raise RuntimeError(f"Mesh shard ranks {row} != group ranks {shard_group_ranks}")

    replicate_group_ranks = dist.get_process_group_ranks(replicate_group)
    for col_idx in range(rank_tensor.shape[1]):
        col = rank_tensor[:, col_idx].tolist()
        if current_rank in col:
            if sorted(col) != sorted(replicate_group_ranks):
                raise RuntimeError(f"Mesh replicate ranks {col} != group ranks {replicate_group_ranks}")


class PrimusTorchFullyShardedDataParallel(_BaseDataParallel):
    """
    Customized FSDP implementation for Primus framework with support for TransformerBlock.

    For models using TransformerBlock (e.g., Flux with heterogeneous layers), FSDP wraps
    the TransformerBlock itself rather than individual TransformerLayer subclasses to avoid
    duplicate mesh_dim_names errors.

    Key difference from base Megatron class:
    - Prevents wrapping of modules that are descendants of already-wrapped modules
    - This avoids the "Invalid mesh_dim_names ('dp', 'dp')" error when using TransformerBlock
      with heterogeneous TransformerLayer subclasses.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        sub_modules_to_wrap: Optional[Set[torch.nn.Module]] = None,
        disable_bucketing: bool = False,
        process_group: Optional[ProcessGroup] = None,
        **kwargs,
    ):
        if not HAVE_FSDP:
            raise RuntimeError("TorchFullyShardedDataParallel requires PyTorch >= 2.4.0 with FSDP 2 support.")

        super().__init__(config=config, module=module)

        # Store ddp_config for later access
        self.ddp_config = ddp_config

        if process_group is None:
            self.process_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        else:
            self.process_group = process_group

        if sub_modules_to_wrap is None:
            sub_modules_to_wrap = {
                TransformerLayer,
                LanguageModelEmbedding,
                RotaryEmbedding,
                tensor_parallel.ColumnParallelLinear,
            }

        if kwargs:
            warning_rank_0(f"PrimusTorchFullyShardedDataParallel: unused args: {kwargs}")

        # Build DeviceMesh from Megatron's process groups
        from megatron.training import get_args

        from primus.core.utils.module_utils import log_rank_0

        args = get_args()
        replicate_degree = getattr(args, "data_parallel_replicate_degree", 1)
        dp_size = dist.get_world_size(self.process_group)

        if replicate_degree > 1:
            shard_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=True
            )
            replicate_group = parallel_state.get_inter_distributed_optimizer_instance_group()

            shard_size = dist.get_world_size(shard_group)

            full_dp_ranks = dist.get_process_group_ranks(self.process_group)
            rank_tensor = torch.tensor(full_dp_ranks).reshape(replicate_degree, shard_size)

            _validate_mesh_ranks(rank_tensor, shard_group, replicate_group)

            mesh = DeviceMesh.from_group(
                [replicate_group, shard_group],
                device_type="cuda",
                mesh=rank_tensor.tolist(),
                mesh_dim_names=("dp_replicate", "dp_shard"),
            )
        else:
            mesh = DeviceMesh.from_group(
                self.process_group,
                device_type="cuda",
                mesh_dim_names=("dp",),
            )

        reshard_after_forward = getattr(self.ddp_config, "reshard_after_forward", True)

        kwargs = {
            "mesh": mesh,
            "reshard_after_forward": reshard_after_forward,
        }

        # When params_dtype is FP32 but training is BF16 (FSDP2 FP32 param optimizer),
        # add MixedPrecisionPolicy so FSDP2 casts to BF16 for forward/backward.
        # param_dtype=bf16 is always required: it ensures forward inputs
        # (activations) get cast to BF16. For params with FP8 AG extensions,
        # FSDP2 uses fsdp_post_all_gather (unaffected by param_dtype).
        # This matches TorchTitan where param_dtype is always set from config.
        use_fp8_all_gather = getattr(args, "use_fsdp2_fp8_all_gather", False)
        if config.params_dtype == torch.float32 and config.bf16:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            )
            kwargs["mp_policy"] = mp_policy
            log_rank_0(
                "FSDP2: MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=bf16) "
                "[FP32 param optimizer: FSDP casts activations to BF16, BF16 reduce]"
            )
        elif (
            config.params_dtype == torch.bfloat16
            and config.bf16
            and getattr(args, "use_fsdp2_bf16_master_weight_optimizer", False)
        ):
            # BF16 master weight optimizer: params are already BF16, so no
            # param_dtype casting is needed. Setting param_dtype=None skips
            # cast_forward_inputs entirely (avoids FP32→BF16 activation cast
            # at every FSDP module boundary). reduce_dtype=bf16 halves
            # ReduceScatter bandwidth; FP32 master weights in the optimizer
            # still ensure full-precision parameter updates.
            mp_policy = MixedPrecisionPolicy(
                param_dtype=None,
                reduce_dtype=torch.bfloat16,
            )
            kwargs["mp_policy"] = mp_policy
            log_rank_0(
                "FSDP2: MixedPrecisionPolicy(param_dtype=None, reduce_dtype=bf16) "
                "[BF16 master weight optimizer: no cast_forward_inputs, BF16 reduce]"
            )

        if replicate_degree > 1:
            log_rank_0(f"FSDP2 Configuration:")
            log_rank_0(f"  mode: HSDP (replicate={replicate_degree}, shard={dp_size // replicate_degree})")
            log_rank_0(
                f"  reshard_after_forward: {reshard_after_forward} "
                f"(ZeRO-{'3' if reshard_after_forward else '2'})"
            )
            log_rank_0(
                f"  Data parallel size: {dp_size} "
                f"(replicate={replicate_degree} x shard={dp_size // replicate_degree})"
            )
        else:
            log_rank_0(f"FSDP2 Configuration:")
            log_rank_0(f"  mode: FSDP")
            log_rank_0(
                f"  reshard_after_forward: {reshard_after_forward} "
                f"(ZeRO-{'3' if reshard_after_forward else '2'})"
            )
            log_rank_0(f"  Data parallel size: {dp_size}")

        # Helper functions to save/restore custom parameter attributes
        def save_custom_attrs(module):
            custom_attrs = {}
            for name, param in module.named_parameters():
                attrs = vars(param)
                if is_float8tensor(param):
                    # disable fp8 transpose cache and perform transposing fp8 weights
                    # at each micro-batch because torch-FSDP doesn't recognize the
                    # micro-batch id, thus removing unnecessary memory stores
                    attrs["_fp8_attrs"]["transpose_invalid"] = False
                    del attrs["_fp8_attrs"]["transpose"]
                custom_attrs[name] = {k: v for k, v in attrs.items()}
            return custom_attrs

        def restore_custom_attrs(module, custom_attrs):
            for name, param in module.named_parameters():
                if name in custom_attrs:
                    for attr_name, attr_value in custom_attrs[name].items():
                        setattr(param, attr_name, attr_value)

        # Save custom attributes that might be removed by FSDP
        attrs = save_custom_attrs(self.module)

        # FP8 all-gather validation
        if (
            use_fp8_all_gather
            and isinstance(reshard_after_forward, int)
            and not isinstance(reshard_after_forward, bool)
        ):
            raise ValueError(
                "FP8 all-gather is incompatible with reshard_after_forward=int (partial reshard)"
            )

        # FSDP2 FP8 all-gather is incompatible with delayed scaling: the delayed
        # forward re-quantizes the weight and does not handle the all-gathered
        # FP8UnshardedWeightTensor subclass produced by the all-gather path, so
        # the combination would miscompute. Reject it explicitly.
        if use_fp8_all_gather:
            from megatron.core.enums import Fp8Recipe

            uses_delayed = (
                getattr(config, "fp8_scaling_strategy", "dynamic") == "delayed"
                or getattr(config, "fp8_recipe", None) == Fp8Recipe.delayed
            )
            if uses_delayed:
                raise ValueError(
                    "use_fsdp2_fp8_all_gather is incompatible with delayed FP8 scaling "
                    "(fp8_recipe='delayed' / fp8_scaling_strategy='delayed'). Use a "
                    "non-delayed recipe (e.g. tensorwise) with FSDP2 FP8 all-gather, or "
                    "disable use_fsdp2_fp8_all_gather."
                )

        # Local transformer implementation does not support ColumnParallelLinear.
        if config.transformer_impl == "local":
            sub_modules_to_wrap = {
                sub_module
                for sub_module in sub_modules_to_wrap
                if sub_module != tensor_parallel.ColumnParallelLinear
            }
        sub_modules_to_wrap = set(sub_modules_to_wrap)

        # Process _fsdp_modules attribute
        for sub_module in self.module.modules():
            fsdp_modules = getattr(sub_module, "_fsdp_modules", [])
            for f in fsdp_modules:
                sub_modules_to_wrap.add(f)

        fp8_ag_stochastic_rounding = getattr(args, "fp8_all_gather_stochastic_rounding", False)
        fp8_ag_deq_after_ag = getattr(args, "fp8_all_gather_deq_requant", False)
        if use_fp8_all_gather:
            from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
                _wrap_fp8_weights_for_all_gather,
            )
            from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
                _build_fp8_config,
            )

            fp8_ag_config = _build_fp8_config(config)
            wrapped_count = _wrap_fp8_weights_for_all_gather(
                self.module,
                fp8_ag_config,
                stochastic_rounding=fp8_ag_stochastic_rounding,
                deq_after_ag=fp8_ag_deq_after_ag,
            )
            sr_tag = " [stochastic rounding]" if fp8_ag_stochastic_rounding else ""
            deq_tag = " [deq+requant]" if fp8_ag_deq_after_ag else ""
            log_rank_0(
                f"FSDP2: Wrapped {wrapped_count} params with FP8 all-gather "
                f"(granularity={fp8_ag_config.granularity}){sr_tag}{deq_tag}"
            )

        # ============================================================================
        # CUSTOM WRAPPING LOGIC: Skip descendants of already-wrapped modules
        # ============================================================================
        wrapped_modules_set = set()
        wrapped_list = []

        for sub_module in self.module.modules():
            should_wrap = any(
                isinstance(sub_module, sub_module_to_wrap) for sub_module_to_wrap in sub_modules_to_wrap
            )
            if not should_wrap:
                continue

            is_descendant = False
            for wrapped_module in wrapped_modules_set:
                if sub_module is not wrapped_module:
                    for child in wrapped_module.modules():
                        if child is sub_module:
                            is_descendant = True
                            break
                    if is_descendant:
                        break

            if is_descendant:
                continue

            fully_shard(sub_module, **kwargs)
            wrapped_modules_set.add(sub_module)
            wrapped_list.append(sub_module)

        log_rank_0(f"FSDP2: wrapped {len(wrapped_list)} inner modules + root")

        prefetch_depth = getattr(self.config, "fsdp_prefetch_depth", 1)
        recompute_on = getattr(self.config, "recompute_granularity", None) is not None
        for i, mod in enumerate(wrapped_list):
            # With activation recompute enabled, the recomputed forward pass triggers
            # wrong-direction forward-prefetch all-gathers that expose communication and
            # inflate peak memory (matches Megatron upstream, which only sets backward
            # prefetch when recompute is on). Skip explicit forward prefetch in that case.
            if not recompute_on:
                fwd_targets = wrapped_list[i + 1 : i + 1 + prefetch_depth]
                if fwd_targets:
                    mod.set_modules_to_forward_prefetch(fwd_targets)

            bwd_start = max(0, i - prefetch_depth)
            bwd_targets = list(reversed(wrapped_list[bwd_start:i]))
            if bwd_targets:
                mod.set_modules_to_backward_prefetch(bwd_targets)

        # Wrap the root module as required by the FSDP API
        fully_shard(self.module, **kwargs)

        if use_fp8_all_gather:
            from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
                precompute_fp8_scales_for_fsdp,
            )

            cache_data = getattr(self.config, "fp8_precompute_data_cache", True)
            use_cpp = getattr(self.config, "use_cpp_fp8_quantize", False)
            precompute_fp8_scales_for_fsdp(
                self.module,
                cache_data=cache_data,
                use_cpp_quantize=use_cpp,
                stochastic_rounding=fp8_ag_stochastic_rounding,
            )

        restore_custom_attrs(self.module, attrs)

        if getattr(args, "overlap_grad_norm", False):
            from primus.backends.megatron.core.optimizer.incremental_grad_norm import (
                IncrementalGradNormAccumulator,
            )

            # Use the shard group for the norm all-reduce (correct for both
            # FSDP and HSDP). For HSDP, replicate-group ranks hold identical
            # gradients after AR, so the full DP group would over-count.
            if replicate_degree > 1:
                norm_reduce_group = shard_group
            else:
                norm_reduce_group = self.process_group

            accumulator = IncrementalGradNormAccumulator(
                shard_process_group=norm_reduce_group,
                device=torch.device("cuda"),
            )
            n_hooked = 0
            for param in self.module.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(accumulator.make_hook())
                    n_hooked += 1
            args._grad_norm_accumulator = accumulator
            log_rank_0(f"FSDP2: Registered incremental grad norm hooks on {n_hooked} params")

    def compile_model(self):
        """
        Delegate torch.compile to the underlying model's compile_model() method.

        Traverses the FSDP2 module hierarchy (up to 3 levels deep) to find
        the actual model (e.g., Flux) and calls its compile_model() if present.
        Skipped if enable_torch_compile is False or no compile_model method is found.
        """
        from primus.core.utils.module_utils import log_rank_0

        try:
            from megatron.training import get_args

            args = get_args()

            # Check if compilation is enabled
            if not getattr(args, "enable_torch_compile", False):
                return

        except Exception:
            # If args not available, skip compilation
            log_rank_0("  ℹ  Cannot access args for torch.compile settings, skipping")
            return

        # FSDP2 wraps models in FSDPFloat16Module, which itself wraps the actual model
        # We need to traverse the hierarchy to find the actual model (e.g., Flux)
        current_module = self.module
        module_path = []

        # Traverse up to 3 levels deep to find a module with compile_model
        for depth in range(3):
            module_type = type(current_module).__name__
            module_path.append(module_type)

            if hasattr(current_module, "compile_model") and callable(
                getattr(current_module, "compile_model")
            ):
                log_rank_0(f"  Found compile_model at depth {depth}: {' -> '.join(module_path)}")
                log_rank_0(f"  Calling compile_model on {module_type}...")
                current_module.compile_model()
                log_rank_0(f"    ✓ Model compilation complete")
                return

            # Try to go deeper if there's a 'module' attribute
            if hasattr(current_module, "module"):
                current_module = current_module.module
            else:
                break

        # No compile_model found - log that we can't compile directly
        # Note: We can't reassign self.module after FSDP wrapping, so if the underlying
        # model doesn't have compile_model, we skip compilation. The model should implement
        # compile_model if torch.compile is desired.
        log_rank_0(f"  ℹ  No compile_model method found in module hierarchy: {' -> '.join(module_path)}")
        log_rank_0(f"    (Model should implement compile_model() if torch.compile is desired)")

    def finish_grad_sync(self, *args, **kwargs):
        """No-op for FSDP2: gradient sync is handled by the FSDP runtime."""

    def load_state_dict(self, state_dict, strict=True):
        """
        No-op because tensors are already loaded in-place by
        `_load_base_checkpoint` with FSDP2."""
