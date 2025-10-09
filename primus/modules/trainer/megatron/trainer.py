###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import dataclasses
import gc
import importlib.util
import os
import statistics
import sys
import time

import megatron
import torch
import torch.distributed as dist
from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.distributed.custom_fsdp import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.distributed.torch_fully_sharded_data_parallel import (
    TorchFullyShardedDataParallel as torch_FSDP,
)
from megatron.core.utils import check_param_hashes_across_dp_replicas, get_model_config
from megatron.training.checkpointing import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)

from primus.backends.megatron.training.utils import is_pipeline_stage_containing_loss

try:
    pass

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    MockGPTDataset,
)
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.rerun_state_machine import (
    RerunDiagnostic,
    RerunErrorInjector,
    RerunMode,
    get_rerun_state_machine,
    initialize_rerun_state_machine,
)
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.utils import check_param_hashes_across_dp_replicas, get_model_config
from megatron.training import (
    ft_integration,
    get_args,
    get_tensorboard_writer,
    get_timers,
    global_vars,
    one_logger_utils,
)
from megatron.training.arguments import moe_freq_type, validate_args
from megatron.training.async_utils import (
    init_persistent_async_worker,
    maybe_finalize_async_save,
)
from megatron.training.checkpointing import (
    checkpoint_exists,
    load_args_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from megatron.training.global_vars import (
    get_args,
    get_one_logger,
    get_tensorboard_writer,
    get_timers,
    get_tokenizer,
    get_wandb_writer,
    set_global_variables,
)
from megatron.training.initialize import (
    _compile_dependencies,
    _init_autoresume,
    _initialize_distributed,
    _initialize_tp_communicators,
    _set_random_seed,
    set_jit_fusion_options,
    setup_logging,
    write_args_to_tensorboard,
)
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.training.training import (
    build_train_valid_test_data_iterators,
    checkpoint_and_decide_exit,
    disable_forward_pre_hook,
    dummy_train_step,
    enable_forward_pre_hook,
    evaluate_and_print_results,
    get_model,
    get_optimizer_param_scheduler,
    post_training_step_callbacks,
    preprocess_common_state_dict,
    print_datetime,
    should_disable_forward_pre_hook,
)
from megatron.training.utils import (
    append_to_progress_log,
    calc_params_l2_norm,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    report_memory,
    unwrap_model,
    update_use_dist_ckpt,
)
from megatron.training.yaml_arguments import validate_yaml
from pretrain_gpt import model_provider

from primus.backends.megatron.training.tokenizer.tokenizer import build_tokenizer
from primus.core.utils import checker, file_utils
from primus.core.utils.flops_estimator import num_floating_point_operations
from primus.core.utils.rocm_mem_info import get_rocm_smi_mem_info
from primus.modules.base_module import BaseModule
from primus.modules.module_utils import (
    debug_rank_0,
    log_kv_rank_0,
    log_rank_0,
    log_rank_last,
    warning_rank_0,
)
from primus.modules.trainer.base_trainer import BaseTrainer

from .utils import set_wandb_writer_patch, validate_args_on_rocm

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()


class MegatronTrainer(BaseTrainer, BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # monkey patch modules
        self.patch_moe_layer()
        self.patch_torch_fsdp()
        self.patch_get_extra_te_kwargs()
        self.patch_file_system_writer()
        self.patch_te_tp_overlap()
        self.patch_mla_attention()
        self.patch_fp8_context()
        self.patch_zbpp()

        self.app_metrics = {}

        # disable all logging handlers
        import logging

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    def patch_pt_replace_te(self, args):

        from megatron.core.models.gpt import (
            gpt_layer_specs,
            gpt_model,
            moe_module_specs,
        )

        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboAttention,
            PrimusTurboColumnParallelLinear,
            PrimusTurboColumnParallelLinearTorch,
            PrimusTurboGroupedMLP,
            PrimusTurboLayerNormColumnParallelLinear,
            PrimusTurboRowParallelLinear,
        )

        if args.use_turbo_attention:
            gpt_layer_specs.TEDotProductAttention = PrimusTurboAttention
        if args.use_turbo_parallel_linear:
            # TE row parallel linear
            gpt_layer_specs.TERowParallelLinear = PrimusTurboRowParallelLinear
            moe_module_specs.TERowParallelLinear = PrimusTurboRowParallelLinear
            # TE layer norn column parallel linear
            gpt_layer_specs.TELayerNormColumnParallelLinear = PrimusTurboLayerNormColumnParallelLinear
            # TE column parallel linear
            gpt_layer_specs.TEColumnParallelLinear = PrimusTurboColumnParallelLinear
            moe_module_specs.TEColumnParallelLinear = PrimusTurboColumnParallelLinear
            # column parallel linear
            gpt_model.tensor_parallel.ColumnParallelLinear = PrimusTurboColumnParallelLinearTorch
        if args.use_turbo_grouped_mlp:
            moe_module_specs.GroupedMLP = PrimusTurboGroupedMLP

    def patch_fp8_context(self):
        from megatron.core import fp8_utils
        from megatron.core.ssm import mamba_block
        from megatron.core.transformer import multi_token_prediction, transformer_block

        from primus.backends.megatron.core.fp8_utils import get_fp8_context

        if self.module_config.fp8:
            warning_rank_0(f"MegatronTrainer: Patch get_fp8_context...")
            transformer_block.get_fp8_context = get_fp8_context
            mamba_block.get_fp8_context = get_fp8_context
            multi_token_prediction.get_fp8_context = get_fp8_context

            fp8_utils.get_fp8_context = get_fp8_context

    def patch_te_tp_overlap(self):
        if not self.module_config.tp_comm_overlap:
            return

        def _check_tp_overlap_cfg():
            if self.module_config.fp8:
                if (
                    self.module_config.tp_comm_overlap_rs
                    or self.module_config.tp_comm_bulk_dgrad
                    or self.module_config.tp_comm_bulk_wgrad
                ):
                    raise NotImplementedError(
                        "FP8 Async-tp not support for rs, bulk overlap! Please set tp_comm_overlap_rs=False, tp_comm_bulk_dgrad=False, tp_comm_bulk_wgrad=False"
                    )

        _check_tp_overlap_cfg()

        import functools

        import transformer_engine as te
        import transformer_engine_torch as tex
        from megatron.core.utils import is_te_min_version

        from primus.backends.transformer_engine import transformer_engine_torch as ptex
        from primus.backends.transformer_engine.pytorch.module.base import (
            get_workspace,
            initialize_ub,
        )

        warning_rank_0(f"MegatronTrainer: Patch transformer_engine tp overlap...")

        tex.CommOverlap = ptex.CommOverlap
        tex.CommOverlapP2P = ptex.CommOverlapP2P
        tex.CommOverlapType = ptex.CommOverlapType
        if is_te_min_version("2.0"):
            from primus.backends.transformer_engine.pytorch.cpp_extensions.gemm import (
                general_gemm,
            )

            prev_general_gemm = te.pytorch.cpp_extensions.general_gemm
            te.pytorch.cpp_extensions.general_gemm = functools.partial(
                general_gemm, orig_func=prev_general_gemm
            )
            te.pytorch.module.linear.general_gemm = functools.partial(
                general_gemm, orig_func=prev_general_gemm
            )
            te.pytorch.module.layernorm_linear.general_gemm = functools.partial(
                general_gemm, orig_func=prev_general_gemm
            )
        else:
            from primus.backends.transformer_engine.pytorch.cpp_extensions.gemm import (
                fp8_gemm,
                gemm,
            )

            prev_gemm = te.pytorch.cpp_extensions.gemm
            prev_fp8_gemm = te.pytorch.cpp_extensions.fp8_gemm

            tex.CommOverlapAlgo = ptex.CommOverlapAlgo
            te.pytorch.cpp_extensions.CommOverlapAlgo = ptex.CommOverlapAlgo
            te.pytorch.cpp_extensions.gemm = functools.partial(gemm, orig_func=prev_gemm)
            te.pytorch.module.linear.gemm = functools.partial(gemm, orig_func=prev_gemm)
            te.pytorch.cpp_extensions.fp8_gemm = functools.partial(fp8_gemm, orig_func=prev_fp8_gemm)
            te.pytorch.module.linear.fp8_gemm = functools.partial(fp8_gemm, orig_func=prev_fp8_gemm)
        te.pytorch.module.base.initialize_ub = initialize_ub
        te.pytorch.module.base.get_workspace = get_workspace
        te.pytorch.cpp_extensions.CommOverlapType = ptex.CommOverlapType

    def patch_get_extra_te_kwargs(self):
        warning_rank_0(f"MegatronTrainer: monkey patch get_extra_te_kwargs...")
        import inspect

        import transformer_engine as te
        from megatron.core.extensions import transformer_engine as te_ext

        # Save the original _get_extra_te_kwargs function
        original_get_extra_te_kwargs = te_ext._get_extra_te_kwargs

        # Create a wrapped version of _get_extra_te_kwargs with custom overrides
        def make_get_extra_te_kwargs_with_override(**overrides):
            def _wrapped(config):
                kwargs = original_get_extra_te_kwargs(config)
                kwargs.update(overrides)
                return kwargs

            return _wrapped

        def has_parameter(cls, param):
            try:
                return param in inspect.signature(cls.__init__).parameters
            except Exception:
                return False

        # Patch TELinear
        def patch_TELinear():
            from megatron.core.extensions.transformer_engine import TELinear

            if not self.module_config.no_fp8_weight_transpose_cache:
                return
            assert has_parameter(
                te.pytorch.Linear, "keep_fp8_weight_transpose_cache"
            ), "Current Transformer-Engine not support this feature"

            orig_init = TELinear.__init__

            def new_init(self, *args, **kwargs):
                # Temporarily override the TE kwargs with our custom flag
                te_ext._get_extra_te_kwargs = make_get_extra_te_kwargs_with_override(
                    keep_fp8_weight_transpose_cache=False
                )
                try:
                    orig_init(self, *args, **kwargs)
                finally:
                    # Always restore the original function after init
                    te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

            TELinear.__init__ = new_init

        # Patch TELayerNormColumnParallelLinear
        def patch_TELayerNormColumnParallelLinear():
            from megatron.core.extensions.transformer_engine import (
                TELayerNormColumnParallelLinear,
            )

            if not self.module_config.no_fp8_weight_transpose_cache:
                return
            assert has_parameter(
                te.pytorch.LayerNormLinear, "keep_fp8_weight_transpose_cache"
            ), "Current Transformer-Engine not support this feature"

            orig_init = TELayerNormColumnParallelLinear.__init__

            def new_init(self, *args, **kwargs):
                # Temporarily override the TE kwargs with our custom flag
                te_ext._get_extra_te_kwargs = make_get_extra_te_kwargs_with_override(
                    keep_fp8_weight_transpose_cache=False
                )
                try:
                    orig_init(self, *args, **kwargs)
                finally:
                    # Always restore the original function after init
                    te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

            TELayerNormColumnParallelLinear.__init__ = new_init

        # Patch TEDelayedScaling
        def patch_TEDelayedScaling():
            from megatron.core.extensions.transformer_engine import TEDelayedScaling

            if not has_parameter(te.common.recipe.DelayedScaling, "reduce_amax"):
                return

            orig_init = TEDelayedScaling.__init__

            def new_init(self, *args, **kwargs):
                # Temporarily override the TE kwargs with our custom flag
                te_ext._get_extra_te_kwargs = make_get_extra_te_kwargs_with_override(reduce_amax=False)
                try:
                    orig_init(self, *args, **kwargs)
                finally:
                    # Always restore the original function after init
                    te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

            TEDelayedScaling.__init__ = new_init

        patch_TELinear()
        patch_TELayerNormColumnParallelLinear()
        patch_TEDelayedScaling()

    def patch_moe_layer(self):
        if self.module_config.use_deprecated_20241209_moe_layer:
            warning_rank_0(f"MegatronTrainer: monkey patch MoELayer with DeprecatedMoELayer...")
            # patch module class
            from primus.backends.megatron.core.transformer.moe.deprecated_20251209.experts import (
                DeprecatedGroupedMLP,
                DeprecatedSequentialMLP,
                DeprecatedTEGroupedMLP,
            )
            from primus.backends.megatron.core.transformer.moe.deprecated_20251209.moe_layer import (
                DeprecatedMoELayer,
                DeprecatedMoESubmodules,
            )

            sys.modules["megatron.core.transformer.moe.moe_layer"].MoELayer = DeprecatedMoELayer
            sys.modules["megatron.core.transformer.moe.moe_layer"].MoESubmodules = DeprecatedMoESubmodules
            sys.modules["megatron.core.transformer.moe.experts"].GroupedMLP = DeprecatedGroupedMLP
            sys.modules["megatron.core.transformer.moe.experts"].SequentialMLP = DeprecatedSequentialMLP
            sys.modules["megatron.core.transformer.moe.experts"].TEGroupedMLP = DeprecatedTEGroupedMLP

            # patch imported module
            from megatron.core.models.gpt import moe_module_specs

            moe_module_specs.MoELayer = DeprecatedMoELayer
            moe_module_specs.MoESubmodules = DeprecatedMoESubmodules
            moe_module_specs.GroupedMLP = DeprecatedGroupedMLP
            moe_module_specs.SequentialMLP = DeprecatedSequentialMLP
            moe_module_specs.TEGroupedMLP = DeprecatedTEGroupedMLP

        if not self.module_config.disable_primus_topk_router:
            warning_rank_0(f"MegatronTrainer: monkey patch TopKRouter...")
            if self.module_config.use_deprecated_20241209_moe_layer:
                from primus.backends.megatron.core.transformer.moe.deprecated_20251209.router import (
                    DeprecatedTopKRouter,
                )

                sys.modules["megatron.core.transformer.moe.router"].TopKRouter = DeprecatedTopKRouter

            # patch module class
            from primus.backends.megatron.core.transformer.moe.router import (
                PrimusTopKRouter,
            )

            sys.modules["megatron.core.transformer.moe.router"].TopKRouter = PrimusTopKRouter

            # patch imported module
            from megatron.core.transformer.moe import moe_layer

            moe_layer.TopKRouter = PrimusTopKRouter

            if self.module_config.use_deprecated_20241209_moe_layer:
                from primus.backends.megatron.core.transformer.moe import (
                    deprecated_20251209,
                )

                deprecated_20251209.moe_layer.TopKRouter = PrimusTopKRouter

        if self.module_config.moe_permute_fusion:
            warning_rank_0(f"MegatronTrainer: monkey patch permutation with latest fusion version...")
            from megatron.core.extensions import (
                transformer_engine as ori_transformer_engine,
            )
            from megatron.core.transformer.moe import moe_utils as ori_moe_utils

            from primus.backends.transformer_engine.pytorch.permutation import (
                moe_permute,
                moe_permute_with_probs,
                moe_sort_chunks_by_index,
                moe_sort_chunks_by_index_with_probs,
                moe_unpermute,
            )

            ori_transformer_engine.fused_permute = moe_permute
            ori_transformer_engine.fused_permute_with_probs = moe_permute_with_probs
            ori_transformer_engine.fused_sort_chunks_by_index = moe_sort_chunks_by_index
            ori_transformer_engine.fused_sort_chunks_by_index_with_probs = moe_sort_chunks_by_index_with_probs
            ori_transformer_engine.fused_unpermute = moe_unpermute

            ori_moe_utils.fused_permute = moe_permute
            ori_moe_utils.fused_permute_with_probs = moe_permute_with_probs
            ori_moe_utils.fused_sort_chunks_by_index = moe_sort_chunks_by_index
            ori_moe_utils.fused_sort_chunks_by_index_with_probs = moe_sort_chunks_by_index_with_probs
            ori_moe_utils.fused_unpermute = moe_unpermute
            ori_moe_utils.HAVE_TE = True

    def patch_mla_attention(self):
        if not self.module_config.fused_padded_mla_attention:
            return

        warning_rank_0(f"MegatronTrainer: monkey patch MLA attention to support padded fusion...")
        # pad module definition
        from megatron.core.transformer import multi_latent_attention

        from primus.backends.megatron.core.transformer.multi_latent_attention import (
            PaddedMLASelfAttention,
        )

        multi_latent_attention.MLASelfAttention = PaddedMLASelfAttention
        # pad imported module
        from megatron.core.models.gpt import gpt_layer_specs

        gpt_layer_specs.MLASelfAttention = PaddedMLASelfAttention

    def patch_torch_fsdp(self):
        if not self.module_config.use_torch_fsdp2:
            return

        warning_rank_0("MegatronTrainer: Patching torch_FSDP2 with Primus implementation...")

        try:
            # Import custom FSDP wrapper
            # Patch Megatron's internal reference to FSDP2 class
            import megatron.core.distributed.torch_fully_sharded_data_parallel as torch_fsdp_module

            from primus.backends.megatron.core.distributed.torch_fully_sharded_data_parallel import (
                PrimusTorchFullyShardedDataParallel,
            )

            torch_fsdp_module.TorchTorchFullyShardedDataParallel = PrimusTorchFullyShardedDataParallel

            # Patch training code reference
            from megatron.training import training

            training.torch_FSDP = PrimusTorchFullyShardedDataParallel

            warning_rank_0("MegatronTrainer: torch_FSDP2 patch applied successfully.")

        except ImportError as e:
            raise RuntimeError("Failed to patch torch_FSDP2: missing dependencies") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred during FSDP patching") from e

    def patch_file_system_writer(self):
        warning_rank_0("MegatronTrainer: Patching FileSystemWriterAsync...")
        try:
            import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

            from primus.backends.megatron.core.dist_checkpointing.strategies.filesystem_async import (
                PrimusFileSystemWriterAsync,
            )

            filesystem_async_module.FileSystemWriterAsync = PrimusFileSystemWriterAsync
        except Exception:
            warning_rank_0("MegatronTrainer: Patch FileSystemWriterAsync failed.")
        else:
            warning_rank_0("MegatronTrainer: Patch FileSystemWriterAsync successfully.")

    def patch_zbpp(self):
        # patch optimizer
        if self.module_config.patch_zero_bubble:
            warning_rank_0(f"MegatronTrainer: Patch ZeroBubble PP")
            import megatron.core.optimizer as optimizer

            from primus.backends.megatron.core.optimizer.zbpp_optimizer import (
                ZeroBubblePPChainedOptimizer,
            )

            optimizer.ChainedOptimizer = ZeroBubblePPChainedOptimizer

            # patch get_forward_backward_func
            import megatron.core.pipeline_parallel as ori_pp

            from primus.backends.megatron.core.pipeline_parallel.schedules import (
                get_forward_backward_func_zbpp,
            )

            ori_pp.get_forward_backward_func = get_forward_backward_func_zbpp

            # patch linear to split d_w and d_input
            import megatron.core.tensor_parallel.layers as ori_layers

            from primus.backends.megatron.core.tensor_parallel.layers import (
                LinearWithGradAccumulationAndAsyncCommunication,
            )

            ori_layers.LinearWithGradAccumulationAndAsyncCommunication = (
                LinearWithGradAccumulationAndAsyncCommunication
            )

            # patch zbv-related code
            if self.module_config.zero_bubble_v_schedule or self.module_config.enable_1f1b_v:
                import megatron.core.parallel_state as ori_parallel_state

                from primus.backends.megatron.core.parallel_state import (
                    default_embedding_ranks,
                    is_pipeline_last_stage,
                    is_rank_in_embedding_group,
                )

                ori_parallel_state.default_embedding_ranks = default_embedding_ranks
                ori_parallel_state.is_pipeline_last_stage = is_pipeline_last_stage
                ori_parallel_state.is_rank_in_embedding_group = is_rank_in_embedding_group

                import megatron.core.distributed.finalize_model_grads as ori_finalize_model_grads

                from primus.backends.megatron.core.distributed.finalize_model_grad import (
                    finalize_model_grads,
                )

                ori_finalize_model_grads.finalize_model_grads = finalize_model_grads

                import megatron.core.transformer.transformer_layer as ori_transformer_layer

                from primus.backends.megatron.core.transformer.transformer_layer import (
                    get_transformer_layer_offset,
                )

                ori_transformer_layer.get_transformer_layer_offset = get_transformer_layer_offset

    def init(self, *init_args, **kwargs):
        allowed_keys = {
            "extra_args_provider",
            "args_defaults",
            "ignore_unknown_args",
            "allow_no_cuda",
            "skip_mpu_initialization",
        }

        invalid_keys = set(kwargs.keys()) - allowed_keys
        if invalid_keys:
            raise TypeError(f"Invalid keyword arguments for MegatronTrainer: {invalid_keys}")

        log_rank_0(f"-run update_primus_config...")
        self.update_primus_config(
            args=self.module_config,
            exp_root_path=self.exp_root_path,
            exp_meta_info=self.exp_meta_info,
        )

        # Initalize and get arguments, timers, and Tensorboard writer.
        log_rank_0(f"-run initialize_megatron...")
        self.initialize_megatron(
            extra_args_provider=kwargs.get("extra_args_provider", None),
            args_defaults=kwargs.get("args_defaults", {}),
            ignore_unknown_args=kwargs.get("ignore_unknown_args", False),
            allow_no_cuda=kwargs.get("allow_no_cuda", False),
            skip_mpu_initialization=kwargs.get("skip_mpu_initialization", False),
        )

        args = get_args()
        # There are some extra limitation on ROCm need extra validate.
        validate_args_on_rocm(args)

        # Enable manually split layers in (interleaved) 1f1b pipeline
        # parallelism by monkey patching
        if args.decoder_pipeline_manual_split_list is not None:
            from .utils import set_manual_pipeline_split_patch, validate_manual_split

            log_rank_0(f"-monkey patch to enable manual pipeline split...")
            if validate_manual_split(args):
                set_manual_pipeline_split_patch(args)

        if args.log_progress:
            append_to_progress_log("Starting job")

        self.log_avg_reset_interval = args.log_avg_reset_interval
        self.log_avg_skip_iterations = args.log_avg_skip_iterations
        self.recent_tflop_throughputs = []
        self.recent_iteration_times = []
        self.recent_token_throughputs = []

        # Initialize fault tolerance
        # NOTE: ft_integration functions other than `setup` are no-op if the FT is not initialized
        if args.enable_ft_package:
            ft_integration.setup(args)
            ft_integration.maybe_setup_simulated_fault()

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

        # Adjust the startup time so it reflects the largest value.
        # This will be closer to what scheduler will see (outside of
        # image ... launches.
        global _TRAIN_START_TIME
        start_time_tensor = torch.tensor([_TRAIN_START_TIME], dtype=torch.double, device="cuda")
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()

        self.app_metrics["app_start_time"] = round(_TRAIN_START_TIME * 1000.0)
        self.app_metrics["app_model_init_start_time"] = round(_TRAIN_START_TIME * 1000.0)

        log_rank_0("time to initialize megatron (seconds): {:.3f}".format(time.time() - _TRAIN_START_TIME))
        print_datetime("after megatron is initialized")
        self.app_metrics["app_model_init_finish_time"] = one_logger_utils.get_timestamp_in_ms()

        # Track E2E metrics on pretrain start
        one_logger_utils.on_pretrain_start()

        # Context used for persisting some state between checkpoint saves.
        if args.non_persistent_ckpt_type == "local":
            try:
                from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
                    LocalCheckpointManager,
                )
                from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (
                    CliqueReplicationStrategy,
                )
            except ModuleNotFoundError:
                raise RuntimeError(
                    "The 'nvidia_resiliency_ext' module is required for local "
                    "checkpointing but was not found. Please ensure it is installed."
                )

            if args.replication:
                repl_strategy = CliqueReplicationStrategy.from_replication_params(
                    args.replication_jump, args.replication_factor
                )
            else:
                repl_strategy = None

            self.checkpointing_context = {
                "local_checkpoint_manager": LocalCheckpointManager(
                    args.non_persistent_local_ckpt_dir, repl_strategy=repl_strategy
                )
            }
        else:
            self.checkpointing_context = {}

        self.setup()

    def update_primus_config(
        self,
        args,
        exp_meta_info,
        exp_root_path,
    ):
        # rank/world_size
        args.rank = self.module_rank
        args.world_size = self.module_world_size
        args.local_rank = self.module_local_rank
        log_kv_rank_0(f"-rank", f"{args.rank}")
        log_kv_rank_0(f"-local_rank", f"{args.local_rank}")
        log_kv_rank_0(f"-world_size", f"{args.world_size}")

        # cuda
        if not args.use_torch_fsdp2 and not args.use_custom_fsdp:
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        else:
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"

        # profile
        if args.profile:
            args.disable_tensorboard = False

        # checkpoint
        ckpt_path = os.path.abspath(os.path.join(exp_root_path, "checkpoints"))
        if args.save is not None:
            warning_rank_0(f" args.save is deprecated, the checkpoint path is: {ckpt_path}")
        args.save = ckpt_path
        log_kv_rank_0(f"-save", f"{args.save}")

        # auto_continue_train
        # Note that if args.auto_continue_train is enabled, check if there are existing save checkpoints
        # for the current training experiment. If checkpoints are found, update the training
        # configuration by disable finetuning (args.finetune), loading the optimizer state
        # (args.no_load_optim), and loading the random number generator state (args.no_load_rng).
        log_kv_rank_0(f"-auto_continue_train", f"{args.auto_continue_train}")
        if args.auto_continue_train:
            latest_file = f"{args.save}/latest_checkpointed_iteration.txt"
            ckpt_exist = file_utils.is_file(latest_file)
            if ckpt_exist:
                with open(latest_file, "r") as file:
                    iter_str = file.read().strip()
                log_rank_0(f"-find '{latest_file}', latest iteration is {iter_str}.")
                if args.load != args.save:
                    warning_rank_0(
                        f"-set args.load={args.save}, path '{args.load}' is deprecated. [auto_continue_train]"
                    )
                    args.load = args.save
                if args.finetune:
                    args.finetune = False
                    warning_rank_0(f"-set args.finetune=False [auto_continue_train]")
                if args.no_load_optim:
                    args.no_load_optim = False
                    warning_rank_0(f"-set args.no_load_optim=False [auto_continue_train]")
                if args.no_load_rng:
                    args.no_load_rng = False
                    warning_rank_0(f"-set args.no_load_rng=False [auto_continue_train]")
                if not args.use_checkpoint_args:
                    args.use_checkpoint_args = True
                    warning_rank_0(f"-set args.use_checkpoint_args=True [auto_continue_train]")
            else:
                log_rank_0(f"-{latest_file} does not exist, skip auto_continue_train.")

        # tensorboard
        if not args.disable_tensorboard:
            tb_path = os.path.abspath(os.path.join(exp_root_path, "tensorboard"))
            if args.tensorboard_dir is not None:
                warning_rank_0(f"args.tensorboard_dir is deprecated, the tensorboard path is: {tb_path}")
            args.tensorboard_dir = tb_path
        else:
            args.tensorboard_dir = None
        log_kv_rank_0(f"-disable_tensorboard", f"{args.disable_tensorboard}")
        log_kv_rank_0(f"  -tensorboard_dir", f"{args.tensorboard_dir}")

        # wandb
        if not args.disable_wandb:
            wandb_path = exp_root_path
            if args.wandb_save_dir is not None:
                warning_rank_0(f"args.wandb_save_dir is deprecated, the wandb path is: {wandb_path}/wandb")
            if not hasattr(args, "wandb_project") or args.wandb_project is None:
                args.wandb_project = f"{exp_meta_info['work_group']}_{exp_meta_info['user_name']}"
                debug_rank_0(f" -create new wandb project name: {args.wandb_project}")
            if not hasattr(args, "wandb_exp_name") or args.wandb_exp_name is None:
                args.wandb_exp_name = exp_meta_info["exp_name"]
                debug_rank_0(f" - create new exp name: {args.wandb_exp_name}")
            args.wandb_save_dir = wandb_path
        elif args.wandb_project is not None:
            args.wandb_project = None
            debug_rank_0(f"args.wandb_project is disabled, as args.disable_wandb=True.")
        log_kv_rank_0(f"-disable_wandb", f"{args.disable_wandb}")
        if not args.disable_wandb and "WANDB_API_KEY" not in os.environ:
            warning_rank_0(
                "The environment variable WANDB_API_KEY is not set. "
                "Please set it before proceeding or enable 'disable_wandb' in yaml config"
            )
        log_kv_rank_0(f"  -wandb_project", f"{args.wandb_project}")
        log_kv_rank_0(f"  -wandb_exp_name", f"{args.wandb_exp_name}")
        log_kv_rank_0(f"  -wandb_save_dir", f"{args.wandb_save_dir}")
        log_kv_rank_0(f"  -wandb_entity", f"{args.wandb_entity}")

        # sink_level: logging_level
        level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        checker.check_true(args.stderr_sink_level in level_map)
        logging_level = level_map[args.stderr_sink_level]
        if args.logging_level is not None:
            warning_rank_0(
                f"-args.logging_level is deprecated, set args.logging_level={logging_level} [stderr_sink_level]"
            )
        args.logging_level = logging_level

        # update data path
        # "data1 data2 data3" -> ['data1', 'data2', 'data3']
        if args.data_path is not None:
            args.data_path = args.data_path.split(" ")
            log_rank_0(f"-data_path: {args.data_path}")

        if args.train_data_path is not None:
            args.train_data_path = args.train_data_path.split(" ")
            log_rank_0(f"-train_data_path: {args.train_data_path}")
        if args.valid_data_path is not None:
            args.valid_data_path = args.valid_data_path.split(" ")
            log_rank_0(f"-valid_data_path: {args.valid_data_path}")
        if args.test_data_path is not None:
            args.test_data_path = args.test_data_path.split(" ")
            log_rank_0(f"-test_data_path: {args.test_data_path}")

        # update sp
        if args.tensor_model_parallel_size == 1:
            args.sequence_parallel = False

        if args.iterations_to_skip is None:
            args.iterations_to_skip = []

        # support moe_freq_type
        args.moe_layer_freq = moe_freq_type(args.moe_layer_freq)

        if args.mock_data:
            args.data_path = None
            args.train_data_path = None
            args.valid_data_path = None
            args.test_data_path = None

    def vocab_size_with_padding(self, orig_vocab_size, args):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        debug_rank_0(
            " -padded vocab (size: {}) with {} dummy tokens "
            "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after)
        )
        return after

    def setup(self):
        args = get_args()
        timers = get_timers()
        # Model, optimizer, and learning rate.
        timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
        self.app_metrics["app_build_optimizer_start_time"] = one_logger_utils.get_timestamp_in_ms()
        log_rank_0(f"-setup_model_and_optimizer...")
        self.model, self.optimizer, self.opt_param_scheduler = self.setup_model_and_optimizer(
            model_provider,
            ModelType.encoder_or_decoder,
            checkpointing_context=self.checkpointing_context,
        )

        timers("model-and-optimizer-setup").stop()
        print_datetime("after model, optimizer, and learning rate " "scheduler are built")
        self.app_metrics["app_build_optimizer_finish_time"] = one_logger_utils.get_timestamp_in_ms()
        self.config = get_model_config(self.model[0])

        # Data stuff.
        self.app_metrics["app_build_dataiters_start_time"] = one_logger_utils.get_timestamp_in_ms()
        timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)

        def train_valid_test_datasets_provider_func(train_val_test_num_samples):
            return self.train_valid_test_datasets_provider(train_val_test_num_samples)

        train_valid_test_datasets_provider_func.is_distributed = True

        if args.virtual_pipeline_model_parallel_size is not None:
            self.train_data_iterator = []
            self.valid_data_iterator = []
            self.test_data_iterator = []
            for i in range(len(self.model)):
                iterators = build_train_valid_test_data_iterators(train_valid_test_datasets_provider_func)
                self.train_data_iterator.append(iterators[0])
                self.valid_data_iterator.append(iterators[1])
                self.test_data_iterator.append(iterators[2])
        else:
            (
                self.train_data_iterator,
                self.valid_data_iterator,
                self.test_data_iterator,
            ) = build_train_valid_test_data_iterators(train_valid_test_datasets_provider_func)
        timers("train/valid/test-data-iterators-setup").stop()
        print_datetime("after dataloaders are built")
        self.app_metrics["app_build_dataiters_finish_time"] = one_logger_utils.get_timestamp_in_ms()

        # Track if training is enabled. Can only be done once args.do_train is assigned after dataloader is built.
        one_logger_utils.track_config_flags(
            args.train_iters,
            args.skip_train,
            args.do_train,
            args.do_valid,
            args.do_test,
            args.dataloader_type,
            args.retro_project_dir,
            args.retro_cyclic_train_iters,
        )

        # Print setup timing.
        log_rank_0("done with setup ...")
        timers.log(
            ["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"],
            barrier=True,
        )

        one_logger = get_one_logger()
        one_logger and one_logger.log_metrics(self.app_metrics)

    def core_gpt_dataset_config_from_args(self, args):
        tokenizer = get_tokenizer()

        return GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=get_blend_from_list(args.data_path),
            blend_per_split=[
                get_blend_from_list(args.train_data_path),
                get_blend_from_list(args.valid_data_path),
                get_blend_from_list(args.test_data_path),
            ],
            split=args.split,
            num_dataset_builder_threads=args.num_dataset_builder_threads,
            path_to_cache=args.data_cache_path,
            mmap_bin_files=args.mmap_bin_files,
            tokenizer=tokenizer,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
        )

    def train_valid_test_datasets_provider(self, train_val_test_num_samples):
        """Build the train test and validation datasets.

        Args:
            train_val_test_num_samples : A list containing the number of samples in train test and validation.
        """
        args = get_args()

        config = self.core_gpt_dataset_config_from_args(args)

        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

        def is_dataset_built_on_rank():
            return (
                parallel_state.is_pipeline_first_stage(ignore_virtual=True)
                or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
            ) and parallel_state.get_tensor_model_parallel_rank() == 0

        log_rank_0("> building train, validation, and test datasets for GPT ...")
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
        ).build()

        log_rank_0("> finished creating GPT datasets ...")

        return train_ds, valid_ds, test_ds

    def initialize_megatron(
        self,
        extra_args_provider=None,
        args_defaults={},
        ignore_unknown_args=False,
        allow_no_cuda=False,
        skip_mpu_initialization=False,
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
    ):
        """Set global variables, initialize distributed, and
        set autoresume and random seeds.
        `allow_no_cuda` should not be set unless using megatron for cpu only
        data processing. In general this arg should not be set unless you know
        what you are doing.
        Returns a function to finalize distributed env initialization
        (optionally, only when args.lazy_mpu_init == True)
        """
        if not allow_no_cuda:
            # Make sure cuda is available.
            assert torch.cuda.is_available(), "Megatron requires CUDA."

        # Note: parse_args is deprecated in megatron trainer, use primus yaml config instead.
        # Parse arguments
        # args = parse_args(extra_args_provider, ignore_unknown_args)

        # Use trainer args from primus
        args = self.module_config

        # Prep for checkpoint conversion.
        if args.ckpt_convert_format is not None:
            assert args.ckpt_convert_save is not None
            assert args.load is not None
            args.exit_on_missing_checkpoint = True

        log_kv_rank_0(f"-load", f"{args.load}")
        log_kv_rank_0(f"-use_checkpoint_args", f"{args.use_checkpoint_args}")
        if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
            checker.check_true(args.load is not None, "--use-checkpoints-args requires --load argument")
            log_rank_0(f"-load_args_from_checkpoint...")
            assert args.non_persistent_ckpt_type != "local", (
                "--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local. "
                "Two-stage checkpoint loading is not implemented, and all arguments must be defined "
                "before initializing LocalCheckpointManager."
            )
            load_args_from_checkpoint(args)

        if args.async_save and args.use_persistent_ckpt_worker:
            init_persistent_async_worker()

        checker.check_true(args.yaml_cfg is None, "Xpipe doesn't support megatron yaml config.")
        if args.yaml_cfg is not None:
            args = validate_yaml(args, args_defaults)
        else:
            if args.decoder_pipeline_manual_split_list is not None:
                from .utils import validate_args_modified

                validate_args_modified(args, args_defaults)
            else:
                validate_args(args, args_defaults)

        # monkey patch _set_wandb_writer before set_global_variables
        log_rank_0(f"-monkey patch megatron.training.global_vars._set_wandb_writer...")
        megatron.training.global_vars._set_wandb_writer = set_wandb_writer_patch

        # set global args, build tokenizer, and set adlr-autoresume,
        # tensorboard-writer, and timers.
        log_rank_0(f"-set_global_variables...")
        set_global_variables(args, build_tokenizer=False)
        args = get_args()

        # set tokenizer
        log_rank_0(f"-build_tokenizer...")
        global_vars._ensure_var_is_not_initialized(global_vars._GLOBAL_TOKENIZER, "tokenizer")
        global_vars._GLOBAL_TOKENIZER = build_tokenizer(args)

        # set logging level
        setup_logging()

        # init rerun state
        def state_save_func():
            return {"rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states()}

        def state_restore_func(state_dict):
            if state_dict["rng_tracker_states"]:
                tensor_parallel.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])

        initialize_rerun_state_machine(
            state_save_func=state_save_func,
            state_restore_func=state_restore_func,
            mode=RerunMode(args.rerun_mode),
            error_injector=RerunErrorInjector(
                error_injection_rate=args.error_injection_rate,
                error_injection_type=RerunDiagnostic(args.error_injection_type),
            ),
            result_rejected_tracker_filename=args.result_rejected_tracker_filename,
        )

        # torch.distributed initialization
        def finish_mpu_init():
            args = get_args()
            # Pytorch distributed.
            log_rank_0(f"-initialize_distributed...")
            _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, None)

            # Random seeds for reproducibility.
            log_kv_rank_0(f"-seeds", f"{args.seed}")
            _set_random_seed(
                args.seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
                args.inference_rng_tracker,
                use_cudagraphable_rng=args.enable_cuda_graph,
            )

            # Setup MoE aux loss scale value.
            if args.num_experts is not None:
                from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler

                MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))

        if skip_mpu_initialization:
            return None

        args = get_args()
        log_kv_rank_0(f"-lazy_mpu_init", f"{args.lazy_mpu_init}")
        if args.lazy_mpu_init:
            # TODO is this still a necessary option?
            args.use_cpu_initialization = True
            # delayed initialization of DDP-related stuff
            # We only set basic DDP globals
            mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
            # and return function for external DDP manager
            # to call when it has DDP initialized
            mpu.set_tensor_model_parallel_rank(args.rank)
            return finish_mpu_init
        else:
            # Megatron's MPU is the master. Complete initialization right away.
            finish_mpu_init()

            # Autoresume.
            _init_autoresume()

            # Compile dependencies.
            if not args.disable_compile_dependencies:
                log_rank_0(f"-compile_dependencies...")
                _compile_dependencies()

            if args.tp_comm_overlap:
                # TODO: Should this be activated with just decoder-tp-comm-overlap too?
                _initialize_tp_communicators()

            # No continuation function
            return None

    def setup_model_and_optimizer(
        self,
        model_provider_func,
        model_type,
        no_wd_decay_cond=None,
        scale_lr_cond=None,
        lr_mult=1.0,
        checkpointing_context=None,
    ):
        """Setup model and optimizer."""
        args = get_args()
        timers = get_timers()
        one_logger = get_one_logger()

        if importlib.util.find_spec("primus_turbo") is not None:
            args = get_args()
            if args.tensor_model_parallel_size == 1:
                if args.enable_primus_turbo:
                    self.patch_pt_replace_te(args)
                    log_rank_0(f"use pt backend...")
                else:
                    log_rank_0(f"use te backend...")
            elif args.enable_primus_turbo:
                log_rank_0(f"primus turbo does not support tp, use te backend...")
        else:
            log_rank_0(f"use te backend...")

        log_rank_0(f"-run get_model")
        model = get_model(model_provider_func, model_type)
        log_rank_0(model)
        # get_megatron_optimizer will use the ddp_config
        if isinstance(model[0], torch_FSDP):
            model[0].ddp_config = DistributedDataParallelConfig()
            model[0].ddp_config.use_custom_fsdp = False

        unwrapped_model = unwrap_model(model)

        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        config.timers = timers
        log_rank_0(f"-run get_megatron_optimizer")
        optimizer = get_megatron_optimizer(
            config,
            model,
            no_wd_decay_cond,
            scale_lr_cond,
            lr_mult,
            use_gloo_process_groups=args.enable_gloo_process_groups,
        )
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

        if args.moe_use_upcycling:
            torch.distributed.barrier()
            assert not checkpoint_exists(args.save), (
                "The upcycling destination directory already exists. "
                "Please check if --moe-use-upcycling is mistakenly enabled. "
                "Upcycling should only be set for the first run when converting the dense model. "
                "All subsequent runs should remove this flag. "
            )
            num_experts = args.num_experts
            args.num_experts = None
            expert_model_parallel_size = args.expert_model_parallel_size
            args.expert_model_parallel_size = 1
            dense_model_for_upcycling = get_model(model_provider_func, model_type)
            args.num_experts = num_experts
            args.expert_model_parallel_size = expert_model_parallel_size
            _, args.num_floating_point_operations_so_far = upcycling_utils.load_and_upcycle_model(
                load_checkpoint,
                unwrapped_model,
                dense_model_for_upcycling,
                load_kwargs={
                    "model": dense_model_for_upcycling,
                    "optimizer": None,
                    "opt_param_scheduler": None,
                },
            )
            args.iteration = 1
            save_checkpoint(
                args.iteration,
                model,
                None,
                None,
                args.num_floating_point_operations_so_far,
            )
            torch.distributed.barrier()
            del dense_model_for_upcycling
            if (args.fp16 or args.bf16) and optimizer is not None:
                optimizer.reload_model_params()
            log_rank_0(f"Upcycled checkpoint saved to {args.save}")

        if (args.load is not None or args.pretrained_checkpoint is not None) and not args.moe_use_upcycling:
            one_logger and one_logger.log_metrics(
                {"load_checkpoint_start_time": one_logger_utils.get_timestamp_in_ms()}
            )
            timers("load-checkpoint", log_level=0).start(barrier=True)

            log_rank_0(f"-run load_checkpoint")
            log_rank_0(f"  -args.load={args.load}")
            args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                model,
                optimizer,
                opt_param_scheduler,
                checkpointing_context=self.checkpointing_context,
                skip_load_to_model_and_opt=HAVE_FSDP2 and args.use_torch_fsdp2,
            )
            timers("load-checkpoint").stop(barrier=True)
            timers.log(["load-checkpoint"])
            one_logger and one_logger.log_metrics(
                {
                    "load_checkpoint_finish_time": one_logger_utils.get_timestamp_in_ms(),
                    "load_checkpoint_time": timers("load-checkpoint").active_time(),
                }
            )
        else:
            args.iteration = 0
            args.num_floating_point_operations_so_far = 0

        # get model without FP16 and/or DDP wrappers
        if (
            args.iteration == 0
            and len(unwrapped_model) == 1
            and hasattr(unwrapped_model[0], "init_state_dict_from_bert")
        ):
            log_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if args.fp16:
                optimizer.reload_model_params()

        # Convert checkpoint format.
        if args.ckpt_convert_format is not None:
            load_ckpt_format = args.ckpt_format
            args.ckpt_format = args.ckpt_convert_format
            args.save = os.path.join(args.ckpt_convert_save, args.ckpt_convert_format)
            update_use_dist_ckpt(args)

            save_checkpoint(
                args.iteration,
                model,
                optimizer,
                opt_param_scheduler,
                args.num_floating_point_operations_so_far,
                preprocess_common_state_dict_fn=preprocess_common_state_dict,
            )

            log_rank_0("> converted checkpoint: %s -> %s." % (load_ckpt_format, args.ckpt_format))
            torch.distributed.barrier()
            exit()

        return model, optimizer, opt_param_scheduler

    def run(self, *args, **kwargs):
        one_logger = get_one_logger()
        args = get_args()

        if args.pp_warmup:
            from .utils import pp_warmup

            log_rank_0(
                "warmup on each rank in parallel to decrease "
                "the first iter time, especially when pp degree is large"
            )
            timers = get_timers()
            timers("pp-warmup", log_level=0).start(barrier=True)
            pp_warmup(args, self.config, self.model, self.optimizer)
            timers("pp-warmup").stop()
            timers.log(["pp-warmup"], barrier=True)

        process_non_loss_data_func = None
        non_loss_data_func = None
        if not args.skip_train:
            log_rank_0("training ...")

            if args.dataloader_type == "cyclic" and args.retro_project_dir:
                assert args.retro_cyclic_train_iters is not None
                args.train_iters = args.retro_cyclic_train_iters
                log_rank_0("retro cyclic train iters : %d" % args.train_iters)

            iteration = 0
            if args.do_train and args.train_iters > 0:
                iteration, num_floating_point_operations_so_far = self.train(
                    self.forward_step,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    self.train_data_iterator,
                    self.valid_data_iterator,
                    process_non_loss_data_func,
                    self.config,
                    self.checkpointing_context,
                    non_loss_data_func,
                )

            print_datetime("after training is done")

            if (
                args.save
                and iteration != 0
                and iteration % args.save_interval != 0
                and not args.disable_last_saving
            ):
                save_checkpoint(
                    iteration,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    self.checkpointing_context,
                    train_data_iterator=self.train_data_iterator,
                    preprocess_common_state_dict_fn=preprocess_common_state_dict,
                )

            one_logger and one_logger.log_metrics(
                {"app_train_loop_finish_time": one_logger_utils.get_timestamp_in_ms()}
            )

        else:
            log_rank_0("skipping training (--skip-train is on) ...")

            iteration = args.iteration

        if args.do_valid:
            prefix = f"iteration {iteration} on validation set"
            evaluate_and_print_results(
                prefix,
                self.forward_step,
                self.valid_data_iterator,
                self.model,
                iteration,
                process_non_loss_data_func,
                self.config,
                verbose=True,
                write_to_tensorboard=not args.skip_train,
                non_loss_data_func=non_loss_data_func,
            )

        if args.do_test:
            prefix = f"iteration {iteration} on test set"
            evaluate_and_print_results(
                prefix,
                self.forward_step,
                self.test_data_iterator,
                self.model,
                iteration,
                process_non_loss_data_func,
                self.config,
                verbose=True,
                write_to_tensorboard=not args.skip_train,
                non_loss_data_func=non_loss_data_func,
            )

        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.finish()

        ft_integration.on_checkpointing_start()
        maybe_finalize_async_save(blocking=True, terminate=True)
        ft_integration.on_checkpointing_end(is_async_finalization=True)

        one_logger and one_logger.log_metrics({"app_finish_time": one_logger_utils.get_timestamp_in_ms()})

        ft_integration.shutdown()
        one_logger_utils.finish()

        # clean up torch pg resources on exit
        if dist.is_initialized():
            dist.destroy_process_group()

    def train(
        self,
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        train_data_iterator,
        valid_data_iterator,
        process_non_loss_data_func,
        config,
        checkpointing_context,
        non_loss_data_func,
    ):
        """Training function: run train_step desired number of times, run validation, checkpoint."""
        args = get_args()
        timers = get_timers()
        one_logger = get_one_logger()

        if args.run_workload_inspector_server:
            try:
                import threading

                from workload_inspector.utils.webserver import run_server

                threading.Thread(target=run_server, daemon=True, args=(torch.distributed.get_rank(),)).start()
            except ModuleNotFoundError:
                log_rank_0("workload inspector module not found.")

        # Write args to tensorboard
        write_args_to_tensorboard()

        # Turn on training mode which enables dropout.
        for model_module in model:
            model_module.train()

        # Tracking loss.
        total_loss_dict = {}

        # Iterations.
        iteration = args.iteration
        # Make sure rerun_state_machine has the right iteration loaded from checkpoint.
        rerun_state_machine = get_rerun_state_machine()
        if rerun_state_machine.current_iteration != iteration:
            log_rank_0(f"Setting rerun_state_machine.current_iteration to {iteration}...")
            rerun_state_machine.current_iteration = iteration

        # Track E2E metrics at the start of training.
        one_logger_utils.on_train_start(
            iteration=iteration,
            consumed_train_samples=args.consumed_train_samples,
            train_samples=args.train_samples,
            seq_length=args.seq_length,
            train_iters=args.train_iters,
            save=args.save,
            async_save=args.async_save,
            log_throughput=args.log_throughput,
            num_floating_point_operations_so_far=args.num_floating_point_operations_so_far,
        )

        num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

        # Setup some training config params.
        config.grad_scale_func = optimizer.scale_loss
        config.timers = timers
        if isinstance(model[0], (custom_FSDP, DDP)) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, (
                "When overlap_grad_reduce is True, config.no_sync_func must be None; "
                "a custom no_sync_func is not supported when overlapping grad-reduce"
            )
            config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
            if len(model) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if args.align_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
                if len(model) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if args.overlap_param_gather and args.align_param_gather:
            config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
            if len(model) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads

        timers("interval-time", log_level=0).start(barrier=True)
        print_datetime("before the start of training step")
        report_memory_flag = True
        pre_hook_enabled = False
        should_exit = False
        exit_code = 0

        if args.manual_gc:
            # Disable the default garbage collector and perform the collection manually.
            # This is to align the timing of garbage collection across ranks.
            assert (
                args.manual_gc_interval >= 0
            ), "Manual garbage collection interval should be larger than or equal to 0"
            gc.disable()
            gc.collect()

        # Singleton initialization of straggler detector.
        if args.log_straggler:
            global stimer
            world = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            mmcnt = args.straggler_minmax_count
            stimer.configure(
                world,
                rank,
                mmcnt=mmcnt,
                enabled=not args.disable_straggler_on_startup,
                port=args.straggler_ctrlr_port,
            )
        num_floating_point_operations_since_last_log_event = 0.0

        num_microbatches = get_num_microbatches()
        eval_duration = 0.0
        eval_iterations = 0

        def get_e2e_base_metrics():
            """Get base metrics values for one-logger to calculate E2E tracking metrics."""
            num_floating_point_operations_since_current_train_start = (
                num_floating_point_operations_so_far - args.num_floating_point_operations_so_far
            )
            return {
                "iteration": iteration,
                "train_duration": timers("interval-time").active_time(),
                "eval_duration": eval_duration,
                "eval_iterations": eval_iterations,
                "total_flops_since_current_train_start": num_floating_point_operations_since_current_train_start,
                "num_floating_point_operations_so_far": num_floating_point_operations_so_far,
                "consumed_train_samples": args.consumed_train_samples,
                "world_size": args.world_size,
                "seq_length": args.seq_length,
            }

        # Cache into one-logger for callback.
        if one_logger:
            with one_logger.get_context_manager():
                one_logger.store_set("get_e2e_base_metrics", get_e2e_base_metrics)

        prof = None
        if args.profile and torch.distributed.get_rank() in args.profile_ranks and args.use_pytorch_profiler:
            activities = [torch.profiler.ProfilerActivity.CUDA]
            if not args.disable_profiler_activity_cpu:
                activities.append(torch.profiler.ProfilerActivity.CPU)
            worker_name = (
                f"primus-megatron-exp[{self.exp_meta_info['exp_name']}]-rank[{torch.distributed.get_rank()}]"
            )
            prof = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=max(args.profile_step_start - 1, 0),
                    warmup=1 if args.profile_step_start > 0 else 0,
                    active=args.profile_step_end - args.profile_step_start,
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    args.tensorboard_dir,
                    worker_name=worker_name,
                    use_gzip=args.torch_profiler_use_gzip,
                ),
                record_shapes=args.torch_profiler_record_shapes,
                with_stack=args.torch_profiler_with_stack,
            )
            prof.start()

        start_iteration = iteration
        # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
        # or random initialization don't propagate to all ranks in first all-gather (which is a
        # no-op if things work correctly).
        if should_disable_forward_pre_hook(args):
            disable_forward_pre_hook(model, param_sync=False)
            # Also remove param_sync_func temporarily so that sync calls made in
            # `forward_backward_func` are no-ops.
            param_sync_func = config.param_sync_func
            config.param_sync_func = None
            pre_hook_enabled = False
        # Also, check weight hash across DP replicas to be very pedantic.
        if args.check_weight_hash_across_dp_replicas_interval is not None:
            assert check_param_hashes_across_dp_replicas(
                model, cross_check=True
            ), "Parameter hashes not matching across DP replicas"
            torch.distributed.barrier()
            log_rank_0(f">>> Weight hashes match after {iteration} iterations...")

        if args.dump_pp_data:
            from .utils import set_dump_pp_data_patch

            set_dump_pp_data_patch()
            log_rank_0(f"dump pp schedule data for visualization")

        # Run training iterations till done.
        while iteration < args.train_iters:
            if args.profile and torch.distributed.get_rank() in args.profile_ranks:
                if args.use_pytorch_profiler:
                    prof.step()
                elif iteration == args.profile_step_start:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

            ft_integration.on_checkpointing_start()
            maybe_finalize_async_save(blocking=False)
            ft_integration.on_checkpointing_end(is_async_finalization=True)

            # Update number of microbatches first without consistency check to decide if a
            # checkpoint should be saved. If the number of microbatches is different
            # from the previous iteration, save a checkpoint. Then run consistency check
            # to make sure training configuration is still valid.
            update_num_microbatches(args.consumed_train_samples, consistency_check=False, verbose=True)
            if get_num_microbatches() != num_microbatches and iteration != 0:
                assert get_num_microbatches() > num_microbatches, (
                    f"Number of microbatches should be increasing due to batch size rampup; "
                    f"instead going from {num_microbatches} to {get_num_microbatches()}"
                )
                if args.save is not None:
                    save_checkpoint_and_time(
                        iteration,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        num_floating_point_operations_so_far,
                        checkpointing_context,
                        train_data_iterator=train_data_iterator,
                    )
            num_microbatches = get_num_microbatches()
            update_num_microbatches(args.consumed_train_samples, consistency_check=True, verbose=True)

            # Completely skip iteration if needed.
            if iteration in args.iterations_to_skip:
                # Dummy train_step to fast forward train_data_iterator.
                dummy_train_step(train_data_iterator)
                iteration += 1
                batch_size = (
                    mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
                )
                args.consumed_train_samples += batch_size
                args.skipped_train_samples += batch_size
                continue

            # Run training step.
            args.curr_iteration = iteration
            ft_integration.on_training_step_start()
            (
                loss_dict,
                skipped_iter,
                should_checkpoint,
                should_exit,
                exit_code,
                grad_norm,
                num_zeros_in_grad,
            ) = self.train_step(
                forward_step_func,
                train_data_iterator,
                model,
                optimizer,
                opt_param_scheduler,
                config,
            )
            ft_integration.on_training_step_end()
            if should_checkpoint:
                save_checkpoint_and_time(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context,
                    train_data_iterator=train_data_iterator,
                )
            if should_exit:
                break

            # Enable forward pre-hooks after first set of forward and backward passes.
            # When running in fp16, skip all NaN iterations until steady-state loss scaling value
            # is reached.
            if iteration == start_iteration:
                if skipped_iter:
                    # Only enable forward pre-hook after a training step has successfully run. Relevant
                    # for fp16 codepath where first XX iterations are skipped until steady-state loss
                    # scale value is reached.
                    start_iteration = iteration + 1
                else:
                    # Enable forward pre-hook after training step has successfully run. All subsequent
                    # forward passes will use the forward pre-hook / `param_sync_func` in
                    # `forward_backward_func`.
                    if should_disable_forward_pre_hook(args):
                        enable_forward_pre_hook(model)
                        config.param_sync_func = param_sync_func
                        pre_hook_enabled = True

            iteration += 1
            batch_size = mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
            args.consumed_train_samples += batch_size
            num_skipped_samples_in_batch = (
                get_current_global_batch_size() - get_current_running_global_batch_size()
            )
            if args.decrease_batch_size_if_needed:
                assert num_skipped_samples_in_batch >= 0
            else:
                assert num_skipped_samples_in_batch == 0
            args.skipped_train_samples += num_skipped_samples_in_batch
            flops_calc = (
                num_floating_point_operations
                if not args.multi_latent_attention
                else self.num_floating_point_operations_mla_moe
            )
            num_floating_point_operations_in_batch = flops_calc(args, batch_size)
            num_floating_point_operations_so_far += num_floating_point_operations_in_batch
            num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

            # Logging.
            if not optimizer.is_stub_optimizer:
                loss_scale = optimizer.get_loss_scale().item()
            else:
                loss_scale = 1.0
            params_norm = None

            if args.log_params_norm:
                params_norm = calc_params_l2_norm(model)
            learning_rate = None
            decoupled_learning_rate = None
            for param_group in optimizer.param_groups:
                if param_group["is_decoupled_lr"]:
                    decoupled_learning_rate = param_group["lr"]
                else:
                    learning_rate = param_group["lr"]
            report_memory_flag = self.training_log(
                loss_dict,
                total_loss_dict,
                learning_rate,
                decoupled_learning_rate,
                iteration,
                loss_scale,
                report_memory_flag,
                skipped_iter,
                grad_norm,
                params_norm,
                num_zeros_in_grad,
            )

            # Evaluation.
            if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
                timers("interval-time").stop()
                if should_disable_forward_pre_hook(args):
                    disable_forward_pre_hook(model)
                    pre_hook_enabled = False
                if args.manual_gc and args.manual_gc_eval:
                    # Collect all objects.
                    gc.collect()
                prefix = f"iteration {iteration}"
                timers("eval-time", log_level=0).start(barrier=True)
                evaluate_and_print_results(
                    prefix,
                    forward_step_func,
                    valid_data_iterator,
                    model,
                    iteration,
                    process_non_loss_data_func,
                    config,
                    verbose=False,
                    write_to_tensorboard=True,
                    non_loss_data_func=non_loss_data_func,
                )
                eval_duration += timers("eval-time").elapsed()
                eval_iterations += args.eval_iters
                timers("eval-time").stop()
                one_logger_utils.track_e2e_metrics()

                if args.manual_gc and args.manual_gc_eval:
                    # Collect only the objects created and used in evaluation.
                    gc.collect(generation=0)
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(model)
                    pre_hook_enabled = True
                timers("interval-time", log_level=0).start(barrier=True)

            # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
            # Some of these only happen at specific iterations.
            post_training_step_callbacks(
                model,
                optimizer,
                opt_param_scheduler,
                iteration,
                prof,
                num_floating_point_operations_since_last_log_event,
            )

            # Checkpoint and decide whether to exit.
            should_exit = checkpoint_and_decide_exit(
                model,
                optimizer,
                opt_param_scheduler,
                iteration,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator,
            )
            if should_exit:
                break

        one_logger_utils.track_e2e_metrics()

        if args.dump_pp_data:
            from .utils import dump_pp_data

            pp_data_dir = os.environ.get("DUMP_PP_DIR", "output/pp_data")
            dump_pp_data(args, get_num_microbatches(), pp_data_dir)
            log_rank_0(f"pp schedule data dumped to {pp_data_dir}")

        # Flush TensorBoard, WandB writers and one-logger.
        writer = get_tensorboard_writer()
        if writer:
            writer.flush()

        # Close out pre-hooks if using distributed optimizer and overlapped param gather.
        if pre_hook_enabled:
            disable_forward_pre_hook(model)

        ft_integration.on_checkpointing_start()
        # This will finalize all unfinalized async request and terminate
        # a persistent async worker if persistent ckpt worker is enabled
        maybe_finalize_async_save(blocking=True, terminate=True)
        ft_integration.on_checkpointing_end(is_async_finalization=True)
        if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
            ft_integration.get_rank_monitor_client().shutdown_workload_monitoring()

        # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
        if should_exit:
            wandb_writer = get_wandb_writer()
            if wandb_writer:
                wandb_writer.finish()
            ft_integration.shutdown()
            sys.exit(exit_code)

        return iteration, num_floating_point_operations_so_far

    def train_step(
        self,
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        no_optimizer_post_validation=False,
    ):
        """Single training step."""
        args = get_args()
        timers = get_timers()

        def run_forward_backward_func(optimizer=None):
            """Forward pass.
            optimizer is not None for running post validation."""
            from megatron.core.pipeline_parallel import get_forward_backward_func

            forward_backward_func = get_forward_backward_func()
            kwargs = {}
            if optimizer is not None:
                kwargs["optimizer"] = optimizer
            return forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches() * args.num_seq_splits,
                seq_length=args.seq_length // args.num_seq_splits,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=False,
                **kwargs,
            )

        rerun_state_machine = get_rerun_state_machine()
        while rerun_state_machine.should_run_forward_backward(data_iterator):
            # Set grad to zero.
            for model_chunk in model:
                model_chunk.zero_grad_buffer()
            optimizer.zero_grad()

            # Forward pass.
            losses_reduced = run_forward_backward_func()

        should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
        if should_exit:
            return {}, True, should_checkpoint, should_exit, exit_code, None, None

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Vision gradients.
        if args.vision_pretraining and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0])
            unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

        # Update parameters.

        timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
        # update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        if get_args().profile:
            torch.cuda.nvtx.range_push("Optimizer")
        if args.patch_zero_bubble and args.enable_optimizer_post_validation:
            if optimizer.post_validation_enabled and not no_optimizer_post_validation:
                optimizer.pre_step(args, timers)
                if get_args().profile:
                    torch.cuda.nvtx.range_pop()
                if get_args().profile:
                    torch.cuda.nvtx.range_push("post_validation_phase")
                update_successful, grad_norm, num_zeros_in_grad = run_forward_backward_func(optimizer)
                if get_args().profile:
                    torch.cuda.nvtx.range_pop()
                # Here num_zeros_in_grad is a fake name, representing for optimizer_rollback
            else:
                update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
                if get_args().profile:
                    torch.cuda.nvtx.range_pop()
            optimizer.record_grad_norm(grad_norm)
        else:
            update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
            if get_args().profile:
                torch.cuda.nvtx.range_pop()

        timers("optimizer").stop()

        # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
        # so we must gather across mp ranks
        update_successful = logical_and_across_model_parallel_group(update_successful)
        # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
        # so we must gather across mp ranks
        grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
        if args.log_num_zeros_in_grad:
            num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

        # Vision momentum.
        if args.vision_pretraining and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0])
            unwrapped_model.update_momentum(args.curr_iteration)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        if is_pipeline_stage_containing_loss():
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0].keys():
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    if isinstance(val, tuple) or isinstance(val, list):
                        numerator += val[0]
                        denominator += val[1]
                    else:
                        # legacy behavior. we average over the number of microbatches,
                        # and so the denominator is 1.
                        numerator += val
                        denominator += 1
                loss_reduced[key] = numerator / denominator
            return (
                loss_reduced,
                skipped_iter,
                should_checkpoint,
                should_exit,
                exit_code,
                grad_norm,
                num_zeros_in_grad,
            )
        return (
            {},
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
        )

    def training_log(
        self,
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        iteration,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
    ):
        """Log training information such as losses, timing, ...."""
        args = get_args()
        timers = get_timers()
        writer = get_tensorboard_writer()
        wandb_writer = get_wandb_writer()
        get_one_logger()

        # Advanced, skipped, and Nan iterations.
        advanced_iters_key = "advanced iterations"
        skipped_iters_key = "skipped iterations"
        nan_iters_key = "nan iterations"
        # Advanced iterations.
        if not skipped_iter:
            total_loss_dict[advanced_iters_key] = total_loss_dict.get(advanced_iters_key, 0) + 1
        else:
            if advanced_iters_key not in total_loss_dict:
                total_loss_dict[advanced_iters_key] = 0
        # Skipped iterations.
        total_loss_dict[skipped_iters_key] = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
        # Update losses and set nan iterations
        got_nan = False
        for key in loss_dict:
            if not skipped_iter:
                total_loss_dict[key] = (
                    total_loss_dict.get(key, torch.tensor([0.0], dtype=torch.float, device="cuda"))
                    + loss_dict[key]
                )
            else:
                value = loss_dict[key].float().sum().item()
                is_nan = value == float("inf") or value == -float("inf") or value != value
                got_nan = got_nan or is_nan
        total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(got_nan)

        # Logging.
        timers_to_log = [
            "forward-backward",
            "forward-compute",
            "backward-compute",
            "batch-generator",
            "forward-recv",
            "forward-send",
            "backward-recv",
            "backward-send",
            "forward-send-forward-recv",
            "forward-send-backward-recv",
            "backward-send-forward-recv",
            "backward-send-backward-recv",
            "forward-backward-send-forward-backward-recv",
            "layernorm-grads-all-reduce",
            "embedding-grads-all-reduce",
            "all-grads-sync",
            "params-all-gather",
            "optimizer-copy-to-main-grad",
            "optimizer-unscale-and-check-inf",
            "optimizer-clip-main-grad",
            "optimizer-count-zeros",
            "optimizer-inner-step",
            "optimizer-copy-main-to-model-params",
            "optimizer",
        ]

        # Calculate batch size.
        batch_size = args.micro_batch_size * args.data_parallel_size * get_num_microbatches()

        # Track app tag & app tag ID
        one_logger_utils.track_app_tag(batch_size, args.world_size, args.seq_length)

        total_iterations = total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]

        # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
        learning_rate = reduce_max_stat_across_model_parallel_group(learning_rate)
        # Tensorboard values.
        # Timer requires all the ranks to call.
        if args.log_timers_to_tensorboard and (iteration % args.tensorboard_log_interval == 0):
            timers.write(timers_to_log, writer, iteration, normalizer=total_iterations)
        if iteration % args.tensorboard_log_interval == 0:
            if wandb_writer:
                wandb_writer.log({"samples vs steps": args.consumed_train_samples}, iteration)
            if writer:
                writer.add_scalar("learning-rate", learning_rate, iteration)
                if args.decoupled_lr is not None:
                    writer.add_scalar("decoupled-learning-rate", decoupled_learning_rate, iteration)
                writer.add_scalar(
                    "learning-rate vs samples",
                    learning_rate,
                    args.consumed_train_samples,
                )
            if wandb_writer:
                wandb_writer.log({"learning-rate": learning_rate}, iteration)
            if writer:
                writer.add_scalar("batch-size", batch_size, iteration)
                writer.add_scalar("batch-size vs samples", batch_size, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"batch-size": batch_size}, iteration)
            for key in loss_dict:
                if writer:
                    writer.add_scalar(key, loss_dict[key], iteration)
                    writer.add_scalar(key + " vs samples", loss_dict[key], args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({key: loss_dict[key]}, iteration)
            if args.log_loss_scale_to_tensorboard:
                if writer:
                    writer.add_scalar("loss-scale", loss_scale, iteration)
                    writer.add_scalar("loss-scale vs samples", loss_scale, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({"loss-scale": loss_scale}, iteration)
            if args.log_world_size_to_tensorboard:
                if writer:
                    writer.add_scalar("world-size", args.world_size, iteration)
                    writer.add_scalar(
                        "world-size vs samples",
                        args.world_size,
                        args.consumed_train_samples,
                    )
                if wandb_writer:
                    wandb_writer.log({"world-size": args.world_size}, iteration)
            if grad_norm is not None:
                if writer:
                    writer.add_scalar("grad-norm", grad_norm, iteration)
                    writer.add_scalar("grad-norm vs samples", grad_norm, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({"grad-norm": grad_norm}, iteration)
            if num_zeros_in_grad is not None:
                if writer:
                    writer.add_scalar("num-zeros", num_zeros_in_grad, iteration)
                    writer.add_scalar(
                        "num-zeros vs samples",
                        num_zeros_in_grad,
                        args.consumed_train_samples,
                    )
                if wandb_writer:
                    wandb_writer.log({"num-zeros": num_zeros_in_grad}, iteration)
            if params_norm is not None:
                if writer:
                    writer.add_scalar("params-norm", params_norm, iteration)
                    writer.add_scalar(
                        "params-norm vs samples",
                        params_norm,
                        args.consumed_train_samples,
                    )
                if wandb_writer:
                    wandb_writer.log({"params-norm": params_norm}, iteration)
            if args.log_memory_to_tensorboard:
                mem_stats = torch.cuda.memory_stats()
                if writer:
                    writer.add_scalar(
                        "mem-reserved-bytes",
                        mem_stats["reserved_bytes.all.current"],
                        iteration,
                    )
                    writer.add_scalar(
                        "mem-allocated-bytes",
                        mem_stats["allocated_bytes.all.current"],
                        iteration,
                    )
                    writer.add_scalar(
                        "mem-max-allocated-bytes",
                        mem_stats["allocated_bytes.all.peak"],
                        iteration,
                    )
                    writer.add_scalar(
                        "mem-allocated-count",
                        mem_stats["allocation.all.current"],
                        iteration,
                    )
                if wandb_writer:
                    wandb_writer.log(
                        {"mem-reserved-bytes": mem_stats["reserved_bytes.all.current"]},
                        iteration,
                    )
                    wandb_writer.log(
                        {"mem-allocated-bytes": mem_stats["allocated_bytes.all.current"]},
                        iteration,
                    )
                    wandb_writer.log(
                        {"mem-max-allocated-bytes": mem_stats["allocated_bytes.all.peak"]},
                        iteration,
                    )
                    wandb_writer.log(
                        {"mem-allocated-count": mem_stats["allocation.all.current"]},
                        iteration,
                    )
        if args.num_experts is not None:
            moe_loss_scale = 1 / get_num_microbatches()
            track_moe_metrics(
                loss_scale=moe_loss_scale,
                iteration=iteration,
                writer=writer,
                wandb_writer=wandb_writer,
                total_loss_dict=total_loss_dict,
                per_layer_logging=args.moe_per_layer_logging,
                moe_layer_freq=args.moe_layer_freq,
                num_layers=args.num_layers,
            )

        if iteration % args.log_interval == 0:
            # Note(wenx): If we want to collect rocm-smi memory information for the first two iterations,
            # place the collection before the timer to minimize its impact on latency measurements for iterations ≥ 3.
            if args.log_throughput:
                if args.use_rocm_mem_info or iteration in args.use_rocm_mem_info_iters:
                    rocm_total_mem, rocm_used_mem, rocm_free_mem = get_rocm_smi_mem_info(
                        self.module_local_rank
                    )

            elapsed_time = timers("interval-time").elapsed(barrier=True)
            elapsed_time_per_iteration = elapsed_time / total_iterations

            flops_calc = (
                num_floating_point_operations
                if not args.multi_latent_attention
                else self.num_floating_point_operations_mla_moe
            )
            throughput = flops_calc(args, batch_size) / (
                elapsed_time_per_iteration * 10**12 * args.world_size
            )

            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar("iteration-time", elapsed_time_per_iteration, iteration)
                if wandb_writer:
                    wandb_writer.log({"iteration-time": elapsed_time_per_iteration}, iteration)
            # log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            log_string = f""
            if hasattr(self, "episode_count") and self.episode_count is not None:
                log_string += f" episode {self.episode_count} |"
            log_string += " iteration {:8d}/{:8d} |".format(iteration, args.train_iters)
            log_string += " consumed samples: {:12d} |".format(args.consumed_train_samples)
            if (
                iteration == self.log_avg_skip_iterations + 1
                or len(self.recent_iteration_times) >= self.log_avg_reset_interval
            ):
                self.recent_iteration_times.clear()
            self.recent_iteration_times.append(elapsed_time_per_iteration * 1000.0)
            log_string += " elapsed time per iteration (ms): {:.1f}/{:.1f} |".format(
                elapsed_time_per_iteration * 1000.0,
                statistics.mean(self.recent_iteration_times),
            )
            if args.log_throughput:
                if (
                    iteration == self.log_avg_skip_iterations + 1
                    or len(self.recent_tflop_throughputs) >= self.log_avg_reset_interval
                ):
                    self.recent_tflop_throughputs.clear()
                self.recent_tflop_throughputs.append(throughput)

                if not args.use_rocm_mem_info:
                    hip_free_mem, hip_total_mem = torch.cuda.mem_get_info()
                    hip_used_mem = hip_total_mem - hip_free_mem
                    hip_mem_usage = hip_used_mem / hip_total_mem
                    log_string += (
                        f" hip mem usage/free/total/usage_ratio: {hip_used_mem/1024/1024/1024:.2f}GB/"
                    )
                    log_string += f"{hip_free_mem/1024/1024/1024:.2f}GB/"
                    log_string += f"{hip_total_mem/1024/1024/1024:.2f}GB/{hip_mem_usage*100:.2f}% |"

                if args.use_rocm_mem_info or iteration in args.use_rocm_mem_info_iters:
                    rocm_mem_usage = rocm_used_mem / rocm_total_mem
                    log_string += (
                        f" rocm mem usage/free/total/usage_ratio: {rocm_used_mem/1024/1024/1024:.2f}GB/"
                    )
                    log_string += f"{rocm_free_mem/1024/1024/1024:.2f}GB/"
                    log_string += f"{rocm_total_mem/1024/1024/1024:.2f}GB/{rocm_mem_usage*100:.2f}% |"

                log_string += (
                    f" throughput per GPU (TFLOP/s/GPU): {throughput:.1f}/"
                    f"{statistics.mean(self.recent_tflop_throughputs):.1f} |"
                )
                token_throughput = args.seq_length * batch_size / elapsed_time_per_iteration / args.world_size
                if (
                    iteration == self.log_avg_skip_iterations + 1
                    or len(self.recent_token_throughputs) >= self.log_avg_reset_interval
                ):
                    self.recent_token_throughputs.clear()
                self.recent_token_throughputs.append(token_throughput)
                log_string += (
                    f" tokens per GPU (tokens/s/GPU): {token_throughput:.1f}/"
                    f"{statistics.mean(self.recent_token_throughputs):.1f} |"
                )
                if args.log_timers_to_tensorboard:
                    if args.use_rocm_mem_info or iteration in args.use_rocm_mem_info_iters:
                        mem_collector = "rocm"
                        used_mem, free_mem, total_mem, mem_usage = (
                            rocm_used_mem,
                            rocm_free_mem,
                            rocm_total_mem,
                            rocm_mem_usage,
                        )
                    else:
                        mem_collector = "hip"
                        used_mem, free_mem, total_mem, mem_usage = (
                            hip_used_mem,
                            hip_free_mem,
                            hip_total_mem,
                            hip_mem_usage,
                        )
                    if writer:
                        writer.add_scalar("throughput(tflops/sec/gpu)", throughput, iteration)
                        writer.add_scalar(
                            "token_throughput(tokens/sec/gpu)",
                            token_throughput,
                            iteration,
                        )
                        writer.add_scalar(
                            f"{mem_collector}_used_mem(GB)",
                            used_mem / 1024 / 1024 / 1024,
                            iteration,
                        )
                        writer.add_scalar(
                            f"{mem_collector}_free_mem(GB)",
                            free_mem / 1024 / 1024 / 1024,
                            iteration,
                        )
                        writer.add_scalar(
                            f"{mem_collector}_total_mem(GB)",
                            total_mem / 1024 / 1024 / 1024,
                            iteration,
                        )
                        writer.add_scalar(f"{mem_collector}_mem_usage(%)", mem_usage * 100.0, iteration)
                    if wandb_writer:
                        wandb_writer.log({"throughput(tflops/sec/gpu)": throughput}, iteration)
                        wandb_writer.log(
                            {"token_throughput(tokens/sec/gpu)": token_throughput},
                            iteration,
                        )
                        wandb_writer.log(
                            {f"{mem_collector}_used_mem(GB)": used_mem / 1024 / 1024 / 1024},
                            iteration,
                        )
                        wandb_writer.log(
                            {f"{mem_collector}_free_mem(GB)": free_mem / 1024 / 1024 / 1024},
                            iteration,
                        )
                        wandb_writer.log(
                            {f"{mem_collector}_total_mem(GB)": total_mem / 1024 / 1024 / 1024},
                            iteration,
                        )
                        wandb_writer.log({f"{mem_collector}_mem_usage(%)": mem_usage * 100.0}, iteration)
            assert learning_rate is not None
            # Decoupled_learning_rate should be not None only on first and last pipeline stage.
            log_string += " learning rate: {:.6E} |".format(learning_rate)
            if args.decoupled_lr is not None and (
                mpu.is_pipeline_first_stage(ignore_virtual=True)
                or mpu.is_pipeline_last_stage(ignore_virtual=True)
            ):
                assert decoupled_learning_rate is not None
                log_string += " decoupled learning rate: {:.6E} |".format(decoupled_learning_rate)
            else:
                assert decoupled_learning_rate is None
            log_string += " global batch size: {:5d} |".format(batch_size)
            for key in total_loss_dict:
                if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                    avg = total_loss_dict[key].item() / float(max(1, total_loss_dict[advanced_iters_key]))
                    if avg > 0.0:
                        log_string += " {}: {:.6E} |".format(key, avg)
                    total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device="cuda")
            log_string += " loss scale: {:.1f} |".format(loss_scale)
            if grad_norm is not None:
                log_string += " grad norm: {:.3f} |".format(grad_norm)
            if num_zeros_in_grad is not None:
                log_string += " num zeros: {:.1f} |".format(num_zeros_in_grad)
            if params_norm is not None:
                log_string += " params norm: {:.3f} |".format(params_norm)
            log_string += " number of skipped iterations: {:3d} |".format(total_loss_dict[skipped_iters_key])
            log_string += " number of nan iterations: {:3d} |".format(total_loss_dict[nan_iters_key])
            total_loss_dict[advanced_iters_key] = 0
            total_loss_dict[skipped_iters_key] = 0
            total_loss_dict[nan_iters_key] = 0

            if get_args().patch_zero_bubble and get_args().zero_bubble_v_schedule or get_args().enable_1f1b_v:
                log_rank_0(log_string)
            else:
                log_rank_last(log_string)
            if report_memory_flag and learning_rate > 0.0:
                # Report memory after optimizer state has been initialized.
                if torch.distributed.get_rank() == 0:
                    num_microbatches = get_num_microbatches()
                    report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
                report_memory("(after {} iterations)".format(iteration))
                report_memory_flag = False

            # Removed to avoid global sync in zero bubble schedules.
            if not (get_args().zero_bubble_v_schedule or get_args().patch_zero_bubble):
                timers.log(timers_to_log, normalizer=args.log_interval)

        return report_memory_flag

    def num_floating_point_operations_mla_moe(self, args, batch_size):
        # MoE.
        gated_linear_multiplier = 3 / 2 if args.swiglu else 1
        num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
        ffn_hidden_size = (
            args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
        )
        shared_expert_ffn_hidden_size = (
            0
            if args.moe_shared_expert_intermediate_size is None
            else args.moe_shared_expert_intermediate_size
        )

        # The 12x term below comes from the following factors; for more details, see
        # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
        #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
        #       in MLP layer).
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        expansion_factor_fw_bw = 3
        expansion_factor_gemm_stack = 2
        expansion_factor_gemm_ops = 2

        return (
            expansion_factor_fw_bw
            * expansion_factor_gemm_ops
            * batch_size
            * args.seq_length
            * args.num_layers
            * args.hidden_size
            * args.hidden_size
            * (
                # Attention - q.
                (
                    (
                        # Attention - q_proj.
                        args.num_attention_heads
                        * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                        / args.hidden_size
                    )
                    if not args.q_lora_rank
                    else (
                        # Attention - q_down_proj.
                        (args.q_lora_rank / args.hidden_size)
                        # Attention - q_up_proj.
                        + (
                            (args.q_lora_rank / args.hidden_size)
                            * (
                                args.num_attention_heads
                                * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                                / args.hidden_size
                            )
                        )
                    )
                )
                # Attention - kv_down_proj.
                + ((args.kv_lora_rank + args.qk_pos_emb_head_dim) / args.hidden_size)
                # Attention - kv_up_proj.
                + (
                    (args.kv_lora_rank / args.hidden_size)
                    * (args.num_attention_heads * (args.qk_head_dim + args.v_head_dim) / args.hidden_size)
                )
                # Attention - Q*K.
                + (
                    (args.seq_length / args.hidden_size)
                    * (
                        args.num_attention_heads
                        * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                        / args.hidden_size
                    )
                )
                # Attention - A(ttention Score)*V.
                + (
                    (args.seq_length / args.hidden_size)
                    * (args.num_attention_heads * args.v_head_dim / args.hidden_size)
                )
                # Attention - output proj.
                + (args.num_attention_heads * args.v_head_dim / args.hidden_size)
                # MLP.
                + (
                    (ffn_hidden_size / args.hidden_size)
                    * num_experts_routed_to
                    * gated_linear_multiplier
                    * expansion_factor_gemm_stack
                )
                # Shared Experts.
                + (
                    (shared_expert_ffn_hidden_size / args.hidden_size)
                    * gated_linear_multiplier
                    * expansion_factor_gemm_stack
                )
                # Logit. (untie_embeddings_and_output_weights=True)
                + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
                * expansion_factor_gemm_stack
            )
        )
