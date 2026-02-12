###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Custom Llama2 recipe for Primus.

This is a custom recipe based on Megatron-Bridge's llama2.py recipe,
but placed in Primus for easier customization and extension.
"""

import gc
import os
import sys
import time
from collections import deque
from datetime import timedelta
from typing import Any, Callable, List, Optional, Union

import torch
from typing_extensions import TypedDict, Unpack

from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches,
)
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.parallel_state import update_pg_timeout
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator, get_rerun_state_machine
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.utils import check_param_hashes_across_dp_replicas, get_model_config
from megatron.core.full_cuda_graph import FullCudaGraphWrapper

# Replace Megatron-Bridge's logging with Primus's logging so that it is visible in the output logs
from primus.modules.module_utils import log_rank_0, log_rank_last
from megatron.bridge.utils import common_utils
common_utils.print_rank_0 = log_rank_0
common_utils.print_rank_last = log_rank_last
common_utils.warn_rank_0 = log_rank_0
common_utils.warn_rank_last = log_rank_last

from megatron.bridge import AutoBridge
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.recipes.utils.finetune_utils import default_squad_config
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import maybe_finalize_async_save
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    FinetuningDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.mixed_precision import (
    MixedPrecisionConfig,
    bf16_mixed,
    register,
)
from megatron.bridge.training.nvrx_straggler import safe_shutdown_nvrx_straggler_manager
from megatron.bridge.training.profiling import (
    handle_profiling_step,
    handle_profiling_stop,
    initialize_pytorch_profiler,
    should_profile_rank,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tensor_inspect import (
    tensor_inspect_end_if_enabled,
    tensor_inspect_step_if_enabled,
)
from megatron.bridge.training.utils import flop_utils
from megatron.bridge.training.utils.train_utils import (
    calc_params_l2_norm,
    prepare_forward_step_func,
    training_log,
)
from megatron.bridge.training.train import (
    train_step,
    should_disable_forward_pre_hook,
    enable_forward_pre_hook,
    disable_forward_pre_hook,
    maybe_synchronize_training_step,
    maybe_report_stragglers,
    maybe_check_weight_hash_across_dp_replicas,
    maybe_run_manual_gc,
    checkpoint_and_decide_exit,
    save_checkpoint_and_time,
    _should_skip_and_handle_iteration,
    _delete_cuda_graphs,
    _maybe_register_fsdp_buffers,
)

@register
def bf16_with_fp8_hybrid() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with MXFP8.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with MXFP8 mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "delayed"
    cfg.fp8_amax_history_len = 4
    cfg.fp8_amax_compute_algo = "most_recent"
    cfg.fp8_param_gather = True
    return cfg


class Llama2CustomKwargs(TypedDict, total=False):
    """Typed options accepted by custom Llama2 recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: Optional[str]
    name: str
    # Dataset configuration
    data_paths: Optional[List[str]]
    data_args_path: Optional[str]
    train_data_path: Optional[List[str]]
    valid_data_path: Optional[List[str]]
    test_data_path: Optional[List[str]]
    per_split_data_args_path: Optional[str]
    mock: bool
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: Optional[torch.dtype]
    virtual_pipeline_model_parallel_size: Optional[int]
    context_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]
    eval_interval: int
    save_interval: int
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_eps: float = 1e-8
    weight_decay: float = 0.0001
    eval_iters: int = 32
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool
    packed_train_data_path: str | None
    packed_val_data_path: str | None
    packed_metadata_path: str | None


def llama2_70b_lora_config(**user_kwargs: Unpack[Llama2CustomKwargs]) -> ConfigContainer:
    """
    Return a custom pre-training config for Llama-2 70B.

    This is a custom variant that can be modified without changing Megatron-Bridge code.
    See `_llama2_lora` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", "lora")
    recommended_kwargs: Llama2CustomKwargs = {
        "hf_path": "meta-llama/Llama-2-70b-hf",
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "train_iters": 1_168_251,
        "global_batch_size": 512,
        "micro_batch_size": 1,
        "lr_warmup_iters": 2000,
        "eval_interval": 2000,
        "save_interval": 2000,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_eps": 1e-8,
        "weight_decay": 0.0001,
        "eval_iters": 32,
        "peft": peft_value,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Llama2CustomKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama2_lora(**combined_kwargs)


def _llama2_lora(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    # Dataset configuration
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
    # Model configuration
    tensor_model_parallel_size: int = 2,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    # Training hyperparameters
    train_iters: int = 1_168_251,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: Optional[int] = None,
    eval_interval: int = 2000,
    save_interval: int = 2000,
    use_null_tokenizer: bool = False,
    pretrained_checkpoint: str | None = None,
    peft: str | PEFT | None = "lora",
    packed_sequence: bool = False,
    packed_train_data_path: str | None = None,
    packed_val_data_path: str | None = None,
    packed_metadata_path: str | None = None,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    adam_eps: float = 1e-5,
    weight_decay: float = 0.1,
    eval_iters: int = 32,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
) -> ConfigContainer:
    """
    Create a custom pre-training configuration for Llama2 models.

    This is based on Megatron-Bridge's llama2 recipe but can be customized
    for Primus-specific needs.

    Args:
        hf_path (str): HuggingFace model path (e.g., "meta-llama/Llama-2-70b-hf").
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        data_paths (Optional[List[str]]): List of paths to dataset files. If None, mock data will be used.
        data_args_path (Optional[str]): Path to file containing data arguments.
        train_data_path (Optional[List[str]]): List of training data paths.
        valid_data_path (Optional[List[str]]): List of validation data paths.
        test_data_path (Optional[List[str]]): List of test data paths.
        per_split_data_args_path (Optional[str]): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data. If True, ignores data_paths.
        tensor_model_parallel_size (int): Degree of tensor model parallelism.
        pipeline_model_parallel_size (int): Degree of pipeline model parallelism.
        pipeline_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size (Optional[int]): Size of virtual pipeline parallelism.
        context_parallel_size (int): Degree of context parallelism to be passed to model_config.
        sequence_parallel (bool): Whether to use sequence parallelism.
        use_megatron_fsdp (bool): Whether to use Megatron FSDP.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (Optional[int]): Number of iterations over which to decay the LR.
        eval_interval (int): Evaluation interval.
        save_interval (int): Save interval.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.
        comm_overlap_config (Optional[CommOverlapConfig]): Communication overlap configuration for the model.
        adam_beta1 (float): Beta1 parameter for Adam optimizer.
        adam_beta2 (float): Beta2 parameter for Adam optimizer.
        adam_eps (float): Epsilon parameter for Adam optimizer.
        weight_decay (float): Weight decay parameter for Adam optimizer.
        eval_iters (int): Number of iterations to run for evaluation validation/test for.
        pretrained_checkpoint (str | None): Path to pretrained checkpoint to load.
        peft (str | PEFT | None): PEFT configuration (e.g., "lora" or LoRA object).
        packed_sequence (bool): Whether to use packed sequences.
        packed_train_data_path (str | None): Path to packed training data.
        packed_val_data_path (str | None): Path to packed validation data.
        packed_metadata_path (str | None): Path to packed metadata.

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=True) #GPTProvider
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length
    model_cfg.perform_initialization = True
    model_cfg.fp16 = False
    model_cfg.bf16 = True
    model_cfg.params_dtype = torch.bfloat16
    model_cfg.autocast_dtype = torch.bfloat16
    model_cfg.pipeline_dtype = torch.bfloat16
    model_cfg.gradient_accumulation_fusion = False
    model_cfg.cross_entropy_loss_fusion = False
    model_cfg.bias_dropout_fusion = False
    model_cfg.fp8 = 'hybrid'
    model_cfg.fp8_recipe = 'delayed'
    model_cfg.fp8_amax_history_len = 4
    model_cfg.disable_parameter_transpose_cache = True
    model_cfg.use_transformer_engine_full_layer_spec = False # Doesn't work beacuse of RMSNorm is not supported in FusedLayerNorm
    model_cfg.cpu_offloading = False
    model_cfg.cpu_offloading_num_layers = 0
    model_cfg.empty_unused_memory_level = 2

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        weight_decay=weight_decay,
        max_lr=lr,
        min_lr=min_lr,
    )

    peft_config = LoRA(
        dim=16,
        alpha=32,
        dropout=0.1,
        dropout_position="pre",
        lora_A_init_method="kaiming",
        lora_B_init_method="zero",
        a2a_experimental=True,
        target_modules=["linear_qkv", "linear_proj"],
    )

    # Packed sequence configuration
    # if packed_sequence:
    #     packed_sequence_specs = PackedSequenceSpecs(
    #         packed_sequence_size=seq_length,  # Must be > 0 to use packed files
    #         tokenizer_model_name=hf_path,
    #         packed_train_data_path=packed_train_data_path or "/data/train.npy",
    #         packed_val_data_path=packed_val_data_path or "/data/validation.npy",
    #         packed_metadata_path=packed_metadata_path or "/data/packed_metadata.jsonl",  # Metadata for packed sequences
    #     )
    # else:
    #     packed_sequence_specs = None

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
            empty_unused_memory_level=0, # 0: No empty, 1: Empty at end of eval, 2: Empty at end of eval and train. 
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=False,
            use_megatron_fsdp=False,
            keep_fp8_transpose_cache=False,
        ),
        dataset=default_squad_config(seq_length, packed_sequence),
        # dataset=FinetuningDatasetConfig(
        #     dataset_root="/data",  # Point to your .npy files directory
        #     seq_length=seq_length,
        #     seed=1234,
        #     packed_sequence_specs=packed_sequence_specs,
        #     # Dataloader config parameters
        #     data_sharding=True,
        #     dataloader_type="batch",  # "batch" is recommended for fine-tuning
        #     num_workers=1,
        # ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer" if use_null_tokenizer else "HuggingFaceTokenizer",
            tokenizer_model=hf_path if not use_null_tokenizer else None,
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE if use_null_tokenizer else None,
        ),
        checkpoint=CheckpointConfig(
            save_interval=None,
            save=None,
            load=None,
            save_optim=False,
            save_rng=False,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            replication_factor=0,
            most_recent_k=0,
            finetune=True,
            load_optim=False,
            load_rng=False,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
        peft=peft_config,
    )

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
        )

    return cfg

from megatron.bridge.training import eval 
def evaluate_and_print_results_custom(
    state: GlobalState,
    prefix: str,
    forward_step_func: ForwardStepCallable,
    data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    model: list[MegatronModule],
    config: ConfigContainer,
    verbose: bool = False,
    write_to_tensorboard: bool = True,
    process_non_loss_data_func: Optional[Callable] = None,
    non_loss_data_func: Optional[Callable] = None,
) -> None:
    """Helper function to evaluate and dump results on screen.

    Args:
        state (GlobalState): The global state object.
        prefix (str): Prefix for logging evaluation results.
        forward_step_func (Callable): The function that performs a forward step.
        data_iterator (Optional[Union[RerunDataIterator, list[RerunDataIterator]]]): Iterator over evaluation data.
        model (list[MegatronModule]): list of model chunks.
        config (ConfigContainer): Configuration container (potentially redundant).
        verbose (bool, optional): Whether to print evaluation progress. Defaults to False.
        write_to_tensorboard (bool, optional): Whether to write results to TensorBoard. Defaults to True.
        process_non_loss_data_func (Optional[Callable], optional): Function to process non-loss data. Defaults to None.
        non_loss_data_func (Optional[Callable], optional): Function to compute non-loss data. Defaults to None.
    """
    log_rank_0(f"Evaluating and printing results at {prefix}")
    def is_last_rank():
        return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)
    import math
    if write_to_tensorboard:
        writer = state.tensorboard_logger
    else:
        writer = None

    wandb_writer = state.wandb_logger

    log_rank_0(f"Calling eval.evaluate")
    total_loss_dict, collected_non_loss_data, timelimit = eval.evaluate(
        state, forward_step_func, data_iterator, model, process_non_loss_data_func, config, verbose, non_loss_data_func
    )
    log_rank_0(f"Evaluation completed")

    # Timelimit hit during evaluation
    if timelimit:
        return
    string = f" validation loss at {prefix} | "
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)
        if writer:
            writer.add_scalar("{} validation".format(key), total_loss_dict[key].item(), state.train_state.step)
            writer.add_scalar(
                "{} validation vs samples".format(key),
                total_loss_dict[key].item(),
                state.train_state.consumed_train_samples,
            )
            if state.cfg.logger.log_validation_ppl_to_tensorboard:
                writer.add_scalar("{} validation ppl".format(key), ppl, state.train_state.step)
                writer.add_scalar(
                    "{} validation ppl vs samples".format(key), ppl, state.train_state.consumed_train_samples
                )

        if wandb_writer and is_last_rank():
            wandb_writer.log({"{} validation".format(key): total_loss_dict[key].item()}, state.train_state.step)
            if state.cfg.logger.log_validation_ppl_to_tensorboard:
                wandb_writer.log({"{} validation ppl".format(key): ppl}, state.train_state.step)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, state.train_state.step, writer)

    length = len(string) + 1
    log_rank_last("-" * length)
    log_rank_last(string)
    log_rank_last("-" * length)

eval.evaluate_and_print_results = evaluate_and_print_results_custom

def warmup_eval(
    state: GlobalState,
    forward_step_func: ForwardStepCallable,
    data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    model: list[MegatronModule],
    config: ConfigContainer,
    verbose: bool = False,
    process_non_loss_data_func: Optional[Callable] = None,
    non_loss_data_func: Optional[Callable] = None,
    num_warmup_iters: int = 10,
) -> None:
    log_rank_0(f"Starting warmup eval...")
    for i in range(num_warmup_iters):
        log_rank_0(f"Warmup eval iteration {i} running...")
        eval.evaluate(
            state, forward_step_func, data_iterator, model, process_non_loss_data_func, config, verbose, non_loss_data_func
        )
        log_rank_0(f"Warmup eval iteration {i} completed")
    log_rank_0(f"Warmup eval completed")

def megatron_bridge_train_override(
    forward_step_func: ForwardStepCallable,
    model: list[MegatronModule],
    optimizer: MegatronOptimizer,
    scheduler: OptimizerParamScheduler,
    train_data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    valid_data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    global_state: GlobalState,
    checkpointing_context: dict[str, Any],
    pg_collection: ProcessGroupCollection,
    process_non_loss_data_func: Optional[Callable] = None,
    non_loss_data_func: Optional[Callable] = None,
) -> None:
    """Main training loop.

    Handles the overall training process, including the iteration loop,
    calling train_step, evaluation, checkpointing, logging, and exit conditions.

    Args:
        forward_step_func: Callable that executes a single forward step.
        model: list of model chunks (potentially wrapped in DDP).
        optimizer: The optimizer instance.
        scheduler: The learning rate scheduler instance.
        train_data_iterator: Iterator for the training dataset.
        valid_data_iterator: Iterator for the validation dataset.
        global_state: The GlobalState object holding various training states.
        checkpointing_context: Context dictionary for checkpointing.
        process_non_loss_data_func: Optional function to process non-loss data during evaluation.
        non_loss_data_func: Optional function to compute non-loss data during evaluation.

    Warnings:
        This is an experimental API and is subject to change in backwards
        incompatible ways without notice.
    """
    config: ConfigContainer = global_state.cfg
    model_config = get_model_config(model[0])
    train_config = config.train
    timers = global_state.timers
    straggler_timer = global_state.straggler_timer
    energy_monitor = global_state.energy_monitor

    # Prepare forward_step_func (check signature and inject state if needed).
    # This is done once to prevent creating new partial objects every iteration.
    #
    # Note on reference semantics:
    # - functools.partial stores a reference to global_state, not a copy
    # - When global_state.train_state.step changes, the partial sees the updated value
    # - This is safe because GlobalState is a mutable object passed by reference
    #
    # For functors (classes with __call__ defined):
    # - For functors: partial(functor_instance, state) still allows functor's internal state to work
    # - inspect.signature() properly inspects the __call__ method of functors
    wrapped_forward_step_func = prepare_forward_step_func(forward_step_func, global_state)

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Make sure rerun_state_machine has the right iteration loaded from checkpoint.
    rerun_state_machine = get_rerun_state_machine()
    if rerun_state_machine.current_iteration != global_state.train_state.step:
        log_rank_0(f"Setting rerun_state_machine.current_iteration to {global_state.train_state.step}...")
        rerun_state_machine.current_iteration = global_state.train_state.step

    num_floating_point_operations_so_far = global_state.train_state.floating_point_operations_so_far
    num_floating_point_operations_since_last_log_event = 0.0

    if energy_monitor is not None:
        energy_monitor.setup()
        energy_monitor.resume()

    timers("interval-time", log_level=0).start(barrier=True)
    report_memory_flag = True
    pre_hook_enabled = False
    should_exit = False
    exit_code = 0

    if train_config.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert train_config.manual_gc_interval >= 0, (
            "Manual garbage collection interval should be larger than or equal to 0"
        )
        gc.disable()
        gc.collect()

    if config.straggler and config.straggler.log_straggler:
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = config.straggler.straggler_minmax_count
        straggler_timer.configure(
            world,
            rank,
            mmcnt=mmcnt,
            enabled=not config.straggler.disable_straggler_on_startup,
            port=config.straggler.straggler_ctrlr_port,
        )

    # Initialize NVRx straggler detection if enabled
    nvrx_straggler_manager = global_state.nvrx_straggler_manager
    if nvrx_straggler_manager is not None:
        try:
            # Initialize the straggler detector first
            nvrx_straggler_manager.initialize()
            # Wrap the train_step function for monitoring
            # Note: The nvidia-resiliency-ext library will monitor the actual train_step calls
            nvrx_straggler_manager.wrap_train_step_function(train_step)
        except Exception as e:
            log_rank_0(f"Failed to initialize NVRx straggler detection: {e}")
            # Set to None to disable further checks
            global_state._nvrx_straggler_manager = None

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    prof = None
    nsys_nvtx_context = None  # NVTX context for nsys profiling
    prof_config = config.profiling
    if prof_config and should_profile_rank(prof_config, torch.distributed.get_rank()):
        if prof_config.use_pytorch_profiler:
            prof = initialize_pytorch_profiler(prof_config, config.logger.tensorboard_dir)
            prof.start()

    # Initialize RPD profiler if enabled
    rpd = None
    rpd_status = None
    profiler_type = os.getenv("PROFILER", '')
    rpd_warmup_steps = int(os.getenv("RPD_WARMUP_STEPS", "0"))
    rpd_active_steps = int(os.getenv("RPD_ACTIVE_STEPS", "100"))
    if profiler_type == 'rpd':
        try:
            from rpdTracerControl import rpdTracerControl
            rank = torch.distributed.get_rank()
            rpd_filename = os.getenv("RPD_TRACE_FILENAME", f"trace.rpd")
            rpdTracerControl.setFilename(name=rpd_filename, append=True)
            rpd = rpdTracerControl()
            log_rank_0(f"RPD profiler initialized. Will start at step {rpd_warmup_steps} and stop at step {rpd_warmup_steps + rpd_active_steps}")
        except Exception as e:
            log_rank_0(f"Failed to initialize RPD profiler: {e}")
            rpd = None
    else:
        log_rank_0(f"### Profiler type is {profiler_type}")

    # Megatron FSDP and FSDP2 does not have this hook
    should_toggle_forward_pre_hook = should_disable_forward_pre_hook(
        config.ddp.use_megatron_fsdp,
        config.optimizer.use_distributed_optimizer,
        config.ddp.overlap_param_gather,
    )
    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_toggle_forward_pre_hook:
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = model_config.param_sync_func
        model_config.param_sync_func = None
        pre_hook_enabled = False
    # Also, check weight hash across DP replicas to be very pedantic.
    if train_config.check_weight_hash_across_dp_replicas_interval is not None:
        assert check_param_hashes_across_dp_replicas(model, cross_check=True), (
            "Parameter hashes not matching across DP replicas"
        )
        torch.distributed.barrier()
        log_rank_0(f">>> Weight hashes match after {global_state.train_state.step} iterations...")

    # Capture CUDA Graphs.
    cuda_graph_helper = None
    if model_config.cuda_graph_impl == "transformer_engine":
        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=model_config,
            seq_length=config.model.seq_length,
            micro_batch_size=config.train.micro_batch_size,
            optimizers=[optimizer],
        )

    # Track train step elapsed time for throughput logging
    history_wct = None
    if config.logger.log_throughput_to_tensorboard:
        history_wct = deque(maxlen=config.logger.throughput_window_size + 1)

    # Wrap forward_backward_func for Full iteration CUDA graph
    forward_backward_func = get_forward_backward_func()
    if config.model.cuda_graph_impl == "local" and "full_iteration" in config.model.cuda_graph_scope:
        forward_backward_func = FullCudaGraphWrapper(
            forward_backward_func, cuda_graph_warmup_steps=config.model.cuda_graph_warmup_steps
        )

    warmup_eval(
        global_state,
        forward_step_func,
        valid_data_iterator,
        model,
        config,
        verbose=False,
        process_non_loss_data_func=process_non_loss_data_func,
        non_loss_data_func=non_loss_data_func,
        num_warmup_iters=5,
    )

    start_iteration = global_state.train_state.step
    log_rank_0(f"Starting training loop at iteration {start_iteration}")

    # Run training iterations till done.
    while global_state.train_state.step < train_config.train_iters:
        # Handle RPD profiling start
        if rpd and global_state.train_state.step >= rpd_warmup_steps and not rpd_status:
            log_rank_0(f"Starting RPD profiling at step {global_state.train_state.step}")
            rpd.start()
            rpd_status = "running"

        # Handle RPD profiling stop
        if rpd and rpd_status == "running" and global_state.train_state.step >= rpd_warmup_steps + rpd_active_steps:
            log_rank_0(f"Stopping RPD profiling at step {global_state.train_state.step}")
            rpd.stop()
            rpd_status = "finished"

        # Handle profiling for this step
        nvtx_ctx = handle_profiling_step(
            prof_config,
            global_state.train_state.step,
            torch.distributed.get_rank(),
            prof,
        )
        if nvtx_ctx is not None:
            nsys_nvtx_context = nvtx_ctx

        fault_tolerance.on_checkpointing_start(global_state)
        maybe_finalize_async_save(global_state=global_state, ckpt_cfg=config.checkpoint, blocking=False)
        fault_tolerance.on_checkpointing_end(global_state=global_state, is_async_finalization=True)

        # Update the timeout for all process groups after initialization
        # We update the timeout after the first successful iteration,
        # which takes longer than others usually
        if global_state.train_state.step == start_iteration + 1:
            distributed_timeout_seconds_after_init = global_state.cfg.dist.distributed_timeout_seconds_after_init
            if distributed_timeout_seconds_after_init is not None:
                update_pg_timeout(timedelta(seconds=distributed_timeout_seconds_after_init))

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(global_state.train_state.consumed_train_samples, consistency_check=False, verbose=True)
        if get_num_microbatches() != num_microbatches and global_state.train_state.step != 0:
            assert get_num_microbatches() > num_microbatches, (
                f"Number of microbatches should be increasing due to batch size rampup; "
                f"instead going from {num_microbatches} to {get_num_microbatches()}"
            )
            if config.checkpoint.save is not None:
                save_checkpoint_and_time(
                    global_state,
                    model,
                    optimizer,
                    scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context,
                    non_persistent_ckpt=False,  # TODO: implement non-persistent checkpointing
                    train_data_iterator=train_data_iterator,
                )
        num_microbatches = get_num_microbatches()
        update_num_microbatches(global_state.train_state.consumed_train_samples, consistency_check=True, verbose=True)

        # Completely skip iteration if needed.
        if _should_skip_and_handle_iteration(global_state, train_data_iterator, pg_collection):
            continue

        # Capture CUDA Graphs after warmup.
        if (
            model_config.cuda_graph_impl == "transformer_engine"
            and cuda_graph_helper is not None
            and not cuda_graph_helper.graphs_created()
            and global_state.train_state.step - start_iteration == model_config.cuda_graph_warmup_steps
        ):
            if model_config.cuda_graph_warmup_steps > 0 and should_toggle_forward_pre_hook:
                disable_forward_pre_hook(model, param_sync=False)
            cuda_graph_helper.create_cudagraphs()
            if model_config.cuda_graph_warmup_steps > 0 and should_toggle_forward_pre_hook:
                enable_forward_pre_hook(model)
                cuda_graph_helper.cuda_graph_set_manual_hooks()

        # Run training step.
        fault_tolerance.on_training_step_start(global_state)
        (
            loss_dict,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
            log_max_attention_logit,
        ) = train_step(
            wrapped_forward_step_func,
            train_data_iterator,
            model,
            optimizer,
            scheduler,
            global_state,
            pg_collection,
            forward_backward_func,
        )

        fault_tolerance.on_training_step_end(global_state)

        # Advance NVIDIA DLFw Inspect step if enabled
        tensor_inspect_step_if_enabled(config.tensor_inspect)

        if config.logger.log_throughput_to_tensorboard:
            history_wct.append(time.time() - global_state.start_time)

        if should_checkpoint:
            save_checkpoint_and_time(
                global_state,
                model,
                optimizer,
                scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
                non_persistent_ckpt=False,  # TODO: implement non-persistent checkpointing
            )
        if should_exit:
            break

        # Enable forward pre-hooks after first set of forward and backward passes.
        # When running in fp16, skip all NaN iterations until steady-state loss scaling value
        # is reached.
        if global_state.train_state.step == start_iteration:
            if skipped_iter:
                # Only enable forward pre-hook after a training step has successfully run. Relevant
                # for fp16 codepath where first XX iterations are skipped until steady-state loss
                # scale value is reached.
                start_iteration = global_state.train_state.step + 1
            else:
                # Enable forward pre-hook after training step has successfully run. All subsequent
                # forward passes will use the forward pre-hook / `param_sync_func` in
                # `forward_backward_func`.
                if should_toggle_forward_pre_hook:
                    enable_forward_pre_hook(model)
                    model_config.param_sync_func = param_sync_func
                    pre_hook_enabled = True
                    # Set the manual hooks here since it's not set right after the capturing.
                    if (
                        model_config.cuda_graph_impl == "transformer_engine"
                        and model_config.cuda_graph_warmup_steps == 0
                    ):
                        assert cuda_graph_helper.graphs_created(), "CUDA Graphs should have been created."
                        cuda_graph_helper.cuda_graph_set_manual_hooks()

        global_state.train_state.step += 1

        # If fsdp_manual_registration is enabled, manually register FSDP communication buffers after one training step.
        if global_state.train_state.step == start_iteration + 1 and config.ddp.use_megatron_fsdp:
            _maybe_register_fsdp_buffers(config, model)

        dp_size = pg_collection.dp.size()
        batch_size = dp_size * train_config.micro_batch_size * get_num_microbatches()
        global_state.train_state.consumed_train_samples += batch_size
        num_skipped_samples_in_batch = get_current_global_batch_size() - get_current_running_global_batch_size()
        if train_config.decrease_batch_size_if_needed:
            assert num_skipped_samples_in_batch >= 0
        else:
            assert num_skipped_samples_in_batch == 0
        global_state.train_state.skipped_train_samples += num_skipped_samples_in_batch
        num_floating_point_operations_in_batch = flop_utils.num_floating_point_operations(config, batch_size)
        global_state.train_state.floating_point_operations_so_far += num_floating_point_operations_in_batch
        num_floating_point_operations_so_far = global_state.train_state.floating_point_operations_so_far
        num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

        # Logging.
        if hasattr(optimizer, "is_stub_optimizer") and not optimizer.is_stub_optimizer:
            loss_scale = optimizer.get_loss_scale().item()
        else:
            loss_scale = 1.0
        params_norm = None

        if config.logger.log_params_norm:
            params_norm = calc_params_l2_norm(model, model_config, use_megatron_fsdp=config.dist.use_megatron_fsdp)
        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if len(param_group) == 0:
                continue
            if param_group["is_decoupled_lr"]:
                decoupled_learning_rate = param_group["lr"]
            else:
                learning_rate = param_group["lr"]
        report_memory_flag = training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
            config,
            global_state,
            history_wct,
            model,
            log_max_attention_logit,
        )

        if (
            global_state.train_state.do_valid
            and train_config.eval_interval
            and global_state.train_state.step % train_config.eval_interval == 0
        ):
            if energy_monitor is not None:
                energy_monitor.pause()
            timers("interval-time").stop()
            if should_toggle_forward_pre_hook:
                disable_forward_pre_hook(model)
                pre_hook_enabled = False
            if train_config.manual_gc and train_config.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f"iteration {global_state.train_state.step}"
            timers("eval-time", log_level=0).start(barrier=True)
            evaluate_and_print_results_custom(
                global_state,
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                model_config,
                verbose=False,
                write_to_tensorboard=True,
                process_non_loss_data_func=process_non_loss_data_func,
                non_loss_data_func=non_loss_data_func,
            )
            eval_duration += timers("eval-time").elapsed()
            eval_iterations += train_config.eval_iters
            timers("eval-time").stop()

            if train_config.manual_gc and train_config.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if should_toggle_forward_pre_hook:
                enable_forward_pre_hook(model)
                pre_hook_enabled = True
            timers("interval-time", log_level=0).start(barrier=True)
            if energy_monitor is not None:
                energy_monitor.resume()

        # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
        # Some of these only happen at specific iterations.
        maybe_synchronize_training_step(config.train.train_sync_interval, global_state.train_state.step)
        num_floating_point_operations_since_last_log_event = maybe_report_stragglers(
            config.logger.log_interval,
            bool(getattr(config.straggler, "log_straggler", False)),
            straggler_timer,
            global_state.train_state.step,
            num_floating_point_operations_since_last_log_event,
        )
        maybe_check_weight_hash_across_dp_replicas(
            model,
            config.train.check_weight_hash_across_dp_replicas_interval,
            global_state.train_state.step,
            should_toggle_forward_pre_hook,
        )
        handle_profiling_stop(
            config.profiling,
            global_state.train_state.step,
            torch.distributed.get_rank(),
            prof,
            nsys_nvtx_context,
        )
        maybe_run_manual_gc(
            config.train.manual_gc,
            config.train.manual_gc_interval,
            global_state.train_state.step,
        )

        # Checkpoint and decide whether to exit.
        should_exit = checkpoint_and_decide_exit(
            global_state,
            model,
            optimizer,
            scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            train_data_iterator,
        )
        if should_exit:
            break

    _delete_cuda_graphs(cuda_graph_helper)

    # Stop RPD profiler if still running
    if rpd and rpd_status == "running":
        log_rank_0(f"Stopping RPD profiling at training end (step {global_state.train_state.step})")
        rpd.stop()
        rpd_status = "finished"

    # Flush TensorBoard, WandB writers and one-logger.
    writer = global_state.tensorboard_logger
    if writer:
        writer.flush()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    # This will finalize all unfinalized async request and terminate
    # a persistent async worker if persistent ckpt worker is enabled
    fault_tolerance.on_checkpointing_start(global_state)
    maybe_finalize_async_save(global_state=global_state, ckpt_cfg=config.checkpoint, blocking=True, terminate=True)
    fault_tolerance.on_checkpointing_end(global_state=global_state, is_async_finalization=True)

    # Shutdown NVRx straggler detection if enabled
    safe_shutdown_nvrx_straggler_manager(global_state.nvrx_straggler_manager)

    if energy_monitor is not None:
        energy_monitor.lap()
        total_energy = energy_monitor.get_total()
        log_rank_0(f"Total training energy (GPU): {total_energy / 1e6} MJ")
        energy_monitor.shutdown()

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        # Stop RPD profiler if still running before exit
        if rpd and rpd_status == "running":
            log_rank_0(f"Stopping RPD profiling before exit at step {global_state.train_state.step}")
            rpd.stop()
            rpd_status = "finished"

        # Close NVIDIA DLFw Inspect if enabled
        tensor_inspect_end_if_enabled(config.tensor_inspect)
        maybe_finalize_async_save(global_state=global_state, ckpt_cfg=config.checkpoint, blocking=True, terminate=True)
        wandb_writer = global_state.wandb_logger
        if wandb_writer:
            wandb_writer.finish()
        fault_tolerance.shutdown(global_state)
        sys.exit(exit_code)

    # Close NVIDIA DLFw Inspect at clean finish
    tensor_inspect_end_if_enabled(config.tensor_inspect)

from megatron.bridge.training import train
train.train = megatron_bridge_train_override
