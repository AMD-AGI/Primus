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

from primus.backends.megatron_bridge.recipes.mlperf_llama2_70b import _log_suppression  # noqa: F401, E402

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
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator, RerunMode, get_rerun_state_machine
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import check_param_hashes_across_dp_replicas, get_model_config
from megatron.core.full_cuda_graph import FullCudaGraphWrapper

from primus.core.utils.module_utils import log_rank_0 as _orig_log_rank_0, log_rank_last as _orig_log_rank_last

_log_suppression.reapply_quiet_logger_levels()

_verbose_logging = os.environ.get("VERBOSE_TRAINING_LOG", "0") == "1"
if _verbose_logging:
    log_rank_0 = _orig_log_rank_0
    log_rank_last = _orig_log_rank_last
else:
    def log_rank_0(*args, **kwargs): pass
    def log_rank_last(*args, **kwargs): pass

from megatron.bridge.utils import common_utils

# Megatron-Bridge training_log prints iteration / TFLOP/s / loss via print_rank_0
# (plain stdout). Only silence Primus log_rank_0 helpers — not Megatron metrics.
_megatron_print_rank_0 = common_utils.print_rank_0
_megatron_print_rank_last = common_utils.print_rank_last
_megatron_warn_rank_0 = common_utils.warn_rank_0
common_utils.print_rank_0 = _megatron_print_rank_0
common_utils.print_rank_last = _megatron_print_rank_last
if _verbose_logging:
    common_utils.warn_rank_0 = _megatron_warn_rank_0
else:
    common_utils.warn_rank_0 = log_rank_0


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _log_training_gpu_mem(tag: str, memory_keys=None) -> None:
    """Rank-0 CUDA memory snapshot when ``PRIMUS_LOG_GPU_MEM=1``."""
    if not _truthy_env("PRIMUS_LOG_GPU_MEM", default=False):
        return
    try:
        from megatron.bridge.training.utils.train_utils import report_memory

        if not torch.cuda.is_available():
            _orig_log_rank_0(f"[GPU mem] {tag} | CUDA not available")
            return
        torch.cuda.synchronize()
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / (1024**3)
        reserved = torch.cuda.memory_reserved(dev) / (1024**3)
        peak_alloc = torch.cuda.max_memory_allocated(dev) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(dev) / (1024**3)
        _orig_log_rank_0(
            f"[GPU mem] {tag} | "
            f"allocated={alloc:.2f} GiB reserved={reserved:.2f} GiB "
            f"max_alloc={peak_alloc:.2f} GiB max_reserved={peak_reserved:.2f} GiB"
        )
        mem = report_memory(memory_keys)
        if mem:
            detail = " | ".join(f"{k}={v}" for k, v in sorted(mem.items()))
            _orig_log_rank_0(f"[GPU mem detail] {tag} | {detail}")
    except Exception as exc:
        _orig_log_rank_0(f"[GPU mem] {tag} | failed ({type(exc).__name__}: {exc})")


from megatron.bridge import AutoBridge
from megatron.bridge.data.finetuning import prepare_finetuning_batch
from megatron.bridge.data.iterator_utils import make_data_iterator_list
from primus.backends.megatron_bridge.patches.mlperf_llama2_70b.lora import LoRA
from primus.backends.megatron_bridge.patches.mlperf_llama2_70b.resettable_data_iterator import (
    ResettableDataIterator,
)
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.recipes.utils.finetune_utils import default_squad_config
from megatron.bridge.peft.base import PEFT
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
    ProfilingConfig,
    RerunStateMachineConfig,
    RNGConfig,
    StragglerDetectionConfig,
    TensorInspectConfig,
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
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.tensor_inspect import (
    tensor_inspect_end_if_enabled,
    tensor_inspect_step_if_enabled,
)
from megatron.bridge.training.utils import flop_utils
from megatron.bridge.training.utils import train_utils as _megatron_train_utils
from megatron.bridge.training.utils.train_utils import (
    calc_params_l2_norm,
    prepare_forward_step_func,
    training_log,
)

# train_utils binds print_rank_* at import time; rebind so training_log always uses Primus loggers even if
# this module was imported after an earlier train_utils load.
_megatron_train_utils.print_rank_0 = log_rank_0
_megatron_train_utils.print_rank_last = log_rank_last
from megatron.bridge.utils.common_utils import is_last_rank
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

# Importing this module installs a monkey-patch that swaps Megatron-Bridge's
from primus.backends.megatron_bridge.recipes.mlperf_llama2_70b import nemo_loss as _nemo_loss  # noqa: F401

MLPERF_TARGET_LOSS = 0.925

# Sticky flag: once ``evaluate_and_print_results_custom`` observes an lm-loss
# value strictly below ``MLPERF_TARGET_LOSS`` we flip this to True. Subsequent
# calls to ``evaluate_and_print_results_custom`` / ``warmup_eval`` become
# no-ops that return ``(should_exit=True, None)`` without running another full
# validation pass. Prevents redundant evals from a resumed loop or any outer
# re-entry path after the target is met.
_TARGET_LOSS_REACHED: bool = False

# ---------------------------------------------------------------------------
# MLPerf logging singleton (initialised lazily in the recipe config function)
# ---------------------------------------------------------------------------
_sft_logger = None

def _get_sft_logger():
    return _sft_logger

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


@register
def bf16_with_mxfp4_mixed() -> MixedPrecisionConfig:
    """BF16 + MXFP4 (e2m1) mixed precision, registered here so it resolves when
    ``precision_config="bf16_with_mxfp4_mixed"`` is passed from YAML before
    ``runtime_config_update`` runs.
    """
    cfg = bf16_mixed()
    cfg.fp8 = None
    cfg.fp4 = "e2m1"
    cfg.fp4_recipe = "mxfp4"
    cfg.fp8_recipe = "delayed"
    cfg.fp8_amax_history_len = 4
    cfg.fp8_amax_compute_algo = "most_recent"
    cfg.fp8_reduce_amax = False
    cfg.fp8_interval = 1
    cfg.fp8_margin = 0
    cfg.fp8_dot_product_attention = False
    cfg.fp8_param_gather = False
    cfg.grad_reduce_in_fp32 = False
    return cfg


class Timer:
    def __init__(self, gbs):
        self.start_time = None
        self.stop_time = None
        self.elapsed_time = 0
        self.samples = 0
        self.gbs = gbs
        self.consumed_samples = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        self.samples += self.gbs
        self.consumed_samples += self.gbs
        self.elapsed_time += self.stop_time - self.start_time

    def get_throughput(self):
        throughput = self.samples / self.elapsed_time
        self.samples = 0
        self.elapsed_time = 0
        return throughput


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
    eval_iters: int = 22
    clip_grad: float = 0.3
    pretrained_checkpoint: str | None
    packed_sequence: bool
    packed_train_data_path: str | None
    packed_val_data_path: str | None
    packed_metadata_path: str | None
    dataset_type: str
    seed: int
    check_for_nan_in_loss: bool
    te_fused_lora_include_modules: Optional[List[str]]
    te_fused_lora_exclude_modules: Optional[List[str]]
    # Optional MXFP4-phase activation recompute (left None outside the MXFP4 recipe).
    recompute_granularity: Optional[str]
    recompute_method: Optional[str]
    recompute_num_layers: Optional[int]
    # TE attention backend ("flash", "fused", "unfused", "local", "auto"). None keeps Megatron's default.
    attention_backend: Optional[str]


def llama2_70b_lora_config(**user_kwargs: Unpack[Llama2CustomKwargs]) -> ConfigContainer:
    """
    Return a custom pre-training config for Llama-2 70B.

    This is a custom variant that can be modified without changing Megatron-Bridge code.
    See `_llama2_lora` for the full list of parameters.
    """
    recommended_kwargs: Llama2CustomKwargs = {
        "hf_path": "meta-llama/Llama-2-70b-hf",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "train_iters": 1000,
        "global_batch_size": 8,
        "micro_batch_size": 1,
        "eval_interval": 48,
        "eval_iters": 22,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "weight_decay": 0.0001,
        "clip_grad": 0.3,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Llama2CustomKwargs = {**recommended_kwargs, **user_kwargs}
    cfg = _llama2_lora(**combined_kwargs)

    # --- MLPerf logging: init phase ---
    if os.getenv("ENABLE_MLLOG", "0") == "1":
        global _sft_logger
        try:
            from primus_mllog import MLPerfSFTLogger

            kw = combined_kwargs
            gbs = kw.get("global_batch_size", 8)
            mbs = kw.get("micro_batch_size", 1)
            _sft_logger = MLPerfSFTLogger(
                global_batch_size=gbs, micro_batch_size=mbs,
            )
            _sft_logger.log_cache_clear_and_init_start()

            data_root = os.getenv("DATA_PATH", "/data")
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            tp = kw.get("tensor_model_parallel_size", 1)
            pp = kw.get("pipeline_model_parallel_size", 1)
            dp_size = world_size // (tp * pp)
            init_cfg = MLPerfSFTLogger.extract_sft_configs(
                train_gbs=gbs,
                train_mbs=mbs,
                train_iters=kw.get("train_iters", 1000),
                eval_iters=kw.get("eval_iters", 22),
                seq_length=kw.get("seq_length", 8192),
                seed=kw.get("seed", int(os.getenv("SEED", "1234"))),
                lr=kw.get("lr", float(os.getenv("LR", "0.0004"))),
                weight_decay=kw.get("weight_decay", 0.0001),
                clip_grad=kw.get("clip_grad", 0.3),
                lr_warmup_iters=kw.get("lr_warmup_iters", 0),
                adam_beta1=kw.get("adam_beta1", 0.9),
                adam_beta2=kw.get("adam_beta2", 0.999),
                adam_eps=kw.get("adam_eps", 1e-8),
                lora_rank=16,
                lora_alpha=32,
                data_root=data_root,
                data_parallel_size=dp_size,
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                context_parallel_size=int(os.getenv("MLLOG_CONTEXT_PARALLELISM", "1")),
                config_filename=os.getenv("MLLOG_CONFIG_FILENAME", ""),
                lowest_numerical_precision_linear=os.getenv("MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR", "mxfp4"),

            )
            _sft_logger.log_init_params(init_cfg)
        except ImportError:
            _orig_log_rank_0("primus_mllog not installed — MLPerf logging disabled")
            _sft_logger = None
        except Exception as exc:
            _orig_log_rank_0(f"MLPerf logging init failed ({type(exc).__name__}: {exc}) — disabled")
            _sft_logger = None

    return cfg


def llama2_70b_lora_mxfp4_config(**user_kwargs: Unpack[Llama2CustomKwargs]) -> ConfigContainer:
    """Llama-2 70B LoRA with MXFP4 mixed precision (``bf16_with_mxfp4_mixed``).

    Matches MLPerf 6.0 NeMo MI355X FP4 defaults: full/block recompute over 8
    layers; FusedAttention backend (CK / AOTriton on ROCm) instead of
    FlashAttention (which is not validated for the MXFP4 path on MI355X).
    All defaults are user-overridable via ``user_kwargs``.
    """
    mxfp4_defaults: Llama2CustomKwargs = {
        "precision_config": "bf16_with_mxfp4_mixed",
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "attention_backend": "fused",
    }
    combined_kwargs: Llama2CustomKwargs = {**mxfp4_defaults, **user_kwargs}
    return llama2_70b_lora_config(**combined_kwargs)


def _llama2_lora(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    # Training hyperparameters
    train_iters: int = 1000,
    global_batch_size: int = 8,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    lr: float = 4e-4,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 0,
    lr_decay_iters: Optional[int] = None,
    eval_interval: int = 48,
    eval_iters: int = 22,
    use_null_tokenizer: bool = False,
    pretrained_checkpoint: str | None = None,
    packed_sequence: bool = False,
    packed_train_data_path: str | None = None,
    packed_val_data_path: str | None = None,
    packed_metadata_path: str | None = None,
    dataset_type: str = "mlperf_dataset",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0001,
    clip_grad: float = 0.3,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_with_mxfp4_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    seed: int = 1234,
    check_for_nan_in_loss: bool = False,
    te_fused_lora_include_modules: Optional[List[str]] = None,
    te_fused_lora_exclude_modules: Optional[List[str]] = None,
    recompute_granularity: Optional[str] = None,
    recompute_method: Optional[str] = None,
    recompute_num_layers: Optional[int] = None,
    attention_backend: Optional[str] = None,
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
        dataset_type (str): Dataset type to use. Either "squad" (default) or "mlperf_dataset".
        use_transformer_engine_op_fuser (bool): If True, set ``model_cfg.use_transformer_engine_op_fuser``
            (TE op-fuser path on the backbone, e.g. fused MLP). Set False if LM/validation loss diverges
            from a known-good run.
        stable_lora_with_te_op_fuser (bool): Single Primus knob for the **stable LoRA + op fuser** combo.
            If True (default): backbone follows ``use_transformer_engine_op_fuser``, but LoRA always
            uses unfused :class:`LoRALinear` (``use_te_fused_lora=False``) so loss matches the safe path.
            If False: **legacy** behavior — LoRA uses ``use_te_fused_lora = use_transformer_engine_op_fuser``
            (when TP=1, fused :class:`TEFusedLoRALinear` tracks backbone op fuser, as in older Bridge).
        te_fused_lora_include_modules (Optional[List[str]]): Passed to :class:`LoRA`; only applies when
            fused LoRA is enabled (``stable_lora_with_te_op_fuser=False`` and backbone op fuser on).
        te_fused_lora_exclude_modules (Optional[List[str]]): Passed to :class:`LoRA`; same scope as include.

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf")
    bridge = AutoBridge.from_hf_config(config)
    model_cfg = bridge.to_megatron_provider(load_weights=False) #GPTProvider
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = None
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
    # Fusions / CE: TE parallel cross-entropy; grad-acc fusion (not used with Megatron FSDP).
    model_cfg.cross_entropy_loss_fusion = False
    model_cfg.cross_entropy_fusion_impl = "native"
    model_cfg.gradient_accumulation_fusion = False
    model_cfg.bias_dropout_fusion = True
    model_cfg.fused_single_qkv_rope = False
    model_cfg.apply_rope_fusion = True
    model_cfg.use_transformer_engine_op_fuser = False
    # Activation offload hint for TP paths (distinct from cpu_offloading / cpu_offloading_num_layers).
    model_cfg.cpu_offloading_activations = True

    # MXFP4 weights (fp4/e2m1); fp8=None during MXFP4 phase. FP8_* env vars in
    # config_MI355X_1x8x1.sh configure healing (step 340) and TE delayed scaling,
    # not Megatron model_cfg.fp8 (which stays None until healing).
    model_cfg.fp8 = None
    model_cfg.fp4 = "e2m1"
    model_cfg.fp4_recipe = "mxfp4"
    model_cfg.fp8_param_gather = False
    model_cfg.grad_reduce_in_fp32 = False

    # Used when cuda_graph_impl is not "none" (harmless when graphs are disabled).
    model_cfg.cuda_graph_retain_backward_graph = True
    model_cfg.cuda_graph_use_single_mempool = True
    model_cfg.fp8_recipe = "delayed"
    model_cfg.fp8_amax_history_len = 4
    model_cfg.fp8_amax_compute_algo = "most_recent"
    model_cfg.fp8_dot_product_attention = False
    model_cfg.disable_parameter_transpose_cache = False
    model_cfg.fine_grained_activation_offloading = False
    model_cfg.use_transformer_engine_full_layer_spec = False # Doesn't work beacuse of RMSNorm is not supported in FusedLayerNorm
    model_cfg.cpu_offloading = False
    model_cfg.cpu_offloading_num_layers = 0
    model_cfg.empty_unused_memory_level = 0 # 0: No empty, 1: Empty at end of eval, 2: Empty at end of eval and train.
    # Optional MXFP4-style activation recompute (left untouched when kwargs are None).
    if recompute_granularity is not None:
        model_cfg.recompute_granularity = recompute_granularity
    if recompute_method is not None:
        model_cfg.recompute_method = recompute_method
    if recompute_num_layers is not None:
        model_cfg.recompute_num_layers = recompute_num_layers
    # Pin TE's attention backend ("fused" forces CK / AOTriton on ROCm for the MXFP4 path).
    # Leave as None to keep Megatron's "auto" default (which picks FlashAttention on ROCm).
    if attention_backend is not None:
        try:
            model_cfg.attention_backend = AttnBackend[attention_backend]
        except KeyError as e:
            raise ValueError(
                f"Unknown attention_backend {attention_backend!r}; expected one of "
                f"{[b.name for b in AttnBackend]}."
            ) from e
    # Disable attention QK clipping / max-logit scans in the optimizer path (extra GPU work per step).
    if hasattr(model_cfg, "qk_clip"):
        model_cfg.qk_clip = False
    if hasattr(model_cfg, "log_max_attention_logit"):
        model_cfg.log_max_attention_logit = False

    from megatron.bridge.training.config import OptimizerConfig, SchedulerConfig
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=lr,
        min_lr=min_lr,
        clip_grad=clip_grad,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        overlap_param_gather_with_optimizer_step=True,
        barrier_with_L1_time=False,
        log_num_zeros_in_grad=False,
    )

    scheduler = SchedulerConfig(
        lr_decay_style="cosine",
        lr_decay_iters=lr_decay_iters,  # same as max_steps in nemo
        lr_warmup_iters=lr_warmup_iters, # value 0 same as nemo
        lr_warmup_init=0.0,
        start_weight_decay=weight_decay,
        end_weight_decay=weight_decay, 
        weight_decay_incr_style="constant",
        override_opt_param_scheduler=True,
    )

    peft_config = LoRA(
        dim=16,
        alpha=32,
        dropout=0.1,
        dropout_position="pre",
        lora_A_init_method="xavier",
        lora_B_init_method="zero",
        a2a_experimental=True,
        target_modules=["linear_qkv", "linear_proj"],
        use_te_fused_lora=False,
        te_fused_lora_include_modules=te_fused_lora_include_modules,
        te_fused_lora_exclude_modules=(
            te_fused_lora_exclude_modules if te_fused_lora_exclude_modules is not None else []
        ),
    )

    # Dataset configuration - switch between squad and mlperf_dataset
    if dataset_type == "squad":
        dataset_cfg = default_squad_config(seq_length, packed_sequence)
    elif dataset_type == "mlperf_dataset":
        if packed_sequence:
            packed_sequence_specs = PackedSequenceSpecs(
                packed_sequence_size=seq_length,
                tokenizer_model_name=hf_path,
                packed_train_data_path=packed_train_data_path or "/data/train.npy",
                packed_val_data_path=packed_val_data_path or "/data/validation.npy",
                packed_metadata_path=packed_metadata_path or "/data/packed_metadata.jsonl",
            )
        else:
            packed_sequence_specs = None
        dataset_cfg = FinetuningDatasetConfig(
            dataset_root="/data",
            seq_length=seq_length,
            seed=seed,
            packed_sequence_specs=packed_sequence_specs,
            data_sharding=True,
            dataloader_type="batch",
            num_workers=1,
            do_test=False,
            do_validation=True,
            dataset_kwargs={"return_cu_seqlen": False},
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}. Expected 'squad' or 'mlperf_dataset'.")

    dataset_cfg.num_workers = 0
    dataset_cfg.memmap_workers = 1 # needs to be 1>0
    dataset_cfg.pin_memory = True
    dataset_cfg.persistent_workers = False
    dataset_cfg.dataloader_type = "batch"

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            # Manual GC aligns collections across ranks but adds periodic host pauses; use default GC for best step time.
            manual_gc=False,
            manual_gc_interval=0,
            manual_gc_eval=False,
            empty_unused_memory_level=0, # 0: No empty, 1: Empty at end of eval, 2: Empty at end of eval and train.
            train_sync_interval=None,
            check_weight_hash_across_dp_replicas_interval=None,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            # Per-step NaN grad scan + sync; disable for throughput when loss NaN checks are off.
            check_for_nan_in_grad=False,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=False,
            use_distributed_optimizer=True,
            # gradient_reduce_div_fusion=True,
            # pad_buckets_for_high_nccl_busbw=True,
            use_megatron_fsdp=False,
             keep_fp8_transpose_cache=(
                os.getenv("ENABLE_TRANSPOSE_CACHE", "").strip().lower()
                in ("1", "true", "yes", "on")
            ),
            fp8_param_gather=False,
        ),
        dataset=dataset_cfg,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=None,
            # Per-step / log-interval overhead toggles (keep off for throughput).
            log_params_norm=False,
            log_throughput=True,
            log_energy=False,
            log_progress=False,
            timing_log_level=0,
            log_loss_scale_to_tensorboard=False,
            log_timers_to_tensorboard=False,
            log_throughput_to_tensorboard=False,
            log_validation_ppl_to_tensorboard=False,
            log_memory_to_tensorboard=False,
            log_runtime_to_tensorboard=False,
            log_world_size_to_tensorboard=False,
            log_l2_norm_grad_to_tensorboard=False,
            wandb_project=None,
            wandb_exp_name=None,
            wandb_save_dir=None,
            wandb_entity=None,
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
            load_main_params_from_ckpt=False,
            load_optim=False,
            load_rng=False,
        ),
        rng=RNGConfig(seed=seed, te_rng_tracker=True),
        rerun_state_machine=RerunStateMachineConfig(
            check_for_nan_in_loss=check_for_nan_in_loss,
        ),
        straggler=StragglerDetectionConfig(log_straggler=False),
        tensor_inspect=TensorInspectConfig(enabled=False),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
        peft=peft_config,
        profiling=ProfilingConfig(
            use_pytorch_profiler=False,
            profile_step_start=140,
            profile_step_end=144,
            profile_ranks=list(range(8)),  # 1 node × 8 GPUs
            record_shapes=False,
            nvtx_ranges=False,
        ),
    )

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
        )

    return cfg

def _reset_data_iterator(data_iterator):
    """Reset data iterator to the beginning for deterministic evaluation.
    Traverses through RerunDataIterator wrappers to find and reset any
    ResettableDataIterator, ensuring evaluation always starts from the
    same data point.
    """
    if data_iterator is None:
        return
    if isinstance(data_iterator, list):
        for it in data_iterator:
            _reset_data_iterator(it)
        return
    if isinstance(data_iterator, RerunDataIterator):
        inner = data_iterator.iterable
        if isinstance(inner, ResettableDataIterator):
            inner.reset()
            data_iterator.saved_microbatches.clear()
            data_iterator.replaying = False
            data_iterator.replay_pos = 0
    elif isinstance(data_iterator, ResettableDataIterator):
        data_iterator.reset()
def evaluate(
    state: GlobalState,
    forward_step_func: ForwardStepCallable,
    data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    model: list[MegatronModule],
    process_non_loss_data_func: Optional[Callable],
    config: ConfigContainer,
    verbose: bool = False,
    non_loss_data_func: Optional[Callable] = None,
) -> tuple[Optional[dict[str, torch.Tensor]], Optional[Any], bool]:
    """Evaluation function (from eval_m.py).
    Validation loss aggregation matches NeMo: mean of per-microbatch means
    (same as NeMo MaskedTokenLossReduction with validation_step=True, val_drop_last=True).
    """
    _reset_data_iterator(data_iterator)
    wrapped_forward_step = prepare_forward_step_func(forward_step_func, state)
    timers = state.timers
    timers("evaluate", log_level=0).start(barrier=True)
    for model_module in model:
        model_module.eval()
    pg_collection = get_pg_collection(model)
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)
    total_loss_dict = {}
    eval_batch_size = state.cfg.train.global_batch_size
    eval_num_microbatches = eval_batch_size // (
        state.cfg.train.micro_batch_size * state.cfg.data_parallel_size
    )
    with torch.no_grad():
        if verbose:
            log_rank_0(f"Evaluating on {state.cfg.train.eval_iters * eval_batch_size} samples")
        if state.cfg.model.cuda_graph_impl == "local" and "full_iteration" in state.cfg.model.cuda_graph_scope:
            forward_backward_func = FullCudaGraphWrapper(
                get_forward_backward_func(),
                cuda_graph_warmup_steps=state.cfg.model.cuda_graph_warmup_steps,
            )
        else:
            forward_backward_func = get_forward_backward_func()
        iteration = 0
        while iteration < state.cfg.train.eval_iters:
            iteration += 1
            if verbose:
                log_rank_0(f"Evaluating iter {iteration}/{state.cfg.train.eval_iters}")
            seq_length = state.cfg.model.seq_length
            eval_data_iterator = data_iterator
            if state.cfg.dataset.dataloader_type == "batch":
                eval_microbatch_iterator, seq_length = prepare_finetuning_batch(
                    data_iterator=data_iterator,
                    num_microbatches=eval_num_microbatches,
                    default_seq_length=state.cfg.model.seq_length,
                    seq_key="tokens",
                )
                eval_data_iterator = make_data_iterator_list(
                    model=model,
                    data_iterator=eval_microbatch_iterator,
                )
            config.timers = None
            fault_tolerance.on_eval_step_start(state)
            loss_dicts = forward_backward_func(
                forward_step_func=wrapped_forward_step,
                data_iterator=eval_data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=seq_length,
                micro_batch_size=state.cfg.train.micro_batch_size,
                forward_only=True,
            )
            fault_tolerance.on_eval_step_end(state)
            config.timers = state.timers
            if is_pp_last_stage(pg_collection.pp):
                for key in loss_dicts[0].keys():
                    if key not in total_loss_dict:
                        total_loss_dict[key] = torch.zeros(2, dtype=torch.float32, device="cuda")
                    val = [x[key].reshape(-1) for x in loss_dicts]
                    if val[0].numel() == 2:
                        # NeMo MLPerf-equivalent: accumulate raw [sum, count] across micros & eval iters.
                        # Per-micro [sum, count] is already DP/CP-reduced inside
                        # nemo_loss.MaskedTokenLossReduction.forward, so no extra all_reduce here.
                        per_iter = torch.vstack([v.float() for v in val]).sum(dim=0)   # [Σ_micro sum, Σ_micro count]
                        total_loss_dict[key] += per_iter
                    elif val[0].numel() == 1:
                        # legacy single-scalar branch 
                        micro_sum = torch.stack([v.float() for v in val], dim=0).sum()
                        total_loss_dict[key][0] += micro_sum
                        total_loss_dict[key][1] += float(len(loss_dicts))
                    else:
                        raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")

            state.train_state.consumed_valid_samples += eval_batch_size
            if state.cfg.train.exit_duration_in_mins:
                train_time = (time.time() - state.start_time) / 60.0
                done_cuda = torch.tensor(
                    [train_time > state.cfg.train.exit_duration_in_mins],
                    dtype=torch.int,
                    device="cuda",
                )
                torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = bool(done_cuda.item())
                if done:
                    rerun_state_machine.set_mode(rerun_mode)
                    log_rank_0("Exiting during evaluation, timelimit reached")
                    return None, None, True
        
        collected_non_loss_data = None
        if non_loss_data_func is not None:
            collected_non_loss_data = non_loss_data_func(model)
        elif process_non_loss_data_func is not None and is_last_rank():
            non_loss_data_iterator = data_iterator
            non_loss_seq_length = state.cfg.model.seq_length
            if state.cfg.dataset.dataloader_type == "batch":
                non_loss_microbatch_iterator, non_loss_seq_length = prepare_finetuning_batch(
                    data_iterator=data_iterator,
                    num_microbatches=get_num_microbatches(),
                    default_seq_length=state.cfg.model.seq_length,
                    seq_key="tokens",
                )
                non_loss_data_iterator = make_data_iterator_list(
                    model=model,
                    data_iterator=non_loss_microbatch_iterator,
                )
            collected_non_loss_data = forward_backward_func(
                forward_step_func=wrapped_forward_step,
                data_iterator=non_loss_data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=non_loss_seq_length,
                micro_batch_size=state.cfg.train.micro_batch_size,
                forward_only=True,
                collect_non_loss_data=True,
            )
    for model_module in model:
        model_module.train()
    for key in total_loss_dict:
        val_loss_sum, val_loss_count = total_loss_dict[key]
        if val_loss_count > 0:
            total_loss_dict[key] = val_loss_sum / val_loss_count
        else:
            total_loss_dict[key] = val_loss_sum
    timers("evaluate").stop()
    timers.log(["evaluate"])
    rerun_state_machine.set_mode(rerun_mode)
    return total_loss_dict, collected_non_loss_data, False


from megatron.bridge.training import eval 
def evaluate_and_print_results_custom(
    state: GlobalState,
    prefix: str,
    forward_step_func: ForwardStepCallable,
    data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    model: list[MegatronModule],
    config: ConfigContainer,
    verbose: bool = False,
    write_to_tensorboard: bool = False,
    process_non_loss_data_func: Optional[Callable] = None,
    non_loss_data_func: Optional[Callable] = None,
    throughput: float = 0.0, # samples/sec
) -> tuple:
    """Helper function to evaluate and dump results on screen.

    Args:
        state (GlobalState): The global state object.
        prefix (str): Prefix for logging evaluation results.
        forward_step_func (Callable): The function that performs a forward step.
        data_iterator (Optional[Union[RerunDataIterator, list[RerunDataIterator]]]): Iterator over evaluation data.
        model (list[MegatronModule]): list of model chunks.
        config (ConfigContainer): Configuration container (potentially redundant).
        verbose (bool, optional): Whether to print evaluation progress. Defaults to False.
        write_to_tensorboard (bool, optional): Whether to write results to TensorBoard. Defaults to False.
        process_non_loss_data_func (Optional[Callable], optional): Function to process non-loss data. Defaults to None.
        non_loss_data_func (Optional[Callable], optional): Function to compute non-loss data. Defaults to None.
    """
    global _TARGET_LOSS_REACHED
    if _TARGET_LOSS_REACHED:
        # Target already hit on a previous pass; skip this entire eval.
        # Returning should_exit=True keeps the outer while-loop on its exit path.
        _orig_log_rank_0(
            f"Skipping evaluation at {prefix}: target loss < "
            f"{MLPERF_TARGET_LOSS} already reached."
        )
        return True, None

    log_rank_0(f"Evaluating and printing results at {prefix}")
    should_exit = False
    def is_last_rank():
        return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)
    import math
    if write_to_tensorboard:
        writer = state.tensorboard_logger
    else:
        writer = None

    wandb_writer = state.wandb_logger
        
    eval_start = time.time()
    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        state, forward_step_func, data_iterator, model, process_non_loss_data_func, config, verbose, non_loss_data_func
    )
    eval_duration = time.time() - eval_start
    # Timelimit hit during evaluation
    if timelimit:
        return False, None
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

    string += "throughput: {:.6E} | ".format(throughput)
    string += "eval duration: {:.6E} | ".format(eval_duration)
    if writer:
        writer.add_scalar("throughput samples/sec", throughput, state.train_state.step)
        writer.add_scalar("throughput samples/sec vs samples", throughput, state.train_state.consumed_train_samples)

    if wandb_writer and is_last_rank():
        wandb_writer.log({"throughput samples/sec": throughput}, state.train_state.step)

    length = len(string) + 1
    # Match training logs: emit on rank 0 so validation lines show up in typical Primus rank-0 log streams.
    log_rank_0("-" * length)
    log_rank_0(string)
    log_rank_0("-" * length)
    # Guard against non-PP-last ranks (or otherwise empty total_loss_dict) where
    # "lm loss" may not be populated; only run the MLPerf early-exit check when
    # the key is present.
    eval_loss_value = None
    if total_loss_dict and "lm loss" in total_loss_dict:
        eval_loss_value = total_loss_dict['lm loss'].item()
        if eval_loss_value < MLPERF_TARGET_LOSS:
            should_exit = True
            _TARGET_LOSS_REACHED = True
            log_rank_0(f"Validation loss is less than {MLPERF_TARGET_LOSS}, exiting training")
    return should_exit, eval_loss_value

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
        evaluate_and_print_results_custom(
            state,
            f"warmup iteration {i}",
            forward_step_func,
            data_iterator,
            model,
            config,
            verbose=verbose,
            write_to_tensorboard=False,
            process_non_loss_data_func=process_non_loss_data_func,
            non_loss_data_func=non_loss_data_func,
        )
        log_rank_0(f"Warmup eval iteration {i} completed")
    log_rank_0(f"Warmup eval completed")

class _SyntheticSFTDataIterator:
    """Infinite iterator yielding synthetic finetuning batches for warmup.

    Produces random token tensors matching the SFT packed-sequence shape expected
    by prepare_finetuning_batch / forward_backward_func.
    """

    def __init__(self, seq_length: int, micro_batch_size: int, vocab_size: int = 32000):
        self._seq_length = seq_length
        self._mbs = micro_batch_size
        self._vocab_size = vocab_size

    def __iter__(self):
        return self

    def __next__(self):
        sl = self._seq_length
        mbs = self._mbs
        tokens = torch.randint(0, self._vocab_size, (mbs, sl), dtype=torch.int64, device="cuda")
        labels = torch.randint(0, self._vocab_size, (mbs, sl), dtype=torch.int64, device="cuda")
        loss_mask = torch.ones(mbs, sl, dtype=torch.float32, device="cuda")
        position_ids = torch.arange(sl, dtype=torch.int64, device="cuda").unsqueeze(0).expand(mbs, -1)
        return {"tokens": tokens, "labels": labels, "loss_mask": loss_mask, "position_ids": position_ids}


def run_synthetic_warmup(
    forward_step_func: ForwardStepCallable,
    model: list[MegatronModule],
    optimizer: MegatronOptimizer,
    scheduler: OptimizerParamScheduler,
    global_state: GlobalState,
    pg_collection: ProcessGroupCollection,
) -> None:
    """Run synthetic warmup steps to pre-compile JIT kernels before RUN_START.

    Mirrors NeMo's warmup behavior: runs N training (fwd+bwd+opt) steps and
    M validation (fwd-only) steps with random data, then restores all state so
    that real training starts from a clean slate.

    Controlled by environment variables:
        SYNTH_WARMUP_STEPS       - number of training warmup steps (default 5, 0 disables)
        SYNTH_WARMUP_VALID_STEPS - number of validation warmup steps (default 5, 0 disables)
    """
    warmup_train_steps = int(os.getenv("SYNTH_WARMUP_STEPS", "5"))
    warmup_valid_steps = int(os.getenv("SYNTH_WARMUP_VALID_STEPS", "5"))

    if warmup_train_steps <= 0 and warmup_valid_steps <= 0:
        return

    config = global_state.cfg
    seq_length = config.model.seq_length
    mbs = config.train.micro_batch_size
    gbs = config.train.global_batch_size
    dp_size = getattr(config, "data_parallel_size", 1)
    num_microbatches = gbs // (mbs * dp_size)

    synth_iter = _SyntheticSFTDataIterator(seq_length, mbs)
    wrapped_forward_step = prepare_forward_step_func(forward_step_func, global_state)
    forward_backward_func = get_forward_backward_func()

    # --- Save model parameters ---
    models = model if isinstance(model, (list, tuple)) else [model]
    saved_params = {}
    for m in models:
        for name, p in m.named_parameters():
            if p.requires_grad:
                saved_params[(id(m), name)] = p.data.to("cpu", copy=True)

    # --- Save optimizer state (neuter to prevent real updates) ---
    saved_opt_states = []
    inner_opts = []
    if hasattr(optimizer, "chained_optimizers"):
        for sub in optimizer.chained_optimizers:
            inner_opts.append(getattr(sub, "optimizer", sub))
    else:
        inner_opts.append(getattr(optimizer, "optimizer", optimizer))

    for inner in inner_opts:
        saved = []
        for group in inner.param_groups:
            state = {}
            for key in ("betas", "weight_decay", "bias_correction"):
                if key in group:
                    state[key] = group[key]
            saved.append(state)
            if "betas" in group:
                group["betas"] = [1.0, 1.0]
            if "weight_decay" in group:
                group["weight_decay"] = 0.0
            if "bias_correction" in group:
                group["bias_correction"] = False
        saved_opt_states.append(saved)

    # --- Save LR scheduler state ---
    saved_sched = {}
    if scheduler is not None:
        for k in ("num_steps", "num_floating_point_operations_so_far"):
            if hasattr(scheduler, k):
                saved_sched[k] = getattr(scheduler, k)

    # --- Training warmup steps (fwd + bwd + optimizer) ---
    if warmup_train_steps > 0:
        for m in models:
            m.train()
        for step in range(1, warmup_train_steps + 1):
            train_step(
                wrapped_forward_step,
                synth_iter,
                model,
                optimizer,
                scheduler,
                global_state,
                pg_collection,
                forward_backward_func,
            )
            torch.cuda.synchronize()

    # --- Validation warmup steps (forward-only) ---
    if warmup_valid_steps > 0:
        for m in models:
            m.eval()
        with torch.no_grad():
            for step in range(1, warmup_valid_steps + 1):
                forward_backward_func(
                    forward_step_func=wrapped_forward_step,
                    data_iterator=synth_iter,
                    model=model,
                    num_microbatches=num_microbatches,
                    seq_length=seq_length,
                    micro_batch_size=mbs,
                    forward_only=True,
                )
                torch.cuda.synchronize()
        for m in models:
            m.train()

    # --- Restore model parameters ---
    for m in models:
        for name, p in m.named_parameters():
            key = (id(m), name)
            if key in saved_params:
                p.data.copy_(saved_params[key].to(p.device))
    del saved_params

    # --- Restore optimizer state ---
    for inner, saved in zip(inner_opts, saved_opt_states):
        for group, state in zip(inner.param_groups, saved):
            for key, val in state.items():
                group[key] = val

    if hasattr(optimizer, "reload_model_params"):
        optimizer.reload_model_params()

    # --- Zero optimizer state tensors (exp_avg / exp_avg_sq) ---
    for inner in inner_opts:
        for param_states in inner.state.values():
            for k, v in param_states.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    v.zero_()

    # --- Restore LR scheduler and re-sync param_groups['lr'] ---
    if scheduler is not None:
        for k, v in saved_sched.items():
            setattr(scheduler, k, v)
        scheduler.step(0)

    # --- Reset FP8 state ---
    for m in models:
        for module in m.modules():
            if hasattr(module, "fp8_initialized"):
                module.fp8_initialized = False
                if hasattr(module, "reset_fp8_meta_tensors"):
                    try:
                        module.reset_fp8_meta_tensors()
                    except Exception:
                        pass

    # --- Seed FP8 amax_history to prevent scale=inf after reset ---
    for m in models:
        for module in m.modules():
            fp8_meta = getattr(module, "fp8_meta", None)
            if fp8_meta is None or not isinstance(fp8_meta, dict):
                continue
            for key in ("scaling_fwd", "scaling_bwd"):
                if key not in fp8_meta:
                    continue
                tensor_meta = fp8_meta[key]
                if hasattr(tensor_meta, "amax_history"):
                    tensor_meta.amax_history.fill_(1.0)

    # --- Reset FP4/MXFP4 state (block scales, quantizer caches) ---
    for m in models:
        for module in m.modules():
            is_fp4 = bool(getattr(module, "fp4", False)) or hasattr(module, "fp4_initialized")
            if not is_fp4:
                fp8_meta = getattr(module, "fp8_meta", None)
                if isinstance(fp8_meta, dict):
                    for key in ("scaling_fwd", "scaling_bwd"):
                        tm = fp8_meta.get(key)
                        if tm is not None and ("FP4" in type(tm).__name__ or "Fp4" in type(tm).__name__):
                            is_fp4 = True
                            break
            if not is_fp4:
                continue
            if hasattr(module, "fp4_initialized"):
                module.fp4_initialized = False
            if hasattr(module, "fp8_initialized"):
                module.fp8_initialized = False
            if hasattr(module, "reset_fp4_meta_tensors"):
                try:
                    module.reset_fp4_meta_tensors()
                except Exception:
                    pass
            elif hasattr(module, "reset_fp8_meta_tensors"):
                try:
                    module.reset_fp8_meta_tensors()
                except Exception:
                    pass

    # --- Clear gradients ---
    for m in models:
        if hasattr(m, "zero_grad_buffer"):
            m.zero_grad_buffer()
        m.zero_grad(set_to_none=True)

    # --- Reset consumed samples / step counter back to 0 ---
    global_state.train_state.step = 0
    global_state.train_state.consumed_train_samples = 0

    # --- Clear CUDA cache (single clear after warmup, before RUN_START) ---
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


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

    # import ctypes
    # import gc
    # libc = ctypes.CDLL("libc.so.6")
    # gc.collect()
    # libc.malloc_trim(0)

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
        log_rank_0(f"Model module: {model_module}")

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

    # interval-time is started/stopped once per iteration around train_step only (see training loop).
    # A single long-running interval-time includes logging, hooks, and other post-step work, which
    # inflates "elapsed time per iteration (ms)" vs profiler / train-step wall time.
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
            trace_dir = config.logger.tensorboard_dir or os.path.join(os.getcwd(), "torch_profiler_traces")
            prof = initialize_pytorch_profiler(prof_config, trace_dir)
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

    start_iteration = global_state.train_state.step
    log_rank_0(f"Starting training loop at iteration {start_iteration}")
    
    dp_size = pg_collection.dp.size()
    batch_size = dp_size * train_config.micro_batch_size * get_num_microbatches()
    timer = Timer(batch_size)

    # Synthetic warmup: pre-compile JIT kernels before measured training begins.
    run_synthetic_warmup(
        forward_step_func, model, optimizer, scheduler, global_state, pg_collection,
    )
    _log_training_gpu_mem("after synthetic warmup", config.logger.memory_keys)

    sft_logger = _get_sft_logger()

    # MLPerf logging: transition from init to training (after warmup)
    if sft_logger is not None:
        sft_logger.log_init_stop_run_start()

    # Wall-clock for the training loop only (first through last train step; warmup eval disabled).
    training_wall_start = time.perf_counter()
    # Skip the first N interval evals (e.g. 48, 96, 144 when eval_interval=48); run from step 4*interval (192).
    eval_skip_first_n = 3

    # One-shot MXFP4 healing env banner (no-op unless HEALING_ITER > 0 and
    # MXFP4_HEALING_PHASE_LOG is on). Safe to call even when the healing
    # module isn't imported — gracefully degrades.
    try:
        from primus.backends.megatron_bridge.recipes.mlperf_llama2_70b.mxfp4_healing import (
            log_healing_env_banner_once,
        )

        log_healing_env_banner_once()
    except Exception:  # noqa: BLE001
        pass

    # NeMo / MLPerf reference (MI355X implementation): CustomCallback uses time.time() in Lightning
    # on_train_batch_start / on_train_batch_end — i.e. wall time over training_step only, no cuda.synchronize.
    # We mirror that by timing megatron.train_step only, then averaging over logger.log_interval like interval-time.
    nemo_style_iter_seconds_accum = 0.0
    nemo_style_iter_count = 0

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

        # Update the timeout for all process groups after initialization
        # We update the timeout after the first successful iteration,
        # which takes longer than others usually
        if global_state.train_state.step == start_iteration + 1:
            distributed_timeout_seconds_after_init = global_state.cfg.dist.distributed_timeout_seconds_after_init
            if distributed_timeout_seconds_after_init is not None:
                update_pg_timeout(timedelta(seconds=distributed_timeout_seconds_after_init))


        # Capture CUDA Graphs after warmup.
        # if (
        #     model_config.cuda_graph_impl == "transformer_engine"
        #     and cuda_graph_helper is not None
        #     and not cuda_graph_helper.graphs_created()
        #     and global_state.train_state.step - start_iteration == model_config.cuda_graph_warmup_steps
        # ):
        #     if model_config.cuda_graph_warmup_steps > 0 and should_toggle_forward_pre_hook:
        #         disable_forward_pre_hook(model, param_sync=False)
        #     cuda_graph_helper.create_cudagraphs()
        #     if model_config.cuda_graph_warmup_steps > 0 and should_toggle_forward_pre_hook:
        #         enable_forward_pre_hook(model)
        #         cuda_graph_helper.cuda_graph_set_manual_hooks()

        # Run training step.
        timers("interval-time", log_level=0).start(barrier=False)
        timer.start()
        fault_tolerance.on_training_step_start(global_state)
        _nemo_t0 = time.time()
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
        nemo_style_iter_seconds_accum += time.time() - _nemo_t0
        nemo_style_iter_count += 1
        fault_tolerance.on_training_step_end(global_state)
        timer.stop()
        timers("interval-time").stop()

        # Advance NVIDIA DLFw Inspect step if enabled
        tensor_inspect_step_if_enabled(config.tensor_inspect)

        if config.logger.log_throughput_to_tensorboard:
            history_wct.append(time.time() - global_state.start_time)
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

        # MXFP4 healing: when train_state.step + 1 == HEALING_ITER, restore FP8
        # weights from the CPU stash (see ``mxfp4_healing``) and
        # switch ``megatron.core.fp4_utils`` out of the MXFP4 phase. No-op when
        # HEALING_ITER == 0 (default) so BF16/MXFP8 submission runs are
        # unaffected.
        try:
            from primus.backends.megatron_bridge.recipes.mlperf_llama2_70b import (
                mxfp4_healing as _mxh,
            )

            if _mxh.healing_iter() > 0:
                _mxh.apply_healing_after_step(model, model_config, global_state.train_state.step)
            _mxh.log_training_step_phase(global_state.train_state.step)
        except Exception as _heal_err:  # noqa: BLE001
            _orig_log_rank_0(
                f"[mxfp4_healing] Failed at step={global_state.train_state.step}: "
                f"{type(_heal_err).__name__}: {_heal_err}"
            )
            raise

        # If fsdp_manual_registration is enabled, manually register FSDP communication buffers after one training step.
        # if global_state.train_state.step == start_iteration + 1 and config.ddp.use_megatron_fsdp:
        #     _maybe_register_fsdp_buffers(config, model)

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
        nemo_elapsed_time_per_iter_sec = None
        if (
            config.logger.log_interval
            and global_state.train_state.step % config.logger.log_interval == 0
            and nemo_style_iter_count > 0
        ):
            nemo_elapsed_time_per_iter_sec = (
                nemo_style_iter_seconds_accum / nemo_style_iter_count
            )
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
            nemo_elapsed_time_per_iter_sec=nemo_elapsed_time_per_iter_sec,
        )
        if config.logger.log_interval and global_state.train_state.step % config.logger.log_interval == 0:
            _log_training_gpu_mem(
                f"step {global_state.train_state.step}",
                config.logger.memory_keys,
            )
        if nemo_elapsed_time_per_iter_sec is not None:
            nemo_style_iter_seconds_accum = 0.0
            nemo_style_iter_count = 0

        # MLPerf logging: per-step train loss
        if sft_logger is not None and not skipped_iter and loss_dict:
            sft_logger.on_train_step(
                step=global_state.train_state.step,
                loss_dict=loss_dict,
                lr=learning_rate,
                consumed_samples=global_state.train_state.consumed_train_samples,
            )

        if (
            global_state.train_state.do_valid
            and train_config.eval_interval
            and global_state.train_state.step % train_config.eval_interval == 0
            and global_state.train_state.step > eval_skip_first_n * train_config.eval_interval
        ):
            if energy_monitor is not None:
                energy_monitor.pause()
            if should_toggle_forward_pre_hook:
                disable_forward_pre_hook(model)
                pre_hook_enabled = False
            if train_config.manual_gc and train_config.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f"iteration {global_state.train_state.step}"
            timers("eval-time", log_level=0).start(barrier=True)
            assert timer.consumed_samples == global_state.train_state.consumed_train_samples, "Timer and global_state sample mismatch"

            if sft_logger is not None:
                sft_logger.on_eval_start(global_state.train_state.consumed_train_samples)

            should_exit, eval_loss_value = evaluate_and_print_results_custom(
                global_state,
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                model_config,
                verbose=False,
                write_to_tensorboard=False,
                process_non_loss_data_func=process_non_loss_data_func,
                non_loss_data_func=non_loss_data_func,
                throughput=timer.get_throughput(),
            )

            if sft_logger is not None and eval_loss_value is not None:
                target_hit = sft_logger.on_eval_end(
                    global_state.train_state.consumed_train_samples, eval_loss_value,
                )
                if target_hit:
                    should_exit = True
            if should_exit:
                exit_code = 0
            eval_duration += timers("eval-time").elapsed()
            eval_iterations += train_config.eval_iters
            timers("eval-time").stop()

            if train_config.manual_gc and train_config.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if should_toggle_forward_pre_hook:
                enable_forward_pre_hook(model)
                pre_hook_enabled = True
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
        if should_exit:
            break

    training_wall_elapsed_s = time.perf_counter() - training_wall_start
    log_rank_0(
        f"Training loop finished: wall time {training_wall_elapsed_s:.2f} s "
        f"({training_wall_elapsed_s / 60.0:.2f} min); "
        f"final_iteration={global_state.train_state.step}; "
        f"consumed_train_samples={global_state.train_state.consumed_train_samples}"
    )

    # MLPerf logging: end of training
    if sft_logger is not None:
        sft_logger.log_run_stop(global_state.train_state.consumed_train_samples)

    _delete_cuda_graphs(cuda_graph_helper)

    # Stop RPD profiler if still running
    if rpd and rpd_status == "running":
        log_rank_0(f"Stopping RPD profiling at training end (step {global_state.train_state.step})")
        rpd.stop()
        rpd_status = "finished"

    # Flush TensorBoard writer if present (TB/WandB disabled in LoggerConfig for this recipe).
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
        if exit_code != 0:
            sys.exit(exit_code)

    # Close NVIDIA DLFw Inspect at clean finish
    tensor_inspect_end_if_enabled(config.tensor_inspect)

# ---------------------------------------------------------------------------
# Install ``megatron_bridge_train_override`` on the megatron-bridge training
# modules. When ``PRE_QUANTIZED_MODEL`` is enabled, the override is first
# wrapped by ``install_pre_quantize_wrap`` so that the very first entry into
# the training loop pre-quantizes all TE Linear / LayerNormLinear weights to
# MXFP4 (and stashes FP8 copies on CPU for healing) before delegating to the
# override. When disabled (default), the override is installed directly (no
# wrap, zero runtime cost).
# ---------------------------------------------------------------------------
from megatron.bridge.training import train as _mb_train_mod
from primus.backends.megatron_bridge.recipes.mlperf_llama2_70b.pre_quantize_mxfp4 import (
    install_pre_quantize_wrap,
)

_installed_train = install_pre_quantize_wrap(megatron_bridge_train_override)
setattr(megatron_bridge_train_override, "_primus_llama2_custom_train_override", True)

_mb_train_mod.train = _installed_train

try:
    from megatron.bridge.training import pretrain as _mb_pretrain_mod
    _mb_pretrain_mod.train = _installed_train
except ImportError:
    pass

try:
    from megatron.bridge.training import finetune as _mb_finetune_mod
    if hasattr(_mb_finetune_mod, "train"):
        _mb_finetune_mod.train = _installed_train
except ImportError:
    pass

# Keep the historical name `train` bound to the training module so any
# lingering `train.train = ...` expectations elsewhere resolve.
train = _mb_train_mod