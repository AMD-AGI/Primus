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

import os
from typing import List, Optional, Union

import torch
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
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
from megatron.bridge.training.mixed_precision import (
    MixedPrecisionConfig,
    bf16_mixed,
    register,
)


@register
def bf16_with_fp8_hybrid() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with MXFP8.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with MXFP8 mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "e4m3"
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


def llama2_70b_pretrain_config(**user_kwargs: Unpack[Llama2CustomKwargs]) -> ConfigContainer:
    """
    Return a custom pre-training config for Llama-2 70B.

    This is a custom variant that can be modified without changing Megatron-Bridge code.
    See `_llama2_common` for the full list of parameters.
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
    return _llama2_common(**combined_kwargs)


def _llama2_common(
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
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

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
    if packed_sequence:
        packed_sequence_specs = PackedSequenceSpecs(
            packed_sequence_size=seq_length,  # Must be > 0 to use packed files
            tokenizer_model_name=hf_path,
            packed_train_data_path=packed_train_data_path or "/data/train.npy",
            packed_val_data_path=packed_val_data_path or "/data/validation.npy",
            packed_metadata_path=packed_metadata_path or "/data/packed_metadata.jsonl",  # Metadata for packed sequences
        )
    else:
        packed_sequence_specs = None

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
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,  # need use_distributed_optimizer=True
        ),
        dataset=FinetuningDatasetConfig(
            dataset_root="/data",  # Point to your .npy files directory
            seq_length=seq_length,
            seed=1234,
            packed_sequence_specs=packed_sequence_specs,
            # Dataloader config parameters
            data_sharding=True,
            dataloader_type="batch",  # "batch" is recommended for fine-tuning
            num_workers=1,
        ),

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
            save_interval=save_interval,
            save=None,
            load=None,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
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
