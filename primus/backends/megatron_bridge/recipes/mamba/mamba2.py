###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Mamba2 finetuning recipes for Megatron-Bridge.

These recipes extend the upstream Mamba2 pretrain-only recipes with
finetuning support (SFT, LoRA/DoRA).  They reuse the model providers
and utility helpers shipped with Megatron-Bridge so that no third-party
code needs to be modified.
"""

import os

import torch
from typing_extensions import TypedDict, Unpack

from megatron.bridge.models.mamba import (
    MambaModelProvider130M,
    MambaModelProvider370M,
    MambaModelProvider780M,
    MambaModelProvider1P3B,
    MambaModelProvider2P7B,
    NVIDIAMambaHybridProvider8B,
    NVIDIAMambaModelProvider8B,
)
from megatron.bridge.recipes.utils.finetune_utils import (
    default_peft_config,
    default_squad_config,
)
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


# ---------------------------------------------------------------------------
# Typed kwargs for Mamba2 finetuning recipes
# ---------------------------------------------------------------------------

class Mamba2FinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Mamba2 finetuning recipe helpers."""

    # Core identifiers
    dir: str | None
    name: str

    # Finetuning-specific
    pretrained_checkpoint: str | None
    peft: str | None
    packed_sequence: bool

    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    sequence_parallel: bool

    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    eval_interval: int
    save_interval: int

    # Optimizer
    finetune_lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None

    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


# ---------------------------------------------------------------------------
# Public finetune recipe helpers
# ---------------------------------------------------------------------------

def mamba2_130m_finetune_config(**user_kwargs: Unpack[Mamba2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Mamba2 130M."""
    recommended: Mamba2FinetuneKwargs = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
    }
    kwargs: Mamba2FinetuneKwargs = {**recommended, **user_kwargs}
    return _mamba2_finetune_common(model_provider=MambaModelProvider130M, **kwargs)


def mamba2_370m_finetune_config(**user_kwargs: Unpack[Mamba2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Mamba2 370M."""
    recommended: Mamba2FinetuneKwargs = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
    }
    kwargs: Mamba2FinetuneKwargs = {**recommended, **user_kwargs}
    return _mamba2_finetune_common(model_provider=MambaModelProvider370M, **kwargs)


def mamba2_780m_finetune_config(**user_kwargs: Unpack[Mamba2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Mamba2 780M."""
    recommended: Mamba2FinetuneKwargs = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
    }
    kwargs: Mamba2FinetuneKwargs = {**recommended, **user_kwargs}
    return _mamba2_finetune_common(model_provider=MambaModelProvider780M, **kwargs)


def mamba2_1p3b_finetune_config(**user_kwargs: Unpack[Mamba2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Mamba2 1.3B."""
    recommended: Mamba2FinetuneKwargs = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
    }
    kwargs: Mamba2FinetuneKwargs = {**recommended, **user_kwargs}
    return _mamba2_finetune_common(model_provider=MambaModelProvider1P3B, **kwargs)


def mamba2_2p7b_finetune_config(**user_kwargs: Unpack[Mamba2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Mamba2 2.7B."""
    recommended: Mamba2FinetuneKwargs = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
    }
    kwargs: Mamba2FinetuneKwargs = {**recommended, **user_kwargs}
    return _mamba2_finetune_common(model_provider=MambaModelProvider2P7B, **kwargs)


def mamba2_8b_finetune_config(**user_kwargs: Unpack[Mamba2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Mamba2 8B."""
    recommended: Mamba2FinetuneKwargs = {
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
    }
    kwargs: Mamba2FinetuneKwargs = {**recommended, **user_kwargs}
    return _mamba2_finetune_common(model_provider=NVIDIAMambaModelProvider8B, **kwargs)


def mamba2_hybrid_8b_finetune_config(**user_kwargs: Unpack[Mamba2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Mamba2 Hybrid 8B."""
    recommended: Mamba2FinetuneKwargs = {
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
    }
    kwargs: Mamba2FinetuneKwargs = {**recommended, **user_kwargs}
    return _mamba2_finetune_common(model_provider=NVIDIAMambaHybridProvider8B, **kwargs)


# ---------------------------------------------------------------------------
# Common finetuning configuration builder
# ---------------------------------------------------------------------------

def _mamba2_finetune_common(
    model_provider: (
        type[MambaModelProvider130M]
        | type[MambaModelProvider370M]
        | type[MambaModelProvider780M]
        | type[MambaModelProvider1P3B]
        | type[MambaModelProvider2P7B]
        | type[NVIDIAMambaModelProvider8B]
        | type[NVIDIAMambaHybridProvider8B]
    ),
    dir: str | None = None,
    name: str = "default",
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    peft: str | None = "none",
    packed_sequence: bool = False,
    # Training hyperparameters
    train_iters: int = 1000,
    global_batch_size: int = 128,
    micro_batch_size: int = 4,
    seq_length: int = 2048,
    eval_interval: int = 30,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 5.0e-6,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: int | None = None,
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
) -> ConfigContainer:
    """
    Create a finetuning configuration for Mamba 2.x models.

    This mirrors the pretrain ``_mamba2_common`` helper but replaces the
    GPT-dataset configuration with a SQuAD-based HFDatasetConfig and adds
    support for PEFT adapters (LoRA / DoRA) and pretrained-checkpoint
    loading — following the same pattern used by the Llama-3 and NemotronH
    finetuning recipes in Megatron-Bridge.
    """
    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Model
    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
    )

    # Optimizer & scheduler (lower LR for finetuning)
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        weight_decay=0.1,
        max_lr=finetune_lr,
        min_lr=min_lr,
    )

    # PEFT (LoRA / DoRA / None for full SFT)
    mamba_target_modules = [
        "linear_qkv", "linear_proj", "linear_fc1", "linear_fc2",
        "in_proj", "out_proj",
    ]
    peft_config = default_peft_config(peft, target_modules=mamba_target_modules)

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Tokenizer — use HuggingFace tokenizer (same default as pretrain recipe)
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model="EleutherAI/gpt-neox-20b",
        hf_tokenizer_kwargs={"use_fast": True},
    )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=10,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=False,
            use_distributed_optimizer=True,
        ),
        dataset=default_squad_config(seq_length, packed_sequence),
        logger=logger_cfg,
        tokenizer=tokenizer_cfg,
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            dist_ckpt_strictness="log_all",
        ),
        rng=RNGConfig(seed=5678),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg


__all__ = [
    "mamba2_130m_finetune_config",
    "mamba2_370m_finetune_config",
    "mamba2_780m_finetune_config",
    "mamba2_1p3b_finetune_config",
    "mamba2_2p7b_finetune_config",
    "mamba2_8b_finetune_config",
    "mamba2_hybrid_8b_finetune_config",
]
