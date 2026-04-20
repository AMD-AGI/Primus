###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Zebra Llama (hybrid Mamba+MLA) recipes for Megatron-Bridge.

These recipes define model providers and training configurations for the
Zebra Llama family of hybrid models that combine Mamba SSM layers with
Multi-Latent Attention (MLA).  The model providers live on the Primus side
so that no third-party code needs to be modified.
"""

import os
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import torch
from typing_extensions import TypedDict, Unpack

from megatron.core.models.mamba import MambaModel as MCoreMambaModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import MLATransformerConfig
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.finetune_utils import (
    default_peft_config,
    default_squad_config,
)
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


# ---------------------------------------------------------------------------
# Hybrid Mamba+MLA Model Provider (extends MLATransformerConfig)
# ---------------------------------------------------------------------------


def _get_hybrid_mamba_mla_stack_spec(config: "ZebraLlamaMambaMLAProvider") -> ModuleSpec:
    """Return the Primus hybrid Mamba+MLA stack spec.

    The import is deferred so that the heavyweight module-level construction
    inside ``hybrid_mamba_mla_layer_specs`` (TE layers, MoE spec, etc.) only
    runs at model-build time, not when the recipe module is first loaded.
    """
    from primus.backends.megatron.core.models.hybrid.hybrid_mamba_mla_layer_specs import (
        hybrid_stack_spec,
    )

    return hybrid_stack_spec


@dataclass
class ZebraLlamaMambaMLAProvider(MLATransformerConfig, ModelProviderMixin[MCoreMambaModel]):
    """Configuration and provider for Zebra Llama hybrid Mamba+MLA models.

    This class combines MLATransformerConfig (for MLA attention parameters)
    with MCoreMambaModel to create a hybrid model that interleaves Mamba SSM
    layers with Multi-Latent Attention layers.
    """

    # ---- Model configuration ----
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    fp16: bool = False
    bf16: bool = True
    is_hybrid_model: bool = True

    # ---- Mamba-specific parameters ----
    mamba_num_groups: int = 8
    hybrid_attention_ratio: float = 0.25
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: Optional[str] = None
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none"
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = False
    make_vocab_size_divisible_by: int = 128
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    attention_backend: AttnBackend = AttnBackend.auto
    deallocate_pipeline_outputs: bool = True
    bias_dropout_fusion: bool = True
    cross_entropy_loss_fusion: bool = True

    mamba_stack_spec: Union[
        ModuleSpec,
        Callable[[], ModuleSpec],
        Callable[["ZebraLlamaMambaMLAProvider"], ModuleSpec],
    ] = _get_hybrid_mamba_mla_stack_spec

    vocab_size: Optional[int] = None
    should_pad_vocab: bool = False
    hf_model_id: Optional[str] = None
    _pg_collection: Optional[ProcessGroupCollection] = None

    restore_modelopt_state: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreMambaModel:
        """Instantiate a Megatron Core Mamba model with MLA attention support."""
        mamba_stack_spec = self.mamba_stack_spec
        if not isinstance(mamba_stack_spec, ModuleSpec):
            import inspect

            if len(inspect.signature(mamba_stack_spec).parameters) > 0:
                mamba_stack_spec = mamba_stack_spec(self)
            else:
                mamba_stack_spec = mamba_stack_spec()

        assert getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None, (
            "Virtual pipeline model parallelism is temporarily unsupported in SSM/Mamba "
            "models due to upstream MCore MambaModel API dependency"
        )

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        return MCoreMambaModel(
            self,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self.seq_length,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process or is_pp_first_stage(self._pg_collection.pp),
            post_process=post_process or is_pp_last_stage(self._pg_collection.pp),
            pg_collection=self._pg_collection,
        )


# ---------------------------------------------------------------------------
# Zebra Llama 1B Provider
# ---------------------------------------------------------------------------


@dataclass
class ZebraLlama1BModelProvider(ZebraLlamaMambaMLAProvider):
    """Configuration for Zebra Llama 1B (hybrid Mamba+MLA).

    Architecture summary:
        - 32 layers with 25% attention ratio (8 MLA + 24 Mamba layers)
        - hidden_size=2048, ffn_hidden_size=8192
        - Multi-Latent Attention with q_lora_rank=1344, kv_lora_rank=128
        - Mamba SSM with state_dim=64, head_dim=64, 8 groups
        - SwiGLU activation in MLP layers
        - Tokenizer: meta-llama/Llama-3.2-1B
    """

    # Layer counts and sizes
    num_layers: int = 32
    hidden_size: int = 2048
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 32
    seq_length: int = 8192

    # Hybrid Mamba parameters
    hybrid_attention_ratio: float = 0.25
    mamba_num_groups: int = 8

    # MLA parameters
    multi_latent_attention: bool = True
    q_lora_rank: int = 1344
    kv_lora_rank: int = 128
    qk_head_dim: int = 32
    qk_pos_emb_head_dim: int = 32
    v_head_dim: int = 64
    rotary_scaling_factor: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0

    # SwiGLU activation
    gated_linear_unit: bool = True

    # Position embedding — MLA uses its own internal positional encoding
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none"
    rotary_base: float = 500000
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-5


# ---------------------------------------------------------------------------
# Zebra Llama 3B Provider
# ---------------------------------------------------------------------------


@dataclass
class ZebraLlama3BModelProvider(ZebraLlamaMambaMLAProvider):
    """Configuration for Zebra Llama 3B (hybrid Mamba+MLA).

    Architecture summary:
        - 56 layers with 25% attention ratio (14 MLA + 42 Mamba layers)
        - hidden_size=3072, ffn_hidden_size=8192
        - Multi-Latent Attention with q_lora_rank=1536, kv_lora_rank=128
        - Mamba SSM with 8 groups
        - SwiGLU activation in MLP layers
        - Tokenizer: meta-llama/Llama-3.2-3B
    """

    # Layer counts and sizes
    num_layers: int = 56
    hidden_size: int = 3072
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 24
    seq_length: int = 8192

    # Hybrid Mamba parameters
    hybrid_attention_ratio: float = 0.25
    mamba_num_groups: int = 8

    # MLA parameters
    multi_latent_attention: bool = True
    q_lora_rank: int = 1536
    kv_lora_rank: int = 128
    qk_head_dim: int = 64
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    rotary_scaling_factor: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0

    # SwiGLU activation
    gated_linear_unit: bool = True

    # Position embedding — MLA uses its own internal positional encoding
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none"
    rotary_base: float = 500000
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-5


# ---------------------------------------------------------------------------
# Zebra Llama 8B Provider
# ---------------------------------------------------------------------------


@dataclass
class ZebraLlama8BModelProvider(ZebraLlamaMambaMLAProvider):
    """Configuration for Zebra Llama 8B (hybrid Mamba+MLA).

    Architecture summary:
        - 64 layers with 25% attention ratio (16 MLA + 48 Mamba layers)
        - hidden_size=4096, ffn_hidden_size=14436
        - Multi-Latent Attention with q_lora_rank=2048, kv_lora_rank=160
        - Mamba SSM with 8 groups
        - SwiGLU activation in MLP layers
        - Tokenizer: meta-llama/Llama-3.1-8B
    """

    # Layer counts and sizes
    num_layers: int = 64
    hidden_size: int = 4096
    ffn_hidden_size: int = 14436
    num_attention_heads: int = 32
    seq_length: int = 8192

    # Hybrid Mamba parameters
    hybrid_attention_ratio: float = 0.25
    mamba_num_groups: int = 8

    # MLA parameters
    multi_latent_attention: bool = True
    q_lora_rank: int = 2048
    kv_lora_rank: int = 160
    qk_head_dim: int = 64
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    rotary_scaling_factor: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0

    # SwiGLU activation
    gated_linear_unit: bool = True

    # Position embedding — MLA uses its own internal positional encoding
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none"
    rotary_base: float = 500000
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-5


# ---------------------------------------------------------------------------
# Typed kwargs for recipe helpers
# ---------------------------------------------------------------------------


class ZebraLlamaPretrainKwargs(TypedDict, total=False):
    """Typed options accepted by Zebra Llama pretrain recipe helpers."""

    dir: str | None
    name: str

    # Dataset
    data_paths: list[str] | None
    data_args_path: str | None
    train_data_path: list[str] | None
    valid_data_path: list[str] | None
    test_data_path: list[str] | None
    per_split_data_args_path: str | None
    mock: bool

    # Model parallelism
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
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None

    # Tokenizer
    use_null_tokenizer: bool

    # Precision / overlap
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


class ZebraLlamaFinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Zebra Llama finetune recipe helpers."""

    dir: str | None
    name: str

    # Finetuning-specific
    pretrained_checkpoint: str | None
    peft: str | None
    packed_sequence: bool

    # Model parallelism
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

    # Precision / overlap
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


# ---------------------------------------------------------------------------
# Public pretrain recipe
# ---------------------------------------------------------------------------


def zebra_llama_1b_pretrain_config(
    **user_kwargs: Unpack[ZebraLlamaPretrainKwargs],
) -> ConfigContainer:
    """Return a pre-training config for Zebra Llama 1B (hybrid Mamba+MLA)."""
    recommended: ZebraLlamaPretrainKwargs = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "use_null_tokenizer": False,
    }
    kwargs: ZebraLlamaPretrainKwargs = {**recommended, **user_kwargs}
    return _zebra_llama_pretrain_common(
        model_provider=ZebraLlama1BModelProvider,
        tokenizer_model="meta-llama/Llama-3.2-1B",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Public finetune recipe
# ---------------------------------------------------------------------------


def zebra_llama_1b_finetune_config(
    **user_kwargs: Unpack[ZebraLlamaFinetuneKwargs],
) -> ConfigContainer:
    """Return a finetuning config for Zebra Llama 1B (hybrid Mamba+MLA)."""
    recommended: ZebraLlamaFinetuneKwargs = {
        "name": "zebra_llama_1b",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
    }
    kwargs: ZebraLlamaFinetuneKwargs = {**recommended, **user_kwargs}
    return _zebra_llama_finetune_common(
        model_provider=ZebraLlama1BModelProvider,
        tokenizer_model="meta-llama/Llama-3.2-1B",
        **kwargs,
    )


def zebra_llama_3b_pretrain_config(
    **user_kwargs: Unpack[ZebraLlamaPretrainKwargs],
) -> ConfigContainer:
    """Return a pre-training config for Zebra Llama 3B (hybrid Mamba+MLA)."""
    recommended: ZebraLlamaPretrainKwargs = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "use_null_tokenizer": False,
    }
    kwargs: ZebraLlamaPretrainKwargs = {**recommended, **user_kwargs}
    return _zebra_llama_pretrain_common(
        model_provider=ZebraLlama3BModelProvider,
        tokenizer_model="meta-llama/Llama-3.2-3B",
        **kwargs,
    )


def zebra_llama_3b_finetune_config(
    **user_kwargs: Unpack[ZebraLlamaFinetuneKwargs],
) -> ConfigContainer:
    """Return a finetuning config for Zebra Llama 3B (hybrid Mamba+MLA)."""
    recommended: ZebraLlamaFinetuneKwargs = {
        "name": "zebra_llama_3b",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
    }
    kwargs: ZebraLlamaFinetuneKwargs = {**recommended, **user_kwargs}
    return _zebra_llama_finetune_common(
        model_provider=ZebraLlama3BModelProvider,
        tokenizer_model="meta-llama/Llama-3.2-3B",
        **kwargs,
    )


def zebra_llama_8b_pretrain_config(
    **user_kwargs: Unpack[ZebraLlamaPretrainKwargs],
) -> ConfigContainer:
    """Return a pre-training config for Zebra Llama 8B (hybrid Mamba+MLA)."""
    recommended: ZebraLlamaPretrainKwargs = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "use_null_tokenizer": False,
    }
    kwargs: ZebraLlamaPretrainKwargs = {**recommended, **user_kwargs}
    return _zebra_llama_pretrain_common(
        model_provider=ZebraLlama8BModelProvider,
        tokenizer_model="meta-llama/Llama-3.1-8B",
        **kwargs,
    )


def zebra_llama_8b_finetune_config(
    **user_kwargs: Unpack[ZebraLlamaFinetuneKwargs],
) -> ConfigContainer:
    """Return a finetuning config for Zebra Llama 8B (hybrid Mamba+MLA)."""
    recommended: ZebraLlamaFinetuneKwargs = {
        "name": "zebra_llama_8b",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
    }
    kwargs: ZebraLlamaFinetuneKwargs = {**recommended, **user_kwargs}
    return _zebra_llama_finetune_common(
        model_provider=ZebraLlama8BModelProvider,
        tokenizer_model="meta-llama/Llama-3.1-8B",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Common pretrain builder
# ---------------------------------------------------------------------------


def _zebra_llama_pretrain_common(
    model_provider: type[ZebraLlamaMambaMLAProvider],
    tokenizer_model: str | None = None,
    dir: str | None = None,
    name: str = "default",
    # Dataset
    data_paths: list[str] | None = None,
    data_args_path: str | None = None,
    train_data_path: list[str] | None = None,
    valid_data_path: list[str] | None = None,
    test_data_path: list[str] | None = None,
    per_split_data_args_path: str | None = None,
    mock: bool = False,
    # Model parallelism
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    # Training hyperparameters
    train_iters: int = 100,
    global_batch_size: int = 64,
    micro_batch_size: int = 8,
    seq_length: int = 8192,
    lr: float = 2.0e-4,
    min_lr: float = 2.0e-5,
    lr_warmup_iters: int = 200,
    lr_decay_iters: int | None = 10000,
    # Tokenizer
    use_null_tokenizer: bool = False,
    # Precision / overlap
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
) -> ConfigContainer:
    """Create a pre-training configuration for Zebra Llama hybrid models."""
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
    )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        weight_decay=0.1,
        max_lr=lr,
        min_lr=min_lr,
    )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=100,
            eval_iters=0,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            use_distributed_optimizer=True,
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=True,
            seq_length=seq_length,
            num_dataset_builder_threads=1,
            blend=blend,
            blend_per_split=blend_per_split,
            split=split,
            data_sharding=True,
            dataloader_type="single",
            num_workers=8,
            skip_getting_attention_mask_from_dataset=True,
        ),
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        tokenizer=(
            TokenizerConfig(
                tokenizer_type="NullTokenizer",
                tokenizer_model=None,
                vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
            )
            if use_null_tokenizer
            else TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=tokenizer_model or "meta-llama/Llama-3.2-1B",
                hf_tokenizer_kwargs={"use_fast": True},
            )
        ),
        checkpoint=CheckpointConfig(
            save_interval=10000,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch",
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg


# ---------------------------------------------------------------------------
# Common finetune builder
# ---------------------------------------------------------------------------


def _zebra_llama_finetune_common(
    model_provider: type[ZebraLlamaMambaMLAProvider],
    tokenizer_model: str = "meta-llama/Llama-3.2-1B",
    dir: str | None = None,
    name: str = "default",
    # Model parallelism
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
    # Precision / overlap
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
) -> ConfigContainer:
    """Create a finetuning configuration for Zebra Llama hybrid models."""
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
    )

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

    # PEFT target modules (Mamba + attention + MLP projections)
    mamba_mla_target_modules = [
        "linear_q_proj",
        "linear_q_down_proj",
        "linear_q_up_proj",
        "linear_kv_down_proj",
        "linear_kv_up_proj",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
        "in_proj",
        "out_proj",
    ]
    peft_config = default_peft_config(peft, target_modules=mamba_mla_target_modules)

    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=tokenizer_model,
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
    "ZebraLlamaMambaMLAProvider",
    "ZebraLlama1BModelProvider",
    "ZebraLlama3BModelProvider",
    "ZebraLlama8BModelProvider",
    "zebra_llama_1b_pretrain_config",
    "zebra_llama_1b_finetune_config",
    "zebra_llama_3b_pretrain_config",
    "zebra_llama_3b_finetune_config",
    "zebra_llama_8b_pretrain_config",
    "zebra_llama_8b_finetune_config",
]
