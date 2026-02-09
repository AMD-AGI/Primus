###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import os
from typing import Any

from MaxText.configs.types import (  # Run and Checkpointing; Data Types and Quantization; Core Model Architecture; Attention Mechanisms; Mixture of Experts; Parallelism and Layout; Training, Optimization, and Fine-Tuning; Reinforcement Learning; Positional Embeddings; Dataset Loading and Tokenization; Inference; Development and Debugging; Metrics and Monitoring; Multimodal; Derived
    AOT,
    GRPO,
    MTP,
    VLLM,
    AdamW,
    Attention,
    Checkpointing,
    DatasetGeneral,
    DataTypes,
    DcnParallelism,
    Debug,
    Decoding,
    DeepSeekMoE,
    DerivedValues,
    DevelopmentAndDebugging,
    EmergencyCheckpointing,
    FineTuning,
    GcpMonitoring,
    Goodput,
    GrainDataset,
    HardwareAndMesh,
    HfDataset,
    HloDump,
    IciParallelism,
    InferenceBenchmark,
    InferenceGeneral,
    InferenceLayout,
    InferenceServer,
    LayoutAndSharding,
    Llama4Attention,
    Logits,
    MaxTextConfig,
    Metrics,
    MlaAttention,
    MoBa,
    ModelArchitecture,
    MoEGeneral,
    MoEKernels,
    MultimodalGeneral,
    Optimizer,
    OrbaxStorage,
    PagedAttention,
    PipelineParallelism,
    PositionalEmbedding,
    PrefixCaching,
    Profiling,
    Quantization,
    Qwen3Next,
    RematAndOffload,
    Reward,
    RLDataset,
    RLEvaluation,
    RLHardware,
    Rope,
    RunInfo,
    SpecialTokens,
    SplashAttention,
    StackTrace,
    Tensorboard,
    TfdsDataset,
    Tokenizer,
    TrainingLoop,
    VisionProjector,
    VisionTower,
    YarnRope,
)
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.fields import Field


class PrimusMoEGeneral(MoEGeneral):
    expert_balance: bool = Field(False, description="Whether to use expert balancing.")


class PrimusDevelopmentAndDebugging(DevelopmentAndDebugging):
    jax_distributed_heartbeat_timeout_seconds: int = Field(
        100,
        description="How long before a missing heartbeat marks a task as dead. Increase for slow NFS checkpoint restores.",
    )


class PrimusTurboConfig(BaseModel):
    enable_primus_turbo: bool = Field(False, description="Whether to enable Primus Turbo.")
    use_turbo_grouped_gemm: bool = Field(False, description="Whether to use turbo grouped gemm.")


class PrimusWandbConfig(BaseModel):
    enable_wandb: bool = Field(False, description="Whether to enable WandB.")
    wandb_project: None | str = Field(None, description="The name of the WandB project.")
    wandb_exp_name: None | str = Field(
        None, description="The name of the WandB experiment, derived from the run_name if not set."
    )
    wandb_save_dir: None | str = Field(None, description="The directory to save the WandB logs.")


class PrimusMaxTextConfig(
    # Run and Checkpointing
    RunInfo,
    Checkpointing,
    OrbaxStorage,
    EmergencyCheckpointing,
    # Data Types and Quantization
    DataTypes,
    Quantization,
    # Core Model Architecture
    ModelArchitecture,
    MTP,
    Logits,
    # Attention Mechanisms
    Attention,
    MlaAttention,
    MoBa,
    Llama4Attention,
    SplashAttention,
    PagedAttention,
    # Mixture of Experts - REPLACED with PrimusMoEGeneral
    PrimusMoEGeneral,  # Replaces MoEGeneral
    MoEKernels,
    DeepSeekMoE,
    Qwen3Next,
    # Parallelism and Layout
    HardwareAndMesh,
    LayoutAndSharding,
    DcnParallelism,
    IciParallelism,
    PipelineParallelism,
    # Training, Optimization, and Fine-Tuning
    RematAndOffload,
    TrainingLoop,
    Optimizer,
    AdamW,
    FineTuning,
    # Reinforcement Learning
    RLHardware,
    VLLM,
    GRPO,
    RLDataset,
    RLEvaluation,
    Reward,
    SpecialTokens,
    # Positional Embeddings
    PositionalEmbedding,
    Rope,
    YarnRope,
    # Dataset Loading and Tokenization
    DatasetGeneral,
    TfdsDataset,
    HfDataset,
    GrainDataset,
    Tokenizer,
    # Inference
    InferenceGeneral,
    Decoding,
    InferenceLayout,
    InferenceServer,
    InferenceBenchmark,
    PrefixCaching,
    # Development and Debugging - REPLACED with PrimusDevelopmentAndDebugging
    AOT,
    PrimusDevelopmentAndDebugging,  # Replaces DevelopmentAndDebugging
    Profiling,
    HloDump,
    StackTrace,
    # Metrics and Monitoring
    Metrics,
    Goodput,
    GcpMonitoring,
    Tensorboard,
    # Multimodal
    MultimodalGeneral,
    VisionTower,
    VisionProjector,
    # Primus-specific configs - ADDED
    PrimusTurboConfig,
    PrimusWandbConfig,
    # Derived
    DerivedValues,
):
    """
    The main configuration object for Primus MaxText.

    This class extends MaxTextConfig with Primus-specific configurations:
    - Replaces MoEGeneral with PrimusMoEGeneral (adds expert_balance)
    - Replaces DevelopmentAndDebugging with PrimusDevelopmentAndDebugging (adds jax_distributed_heartbeat_timeout_seconds)
    - Adds PrimusTurboConfig (Primus Turbo optimizations)
    - Adds PrimusWandbConfig (WandB integration)

    All other functionality from MaxTextConfig is preserved.
    """

    debug: Debug = Field(default_factory=Debug)
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def load_model_specific_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """This method is a no-op because `pyconfig` handles model-specific config loading."""
        return values

    @model_validator(mode="after")
    def set_derived_and_validate_values(self) -> "PrimusMaxTextConfig":
        """
        Computes all derived values and runs all cross-field validations after initial parsing.
        This calls the MaxTextConfig's validation logic and then adds any Primus-specific validations.
        """
        # Call MaxTextConfig's validation logic directly since we're using composition via multiple inheritance
        # rather than direct inheritance. MaxTextConfig.set_derived_and_validate_values expects a MaxTextConfig
        # instance, but since we have all the same base classes, we can call it on self.
        # We need to temporarily cast self to MaxTextConfig for the method call, or call the method directly.
        # Actually, since MaxTextConfig's method works on the same fields we have, we can call it directly.
        MaxTextConfig.set_derived_and_validate_values(self)

        # Add any Primus-specific validations here if needed
        if self.wandb_save_dir is None or self.wandb_save_dir == "" and self.base_output_directory:
            self.wandb_save_dir = os.path.join(self.base_output_directory, "wandb")

        if self.wandb_project is None or self.wandb_project == "":
            self.wandb_project = os.getenv("WANDB_PROJECT", "Primus-MaxText-Pretrain")

        if self.wandb_exp_name is None or self.wandb_exp_name == "" and self.run_name:
            self.wandb_exp_name = self.run_name

        if self.enable_wandb and "WANDB_API_KEY" not in os.environ:
            raise ValueError("WANDB_API_KEY is not set. Please set it or login wandb before proceeding")

        if not self.enable_primus_turbo:
            self.use_turbo_grouped_gemm = False

        return self
