"""Axis -> override translator.

Single source of truth for "given an `axis` name from `axis_taxonomy.md` and
a candidate `value`, what does it become at execution time?".

Three execution channels exist today:

- ``trainer_override``: a key/value to merge into ``modules.pre_trainer.overrides``
  in the Primus exp YAML (handled by ``submit.run`` via ``--override``).
- ``env``: an environment variable to inject into the launcher (handled by
  ``submit.run`` via ``--set-env``).  Boolean values are normalized to ``"1"``
  / ``"0"`` since the trainer reads them as strings.
- ``structural``: same as ``trainer_override`` but flagged so the caller knows
  to re-run ``constraint.check`` / re-estimate memory before submission.

The translator deliberately does NOT cross-reference the engine output;
``pilot.tools.diagnose`` returns axis names, this module turns those names
into concrete (channel, key, value) tuples. The engine never knows whether an
axis is ultimately a YAML override or an env var; that's a deployment
concern owned here.

If an axis is unknown, ``translate`` returns ``None`` and the caller is
expected to log + skip.  Adding a new axis means appending one entry below
and (if structural) updating ``axis_taxonomy.md`` §2.x in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Channel = Literal["trainer_override", "env", "structural"]


@dataclass(frozen=True)
class AxisAction:
    """One concrete change derived from a (axis, value) pair."""

    axis: str
    value: Any
    channel: Channel
    key: str  # YAML override key (channel != "env") OR env var name (channel == "env")
    rendered_value: Any  # what actually goes into overrides / env (e.g. bool -> "1"/"0")


_TRAINER_OVERRIDE_AXES: dict[str, str] = {
    # Communication / overlap (weakly_local).
    "overlap_grad_reduce": "overlap_grad_reduce",
    "overlap_param_gather": "overlap_param_gather",
    "gradient_accumulation_fusion": "gradient_accumulation_fusion",
    "turbo_deepep_use_comm_stream": "turbo_deepep_use_comm_stream",
    "turbo_deepep_num_cu": "turbo_deepep_num_cu",
    "moe_shared_expert_overlap": "moe_shared_expert_overlap",
    "moe_router_force_load_balancing": "moe_router_force_load_balancing",
    "moe_router_dtype": "moe_router_dtype",
    "turbo_sync_free_moe_stage": "turbo_sync_free_moe_stage",
    "attention_kernel": "attention_backend",
    # Recompute / memory (strongly_local).
    "recompute_granularity": "recompute_granularity",
    "recompute_method": "recompute_method",
    "recompute_num_layers": "recompute_num_layers",
    "optimizer_offload": "optimizer_offload",
    # FP8 / precision (axis_taxonomy.md §2.6).
    "fp8_recipe": "fp8_recipe",
    "accumulate_allreduce_grads_in_fp32": "accumulate_allreduce_grads_in_fp32",
    "attention_softmax_in_fp32": "attention_softmax_in_fp32",
    "attention_dropout": "attention_dropout",
    # Megatron fusion knobs (axis_taxonomy.md §2.7).
    "apply_rope_fusion": "apply_rope_fusion",
    "bias_activation_fusion": "bias_activation_fusion",
    "bias_dropout_fusion": "bias_dropout_fusion",
    "masked_softmax_fusion": "masked_softmax_fusion",
    # CUDA graphs (axis_taxonomy.md §2.8 — stack-blocked today; still
    # registered so DIAGNOSE can name them and constraint.check can mutex
    # them out).
    "enable_cuda_graph": "enable_cuda_graph",
    "external_cuda_graph": "external_cuda_graph",
    "cuda_graph_impl": "cuda_graph_impl",
    "cuda_graph_scope": "cuda_graph_scope",
    # PP-only knobs (axis_taxonomy.md §2.9; gated by constraint.check).
    "defer_embedding_wgrad_compute": "defer_embedding_wgrad_compute",
    "overlap_p2p_communication": "overlap_p2p_communication",
    "overlap_param_gather_with_optimizer_step": "overlap_param_gather_with_optimizer_step",
    # Host-launch / runtime tuning (axis_taxonomy.md §2.10).
    "manual_gc": "manual_gc",
    "manual_gc_interval": "manual_gc_interval",
}

_STRUCTURAL_AXES: dict[str, str] = {
    "tensor_model_parallel_size": "tensor_model_parallel_size",
    "pipeline_model_parallel_size": "pipeline_model_parallel_size",
    "expert_model_parallel_size": "expert_model_parallel_size",
    "context_parallel_size": "context_parallel_size",
    "virtual_pipeline_model_parallel_size": "virtual_pipeline_model_parallel_size",
    "micro_batch_size": "micro_batch_size",
    "global_batch_size": "global_batch_size",
    "seq_length": "seq_length",
}

_ENV_AXES: dict[str, str] = {
    "MOE_BUFFER_PCT": "MOE_BUFFER_PCT",
    "MOE_PERMUTE_FUSION": "MOE_PERMUTE_FUSION",
    "NCCL_BUFFSIZE": "NCCL_BUFFSIZE",
    "NCCL_MIN_NCHANNELS": "NCCL_MIN_NCHANNELS",
    "NCCL_NET_GDR_LEVEL": "NCCL_NET_GDR_LEVEL",
    "NCCL_IB_DISABLE": "NCCL_IB_DISABLE",
    "NCCL_IB_HCA": "NCCL_IB_HCA",
    "RCCL_MSCCL_ENABLE": "RCCL_MSCCL_ENABLE",
    "PYTORCH_HIP_ALLOC_CONF": "PYTORCH_HIP_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF": "PYTORCH_CUDA_ALLOC_CONF",
    # Host-launch / runtime tuning (axis_taxonomy.md §2.10).
    "OMP_NUM_THREADS": "OMP_NUM_THREADS",
    "GPU_MAX_HW_QUEUES": "GPU_MAX_HW_QUEUES",
    "MIOPEN_FIND_MODE": "MIOPEN_FIND_MODE",
    # RCCL extras (axis_taxonomy.md §2.11).
    "RCCL_PROTO": "RCCL_PROTO",
    "RCCL_ALGO": "RCCL_ALGO",
    "RCCL_NTHREADS": "RCCL_NTHREADS",
    "TORCH_NCCL_HIGH_PRIORITY": "TORCH_NCCL_HIGH_PRIORITY",
    # ROCm / HSA (axis_taxonomy.md §2.12; HSA_ENABLE_INTERRUPT carries a
    # DANGER note — engine must NEVER emit value=0 unless explicitly
    # acknowledged).
    "HSA_NO_SCRATCH_RECLAIM": "HSA_NO_SCRATCH_RECLAIM",
    "HSA_ENABLE_INTERRUPT": "HSA_ENABLE_INTERRUPT",
    # Profile-blocker (axis_taxonomy.md §2.14 MUTEX-PROFILE-HIPBLASLT).
    "PRIMUS_HIPBLASLT_TUNING": "PRIMUS_HIPBLASLT_TUNING",
}


def _render_env_value(value: Any) -> str:
    """Trainer reads env values as strings; normalize booleans to '1'/'0'."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def translate(axis: str, value: Any) -> AxisAction | None:
    """Translate one (axis, value) pair.

    Returns ``None`` for unknown axes so the caller can drop them with a
    warning; raising would force the caller to handle every taxonomy gap as
    an exception, which is hostile to the LLM-driven layer above.
    """
    if axis in _TRAINER_OVERRIDE_AXES:
        return AxisAction(
            axis=axis,
            value=value,
            channel="trainer_override",
            key=_TRAINER_OVERRIDE_AXES[axis],
            rendered_value=value,
        )
    if axis in _STRUCTURAL_AXES:
        return AxisAction(
            axis=axis,
            value=value,
            channel="structural",
            key=_STRUCTURAL_AXES[axis],
            rendered_value=value,
        )
    if axis in _ENV_AXES:
        return AxisAction(
            axis=axis,
            value=value,
            channel="env",
            key=_ENV_AXES[axis],
            rendered_value=_render_env_value(value),
        )
    return None


def is_known(axis: str) -> bool:
    return axis in _TRAINER_OVERRIDE_AXES or axis in _STRUCTURAL_AXES or axis in _ENV_AXES


def channel_of(axis: str) -> Channel | None:
    if axis in _TRAINER_OVERRIDE_AXES:
        return "trainer_override"
    if axis in _STRUCTURAL_AXES:
        return "structural"
    if axis in _ENV_AXES:
        return "env"
    return None
