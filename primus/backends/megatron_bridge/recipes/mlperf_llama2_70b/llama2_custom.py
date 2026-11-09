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

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import gc
import math
import os
import sys
import time
from collections import deque
from datetime import timedelta
from typing import Any, Callable, List, Optional, Union

import _log_suppression  # noqa: F401, E402  # must precede noisy imports
import torch
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
)
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.parallel_state import update_pg_timeout
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import (
    RerunDataIterator,
    RerunMode,
    get_rerun_state_machine,
)
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import check_param_hashes_across_dp_replicas, get_model_config
from typing_extensions import TypedDict, Unpack

from primus.core.utils.module_utils import log_rank_0 as _orig_log_rank_0
from primus.core.utils.module_utils import log_rank_last as _orig_log_rank_last

_log_suppression.reapply_quiet_logger_levels()

log_rank_0 = _orig_log_rank_0
log_rank_last = _orig_log_rank_last

from megatron.bridge.utils import common_utils

common_utils.print_rank_0 = log_rank_0
common_utils.print_rank_last = log_rank_last
common_utils.warn_rank_0 = log_rank_0
common_utils.warn_rank_last = log_rank_last

import numpy as np
from megatron.bridge import AutoBridge
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.datasets.sft import GPTSFTPackedDataset
from megatron.bridge.data.finetuning import prepare_finetuning_batch
from megatron.bridge.data.iterator_utils import make_data_iterator_list

from primus.backends.megatron_bridge.patches.mlperf_llama2_70b.resettable_data_iterator import (
    ResettableDataIterator,
)

_orig_build_loss_mask = GPTSFTPackedDataset._build_loss_mask


# The .npy stores loss_mask on the target tokens themselves (labels != -100), not
# rolled onto the preceding position as NeMo's packer does, while labels[t] is
# input_ids[t + 1]. Read the mask at t + 1 so each label is scored by its own mask,
# as the reference's HF shifted-label loss does.
def _build_loss_mask_label_aligned(self, processed_example):
    if not self.answer_only_loss:
        return _orig_build_loss_mask(self, processed_example)
    seq_boundaries = processed_example["seq_boundaries"]
    return np.concatenate(
        [
            processed_example["loss_mask"][seq_boundaries[i] + 1 : seq_boundaries[i + 1]]
            for i in range(len(seq_boundaries) - 1)
        ]
    )


GPTSFTPackedDataset._build_loss_mask = _build_loss_mask_label_aligned

import megatron.bridge.data.samplers as _bridge_samplers

from primus.core.patches import PatchContext, register_patch


# Score every validation sample once.
@register_patch(
    "mlperf_llama2_70b.full_validation_pass",
    backend="megatron_bridge",
    phase="before_train",
    priority=60,
    description="Score every validation row exactly once per eval",
)
def _patch_full_validation_pass(ctx: PatchContext) -> None:
    orig_build_loader = _bridge_samplers.build_pretraining_data_loader

    def _build_loader(dataset, consumed_samples, dataloader_type, *args, **kwargs):
        if kwargs.get("name") != "validation" or dataset is None:
            return orig_build_loader(dataset, consumed_samples, dataloader_type, *args, **kwargs)
        kwargs.pop("name")
        kwargs.pop("eval_iters", None)
        kwargs["drop_last"] = False
        return orig_build_loader(dataset, 0, "batch", *args, **kwargs)

    _bridge_samplers.build_pretraining_data_loader = _build_loader


import functools

import megatron.bridge.training.setup as _bridge_setup


class _DeferredDataIterator:
    def __init__(self, build, index):
        self._build = build
        self._index = index


def _materialize_data_iterators(*iterators) -> None:
    # Morph in place so every holder of the placeholder, including Bridge's pretrain(), sees the real iterator.
    for it in iterators:
        if isinstance(it, _DeferredDataIterator):
            real = it._build()[it._index]
            if real is not None:
                it.__class__ = type(real)
                it.__dict__ = real.__dict__


# The clock must start before the dataset is touched, so the loaders are built after run_start.
@register_patch(
    "mlperf_llama2_70b.defer_data_iterators",
    backend="megatron_bridge",
    phase="before_train",
    priority=60,
    description="Build the train and validation iterators after run_start",
)
def _patch_defer_data_iterators(ctx: PatchContext) -> None:
    orig_setup_data_iterators = _bridge_setup.setup_data_iterators

    def _setup_data_iterators(cfg, train_state, model_length, train_valid_test_datasets_provider):
        build = functools.cache(
            lambda: orig_setup_data_iterators(
                cfg=cfg,
                train_state=train_state,
                model_length=model_length,
                train_valid_test_datasets_provider=train_valid_test_datasets_provider,
            )
        )
        train_state.do_train = cfg.train.train_iters > 0
        train_state.do_valid = cfg.train.eval_iters > 0
        train_state.do_test = False
        return _DeferredDataIterator(build, 0), _DeferredDataIterator(build, 1), None

    _bridge_setup.setup_data_iterators = _setup_data_iterators


from megatron.bridge.recipes.utils.finetune_utils import default_squad_config
from megatron.bridge.recipes.utils.tokenizer_utils import (
    DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
)
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
    should_profile_rank,
)

from primus.backends.megatron_bridge.patches.mlperf_llama2_70b.lora import LoRA

_profiling_utils = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "utilities", "metrics", "profiling"
)
if not os.path.isdir(_profiling_utils):
    _profiling_utils = "/workspace/utilities/metrics/profiling"
sys.path.insert(0, _profiling_utils)
# utilities/ is mounted only by dev/run_with_docker_dev.sh; a non-profiled run
# launched without it must still import this recipe.
try:
    from full_run_profiler_handler import FullRunProfilerHandler
except ImportError:
    FullRunProfilerHandler = None
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tensor_inspect import (
    tensor_inspect_end_if_enabled,
    tensor_inspect_step_if_enabled,
)
from megatron.bridge.training.utils import flop_utils
from megatron.bridge.training.utils import train_utils as _megatron_train_utils
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import (
    calc_params_l2_norm,
    prepare_forward_step_func,
    training_log,
)

# train_utils binds print_rank_* at import time; rebind so training_log always uses Primus loggers even if
# this module was imported after an earlier train_utils load.
_megatron_train_utils.print_rank_0 = log_rank_0
_megatron_train_utils.print_rank_last = log_rank_last
from megatron.bridge.training.train import (
    _delete_cuda_graphs,
    disable_forward_pre_hook,
    enable_forward_pre_hook,
    maybe_check_weight_hash_across_dp_replicas,
    maybe_report_stragglers,
    maybe_run_manual_gc,
    maybe_synchronize_training_step,
    should_disable_forward_pre_hook,
    train_step,
)
from megatron.bridge.utils.common_utils import is_last_rank

# Importing this module installs a monkey-patch that swaps Megatron-Bridge's
from primus.backends.megatron_bridge.recipes.mlperf_llama2_70b import (  # noqa: F401
    nemo_loss as _nemo_loss,
)

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
    cfg.fp8_dot_product_attention = _env_flag("FP8_DPA", "false")
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
    enable_primus_turbo: bool
    use_turbo_attention: bool
    use_turbo_gemm: bool


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_flag(name: str, default: str = "false") -> bool:
    return _as_bool(os.environ.get(name, default), default=_as_bool(default))


#: Attributes ``primus_turbo.py`` reads off ``get_args()`` without a getattr
#: default, so a missing one is an AttributeError rather than a fallback. The
#: values here are the inert choices for a dense 1x8x1 LoRA run; anything the
#: Primus config already carries is left alone.
_TURBO_ARGS_FALLBACKS = (
    ("offload", False),
    ("offload_ops", ()),
    ("deterministic_mode", False),
    ("patch_primus_pipeline", False),
    ("patch_zero_bubble", False),
    ("enable_zero_bubble", False),
    ("pp_algorithm", ""),
    ("enable_turbo_attention_float8", False),
    ("use_turbo_grouped_gemm", False),
    ("moe_router_force_load_balancing", False),
    ("moe_router_padding_for_quantization", False),
    ("turbo_deepep_num_cu", 0),
    ("turbo_deepep_use_comm_stream", False),
    ("turbo_sync_free_moe_stage", 0),
)


def _turbo_log(message: str) -> None:
    """Record which Turbo path was taken, under TURBO_DEBUG=1.

    Not the Primus logger and not stderr: the recipe sets
    ``stderr_sink_level: ERROR``, so anything below that is dropped and cannot
    be used to confirm whether an install ran.
    """
    if os.getenv("TURBO_DEBUG") != "1":
        return
    try:
        with open("/results/turbo_decide.log", "a") as fh:
            fh.write(message + "\n")
    except Exception:
        pass


def _run_forward_pre_hooks(module, args) -> None:
    """Run ``module``'s forward-pre-hooks without calling its forward.

    The distributed optimizer waits on and dispatches each submodule's parameter
    all-gather from these hooks, so skipping linear_fc1/linear_fc2.forward must
    not skip them.
    """
    for hook in module._forward_pre_hooks.values():
        result = hook(module, args)
        if result is not None:
            args = result if isinstance(result, tuple) else (result,)
    kwargs: dict = {}
    for hook in getattr(module, "_forward_pre_hooks_with_kwargs", {}).values():
        result = hook(module, args, kwargs)
        if result is not None:
            args, kwargs = result


def _install_frozen_fused_mlp(state_mgr) -> None:
    import turbo_fp4_frozen
    from megatron.core.transformer.mlp import MLP
    from primus_turbo.pytorch.ops.normalization import rmsnorm

    orig_forward = MLP.forward
    logged = set()

    def forward(self, hidden_states, per_token_scale=None):
        if per_token_scale is None and state_mgr.is_turbo_fp4_enabled():
            fc1, fc2 = self.linear_fc1, self.linear_fc2
            w1, w2 = fc1._parameters.get("weight"), fc2._parameters.get("weight")
            quant = state_mgr.get_turbo_quant_config()
            config = quant.data() if quant is not None and quant.mxfp4_scaling() else None
            x2d = hidden_states.reshape(-1, hidden_states.shape[-1])
            checks = {
                "weights": w1 is not None and w2 is not None,
                "no_bias": not any(
                    isinstance(b, torch.Tensor) and b.numel() > 0
                    for b in (getattr(fc1, "bias", None), getattr(fc2, "bias", None))
                ),
                "rmsnorm": self.config.normalization == "RMSNorm",
                "norm_in_fc1": getattr(fc1, "layer_norm_weight", None) is not None,
            }
            if all(checks.values()):
                checks["frozen_pairs_and_shapes"] = turbo_fp4_frozen.fused_mlp_eligible(x2d, w1, w2, config)
            if all(checks.values()):
                _run_forward_pre_hooks(fc1, (hidden_states,))
                _run_forward_pre_hooks(fc2, (hidden_states,))
                if getattr(fc1, "_skip_fused_norm", False):
                    x = x2d
                else:
                    norm_weight = fc1.layer_norm_weight
                    if fc1.zero_centered_gamma:
                        norm_weight = norm_weight + 1
                    x = rmsnorm(x2d, norm_weight, fc1.eps)
                y = turbo_fp4_frozen.frozen_fused_mlp(x, w1, w2, config)
                if "fused" not in logged:
                    logged.add("fused")
                    _turbo_log(f"[FusedDenseMLP] fused FlyDSL MLP taken, x={tuple(x.shape)}")
                return y.view(*hidden_states.shape[:-1], y.shape[-1]), None
            if "fallback" not in logged:
                logged.add("fallback")
                _turbo_log(
                    f"[FusedDenseMLP] fallback to stock MLP under MXFP4: x={tuple(x2d.shape)} "
                    f"dtype={x2d.dtype} config={config is not None} checks={checks} "
                    f"fc1={type(fc1).__name__} w1={type(w1).__name__} "
                    f"w1_pair={hasattr(w1, '_turbo_fp4_pair')} w1_rg={getattr(w1, 'requires_grad', None)} "
                    f"preshuffle={getattr(config, 'use_preshuffle', None)}"
                )
        return orig_forward(self, hidden_states, per_token_scale)

    MLP.forward = forward
    _turbo_log("[FusedDenseMLP] MLP.forward routed through turbo_fp4_frozen.frozen_fused_mlp")


def _install_fused_residual_norm() -> None:
    if int(os.getenv("HEALING_ITER", "0") or 0) > 0 and not _env_flag("PRIMUS_TURBO_FP8_HEALING", "false"):
        raise RuntimeError(
            "PRIMUS_FUSED_RESIDUAL_NORM with healing needs PRIMUS_TURBO_FP8_HEALING=1: "
            "TE's LayerNormLinear ignores _skip_fused_norm and would normalize twice"
        )
    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frn

    # LoRALinear keeps the fused-norm linear in to_wrap and does not forward its
    # attributes, so the stock lookup misses the QKV gamma and the cross-layer carry
    # is never consumed.
    def _qkv_fused_ln_params(layer):
        attn = getattr(layer, "self_attention", None)
        qkv = getattr(attn, "linear_qkv", None) if attn is not None else None
        qkv = getattr(qkv, "to_wrap", qkv)
        ln_w = getattr(qkv, "layer_norm_weight", None) if qkv is not None else None
        if ln_w is None:
            return None
        eps = getattr(qkv, "eps", None) or getattr(layer.config, "layernorm_epsilon", 1e-5)
        return qkv, ln_w, eps

    frn._qkv_fused_ln_params = _qkv_fused_ln_params

    orig_can_fuse = frn._can_fuse
    orig_run = frn._run_rmsnorm_residual
    seen = {}

    def _can_fuse(layer):
        ok = orig_can_fuse(layer)
        if "can_fuse" not in seen:
            seen["can_fuse"] = ok
            _turbo_log(
                f"[FusedResidualNorm] first layer can_fuse={ok} "
                f"hidden_dropout={getattr(layer, 'hidden_dropout', None)} "
                f"qkv_params={_qkv_fused_ln_params(layer) is not None} "
                f"v2_linked={getattr(layer, '_v2_next_layer', None) is not None}"
            )
        return ok

    def _run_rmsnorm_residual(*args, **kwargs):
        if "run" not in seen:
            seen["run"] = True
            _turbo_log(f"[FusedResidualNorm] rmsnorm_residual taken, x={tuple(args[0].shape)}")
        return orig_run(*args, **kwargs)

    frn._can_fuse = _can_fuse
    frn._run_rmsnorm_residual = _run_rmsnorm_residual
    if not frn.install():
        raise RuntimeError("fused_residual_rmsnorm.install() failed; see [fused_residual_rmsnorm] log")
    _turbo_log(
        "[FusedResidualNorm] installed "
        f"V1={_env_flag('PRIMUS_FUSED_RESIDUAL_NORM', 'false')} "
        f"V2={_env_flag('PRIMUS_FUSED_RESIDUAL_NORM_V2', 'false')}"
    )


def _install_primus_args_bridge():
    """Route Turbo's ``get_args()`` at the Primus config on the Bridge path.

    ``primus_turbo.py`` imports ``get_args`` from ``megatron.training.global_vars``,
    whose ``_GLOBAL_ARGS`` only ever gets set by Megatron-LM's own argparse
    entrypoint. Megatron-Bridge never goes through it, so every Turbo module that
    reads a config value -- ``PrimusTurboAttention.__init__`` among them -- dies on
    ``AssertionError: args is not initialized``. Primus's *own* globals are
    populated here, so fall back to those.

    Returns the Primus args object.
    """
    from primus.backends.megatron.training.global_vars import get_primus_args

    try:
        primus_args = get_primus_args()
    except Exception as exc:
        raise RuntimeError("Primus args are not initialized; cannot wire Primus-Turbo.") from exc

    for _name, _value in _TURBO_ARGS_FALLBACKS:
        if not hasattr(primus_args, _name):
            setattr(primus_args, _name, _value)

    # Populate Megatron's own ``_GLOBAL_ARGS`` rather than only wrapping
    # ``get_args``. Turbo modules do ``from megatron.training.global_vars import
    # get_args`` at import time, so they hold a direct reference to the original
    # function and never see a replaced module attribute; filling the global the
    # original function reads fixes every consumer at once.
    try:
        import megatron.training.global_vars as megatron_global_vars

        if getattr(megatron_global_vars, "_GLOBAL_ARGS", None) is None:
            megatron_global_vars.set_args(primus_args)
    except Exception:
        pass

    def _safe_get_args(orig):
        def _inner(*args, **kwargs):
            try:
                return orig(*args, **kwargs)
            except AssertionError:
                return get_primus_args()

        return _inner

    from primus.backends.megatron.core.extensions import (
        primus_turbo as primus_turbo_ext,
    )

    primus_turbo_ext.get_args = _safe_get_args(primus_turbo_ext.get_args)
    primus_turbo_ext.use_split_wgrad_op = lambda: False

    # Every Turbo linear picks its FP4 path with `if get_num_microbatches() == 1`,
    # and this recipe is GBS 8 / MBS 1 / DP 8 -- exactly one microbatch. That
    # branch is documented in-tree as "quantizes the bf16 weight internally and
    # has no persistent cache": it hands the raw parameter to gemm_fp4, which
    # re-quantizes it on every step and, with no transpose cache, dequantizes and
    # re-quantizes again for the column direction. Both are pure HBM traffic on a
    # weight that LoRA freezes. The >1 branch instead builds a
    # QuantizedTensorPair once via create_quantized_weight and reuses it.
    #
    # The count is never read for anything else -- all ten uses in that module are
    # this same ==1 test -- so reporting 2 selects the cached path everywhere.
    if _env_flag("PRIMUS_TURBO_FP4_WEIGHT_CACHE", "false"):
        _orig_num_mb = primus_turbo_ext.get_num_microbatches

        def _num_microbatches_at_least_two():
            try:
                return max(2, int(_orig_num_mb()))
            except Exception:
                return 2

        primus_turbo_ext.get_num_microbatches = _num_microbatches_at_least_two
        _turbo_log("[TurboWeightCache] forced the cached QuantizedTensorPair path")

        # The cached path still rebuilds its buffers on the first microbatch of
        # every step, which re-quantizes the weight. For a LoRA-frozen base
        # weight that is both wasted and impossible: the parameter is a TE
        # MXFP4Tensor, and re-quantizing it lands in TE's dequantize stub
        # ("MXFP4 dequantization from packed FP4 not yet implemented"). When
        # pre_quantize_mxfp4 has already built the pair from the high-precision
        # weight it hangs it on the parameter, so reuse that instead.
        _orig_make_buffers = getattr(primus_turbo_ext, "_maybe_create_quantized_weight_buffers", None)

        if _orig_make_buffers is not None:

            _fp8_healing = _env_flag("PRIMUS_TURBO_FP8_HEALING", "false")

            def _reuse_prequantized_weight_buffers(weight, *args, **kwargs):
                pair = getattr(weight, "_turbo_fp4_pair", None)
                if pair is not None:
                    return pair
                # After healing the parameter is the FP8 stash (TE Float8Tensor,
                # per-tensor E4M3, scale 1); serve it to the frozen FP8 GEMM as-is.
                if _fp8_healing:
                    import turbo_fp8_frozen

                    w8 = turbo_fp8_frozen.frozen_fp8_weight(weight)
                    if w8 is not None:
                        return w8, None
                return _orig_make_buffers(weight, *args, **kwargs)

            primus_turbo_ext._maybe_create_quantized_weight_buffers = _reuse_prequantized_weight_buffers

        # The cached path then bridges the quantized weight's gradient back onto
        # the parameter's main_grad, and asserts main_grad exists. Under LoRA the
        # base weights are frozen, so DDP never allocates one and there is no
        # base wgrad to bridge -- the adapters carry the gradient. Skip the
        # bridge for those and hand the GEMM the pair directly.
        _orig_bridge = getattr(primus_turbo_ext, "_bridge_weight_grad", None)

        if _orig_bridge is not None:

            def _bridge_weight_grad_if_trainable(x, weight, weight_buffer, fuse_wgrad_accum=False):
                if not hasattr(weight, "main_grad"):
                    return x, weight_buffer
                return _orig_bridge(x, weight, weight_buffer, fuse_wgrad_accum=fuse_wgrad_accum)

            primus_turbo_ext._bridge_weight_grad = _bridge_weight_grad_if_trainable

        # Skipping the bridge is not enough on its own. forward_internal asks
        # _fuse_wgrad_accum_pattern separately, and with a pattern the gemm runs
        # a beta=1 epilogue that accumulates into main_grad *on the quantized
        # buffer* -- a target only the bridge installs. Skip one without the
        # other and the gemm accumulates into an unset target, which is how the
        # first working run produced a rising loss and then NaN.
        _orig_fuse_pattern = getattr(primus_turbo_ext, "_fuse_wgrad_accum_pattern", None)

        if _orig_fuse_pattern is not None:

            def _fuse_wgrad_accum_pattern_if_trainable(config, weight):
                if not hasattr(weight, "main_grad"):
                    return None
                return _orig_fuse_pattern(config, weight)

            primus_turbo_ext._fuse_wgrad_accum_pattern = _fuse_wgrad_accum_pattern_if_trainable

    return primus_args


def _install_turbo_attention(*, attention_float8: bool) -> None:
    """Swap TE fused attention for Turbo's flash-attention on the Bridge path.

    Only reachable with ``use_turbo_gemm=False``: ``_install_flydsl_fp4_gemms``
    pins ``use_turbo_attention`` off and asserts the spec provider hands back
    TE attention, so the two are mutually exclusive by construction.
    """
    primus_args = _install_primus_args_bridge()
    primus_args.enable_primus_turbo = True
    primus_args.use_turbo_attention = True
    primus_args.enable_turbo_attention_float8 = attention_float8
    if hasattr(primus_args, "use_turbo_gemm"):
        primus_args.use_turbo_gemm = False

    # PrimusTurboAttention.forward always passes ``sink=`` (sink attention,
    # upstream PR 208), but only the BF16 ``flash_attn_func`` in this
    # primus-turbo pin grew that parameter -- the FP8 entry point raises
    # TypeError. Llama-2 has no sink parameters, so the argument is always None
    # and dropping it is a no-op; refuse rather than silently ignore a real one.
    if attention_float8:
        import inspect as _insp

        import primus_turbo.pytorch as _turbo_pt

        _fp8_attn = _turbo_pt.ops.flash_attn_fp8_func
        try:
            _params = _insp.signature(_fp8_attn).parameters
            _takes_sink = "sink" in _params or any(
                p.kind == _insp.Parameter.VAR_KEYWORD for p in _params.values()
            )
        except (TypeError, ValueError):
            _takes_sink = False
        if not _takes_sink:

            def _fp8_attn_without_sink(*args, sink=None, **kwargs):
                if sink is not None:
                    raise ValueError(
                        "FP8 attention in this primus-turbo build has no sink "
                        "support; disable use_sink_attention."
                    )
                return _fp8_attn(*args, **kwargs)

            _turbo_pt.ops.flash_attn_fp8_func = _fp8_attn_without_sink

    import megatron.core.extensions.transformer_engine as te_spec_mod
    from megatron.core.extensions.transformer_engine import TELinear

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboAttention,
    )
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    te_spec_mod.TESpecProvider = PrimusTurboSpecProvider
    for _mod_path, _attr in (
        ("megatron.core.extensions.transformer_engine_spec_provider", "TESpecProvider"),
        ("megatron.core.models.gpt.gpt_layer_specs", "TESpecProvider"),
    ):
        try:
            import importlib

            setattr(importlib.import_module(_mod_path), _attr, PrimusTurboSpecProvider)
        except Exception:
            pass

    provider = PrimusTurboSpecProvider()
    attn_cls = provider.core_attention()
    linear_cls = provider.linear()
    if attn_cls is not PrimusTurboAttention:
        raise RuntimeError(
            "Turbo attention requested, but spec provider.core_attention() is "
            f"{attn_cls.__module__}.{attn_cls.__name__}"
        )
    if linear_cls is not TELinear:
        raise RuntimeError(
            "Turbo attention runs with TE linear layers, but spec "
            f"provider.linear() is {linear_cls.__module__}.{linear_cls.__name__}"
        )

    import turbo_attention_thd

    turbo_attention_thd.install()

    _turbo_log(
        "[TurboAttention] PrimusTurboAttention selected; "
        f"linear={linear_cls.__name__}; fp8={attention_float8}; "
        f"backend={os.getenv('PRIMUS_TURBO_ATTN_BACKEND', '<unset>')}"
    )


def _install_flydsl_fp4_gemms(
    *,
    enable_primus_turbo: bool,
    use_turbo_gemm: bool,
    use_turbo_attention: bool,
) -> None:
    """Wire Primus-Turbo FlyDSL MXFP4 GEMMs on Bridge, optionally with Turbo attention.

    Megatron-Bridge does not reliably apply the Megatron ``before_train`` Turbo
    patches, so this recipe:
      * sets Primus args (``use_turbo_gemm``)
      * replaces ``TESpecProvider`` with ``PrimusTurboSpecProvider``
      * nests ``primus_turbo_fp4_autocast`` around Megatron ``get_fp4_context``
      * falls back to TE Linear.forward after MXFP4 healing (FP8 DelayedScaling)
      * with ``use_turbo_attention``, takes ``PrimusTurboAttention`` (FlyDSL flash
        attention) with THD support from ``turbo_attention_thd``; else TE attention
    """
    if not enable_primus_turbo or not use_turbo_gemm:
        return

    try:
        import primus_turbo  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "use_turbo_gemm=True requires primus_turbo in the image. "
            "Rebuild llama2_sft/primus with Primus-Turbo installed."
        ) from exc

    os.environ.setdefault("PRIMUS_TURBO_GEMM_BACKEND", "flydsl")
    os.environ.setdefault("PRIMUS_TURBO_AUTO_TUNE", "1")

    primus_args = _install_primus_args_bridge()

    primus_args.enable_primus_turbo = True
    primus_args.use_turbo_attention = bool(use_turbo_attention)
    if use_turbo_attention:
        primus_args.enable_turbo_attention_float8 = False

    # Gradient stochastic rounding. Primus resolves this off the model config
    # and falls back to get_args(), which _install_primus_args_bridge populates,
    # so setting it here is enough to reach fp4_utils' quant config.
    #
    # Worth having because the FlyDSL path diverges at the recipe's 6e-4 while
    # training cleanly at 2e-4: round-to-nearest on MXFP4 gradients accumulates
    # a bias every step, and SR is the standard remedy for exactly that. The
    # coarse 32x32 weight scales make this regime more bias-prone than TE's
    # 1x32, which is consistent with TE being stable at the same LR.
    if "MXFP4_GRADIENT_SR" in os.environ:
        primus_args.mxfp4_gradient_stochastic_rounding = _env_flag("MXFP4_GRADIENT_SR", "false")
        _turbo_log(
            "[TurboGradSR] mxfp4_gradient_stochastic_rounding="
            f"{primus_args.mxfp4_gradient_stochastic_rounding}"
        )

    if hasattr(primus_args, "use_turbo_gemm"):
        primus_args.use_turbo_gemm = True
    elif hasattr(primus_args, "use_turbo_parallel_linear"):
        primus_args.use_turbo_parallel_linear = True

    import inspect as _inspect

    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TELinear,
        TERowParallelLinear,
    )

    # The Megatron pin calls the TE 2.7+ graph-capture setter, while the TE
    # pin in this image stores the same device flag directly on its
    # quantization state. Backfill the setter so local full-iteration graphs
    # can capture MXFP4 weight-cache update decisions.
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboColumnParallelLinear,
        PrimusTurboLayerNormColumnParallelLinear,
        PrimusTurboLinear,
        PrimusTurboLowPrecisionGlobalStateManager,
        PrimusTurboQuantConfig,
        PrimusTurboRowParallelLinear,
        primus_turbo_fp4_autocast,
    )
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    if not hasattr(FP8GlobalStateManager, "set_skip_fp8_weight_update_tensor"):

        @classmethod
        def _set_skip_fp8_weight_update_tensor(cls, skip: bool) -> None:
            state = cls.quantization_state
            if state.skip_fp8_weight_update_tensor is None:
                state.skip_fp8_weight_update_tensor = torch.empty(1, dtype=torch.float32, device="cuda")
            state.skip_fp8_weight_update_tensor.fill_(skip)

        FP8GlobalStateManager.set_skip_fp8_weight_update_tensor = _set_skip_fp8_weight_update_tensor

    def _drop_unsupported_init_kwargs(cls):
        orig = cls.__init__
        params = _inspect.signature(orig).parameters
        if any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return

        def _init(self, *args, **kwargs):
            allowed = {k: v for k, v in kwargs.items() if k in params}
            return orig(self, *args, **allowed)

        cls.__init__ = _init

    for _te_cls in (
        TELinear,
        TEColumnParallelLinear,
        TERowParallelLinear,
        TELayerNormColumnParallelLinear,
    ):
        _drop_unsupported_init_kwargs(_te_cls)

    import megatron.core.extensions.transformer_engine as te_spec_mod

    te_spec_mod.TESpecProvider = PrimusTurboSpecProvider
    try:
        import megatron.core.extensions.transformer_engine_spec_provider as te_spec_provider_mod

        te_spec_provider_mod.TESpecProvider = PrimusTurboSpecProvider
    except Exception:
        pass
    try:
        from megatron.core.models.gpt import gpt_layer_specs

        gpt_layer_specs.TESpecProvider = PrimusTurboSpecProvider
    except Exception:
        pass

    provider = PrimusTurboSpecProvider()
    linear_cls = provider.linear()
    attn_cls = provider.core_attention()
    norm_cls = provider.layer_norm()
    if linear_cls is not PrimusTurboLinear:
        raise RuntimeError(
            "FlyDSL FP4 GEMMs requested, but spec provider.linear() is "
            f"{linear_cls.__module__}.{linear_cls.__name__}"
        )
    if use_turbo_attention:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboAttention,
        )

        if attn_cls is not PrimusTurboAttention:
            raise RuntimeError(
                "Turbo attention requested, but spec provider.core_attention() is "
                f"{attn_cls.__module__}.{attn_cls.__name__}"
            )
        import turbo_attention_thd

        turbo_attention_thd.install()
    elif attn_cls is not TEDotProductAttention:
        raise RuntimeError(
            "TE fused attention required, but spec provider.core_attention() is "
            f"{attn_cls.__module__}.{attn_cls.__name__}"
        )

    def _te_fallback_forward(cls):
        orig = cls.forward

        def forward(self, *args, **kwargs):
            if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled() or (
                PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled()
            ):
                return orig(self, *args, **kwargs)
            return super(cls, self).forward(*args, **kwargs)

        cls.forward = forward

    for turbo_cls in (
        PrimusTurboLinear,
        PrimusTurboRowParallelLinear,
        PrimusTurboColumnParallelLinear,
        PrimusTurboLayerNormColumnParallelLinear,
    ):
        # The Turbo forward reads self.te_quant_params, which a newer mcore sets
        # from TE quantization params this image's TE does not have -- the
        # container's own [compat] step stubs those types out, and
        # _drop_unsupported_init_kwargs strips the kwarg, so nothing ever assigns
        # it. None is the correct value, not merely a safe one:
        # _get_fp8_autocast_for_quant_params returns nullcontext() for None, and
        # quantization here comes from the Turbo MXFP4 autocast, not from TE.
        if not hasattr(turbo_cls, "te_quant_params"):
            turbo_cls.te_quant_params = None
        _te_fallback_forward(turbo_cls)

    # LoRA sets return_layernorm_output on every TELayerNormColumnParallelLinear and
    # feeds the adapter the norm output it gets back. The Turbo forward_internal
    # ignores the flag and returns (out, None), so Bridge falls back to the raw
    # residual stream as the adapter input. LoRA B starts at zero, so step 1 is
    # exact and every step after it is wrong (loss 4.1 -> 9.3 at step 2).
    # Normalize here and run the stock forward on its pre-normalized
    # _skip_fused_norm path, so the norm is still computed once.
    from primus_turbo.pytorch.ops.normalization import rmsnorm as _turbo_rmsnorm

    _orig_ln_forward_internal = PrimusTurboLayerNormColumnParallelLinear.forward_internal

    def _ln_forward_internal(self, x, is_first_microbatch=False):
        if not getattr(self, "return_layernorm_output", False):
            return _orig_ln_forward_internal(self, x, is_first_microbatch)
        if getattr(self, "_skip_fused_norm", False):
            out, bias = _orig_ln_forward_internal(self, x, is_first_microbatch)
            return (out, x), bias
        if self.config.normalization != "RMSNorm":
            raise RuntimeError(
                "return_layernorm_output on PrimusTurboLayerNormColumnParallelLinear "
                f"is only handled for RMSNorm, not {self.config.normalization}"
            )
        norm_weight = self.layer_norm_weight
        if self.zero_centered_gamma:
            norm_weight = norm_weight + 1
        ln_out = _turbo_rmsnorm(x, norm_weight, self.eps)
        prev_skip = self.__dict__.get("_skip_fused_norm", None)
        self._skip_fused_norm = True
        try:
            out, bias = _orig_ln_forward_internal(self, ln_out, is_first_microbatch)
        finally:
            if prev_skip is None:
                del self._skip_fused_norm
            else:
                self._skip_fused_norm = prev_skip
        return (out, ln_out), bias

    PrimusTurboLayerNormColumnParallelLinear.forward_internal = _ln_forward_internal
    _turbo_log("[TurboLNLinear] return_layernorm_output honored for LoRA adapters")

    from contextlib import contextmanager

    import megatron.core.fp4_utils as fp4_utils

    orig_fp4_context = fp4_utils.get_fp4_context

    _fp4_ctx_logged = []

    @contextmanager
    def _turbo_fp4_context(config, layer_no: int = -1, is_init: bool = False):
        inner = orig_fp4_context(config, layer_no, is_init)
        recipe = str(getattr(config, "fp4_recipe", "")).lower()
        use_mxfp4 = bool(getattr(config, "fp4", None)) and "mxfp4" in recipe
        if not _fp4_ctx_logged:
            _fp4_ctx_logged.append(True)
            _turbo_log(
                f"[TurboFP4Ctx] fp4={getattr(config, 'fp4', None)!r} "
                f"fp4_recipe={getattr(config, 'fp4_recipe', None)!r} "
                f"use_mxfp4={use_mxfp4} is_init={is_init}"
            )
        if is_init or not use_mxfp4:
            with inner:
                yield
            return
        from primus_turbo.pytorch.core.low_precision import (
            Format,
            ScaleDtype,
            ScalingGranularity,
            ScalingStrategy,
        )

        quant = PrimusTurboQuantConfig(
            format=Format.E2M1_X2,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            strategy=ScalingStrategy.DYNAMIC,
            scale_dtype=ScaleDtype.E8M0,
            block_size=32,
        )
        with inner:
            with primus_turbo_fp4_autocast(
                enabled=True,
                enabled_turbo=True,
                turbo_quant_config=quant,
            ):
                yield

    # Installing on fp4_utils alone is not enough, and that was the whole
    # problem: transformer_block.py does `from megatron.core.fp4_utils import
    # get_fp4_context` at import time, so it holds a direct reference and never
    # sees a replaced module attribute -- while being the module that actually
    # calls it. The Turbo MXFP4 autocast was therefore never established, every
    # PrimusTurbo linear fell back to TE through _te_fallback_forward, and the
    # GEMM backend knobs had nothing to select between. Primus ships both the
    # ROCm implementation and the consumer list, but registers them through its
    # own patch registry, which Megatron-Bridge never runs.
    # Healing swaps the weights to FP8 and repoints fp4_utils.get_fp4_recipe at the
    # healing recipe, but neither Turbo context looks at either: they keep entering
    # the Turbo MXFP4 autocast, now over FP8 weights with no Turbo pair. After healing,
    # hand back to Megatron's own context, which builds TE's autocast from the patched
    # recipe -- the same post-healing path the TE recipe takes.
    # PRIMUS_TURBO_FP8_HEALING=1 keeps the post-healing phase on the Turbo linears
    # instead: a Turbo FP8 autocast (tensorwise, HYBRID like the healing recipe, TE
    # FP8 off) whose GEMMs turbo_fp8_frozen serves from the FP8 stash on FlyDSL.
    _fp8_healing_ctx = _env_flag("PRIMUS_TURBO_FP8_HEALING", "false")
    if _fp8_healing_ctx:
        from primus_turbo.pytorch.core.low_precision import Format as _Fmt
        from primus_turbo.pytorch.core.low_precision import ScalingGranularity as _Gran

        from primus.backends.megatron.core.extensions.primus_turbo import (
            primus_turbo_fp8_autocast,
        )

        _fp8_quant = PrimusTurboQuantConfig(format=_Fmt.HYBRID, granularity=_Gran.TENSORWISE)

        @contextmanager
        def _turbo_fp8_healing_context():
            with primus_turbo_fp8_autocast(enabled=False, enabled_turbo=True, turbo_quant_config=_fp8_quant):
                yield

    def _healing_aware(ctx_fn):
        def _ctx(config, layer_no: int = -1, is_init: bool = False):
            if not is_init and not getattr(fp4_utils, "_mxfp4_phase", True):
                if _fp8_healing_ctx:
                    return _turbo_fp8_healing_context()
                return orig_fp4_context(config, layer_no, is_init)
            return ctx_fn(config, layer_no, is_init)

        return _ctx

    if _env_flag("PRIMUS_TURBO_FP4_AUTOCAST", "false"):
        fp4_ctx = _turbo_fp4_context
        try:
            from primus.backends.megatron.core.fp4_utils import (
                get_fp4_context as _primus_fp4_ctx,
            )

            fp4_ctx = _primus_fp4_ctx
        except Exception:
            pass
        _fp4_ctx_name = "primus" if fp4_ctx is not _turbo_fp4_context else "local"
        fp4_ctx = _healing_aware(fp4_ctx)

        import importlib as _importlib

        _fp4_consumers = [fp4_utils]
        for _mod_name in (
            "megatron.core.transformer.transformer_block",
            "megatron.core.ssm.mamba_block",
            "megatron.core.transformer.multi_token_prediction",
        ):
            try:
                _fp4_consumers.append(_importlib.import_module(_mod_name))
            except Exception:
                continue
        for _mod in _fp4_consumers:
            if getattr(_mod, "get_fp4_context", None) is not None:
                _mod.get_fp4_context = fp4_ctx
        _turbo_log(
            f"[TurboFP4Ctx] installed {_fp4_ctx_name} get_fp4_context (healing-aware) "
            f"into {[m.__name__ for m in _fp4_consumers]}"
        )
    else:
        fp4_utils.get_fp4_context = _healing_aware(_turbo_fp4_context)

    try:
        from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager

        flydsl = getattr(BackendType, "FLYDSL", None) or getattr(BackendType, "flydsl", None)
        if flydsl is not None:
            GlobalBackendManager.set_gemm_backend(flydsl)
    except Exception:
        pass

    # LoRA freezes every base weight, so stock gemm_fp4's activation column
    # half, grad_out column half and wgrad GEMM are discarded work (960 GEMMs a
    # step for 640 used). This routes frozen 1x32 weight pairs to a forward +
    # dgrad-only FlyDSL path; quantize and GEMM both run FlyDSL kernels.
    if _env_flag("PRIMUS_TURBO_FP4_FROZEN_WEIGHT", "false"):
        import turbo_fp4_frozen

        turbo_fp4_frozen.install()
        _turbo_log("[FrozenFP4] gemm_fp4 routed through turbo_fp4_frozen")

    # qkv and proj LoRA adapters folded into the frozen MXFP4 base GEMM's forward
    # epilogue and dgrad accumulate.
    if _env_flag("PRIMUS_TURBO_LORA_FP4_FUSED", "false"):
        if not _env_flag("PRIMUS_TURBO_FP4_FROZEN_WEIGHT", "false"):
            raise RuntimeError("PRIMUS_TURBO_LORA_FP4_FUSED=1 needs PRIMUS_TURBO_FP4_FROZEN_WEIGHT=1")
        import turbo_lora_fp4

        turbo_lora_fp4.install()
        _turbo_log("[LoRAFP4] LoRALinear qkv/proj routed through lora_frozen_linear")

    if _env_flag("PRIMUS_TURBO_FP8_HEALING", "false"):
        if not _env_flag("PRIMUS_TURBO_FP4_WEIGHT_CACHE", "false"):
            raise RuntimeError("PRIMUS_TURBO_FP8_HEALING=1 needs PRIMUS_TURBO_FP4_WEIGHT_CACHE=1")
        import turbo_fp8_frozen

        turbo_fp8_frozen.install()
        _turbo_log("[FrozenFP8] gemm_fp8 routed through turbo_fp8_frozen")

    # fc1 -> SwiGLU -> fc2 as two FlyDSL GEMMs with the SwiGLU (and its MXFP4
    # quantize) fused into the fc1 forward and fc2 dgrad epilogues, replacing TE's
    # gated_act / dgated_act. MLP weights
    # carry no LoRA adapters, so the frozen forward + dgrad path covers them.
    # Only taken inside the Turbo MXFP4 autocast; after healing the stock path runs.
    if _env_flag("PRIMUS_TURBO_FUSED_DENSE_MLP", "false"):
        if not _env_flag("PRIMUS_TURBO_FP4_FROZEN_WEIGHT", "false"):
            raise RuntimeError("PRIMUS_TURBO_FUSED_DENSE_MLP=1 needs PRIMUS_TURBO_FP4_FROZEN_WEIGHT=1")
        _install_frozen_fused_mlp(PrimusTurboLowPrecisionGlobalStateManager)

    # Residual add + RMSNorm in one Triton kernel, as in gpt-oss and the 8B. The
    # Primus patch is registered for the megatron backend only, so it never runs
    # under megatron_bridge unless installed here, before the model is built.
    if _env_flag("PRIMUS_FUSED_RESIDUAL_NORM", "false") or _env_flag(
        "PRIMUS_FUSED_RESIDUAL_NORM_V2", "false"
    ):
        _install_fused_residual_norm()

    _orig_log_rank_0(
        "[TurboGEMM] PrimusTurboSpecProvider selected PrimusTurboLinear; "
        f"attention={attn_cls.__name__}; "
        f"norm={norm_cls.__name__}; "
        f"PRIMUS_TURBO_GEMM_BACKEND={os.getenv('PRIMUS_TURBO_GEMM_BACKEND', '<unset>')}"
    )


def _install_te_graph_compat() -> bool:
    """Give mcore's CUDA-graph code the TE setter our pinned TE never shipped.

    ``megatron.core.transformer.cuda_graphs`` calls
    ``FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(bool)`` at three
    points around capture and replay. Our pinned TE has no such method -- it
    keeps the flag as ``quantization_state.skip_fp8_weight_update_tensor``, an
    ``Optional[Tensor]`` in which ``None`` means "do not skip" -- so enabling
    MCORE_CUDA_GRAPH dies with AttributeError during capture.

    TE does this exact job itself in ``graph.py`` when capturing its own graphs,
    so this mirrors that code rather than inventing semantics: a lazily
    allocated float32 tensor of shape (1,) on cuda, filled in place. float32
    rather than bool is TE's choice, and the consumers in ``linear.py``,
    ``layernorm_linear.py`` and ``layernorm_mlp.py`` read whatever TE allocated,
    so the dtype has to match theirs rather than look more correct than theirs.

    Filling in place is the entire point. Capture records the tensor by
    reference, which is why mcore hands the graph a tensor instead of a Python
    bool, so this must allocate at most once and never rebind the attribute
    afterwards -- a rebind would leave the captured graph reading a tensor
    nobody writes to again.

    Returns whether the method was added. False means TE already provides it, in
    which case TE's own implementation is left alone.
    """
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    if hasattr(FP8GlobalStateManager, "set_skip_fp8_weight_update_tensor"):
        return False

    @classmethod
    def set_skip_fp8_weight_update_tensor(cls, value: bool) -> None:
        qstate = cls.quantization_state
        if qstate.skip_fp8_weight_update_tensor is None:
            qstate.skip_fp8_weight_update_tensor = torch.empty(1, dtype=torch.float32, device="cuda")
        qstate.skip_fp8_weight_update_tensor.fill_(value)

    FP8GlobalStateManager.set_skip_fp8_weight_update_tensor = set_skip_fp8_weight_update_tensor
    return True


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
                global_batch_size=gbs,
                micro_batch_size=mbs,
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
                lowest_numerical_precision_in_linear=os.getenv(
                    "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR", "mxfp4"
                ),
                lowest_numerical_precision_in_attn=os.getenv(
                    "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN", "bfloat16"
                ),
                lowest_numerical_precision_in_comm=os.getenv(
                    "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM", "bfloat16"
                ),
            )
            _sft_logger.log_init_params(init_cfg)
        except ImportError:
            _orig_log_rank_0("primus_mllog is required when ENABLE_MLLOG=1")
            raise
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
        "attention_backend": os.environ.get("PRIMUS_ATTENTION_BACKEND", "fused"),
        "enable_primus_turbo": _env_flag("ENABLE_PRIMUS_TURBO", "true"),
        "use_turbo_attention": _env_flag("PRIMUS_USE_TURBO_ATTENTION", "false"),
        "use_turbo_gemm": _env_flag("PRIMUS_USE_TURBO_GEMM", "true"),
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
    enable_primus_turbo: bool = True,
    use_turbo_attention: bool = False,
    use_turbo_gemm: bool = True,
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
    model_cfg = bridge.to_megatron_provider(load_weights=False)  # GPTProvider
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
    # Keep the switches environment-controlled so the MI355X recipe can A/B
    # Megatron's TE fused cross entropy and full-iteration graph capture.
    model_cfg.cross_entropy_loss_fusion = _env_flag("CROSS_ENTROPY_LOSS_FUSION", "false")
    model_cfg.cross_entropy_fusion_impl = os.getenv("CROSS_ENTROPY_LOSS_FUSION_IMPL", "native")
    model_cfg.gradient_accumulation_fusion = _env_flag("GRADIENT_ACCUMULATION_FUSION", "false")
    model_cfg.bias_dropout_fusion = True
    model_cfg.fused_residual_rmsnorm = _env_flag("FUSED_RESIDUAL_RMSNORM", "false")
    model_cfg.fused_single_qkv_rope = _env_flag("FUSED_SINGLE_QKV_ROPE", "true")
    model_cfg.apply_rope_fusion = True
    if model_cfg.fused_single_qkv_rope and _env_flag("PRIMUS_TURBO_FLYDSL_ROPE", "false"):
        import turbo_rope

        turbo_rope.install()
        _turbo_log("[FlyDSLRoPE] TE fused QKV RoPE replaced by Turbo fused_qkv_rope")
    model_cfg.use_transformer_engine_op_fuser = _env_flag("USE_TE_OPS", "false")
    # Activation offload hint for TP paths (distinct from cpu_offloading / cpu_offloading_num_layers).
    model_cfg.cpu_offloading_activations = True

    # MXFP4 configuration
    model_cfg.fp8 = None
    model_cfg.fp4 = "e2m1"
    model_cfg.fp4_recipe = "mxfp4"
    model_cfg.fp8_param_gather = False
    model_cfg.grad_reduce_in_fp32 = False

    # Used when cuda_graph_impl is not "none" (harmless when graphs are disabled).
    model_cfg.cuda_graph_retain_backward_graph = True
    model_cfg.cuda_graph_use_single_mempool = True
    if _env_flag("MCORE_CUDA_GRAPH", "false"):
        from megatron.core.transformer.enums import CudaGraphScope

        model_cfg.cuda_graph_impl = os.getenv("CUDA_GRAPH_IMPL", "local").strip() or "local"
        model_cfg.cuda_graph_scope = [CudaGraphScope.full_iteration]
        model_cfg.cuda_graph_warmup_steps = int(os.getenv("CUDA_GRAPH_WARMUP_STEPS", "1"))
        model_cfg.use_te_rng_tracker = True
        # Only patch TE when graphs are actually on, so the no-flag baseline is
        # left byte-identical.
        shimmed = _install_te_graph_compat()
        # Announced on stderr, not log_rank_0: the configs pin PRIMUS_LOG_LEVEL=ERROR
        # and _log_suppression filters fd 1, which is why earlier graph runs could
        # not be confirmed from their logs either way.
        print(
            f"MCORE_CUDA_GRAPH enabled: impl={model_cfg.cuda_graph_impl} "
            f"scope={model_cfg.cuda_graph_scope} "
            f"warmup={model_cfg.cuda_graph_warmup_steps} te_graph_shim={shimmed}",
            file=sys.stderr,
            flush=True,
        )
    model_cfg.fp8 = None
    model_cfg.fp8_recipe = "delayed"
    model_cfg.fp8_amax_history_len = 4
    model_cfg.disable_parameter_transpose_cache = False
    model_cfg.fine_grained_activation_offloading = False
    model_cfg.use_transformer_engine_full_layer_spec = (
        False  # Doesn't work beacuse of RMSNorm is not supported in FusedLayerNorm
    )
    model_cfg.cpu_offloading = False
    model_cfg.cpu_offloading_num_layers = 0
    model_cfg.empty_unused_memory_level = (
        0  # 0: No empty, 1: Empty at end of eval, 2: Empty at end of eval and train.
    )
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
    # TE reads these two TransformerConfig fields into its fp8_dpa / fp8_mha recipe flags
    # (transformer_engine.py:1900), which is the only supported way to ask for
    # the FP8 fused-attention backend; the NVTE_* env vars alone do not reach it.
    model_cfg.fp8_dot_product_attention = _env_flag("FP8_DOT_PRODUCT_ATTENTION", "false")
    model_cfg.fp8_multi_head_attention = _env_flag("FP8_MULTI_HEAD_ATTENTION", "false")

    _turbo_on = _as_bool(enable_primus_turbo, True)
    _turbo_gemm = _as_bool(use_turbo_gemm, True)
    _turbo_attn = _as_bool(use_turbo_attention, False)
    _turbo_log(
        f"[TurboDecide] turbo={_turbo_on} gemm={_turbo_gemm} attn={_turbo_attn} "
        f"env_attn={os.getenv('PRIMUS_USE_TURBO_ATTENTION', '<unset>')} "
        f"env_gemm={os.getenv('PRIMUS_USE_TURBO_GEMM', '<unset>')}"
    )
    if _turbo_on and _turbo_attn and not _turbo_gemm:
        _install_turbo_attention(
            attention_float8=_env_flag("PRIMUS_TURBO_ATTENTION_FP8", "false"),
        )
    else:
        _install_flydsl_fp4_gemms(
            enable_primus_turbo=_turbo_on,
            use_turbo_gemm=_turbo_gemm,
            use_turbo_attention=_turbo_attn,
        )
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
        lr_warmup_iters=lr_warmup_iters,  # value 0 same as nemo
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
            # GPTSFTPackedDataset only shuffles (per epoch, seeded) when this is set; without it
            # train.npy is read in file order, which is all CRS reports followed by all GAO reports.
            max_train_samples=int(math.ceil(global_batch_size * train_iters * 1.005)),
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}. Expected 'squad' or 'mlperf_dataset'.")

    dataset_cfg.num_workers = 0
    dataset_cfg.memmap_workers = 1  # needs to be 1>0
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
            empty_unused_memory_level=0,  # 0: No empty, 1: Empty at end of eval, 2: Empty at end of eval and train.
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
                os.getenv("ENABLE_TRANSPOSE_CACHE", "").strip().lower() in ("1", "true", "yes", "on")
            ),
            fp8_param_gather=False,
        ),
        dataset=dataset_cfg,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=os.getenv("TORCHPROF_OUTPUT_DIR"),
            # Per-step / log-interval overhead toggles (keep off for throughput).
            log_params_norm=False,
            log_throughput=False,
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
            # PrimusTurboAttention's sharded_state_dict requests a
            # core_attention._extra_state entry that the converted Llama-2-70b
            # checkpoint does not carry. DIST_CKPT_STRICTNESS selects how that
            # missing key is treated. The recipe default is assume_ok_unexpected.
            dist_ckpt_strictness=os.getenv("DIST_CKPT_STRICTNESS", "assume_ok_unexpected"),
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
            use_pytorch_profiler=os.getenv("PRIMUS_PROFILE", "").lower() in ("1", "true", "yes", "on"),
            profile_step_start=int(os.getenv("PRIMUS_PROFILE_STEP_START", "140")),
            profile_step_end=int(os.getenv("PRIMUS_PROFILE_STEP_END", "144")),
            profile_ranks=[
                int(rank)
                for rank in os.getenv("PRIMUS_PROFILE_RANKS", "0,1,2,3,4,5,6,7").split(",")
                if rank.strip()
            ],
            # Off by default: shapes roughly double trace size and add launch
            # overhead. Needed to attribute GEMM time to individual shapes and
            # check them against the AITER tuning table.
            record_shapes=_env_flag("PRIMUS_PROFILE_RECORD_SHAPES", "false"),
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
        if (
            state.cfg.model.cuda_graph_impl == "local"
            and "full_iteration" in state.cfg.model.cuda_graph_scope
        ):
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
                        per_iter = torch.vstack([v.float() for v in val]).sum(
                            dim=0
                        )  # [Σ_micro sum, Σ_micro count]
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
    throughput: float = 0.0,  # samples/sec
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
            f"Skipping evaluation at {prefix}: target loss < " f"{MLPERF_TARGET_LOSS} already reached."
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
        state,
        forward_step_func,
        data_iterator,
        model,
        process_non_loss_data_func,
        config,
        verbose,
        non_loss_data_func,
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
            writer.add_scalar(
                "{} validation".format(key), total_loss_dict[key].item(), state.train_state.step
            )
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
            wandb_writer.log(
                {"{} validation".format(key): total_loss_dict[key].item()}, state.train_state.step
            )
            if state.cfg.logger.log_validation_ppl_to_tensorboard:
                wandb_writer.log({"{} validation ppl".format(key): ppl}, state.train_state.step)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, state.train_state.step, writer)

    string += "throughput: {:.6E} | ".format(throughput)
    string += "eval duration: {:.6E} | ".format(eval_duration)
    if writer:
        writer.add_scalar("throughput samples/sec", throughput, state.train_state.step)
        writer.add_scalar(
            "throughput samples/sec vs samples", throughput, state.train_state.consumed_train_samples
        )

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
        eval_loss_value = total_loss_dict["lm loss"].item()
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


def _warmup_frozen_fp8_gemms(model, global_state) -> None:
    """Compile the post-healing FlyDSL FP8 GEMMs before RUN_START, for every frozen base
    linear shape of the first decoder layer."""
    import turbo_fp8_frozen

    chunks = model if isinstance(model, (list, tuple)) else [model]
    layer0 = _mxh_extract_layers(chunks[0])[0]
    shapes = []
    for name, mod in layer0.named_modules():
        w = getattr(mod, "_parameters", {}).get("weight")
        if w is not None and w.ndim == 2 and not w.requires_grad:
            # PRIMUS_TURBO_FP4_DROP_TE_WEIGHT empties the TE storage that w.shape derives from.
            shape = (getattr(mod, "out_features", None), getattr(mod, "in_features", None))
            if None in shape:
                shape = tuple(w.shape)
            if tuple(w.shape) != shape:
                _turbo_log(
                    f"[FP8Warmup] {name}: {type(w).__name__} reports {tuple(w.shape)}, module says {shape}"
                )
            shapes.append(shape)
    cfg = global_state.cfg
    turbo_fp8_frozen.warmup(shapes, cfg.model.seq_length * cfg.train.micro_batch_size)


def _mxh_extract_layers(chunk):
    m = chunk
    while not hasattr(m, "decoder") and hasattr(m, "module"):
        m = m.module
    return m.decoder.layers


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

    if _env_flag("PRIMUS_TURBO_FP8_HEALING", "false") and int(os.getenv("HEALING_ITER", "0") or 0) > 0:
        _warmup_frozen_fp8_gemms(model, global_state)

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
    # TE FusedAdam keeps the bias-correction step in group["step"] and recreates it
    # as 1 on the next update when absent, so the first real step is step 1 for any
    # SYNTH_WARMUP_STEPS.
    for inner, saved in zip(inner_opts, saved_opt_states):
        for group, state in zip(inner.param_groups, saved):
            for key, val in state.items():
                group[key] = val
            group.pop("step", None)

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
    # WARMUP_KEEP_CACHE=1 keeps the warm-up's pool: releasing it here makes the
    # first measured steps map it back inside the clock.
    torch.cuda.synchronize()
    if not _env_flag("WARMUP_KEEP_CACHE", "false"):
        torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _drop_turbo_fp4_pairs(model) -> None:
    """Free the Turbo MXFP4 weight pairs before healing installs the FP8 weights.

    Healing releases the TE MXFP4 weight by unlinking the parameter, but the pair
    pre_quantize_mxfp4 built for the FlyDSL GEMM lives on the module buffers and on
    the parameter, outside healing's view. Left in place it is ~70 GiB resident next
    to the ~64 GiB FP8 weights, and the first post-healing step OOMs at 262 of 288
    GiB. After healing the Turbo linears take TE's forward and never read the pair.
    """
    chunks = model if isinstance(model, (list, tuple)) else [model]
    n_dropped = 0
    for chunk in chunks:
        for m in chunk.modules():
            dropped = False
            for attr in ("quantized_weight_buffer", "quantized_weight_t_buffer"):
                if getattr(m, attr, None) is not None:
                    setattr(m, attr, None)
                    dropped = True
            w = m._parameters.get("weight") if hasattr(m, "_parameters") else None
            if w is not None and hasattr(w, "_turbo_fp4_pair"):
                del w._turbo_fp4_pair
                dropped = True
            n_dropped += dropped
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    _turbo_log(f"[TurboWeights] dropped {n_dropped} Turbo MXFP4 weight pairs before healing")


_STASH_PREFETCH: dict = {"stream": None}


def _prefetch_fp8_stash(stash, budget_bytes: Optional[int] = None) -> int:
    """Copy the FP8 healing stash tensors still on the host to the GPU, up to ``budget_bytes``.

    The caching allocator keeps a pool per stream, so the destinations are allocated
    on the current stream and only the copy runs on the side stream. Allocating on the
    copy stream, as healing's own restore does, maps new memory instead of reusing the
    ~220 GiB cached on the compute stream, and near the HBM limit the allocator then
    frees its whole cache. Healing's ``.to(dev, non_blocking=True)`` hands back a tensor
    already on the device unchanged. Synchronize the stream before healing.
    """
    dev = torch.cuda.current_device()
    stream = _STASH_PREFETCH["stream"] or torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    copied = 0
    with torch.no_grad():
        for weight in (w for layer in stash for w in layer):
            data = getattr(weight, "_data", None)
            if data is None or data.is_cuda:
                continue
            nbytes = data.numel() * data.element_size()
            if budget_bytes is not None and copied + nbytes > budget_bytes:
                break
            dst = torch.empty(data.shape, dtype=data.dtype, device=dev)
            with torch.cuda.stream(stream):
                dst.copy_(data, non_blocking=True)
            weight._data = dst
            copied += nbytes
    _STASH_PREFETCH["stream"] = stream
    _turbo_log(f"[HealPrefetch] copying {copied / 2**30:.1f} GiB of the FP8 stash")
    return copied


def _install_healing_timers(mxh) -> None:
    """HEALING_TIMING=1: log wall time and GPU memory around each healing phase."""
    import time as _time

    def _mem():
        return (
            f"alloc={torch.cuda.memory_allocated() / 2**30:.1f}G "
            f"reserved={torch.cuda.memory_reserved() / 2**30:.1f}G"
        )

    def _timed(name, fn):
        def _inner(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            _turbo_log(f"[HealTiming] {name}: {_time.perf_counter() - t0:.3f}s {_mem()}")
            return out

        return _inner

    for name in ("_reset_full_iteration_cuda_graphs", "_set_quantized_params_cpu", "_reset_fp8_state"):
        setattr(mxh, name, _timed(name, getattr(mxh, name)))
    _orig_empty_cache = torch.cuda.empty_cache
    torch.cuda.empty_cache = _timed("empty_cache", _orig_empty_cache)
    globals()["_drop_turbo_fp4_pairs"] = _timed("_drop_turbo_fp4_pairs", _drop_turbo_fp4_pairs)
    _orig_apply = mxh.apply_healing_after_step

    def _apply(*args, **kwargs):
        _turbo_log(f"[HealTiming] before healing: {_mem()}")
        try:
            return _timed("apply_healing_after_step", _orig_apply)(*args, **kwargs)
        finally:
            torch.cuda.empty_cache = _orig_empty_cache

    mxh.apply_healing_after_step = _apply


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

    global_state.train_state.floating_point_operations_so_far
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
        assert (
            train_config.manual_gc_interval >= 0
        ), "Manual garbage collection interval should be larger than or equal to 0"
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

    get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    prof = None
    full_run_handler = None
    nsys_nvtx_context = None  # NVTX context for nsys profiling
    prof_config = config.profiling
    if torch.distributed.get_rank() == 0:
        _dbg_msg = (
            f"[PROFILER DEBUG] PRIMUS_PROFILE={os.getenv('PRIMUS_PROFILE','<unset>')}, "
            f"use_pytorch_profiler={prof_config.use_pytorch_profiler if prof_config else 'no_config'}, "
            f"should_profile={should_profile_rank(prof_config, 0) if prof_config else False}"
        )
        print(_dbg_msg, flush=True)
        try:
            with open("/results/profiler_debug.txt", "w") as _f:
                _f.write(_dbg_msg + "\n")
        except FileNotFoundError:
            pass
    if prof_config and should_profile_rank(prof_config, torch.distributed.get_rank()):
        if prof_config.use_pytorch_profiler:
            trace_dir = config.logger.tensorboard_dir or os.path.join(os.getcwd(), "torch_profiler_traces")
            window_size = int(os.getenv("PRIMUS_PROFILE_WINDOW_SIZE", "4"))
            total_steps = config.train.train_iters
            if FullRunProfilerHandler is None:
                raise ImportError(
                    "PyTorch profiling requested but full_run_profiler_handler is not "
                    f"importable; mount utilities/ at /workspace/utilities (looked in {_profiling_utils})"
                )
            full_run_handler = FullRunProfilerHandler(
                trace_dir,
                total_steps,
                window_size,
                record_shapes=prof_config.record_shapes,
            )
            print(
                f"[FullRunProfiler] Enabled (repeat=1 per window): "
                f"window_size={window_size}, total_steps={total_steps}, "
                f"midpoint_window={full_run_handler.midpoint_window}",
                flush=True,
            )

    # Initialize RPD profiler if enabled
    rpd = None
    rpd_status = None
    profiler_type = os.getenv("PROFILER", "")
    rpd_warmup_steps = max(0, int(os.getenv("RPD_WARMUP_STEPS", "0")))
    rpd_active_steps = int(os.getenv("RPD_ACTIVE_STEPS", "999999"))
    if profiler_type == "rpd":
        try:
            from rpdTracerControl import rpdTracerControl

            rank = torch.distributed.get_rank()
            rpd_filename = os.getenv("RPD_TRACE_FILENAME", f"trace.rpd")
            rpdTracerControl.setFilename(name=rpd_filename, append=True)
            rpd = rpdTracerControl()
            log_rank_0(
                f"RPD profiler initialized. Will start at step {rpd_warmup_steps} and stop at step {rpd_warmup_steps + rpd_active_steps}"
            )
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
        assert check_param_hashes_across_dp_replicas(
            model, cross_check=True
        ), "Parameter hashes not matching across DP replicas"
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
        forward_step_func,
        model,
        optimizer,
        scheduler,
        global_state,
        pg_collection,
    )

    sft_logger = _get_sft_logger()

    # MLPerf logging: transition from init to training (after warmup)
    if sft_logger is not None:
        sft_logger.log_init_stop_run_start()

    _materialize_data_iterators(train_data_iterator, valid_data_iterator)

    # Wall-clock for the training loop only (first through last train step; warmup eval disabled).
    training_wall_start = time.perf_counter()
    # Skip the first N interval evals (e.g. 48, 96, 144 when eval_interval=48); run from step 4*interval (192).
    eval_skip_first_n = 3

    # LIMIT_VAL_BATCHES=0 disables all evaluation (matches NeMo profiling behavior).
    limit_val_batches = float(os.getenv("LIMIT_VAL_BATCHES", "0"))
    if limit_val_batches == 0:
        log_rank_0("LIMIT_VAL_BATCHES=0: evaluation disabled for this run")

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
        if (
            rpd
            and rpd_status == "running"
            and global_state.train_state.step >= rpd_warmup_steps + rpd_active_steps
        ):
            log_rank_0(f"Stopping RPD profiling at step {global_state.train_state.step}")
            rpd.stop()
            rpd_status = "finished"

        # Handle profiling for this step
        if full_run_handler is not None:
            full_run_handler.pre_step(global_state.train_state.step)
        else:
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
            distributed_timeout_seconds_after_init = (
                global_state.cfg.dist.distributed_timeout_seconds_after_init
            )
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

        # Advance profiler after the training step completes
        if full_run_handler is not None:
            full_run_handler.post_step(global_state.train_state.step)

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
        # weights from the CPU stash (see ``mlperf_llama2_70b.mxfp4_healing``) and
        # switch ``megatron.core.fp4_utils`` out of the MXFP4 phase. No-op when
        # HEALING_ITER == 0 (default) so BF16/MXFP8 submission runs are
        # unaffected.
        try:
            from primus.backends.megatron_bridge.recipes.mlperf_llama2_70b import (
                mxfp4_healing as _mxh,
            )

            if _mxh.healing_iter() > 0:
                _real_empty_cache = torch.cuda.empty_cache
                _prefetch_gb = float(os.getenv("HEALING_PREFETCH_GB", "0") or 0)
                if (
                    _prefetch_gb > 0
                    and global_state.train_state.step + 1
                    == _mxh.healing_iter() - int(os.getenv("HEALING_PREFETCH_STEPS", "4"))
                    and not _mxh.is_fp8_healing_phase()
                    and _mxh._FP8_CPU_PARAMS
                ):
                    _prefetch_fp8_stash(_mxh._FP8_CPU_PARAMS, int(_prefetch_gb * 2**30))
                if (
                    global_state.train_state.step + 1 == _mxh.healing_iter()
                    and not _mxh.is_fp8_healing_phase()
                    and _mxh._FP8_CPU_PARAMS
                ):
                    # HEALING_KEEP_CACHE=1: healing's empty_cache calls unmap ~200 GiB
                    # (~1.6 s) that the first FP8 step then maps back (~2 s). Keeping
                    # the cache only works with the stash copied into it first: the
                    # restore allocates on its own stream, which cannot reuse the
                    # compute stream's cache, overflows HBM and makes the allocator
                    # free everything anyway.
                    if _env_flag("HEALING_KEEP_CACHE", "false"):
                        torch.cuda.empty_cache = lambda: None
                    if _env_flag("HEALING_TIMING", "false"):
                        _install_healing_timers(_mxh)
                    _drop_turbo_fp4_pairs(model)
                    if _env_flag("HEALING_KEEP_CACHE", "false"):
                        _prefetch_fp8_stash(_mxh._FP8_CPU_PARAMS)
                    if _STASH_PREFETCH["stream"] is not None:
                        _STASH_PREFETCH["stream"].synchronize()
                try:
                    _mxh.apply_healing_after_step(model, model_config, global_state.train_state.step)
                finally:
                    torch.cuda.empty_cache = _real_empty_cache
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
        num_skipped_samples_in_batch = (
            get_current_global_batch_size() - get_current_running_global_batch_size()
        )
        if train_config.decrease_batch_size_if_needed:
            assert num_skipped_samples_in_batch >= 0
        else:
            assert num_skipped_samples_in_batch == 0
        global_state.train_state.skipped_train_samples += num_skipped_samples_in_batch
        num_floating_point_operations_in_batch = flop_utils.num_floating_point_operations(config, batch_size)
        global_state.train_state.floating_point_operations_so_far += num_floating_point_operations_in_batch
        global_state.train_state.floating_point_operations_so_far
        num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

        # Logging.
        if hasattr(optimizer, "is_stub_optimizer") and not optimizer.is_stub_optimizer:
            loss_scale = optimizer.get_loss_scale().item()
        else:
            loss_scale = 1.0
        params_norm = None

        if config.logger.log_params_norm:
            params_norm = calc_params_l2_norm(
                model, model_config, use_megatron_fsdp=config.dist.use_megatron_fsdp
            )
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
            nemo_elapsed_time_per_iter_sec = nemo_style_iter_seconds_accum / nemo_style_iter_count
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
        if nemo_elapsed_time_per_iter_sec is not None:
            nemo_style_iter_seconds_accum = 0.0
            nemo_style_iter_count = 0

        if os.environ.get("LORA_FP4_PARITY") == "1":
            import turbo_lora_fp4

            turbo_lora_fp4.log_step(global_state.train_state.step, loss_dict, grad_norm)

        # MLPerf logging: per-step train loss
        if sft_logger is not None and not skipped_iter and loss_dict:
            sft_logger.on_train_step(
                step=global_state.train_state.step,
                loss_dict=loss_dict,
                lr=learning_rate,
                consumed_samples=global_state.train_state.consumed_train_samples,
            )

        if (
            limit_val_batches > 0
            and global_state.train_state.do_valid
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
            assert (
                timer.consumed_samples == global_state.train_state.consumed_train_samples
            ), "Timer and global_state sample mismatch"

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
                    global_state.train_state.consumed_train_samples,
                    eval_loss_value,
                )
                if target_hit:
                    should_exit = True
                    if rpd and rpd_status == "running":
                        log_rank_0(
                            f"Stopping RPD profiling at convergence (step {global_state.train_state.step})"
                        )
                        rpd.stop()
                        rpd_status = "finished"
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

    # Write profiler summary if active
    if full_run_handler is not None:
        full_run_handler.finalize()
        full_run_handler.write_summary()
        try:
            with open("/results/profiler_debug.txt", "a") as _f:
                _f.write(
                    f"[FullRunProfiler] Collected {len(full_run_handler.kernel_stats)} unique kernels "
                    f"across {full_run_handler.current_window} windows\n"
                )
        except FileNotFoundError:
            pass

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
    maybe_finalize_async_save(
        global_state=global_state, ckpt_cfg=config.checkpoint, blocking=True, terminate=True
    )
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
        maybe_finalize_async_save(
            global_state=global_state, ckpt_cfg=config.checkpoint, blocking=True, terminate=True
        )
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
