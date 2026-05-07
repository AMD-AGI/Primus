###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from contextlib import contextmanager, nullcontext
from typing import Callable, List, Optional, Tuple

import primus_turbo.pytorch as primus_turbo_torch
import torch
import torch.distributed as dist
import transformer_engine as te
from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEGroupedLinear,
    TELayerNormColumnParallelLinear,
    TELinear,
    TEQuantizationParams,
    TEQuantizationRecipe,
    TERowParallelLinear,
    condition_init_method,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_pg_size
from megatron.training.global_vars import get_args
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    ScalingStrategy,
    check_fp8_support,
    check_mxfp4_support,
    check_mxfp8_support,
)
from torch import Tensor
from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.fp8 import DelayedScaling, FP8GlobalStateManager, Recipe

from primus.core.pipeline_parallel.handler.offload_handler import OFFLOAD_BUFFER


def _call_fp8_autocast_enter(
    *,
    enabled: bool,
    calibrating: bool,
    fp8_recipe: Optional[Recipe],
    fp8_group: Optional[dist_group_type],
    _graph: bool,
) -> None:
    """Dispatch to whichever FP8 enter API the installed TE exposes."""
    enter_fn = getattr(FP8GlobalStateManager, "autocast_enter", None)
    if enter_fn is None:
        enter_fn = getattr(FP8GlobalStateManager, "fp8_autocast_enter", None)
    if enter_fn is None:
        raise AttributeError("FP8GlobalStateManager has no autocast enter API")
    enter_fn(
        enabled=enabled,
        calibrating=calibrating,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
        _graph=_graph,
    )


def _call_fp8_autocast_exit(enabled: bool, *, _graph: bool) -> None:
    """Dispatch to whichever FP8 exit API the installed TE exposes."""
    exit_fn = getattr(FP8GlobalStateManager, "autocast_exit", None)
    if exit_fn is None:
        exit_fn = getattr(FP8GlobalStateManager, "fp8_autocast_exit", None)
    if exit_fn is None:
        raise AttributeError("FP8GlobalStateManager has no autocast exit API")
    exit_fn(enabled, _graph=_graph)


def _is_fp4_or_fp8_enabled():
    return (
        PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled()
        or PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled()
    )


def _use_split_wgrad_op():
    args = get_args()

    enable_split_wgrad_op = False
    if args.patch_primus_pipeline and args.pp_algorithm in [
        "zero-bubble",
        "zero-bubble-heuristic",
        "zbv-formatted",
        "v-half",
        "v-min",
    ]:
        enable_split_wgrad_op = True
        return True

    elif args.patch_zero_bubble and args.enable_zero_bubble:
        enable_split_wgrad_op = True

    if enable_split_wgrad_op:
        assert (
            not _is_fp4_or_fp8_enabled()
        ), "split wgrad op is not supported when turbo fp8 or fp4 is enabled."

    return enable_split_wgrad_op


class PrimusTurboQuantConfig:

    def __init__(
        self,
        format: Format = Format.E4M3,
        granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
        strategy: ScalingStrategy = ScalingStrategy.DYNAMIC,
        scale_dtype: ScaleDtype = ScaleDtype.FP32,
        block_size: int = None,
    ):
        self._is_fp4 = False
        self._is_fp8 = False
        if format == Format.E2M1_X2:
            # FP4
            self._quant_config = Float4QuantConfig(
                format=format,
                granularity=granularity,
                strategy=strategy,
                scale_dtype=scale_dtype,
                block_size=block_size,
            )
            self._is_fp4 = True
        else:
            # FP8
            self._quant_config = Float8QuantConfig(
                format=format,
                granularity=granularity,
                strategy=strategy,
                scale_dtype=scale_dtype,
                block_size=block_size,
            )
            self._is_fp8 = True

    def data(self):
        return self._quant_config

    def is_fp4(self):
        return self._is_fp4

    def is_fp8(self):
        return self._is_fp8

    def block_scaling(self):
        return (
            self._quant_config.granularity == ScalingGranularity.BLOCKWISE
            and self._quant_config.strategy == ScalingStrategy.DYNAMIC
        )

    def current_scaling(self):
        return (
            self._quant_config.granularity == ScalingGranularity.TENSORWISE
            and self._quant_config.strategy == ScalingStrategy.DYNAMIC
        )

    def mxfp8_scaling(self):
        # NOTE: The mxfp8 recipe only support e4m3 format in megatron-lm backend.
        return (
            self._quant_config.granularity == ScalingGranularity.MX_BLOCKWISE
            and self._quant_config.strategy == ScalingStrategy.DYNAMIC
            and self._quant_config.format == Format.E4M3
        )

    def mxfp4_scaling(self):
        return (
            self._quant_config.granularity == ScalingGranularity.MX_BLOCKWISE
            and self._quant_config.strategy == ScalingStrategy.DYNAMIC
            and self._quant_config.format == Format.E2M1_X2
            and self._quant_config.scale_dtype == ScaleDtype.E8M0
        )


class PrimusTurboLowPrecisionGlobalStateManager(FP8GlobalStateManager):
    PRIMUS_TURBO_QUANT_CONFIG: PrimusTurboQuantConfig = None
    PRIMUS_TURBO_FP8_ENABLED: bool = False
    PRIMUS_TURBO_FP4_ENABLED: bool = False

    @classmethod
    def is_turbo_fp8_enabled(cls) -> bool:
        """Is FP8 enabled"""
        return cls.PRIMUS_TURBO_FP8_ENABLED

    @classmethod
    def is_turbo_fp4_enabled(cls) -> bool:
        """Is FP4 enabled"""
        return cls.PRIMUS_TURBO_FP4_ENABLED

    @classmethod
    def reset(cls) -> None:
        """Reset the global state"""
        FP8GlobalStateManager.reset()

        cls.PRIMUS_TURBO_FP8_ENABLED = False
        cls.PRIMUS_TURBO_FP4_ENABLED = False
        cls.PRIMUS_TURBO_QUANT_CONFIG = None

    @classmethod
    def autocast_enter(
        cls,
        enabled: bool = False,
        calibrating: bool = False,
        fp8_recipe: Optional[Recipe] = None,
        fp8_group: Optional[dist_group_type] = None,
        _graph: bool = False,
        enabled_turbo: bool = False,
        turbo_quant_config: Optional[PrimusTurboQuantConfig] = None,
    ) -> None:
        _call_fp8_autocast_enter(
            enabled=enabled,
            calibrating=calibrating,
            fp8_recipe=fp8_recipe,
            fp8_group=fp8_group,
            _graph=_graph,
        )

        # Default is fp8 tensorwise
        turbo_quant_config = PrimusTurboQuantConfig() if turbo_quant_config is None else turbo_quant_config

        cls.PRIMUS_TURBO_FP8_ENABLED = enabled_turbo and turbo_quant_config.is_fp8()
        cls.PRIMUS_TURBO_FP4_ENABLED = enabled_turbo and turbo_quant_config.is_fp4()
        cls.PRIMUS_TURBO_QUANT_CONFIG = turbo_quant_config

        if enabled_turbo:
            fp8_available, reason_for_no_fp8 = check_fp8_support()
            assert fp8_available, reason_for_no_fp8
            if turbo_quant_config.mxfp8_scaling():
                mxfp8_available, reason_for_no_mxfp8 = check_mxfp8_support()
                assert mxfp8_available, reason_for_no_mxfp8
            if turbo_quant_config.mxfp4_scaling():
                mxfp4_available, reason_for_no_mxfp4 = check_mxfp4_support()
                assert mxfp4_available, reason_for_no_mxfp4

    @classmethod
    def get_turbo_quant_config(cls) -> PrimusTurboQuantConfig:
        """Return the turbo's quant_config"""
        return cls.PRIMUS_TURBO_QUANT_CONFIG

    @classmethod
    def get_fp8_autocast_state(
        cls,
    ) -> Tuple[bool, bool, Recipe, dist_group_type, bool, bool, PrimusTurboQuantConfig]:
        """FP8 autocast state getter"""
        return (
            FP8GlobalStateManager.FP8_ENABLED,
            FP8GlobalStateManager.FP8_CALIBRATION,
            FP8GlobalStateManager.FP8_RECIPE,
            FP8GlobalStateManager.FP8_DISTRIBUTED_GROUP,
            FP8GlobalStateManager.IS_FIRST_FP8_MODULE,
            FP8GlobalStateManager.FP8_GRAPH_CAPTURING,
            cls.PRIMUS_TURBO_FP8_ENABLED,
            cls.PRIMUS_TURBO_FP4_ENABLED,
            cls.PRIMUS_TURBO_QUANT_CONFIG,
        )

    @classmethod
    def set_fp8_autocast_state(
        cls,
        fp8_state: Tuple[bool, bool, DelayedScaling, dist_group_type, bool, bool, PrimusTurboQuantConfig],
    ) -> None:
        """FP8 autocast state setter"""
        (
            FP8GlobalStateManager.FP8_ENABLED,
            FP8GlobalStateManager.FP8_CALIBRATION,
            FP8GlobalStateManager.FP8_RECIPE,
            FP8GlobalStateManager.FP8_DISTRIBUTED_GROUP,
            FP8GlobalStateManager.IS_FIRST_FP8_MODULE,
            FP8GlobalStateManager.FP8_GRAPH_CAPTURING,
            cls.PRIMUS_TURBO_FP8_ENABLED,
            cls.PRIMUS_TURBO_FP4_ENABLED,
            cls.PRIMUS_TURBO_QUANT_CONFIG,
        ) = fp8_state


@contextmanager
def primus_turbo_fp8_autocast(
    enabled: bool = True,
    calibrating: bool = False,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    _graph: bool = False,
    enabled_turbo: bool = False,
    turbo_quant_config: Optional[PrimusTurboQuantConfig] = None,
) -> None:  # type: ignore
    fp8_state = PrimusTurboLowPrecisionGlobalStateManager.get_fp8_autocast_state()
    PrimusTurboLowPrecisionGlobalStateManager.autocast_enter(
        enabled=enabled,
        calibrating=calibrating,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
        _graph=_graph,
        enabled_turbo=enabled_turbo,
        turbo_quant_config=turbo_quant_config,
    )
    try:
        yield
    finally:
        PrimusTurboLowPrecisionGlobalStateManager.set_fp8_autocast_state(fp8_state)
        # Use the base TE state manager so depth accounting stays in sync
        # across both old and new TE autocast APIs.
        _call_fp8_autocast_exit(enabled, _graph=_graph)


@contextmanager
def primus_turbo_fp4_autocast(
    enabled: bool = True,
    calibrating: bool = False,
    fp4_recipe: Optional[Recipe] = None,
    fp4_group: Optional[dist_group_type] = None,
    _graph: bool = False,
    enabled_turbo: bool = False,
    turbo_quant_config: Optional[PrimusTurboQuantConfig] = None,
) -> None:  # type: ignore
    # TE currently uses fp8_autocast for fp8 and fp4 quantization.
    fp8_state = PrimusTurboLowPrecisionGlobalStateManager.get_fp8_autocast_state()
    PrimusTurboLowPrecisionGlobalStateManager.autocast_enter(
        enabled=enabled,
        calibrating=calibrating,
        fp8_recipe=fp4_recipe,
        fp8_group=fp4_group,
        _graph=_graph,
        enabled_turbo=enabled_turbo,
        turbo_quant_config=turbo_quant_config,
    )
    try:
        yield
    finally:
        PrimusTurboLowPrecisionGlobalStateManager.set_fp8_autocast_state(fp8_state)
        _call_fp8_autocast_exit(enabled, _graph=_graph)


def _get_fp8_autocast_for_quant_recipe(qrecipe: TEQuantizationRecipe):
    if FP8GlobalStateManager.is_fp8_enabled():
        if not qrecipe.override_quantized_autocast:
            return nullcontext()
    else:
        if not qrecipe.override_nonquantized_autocast:
            return nullcontext()

    if qrecipe.fp8_quantization_recipe is None and qrecipe.fp4_quantization_recipe is None:
        # Force BF16 for this layer and override autocast
        return primus_turbo_fp8_autocast(enabled=False, enabled_turbo=False)
    else:
        if (
            qrecipe.fp8_quantization_recipe == Fp8Recipe.custom
            or qrecipe.fp4_quantization_recipe == Fp4Recipe.custom
        ):
            assert qrecipe.custom_recipe_factory is not None
            assert False, "Custom recipe is not supported for Primus-Turbo"

        elif qrecipe.fp8_quantization_recipe is not None:
            from primus.backends.megatron.core.fp8_utils import (
                MXFP8_SCALING_BLOCK_SIZE,
                SCALING_BLOCK_SIZE,
            )

            if qrecipe.fp8_format == "e4m3":
                fp8_format = Format.E4M3
            elif qrecipe.fp8_format == "hybrid":
                fp8_format = Format.HYBRID
            else:
                raise ValueError(f"Unhandled fp8_format {qrecipe.fp8_format}")

            if qrecipe.fp8_quantization_recipe == Fp8Recipe.tensorwise:
                quant_recipe = PrimusTurboQuantConfig(
                    granularity=ScalingGranularity.TENSORWISE, format=fp8_format
                )
            elif qrecipe.fp8_quantization_recipe == Fp8Recipe.blockwise:
                quant_recipe = PrimusTurboQuantConfig(
                    granularity=ScalingGranularity.BLOCKWISE, format=fp8_format, block_size=SCALING_BLOCK_SIZE
                )
            elif qrecipe.fp8_quantization_recipe == Fp8Recipe.mxfp8:
                quant_recipe = PrimusTurboQuantConfig(
                    granularity=ScalingGranularity.MX_BLOCKWISE,
                    format=fp8_format,
                    block_size=MXFP8_SCALING_BLOCK_SIZE,
                    scale_dtype=ScaleDtype.E8M0,
                )
            else:
                raise ValueError(f"Unhandled fp8 recipe: {qrecipe.fp8_quantization_recipe}")

            return primus_turbo_fp8_autocast(
                enabled=False, enabled_turbo=True, turbo_quant_config=quant_recipe
            )
        else:
            # Fp4 configured.
            if qrecipe.fp4_quantization_recipe == Fp4Recipe.nvfp4:
                assert False, "NVFP4 is not supported for Primus-Turbo"
            elif qrecipe.fp4_quantization_recipe == Fp4Recipe.mxfp4:
                from primus.backends.megatron.core.fp4_utils import (
                    MXFP4_SCALING_BLOCK_SIZE,
                )

                quant_recipe = PrimusTurboQuantConfig(
                    granularity=ScalingGranularity.MX_BLOCKWISE,
                    format=Format.E2M1_X2,
                    block_size=MXFP4_SCALING_BLOCK_SIZE,
                    scale_dtype=ScaleDtype.E8M0,
                )
            else:
                raise ValueError(f"Unhandled fp4 recipe: {qrecipe.fp4_quantization_recipe}")

            return primus_turbo_fp4_autocast(
                enabled=False, enabled_turbo=True, turbo_quant_config=quant_recipe
            )


def _get_fp8_autocast_for_quant_params(qparams: TEQuantizationParams | None, training: bool):
    if qparams is None:
        return nullcontext()
    elif not training and qparams.evaluation_recipe is not None:
        return _get_fp8_autocast_for_quant_recipe(qparams.evaluation_recipe)
    else:
        return _get_fp8_autocast_for_quant_recipe(qparams.training_recipe)


class PrimusTurboAttention(te.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().

    Supports sink attention (PR 208) when use_sink_attention is enabled.
    GPT-OSS style sink attention uses learned sink parameters per attention head,
    which act as virtual attention targets that help stabilize attention patterns
    especially with sliding window attention.

    Primus-Turbo API (flash_attn_interface.py):
        flash_attn_func(..., sink: Optional[torch.Tensor] = None)
        - sink: learned sink parameters, shape (num_attention_heads,)
        - When sink is provided, the Triton backend is automatically used
          (C++ backend does not support sink attention)

    Reference: gpt-oss/gpt_oss/triton/attention.py
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        self.config = config
        self.qkv_format: str = "sbhd"
        self.softmax_scale = softmax_scale
        self.layer_number = layer_number

        args = get_args()

        # Sink attention configuration (PR 208) - GPT-OSS style learned sinks
        # Reference: Primus-Turbo/primus_turbo/pytorch/ops/attention/flash_attn_interface.py
        # Note: We store config here but create self.sinks AFTER super().__init__()
        # because PyTorch requires Module.__init__() to be called before assigning parameters
        _use_sink_attention = getattr(args, "use_sink_attention", False)
        # Sliding window size (gpt-oss uses 128, applied to even layers only)
        self.sink_sliding_window = getattr(args, "sink_sliding_window", 0)
        # Whether to apply sliding window only to even layers (gpt-oss pattern)
        self.sink_window_even_layers_only = getattr(args, "sink_window_even_layers_only", True)

        # Note: Sink attention is currently only supported in non-CP mode
        # (flash_attn_usp_func does not support sink parameter yet)
        if _use_sink_attention and self.config.context_parallel_size > 1:
            import warnings

            warnings.warn(
                "Sink attention is not supported with Context Parallel (CP > 1). "
                "Disabling sink attention for this configuration."
            )
            _use_sink_attention = False

        # Store for later use after super().__init__()
        self._init_sink_attention = _use_sink_attention
        self._num_heads_for_sinks = self.config.num_attention_heads

        self.offload = args.offload and "attn" in args.offload_ops
        if args.enable_turbo_attention_float8:
            self.attn = (
                primus_turbo_torch.ops.flash_attn_fp8_usp_func
                if self.config.context_parallel_size > 1
                else primus_turbo_torch.ops.flash_attn_fp8_func
            )
        else:
            self.attn = (
                primus_turbo_torch.ops.flash_attn_usp_func
                if self.config.context_parallel_size > 1
                else primus_turbo_torch.ops.flash_attn_func
            )
        if pg_collection is None:
            # For backward compatibility, remove in v0.14 and raise error
            # raise ValueError("TEDotProductAttention was called without ProcessGroupCollection")
            pg_collection = ProcessGroupCollection(
                tp=get_tensor_model_parallel_group(check_initialized=False),
                cp=get_context_parallel_group(check_initialized=False),
                hcp=get_hierarchical_context_parallel_groups(check_initialized=False),
            )
        else:
            assert hasattr(pg_collection, "tp"), "TEDotProductAttention pg_collection must have tp pg"
            assert hasattr(pg_collection, "cp"), "TEDotProductAttention pg_collection must have cp pg"
            if cp_comm_type == "a2a+p2p":
                assert hasattr(
                    pg_collection, "hcp"
                ), "TEDotProductAttention pg_collection must have hierarchical cp pg"

        self.attn_kwargs = {}
        if self.config.context_parallel_size > 1:
            self.attn_kwargs["ulysses_group"] = pg_collection.cp
            # TODO (limou)
            # enable ring attention
            self.attn_kwargs["ring_group"] = dist.new_group(ranks=[dist.get_rank()])

        assert config.window_size is None, "primus_turbo does not support sliding window attention"
        # Check version

        kv_channels = (
            (k_channels, v_channels)
            if k_channels is not None and v_channels is not None
            else self.config.kv_channels
        )

        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=kv_channels,
            num_gqa_groups=self.config.num_query_groups,
            attention_dropout=(
                self.config.attention_dropout if attention_dropout is None else attention_dropout
            ),
            qkv_format="sbhd",
            attn_mask_type=attn_mask_type.name,
            window_size=None,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=None,
            tp_group=pg_collection.tp,
            layer_number=layer_number,
            attention_type=attention_type,
            # cp is not support
            softmax_scale=softmax_scale,
        )

        # Initialize learned sink parameters AFTER super().__init__()
        # Shape: (num_attention_heads,) - one sink value per head
        # This matches gpt-oss model: self.sinks = torch.nn.Parameter(torch.empty(num_attention_heads))
        self.use_sink_attention = self._init_sink_attention
        if self.use_sink_attention:
            self.sinks = torch.nn.Parameter(torch.zeros(self._num_heads_for_sinks, dtype=torch.bfloat16))
        else:
            self.sinks = None
        # Clean up temporary attributes
        del self._init_sink_attention
        del self._num_heads_for_sinks

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward."""
        SUPPORTED_QKV_FORMATS = "sbhd"

        packed_seq_kwargs = (
            {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )

        qkv_format = packed_seq_kwargs.get("qkv_format", self.qkv_format)
        assert (
            qkv_format in SUPPORTED_QKV_FORMATS
        ), f"qkv_format only support {SUPPORTED_QKV_FORMATS}, but got {qkv_format}"
        # NOTE(ruibin): The layout of q, k and v is (S, B, H, D). But attn accept the shape of qkv is (B, S, H, D).
        query, key, value = [x.permute(1, 0, 2, 3) for x in (query, key, value)]
        mask_type = attn_mask_type.name
        if mask_type == AttnMaskType.causal.name:
            causal = True
        elif mask_type == AttnMaskType.no_mask.name:
            causal = False
        else:
            raise ValueError(f"Unsupported mask type: {mask_type}")

        # Sink attention support (PR 208) - GPT-OSS style
        # Learned sinks act as virtual attention targets that help stabilize
        # attention patterns, especially with sliding window attention.
        #
        # Primus-Turbo API (flash_attn_interface.py line 316-348):
        #   flash_attn_func(..., sink: Optional[torch.Tensor] = None)
        #   - sink: learned sink parameters, shape (num_attention_heads,)
        #   - When sink is provided, Triton backend is automatically used
        #
        # Reference: gpt-oss/gpt_oss/triton/attention.py
        sink_tensor = None
        window_size = (-1, -1)

        if self.use_sink_attention and self.sinks is not None:
            sink_tensor = self.sinks

            # Apply sliding window based on layer pattern (gpt-oss: even layers only)
            # gpt-oss pattern: self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
            if self.sink_sliding_window > 0:
                if self.sink_window_even_layers_only:
                    # Only apply sliding window to even layers (layer_number is 1-indexed in Megatron)
                    if (self.layer_number - 1) % 2 == 0:
                        window_size = (self.sink_sliding_window, 0)
                else:
                    window_size = (self.sink_sliding_window, 0)
        if self.offload:
            OFFLOAD_BUFFER.add_offload_tensor(f"attn_q", query)
            OFFLOAD_BUFFER.add_offload_tensor(f"attn_k", key)
            OFFLOAD_BUFFER.add_offload_tensor(f"attn_v", value)

        o = self.attn(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            window_size=window_size,
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            sink=sink_tensor,  # PR 208: pass sink tensor to Primus-Turbo
            **self.attn_kwargs,
        )

        # NOTE(ruibin): The output of attn is BSHD. Use permute to convert the layout to SBHD.
        o = o.permute(1, 0, 2, 3).contiguous()
        o = o.view(o.shape[0], o.shape[1], -1)

        return o


class PrimusTurboLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: Optional[str] = None,
        is_expert: bool = False,
        symmetric_ar_type: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=is_expert,
            symmetric_ar_type=symmetric_ar_type,
            tp_group=tp_group,
        )

        tp_size = get_pg_size(self._tp_group)
        assert tp_size == 1, "PrimusTurboLinear only supports tensor parallel size = 1"

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, {"weight": 1}, sharded_offsets)

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def forward(self, x: torch.Tensor):
        _is_first_microbatch = self.is_first_microbatch

        # Rewrite quant context
        quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

        with quant_context:
            out = self.forward_internal(x, _is_first_microbatch)

        self.is_first_microbatch = False

        return out

    def forward_internal(
        self,
        x: torch.Tensor,
        is_first_microbatch: bool = False,
    ):
        weight = self._parameters["weight"]
        if self.use_bias:
            bias_tensor = torch.cat([getattr(self, name) for name in self.bias_names])
        original_shape = x.size()
        if not x.is_contiguous():
            x = x.contiguous()
        x = x.view(-1, original_shape[-1])

        if self.offload:
            OFFLOAD_BUFFER.add_offload_tensor(f"linear_input", x)

        if _use_split_wgrad_op():
            from .zbpp_gemm import gemm_with_weight_gradient_store

            out = gemm_with_weight_gradient_store(x, weight, bias=None)
        else:
            if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
                quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
                assert (
                    quant_config.mxfp8_scaling()
                    or quant_config.current_scaling()
                    or quant_config.block_scaling()
                ), "Turbo FP8 is enabled but quant config is not mxfp8, current scaling, or block scaling."

                out = primus_turbo_torch.ops.gemm_fp8(
                    x,
                    weight,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=None,
                    config=quant_config.data(),
                )
            elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
                quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
                assert quant_config.mxfp4_scaling(), "Turbo FP4 is enabled but quant config is not mxfp4."

                out = primus_turbo_torch.ops.gemm_fp4(
                    x,
                    weight,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=None,
                    config=quant_config.data(),
                )
            else:
                out = primus_turbo_torch.ops.gemm(x, weight, trans_a=False, trans_b=True, out_dtype=None)

        out = out.view(original_shape[0], original_shape[1], -1)

        if self.use_bias:
            out = out + bias_tensor

        return out, None


class PrimusTurboRowParallelLinear(TERowParallelLinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not input_is_parallel:
            raise ValueError(f"{__class__.__name__} layers do not support input_is_parallel = False")

        args = get_args()
        self.offload = args.offload and "row_parallel_gemm" in args.offload_ops
        assert not self.offload, "gemm offload still have some problems"

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )

        tp_size = get_pg_size(self._tp_group)
        assert tp_size == 1, "PrimusTurboRowParallelLinear only supports tensor parallel size = 1"

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, {"weight": 1}, sharded_offsets)

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def forward(self, x: torch.Tensor):
        _is_first_microbatch = self.is_first_microbatch

        # Rewrite quant context
        quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

        with quant_context:
            out = self.forward_internal(x, _is_first_microbatch)

        self.is_first_microbatch = False

        return out

    def forward_internal(
        self,
        x: torch.Tensor,
        is_first_microbatch: bool = False,
    ):
        weight = self._parameters["weight"]

        if self.use_bias:
            bias_tensor = torch.cat([getattr(self, name) for name in self.bias_names])
        original_shape = x.size()
        if not x.is_contiguous():
            x = x.contiguous()

        if self.offload:
            OFFLOAD_BUFFER.add_offload_tensor(f"row_parallel_linear_input", x)

        x = x.view(-1, original_shape[-1])

        if self.offload:
            OFFLOAD_BUFFER.add_offload_tensor(f"row_parallel_linear_input", x)

        if _use_split_wgrad_op():
            from .zbpp_gemm import gemm_with_weight_gradient_store

            out = gemm_with_weight_gradient_store(x, weight, bias=None)
        else:
            if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
                quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
                assert (
                    quant_config.mxfp8_scaling()
                    or quant_config.current_scaling()
                    or quant_config.block_scaling()
                ), "Turbo FP8 is enabled but quant config is not mxfp8, current scaling, or block scaling."

                out = primus_turbo_torch.ops.gemm_fp8(
                    x,
                    weight,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=None,
                    config=quant_config.data(),
                )
            elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
                quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
                assert quant_config.mxfp4_scaling(), "Turbo FP4 is enabled but quant config is not mxfp4."

                out = primus_turbo_torch.ops.gemm_fp4(
                    x,
                    weight,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=None,
                    config=quant_config.data(),
                )
            else:
                out = primus_turbo_torch.ops.gemm(x, weight, trans_a=False, trans_b=True, out_dtype=None)

        out = out.view(original_shape[0], original_shape[1], -1)

        if self.use_bias:
            out = out + bias_tensor

        return out, None


class PrimusTurboColumnParallelLinear(TEColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        stride: int = 1,  # TODO(ruibin): compatible with Megatron-LM. Not used.
    ):
        args = get_args()
        self.offload = args.offload and "column_parallel_gemm" in args.offload_ops
        assert not self.offload, "gemm offload still have some problems"

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
            stride=stride,
        )

        tp_size = get_pg_size(self._tp_group)
        assert tp_size == 1, "PrimusTurboColumnParallelLinear only supports tensor parallel size = 1"

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def forward(self, x: torch.Tensor):
        _is_first_microbatch = self.is_first_microbatch

        # Rewrite quant context
        quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

        with quant_context:
            out = self.forward_internal(x, _is_first_microbatch)

        self.is_first_microbatch = False

        return out

    def forward_internal(
        self,
        x: torch.Tensor,
        is_first_microbatch: bool = False,
    ):
        weight = self._parameters["weight"]
        if self.use_bias:
            bias_tensor = torch.cat([getattr(self, name) for name in self.bias_names])
        original_shape = x.size()
        if not x.is_contiguous():
            x = x.contiguous()
        x = x.view(-1, original_shape[-1])

        if self.offload:
            OFFLOAD_BUFFER.add_offload_tensor(f"column_parallel_linear_input", x)

        if _use_split_wgrad_op():
            from .zbpp_gemm import gemm_with_weight_gradient_store

            out = gemm_with_weight_gradient_store(x, weight, bias=None)
        else:
            if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
                quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
                assert (
                    quant_config.mxfp8_scaling()
                    or quant_config.current_scaling()
                    or quant_config.block_scaling()
                ), "Turbo FP8 is enabled but quant config is not mxfp8, current scaling, or block scaling."

                out = primus_turbo_torch.ops.gemm_fp8(
                    x,
                    weight,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=None,
                    config=quant_config.data(),
                )
            elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
                quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
                assert quant_config.mxfp4_scaling(), "Turbo FP4 is enabled but quant config is not mxfp4."

                out = primus_turbo_torch.ops.gemm_fp4(
                    x,
                    weight,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=None,
                    config=quant_config.data(),
                )
            else:
                out = primus_turbo_torch.ops.gemm(x, weight, trans_a=False, trans_b=True, out_dtype=None)

        out = out.view(original_shape[0], original_shape[1], -1)

        if self.use_bias:
            out = out + bias_tensor

        return out, None


class PrimusTurboColumnParallelLinearTorch(ColumnParallelLinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        disable_grad_reduce: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        args = get_args()
        self.offload = args.offload and "column_parallel_gemm" in args.offload_ops
        assert not self.offload, "gemm offload still have some problems"

        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=gather_output,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            disable_grad_reduce=disable_grad_reduce,
            tp_group=tp_group,
        )
        tp_size = get_pg_size(tp_group)
        assert tp_size == 1, "PrimusTurboColumnParallelLinearTorch only supports tensor parallel size = 1"

        self.is_first_microbatch = True

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        _is_first_microbatch = self.is_first_microbatch

        # Rewrite quant context
        quant_context = _get_fp8_autocast_for_quant_params(
            getattr(self, "te_quant_params", None), self.training
        )

        with quant_context:
            out = self.forward_internal(x, weight, runtime_gather_output, _is_first_microbatch)

        self.is_first_microbatch = False

        return out

    def forward_internal(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        is_first_microbatch: bool = False,
    ):
        if weight is None:
            weight = self.weight

        bias_tensor = self.bias if not self.skip_bias_add else None

        original_shape = x.size()
        if not x.is_contiguous():
            x = x.contiguous()
        x = x.view(-1, original_shape[-1])

        if self.offload:
            OFFLOAD_BUFFER.add_offload_tensor(f"column_parallel_linear_torch_input", x)

        if _use_split_wgrad_op():
            from .zbpp_gemm import gemm_with_weight_gradient_store

            out = gemm_with_weight_gradient_store(x, weight, bias=None)
        else:
            if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
                quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
                assert (
                    quant_config.mxfp8_scaling()
                    or quant_config.current_scaling()
                    or quant_config.block_scaling()
                ), "Turbo FP8 is enabled but quant config is not mxfp8, current scaling, or block scaling."

                out = primus_turbo_torch.ops.gemm_fp8(
                    x,
                    weight,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=None,
                    config=quant_config.data(),
                )
            elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
                quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
                assert quant_config.mxfp4_scaling(), "Turbo FP4 is enabled but quant config is not mxfp4."

                out = primus_turbo_torch.ops.gemm_fp4(
                    x,
                    weight,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=None,
                    config=quant_config.data(),
                )
            else:
                out = primus_turbo_torch.ops.gemm(x, weight, trans_a=False, trans_b=True, out_dtype=None)

        out = out.view(original_shape[0], original_shape[1], -1)

        return out, bias_tensor


class PrimusTurboLayerNormColumnParallelLinear(TELayerNormColumnParallelLinear):
    """
    Wrapper for the Transformer-Engine's `LayerNormLinear` layer that combines
    layernorm and linear layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        stride: int = 1,
    ):
        args = get_args()
        self.offload = args.offload and "column_parallel_gemm" in args.offload_ops
        assert not self.offload, "gemm offload still have some problems"

        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
            stride=stride,
        )

        tp_size = get_pg_size(self._tp_group)
        assert tp_size == 1, "PrimusTurboLayerNormColumnParallelLinear only supports tensor parallel size = 1"

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def forward(self, x):
        _is_first_microbatch = self.is_first_microbatch

        # Rewrite quant context
        quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

        with quant_context:
            out = self.forward_internal(x, _is_first_microbatch)

        self.is_first_microbatch = False

        return out

    def forward_internal(self, x, is_first_microbatch: bool = False):
        """Forward."""
        if self.config.normalization == "LayerNorm":
            norm_out = torch.nn.functional.layer_norm(
                x, [x.size(-1)], self.layer_norm_weight, self.layer_norm_bias, self.eps
            )
        elif self.config.normalization == "RMSNorm":
            norm_out = torch.nn.functional.rms_norm(x, [x.size(-1)], self.layer_norm_weight, self.eps)
        else:
            assert False, "Not support normalization type."

        weight = self._parameters["weight"]
        if self.use_bias:
            bias_tensor = torch.cat([getattr(self, name) for name in self.bias_names])
        else:
            bias_tensor = None

        original_shape = x.size()
        if not norm_out.is_contiguous():
            norm_out = norm_out.contiguous()
        inp = norm_out.view(-1, original_shape[-1])

        if _use_split_wgrad_op():
            from .zbpp_gemm import gemm_with_weight_gradient_store

            out = gemm_with_weight_gradient_store(inp, weight, bias=None)
        else:
            if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
                quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
                assert (
                    quant_config.mxfp8_scaling()
                    or quant_config.current_scaling()
                    or quant_config.block_scaling()
                ), "Turbo FP8 is enabled but quant config is not mxfp8, current scaling, or block scaling."

                out = primus_turbo_torch.ops.gemm_fp8(
                    inp,
                    weight,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=None,
                    config=quant_config.data(),
                )
            elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
                quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
                assert quant_config.mxfp4_scaling(), "Turbo FP4 is enabled but quant config is not mxfp4."

                out = primus_turbo_torch.ops.gemm_fp4(
                    inp,
                    weight,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=None,
                    config=quant_config.data(),
                )
            else:
                out = primus_turbo_torch.ops.gemm(inp, weight, trans_a=False, trans_b=True, out_dtype=None)

        out = out.view(original_shape[0], original_shape[1], -1)

        if self.use_bias:
            out = out + bias_tensor

        return out, None


def fused_bias_act_with_probs(
    intermediate_parallel: torch.Tensor,
    bias_parallel: torch.Tensor,
    permuted_probs: torch.Tensor,
    tokens_per_experts: torch.Tensor,
    activation_func: str,
):
    assert intermediate_parallel.ndim == 2
    assert permuted_probs.ndim == 1
    assert tokens_per_experts.device == intermediate_parallel.device

    # TODO(ruibin): fuse bias addition with activation function
    if bias_parallel is not None:
        intermediate_parallel = intermediate_parallel + bias_parallel

    num_tokens = intermediate_parallel.shape[0]
    row_mask = primus_turbo_torch.ops.tokens_per_expert_to_mask(tokens_per_experts, num_tokens)

    # TODO(ruibin): support more activation functions
    if activation_func == "silu":
        fused_act_with_probs = primus_turbo_torch.ops.swiglu_with_probs
    elif activation_func == "gelu":
        fused_act_with_probs = primus_turbo_torch.ops.geglu_with_probs
    else:
        raise ValueError(f"Activation function {activation_func} is not supported.")

    return fused_act_with_probs(intermediate_parallel, permuted_probs, row_mask)


class PrimusTurboGroupedLinear(TEGroupedLinear):
    """
    Wrapper for the PrimusTurbo `grouped_gemm` ops.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        args = get_args()
        self.offload = args.offload and "column_parallel_gemm" in args.offload_ops
        assert not self.offload, "gemm offload still have some problems"

        super().__init__(
            num_gemms,
            input_size,
            output_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            pg_collection=pg_collection,
        )

        tp_size = get_pg_size(self._tp_group)
        assert tp_size == 1, "PrimusTurboGroupedLinear only supports tensor parallel size = 1"

    def _stack_grouped_linear_weight(self) -> torch.Tensor:
        weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        return torch.stack(weights, dim=0).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, m_splits: torch.Tensor):
        _is_first_microbatch = self.is_first_microbatch
        quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

        with quant_context:
            out = self.forward_internal(x, m_splits, _is_first_microbatch)

        self.is_first_microbatch = False

        return out

    def forward_internal(
        self,
        x: torch.Tensor,
        m_splits: torch.Tensor,
        is_first_microbatch: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward step of the legacy PrimusTurbo grouped-gemm MLP."""
        weights = self._stack_grouped_linear_weight()
        # NOTE: keep x and m_splits on the same device
        m_splits = m_splits.to(x.device)

        if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
            quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
            out = primus_turbo_torch.ops.grouped_gemm_fp8(
                x,
                weights,
                m_splits,
                trans_b=False,
                config=quant_config.data(),
            )
        elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
            assert False, "FP4 is not supported in PrimusTurboGroupedLinear"
        else:
            out = primus_turbo_torch.ops.grouped_gemm(x, weights, m_splits, trans_b=False)

        return out, None


class PrimusTurboColumnParallelGroupedLinear(PrimusTurboGroupedLinear):
    """
    Wrapper for the PrimusTurboGroupedLinear layer but specialized
    to column-parallel style.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            pg_collection=pg_collection,
        )

        tp_size = get_pg_size(self._tp_group)
        assert tp_size == 1, "PrimusTurboColumnParallelGroupedLinear only supports tensor parallel size = 1"


class PrimusTurboRowParallelGroupedLinear(PrimusTurboGroupedLinear):
    """
    Wrapper for the PrimusTurboGroupedLinear layer but specialized
    to row-parallel style.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            pg_collection=pg_collection,
        )

        tp_size = get_pg_size(self._tp_group)
        assert tp_size == 1, "PrimusTurboColumnParallelGroupedLinear only supports tensor parallel size = 1"


class PrimusTurboDeepEPTokenDispatcher(MoETokenDispatcher):
    """
    PrimusTurbo token dispatcher using DeepEP.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """
        Initialize the Flex token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)

        assert self.tp_size * self.ep_size > 1, "Flex token dispatcher requires TPxEP > 1"
        assert (
            self.config.moe_enable_deepep
        ), "DeepEP is not enabled. Please set --moe-enable-deepep to use DeepEP backend."
        assert (
            self.config.moe_pad_expert_input_to_capacity is False
        ), "Flex token dispatcher does not support --moe-pad-expert-input-to-capacity"

        args = get_args()

        # enable sync-free moe to elimiate deepep cpu busy-wait
        num_worst_tokens, permute_max_token_num = 0, 0
        if args.turbo_sync_free_moe_stage > 1:
            if args.sequence_parallel:
                seq_length = args.seq_length // self.tp_size
            else:
                seq_length = args.seq_length
            num_tokens = seq_length // args.context_parallel_size * args.micro_batch_size
            num_worst_tokens = num_tokens * self.tp_ep_group.size()
            if args.turbo_sync_free_moe_stage > 2:
                # fully sync-free moe
                permute_max_token_num = num_worst_tokens * config.moe_router_topk

        use_turbo_grouped_gemm = args.use_turbo_grouped_gemm or args.use_turbo_grouped_mlp
        self.deepep_dispatcher = primus_turbo_torch.modules.DeepEPTokenDispatcher(
            num_experts=config.num_moe_experts,
            router_topk=config.moe_router_topk,
            ep_group=self.ep_group,
            tp_group=self.tp_group,
            tp_ep_group=self.tp_ep_group,
            expert_capacity_factor=config.moe_expert_capacity_factor,
            permute_fusion=config.moe_permute_fusion,
            permute_max_token_num=permute_max_token_num,
            deepep_use_comm_stream=args.turbo_deepep_use_comm_stream,
            deepep_num_use_cu=args.turbo_deepep_num_cu,
            deepep_num_worst_tokens=num_worst_tokens,
            deepep_use_cuda_num_tokens_per_expert=use_turbo_grouped_gemm,
            deepep_async_finish=True,
            deepep_allocate_on_comm_stream=True,
        )
        # This is just a place holder.
        # The communication manager class is not used in Primus Turbo's DeepEP dispatcher.
        # But it may get referenced in some Megatron code paths.
        self._comm_manager = self.deepep_dispatcher

        self.moe_router_force_load_balancing = args.moe_router_force_load_balancing

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Initializes routing metadata and prepares tensors for fused dispatch.

        This method reshapes input tensors and processes routing information into a
        unified format, where the routing map is expanded to cover the TPxEP communication domain,
        enabling the token dispatch logic to be agnostic to parallelism strategies.

        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed
            routing_map (torch.Tensor): Map indicating which expert each token should be routed to
            probs (torch.Tensor): Routing probabilities for each token-expert pair

        Returns:
            A tuple of reshaped hidden states and token probabilities.
        """
        self.hidden_shape = hidden_states.shape
        # view as [num_tokens, hidden_size]
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        num_tokens = hidden_states.shape[0]

        # when force_load_balancing, we use even token_indices to make sure each expert get same number of tokens
        token_indices = None
        if self.moe_router_force_load_balancing:
            token_indices = (
                torch.arange(num_tokens * self.config.moe_router_topk, device=hidden_states.device).view(
                    num_tokens, self.config.moe_router_topk
                )
                % self.config.num_moe_experts
            )

        hidden_states, probs = self.deepep_dispatcher._pre_dispatch(
            hidden_states, probs, routing_map, token_indices
        )
        return hidden_states, probs

    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Execute fused permutation and AlltoAll communication.

        This method currently leverages DeepEP's fused dispatch kernel, which combines token
        permutation and AlltoAll communication into a single optimized operation.
        The fused approach reduces memory bandwidth requirements and enables better
        overlap between computation and communication operations.

        Args:
            hidden_states (torch.Tensor): Preprocessed hidden states to be dispatched
            probs (torch.Tensor): Routing probabilities (unused in current implementation)
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            A tuple of dispatched tokens and probabilities.
        """
        dispatched_tokens, dispatched_probs = self.deepep_dispatcher._exec_dispatch(hidden_states, probs)
        return dispatched_tokens, dispatched_probs

    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Converts dispatched tokens to a per-expert format for expert processing.

        This method transforms the output of the fused dispatch into the tensor
        organization required for the expert computation.

        Args:
            hidden_states (torch.Tensor): Hidden states after fused dispatch
            probs (torch.Tensor): Routing probabilities after fused dispatch

        Returns:
            A tuple of permuted tokens, token counts per expert, and permuted probabilities.
        """
        permuted_input, tokens_per_expert, permuted_probs = self.deepep_dispatcher._post_dispatch(
            hidden_states, probs
        )
        if self.config.moe_router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return permuted_input, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states: torch.Tensor):
        """Pre-processes hidden states before combining them after expert processing.

        This method restores the hidden states to their original ordering before expert processing
        by using the communication manager's restoration function.
        """
        hidden_states = self.deepep_dispatcher._pre_combine(hidden_states)
        return hidden_states

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """Executes fused un-permutation and communication using DeepEP kernels.

        This is the inverse of the `token_dispatch` operation.

        Args:
            hidden_states (torch.Tensor): Expert outputs ready for combination
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            Combined tokens after fused un-permutation and communication.
        """
        combined_tokens = self.deepep_dispatcher._exec_combine(hidden_states)
        return combined_tokens

    def combine_postprocess(self, hidden_states: torch.Tensor):
        """
        Restores the original tensor shape and finalizes the MoE layer output.

        This method performs the final step of the MoE token processing pipeline
        by reshaping the combined tokens back to their original input dimensions.

        Args:
            hidden_states (torch.Tensor): Combined tokens.

        Returns:
            The final MoE layer output reshaped to its original dimensions.
        """
        hidden_states = self.deepep_dispatcher._post_combine(hidden_states)
        return hidden_states.view(self.hidden_shape)


class PrimusTurboRMSNorm(te.pytorch.RMSNorm):
    def __init__(self, *args, **kwargs):
        assert "device" in kwargs
        assert "dtype" in kwargs or "params_dtype" in kwargs, "device and dtype must be provided"
        super().__init__(*args, **kwargs)
        self.rms_norm_func = primus_turbo_torch.modules.RMSNorm(
            normalized_shape=kwargs["hidden_size"],
            eps=self.eps,
            device=kwargs["device"],
            dtype=kwargs["dtype"] if "dtype" in kwargs else kwargs["params_dtype"],
        )

    def forward(self, x):
        return self.rms_norm_func(x)
