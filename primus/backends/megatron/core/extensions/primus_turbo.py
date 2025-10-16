###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import functools
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

import grouped_gemm
import primus_turbo.pytorch as pt
import primus_turbo.pytorch.ops.activation as turbo_moe_activation
import torch
import torch.nn.functional as F
import transformer_engine as te
from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import TELinear, condition_init_method
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
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from megatron.training.global_vars import get_args
from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    ScalingGranularity,
    ScalingStrategy,
    check_fp8_support,
)
from primus_turbo.pytorch.ops.moe.tokens_per_expert_to_mask import (
    tokens_per_expert_to_mask as turbo_tokens_per_expert_to_mask,
)
from torch import Tensor
from transformer_engine.pytorch.fp8 import (
    DelayedScaling,
    FP8GlobalStateManager,
    Recipe,
    dist_group_type,
)


class PrimusTurboFloat8QuantConfig(Float8QuantConfig):

    def block_scaling(self):
        return self.granularity == ScalingGranularity.BLOCKWISE

    def current_scaling(self):
        return self.granularity == ScalingGranularity.TENSORWISE and self.strategy == ScalingStrategy.DYNAMIC


class PrimusTurboFP8GlobalStateManager(FP8GlobalStateManager):
    PRIMUS_TURBO_FP8_QUANT_CONFIG: PrimusTurboFloat8QuantConfig = None
    PRIMUS_TURBO_FP8_ENABLED: bool = False

    @classmethod
    def is_turbo_fp8_enabled(cls) -> bool:
        """Is FP8 enabled"""
        return cls.PRIMUS_TURBO_FP8_ENABLED

    @classmethod
    def reset(cls) -> None:
        """Reset the global state"""
        FP8GlobalStateManager.reset()

        cls.PRIMUS_TURBO_FP8_ENABLED = False
        cls.PRIMUS_TURBO_FP8_QUANT_CONFIG = None

    @classmethod
    def fp8_autocast_enter(
        cls,
        enabled: bool = False,
        calibrating: bool = False,
        fp8_recipe: Optional[Recipe] = None,
        fp8_group: Optional[dist_group_type] = None,
        _graph: bool = False,
        enabled_turbo: bool = False,
        turbo_fp8_quant_config: Optional[PrimusTurboFloat8QuantConfig] = None,
    ) -> None:
        FP8GlobalStateManager.fp8_autocast_enter(
            enabled=enabled,
            calibrating=calibrating,
            fp8_recipe=fp8_recipe,
            fp8_group=fp8_group,
            _graph=_graph,
        )

        turbo_fp8_quant_config = (
            PrimusTurboFloat8QuantConfig() if turbo_fp8_quant_config is None else turbo_fp8_quant_config
        )

        cls.PRIMUS_TURBO_FP8_ENABLED = enabled_turbo
        cls.PRIMUS_TURBO_FP8_QUANT_CONFIG = turbo_fp8_quant_config

        if enabled_turbo:
            fp8_available, reason_for_no_fp8 = check_fp8_support()
            assert fp8_available, reason_for_no_fp8

    @classmethod
    def get_turbo_fp8_quant_config(cls) -> PrimusTurboFloat8QuantConfig:
        """Return the turbo's fp8 quant_config"""
        return cls.PRIMUS_TURBO_FP8_QUANT_CONFIG

    @classmethod
    def get_fp8_autocast_state(
        cls,
    ) -> Tuple[bool, bool, Recipe, dist_group_type, bool, bool, PrimusTurboFloat8QuantConfig]:
        """FP8 autocast state getter"""
        return (
            cls.FP8_ENABLED,
            cls.FP8_CALIBRATION,
            cls.FP8_RECIPE,
            cls.FP8_DISTRIBUTED_GROUP,
            cls.IS_FIRST_FP8_MODULE,
            cls.FP8_GRAPH_CAPTURING,
            cls.PRIMUS_TURBO_FP8_ENABLED,
            cls.PRIMUS_TURBO_FP8_QUANT_CONFIG,
        )

    @classmethod
    def set_fp8_autocast_state(
        cls,
        fp8_state: Tuple[
            bool, bool, DelayedScaling, dist_group_type, bool, bool, PrimusTurboFloat8QuantConfig
        ],
    ) -> None:
        """FP8 autocast state setter"""
        (
            cls.FP8_ENABLED,
            cls.FP8_CALIBRATION,
            cls.FP8_RECIPE,
            cls.FP8_DISTRIBUTED_GROUP,
            cls.IS_FIRST_FP8_MODULE,
            cls.FP8_GRAPH_CAPTURING,
            cls.PRIMUS_TURBO_FP8_ENABLED,
            cls.PRIMUS_TURBO_FP8_QUANT_CONFIG,
        ) = fp8_state


@contextmanager
def primus_turbo_fp8_autocast(
    enabled: bool = True,
    calibrating: bool = False,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    _graph: bool = False,
    enabled_turbo: bool = False,
    turbo_fp8_quant_config: Optional[PrimusTurboFloat8QuantConfig] = None,
) -> None:  # type: ignore
    fp8_state = PrimusTurboFP8GlobalStateManager.get_fp8_autocast_state()
    PrimusTurboFP8GlobalStateManager.fp8_autocast_enter(
        enabled=enabled,
        calibrating=calibrating,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
        _graph=_graph,
        enabled_turbo=enabled_turbo,
        turbo_fp8_quant_config=turbo_fp8_quant_config,
    )
    try:
        yield
    finally:
        PrimusTurboFP8GlobalStateManager.set_fp8_autocast_state(fp8_state)
        PrimusTurboFP8GlobalStateManager.fp8_autocast_exit(enabled, _graph=_graph)


class PrimusTurboAttention(te.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
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

        args = get_args()
        if args.enable_turbo_attention_float8:
            self.attn = pt.ops.attention_fp8_blockwise
            self.attention_backend = "triton"
        else:
            self.attn = pt.ops.flash_attn_func
            self.attention_backend = "ck"
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
        self.cp_param_bundle = None
        if self.config.context_parallel_size > 1:
            self.cp_param_bundle = {"cp_group": pg_collection.cp, "cp_comm_type": cp_comm_type}

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
        packed_seq_kwargs = (
            {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )

        qkv_format = packed_seq_kwargs.get("qkv_format", self.qkv_format)
        assert qkv_format in ("sbhd", "bhsd"), "qkv_format only support bshd, but got {qkv_format}"
        if qkv_format == "sbhd":
            query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
        mask_type = attn_mask_type.name
        if mask_type == AttnMaskType.causal.name:
            causal = True
        elif mask_type == AttnMaskType.no_mask.name:
            causal = False
        else:
            raise ValueError(f"Unsupported mask type: {mask_type}")

        o = self.attn(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            backend_type=self.attention_backend,
            cp_param_bundle=self.cp_param_bundle,
        )

        o = o.reshape(o.shape[0], o.shape[1], -1).transpose(0, 1)
        if not o.is_contiguous():
            o = o.contiguous()
        return o


class PrimusTurboRowParallelLinear(TELinear):
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
            raise ValueError("Transformer Engine linear layers do not support input_is_parallel = False")

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        args = get_args()
        if args.patch_zero_bubble and args.enable_zero_bubble:
            from .zbpp_gemm import gemm_with_weight_gradient_store

            self.gemm = lambda a, b, bias=None: gemm_with_weight_gradient_store(a, b, bias=bias)
        else:
            self.gemm = lambda a, b, trans_a=False, trans_b=True, out_dtype=None: pt.ops.gemm(
                a, b, trans_a=trans_a, trans_b=trans_b, out_dtype=out_dtype
            )

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
            # We don't currently use this for row parallel layers # pylint: disable=line-too-long
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, {"weight": 1}, sharded_offsets)

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def forward(
        self,
        input_: torch.Tensor,
    ):
        # weights = [getattr(self, name) for name in self.weight_names]
        # weights = torch.cat(weights, dim=0)  # or set weights = self._parameters['weight']
        weights = self._parameters["weight"]
        if self.use_bias:
            bias_tensor = torch.cat([getattr(self, name) for name in self.bias_names])
        original_shape = input_.size()
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        input_ = input_.view(-1, original_shape[-1])

        if PrimusTurboFP8GlobalStateManager.is_turbo_fp8_enabled():
            quant_config = PrimusTurboFP8GlobalStateManager.get_turbo_fp8_quant_config()
            if quant_config.block_scaling():
                fp8_gemm = pt.ops.gemm_fp8_blockwise
            elif quant_config.current_scaling():
                fp8_gemm = pt.ops.gemm_fp8
            else:
                raise ValueError("Not support quant config.")

            out = fp8_gemm(input_, weights, trans_a=False, trans_b=True, out_dtype=None, config=quant_config)
        else:
            out = self.gemm(input_, weights)

        out = out.view(original_shape[0], original_shape[1], -1)
        if self.te_return_bias:
            return out, bias_tensor
        if self.use_bias:
            return out + bias_tensor, None
        return out, None


class PrimusTurboColumnParallelLinear(TELinear):
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
    ):
        if gather_output:
            raise ValueError("Transformer Engine linear layers do not support gather_output = True")
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        args = get_args()

        if args.patch_zero_bubble and args.enable_zero_bubble:
            from .zbpp_gemm import gemm_with_weight_gradient_store

            self.gemm = lambda a, b, bias=None: gemm_with_weight_gradient_store(a, b, bias=bias)
        else:
            self.gemm = lambda a, b, trans_a=False, trans_b=True, out_dtype=None: pt.ops.gemm(
                a, b, trans_a=trans_a, trans_b=trans_b, out_dtype=out_dtype
            )

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )

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

    def forward(
        self,
        input_: torch.Tensor,
    ):
        # weights = [getattr(self, name) for name in self.weight_names]
        # weights = torch.cat(weights, dim=0)  # or set weights = self._parameters['weight']
        weights = self._parameters["weight"]
        if self.use_bias:
            bias_tensor = torch.cat([getattr(self, name) for name in self.bias_names])
        original_shape = input_.size()
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        input_ = input_.view(-1, original_shape[-1])

        if PrimusTurboFP8GlobalStateManager.is_turbo_fp8_enabled():
            quant_config = PrimusTurboFP8GlobalStateManager.get_turbo_fp8_quant_config()
            if quant_config.block_scaling():
                fp8_gemm = pt.ops.gemm_fp8_blockwise
            elif quant_config.current_scaling():
                fp8_gemm = pt.ops.gemm_fp8
            else:
                raise ValueError("Not support quant config.")

            out = fp8_gemm(input_, weights, trans_a=False, trans_b=True, out_dtype=None, config=quant_config)
        else:
            out = self.gemm(input_, weights)

        out = out.view(original_shape[0], original_shape[1], -1)
        if self.te_return_bias:
            return out, bias_tensor
        if self.use_bias:
            return out + bias_tensor, None
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
        if args.patch_zero_bubble and args.enable_zero_bubble:
            from .zbpp_gemm import gemm_with_weight_gradient_store

            self.gemm = lambda a, b, bias=None: gemm_with_weight_gradient_store(a, b, bias=bias)
        else:
            self.gemm = lambda a, b, trans_a=False, trans_b=True, out_dtype=None: pt.ops.gemm(
                a, b, trans_a=trans_a, trans_b=trans_b, out_dtype=out_dtype
            )

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

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        if weight is None:
            weight = self.weight
        bias_tensor = self.bias if not self.skip_bias_add else None

        original_shape = input_.size()
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        input_ = input_.view(-1, original_shape[-1])

        if PrimusTurboFP8GlobalStateManager.is_turbo_fp8_enabled():
            quant_config = PrimusTurboFP8GlobalStateManager.get_turbo_fp8_quant_config()
            if quant_config.block_scaling():
                fp8_gemm = pt.ops.gemm_fp8_blockwise
            elif quant_config.current_scaling():
                fp8_gemm = pt.ops.gemm_fp8
            else:
                raise ValueError("Not support quant config.")

            out = fp8_gemm(input_, weight, trans_a=False, trans_b=True, out_dtype=None, config=quant_config)
        else:
            out = self.gemm(input_, weight)
        out = out.view(original_shape[0], original_shape[1], -1)

        return out, bias_tensor


class PrimusTurboLayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
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
    ):
        self.config = config

        if gather_output:
            raise ValueError("Primus Turbo linear layers do not support gather_output = True")

        if is_expert:
            raise ValueError("Primus Turbo linear layers do not yet support MoE")

        if skip_weight_param_allocation:
            raise ValueError("Primus Turbo linear layers do not support skip_weight_param_allocation")

        # TODO: For backward compatibility, remove in v0.15.
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias

        args = get_args()
        if args.patch_zero_bubble and args.enable_zero_bubble:

            from .zbpp_gemm import gemm_with_weight_gradient_store

            self.gemm = lambda a, b, bias=None: gemm_with_weight_gradient_store(a, b, bias=bias)
        else:
            self.gemm = lambda a, b, trans_a=False, trans_b=True, out_dtype=None: pt.ops.gemm(
                a, b, trans_a=trans_a, trans_b=trans_b, out_dtype=out_dtype
            )

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            eps=self.config.layernorm_epsilon,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=tp_group if torch.distributed.is_initialized() else None,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=None,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            normalization=self.config.normalization,
            return_bias=self.te_return_bias,
            parallel_mode="column",
            return_layernorm_output=False,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
        )

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
        """Forward."""

        if self.config.normalization == "LayerNorm":
            norm_out = torch.nn.functional.layer_norm(
                x, [x.size(-1)], self.layer_norm_weight, self.layer_norm_bias, self.eps
            )
        elif self.config.normalization == "RMSNorm":
            norm_out = torch.nn.functional.rms_norm(x, [x.size(-1)], self.layer_norm_weight, self.eps)
        # weights = [getattr(self, name) for name in self.weight_names]
        # weights = torch.cat(weights, dim=0)
        weights = self._parameters["weight"]
        if self.use_bias:
            bias_tensor = torch.cat([getattr(self, name) for name in self.bias_names])
        original_shape = x.size()
        if not norm_out.is_contiguous():
            norm_out = norm_out.contiguous()
        inp = norm_out.view(-1, original_shape[-1])

        if PrimusTurboFP8GlobalStateManager.is_turbo_fp8_enabled():
            quant_config = PrimusTurboFP8GlobalStateManager.get_turbo_fp8_quant_config()
            if quant_config.block_scaling():
                fp8_gemm = pt.ops.gemm_fp8_blockwise
            elif quant_config.current_scaling():
                fp8_gemm = pt.ops.gemm_fp8
            else:
                raise ValueError("Not support quant config.")

            out = fp8_gemm(inp, weights, trans_a=False, trans_b=True, out_dtype=None, config=quant_config)
        else:
            out = self.gemm(inp, weights)

        out = out.view(original_shape[0], original_shape[1], -1)
        if self.te_return_bias:
            return out, bias_tensor
        if self.use_bias:
            return out + bias_tensor, None
        return out, None


class PrimusTurboGroupedMLP(GroupedMLP):
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            num_local_experts,
            config,
            pg_collection,
        )
        args = get_args()
        grouped_gemm_backend = args.grouped_gemm_backend
        self.grouped_gemm_backend = grouped_gemm_backend

        if args.patch_zero_bubble and args.enable_zero_bubble:
            from .zbpp_gemm import grouped_gemm_with_weight_gradient_store

            self.grouped_gemm = functools.partial(
                grouped_gemm_with_weight_gradient_store, gg_backend=grouped_gemm_backend
            )
        else:
            if grouped_gemm_backend == "turbo-gg":
                self.grouped_gemm = pt.ops.grouped_gemm
            elif grouped_gemm_backend == "lagacy-gg":
                self.grouped_gemm = grouped_gemm.ops.gmm
            else:
                raise NotImplementedError(f"Grouped gemm backend {grouped_gemm_backend} not implemented")

        if args.use_turbo_fused_act_with_probs:
            assert self.config.gated_linear_unit, "turbo_fused_act_with_probs only support with GLU."

            if self.config.activation_func == F.silu:
                turbo_fused_act_with_probs = turbo_moe_activation.swiglu_with_probs
            elif self.config.activation_func == F.gelu:
                turbo_fused_act_with_probs = turbo_moe_activation.geglu_with_probs
            else:
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            def _activation_func_with_probs(x, probs, tokens_per_experts):
                assert x.ndim == 2
                assert probs.ndim == 1
                num_tokens = x.shape[0]
                row_mask = turbo_tokens_per_expert_to_mask(tokens_per_experts, num_tokens)
                return turbo_fused_act_with_probs(x, probs, row_mask)

            self.activation_func_with_probs = _activation_func_with_probs

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the GroupedMLP."""
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        args = get_args()
        gemm_kargs = [dict(), dict()]
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            if args.patch_zero_bubble and args.enable_zero_bubble:

                w1 = self.weight1
                w2 = self.weight2

                gemm_kargs[0]["weight_reshape_size"] = (self.num_local_experts, self.config.hidden_size, -1)
                gemm_kargs[1]["weight_reshape_size"] = (self.num_local_experts, -1, self.config.hidden_size)
            else:
                w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
                w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            if self.grouped_gemm_backend == "turbo-gg":
                tokens_per_expert = tokens_per_expert.cuda()
            assert w1.is_contiguous(), "w1 must be contiguous"
            assert w2.is_contiguous(), "w2 must be contiguous"
            fc1_output = self.grouped_gemm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False, **(gemm_kargs[0])
            )
            if self.activation_recompute:
                if args.use_turbo_fused_act_with_probs:
                    intermediate_parallel = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs,
                        fc1_output,
                        permuted_probs,
                        tokens_per_expert,
                    )
                else:
                    intermediate_parallel = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs, fc1_output, permuted_probs.unsqueeze(-1)
                    )
                fc2_output = self.grouped_gemm(
                    intermediate_parallel, w2, tokens_per_expert, trans_b=False, **(gemm_kargs[1])
                )
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                if args.use_turbo_fused_act_with_probs:
                    intermediate_parallel = self.activation_func_with_probs(
                        fc1_output, permuted_probs, tokens_per_expert
                    )
                else:
                    intermediate_parallel = self.activation_func_with_probs(
                        fc1_output, permuted_probs.unsqueeze(-1)
                    )
                fc2_output = self.grouped_gemm(
                    intermediate_parallel, w2, tokens_per_expert, trans_b=False, **(gemm_kargs[1])
                )
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0
            # Make sure params of experts still have gradients even given zero tokens.
            assert not args.patch_zero_bubble, "Zero bubble not support torch.matmul backend yet"
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                if args.use_turbo_fused_act_with_probs:
                    h = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs, h, permuted_probs, tokens_per_expert
                    )
                else:
                    h = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs, h, permuted_probs.unsqueeze(-1)
                    )
                fc2_output = torch.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                if args.use_turbo_fused_act_with_probs:
                    h = self.activation_func_with_probs(h, permuted_probs, tokens_per_expert)
                else:
                    h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
                fc2_output = torch.matmul(h, w2)

        return fc2_output, None


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

        self.deepep_dispatcher = pt.modules.DeepEPTokenDispatcher(
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
            deepep_use_cuda_num_tokens_per_expert=(
                args.use_turbo_grouped_mlp
                and args.moe_use_legacy_grouped_gemm
                and args.grouped_gemm_backend == "turbo-gg"
            ),
            deepep_async_finish=True,
            deepep_allocate_on_comm_stream=True,
        )

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
