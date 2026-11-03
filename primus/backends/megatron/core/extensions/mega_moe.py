###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaMoE layer, drop-in for Megatron MoELayer (EP-only, bf16 params).

``turbo_mega_moe_precision`` selects the expert class; the flavours differ only in which stage pair
they call, so precision is decided once when the layer is built and nowhere else. ``mxfp8`` is an
op-internal change: w1/w2 stay bf16 parameters and the op maintains their mxfp8 quant in an
internal cache. Because the precision-aware optimizer may not bump ``w._version``, the fp8 path
advances a separate generation counter (``advance_weight_generation()``) at the optimizer-step
boundary to invalidate that cache, so initialization, checkpointing and the optimizer remain
unchanged.

The expert activation is SiLU-SwiGLU or Megatron's ``quick_geglu`` (``activation_func =
quick_gelu`` with ``glu_linear_offset`` / ``activation_func_clamp_value``, i.e. MiniMax-M3's and
GPT-OSS's ``swigluoai``); see :func:`mega_moe_activation`.

``add_bias_linear`` gives every expert an fc1 and an fc2 bias (GPT-OSS), applied the way Megatron's
grouped experts do: the fc1 bias before the activation and its clamp, the fc2 bias scaled by the
routing weight. Both flavours take them; the ops own how each is applied.
"""

import contextlib
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
from primus_turbo.pytorch.ops.moe.fused_mega_moe import (
    GLUActivation,
    fused_mega_moe_stage1,
    fused_mega_moe_stage2,
)
from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
    fused_mega_moe_fp8_stage1,
    fused_mega_moe_fp8_stage2,
)

# Megatron's quick_gelu hardcodes its sigmoid scale.
_QUICK_GELU_ALPHA = 1.702


def _is_silu(activation_func) -> bool:
    return activation_func in (F.silu, torch.nn.SiLU)


def mega_moe_activation(config: TransformerConfig) -> Optional[GLUActivation]:
    """The MegaMoE kernel activation for ``config``; ``None`` keeps the kernels' SiLU default.

    ``quick_gelu`` is Megatron's ``quick_geglu``: ``g * sigmoid(1.702 g) * (u + glu_linear_offset)``
    with the gate clamped from above only and ``up`` on both sides by ``activation_func_clamp_value``
    (``moe/experts.py``), which is MiniMax-M3's ``swigluoai``.

    SiLU always gets the default, whatever ``activation_func_clamp_value`` says: DeepSeek-V4 runs
    MegaMoE with a SiLU config that sets it, and honouring it would change that model's numerics.
    """
    if _is_silu(config.activation_func):
        return None
    assert config.activation_func is quick_gelu, f"no MegaMoE activation for {config.activation_func!r}"
    limit = config.activation_func_clamp_value
    return GLUActivation(
        alpha=_QUICK_GELU_ALPHA,
        glu_offset=float(config.glu_linear_offset),
        gate_clamp_lo=None,
        gate_clamp_hi=limit,
        up_clamp=limit,
    )


def mega_moe_precision() -> str:
    """Read ``turbo_mega_moe_precision`` off the Megatron args namespace."""
    from megatron.training import get_args

    supported = ("bf16", "mxfp8")
    precision = getattr(get_args(), "turbo_mega_moe_precision", "bf16") or "bf16"
    assert precision in supported, f"turbo_mega_moe_precision must be one of {supported}, got {precision!r}"
    return precision


def mega_moe_router(router):
    """The router builder MegaMoE uses: Megatron's ``TopKRouter`` becomes ``PrimusTopKRouter``.

    The ``megatron.moe.primus_topk_router`` patch makes that swap in ``MoELayer.__init__``, which this
    layer replaces, and in specs from the GPT MoE spec builders it wraps, which reach only modules that
    import those builders after the patch runs. So MegaMoE applies the same rule itself, unless
    ``disable_primus_topk_router`` is set.
    """
    from megatron.training import get_args

    from primus.backends.megatron.core.transformer.moe.router import (
        primus_topk_router_for,
    )

    primus_router = primus_topk_router_for(router)
    if primus_router is router or getattr(get_args(), "disable_primus_topk_router", False):
        return router
    return primus_router


class MegaMoEWeightModule(MegatronModule):
    """Callable expert-weight module used as a DDP overlap boundary.

    ``forward`` returns ``(weight, bias)``; ``bias`` is None unless ``bias_shape`` is given."""

    def __init__(self, config: TransformerConfig, weight_shape, bias_shape=None) -> None:
        super().__init__(config)
        device = torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = torch.nn.Parameter(torch.empty(weight_shape, device=device, dtype=config.params_dtype))
        # zeros, like the bias of Megatron's grouped expert linears
        self.bias = (
            torch.nn.Parameter(torch.zeros(bias_shape, device=device, dtype=config.params_dtype))
            if bias_shape is not None
            else None
        )

    def forward(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.weight, self.bias

    def backward_dw(self) -> None:
        # wgrad produced in custom autograd backward
        return None


class MegaMoEExperts(MegatronModule):
    """Two-stage bf16 expert with separately wrapped w1/w2 parameters."""

    def __init__(
        self,
        config: TransformerConfig,
        experts_per_rank: int,
        hidden_size: int,
        intermediate_size: int,
        ep_group,
    ) -> None:
        super().__init__(config)
        self.ep_group = ep_group
        self.experts_per_rank = experts_per_rank
        self.activation = mega_moe_activation(config)
        bias = config.add_bias_linear
        # w1 [g, 2I, H] gate+up; w2 [g, H, I] down; biases b1 [g, 2I], b2 [g, H]
        self.fc1_weight = MegaMoEWeightModule(
            config,
            (experts_per_rank, 2 * intermediate_size, hidden_size),
            (experts_per_rank, 2 * intermediate_size) if bias else None,
        )
        self.fc2_weight = MegaMoEWeightModule(
            config,
            (experts_per_rank, hidden_size, intermediate_size),
            (experts_per_rank, hidden_size) if bias else None,
        )
        # Expert weights are already EP-sharded, so the DP allreduce hook must
        # not touch them.
        expert_parallel = config.expert_model_parallel_size > 1
        for p in self.parameters():
            setattr(p, "allreduce", not expert_parallel)

    def reset_parameters(self, ep_rank: int) -> None:
        """Init expert weights exactly like Megatron TEGroupedLinear."""
        init_fc1 = self.config.init_method
        init_fc2 = self.config.output_layer_init_method
        assert init_fc1 is not None and init_fc2 is not None, "config init methods are unset"
        # fc1 <- init_method, fc2 <- output_layer_init_method
        weights = (
            (self.fc1_weight.weight, init_fc1),
            (self.fc2_weight.weight, init_fc2),
        )
        for module in (self.fc1_weight, self.fc2_weight):
            if module.bias is not None:
                module.bias.data.zero_()

        if self.config.use_cpu_initialization:
            # cpu rng is rank-identical: draw all experts, keep this rank's shard
            first_expert = ep_rank * self.experts_per_rank
            for weight, init_method in weights:
                master = torch.empty(weight.shape[1:], dtype=torch.float32)
                for e in range(self.config.num_moe_experts):
                    init_method(master)
                    if first_expert <= e < first_expert + self.experts_per_rank:
                        weight.data[e - first_expert].copy_(master)
            return

        tracker = get_cuda_rng_tracker()
        rng_fork = (
            tracker.fork(get_expert_parallel_rng_tracker_name())
            if tracker.is_initialized()
            else contextlib.nullcontext()
        )
        with rng_fork:
            # init each expert as a 2D [out, in] slice, like TE per-gemm weights
            for weight, init_method in weights:
                for expert_weight in weight.data:
                    init_method(expert_weight)

    def forward(self, x, topk_idx, topk_weights):
        # w2 is fetched only after stage1 so the two weight modules keep their fc1-then-fc2 order.
        w1, b1 = self.fc1_weight()
        l1_out, dwib, handle = fused_mega_moe_stage1(x, topk_idx, topk_weights, w1, self.ep_group)
        w2, b2 = self.fc2_weight()
        # stage2 owns the activation, so fc1's bias, which sits in front of it, is applied there
        return fused_mega_moe_stage2(
            l1_out,
            dwib,
            handle,
            topk_idx,
            topk_weights,
            w2,
            self.ep_group,
            self.activation,
            fc1_bias=b1,
            fc2_bias=b2,
        )

    def backward_dw(self) -> None:
        # match native fc2-then-fc1 order
        self.fc2_weight.backward_dw()
        self.fc1_weight.backward_dw()


class MegaMoEFP8Experts(MegaMoEExperts):
    """MXFP8 sibling: same parameters and same two-stage shape, different stage pair.

    The fp8 stages thread an extra opaque ``state`` from stage1 to stage2, which carries the
    quantized operands that cannot ride autograd's gradient slots.
    """

    # The mxfp8 weight-quant cache is dropped by the megatron.turbo.mega_moe_weight_generation
    # patch, which advances the generation once per optimizer step. Doing it there rather than
    # here keeps the refresh tied to the weights actually moving, not to how often this layer
    # happens to run a forward.
    def forward(self, x, topk_idx, topk_weights):
        w1, b1 = self.fc1_weight()
        l1_out, dwib, handle, state = fused_mega_moe_fp8_stage1(x, topk_idx, topk_weights, w1, self.ep_group)
        w2, b2 = self.fc2_weight()
        return fused_mega_moe_fp8_stage2(
            l1_out,
            dwib,
            handle,
            state,
            topk_idx,
            topk_weights,
            w2,
            self.ep_group,
            self.activation,
            fc1_bias=b1,
            fc2_bias=b2,
        )


class PrimusTurboMegaMoELayer(MegatronModule):
    """EP MoE layer: Megatron router -> two-stage experts -> shared expert."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[object] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        hash_moe_layer_threshold: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(config)

        assert (
            pg_collection is not None and pg_collection.ep is not None
        ), "MegaMoE requires an expert-parallel process group"
        assert submodules is not None, "MegaMoE requires MoESubmodules (router/shared_experts)"
        self._assert_supported_config(config)

        self.config = config
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer

        # experts sharded evenly across EP group
        self.ep_group = pg_collection.ep
        self.ep_size = self.ep_group.size()
        self.ep_rank = self.ep_group.rank()
        assert config.num_moe_experts % self.ep_size == 0, "num_experts must be divisible by EP size"
        self.experts_per_rank = config.num_moe_experts // self.ep_size

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_ffn_hidden_size

        # router owns all routing logic
        self.router = mega_moe_router(submodules.router)(
            config=config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            # passed by Megatron-LM main (core 0.20), whose routers take it
            **(
                {"hash_moe_layer_threshold": hash_moe_layer_threshold}
                if hash_moe_layer_threshold is not None
                else {}
            ),
        )

        # separate w1/w2 modules give DDP overlap boundaries
        experts_cls = MegaMoEFP8Experts if mega_moe_precision() == "mxfp8" else MegaMoEExperts
        self.experts = experts_cls(
            config,
            self.experts_per_rank,
            self.hidden_size,
            self.intermediate_size,
            self.ep_group,
        )
        if config.perform_initialization:
            self.reset_parameters()

        # optional shared expert
        self.use_shared_expert = config.moe_shared_expert_intermediate_size is not None
        if self.use_shared_expert:
            assert (
                submodules.shared_experts is not None
            ), "Shared experts builder is not provided in the module spec."
            shared_experts_kwargs = dict(
                config=config, pg_collection=pg_collection, gate=config.moe_shared_expert_gate
            )
            if isinstance(submodules.shared_experts, ModuleSpec):
                self.shared_experts = build_module(submodules.shared_experts, **shared_experts_kwargs)
            else:
                # Megatron-LM core_v0.19+: a builder callable, which also takes the module name
                self.shared_experts = submodules.shared_experts(
                    **shared_experts_kwargs, name=(name + ".shared_experts") if name is not None else None
                )
        else:
            self.shared_experts = None

    def reset_parameters(self) -> None:
        """Init expert weights exactly like Megatron TEGroupedLinear."""
        self.experts.reset_parameters(self.ep_rank)

    @staticmethod
    def _assert_supported_config(config: TransformerConfig) -> None:
        """Only kernel-level constraints; routing features are handled by the router."""
        assert config.tensor_model_parallel_size == 1, "MegaMoE is EP-only (TP==1)"
        # Holds for the fp8 path too: fp8 is internal to the op, the parameters stay bf16.
        assert config.params_dtype == torch.bfloat16, "MegaMoE only supports bf16 params"
        assert config.gated_linear_unit, "MegaMoE fuses a gated (GLU) MLP only"
        # `is`, not ==: Megatron defines quick_gelu twice, and configs carry the fusions one.
        assert _is_silu(config.activation_func) or config.activation_func is quick_gelu, (
            "MegaMoE supports SiLU-SwiGLU and quick_geglu (swigluoai) only, got "
            f"{config.activation_func!r}"
        )
        assert (
            config.moe_expert_capacity_factor is None
        ), "MegaMoE is dropless; moe_expert_capacity_factor must be None"
        assert not config.init_model_with_meta_device, "MegaMoE does not support meta-device init"
        load_balancing_type = config.moe_router_load_balancing_type
        load_balancing_types = (
            load_balancing_type if isinstance(load_balancing_type, (list, tuple)) else [load_balancing_type]
        )
        assert "sinkhorn" not in load_balancing_types, "MegaMoE does not support sinkhorn load balancing"

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        in_shape = hidden_states.shape
        # transpose [bsz, seq] -> [seq, bsz]
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        # Megatron-LM main hands its router the token ids (hash routing) and the packed-sequence
        # params (aux-loss groups) through the MoE layer; older routers take neither
        router_kwargs = {
            key: value
            for key, value in (("input_ids", input_ids), ("packed_seq_params", packed_seq_params))
            if value is not None
        }
        # dense probs [T, E], aux-loss grad baked in
        probs, _ = self.router(hidden_states, padding_mask, **router_kwargs)
        probs = probs.reshape(-1, self.config.num_moe_experts)

        # dense -> sparse top-k
        topk_weights, topk_idx = probs.topk(self.router.topk, dim=-1)

        x = hidden_states.reshape(-1, self.hidden_size).to(torch.bfloat16)
        y = self.experts(
            x,
            topk_idx,
            topk_weights.to(torch.float32),
        )
        y = y.reshape(in_shape).to(hidden_states.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(hidden_states)
        return y, None

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[dict] = None,
    ):
        # delegates below need metadata["dp_cp_group"]
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        prepend_axis_num = len(sharded_offsets)
        edp_rank = parallel_state.get_expert_data_parallel_rank()
        expert_replica_id = (0, 0, edp_rank)

        sharded_sd: dict = {}
        # expert weights and biases: EP-sharded on axis 0
        for module_name in ("fc1_weight", "fc2_weight"):
            module = getattr(self.experts, module_name)
            for param_name in ("weight", "bias"):
                param = getattr(module, param_name)
                if param is None:
                    continue
                key = f"{prefix}experts.{module_name}.{param_name}"
                sharded_sd[key] = ShardedTensor.from_rank_offsets(
                    key,
                    param,
                    *sharded_offsets,
                    (prepend_axis_num, self.ep_rank, self.ep_size),
                    replica_id=expert_replica_id,
                    prepend_axis_num=prepend_axis_num,
                )
        # router + shared expert: delegate to their own sharded_state_dict
        sharded_sd.update(self.router.sharded_state_dict(f"{prefix}router.", sharded_offsets, metadata))
        if self.shared_experts is not None:
            sharded_sd.update(
                self.shared_experts.sharded_state_dict(f"{prefix}shared_experts.", sharded_offsets, metadata)
            )
        return sharded_sd

    def set_layer_number(self, layer_number: int) -> None:
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)

    def backward_dw(self, *args: object, **kwargs: object) -> None:
        return None
