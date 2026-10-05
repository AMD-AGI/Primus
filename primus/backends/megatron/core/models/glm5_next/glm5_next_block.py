###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 (``glm5_next``) decoder layer and block.

Layer (SGLang ``Glm5NextDecoderLayer`` with ``MHCLayerCommunicator``)::

    x, post, comb = mhc_pre(streams, hc_attn_*)        # [s, b, h]
    streams = mhc_post(attn(input_layernorm(x)), streams, post, comb)
    x, post, comb = mhc_pre(streams, hc_ffn_*)
    streams = mhc_post(mlp(pre_mlp_layernorm(x)), streams, post, comb)

Block: ``streams = expand(embeddings)`` on the first PP stage, the layers, then
``final_layernorm(mean(streams))`` on the last. Between PP stages the
``[s, b, n, h]`` stream tensor travels as ``[n * s, b, h]`` (the same carrier
DeepSeek-V4 uses), so the stock 3-D P2P path is untouched apart from the
first-dim scale handled by the PP shape patch.

Sequence parallelism: everything here is per token, so the layer runs on the
local ``s / tp`` shard; the mHC parameters are replicated and carry the
``sequence_parallel`` flag so their partial gradients get summed over TP.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor
from torch import Tensor, nn

try:
    from megatron.core.transformer.transformer_block import (
        get_cpu_offload_context as _get_cpu_offload_context,
    )
except ImportError:  # pragma: no cover
    _get_cpu_offload_context = None

from primus.backends.megatron.core.transformer.glm5_next.fp32_params import (
    KeepFp32ParamsMixin,
)
from primus.backends.megatron.core.transformer.glm5_next.mhc import (
    mhc_contract,
    mhc_expand,
    mhc_post,
    mhc_pre,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Glm5NextLayer",
    "Glm5NextLayerSubmodules",
    "Glm5NextTransformerBlock",
    "Glm5NextTransformerBlockSubmodules",
]


@dataclass
class Glm5NextLayerSubmodules:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp


@dataclass
class Glm5NextTransformerBlockSubmodules:
    layer_specs: Optional[List[ModuleSpec]] = None
    final_layernorm: Optional[Union[ModuleSpec, type]] = None


def _hc_param(*shape: int) -> nn.Parameter:
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    return nn.Parameter(torch.zeros(*shape, dtype=torch.float32, device=device))


class Glm5NextLayer(KeepFp32ParamsMixin, TransformerLayer):
    """One GLM-5.3 decoder layer (KDA or DSA attention; dense MLP or MoE).

    ``TransformerLayer.__init__`` is bypassed (as in ``DeepseekV4HybridLayer``):
    there is no BDA / cross-attention, and the residual is the mHC stream set.
    """

    def __init__(
        self,
        config,
        submodules: Glm5NextLayerSubmodules,
        layer_number: int = 1,
        *,
        layer_idx: int,
        is_kda_layer: bool,
        pg_collection=None,
        vp_stage: Optional[int] = None,
        **_unused,
    ) -> None:
        MegatronModule.__init__(self, config=config)
        self.submodules_config = submodules
        self.layer_idx = int(layer_idx)
        self.layer_number = self.layer_idx + 1
        self.is_kda_layer = bool(is_kda_layer)
        self.pg_collection = pg_collection
        self.vp_stage = vp_stage
        self.hc_mult = int(config.hc_mult)
        hidden = int(config.hidden_size)
        eps = float(config.layernorm_epsilon)

        self.input_layernorm = build_module(
            submodules.input_layernorm, config=config, hidden_size=hidden, eps=eps
        )
        self.self_attention = build_module(
            submodules.self_attention,
            config=config,
            layer_number=self.layer_number,
            pg_collection=pg_collection,
        )
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm, config=config, hidden_size=hidden, eps=eps
        )
        mlp_kwargs = {}
        if isinstance(submodules.mlp, ModuleSpec) and submodules.mlp.module is MoELayer:
            mlp_kwargs["pg_collection"] = pg_collection
        elif isinstance(submodules.mlp, ModuleSpec) and submodules.mlp.module is MLP:
            mlp_kwargs["tp_group"] = pg_collection.tp
        if isinstance(submodules.mlp, ModuleSpec) or isinstance(submodules.mlp, type):
            self.mlp = build_module(submodules.mlp, config=config, **mlp_kwargs)
        else:
            # Megatron >= 0.19 returns MLP builders (functools.partial) instead of ModuleSpecs.
            self.mlp = submodules.mlp(
                config=config, pg_collection=pg_collection, layer_number=self.layer_number
            )
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.layer_number)
        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        n = self.hc_mult
        mix = (2 + n) * n
        self.hc_attn_fn = _hc_param(mix, n * hidden)
        self.hc_attn_base = _hc_param(mix)
        self.hc_attn_scale = _hc_param(3)
        self.hc_ffn_fn = _hc_param(mix, n * hidden)
        self.hc_ffn_base = _hc_param(mix)
        self.hc_ffn_scale = _hc_param(3)
        hc_names = ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale")
        self._keep_fp32_enabled = bool(getattr(config, "glm5_keep_fp32_params", True))
        self._fp32_param_names = hc_names
        for name in hc_names:
            setattr(getattr(self, name), "sequence_parallel", bool(config.sequence_parallel))
        self._reset_hc_parameters()

        self._hc_kwargs = dict(
            rms_eps=eps,
            hc_eps=float(config.hc_eps),
            sinkhorn_iters=int(config.hc_sinkhorn_iters),
            post_mult=float(config.hc_post_mult),
        )

    def _reset_hc_parameters(self) -> None:
        if not self.config.perform_initialization:
            return
        with torch.no_grad():
            for fn in (self.hc_attn_fn, self.hc_ffn_fn):
                fn.normal_(0.0, float(self.config.init_method_std))
            for scale in (self.hc_attn_scale, self.hc_ffn_scale):
                scale.fill_(1.0)
            for base in (self.hc_attn_base, self.hc_ffn_base):
                base.zero_()

    def _sub(self, streams: Tensor, fn, scale, base, norm, body) -> Tensor:
        x, post, comb = mhc_pre(streams, fn, scale, base, **self._hc_kwargs)
        out, bias = body(norm(x))
        if bias is not None:
            out = out + bias
        return mhc_post(out, streams, post, comb)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        packed_seq_params=None,
        **kwargs,
    ):
        """``hidden_states``: ``[s, b, n, h]`` stream set. Returns ``(streams, None)``."""
        streams = self._sub(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.input_layernorm,
            lambda x: self.self_attention(
                x, attention_mask=attention_mask, packed_seq_params=packed_seq_params
            ),
        )
        streams = self._sub(
            streams,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.pre_mlp_layernorm,
            self.mlp,
        )
        return streams, None


class Glm5NextTransformerBlock(TransformerBlock):
    """GLM-5.3 decoder: mHC stream expand / layers / mean-contract + final norm."""

    def __init__(
        self,
        config,
        spec=None,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection=None,
        vp_stage=None,
        submodules: Optional[Glm5NextTransformerBlockSubmodules] = None,
    ) -> None:
        MegatronModule.__init__(self, config=config)
        self.spec = spec
        self.submodules = submodules
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage
        self.pg_collection = pg_collection
        self.input_tensor = None
        self.hc_mult = int(config.hc_mult)

        self.offload_context, self.group_prefetch_offload_commit_async = nullcontext(), None
        if _get_cpu_offload_context is not None:
            offload_args = [
                config.cpu_offloading,
                config.cpu_offloading_num_layers,
                config.num_layers,
                config.cpu_offloading_activations,
                config.cpu_offloading_weights,
                config.cpu_offloading_double_buffering,
            ]
            # Megatron >= 0.19 added a required retain_pinned_cpu_buffers argument.
            if hasattr(config, "cpu_offloading_retain_pinned_cpu_buffers"):
                offload_args.append(config.cpu_offloading_retain_pinned_cpu_buffers)
            self.offload_context, self.group_prefetch_offload_commit_async = _get_cpu_offload_context(
                *offload_args
            )
            config._cpu_offloading_context = self.offload_context if config.cpu_offloading else None

        layer_specs = submodules.layer_specs if submodules is not None else None
        assert layer_specs is not None, "Glm5NextTransformerBlock requires submodules.layer_specs."
        self.layers = nn.ModuleList(
            [
                build_module(spec, config=config, pg_collection=pg_collection, vp_stage=vp_stage)
                for spec in layer_specs
            ]
        )
        self.global_layer_indices = [int(layer.layer_idx) for layer in self.layers]

        if self.post_process and self.post_layer_norm and submodules.final_layernorm is not None:
            self.final_layernorm = build_module(
                submodules.final_layernorm,
                config=config,
                hidden_size=config.hidden_size,
                eps=config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None

    def set_input_tensor(self, input_tensor: Tensor):
        self.input_tensor = input_tensor

    @property
    def num_layers_per_pipeline_rank(self) -> int:
        return len(self.layers)

    # ---- stream carrier across PP boundaries ---------------------------
    def _lift(self, hidden_states: Tensor) -> Tensor:
        n = self.hc_mult
        if self.pre_process:
            return mhc_expand(hidden_states, n)
        ns, b, h = hidden_states.shape
        assert ns % n == 0, f"PP carrier first dim {ns} is not a multiple of hc_mult={n}"
        return hidden_states.view(n, ns // n, b, h).permute(1, 2, 0, 3)

    def _lower(self, streams: Tensor) -> Tensor:
        s, b, n, h = streams.shape
        return streams.permute(2, 0, 1, 3).reshape(n * s, b, h)

    # ---- layer loop ------------------------------------------------------
    def _recompute_local(self) -> Optional[set]:
        cfg = self.config
        if not self.training or getattr(cfg, "recompute_granularity", None) != "full":
            return None
        n_local = len(self.layers)
        num = int(getattr(cfg, "recompute_num_layers", 0) or 0)
        if n_local == 0 or num <= 0:
            return None
        method = getattr(cfg, "recompute_method", None) or "block"
        if method == "block":
            return set(range(min(num, n_local)))
        if method == "uniform":
            return set(range(n_local))
        raise ValueError(f"Invalid recompute_method for GLM-5.3: {method!r}")

    def _layer_fp8_context(self, global_idx: int):
        if getattr(self.config, "fp4", None):
            from megatron.core import fp4_utils

            return fp4_utils.get_fp4_context(self.config, global_idx)
        if not self.config.fp8:
            return nullcontext()
        from megatron.core import fp8_utils

        return fp8_utils.get_fp8_context(self.config, global_idx)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ) -> Tensor:
        if not self.pre_process:
            hidden_states = self.input_tensor
        assert hidden_states is not None, "Glm5NextTransformerBlock received no hidden_states"

        x = self._lift(hidden_states)
        recompute = self._recompute_local()
        for local_idx, layer in enumerate(self.layers):
            global_idx = self.global_layer_indices[local_idx]
            if recompute is not None and local_idx in recompute:

                def _run(h, _layer=layer, _gi=global_idx):
                    with self._layer_fp8_context(_gi):
                        out, _ = _layer(h, attention_mask=attention_mask, packed_seq_params=packed_seq_params)
                    return out

                x = tensor_parallel.checkpoint(_run, self.config.distribute_saved_activations, x)
            else:
                with self.offload_context, self._layer_fp8_context(global_idx):
                    x, _ = layer(x, attention_mask=attention_mask, packed_seq_params=packed_seq_params)
                if (
                    torch.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    x = self.group_prefetch_offload_commit_async(x)

        if self.post_process:
            out = mhc_contract(x)
            if self.final_layernorm is not None:
                out = self.final_layernorm(out)
        else:
            out = self._lower(x)
        return make_viewless_tensor(inp=out, requires_grad=out.requires_grad, keep_graph=True)
