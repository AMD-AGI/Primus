###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 Mixture-of-Experts module.

Reference: techblog §4 ("MoE: hash routing + sqrtsoftplus + shared experts")
and ``DeepSeek-V4-Flash/inference/model.py:MoE``.

V4's MoE block has three pieces:

1. **Router** — either :class:`DeepseekV4HashRouter` (first
   ``num_hash_layers`` layers) or :class:`DeepseekV4LearnedRouter` (the
   rest). Both produce the same ``(probs, routing_map)`` shape contract:
   ``[N, num_experts]``. The two routers share a learned gate weight;
   only the *selection* differs (top-K argmax for the learned router,
   ``tid2eid`` lookup for the hash router). Routing weights always come
   from the same ``v4_score_fn(linear(hidden, weight))`` path.
2. **Routed experts** — ``num_experts`` clamped-SwiGLU MLPs. Each token
   contributes to ``moe_router_topk`` of them, weighted by the router
   probability. The clamp is **pre-multiplication**:
   ``SiLU(clamp(gate, max=alpha)) * clamp(up, +/- alpha)``.
3. **Shared expert(s)** — always-on MLP(s) whose output is added to every
   token's contribution. V4-Flash has 1 shared expert with the same
   ``moe_intermediate_size`` as the routed experts.

Plan-2 P14 contract:

P14 phase-1 (committed in 1a8bf32e) — math + parameter-layout
faithfulness: pre-multiplication clamped SwiGLU activation, learned
router rewritten with HF-aligned scoring + bias-only-for-selection
semantics, hash router rewritten with a learnable gate weight + frozen
``tid2eid`` Parameter.

P14 phase-2 (this commit) — structural bring-up:
* :class:`DeepseekV4MoE` now subclasses :class:`MegatronModule` (was
  ``nn.Module``) so it integrates with Megatron's spec lifecycle and
  shares config plumbing with the rest of the V4 stack.
* CPU-friendly local-experts path: when ``pg_collection`` is ``None``
  (or when the grouped backend does not declare clamped-SwiGLU support),
  :class:`DeepseekV4MoE` builds a :class:`nn.ModuleList` of
  :class:`ClampedSwiGLUMLP` routed experts plus a single
  :class:`ClampedSwiGLUMLP` shared expert and runs a per-expert dispatch
  loop in ``forward`` that mirrors the HF reference exactly. This makes
  the MoE forward unit-testable on CPU at G5 (1L MoE forward agreement
  vs HF reference within 1e-3 fp32) without requiring distributed init.
* :meth:`set_layer_number` mirrors :class:`BaseMoELayer` so this module
  slots into ``TransformerLayer`` via the spec lifecycle.
* :attr:`local_expert_indices` exposed for compatibility with downstream
  tooling that expects the ``BaseMoELayer`` public surface.

Aux-loss / z-loss inheritance via :class:`TopKRouter` is left as a
follow-up: the V4 routers are standalone ``nn.Module``\\ s rather than
subclasses of Megatron's :class:`TopKRouter` (the parent registers CUDA
buffers in ``__init__`` and is impractical to instantiate on CPU). The
distributed re-validation phase (P19) will re-introduce that path
behind a TopKRouter subclass once the CUDA-buffer init is gated by a
device check upstream.
"""

from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.clamped_swiglu import ClampedSwiGLUMLP
from primus.backends.megatron.core.transformer.moe.v4_hash_router import (
    DeepseekV4HashRouter,
)
from primus.backends.megatron.core.transformer.moe.v4_topk_router import (
    DeepseekV4LearnedRouter,
)

logger = logging.getLogger(__name__)


@dataclass
class DeepseekV4MoESubmodules:
    """Spec tree for V4 MoE construction."""

    hash_router: Optional[Union[ModuleSpec, type]] = DeepseekV4HashRouter
    learned_router: Optional[Union[ModuleSpec, type]] = DeepseekV4LearnedRouter
    token_dispatcher: Optional[Union[ModuleSpec, type]] = MoEAlltoAllTokenDispatcher
    grouped_experts: Optional[Union[ModuleSpec, type]] = None
    shared_expert: Optional[Union[ModuleSpec, type]] = SharedExpertMLP


class DeepseekV4MoE(MegatronModule):
    """V4 MoE FFN sub-block.

    Args:
        config: runtime DeepSeek-V4 config. Core MoE dimensions and router
            options are read directly from config.
        layer_idx: 0-based decoder layer index. Used to pick router type
            against ``num_hash_layers``.
        pg_collection: Megatron process-group collection. When ``None``
            (CPU unit tests), the module skips the distributed dispatcher
            and builds a local :class:`nn.ModuleList` of
            :class:`ClampedSwiGLUMLP` routed experts plus a single
            :class:`ClampedSwiGLUMLP` shared expert; ``forward`` runs a
            per-expert dispatch loop matching the HF reference math.
        submodules: spec tree describing routers / dispatcher / experts /
            shared expert. Must be provided.
        layer_number: optional 1-based layer number used by Megatron's
            spec lifecycle (mirrors :class:`BaseMoELayer.set_layer_number`).
    """

    def __init__(
        self,
        config: DeepSeekV4TransformerConfig,
        *,
        layer_idx: int,
        pg_collection=None,
        submodules: Optional[DeepseekV4MoESubmodules] = None,
        layer_number: Optional[int] = None,
    ) -> None:
        if config is None:
            raise ValueError("DeepSeek-V4 MoE requires config.")
        super().__init__(config=config)
        self.pg_collection = pg_collection
        self.submodules = submodules
        assert self.submodules is not None, "DeepSeek-V4 MoE requires explicit submodules."
        self.layer_number = layer_number

        hidden_size = int(config.hidden_size)
        moe_intermediate_size = int(
            config.moe_ffn_hidden_size or config.moe_intermediate_size or config.ffn_hidden_size
        )
        num_routed_experts = int(config.num_moe_experts)
        moe_router_topk = int(config.moe_router_topk)
        use_shared_expert = config.moe_shared_expert_intermediate_size is not None
        layer_num_hash_layers = int(config.num_hash_layers)
        layer_hash_vocab_size = config.padded_vocab_size or config.vocab_size
        layer_hash_seed = int(config.hash_routing_seed)
        score_function = str(config.moe_router_score_function)
        enable_expert_bias = bool(config.moe_router_enable_expert_bias)
        topk_scaling_factor = float(getattr(config, "moe_router_topk_scaling_factor", 1.0) or 1.0)
        clamp_alpha = float(config.swiglu_limit)

        if num_routed_experts <= 0:
            raise ValueError(f"num_routed_experts must be > 0, got {num_routed_experts}")
        if moe_router_topk <= 0 or moe_router_topk > num_routed_experts:
            raise ValueError(f"moe_router_topk must be in [1, {num_routed_experts}], got {moe_router_topk}")

        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_routed_experts = num_routed_experts
        self.moe_router_topk = moe_router_topk
        self.use_shared_expert = use_shared_expert
        self.layer_idx = int(layer_idx)
        self.num_hash_layers = layer_num_hash_layers
        self.use_hash_router = self.layer_idx < self.num_hash_layers
        self.clamp_alpha = clamp_alpha
        self.moe_token_dispatcher_type = "alltoall"

        # ---- EP placement ----
        self.ep_group = getattr(pg_collection, "ep", None) if pg_collection is not None else None
        self.ep_size = 1
        self.ep_rank = 0
        if self.ep_group is None and dist.is_available() and dist.is_initialized():
            try:
                self.ep_group = parallel_state.get_expert_model_parallel_group()
            except Exception:
                self.ep_group = None
        if self.ep_group is not None and dist.is_available() and dist.is_initialized():
            self.ep_size = int(self.ep_group.size())
            self.ep_rank = int(self.ep_group.rank())

        base = self.num_routed_experts // self.ep_size
        remainder = self.num_routed_experts % self.ep_size
        self.local_num_routed_experts = base + (1 if self.ep_rank < remainder else 0)
        self.local_expert_start = (self.ep_rank * base) + min(self.ep_rank, remainder)
        self.local_expert_end = self.local_expert_start + self.local_num_routed_experts
        # BaseMoELayer-compatible public attribute.
        self.local_expert_indices = list(range(self.local_expert_start, self.local_expert_end))

        # ---- routers ----
        self.router = None
        self.learned_router = None
        self._build_router_modules(
            hash_vocab_size=layer_hash_vocab_size,
            hash_seed=layer_hash_seed,
            score_function=score_function,
            enable_expert_bias=enable_expert_bias,
            topk_scaling_factor=topk_scaling_factor,
        )

        # ---- experts ----
        # Production path: full Megatron dispatcher + grouped-experts.
        # CPU path: a local nn.ModuleList of ClampedSwiGLUMLP experts + a
        # single ClampedSwiGLUMLP shared expert. The CPU path is used when
        # ``pg_collection is None`` so unit tests can drive ``forward``
        # without distributed initialization.
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.grouped_experts: Optional[nn.Module] = None
        self.local_experts: Optional[nn.ModuleList] = None
        self.shared_expert: Optional[nn.Module] = None

        if pg_collection is None:
            self.local_experts = self._build_local_experts(intermediate_size=self.moe_intermediate_size)
            if self.use_shared_expert:
                assert self.config.moe_shared_expert_intermediate_size is not None
                self.shared_expert = ClampedSwiGLUMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=int(self.config.moe_shared_expert_intermediate_size),
                    alpha=self.clamp_alpha,
                    bias=False,
                )
        else:
            self.token_dispatcher = self._build_token_dispatcher()
            self.grouped_experts = self._build_grouped_experts()
            if self.use_shared_expert:
                assert self.config.moe_shared_expert_intermediate_size is not None
                self.shared_expert = self._build_shared_expert_module(
                    intermediate_size=int(self.config.moe_shared_expert_intermediate_size)
                )

    # ------------------------------------------------------------------

    def set_layer_number(self, layer_number: int) -> None:
        """Mirror :class:`BaseMoELayer.set_layer_number` for spec lifecycle.

        Megatron's :class:`TransformerLayer` walks every spec submodule
        with a ``set_layer_number`` method to populate the 1-based layer
        index. The V4 routers are intentionally standalone (CPU-clean),
        but we still need to track ``layer_number`` here so future
        TopKRouter-rooted upgrades plug in without spec changes.
        """
        self.layer_number = layer_number

    def _build_local_experts(self, *, intermediate_size: int) -> nn.ModuleList:
        """Build a local :class:`nn.ModuleList` of clamped-SwiGLU experts.

        Used when ``pg_collection is None`` (CPU unit tests). Each module
        in the list mirrors a single HF reference ``Expert`` (separate
        ``w1`` / ``w2`` / ``w3`` Linears + V4 pre-mul clamp).
        """
        if self.local_num_routed_experts <= 0:
            raise RuntimeError(f"DeepSeek-V4 MoE layer={self.layer_idx} has no local experts.")
        return nn.ModuleList(
            [
                ClampedSwiGLUMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=intermediate_size,
                    alpha=self.clamp_alpha,
                    bias=False,
                )
                for _ in range(self.local_num_routed_experts)
            ]
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_dispatcher_type_from_spec(dispatcher_spec: Optional[Union[ModuleSpec, type]]) -> str:
        module = dispatcher_spec.module if isinstance(dispatcher_spec, ModuleSpec) else dispatcher_spec
        if module is MoEAllGatherTokenDispatcher:
            return "allgather"
        if module is MoEFlexTokenDispatcher:
            return "flex"
        if module is MoEAlltoAllTokenDispatcher or module is None:
            return "alltoall"
        logger.warning(
            "[DeepSeek-V4] unsupported dispatcher module=%s; fallback type to alltoall.",
            getattr(module, "__name__", str(module)),
        )
        return "alltoall"

    def _build_router_modules(
        self,
        *,
        hash_vocab_size: Optional[int],
        hash_seed: int,
        score_function: str,
        enable_expert_bias: bool,
        topk_scaling_factor: float,
    ) -> None:
        if self.use_hash_router:
            if hash_vocab_size is None or hash_vocab_size <= 0:
                raise ValueError(
                    "hash_vocab_size must be provided (and > 0) when layer_idx < num_hash_layers"
                )
            hash_router_spec = self.submodules.hash_router or DeepseekV4HashRouter
            self.router = build_module(
                hash_router_spec,
                hidden_size=self.hidden_size,
                num_experts=self.num_routed_experts,
                topk=self.moe_router_topk,
                vocab_size=hash_vocab_size,
                seed=hash_seed,
                score_function=score_function,
                topk_scaling_factor=topk_scaling_factor,
            )
            self.learned_router = None
            return

        learned_router_spec = self.submodules.learned_router or DeepseekV4LearnedRouter
        self.router = None
        self.learned_router = build_module(
            learned_router_spec,
            hidden_size=self.hidden_size,
            num_experts=self.num_routed_experts,
            topk=self.moe_router_topk,
            score_function=score_function,
            enable_expert_bias=enable_expert_bias,
            topk_scaling_factor=topk_scaling_factor,
        )

    def _build_shared_expert_module(self, *, intermediate_size: int) -> nn.Module:
        shared_expert_spec = self.submodules.shared_expert
        assert isinstance(
            shared_expert_spec, ModuleSpec
        ), "DeepSeek-V4 MoE requires shared_expert ModuleSpec in submodules."
        shared_expert_module = shared_expert_spec.module
        assert shared_expert_module is SharedExpertMLP, "DeepSeek-V4 shared_expert must be SharedExpertMLP."
        if self.config is None or self.pg_collection is None:
            raise RuntimeError("DeepSeek-V4 MoE SharedExpertMLP requires config and pg_collection.")

        # Shared experts must always run with clamped SwiGLU via SharedExpertMLP.
        shared_cfg = copy(self.config)
        shared_cfg.add_bias_linear = False
        shared_cfg.gated_linear_unit = True
        shared_cfg.activation_func = F.silu
        shared_cfg.bias_activation_fusion = False
        shared_cfg.use_te_activation_func = False
        if self.clamp_alpha > 0:
            shared_cfg.activation_func_clamp_value = float(self.clamp_alpha)
        else:
            shared_cfg.activation_func_clamp_value = None
        if int(shared_cfg.moe_shared_expert_intermediate_size or 0) <= 0:
            setattr(
                shared_cfg,
                "moe_shared_expert_intermediate_size",
                int(intermediate_size),
            )

        try:
            return build_module(
                shared_expert_spec,
                config=shared_cfg,
                pg_collection=self.pg_collection,
                gate=bool(shared_cfg.moe_shared_expert_gate),
            )
        except Exception as exc:
            raise RuntimeError(
                f"DeepSeek-V4 MoE shared expert build failed with SharedExpertMLP: {exc}"
            ) from exc

    def _build_token_dispatcher(self) -> MoETokenDispatcher:
        if self.config is None or self.pg_collection is None:
            raise RuntimeError(
                "DeepSeek-V4 MoE requires config and pg_collection for Megatron dispatcher path."
            )
        if self.local_num_routed_experts <= 0:
            raise RuntimeError(
                f"DeepSeek-V4 MoE layer={self.layer_idx} has no local experts for dispatcher path."
            )

        dispatcher_spec: Union[ModuleSpec, type, None] = self.submodules.token_dispatcher
        assert dispatcher_spec is not None, "DeepSeek-V4 MoE requires token_dispatcher spec in submodules."
        requested_dispatcher_type = self._resolve_dispatcher_type_from_spec(dispatcher_spec)
        ep_group = getattr(self.pg_collection, "ep", None)
        tp_ep_group = getattr(self.pg_collection, "tp_ep", None)
        if requested_dispatcher_type == "alltoall" and ep_group is None:
            logger.info(
                "[DeepSeek-V4] MoE layer=%s alltoall dispatcher requires EP group.",
                self.layer_idx,
            )
        if requested_dispatcher_type == "flex" and tp_ep_group is None:
            logger.info(
                "[DeepSeek-V4] MoE layer=%s flex dispatcher requires TPxEP group.",
                self.layer_idx,
            )
        self.moe_token_dispatcher_type = requested_dispatcher_type

        local_expert_indices = list(range(self.local_expert_start, self.local_expert_end))
        try:
            dispatcher = build_module(
                dispatcher_spec,
                num_local_experts=self.local_num_routed_experts,
                local_expert_indices=local_expert_indices,
                config=self.config,
                pg_collection=self.pg_collection,
            )
            logger.info(
                "[DeepSeek-V4] MoE layer=%s dispatcher active via %s.",
                self.layer_idx,
                type(dispatcher).__name__,
            )
            return dispatcher
        except Exception as exc:
            raise RuntimeError(f"DeepSeek-V4 MoE layer={self.layer_idx} dispatcher build failed: {exc}")

    def _route(
        self,
        hidden: torch.Tensor,
        token_ids: Optional[torch.Tensor],
    ):
        """Return ``(probs, routing_map)`` for the current router.

        Hash-routed layers feed both ``hidden`` (for the learned routing
        weights) AND ``token_ids`` (for the static expert ids from
        ``tid2eid``); learned layers only consume ``hidden``.
        """
        if self.use_hash_router:
            assert self.router is not None
            if token_ids is None:
                raise ValueError(
                    f"layer {self.layer_idx} uses DeepseekV4HashRouter; "
                    "token_ids is required (shape [B, S])."
                )
            return self.router(hidden, token_ids)
        assert self.learned_router is not None
        return self.learned_router(hidden)

    # ------------------------------------------------------------------

    def _build_grouped_experts(self):
        grouped_experts_spec: Optional[Union[ModuleSpec, type]] = self.submodules.grouped_experts
        assert (
            grouped_experts_spec is not None
        ), "DeepSeek-V4 MoE requires grouped experts spec in submodules."
        if self.local_num_routed_experts <= 0:
            raise RuntimeError(
                f"DeepSeek-V4 MoE layer={self.layer_idx} has no local experts for grouped backend."
            )

        if self.config is None or self.pg_collection is None:
            raise RuntimeError("DeepSeek-V4 MoE requires config and pg_collection to build grouped experts.")

        try:
            module = build_module(
                grouped_experts_spec,
                num_local_experts=self.local_num_routed_experts,
                config=self.config,
                pg_collection=self.pg_collection,
            )
            if not self._grouped_backend_supports_clamped_swiglu(module):
                raise RuntimeError(
                    "DeepSeek-V4 MoE grouped backend "
                    f"{type(module).__name__} does not declare clamped-SwiGLU support. "
                    "Set `v4_grouped_experts_support_clamped_swiglu=True` only "
                    "after backend parity is validated."
                )
            logger.info(
                "[DeepSeek-V4] MoE layer=%s provider grouped-gemm active via %s.",
                self.layer_idx,
                type(module).__name__,
            )
            return module
        except Exception as exc:
            raise RuntimeError(f"DeepSeek-V4 MoE layer={self.layer_idx} grouped experts build failed: {exc}")

    def _grouped_backend_supports_clamped_swiglu(self, module: nn.Module) -> bool:
        if self.clamp_alpha <= 0:
            return True
        if bool(getattr(module, "supports_clamped_swiglu", False)):
            return True
        if self.config is not None and bool(self.config.v4_grouped_experts_support_clamped_swiglu):
            return True
        return False

    def _dispatcher_expert_forward(
        self,
        permuted_hidden: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> torch.Tensor:
        assert self.grouped_experts is not None
        try:
            grouped_out = self.grouped_experts(
                permuted_hidden,
                tokens_per_expert,
                permuted_probs,
                routing_map=routing_map,
            )
        except TypeError:
            grouped_out = self.grouped_experts(
                permuted_hidden,
                tokens_per_expert,
                permuted_probs,
            )
        if isinstance(grouped_out, tuple):
            return grouped_out[0]
        return grouped_out

    def _dispatcher_forward(
        self,
        hidden: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> torch.Tensor:
        assert self.token_dispatcher is not None
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(hidden, routing_map, probs)
        hidden_states, probs = self.token_dispatcher.token_dispatch(hidden_states, probs)
        expert_input, tokens_per_expert, permuted_probs = self.token_dispatcher.dispatch_postprocess(
            hidden_states, probs
        )

        expert_output = self._dispatcher_expert_forward(
            expert_input,
            tokens_per_expert,
            permuted_probs,
            routing_map,
        )

        combined = self.token_dispatcher.combine_preprocess(expert_output)
        combined = self.token_dispatcher.token_combine(combined)
        return self.token_dispatcher.combine_postprocess(combined)

    def _local_experts_forward(
        self,
        hidden: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert dispatch loop matching the HF reference math.

        Drives :attr:`local_experts` directly (no token dispatcher); used
        on the CPU path when ``pg_collection is None``. The math mirrors
        ``DeepSeek-V4-Flash/inference/model.py:MoE.forward`` exactly:

            for i in local_experts:
                idx = where(routing_map[:, i])
                out[idx] += probs[idx, i] * expert_i(hidden[idx])

        Args:
            hidden: ``[N, D]`` flattened input.
            probs: ``[N, num_experts]`` sparse routing weights (already
                renormalized + scaled by the router).
            routing_map: ``[N, num_experts]`` bool mask, True at
                ``(n, e)`` iff token ``n`` is routed to expert ``e``.

        Returns:
            ``[N, D]`` routed-expert contribution (no shared expert).
        """
        assert self.local_experts is not None
        out = torch.zeros_like(hidden, dtype=hidden.dtype)
        for local_i, global_i in enumerate(self.local_expert_indices):
            mask = routing_map[:, global_i]  # [N]
            if not bool(mask.any()):
                continue
            idx = mask.nonzero(as_tuple=True)[0]  # [n_i]
            weight = probs[idx, global_i].unsqueeze(-1).to(hidden.dtype)  # [n_i, 1]
            expert = self.local_experts[local_i]
            out_idx = expert(hidden[idx])
            out[idx] = out[idx] + weight * out_idx
        return out

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run V4 MoE FFN.

        Args:
            hidden: ``[B, S, D]`` input.
            token_ids: ``[B, S]`` integer token ids, required only when
                ``layer_idx < num_hash_layers``.

        Returns:
            ``[B, S, D]`` output. Sum of routed-expert and shared-expert
            contributions.
        """
        probs, routing_map = self._route(hidden, token_ids)  # [N, E], bool

        if self.local_experts is not None:
            # CPU local-experts path. Reshape to flat then back; the
            # router already returned [N, E] sparse outputs.
            shape = hidden.shape
            flat_hidden = hidden.reshape(-1, self.hidden_size)
            flat_out = self._local_experts_forward(flat_hidden, probs, routing_map)
            if self.shared_expert is not None:
                flat_out = flat_out + self.shared_expert(flat_hidden)
            return flat_out.view(*shape)

        # Production path: Megatron dispatcher + grouped experts.
        out = self._dispatcher_forward(hidden, probs, routing_map)
        if self.shared_expert is not None:
            out = out + self.shared_expert(hidden)
        return out


__all__ = ["DeepseekV4MoE", "DeepseekV4MoESubmodules"]
