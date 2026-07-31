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

Load balancing follows the paper: the auxiliary-loss-free expert bias plus a
sequence-wise balance loss, the latter implemented directly on the learned
router (see :mod:`...moe.v4_seq_balance_loss`) and driven by
``moe_router_load_balancing_type: seq_aux_loss`` + ``moe_aux_loss_coeff``.
It is not inherited from the framework's :class:`TopKRouter` for two reasons: the
V4 routers are standalone ``nn.Module``\\ s (the parent registers CUDA buffers in
``__init__`` and is impractical to instantiate on CPU), and the built-in aux-loss
scores helper only handles ``softmax`` / ``sigmoid`` and rejects V4's
``sqrtsoftplus``. z-loss remains unimplemented. The
distributed re-validation phase (P19) will re-introduce that path
behind a TopKRouter subclass once the CUDA-buffer init is gated by a
device check upstream.
"""

from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

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
from primus.backends.megatron.core.transformer.moe.shared_experts import (
    PrimusSharedExpertMLP,
)
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
        self._force_lb_mode = self._resolve_force_load_balancing_mode()
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
        # True once the dispatcher owns the shared expert, which also means the
        # dispatcher adds its output (see ``_enable_shared_expert_overlap``).
        self.shared_expert_overlap = False
        self.mega_moe_experts: Optional[nn.Module] = None

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
        elif self._mega_moe_requested():
            # MegaMoE fuses dispatch/combine into the grouped GEMMs, so it
            # replaces the dispatcher *and* the grouped experts. The V4 routers
            # and the shared expert are untouched.
            self.mega_moe_experts = self._build_mega_moe_experts()
            if self.use_shared_expert:
                assert self.config.moe_shared_expert_intermediate_size is not None
                self.shared_expert = self._build_shared_expert_module(
                    intermediate_size=int(self.config.moe_shared_expert_intermediate_size)
                )
        else:
            self.token_dispatcher = self._build_token_dispatcher()
            self.grouped_experts = self._build_grouped_experts()
            if self.use_shared_expert:
                assert self.config.moe_shared_expert_intermediate_size is not None
                self.shared_expert = self._build_shared_expert_module(
                    intermediate_size=int(self.config.moe_shared_expert_intermediate_size)
                )
                self.shared_expert_overlap = self._enable_shared_expert_overlap()

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
        # The router is built before this runs, so forward the index it needs
        # to attribute its balance loss to the right layer in the tracker.
        if self.learned_router is not None:
            self.learned_router.layer_number = layer_number

    def _mega_moe_requested(self) -> bool:
        """Whether the fused MegaMoE expert path was asked for.

        The upstream patch that swaps Megatron's ``MoELayer`` cannot reach V4 --
        :class:`DeepseekV4MoE` is built directly by ``deepseek_v4_block`` and
        never instantiates ``MoELayer`` -- so V4 drives
        :class:`MegaMoEExperts` itself.

        The turbo switches live on the Megatron ``args`` namespace, not on
        :class:`TransformerConfig` (same as ``args.enable_primus_turbo`` in
        :mod:`primus.backends.megatron.core.transformer.moe.router`), so read
        them from there. ``get_args`` raises before Megatron is initialized,
        which is the CPU unit-test path -- treat that as "not requested".
        """
        try:
            from megatron.training import get_args

            args = get_args()
        except Exception:
            return False
        return bool(getattr(args, "enable_primus_turbo", False)) and bool(
            getattr(args, "use_turbo_mega_moe", False)
        )

    @staticmethod
    def _resolve_force_load_balancing_mode() -> Optional[str]:
        """Return the requested force-load-balancing mode, or ``None`` if off.

        Read from the Megatron ``args`` namespace like
        :meth:`_mega_moe_requested`; ``get_args`` raises before Megatron is
        initialized (the CPU unit-test path), which counts as "off".
        """
        try:
            from megatron.training import get_args

            args = get_args()
        except Exception:
            return None
        if not bool(getattr(args, "moe_router_force_load_balancing", False)):
            return None
        mode = str(getattr(args, "moe_router_force_load_balancing_type", "uniform"))
        if mode not in ("even", "uniform"):
            raise ValueError(
                f"moe_router_force_load_balancing_type must be 'even' or 'uniform', got {mode!r}"
            )
        return mode

    def _apply_force_load_balancing(
        self, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rewrite the routed expert set so every expert gets a similar share.

        Benchmark aid, mirroring ``moe_router_force_load_balancing`` on Megatron's
        ``TopKRouter``. V4's routers are plain ``nn.Module``s rather than
        ``TopKRouter`` subclasses, so neither upstream's ``apply_random_logits``
        (``TopKRouter.forward``) nor ``PrimusTopKRouter._force_even_routing``
        reaches this path -- this is the V4 equivalent of both, and it covers the
        hash-routed layers too (force load balancing exists to discard the
        router's decision, hash tables included).

        Both modes carry the real per-token top-k probability magnitudes over to
        the new slots, so ``probs`` and ``routing_map`` stay mutually consistent
        and the combine weights keep their true scale:

        * ``even``    -- round-robin ``(token_idx * topk + k) % num_experts``.
          Per-expert counts are exactly equal and step-invariant, so
          grouped-GEMM shapes never change. This is the same cycle
          ``PrimusTurboDeepEPTokenDispatcher.dispatch_preprocess`` derives on its
          own, so when Turbo DeepEP re-applies it the two agree instead of
          dispatching to one expert set while weighting by another.
        * ``uniform`` -- a fresh random k-subset per token every step, i.e. the
          selection ``apply_random_logits`` + top-k produces upstream. Balanced
          only statistically, so it keeps the step-to-step shape variation of
          real routing.
        """
        if self._force_lb_mode is None:
            return probs, routing_map

        num_tokens = routing_map.shape[0]
        topk = self.moe_router_topk
        num_experts = self.num_routed_experts
        device = routing_map.device

        if self._force_lb_mode == "even":
            # topk consecutive experts per token -> distinct while topk <= num_experts.
            slot = torch.arange(num_tokens * topk, device=device).view(num_tokens, topk) % num_experts
        else:
            slot = torch.rand(num_tokens, num_experts, device=device).topk(topk, dim=-1).indices

        new_routing_map = torch.zeros_like(routing_map)
        new_routing_map.scatter_(1, slot, torch.ones_like(slot, dtype=routing_map.dtype))

        # probs is non-zero only on the real top-k, so topk() extracts exactly them.
        topk_vals, _ = torch.topk(probs, topk, dim=1)
        new_probs = torch.zeros_like(probs)
        new_probs.scatter_(1, slot, topk_vals.to(probs.dtype))

        return new_probs, new_routing_map

    def _build_mega_moe_experts(self) -> nn.Module:
        """Build the fused MegaMoE expert path, or explain why it cannot run.

        Every unmet requirement raises rather than falling back: a silent
        fallback leaves the run with neither MegaMoE nor the Turbo DeepEP
        dispatcher that ``use_turbo_mega_moe`` disables, which shows up only as
        an unexplained slowdown.
        """
        from primus.backends.megatron.core.extensions.mega_moe import MegaMoEExperts

        reasons = []
        if self.config.tensor_model_parallel_size != 1:
            reasons.append(f"MegaMoE is EP-only (TP==1), got TP={self.config.tensor_model_parallel_size}")
        if self.config.params_dtype != torch.bfloat16:
            reasons.append(f"MegaMoE only supports bf16, got params_dtype={self.config.params_dtype}")
        if self.ep_group is None or self.ep_size <= 1:
            reasons.append(f"MegaMoE needs an expert-parallel group, got EP={self.ep_size}")
        if self.num_routed_experts % self.ep_size != 0:
            reasons.append(
                f"MegaMoE needs num_experts divisible by EP size, got "
                f"{self.num_routed_experts} % {self.ep_size} != 0"
            )
        if reasons:
            raise ValueError(
                "[DeepSeek-V4] use_turbo_mega_moe=True but the fused MegaMoE path cannot run here: "
                + "; ".join(reasons)
            )

        logger.warning(
            "[DeepSeek-V4] MegaMoE expert path enabled: the fused kernel hardcodes unclamped "
            "SwiGLU, so swiglu_limit=%s is NOT applied to routed experts.",
            self.clamp_alpha,
        )
        experts = MegaMoEExperts(
            self.config,
            self.local_num_routed_experts,
            self.hidden_size,
            self.moe_intermediate_size,
            self.ep_group,
        )
        if self.config.perform_initialization:
            experts.reset_parameters(self.ep_rank)
        logger.info(
            "[DeepSeek-V4] MoE expert path resolved to MegaMoEExperts "
            "(fused dispatch/combine; token dispatcher and grouped experts not built)."
        )
        return experts

    def _mega_moe_forward(self, hidden: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Run the fused MegaMoE expert path.

        ``probs`` is the dense ``[N, E]`` router output while MegaMoE wants the
        compact ``(topk_idx, topk_weights)`` pair. ``probs`` is zero outside the
        selected experts, so ``topk`` recovers exactly the routed set -- the
        same conversion :class:`PrimusTurboMegaMoELayer` performs.
        """
        assert self.mega_moe_experts is not None
        in_shape = hidden.shape
        topk_weights, topk_idx = probs.topk(self.moe_router_topk, dim=-1)
        x = hidden.reshape(-1, self.hidden_size).to(torch.bfloat16)
        y = self.mega_moe_experts(x, topk_idx, topk_weights.to(torch.float32))
        return y.reshape(in_shape).to(hidden.dtype)

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
        # Plan-3 P23: PrimusTurboDeepEPTokenDispatcher is a "flex"
        # variant — V4 spec build chose it explicitly when the user
        # opted into Turbo DeepEP.  We recognise it by class name so
        # this resolver does not require ``primus_turbo`` to be
        # importable on hosts that never opt in (CPU unit tests,
        # Megatron-only consumers).
        module_name = getattr(module, "__name__", "")
        if module_name == "PrimusTurboDeepEPTokenDispatcher":
            return "flex"
        logger.warning(
            "[DeepSeek-V4] unsupported dispatcher module=%s; fallback type to alltoall.",
            module_name or str(module),
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
            seq_balance_loss_coeff=self._seq_balance_loss_coeff(),
            seq_balance_reduce_group=self._seq_balance_reduce_group(),
            layer_number=self.layer_number,
            num_layers=self._total_layers_for_loss_logging(),
        )

    def _seq_balance_loss_coeff(self) -> float:
        """Coefficient of the sequence-wise balance loss, 0 when not selected.

        V4 balances with the auxiliary-loss-free expert bias plus this
        sequence-wise term (paper 2.1), which is what
        ``moe_router_load_balancing_type: seq_aux_loss`` in the model yaml asks
        for. Any other balancing type leaves the loss off -- the framework's
        built-in aux-loss variants are not reachable from this router.
        """
        config = self.config
        if str(getattr(config, "moe_router_load_balancing_type", "")) != "seq_aux_loss":
            return 0.0
        return float(getattr(config, "moe_aux_loss_coeff", 0.0) or 0.0)

    def _seq_balance_reduce_group(self):
        """Group the sequence axis is sharded over, or ``None`` if it is not.

        The loss is per-sequence, so with sequence parallelism (TP) or context
        parallelism a rank only sees part of each sequence and the expert counts
        have to be summed across the shards.
        """
        tp_size = int(getattr(self.config, "tensor_model_parallel_size", 1) or 1)
        cp_size = int(getattr(self.config, "context_parallel_size", 1) or 1)
        sequence_parallel = bool(getattr(self.config, "sequence_parallel", False))
        if cp_size == 1 and not (sequence_parallel and tp_size > 1):
            return None
        group = getattr(self.pg_collection, "tp_cp", None)
        if group is not None:
            return group
        from megatron.core import parallel_state

        return parallel_state.get_tensor_and_context_parallel_group()

    def _total_layers_for_loss_logging(self) -> Optional[int]:
        """Tracker width: decoder layers plus MTP layers, matching the framework."""
        num_layers = getattr(self.config, "num_layers", None)
        if not num_layers:
            return None
        return int(num_layers) + int(getattr(self.config, "mtp_num_layers", 0) or 0)

    def _enable_shared_expert_overlap(self) -> bool:
        """Hand the shared expert to the dispatcher so it runs under the A2A.

        ``moe_shared_expert_overlap`` asks for the shared expert's GEMMs to be
        interleaved with the dispatch / combine all-to-alls instead of running
        after them. The token dispatcher already carries the hooks, and they sit
        in exactly the decomposed calls :meth:`_dispatcher_forward` already
        makes: ``dispatch_preprocess`` kicks off the input comm on a side stream,
        ``dispatch_postprocess`` runs fc1 + activation, and
        ``combine_postprocess`` runs fc2 and **adds the result into the output**.

        That last part is why the return value matters to the caller: once the
        dispatcher owns the shared expert, adding it again in ``forward`` would
        double-count it.

        Returns False (leaving the serial path in place) when the config asks for
        no overlap or the dispatcher does not support it -- the flex dispatcher
        raises ``NotImplementedError`` from ``set_shared_experts``.
        """
        if not bool(getattr(self.config, "moe_shared_expert_overlap", False)):
            return False
        if self.shared_expert is None or self.token_dispatcher is None:
            return False
        if not hasattr(self.shared_expert, "pre_forward_comm"):
            # The CPU ClampedSwiGLUMLP has no overlap protocol.
            return False
        try:
            self.token_dispatcher.set_shared_experts(self.shared_expert)
        except NotImplementedError:
            logger.warning(
                "[V4-MoE] layer %s: %s does not support shared-expert overlap; "
                "running the shared expert serially.",
                self.layer_idx,
                type(self.token_dispatcher).__name__,
            )
            return False
        return True

    def _build_shared_expert_module(self, *, intermediate_size: int) -> nn.Module:
        shared_expert_spec = self.submodules.shared_expert
        assert isinstance(
            shared_expert_spec, ModuleSpec
        ), "DeepSeek-V4 MoE requires shared_expert ModuleSpec in submodules."
        shared_expert_module = shared_expert_spec.module
        assert issubclass(
            shared_expert_module, SharedExpertMLP
        ), "DeepSeek-V4 shared_expert must be (a subclass of) SharedExpertMLP."
        if self.config is None or self.pg_collection is None:
            raise RuntimeError("DeepSeek-V4 MoE SharedExpertMLP requires config and pg_collection.")

        # Shared experts run with clamped SwiGLU. PrimusSharedExpertMLP fuses the
        # clamp+SiLU+mul into a single Triton kernel (matching the routed experts),
        # so we no longer need to force the un-fused eager path.
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

        # Build the Primus fused-SwiGLU shared expert while keeping the spec's
        # submodules (linear layers / activation). The state-dict layout is
        # unchanged since PrimusSharedExpertMLP only overrides the activation.
        fused_shared_expert_spec = ModuleSpec(
            module=PrimusSharedExpertMLP,
            submodules=shared_expert_spec.submodules,
            params=shared_expert_spec.params,
        )
        try:
            return build_module(
                fused_shared_expert_spec,
                config=shared_cfg,
                pg_collection=self.pg_collection,
                gate=bool(shared_cfg.moe_shared_expert_gate),
            )
        except Exception as exc:
            raise RuntimeError(
                f"DeepSeek-V4 MoE shared expert build failed with PrimusSharedExpertMLP: {exc}"
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

        The routed set then goes through :meth:`_apply_force_load_balancing`,
        which is a no-op unless ``moe_router_force_load_balancing`` is set.
        """
        if self.use_hash_router:
            assert self.router is not None
            if token_ids is None:
                raise ValueError(
                    f"layer {self.layer_idx} uses DeepseekV4HashRouter; "
                    "token_ids is required (shape [B, S])."
                )
            probs, routing_map = self.router(hidden, token_ids)
        else:
            assert self.learned_router is not None
            probs, routing_map = self.learned_router(hidden)
        return self._apply_force_load_balancing(probs, routing_map)

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

        # Production path: fused MegaMoE experts, or Megatron dispatcher +
        # grouped experts.
        if self.mega_moe_experts is not None:
            out = self._mega_moe_forward(hidden, probs)
        else:
            out = self._dispatcher_forward(hidden, probs, routing_map)
        if self.shared_expert is not None and not self.shared_expert_overlap:
            out = out + self.shared_expert(hidden)
        return out


__all__ = ["DeepseekV4MoE", "DeepseekV4MoESubmodules"]
