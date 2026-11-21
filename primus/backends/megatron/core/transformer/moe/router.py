###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from megatron.core.transformer.moe.moe_utils import apply_router_token_dropping
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args

from primus.backends.megatron.core.extensions.logits_processor import fused_softcap
from primus.core.utils.module_utils import log_rank_0

_REPLAY_ROUTERS: List["PrimusTopKRouter"] = []
_REPLAY_RECORDS: Dict[str, Dict[int, torch.Tensor]] = {}


def _routing_records(path: str) -> Dict[int, torch.Tensor]:
    """This rank's routes in a Megatron router trace (``--moe-routing-trace-path``): decoder layer -> top-k expert
    ids ``[tokens, topk]`` of all its records, in file order. Read once per process."""
    if path not in _REPLAY_RECORDS:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        files = len(
            [f for f in os.listdir(path) if f.startswith("router_trace_rank") and f.endswith(".jsonl")]
        )
        if files != world:
            raise ValueError(
                f"{path} has {files} router_trace_rank<r>.jsonl files for {world} ranks: replay on the world size "
                "and parallel layout the routes were recorded with"
            )
        per_layer: Dict[int, List[torch.Tensor]] = {}
        with open(os.path.join(path, f"router_trace_rank{rank}.jsonl")) as f:
            for line in f:
                record = json.loads(line)
                if record.get("block", "decoder") != "decoder":
                    continue
                routes = torch.tensor(record["top_indices"], dtype=torch.int32)
                per_layer.setdefault(int(record["layer"]), []).append(routes)
        _REPLAY_RECORDS[path] = {layer: torch.cat(routes) for layer, routes in per_layer.items()}
    return _REPLAY_RECORDS[path]


def _replayed_topk(router: "PrimusTopKRouter", num_tokens: int) -> Optional[torch.Tensor]:
    """Benchmark aid (``moe_routing_replay_path``): the next ``num_tokens`` recorded top-k expert ids of this
    router's layer, wrapping around, or None when replay is off.

    The trace of a decoder layer ``n`` (``decoder.layers.n`` in the trace) drives the MoE layer with
    ``layer_number == n + 1``, or the MoE layers listed in ``moe_routing_replay_layers``, in build order."""
    args = get_args()
    path = args.moe_routing_replay_path
    if not path:
        return None
    if getattr(router, "_replay", None) is None:
        order = len(_REPLAY_ROUTERS)
        layers = args.moe_routing_replay_layers
        if layers and order >= len(layers):
            raise ValueError(
                f"moe_routing_replay_layers has {len(layers)} entries; the model has more MoE layers"
            )
        layer = int(layers[order]) if layers else router.layer_number - 1
        records = _routing_records(path)
        if layer not in records:
            raise ValueError(f"no routes of decoder layer {layer} in {path} (layers {sorted(records)})")
        routes = records[layer]
        if routes.shape[1] != router.topk or int(routes.max()) >= router.config.num_moe_experts:
            raise ValueError(
                f"{path} layer {layer}: top-{routes.shape[1]} routes up to expert {int(routes.max())} do not fit "
                f"top-{router.topk} over {router.config.num_moe_experts} experts"
            )
        _REPLAY_ROUTERS.append(router)
        router._replay = routes.to(router.weight.device, torch.int64)
        router._replay_pos = 0
        if router.router_replay is None:
            router.router_replay = RouterReplay()
        log_rank_0(
            f"[moe_routing_replay] MoE layer {router.layer_number} replays decoder layer {layer} of {path} "
            f"({routes.shape[0]} tokens)"
        )
    routes, pos = router._replay, router._replay_pos
    idx = (torch.arange(num_tokens, device=routes.device) + pos) % routes.shape[0]
    router._replay_pos = (pos + num_tokens) % routes.shape[0]
    return routes[idx]


class PrimusTopKRouter(TopKRouter):
    """Balanced route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:
        super().__init__(config=config, *args, **kwargs)

    def fused_router_and_auxiliary_loss(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            import primus_turbo.pytorch as pt
        except ImportError as e:
            raise ImportError("Failed to import 'primus_turbo'. Please make sure it is installed. ") from e

        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        scores_for_aux_loss, probs, routing_map = pt.ops.fused_group_topk_routing_with_aux_score(
            logits,
            self.config.moe_router_topk,
            self.config.moe_router_num_groups,
            self.config.moe_router_group_topk,
            self.config.moe_router_score_function,
            self.config.moe_router_topk_scaling_factor,
        )

        routing_map = routing_map.bool()

        # Apply token dropping to probs and routing_map.
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            # todo: fuse the following logic into the turbo fused_group_topk_routing_with_aux_score OP
            # (the routing_map here differs from the one in the OP output is regarding less of group limit)
            if self.config.moe_router_num_groups is None or self.config.moe_router_num_groups <= 1:
                routing_map_for_aux_loss = routing_map
            else:
                _, top_indices_for_aux_loss = torch.topk(scores_for_aux_loss, k=self.topk, dim=1)
                routing_map_for_aux_loss = (
                    torch.zeros_like(logits).int().scatter(1, top_indices_for_aux_loss, 1).bool()
                )
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
            probs = self._apply_global_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
        # Update expert bias and tokens_per_expert
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)
        return probs, routing_map

    def routing(self, logits: torch.Tensor, **kwargs):
        args = get_args()

        if args.router_logit_softcapping is not None and args.router_logit_softcapping > 0.0:
            # grok2 router logit softcapping
            fused_softcap(logits, args.router_logit_softcapping)

        replay = _replayed_topk(self, logits.view(-1, self.config.num_moe_experts).shape[0])
        if replay is not None:
            # Megatron's routing replay picks these experts and gathers their probabilities from this router's scores
            self.router_replay.target_topk_idx = replay
            self.router_replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            scores, routing_map = super().routing(logits, **kwargs)
        elif args.enable_primus_turbo and args.moe_use_fused_router_with_aux_score:
            scores, routing_map = self.fused_router_and_auxiliary_loss(logits)
        else:
            scores, routing_map = super().routing(logits, **kwargs)

        assert routing_map.dtype == torch.bool, "routing_map should be boolean"

        # NOTE on ``moe_router_force_load_balancing``:
        #
        # The previous implementation overwrote ``routing_map`` here with a
        # deterministic ``(token_idx * topk + k) % num_experts`` cycle while
        # leaving ``scores`` (the sparse top-k routing probabilities from
        # ``topk_routing_with_score_function``) untouched. Because ``scores``
        # is non-zero only on the *real* top-k experts, the deterministic
        # routing_map rarely overlapped with those non-zero positions, so the
        # MoE combine step multiplied almost every expert output by zero. On
        # a pretrained MoE checkpoint (e.g. DeepSeek-V2-Lite SFT) this
        # collapsed the MoE layer output to essentially just the shared
        # expert path, inflated iter-1 loss by several nats compared to
        # Megatron-Bridge, and broke Native-vs-Bridge A/B loss comparison.
        #
        # The correct (and Bridge-parity) way to do force-load-balancing is
        # to replace the *logits* with random values BEFORE routing, so that
        # both ``scores`` and ``routing_map`` are derived from the same
        # random logits and stay mutually consistent. That replacement is
        # already done in the upstream ``TopKRouter.forward`` via
        # ``apply_random_logits(logits)`` (see
        # third_party/Megatron-LM/megatron/core/transformer/moe/router.py),
        # so by the time we get here ``logits`` is already random when
        # ``args.moe_router_force_load_balancing`` is True. There is nothing
        # extra to do.
        # ``moe_router_force_load_balancing_type`` selects the balancing mode
        # (only relevant when ``moe_router_force_load_balancing`` is on):
        #   * "even"    -> deterministic round-robin so per-expert counts are
        #                  constant every step (constant M_total, no autotune/
        #                  recompile churn); handled by ``_force_even_routing``.
        #                  This must run with DeepEP too: its "even" token_indices
        #                  gather from ``scores``, which are zero off the real
        #                  top-k, and MegaMoE re-derives its top-k from them.
        #   * "uniform" -> Megatron-LM original random-logits balancing (already
        #                  applied upstream in ``TopKRouter.forward``; nothing to
        #                  do here).
        force_load_balancing_type = getattr(args, "moe_router_force_load_balancing_type", "uniform")
        if args.moe_router_force_load_balancing and force_load_balancing_type == "even":
            scores, routing_map = self._force_even_routing(scores, routing_map)

        return scores, routing_map

    def _force_even_routing(
        self, scores: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministically assign each token's top-k slots to experts in a
        round-robin cycle ``(token_idx * topk + k) % num_experts`` so per-expert
        token counts are exactly balanced and step-invariant (constant M_total).

        The original top-k probability magnitudes are carried over to the new
        expert positions (per-slot), keeping ``scores`` and ``routing_map``
        mutually consistent so the MoE combine weights stay correct.
        """
        num_tokens = routing_map.shape[0]
        num_experts = self.config.num_moe_experts
        topk = self.topk
        device = routing_map.device

        # slot[t, k] = deterministic destination expert for token t's k-th slot.
        # topk consecutive experts per token -> distinct while topk <= num_experts.
        slot = (
            torch.arange(num_tokens * topk, device=device).view(num_tokens, topk) % num_experts
        )  # [num_tokens, topk], int64

        new_routing_map = torch.zeros_like(routing_map)
        new_routing_map.scatter_(1, slot, torch.ones_like(slot, dtype=routing_map.dtype))

        # Carry the real per-token top-k probability magnitudes to the new slots.
        # scores is non-zero only on the real top-k, so topk() extracts exactly them.
        topk_vals, _ = torch.topk(scores, topk, dim=1)  # [num_tokens, topk]
        new_scores = torch.zeros_like(scores)
        new_scores.scatter_(1, slot, topk_vals.to(scores.dtype))

        return new_scores, new_routing_map


def primus_topk_router_for(router):
    """``PrimusTopKRouter`` if ``router`` is Megatron's ``TopKRouter`` class, else ``router`` unchanged.

    The swap rule of the ``megatron.moe.primus_topk_router`` patch, shared with MegaMoE. Matching the
    class by name also catches a ``TopKRouter`` that a module bound before the patch ran; subclasses
    such as ``InferenceTopKRouter`` and builders such as a ``functools.partial`` are left alone.
    """
    if isinstance(router, type) and router.__name__ == "TopKRouter":
        return PrimusTopKRouter
    return router
