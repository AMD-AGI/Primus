###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ROCMoE-backed MoE layer for the Megatron backend.

Drop-in replacement for ``megatron.core.transformer.moe.moe_layer.MoELayer``.
The Megatron router (sigmoid + group-topk + expert bias) and the shared expert
are kept as-is; only the dispatch + grouped expert MLP + combine are replaced by
the ROCMoE EP engine (``rocmoe.ROCMoE``), which does its own cross-rank PUSH
dispatch/combine over the expert-parallel group via XGMI/IPC.

Scope (bring-up): EP only, TP=1, PP=1 (no 1F1B), bf16.  The ROCMoE engine reuses
its forward state in backward, so the no-pipeline schedule (fwd->bwd per
microbatch) is required, and per-layer transport buffers must not be shared
(run with ROCMOE_SHARE_BUFFERA=0).  EP must be intra-node (IPC peer access).

Expert weight layout: ROCMoE wants ``w_fc1 [E, 2F, H]`` ([gate|up]) and
``w_fc2 [E, H, F]``, which equals Megatron's per-expert logical ``[out, in]``
weights, so we initialize them directly with the configured init methods (no
checkpoint interop with Megatron's flat GroupedMLP storage).
"""

import os

import torch

from megatron.core import utils as mcore_utils
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import build_module
from megatron.training.global_vars import get_args

from primus.modules.module_utils import log_rank_0

# Env-gated forward profiling (ROCMOE_PROFILE=1): per-segment wall-time probe.
_PROFILE = os.environ.get("ROCMOE_PROFILE", "0") == "1"
_PROFILE_ACC = {"router": 0.0, "convert": 0.0, "engine": 0.0, "shared": 0.0, "n": 0}


class ROCMoELayer(MegatronModule):
    """Mixture-of-experts layer whose dispatch+experts+combine run on ROCMoE."""

    def __init__(
        self,
        config,
        submodules=None,
        layer_number=None,
        pg_collection=None,
        is_mtp_layer=False,
    ):
        super().__init__(config=config)
        self.config = config
        self.submodules = submodules
        self.layer_number = layer_number

        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.tp

        ep_size = mcore_utils.get_pg_size(self.ep_group)
        ep_rank = mcore_utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"
        assert (
            config.num_moe_experts % ep_size == 0
        ), f"num_moe_experts={config.num_moe_experts} not divisible by EP={ep_size}"
        self.num_local_experts = config.num_moe_experts // ep_size

        tp_size = self.tp_group.size() if self.tp_group is not None else 1
        assert tp_size == 1, "ROCMoE bring-up supports TP=1 only"
        assert not config.sequence_parallel, "ROCMoE bring-up does not support sequence parallel"

        # Router stays Megatron's (produces probs + boolean routing_map).
        router_builder = TopKRouter
        if submodules is not None and getattr(submodules, "router", None) is not None:
            router_builder = submodules.router
        self.router = router_builder(
            config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer
        )

        # Shared expert stays Megatron's (added to the routed output).
        self.use_shared_expert = config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = config.moe_shared_expert_overlap
        assert not self.shared_expert_overlap, "ROCMoE does not support shared-expert overlap"
        if self.use_shared_expert:
            self.shared_experts = build_module(
                submodules.shared_experts,
                config=config,
                pg_collection=pg_collection,
                gate=config.moe_shared_expert_gate,
            )
        else:
            self.shared_experts = None

        # ROCMoE engine: fixed per-rank token count = seq_length * micro_batch_size
        # (TP=1, no sequence parallel).
        args = get_args()
        self.seq_length = args.seq_length
        self.micro_batch_size = args.micro_batch_size
        b_per_rank = self.seq_length * self.micro_batch_size
        self.b_per_rank = b_per_rank

        H = config.hidden_size
        F = config.moe_ffn_hidden_size
        K = config.moe_router_topk
        over_provision = float(os.environ.get("ROCMOE_OVER_PROVISION", "2.0"))
        device = torch.device("cuda", torch.cuda.current_device())

        from rocmoe.layer import ROCMoE

        # ``experts`` name => parameters get a ".experts." qualified name, which
        # Megatron's grad machinery uses to recognize expert-parallel params.
        self.experts = ROCMoE(
            hidden=H,
            intermediate=F,
            num_local_experts=self.num_local_experts,
            top_k=K,
            b_per_rank=b_per_rank,
            num_ranks=ep_size,
            rank=ep_rank,
            over_provision=over_provision,
            process_group=self.ep_group,
            device=device,
            dtype=torch.bfloat16,
        )

        self._init_expert_weights()
        self._tag_expert_params()

        log_rank_0(
            "[ROCMoELayer] built layer "
            f"{layer_number}: H={H} F={F} E_local={self.num_local_experts} K={K} "
            f"EP={ep_size} b_per_rank={b_per_rank} over_provision={over_provision} "
            f"shared_expert={self.use_shared_expert}"
        )

    def _init_expert_weights(self):
        """Init ROCMoE-layout weights with Megatron's configured init methods.

        ROCMoE ``w_fc1 [E, 2F, H]`` / ``w_fc2 [E, H, F]`` == per-expert ``[out, in]``,
        so init each expert slice directly (correct fan-in) in fp32 then cast bf16.
        """
        E = self.num_local_experts
        F = self.config.moe_ffn_hidden_size
        H = self.config.hidden_size
        init_method = self.config.init_method
        out_init = self.config.output_layer_init_method
        with torch.no_grad():
            w1 = torch.empty(E, 2 * F, H, device=self.experts.w_fc1.device, dtype=torch.float32)
            w2 = torch.empty(E, H, F, device=self.experts.w_fc2.device, dtype=torch.float32)
            for e in range(E):
                init_method(w1[e])
                out_init(w2[e])
            self.experts.w_fc1.copy_(w1.to(torch.bfloat16))
            self.experts.w_fc2.copy_(w2.to(torch.bfloat16))

    def _tag_expert_params(self):
        """Mark w_fc1/w_fc2 as expert-parallel params for Megatron grad sync."""
        for p in (self.experts.w_fc1, self.experts.w_fc2):
            if not hasattr(p, "tensor_model_parallel"):
                set_tensor_model_parallel_attributes(p, is_parallel=False, dim=-1, stride=1)
            # Expert params are NOT all-reduced across the full DP group; they are
            # reduced only over the expert-data-parallel group (handled separately).
            setattr(p, "allreduce", False)
            setattr(p, "sequence_parallel", False)

    def set_layer_number(self, layer_number):
        """Set the layer number for the MoE layer (called from transformer_layer)."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)

    def forward(self, hidden_states, *args, **kwargs):
        """hidden_states [S, B, H] -> (output [S, B, H], mlp_bias=None)."""
        if _PROFILE:
            return self._forward_profiled(hidden_states)

        S, Bsz, H = hidden_states.shape
        K = self.config.moe_router_topk

        # Megatron router: probs/routing_map are [num_tokens, num_experts].
        probs, routing_map = self.router(hidden_states)

        # Selected global expert ids per token + their (differentiable) weights.
        topk_idx = routing_map.to(torch.int32).topk(K, dim=1).indices
        topk_weight = torch.gather(probs.to(torch.float32), 1, topk_idx.to(torch.int64))

        x = hidden_states.reshape(-1, H).to(torch.bfloat16).contiguous()
        y = self.experts(x, topk_idx.contiguous(), topk_weight.contiguous())
        output = y.view(S, Bsz, H)

        if self.use_shared_expert:
            output = output + self.shared_experts(hidden_states)

        return output, None

    def _forward_profiled(self, hidden_states):
        """Same as forward() but wall-times each segment (ROCMOE_PROFILE=1).

        Each segment is bounded by torch.cuda.synchronize() so the host clock
        captures the true wall cost (incl. ROCMoE's internal cross-rank host
        barriers + device sync).  Perturbs timing (adds syncs) -- profiling only.
        """
        import time

        S, Bsz, H = hidden_states.shape
        K = self.config.moe_router_topk
        acc = _PROFILE_ACC

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs, routing_map = self.router(hidden_states)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        topk_idx = routing_map.to(torch.int32).topk(K, dim=1).indices
        topk_weight = torch.gather(probs.to(torch.float32), 1, topk_idx.to(torch.int64))
        x = hidden_states.reshape(-1, H).to(torch.bfloat16).contiguous()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        y = self.experts(x, topk_idx.contiguous(), topk_weight.contiguous())
        output = y.view(S, Bsz, H)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if self.use_shared_expert:
            output = output + self.shared_experts(hidden_states)
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        acc["router"] += t1 - t0
        acc["convert"] += t2 - t1
        acc["engine"] += t3 - t2
        acc["shared"] += t4 - t3
        acc["n"] += 1
        if acc["n"] % 200 == 0:
            tot = acc["router"] + acc["convert"] + acc["engine"] + acc["shared"]
            log_rank_0(
                f"[ROCMoE profile] fwd calls={acc['n']} avg ms/call -- "
                f"router={acc['router'] / acc['n'] * 1e3:.2f} "
                f"convert={acc['convert'] / acc['n'] * 1e3:.2f} "
                f"engine={acc['engine'] / acc['n'] * 1e3:.2f} "
                f"shared={acc['shared'] / acc['n'] * 1e3:.2f} | "
                f"engine share={acc['engine'] / tot * 100:.1f}%"
            )
        return output, None
