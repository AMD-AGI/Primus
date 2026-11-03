###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling MoE: sigmoid top-k routing over ``n_routed`` experts, a softmax of
log-sigmoid logits over {selected routed, shared} experts scaled by
``route_scale * global_scale``, and shared experts weighted by their own
router gammas (``shared_expert_sink``).

New math: Megatron's MoELayer adds the shared expert output unweighted and its
router has no shared-expert logits, so the routing and the combine are Inkling-owned.

Expert parallelism: each rank owns ``n_routed / EP`` experts. The tokens, the
routing indices and weights are all-gathered over the expert-parallel group,
every rank applies its own experts, and the partial outputs are
reduce-scattered back to their owners (an autograd pair: AG forward / RS
backward and the reverse). Megatron's token dispatchers cannot be used here:
``MoELayer.forward`` refuses tensor parallelism without sequence parallelism, a
layout this port supports, and the shared-expert gammas come from the same
router call. Expert weights carry ``allreduce=False`` so DDP reduces them over
the expert-data-parallel group. Tensor parallelism replicates the MoE (its
inputs are identical on every TP rank).

Precision: the router logits, the top-k choice and the routing weights are
computed in fp32. The routed experts then run in the activation dtype (bf16 in
training), as HF does: each expert output is multiplied by its routing weight cast
to that dtype, and the routed contributions are summed in that dtype (``index_add``,
then the EP reduce-scatter), as HF's ``index_add_`` into a bf16 buffer. The shared
experts' outputs are weighted by their gammas (cast to the activation dtype) before their
down projection and accumulated with one activation-dtype GEMM per shared expert (``addmm``:
fp32 accumulator inside each GEMM, the running sum rounded to the activation dtype after each
expert). HF's ``InklingSharedExperts`` sums the per-expert outputs in fp32 and casts back
once: the same math, with one extra bf16 rounding per shared expert after the first (2 shared
experts in Inkling; identical in fp32, the dtype of the logprob parity check). Under
``--deterministic-mode`` Megatron sets ``torch.use_deterministic_algorithms(True)``,
which makes ``index_add`` take PyTorch's deterministic path (as Megatron's own
``moe_utils`` unpermute relies on). ``expert_bias`` (HF ``gate.bias``) is an fp32 buffer kept in
fp32 by ``TopKRouter._maintain_float32_expert_bias``; with
``moe_router_enable_expert_bias`` Megatron's ``finalize_model_grads`` updates it
from ``local_tokens_per_expert`` (aux-loss-free load balancing). Parameters are
stored in the training dtype (as Megatron's own router weight) under
``Float16Module``; the optimizer keeps their fp32 main copies.
"""

import functools
import importlib.util
import logging
import sys

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

from primus.backends.megatron.core.transformer.inkling.inkling_layers import (
    init_seed,
    seeded_normal,
)

logger = logging.getLogger(__name__)


def _swiglu(gate, up):
    return F.silu(gate) * up


def _count_ids(ids, n):
    """Occurrences of each id in ``range(n)`` as int64, without a host sync (``torch.bincount``
    on a GPU reads the ids' min / max on the host to size its bins)."""
    flat = ids.reshape(-1)
    return torch.zeros(n, dtype=torch.int64, device=flat.device).scatter_add_(
        0, flat, torch.ones_like(flat, dtype=torch.int64)
    )


@functools.lru_cache(maxsize=None)
def _warn_moe_act_ignored():
    logger.warning(
        "InklingMoE: recompute_modules has 'moe_act', but the routed experts do not run on the "
        "Primus-Turbo grouped GEMM (enable_primus_turbo and use_turbo_grouped_gemm are not both on); the per-expert "
        "fallback does not recompute the SwiGLU, so 'moe_act' is ignored"
    )


class _KeepInGraph(torch.autograd.Function):
    """Identity on ``x`` that ties ``params`` into the graph with zero gradients.

    Used only when none of this rank's experts received a token (so the expert
    weights and the gathered inputs are otherwise absent from the graph): every rank then
    still runs the same gradient hooks and EP collectives, and unlike ``x + 0 * p.sum()``
    a NaN / inf weight does not leak into the output.
    """

    @staticmethod
    def forward(ctx, x, *params):
        ctx.shapes = [(p.shape, p.dtype, p.device) for p in params]
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return (grad, *[torch.zeros(s, dtype=d, device=dev) for s, d, dev in ctx.shapes])


def _turbo_grouped_gemm():
    """``primus_turbo.pytorch.ops.grouped_gemm``, when it is requested (see
    ``_turbo_grouped_gemm_requested``).

    Raises ``ImportError`` (with the fix) when Primus-Turbo is not installed or its grouped GEMM
    does not import: the expert kernel path never depends silently on the container."""
    if importlib.util.find_spec("primus_turbo") is None:
        reason = "Primus-Turbo (primus_turbo) is not installed"
    else:
        try:
            from primus_turbo.pytorch.ops import grouped_gemm

            return grouped_gemm
        except (ImportError, RuntimeError, AttributeError) as e:
            reason = f"Primus-Turbo's grouped GEMM does not import ({type(e).__name__}: {e})"
    raise ImportError(
        f"InklingMoE: enable_primus_turbo and use_turbo_grouped_gemm request the Primus-Turbo "
        f"grouped GEMM for the routed experts, but {reason}. Set use_turbo_grouped_gemm: false to "
        f"run them as a per-expert GEMM loop (same math, slower)."
    )


class InklingTopKRouter(TopKRouter):
    """Sigmoid top-k router with shared-expert logits and a learned ``global_scale``.

    Built by Megatron's :class:`TopKRouter` constructor (process groups, top-k and
    score-function state). ``reset_parameters`` (called from ``Router.__init__``)
    replaces the gate with Inkling's: it also scores the shared experts
    (``[n_routed + n_shared, H]``, layout-independent seeded init), and the weights
    are a softmax of log-sigmoid logits, which Megatron's score functions do not
    cover. ``expert_bias`` (always present: it is a checkpoint value) and
    ``local_tokens_per_expert`` follow Megatron's aux-loss-free load-balancing
    contract.
    """

    def __init__(self, config, pg_collection=None, is_mtp_layer=False, seed=0):
        self._inkling_seed = seed  # read by reset_parameters() inside Router.__init__
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        self.n_routed = config.num_moe_experts
        self.route_scale = config.inkling_route_scale
        self.global_scale = torch.nn.Parameter(torch.ones(1))
        for name in ("expert_bias", "local_tokens_per_expert"):
            if name in self._buffers:
                del self._buffers[name]
            elif hasattr(self, name):
                delattr(self, name)
        self.register_buffer("expert_bias", torch.zeros(self.n_routed, dtype=torch.float32))
        self.register_buffer(
            "local_tokens_per_expert", torch.zeros(self.n_routed, dtype=torch.float32), persistent=False
        )

    def reset_parameters(self):
        """The gate over routed + shared experts with the seeded Inkling init."""
        cfg = self.config
        shape = (cfg.num_moe_experts + cfg.inkling_num_shared_experts, cfg.hidden_size)
        weight = (
            seeded_normal(shape, cfg.init_method_std, self._inkling_seed)
            if cfg.perform_initialization
            else torch.empty(shape)
        )
        self.weight = torch.nn.Parameter(weight.to(cfg.params_dtype))
        setattr(self.weight, "sequence_parallel", cfg.sequence_parallel)
        self.bias = None

    def forward(self, flat, padding_mask=None):  # [N, H]
        _refuse_padding_mask(padding_mask)
        self._maintain_float32_expert_bias()
        logits = F.linear(flat.float(), self.weight.float())
        routed = logits[:, : self.n_routed]
        shared = logits[:, self.n_routed :]
        # as Megatron's TopKRouter._apply_expert_bias: count only grad-enabled forwards, so a
        # recompute (its first, no-grad forward) or a no-grad evaluation counts nothing extra
        count = self.training and self.enable_expert_bias and torch.is_grad_enabled()
        with torch.no_grad():
            choice = routed.sigmoid() + self.expert_bias
            idx = torch.topk(choice, self.topk, dim=-1, sorted=False)[1]
            if count:
                self.local_tokens_per_expert += _count_ids(idx, self.n_routed).float()
        sel = torch.cat([routed.gather(-1, idx), shared], dim=-1)
        lp = F.logsigmoid(sel)
        w = torch.exp(lp - torch.logsumexp(lp, dim=-1, keepdim=True))
        w = w * self.route_scale * self.global_scale.float()
        return w[:, : self.topk], idx, w[:, self.topk :]


def _refuse_padding_mask(padding_mask):
    """Megatron's MoE ``padding_mask`` (tokens left out of routing) is not implemented: refuse
    it instead of silently routing the padded tokens. ``None`` (every caller in Primus's
    pretrain path) is accepted."""
    if padding_mask is not None:
        raise NotImplementedError(
            "Inkling: padding_mask (MoE routing that skips padded tokens) is not supported; "
            "pass padding_mask=None (HF Inkling routes every token)"
        )


def _check_moe_config(config):
    """Refuse the MoE knobs this layer does not implement (instead of ignoring them)."""
    # setting -> (its value, the values this layer supports)
    allowed = {
        "moe_router_load_balancing_type": (config.moe_router_load_balancing_type, ("none",)),
        "moe_aux_loss_coeff": (config.moe_aux_loss_coeff, (0, None)),
        "moe_z_loss_coeff": (config.moe_z_loss_coeff, (0, None)),
        "moe_input_jitter_eps": (config.moe_input_jitter_eps, (None,)),
        "moe_expert_capacity_factor": (config.moe_expert_capacity_factor, (None,)),
        "moe_shared_expert_intermediate_size": (config.moe_shared_expert_intermediate_size, (None,)),
        "moe_router_num_groups": (config.moe_router_num_groups, (None,)),
        "moe_router_force_load_balancing": (
            getattr(config, "moe_router_force_load_balancing", False),
            (False,),
        ),
        "moe_latent_size": (getattr(config, "moe_latent_size", None), (None,)),
        # InklingTopKRouter always scores with sigmoid and Inkling's own weight normalization
        "moe_router_score_function": (config.moe_router_score_function, ("sigmoid",)),
        "moe_router_pre_softmax": (config.moe_router_pre_softmax, (False,)),
        "moe_router_topk_scaling_factor": (config.moe_router_topk_scaling_factor, (None,)),
    }
    if isinstance(config.moe_aux_loss_coeff, (list, tuple)):
        allowed["moe_aux_loss_coeff"] = (sum(config.moe_aux_loss_coeff), (0,))
    bad = {k: v for k, (v, ok) in allowed.items() if v not in ok}
    if bad:
        raise ValueError(f"InklingMoE does not implement these MoE settings: {bad}")


def _turbo_grouped_gemm_requested():
    """``enable_primus_turbo and use_turbo_grouped_gemm`` on the Megatron args, the switches the
    other Primus MoE layers use. ``moe_grouped_gemm`` (Megatron's grouped-expert modules, which
    InklingMoE does not build) does not select the kernel; Megatron requires it for ``moe_act``
    recompute only. These live on ``args``, not on the TransformerConfig;
    Megatron's args exist only after ``megatron.training`` is imported and initialized: when the
    module is not loaded (CPU unit tests, spawned test workers) this is "not requested" without
    importing it, and before initialization ``get_args`` asserts ("not requested" too). No other
    error is caught: a broken Megatron install fails here, not silently in the per-expert loop."""
    training = sys.modules.get("megatron.training")
    if training is None:
        return False
    try:
        args = training.get_args()
    except AssertionError:  # "args is not initialized."
        return False
    return bool(getattr(args, "enable_primus_turbo", False)) and bool(
        getattr(args, "use_turbo_grouped_gemm", False)
    )


class InklingMoE(MoELayer):
    """Inkling MoE on Megatron's :class:`MoELayer`.

    ``MoELayer.__init__`` sets up the process groups, the expert-parallel split
    (``num_local_experts`` / ``local_expert_indices``), the recompute flags and the
    router (an :class:`InklingTopKRouter`, kept under the checkpoint name ``gate``).
    The experts are Inkling's stacked ``gate_up_proj`` / ``down_proj`` and the
    shared experts are weighted by their router gammas, so the forward is
    Inkling-owned (see the module docstring)."""

    def __init__(self, config, layer_number, inkling_layer_idx, pg_collection=None):
        _check_moe_config(config)
        base = config.inkling_init_seed

        def seed(slot, index=0):
            return init_seed(base, inkling_layer_idx, slot, index)

        submodules = MoESubmodules(
            experts=IdentityOp,
            shared_experts=None,
            router=functools.partial(InklingTopKRouter, seed=seed("router")),
        )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=False,
        )
        # The checkpoint names the router ``gate``: register it once under that name and keep
        # ``self.router`` (used by MoELayer.route / set_layer_number) as a plain alias.
        gate = self._modules.pop("router")
        self.gate = gate
        object.__setattr__(self, "router", gate)
        self.experts = None  # Inkling's routed experts are this module's own stacked parameters
        self.token_dispatcher = None  # not used: see the module docstring
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        self.expt_dp_group = getattr(pg_collection, "expt_dp", None)
        self.expt_tp_group = getattr(pg_collection, "expt_tp", None)
        selective = config.recompute_granularity == "selective"
        self.recompute_act = selective and "moe_act" in (config.recompute_modules or [])
        self.grouped_gemm = None
        if _turbo_grouped_gemm_requested():
            self.grouped_gemm = _turbo_grouped_gemm()  # raises when Primus-Turbo is unavailable
        if self.recompute_act and self.grouped_gemm is None:
            _warn_moe_act_ignored()
        H, I = config.hidden_size, config.moe_ffn_hidden_size
        self.ep = self.ep_group.size() if self.ep_group is not None else 1
        self.e0 = self.local_expert_indices[0]
        std = config.init_method_std
        init = config.perform_initialization
        gu = torch.empty(self.num_local_experts, 2 * I, H)
        dn = torch.empty(self.num_local_experts, H, I)
        if init:
            for j in range(self.num_local_experts):
                g = self.e0 + j  # global expert id: the same values under every EP layout
                gu[j] = seeded_normal((2 * I, H), std, seed("gate_up_proj", g))
                dn[j] = seeded_normal((H, I), std, seed("down_proj", g))
        self.gate_up_proj = torch.nn.Parameter(gu)
        self.down_proj = torch.nn.Parameter(dn)
        for p in (self.gate_up_proj, self.down_proj):
            p.allreduce = False  # expert parameter: reduced over the expert-data-parallel group
        S = config.inkling_num_shared_experts

        # shared experts: gate and up fused as [S, 2I, H] (one batched GEMM, like gate_up_proj);
        # the two halves keep their own init seeds
        if init:
            sgu = torch.cat(
                [seeded_normal((S, I, H), std, seed(s)) for s in ("shared_gate_proj", "shared_up_proj")], 1
            )
            sdn = seeded_normal((S, H, I), std, seed("shared_down_proj"))
        else:
            sgu, sdn = torch.empty(S, 2 * I, H), torch.empty(S, H, I)
        self.shared_gate_up_proj = torch.nn.Parameter(sgu)
        self.shared_down_proj = torch.nn.Parameter(sdn)

    def _experts(self, x, idx, w):
        """Sum over this rank's experts of w * expert(x) for every (token, slot) routed to them.

        The per-expert segment sizes are read on the host once per call (one sync per
        layer). With the Primus-Turbo grouped GEMM (``self.grouped_gemm``) the local experts run as two Primus-Turbo grouped
        GEMMs; otherwise as one small dense GEMM pair per expert that received tokens.
        """
        local = idx - self.e0
        hit = (local >= 0) & (local < self.num_local_experts)
        e_all = torch.where(hit, local, self.num_local_experts).reshape(-1)  # last bucket = not on this rank
        order = torch.argsort(e_all, stable=True)
        counts_t = _count_ids(e_all, self.num_local_experts + 1)  # no sync (e_all is in range)
        counts = counts_t.tolist()  # the single host sync
        n_mine = sum(counts[: self.num_local_experts])
        if n_mine == 0:
            # No local expert got a token. Tie the expert weights *and* the (EP-gathered) inputs
            # into the graph with zero gradients: the weights still get a gradient, and the
            # backward of the EP all-gather (a reduce-scatter every EP rank must join) still runs.
            return _KeepInGraph.apply(torch.zeros_like(x), x, w, self.gate_up_proj, self.down_proj)
        order = order[:n_mine]
        tok = torch.div(order, idx.shape[1], rounding_mode="floor")
        slot = order % idx.shape[1]
        xs = x[tok]
        ws = w[tok, slot].unsqueeze(-1)
        if self.grouped_gemm is not None:
            group_lens = counts_t[: self.num_local_experts]  # already int64
            gate, up = self.grouped_gemm(xs, self.gate_up_proj, group_lens, trans_b=True).chunk(2, dim=-1)
            if self.recompute_act and self.training and gate.requires_grad:
                # moe_act (Megatron's semantics): the SwiGLU output is freed once fc2 ran and
                # recomputed from gate / up when fc2's output gradient arrives
                ckpt = tensor_parallel.CheckpointWithoutOutput(fp8=None)  # None = no fp8 (refused)
                act = ckpt.checkpoint(_swiglu, gate, up)
                y = self.grouped_gemm(act, self.down_proj, group_lens, trans_b=True)
                ckpt.discard_output_and_register_recompute(y)
            else:
                y = self.grouped_gemm(_swiglu(gate, up), self.down_proj, group_lens, trans_b=True)
            ys = y * ws.to(y.dtype)
        else:
            # unbind: one backward stack instead of a full-size zero gradient per indexed expert
            gate_up, down = self.gate_up_proj.unbind(0), self.down_proj.unbind(0)
            parts, start = [], 0
            for j, c in enumerate(counts[: self.num_local_experts]):
                if c == 0:
                    continue
                gate, up = F.linear(xs[start : start + c], gate_up[j]).chunk(2, dim=-1)
                y = F.linear(_swiglu(gate, up), down[j])
                parts.append(y * ws[start : start + c].to(y.dtype))
                start += c
            ys = torch.cat(parts, 0)
        return torch.zeros_like(x).index_add_(0, tok, ys.to(x.dtype))

    def _shared(self, x, gammas):  # x [N, H], gammas [N, S]
        # One GEMM over all shared experts' gate / up rows (no [S, N, H] copy of x; backward
        # builds one [N, H] input gradient), then one accumulating GEMM per shared expert (S is
        # small, 2 for Inkling) on its own [H, F] down weight, a view: no [H, S * F] copy of the
        # weights per call (a permute of [S, H, F] is not viewable as [H, S * F]).
        S, two_f, H = self.shared_gate_up_proj.shape
        gate, up = (
            (x @ self.shared_gate_up_proj.reshape(S * two_f, H).t()).view(-1, S, two_f).chunk(2, dim=-1)
        )
        act = _swiglu(gate, up) * gammas.unsqueeze(-1).to(gate.dtype)  # [N, S, F]
        out = act[:, 0] @ self.shared_down_proj[0].t()
        for s in range(1, S):
            out = torch.addmm(out, act[:, s], self.shared_down_proj[s].t())
        return out

    def forward(self, hidden_states, *_args, padding_mask=None, **_kwargs):
        """``[s, b, H] -> ([s, b, H], None)`` (Megatron's ``(output, bias)`` convention).

        ``padding_mask`` (Megatron's MoE-routing padding mask) is refused, not ignored: HF
        Inkling routes every token, and this layer has no masked routing path.

        Selective recompute as Megatron's MoELayer: ``moe`` recomputes this whole layer (the
        router counts tokens for its grad-enabled forward only), ``shared_experts`` the shared
        experts, ``moe_act`` the routed experts' SwiGLU (only on the Primus-Turbo grouped-GEMM
        path, as Megatron's grouped-GEMM experts; the per-expert fallback used without Turbo
        ignores ``moe_act``)."""
        _refuse_padding_mask(padding_mask)
        if self.moe_layer_recompute and self.training:
            out = tensor_parallel.checkpoint(self._forward, False, hidden_states)
        else:
            out = self._forward(hidden_states)
        return out, None

    def _forward(self, hidden_states):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        w, idx, gammas = self.gate(flat)
        if self.ep > 1:
            g_flat = gather_from_sequence_parallel_region(flat, group=self.ep_group)
            g_w = gather_from_sequence_parallel_region(w, group=self.ep_group)
            g_idx = torch.empty((idx.shape[0] * self.ep, idx.shape[1]), dtype=idx.dtype, device=idx.device)
            torch.distributed.all_gather_into_tensor(g_idx, idx.contiguous(), group=self.ep_group)
            routed = self._experts(g_flat, g_idx, g_w)
            routed = reduce_scatter_to_sequence_parallel_region(routed, group=self.ep_group)
        else:
            routed = self._experts(flat, idx, w)
        if self.shared_experts_recompute and self.training and not self.moe_layer_recompute:
            shared = tensor_parallel.checkpoint(self._shared, False, flat, gammas)
        else:
            shared = self._shared(flat, gammas)
        return (routed + shared).view(shape)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """The stacked routed experts are sharded along axis 0 (the global expert id) over EP
        and replicated over TP and the expert-data-parallel group; the rest is replicated."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        expert_keys = ("gate_up_proj", "down_proj")
        # Megatron's default (TP-replicated tensors) for every key; the two expert keys are then
        # replaced below. The module itself is not modified (parameter order stays as built).
        sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        ep_rank = self.ep_group.rank() if self.ep_group is not None else 0
        # replicas of an expert shard: the expert-TP ranks (ETP = TP; the MoE is replicated over
        # TP, so each TP rank holds a full copy) and the expert-data-parallel ranks
        tp_rank = self.expt_tp_group.rank() if self.expt_tp_group is not None else 0
        dp_rank = self.expt_dp_group.rank() if self.expt_dp_group is not None else 0
        n = len(sharded_offsets)
        for k in expert_keys:
            sd[prefix + k] = ShardedTensor.from_rank_offsets(
                prefix + k,
                self._parameters[k],
                *sharded_offsets,
                (n, ep_rank, self.ep),
                replica_id=(0, tp_rank, dp_rank),
                prepend_axis_num=n,
            )
        return sd
