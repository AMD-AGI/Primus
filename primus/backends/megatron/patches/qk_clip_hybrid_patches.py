###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
qk_clip (MuonClip) Hybrid / Distributed-Optimizer Patch
=======================================================

Upstream Megatron-LM's ``qk_clip`` (MuonClip attention-logit clipping) assumes
that every transformer layer has a ``.self_attention`` and that the fp32 master
weight is a full 2-D tensor. Both assumptions break for KDA/Mamba hybrid stacks
and for the distributed optimizer, so enabling ``qk_clip`` on those configs
either raises ``AttributeError`` or silently fails to rescale the master
weight. This patch fixes all three spots in place, keeping the Megatron-LM
submodule pristine:

    1. ``optimizer/qk_clip.py::clip_qk`` -- skip layers without a
       ``qk_clip``-capable ``self_attention`` (hybrid models interleave
       attention with linear/Mamba/KDA layers), and reset the per-head buffer in
       log-only mode so it reports a per-step (not running) max.
    2. ``optimizer/distrib_optimizer.py::_build_model_and_main_param_groups`` --
       record each shard's flat offset (``main_param_shard_start``) so ``clip_qk``
       can locate the sharded fp32 master.
    3. ``transformer/multi_latent_attention.py::MLASelfAttention.clip_qk`` --
       floor the ``eta`` denominator (a head with max logit <= 0 would give a
       negative base and ``eta ** alpha == NaN``) and rescale both the
       DP-replicated bf16 weight and the fp32 master (flat shard under the
       distributed optimizer, or full 2-D otherwise).

These are "source-string rewrite" patches: the edits sit in the middle of
upstream method/function bodies, where a plain wrapping monkey-patch cannot
reach without duplicating large, version-sensitive code (see
``_source_patch_utils``). With no ``qk_clip``-capable attention present the
paths are no-ops, so non-hybrid / non-``qk_clip`` configs are unaffected.
"""

import inspect
import textwrap

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._source_patch_utils import (
    patch_function_source,
    patch_method_source,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.qk_clip_hybrid"

# --- 1. qk_clip.clip_qk: skip non-attention layers, reset buffer in log-only ---
_QKCLIP_ORI = """\
                if hasattr(transformer_layer.self_attention, 'clip_qk'):
                    if (
                        transformer_layer.self_attention.core_attention.current_max_attn_logits
                        is None
                    ):
                        continue
                    torch.distributed.all_reduce(
                        transformer_layer.self_attention.core_attention.current_max_attn_logits,
                        op=torch.distributed.ReduceOp.MAX,
                        group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                    )
                    log_max_attention_logit = max(
                        log_max_attention_logit,
                        torch.max(
                            transformer_layer.self_attention.core_attention.current_max_attn_logits
                        ).item(),
                    )
                    if not log_max_only:
                        transformer_layer.self_attention.clip_qk()"""

_QKCLIP_NEW = """\
                # Hybrid models interleave attention layers with linear/Mamba/KDA
                # layers that have no ``self_attention``; skip those.
                self_attn = getattr(transformer_layer, 'self_attention', None)
                if self_attn is not None and hasattr(self_attn, 'clip_qk'):
                    if self_attn.core_attention.current_max_attn_logits is None:
                        continue
                    torch.distributed.all_reduce(
                        self_attn.core_attention.current_max_attn_logits,
                        op=torch.distributed.ReduceOp.MAX,
                        group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                    )
                    log_max_attention_logit = max(
                        log_max_attention_logit,
                        torch.max(self_attn.core_attention.current_max_attn_logits).item(),
                    )
                    if not log_max_only:
                        self_attn.clip_qk()
                    else:
                        # log-only: clip_qk() (which resets the buffer) isn't
                        # called, so reset here for a per-step (not running) max.
                        self_attn.core_attention.current_max_attn_logits = None"""

# --- 2. distrib_optimizer: record the flat shard offset of each master param ---
_DISTOPT_ORI = """\
                    model_param.main_param = shard_main_param
                    model_param.main_param_sharded = True"""

_DISTOPT_NEW = (
    _DISTOPT_ORI
    + """
                    # Flat offset of this rank's shard in the flattened param
                    # (main_param == model_param.view(-1)[start:start+numel]);
                    # used by qk_clip to rescale the sharded fp32 master.
                    model_param.main_param_shard_start = param_range.start"""
)

# --- 3. MLASelfAttention.clip_qk: eta NaN guard + sharded/replicated rescale ---
_MLA_ORI = """\
            self.qk_clip_balancing_eta = torch.clamp(
                self.config.qk_clip_threshold / self.core_attention.current_max_attn_logits, max=1.0
            ).view(self.num_attention_heads_per_partition, 1, 1)
            assert torch.all(self.qk_clip_balancing_eta <= 1.0)

            # Update q side weight, keep qk_pos_emb_head_dim side weight unchanged
            if self.config.q_lora_rank is None:
                q_proj_weight = self.linear_q_proj.weight
            else:
                q_proj_weight = self.linear_q_up_proj.weight

            # Handle different weight access patterns (main_param vs direct access)
            if hasattr(q_proj_weight, 'main_param'):
                q_proj_weight.main_param.data.copy_(
                    self._clip_q_proj_weight(q_proj_weight.main_param.data)
                )
            q_proj_weight.data.copy_(self._clip_q_proj_weight(q_proj_weight.data))

            # Update k side weight, keep v side weight unchanged
            kv_proj_weight = self.linear_kv_up_proj.weight

            # Handle different weight access patterns
            if hasattr(kv_proj_weight, 'main_param'):
                kv_proj_weight.main_param.data.copy_(
                    self._clip_kv_proj_weight(kv_proj_weight.main_param.data)
                )
            kv_proj_weight.data.copy_(self._clip_kv_proj_weight(kv_proj_weight.data))"""

_MLA_NEW = """\
            # Floor the denominator: a head with max logit <=0 would give a
            # negative eta and eta**alpha = NaN. With the floor, heads with
            # max_logit <= threshold get eta = 1.0 (no clip); an overflowed
            # (+inf) logit yields eta = 0 (fully clipped), which is well-defined.
            _max_logit = self.core_attention.current_max_attn_logits.clamp(min=1e-6)
            self.qk_clip_balancing_eta = torch.clamp(
                self.config.qk_clip_threshold / _max_logit, max=1.0
            ).view(self.num_attention_heads_per_partition, 1, 1)
            assert torch.all(self.qk_clip_balancing_eta <= 1.0)

            # Rescale both the DP-replicated bf16 weight and the fp32 master. With
            # the distributed optimizer the master is a flat shard, so build the
            # full per-element factor (matching weight.view(-1)) and slice this
            # rank's shard by its offset; view_as also covers the non-distributed
            # full 2-D master (start=0).
            eta = self.qk_clip_balancing_eta  # [n, 1, 1], <= 1.0
            n = self.num_attention_heads_per_partition
            a = self.config.qk_head_dim
            alpha = self.config.qk_clip_alpha

            def _rescale(model_weight, rows_per_head, head_factor):
                # head_factor: [n, rows_per_head, 1] fp32 multiplicative factor
                w = model_weight.data
                cols = w.numel() // (n * rows_per_head)
                # (1) replicated bf16 model weight (full)
                w.view(n, rows_per_head, cols).mul_(head_factor.to(w.dtype))
                # (2) fp32 master: flat shard (distributed) or full 2-D (regular)
                mp = getattr(model_weight, 'main_param', None)
                if mp is not None:
                    start = int(getattr(model_weight, 'main_param_shard_start', 0))
                    numel = mp.numel()
                    flat = head_factor.expand(n, rows_per_head, cols).reshape(-1)
                    mp.data.mul_(flat[start : start + numel].to(mp.dtype).view_as(mp.data))

            # q side: content (nope) part *= eta^alpha, rotary (pe) part *= eta
            b_pe = self.config.qk_pos_emb_head_dim
            q_factor = torch.ones(n, a + b_pe, 1, device=eta.device, dtype=torch.float32)
            q_factor[:, :a, :] = torch.pow(eta, alpha)
            q_factor[:, a:, :] = eta
            if self.config.q_lora_rank is None:
                _rescale(self.linear_q_proj.weight, a + b_pe, q_factor)
            else:
                _rescale(self.linear_q_up_proj.weight, a + b_pe, q_factor)

            # k side: k part *= eta^(1-alpha), v part unchanged
            v = self.config.v_head_dim
            kv_factor = torch.ones(n, a + v, 1, device=eta.device, dtype=torch.float32)
            kv_factor[:, :a, :] = torch.pow(eta, 1.0 - alpha)
            _rescale(self.linear_kv_up_proj.weight, a + v, kv_factor)"""


def _patch_distopt_classmethod() -> None:
    """Insert ``main_param_shard_start`` into the distributed-optimizer classmethod.

    ``_build_model_and_main_param_groups`` is a ``@classmethod``, which the shared
    ``patch_method_source`` helper does not handle, so rewrite its source here and
    re-wrap the result as a classmethod.
    """
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    func = DistributedOptimizer.__dict__["_build_model_and_main_param_groups"].__func__
    raw = inspect.getsource(func)
    assert _DISTOPT_ORI in raw, "[qk_clip_hybrid] distrib_optimizer anchor not found"
    # Replace on the raw (still-indented) source so the absolute-column anchor
    # matches, then dedent once so the whole body shifts by a single amount.
    source = textwrap.dedent(raw.replace(_DISTOPT_ORI, _DISTOPT_NEW))
    # Drop the decorator line(s) so the bare def can be exec'd, then re-wrap.
    source = "".join(l for l in source.splitlines(keepends=True) if not l.lstrip().startswith("@"))
    namespace: dict = {}
    exec(source, func.__globals__, namespace)  # noqa: S102
    DistributedOptimizer._build_model_and_main_param_groups = classmethod(namespace[func.__name__])


def _install_qk_clip_hybrid_patches() -> None:
    import megatron.core.optimizer.qk_clip as qk_clip_mod
    import megatron.training.training as training_mod
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention

    if is_patched(qk_clip_mod, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] already applied; skipping.")
        return

    # 1. clip_qk: rebind the module attr and the name already imported by training.
    new_clip_qk = patch_function_source(qk_clip_mod, "clip_qk", _QKCLIP_ORI, _QKCLIP_NEW)
    training_mod.clip_qk = new_clip_qk

    # 2. distributed-optimizer shard offset.
    _patch_distopt_classmethod()

    # 3. MLASelfAttention.clip_qk (inherited unchanged by PrimusMLASelfAttention).
    patch_method_source(MLASelfAttention, "clip_qk", _MLA_ORI, _MLA_NEW)

    mark_patched(qk_clip_mod, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] Patched clip_qk (hybrid skip), distrib_optimizer "
        "(shard offset), and MLASelfAttention.clip_qk (eta guard + sharded rescale)."
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Make qk_clip (MuonClip) work with hybrid (KDA/Mamba) stacks and the "
        "distributed optimizer: skip non-attention layers, record the master "
        "shard offset, and rescale both bf16 weight and fp32 master with an "
        "eta NaN guard."
    ),
    condition=lambda ctx: (
        getattr(get_args(ctx), "qk_clip", False) or getattr(get_args(ctx), "log_max_attention_logit", False)
    ),
)
def patch_qk_clip_hybrid(ctx: PatchContext) -> None:
    # qk_clip rescales the fp32 master through ``model_param.main_param``. With
    # ``use_precision_aware_optimizer`` that master is owned by FusedAdam and
    # ``main_param`` is unset, so the clip would touch only the bf16 weight and
    # get overwritten on the next optimizer step. Reject the combination rather
    # than silently under-clip. (Log-only mode never rescales, so it is fine.)
    args = get_args(ctx)
    if getattr(args, "qk_clip", False) and getattr(args, "use_precision_aware_optimizer", False):
        raise NotImplementedError(
            "qk_clip is not supported together with use_precision_aware_optimizer: "
            "the fp32 master is owned by the fused optimizer and cannot be rescaled "
            "by this patch. Disable one of the two."
        )
    _install_qk_clip_hybrid_patches()
