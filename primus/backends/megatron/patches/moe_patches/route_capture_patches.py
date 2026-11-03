###############################################################################
# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MoE route capture for building co-activation expert placements (PR #166 port).

Env-var gated (``CMOE_CAPTURE=1``; see ``primus/backends/megatron/core/capture.py``
for the full list). When active, every forward records each MoE layer's top-k
expert choices and gate weights per token and writes one ``.pt`` shard per
(iteration, micro-batch, rank) to ``CMOE_CAPTURE_DIR``.
``tools/moe_expert_placement/build_expert_placement.py`` turns those shards into
a placement JSON for ``moe_expert_placement``.

Instead of editing ``GPTModel.forward`` and the router (as the upstream PR does),
this wraps:

* ``GPTModel.forward`` -- ``begin_step`` / ``end_step`` around the decoder, and
* ``TopKRouter.forward`` -- derives the dense ``[tokens, topk]`` selection from
  the returned sparse ``probs`` / ``routing_map`` and records it.

The derived selection is ordered by gate weight, which is the ordering the
placement builder uses (``_winner_first``). Padding-masked tokens record a zero
weight; token dropping is not supported by the capture (it is not supported by
token dedup either).
"""

from __future__ import annotations

import os

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PREFIX = "[Patch:megatron.moe.route_capture]"
_KEY = "megatron.moe.route_capture"


def _capture_requested(ctx: PatchContext) -> bool:
    del ctx
    return os.environ.get("CMOE_CAPTURE", "").strip().lower() in ("1", "true", "yes", "on")


def _wrap_gpt_forward() -> None:
    from megatron.core.models.gpt import gpt_model as gpt_mod

    from primus.backends.megatron.core.capture import get_capturer

    cls = gpt_mod.GPTModel
    if is_patched(cls, _KEY):
        return
    orig_forward = cls.forward

    def forward(self, input_ids, position_ids, attention_mask, decoder_input=None, *a, **kw):
        capturer = get_capturer()
        # Embeddings are only captured with CMOE_CAPTURE_EMBEDDINGS; the decoder input is
        # usually computed inside forward, so only an explicitly passed one is recorded.
        capturer.begin_step(token_ids=input_ids, input_embeddings=decoder_input)
        try:
            return orig_forward(self, input_ids, position_ids, attention_mask, decoder_input, *a, **kw)
        finally:
            capturer.end_step()

    cls.forward = forward
    mark_patched(cls, _KEY)
    log_rank_0(f"{_PREFIX} Wrapped GPTModel.forward with route-capture begin_step/end_step")


def _wrap_router_forward() -> None:
    import torch
    from megatron.core.transformer.moe import router as router_mod

    from primus.backends.megatron.core.capture import get_capturer

    cls = router_mod.TopKRouter
    if is_patched(cls, _KEY):
        return
    orig_forward = cls.forward

    def forward(self, input, *a, **kw):
        probs, routing_map = orig_forward(self, input, *a, **kw)
        capturer = get_capturer()
        if capturer.wants_router and getattr(self, "routing_type", None) != "sinkhorn":
            seq_length, bsz = input.shape[:2]
            topk = self.topk
            # Selected experts first (routing_map), then by gate weight: slot 0 is the
            # highest-weight expert, as the capture format expects.
            key = probs.detach().float() + routing_map.to(torch.float32)
            top_indices = torch.topk(key, k=topk, dim=1).indices
            top_probs = torch.gather(probs.detach(), 1, top_indices)
            layer_number = self.layer_number if self.layer_number is not None else 0
            if getattr(self, "is_mtp_layer", False):
                layer_number = layer_number + self.config.num_layers
            capturer.record_router(
                int(layer_number),
                top_indices.view(seq_length, bsz, topk).transpose(0, 1),
                top_probs.view(seq_length, bsz, topk).transpose(0, 1),
            )
        return probs, routing_map

    cls.forward = forward
    mark_patched(cls, _KEY)
    log_rank_0(f"{_PREFIX} Wrapped TopKRouter.forward to record per-token expert choices")


@register_patch(
    _KEY,
    backend="megatron",
    phase="before_train",
    description="Record MoE routing decisions to disk when CMOE_CAPTURE=1 (input for build_expert_placement.py)",
    condition=_capture_requested,
)
def patch_route_capture(ctx: PatchContext):
    del ctx
    from primus.backends.megatron.core.capture import get_capturer

    capturer = get_capturer()
    os.makedirs(capturer.out_dir, exist_ok=True)
    _wrap_gpt_forward()
    _wrap_router_forward()
    log_rank_0(
        f"{_PREFIX} capture dir={capturer.out_dir} start={capturer.start_iter} "
        f"cons_iters={capturer.cons_iters} interval={capturer.interval}"
    )
