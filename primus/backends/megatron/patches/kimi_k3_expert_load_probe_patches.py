###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Record the global expert-load histogram once per optimizer step.

Phase 2's A/B compares Kimi K3's Quantile Balancing against DeepSeek-V3's
sign rule. The quantity that discriminates them is the *expert load*, and
Megatron logs no such thing: the per-iteration line carries only
``seq_load_balancing_loss``, which is the auxiliary loss rather than the
histogram.

Where the numbers come from
---------------------------
``TopKRouter`` keeps a non-persistent ``local_tokens_per_expert`` buffer of
shape ``[num_moe_experts]`` whenever ``moe_router_enable_expert_bias`` is set
(``router.py``) and accumulates ``routing_map.sum(dim=0)`` into it on every
*training* microbatch (``router.py`` -- the ``is_grad_enabled`` guard means
evaluation contributes nothing). It is zeroed once per global batch by
``reset_model_temporary_tensors`` (``finalize_model_grads.py``), which
``finalize_model_grads`` calls immediately after the expert-bias update.

So the buffers hold exactly one global batch's routing decisions at the moment
``reset_model_temporary_tensors`` is entered, and that is where this probe
reads them.

Why wrap the reset rather than the bias update
----------------------------------------------
``_update_router_expert_bias`` is itself rebound by
``kimi_k3_quantile_balancing_patches`` when the quantile rule is selected.
Wrapping the reset instead makes the probe independent of which bias rule is
installed and of patch ordering, so both arms of the A/B are measured by
identical code on an identical schedule. ``reset_model_temporary_tensors`` is
looked up as a module global, so rebinding the module attribute is sufficient.

``get_updated_expert_bias`` all-reduces a ``torch.stack`` *copy*
(``finalize_model_grads.py`` -> ``moe_utils.py``), so the per-router buffers
are still rank-local here and this probe does its own all-reduce over the same
TPxCPxDP group.

Activation
----------
Entirely driven by the ``expert_load_probe_path`` config parameter (a Kimi K3
field, declared on ``KimiK3TransformerConfig``), which holds the output path
and defaults to ``None``. Unset -- the default everywhere, including every unit
test -- and this module registers a patch whose condition is False, so nothing
is imported, wrapped or allocated. The path is passed through the normal Primus
config / CLI channel; it is deliberately NOT read from an environment variable.
"""

import atexit
import json
import math
import os

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# The Kimi K3 config / CLI key that carries the output path. Read at the args
# layer (get_args(ctx) == module_config.params) the same way every other K3
# patch reads its knobs; declared as a field on KimiK3TransformerConfig so it is
# a first-class parameter rather than an environment variable.
_CONFIG_KEY = "expert_load_probe_path"
_LOG_PREFIX = "[Patch:megatron.kimi_k3.expert_load_probe]"

# Dump the full per-layer histogram this often (in optimizer steps). The scalar
# summaries are written every step; the per-layer arrays are 23 x 32 numbers and
# are only needed to inspect the shape of the imbalance, not to plot it.
_FULL_DUMP_EVERY = 25


def _probe_path(ctx: PatchContext) -> str:
    """Output path from the ``expert_load_probe_path`` config key; "" when unset."""
    args = get_args(ctx)
    return str(getattr(args, _CONFIG_KEY, None) or "").strip()


def _wants_expert_load_probe(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    # Kimi K3 only. This probe wraps finalize_model_grads.reset_model_temporary_tensors
    # -- a function SHARED by every MoE model -- so it must be gated on model_type,
    # not just on moe_router_enable_expert_bias (DeepSeek-V3/V4 also set that flag).
    # Without this a non-K3 run that happened to set expert_load_probe_path would
    # have its grad finalization wrapped. Mirrors kimi_k3_flops_patches.py.
    if getattr(args, "model_type", None) != "kimi_k3":
        return False
    if not _probe_path(ctx):
        return False
    # The buffer this probe reads only exists when the expert bias is enabled.
    return bool(getattr(args, "moe_router_enable_expert_bias", False))


@register_patch(
    "megatron.kimi_k3.expert_load_probe",
    backend="megatron",
    phase="before_train",
    description=(
        "Write the all-reduced expert-load histogram, its entropy and its "
        "max/min ratio to a JSONL file once per optimizer step."
    ),
    priority=90,
    condition=_wants_expert_load_probe,
)
def patch_expert_load_probe(ctx: PatchContext):
    """Wrap ``reset_model_temporary_tensors`` with a measurement."""
    import importlib

    import torch
    from megatron.core import parallel_state
    from megatron.core.utils import get_attr_wrapped_model

    finalize_model_grads = importlib.import_module("megatron.core.distributed.finalize_model_grads")

    path = _probe_path(ctx)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    original = finalize_model_grads.reset_model_temporary_tensors
    state = {"step": 0, "failed": False, "handle": None}

    def _close_handle() -> None:
        handle = state["handle"]
        if handle is not None:
            try:
                handle.close()
            finally:
                state["handle"] = None

    atexit.register(_close_handle)

    def _rank() -> int:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def _write(record: dict) -> None:
        if state["handle"] is None:
            # Line-buffered: a killed run still leaves every completed step on
            # disk. This whole project has lost work to that failure mode.
            state["handle"] = open(path, "a", buffering=1)
        try:
            state["handle"].write(json.dumps(record) + "\n")
            state["handle"].flush()
        except Exception:
            _close_handle()
            raise

    @torch.no_grad()
    def _measure(config, model) -> None:
        counts = []
        clamped = []
        for model_chunk in model:
            for module in get_attr_wrapped_model(model_chunk, "modules")():
                if hasattr(module, "expert_bias") and module.training:
                    counts.append(module.local_tokens_per_expert)
                    if hasattr(module, "local_margin_clamped"):
                        clamped.append(module.local_margin_clamped)
        if not counts:
            return

        state["step"] += 1
        step = state["step"]

        # [num_moe_layers, num_experts], summed over every rank that routed a
        # token -- the same group get_updated_expert_bias reduces over.
        stacked = torch.stack(counts, dim=0).clone()
        torch.distributed.all_reduce(
            stacked,
            group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
        )

        clamp_total = 0
        if clamped:
            stacked_clamp = torch.stack(clamped, dim=0).clone()
            torch.distributed.all_reduce(
                stacked_clamp,
                group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
            )
            clamp_total = int(stacked_clamp.sum().item())

        if _rank() != 0:
            return

        counts_f = stacked.double().cpu()
        num_layers, num_experts = counts_f.shape
        totals = counts_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        probs = counts_f / totals

        # Natural log, so the perfect-balance ceiling is ln(num_experts) and the
        # units match the loss curve.
        entropy = -(probs * probs.clamp(min=1e-300).log()).sum(dim=-1)
        per_layer_max = counts_f.max(dim=-1).values
        per_layer_min = counts_f.min(dim=-1).values
        ratio = torch.where(
            per_layer_min > 0,
            per_layer_max / per_layer_min.clamp(min=1.0),
            torch.full_like(per_layer_max, float("inf")),
        )
        dead = int((counts_f == 0).sum().item())

        # Pooled over layers: one number per step, the cleanest headline series.
        pooled = counts_f.sum(dim=0)
        pooled_p = pooled / pooled.sum().clamp(min=1.0)
        pooled_entropy = float(-(pooled_p * pooled_p.clamp(min=1e-300).log()).sum().item())
        pooled_min = float(pooled.min().item())
        pooled_ratio = float(pooled.max().item() / pooled_min) if pooled_min > 0 else float("inf")

        finite = ratio[torch.isfinite(ratio)]
        record = {
            "step": step,
            "num_moe_layers": int(num_layers),
            "num_experts": int(num_experts),
            "max_entropy": math.log(num_experts),
            "entropy_mean": float(entropy.mean().item()),
            "entropy_min": float(entropy.min().item()),
            "entropy_max": float(entropy.max().item()),
            "maxmin_mean": float(finite.mean().item()) if finite.numel() else float("inf"),
            "maxmin_max": float(ratio.max().item()),
            "pooled_entropy": pooled_entropy,
            "pooled_maxmin": pooled_ratio,
            "dead_experts": dead,
            "tokens_routed": float(counts_f.sum().item()),
            "margin_clamped": clamp_total,
        }
        if step == 1 or step % _FULL_DUMP_EVERY == 0:
            record["per_layer_entropy"] = [round(v, 6) for v in entropy.tolist()]
            record["pooled_counts"] = [int(v) for v in pooled.tolist()]
        _write(record)

    def _instrumented_reset(config, model):
        if not state["failed"]:
            try:
                _measure(config, model)
            except Exception as exc:  # noqa: BLE001 - never take a run down
                state["failed"] = True
                log_rank_0(f"{_LOG_PREFIX}   DISABLED after an error: {exc!r}")
        return original(config, model)

    finalize_model_grads.reset_model_temporary_tensors = _instrumented_reset
    log_rank_0(f"{_LOG_PREFIX}   Wrapped reset_model_temporary_tensors; " f"expert-load histogram -> {path}")
