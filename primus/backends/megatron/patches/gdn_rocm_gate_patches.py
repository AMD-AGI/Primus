###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron upstream GatedDeltaNet ROCm gate patch
==============================================

Qwen3.5 and other models with ``experimental_attention_variant: gated_delta_net``
use Megatron-LM's ``megatron.core.ssm.gated_delta_net.GatedDeltaNet`` (not the
Primus HybridStack copy).  On ROCm, FLA's Triton ``chunk_gated_delta_rule`` can
produce NaN gradients when the gate is fused inside the kernel
(``use_gate_in_kernel=True``, the default).  Pre-computing ``g`` in fp32 and
passing ``use_gate_in_kernel=False`` matches the Primus hybrid GDN path and
avoids NaN gradients during backward on ROCm.
"""

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._source_patch_utils import patch_method_source
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.ssm.gdn_rocm_gate"

# Unique anchor in GatedDeltaNet.forward (megatron/core/ssm/gated_delta_net.py).
_GATE_ORI = (
    "        g = -self.A_log.exp() * F.softplus(alpha.float() + self.dt_bias)  # In fp32\n"
    "        beta = beta.sigmoid()\n"
    '        nvtx_range_pop(suffix="g_and_beta")\n'
    "\n"
    '        nvtx_range_push(suffix="gated_delta_rule")\n'
    "        if self.config.deterministic_mode:\n"
    "            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(\n"
    "                query,\n"
    "                key,\n"
    "                value,\n"
    "                g=g,\n"
    "                beta=beta,\n"
    "                initial_state=None,\n"
    "                output_final_state=False,\n"
    "                use_qk_l2norm_in_kernel=False,\n"
    "            )\n"
    "        else:\n"
    "            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(\n"
    "                query,\n"
    "                key,\n"
    "                value,\n"
    "                g=g,\n"
    "                beta=beta,\n"
    "                initial_state=None,\n"
    "                output_final_state=False,\n"
    "                use_qk_l2norm_in_kernel=False,\n"
    "            )"
)

_GATE_NEW = (
    "        # Pre-compute gate in fp32; use_gate_in_kernel=False avoids FLA Triton NaN on ROCm.\n"
    "        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)\n"
    "        beta = beta.sigmoid()\n"
    '        nvtx_range_pop(suffix="g_and_beta")\n'
    "\n"
    '        nvtx_range_push(suffix="gated_delta_rule")\n'
    "        if self.config.deterministic_mode:\n"
    "            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(\n"
    "                query,\n"
    "                key,\n"
    "                value,\n"
    "                g=g,\n"
    "                beta=beta,\n"
    "                initial_state=None,\n"
    "                output_final_state=False,\n"
    "                use_qk_l2norm_in_kernel=False,\n"
    "            )\n"
    "        else:\n"
    "            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(\n"
    "                query,\n"
    "                key,\n"
    "                value,\n"
    "                g=g.to(query.dtype),\n"
    "                beta=beta,\n"
    "                initial_state=None,\n"
    "                output_final_state=False,\n"
    "                use_qk_l2norm_in_kernel=False,\n"
    "                use_gate_in_kernel=False,\n"
    "            )"
)


def _is_rocm(ctx: PatchContext) -> bool:
    """Return True when running on an AMD ROCm platform."""
    return getattr(torch.version, "hip", None) is not None


def _uses_gated_delta_net(args) -> bool:
    return getattr(args, "experimental_attention_variant", None) == "gated_delta_net"


def _should_patch_gdn_rocm_gate(ctx: PatchContext) -> bool:
    return _is_rocm(ctx) and _uses_gated_delta_net(get_args(ctx))


def _install_gdn_rocm_gate_patch() -> None:
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    if is_patched(GatedDeltaNet, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] GatedDeltaNet already patched; skipping.")
        return

    patch_method_source(GatedDeltaNet, "forward", _GATE_ORI, _GATE_NEW)

    mark_patched(GatedDeltaNet, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] Patched GatedDeltaNet.forward: fp32 gate precompute + "
        "use_gate_in_kernel=False for FLA chunk_gated_delta_rule on ROCm."
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Pre-compute GDN gate in fp32 and disable FLA in-kernel gate fusion on "
        "Megatron upstream GatedDeltaNet to avoid NaN gradients on ROCm."
    ),
    condition=_should_patch_gdn_rocm_gate,
    priority=51,
    tags=["rocm", "gdn"],
)
def patch_gdn_rocm_gate(ctx: PatchContext) -> None:
    _install_gdn_rocm_gate_patch()
