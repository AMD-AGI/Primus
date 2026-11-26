###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText MoE patches (ROCm/maxtext release/v26.10).

- ``maxtext.moe.expert_balance`` (38d321cff): with ``expert_balance: true``,
  ``RoutedMoE.dense_matmul`` replaces the router logits with a deterministic
  rotation so tokens spread uniformly over the experts (benchmarking aid).
- ``maxtext.moe.routing_maps_remat`` (b2632e16d): tag the router top-k expert
  ids with ``checkpoint_name("moe_routing_maps")`` and save that name in the
  ``minimal*`` remat policies, so a bf16 top-k recomputed in the backward pass
  cannot pick different experts than the forward did.
- ``maxtext.moe.turbo_grouped_gemm`` (a1efcbc98 .. 2fb48f50e): Primus-Turbo
  grouped GEMM (``use_turbo_grouped_gemm``, backends ck/hipblaslt/flydsl,
  bf16/fp8) in ``RoutedMoE.sparse_matmul``, including ring-of-experts with
  ``use_ragged_sort`` (non-zero ``group_offset``) for both the turbo GEMM and the
  ``jax.lax.ragged_dot`` fallback. The kernel wrapper lives in
  ``primus.backends.maxtext.kernels.turbo_grouped_gemm``.
"""

import functools
import inspect

from primus.backends.maxtext.patches._source_patch_utils import (
    any_in_source,
    rewrite_source,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


def _import_moe(prefix):
    try:
        from maxtext.layers import moe
    except ImportError as e:
        warning_rank_0(f"{prefix} maxtext.layers.moe not importable; skipped: {e}")
        return None
    return moe


# ---------------------------------------------------------------------------
# expert_balance
# ---------------------------------------------------------------------------


def balanced_gate_logits(gate_logits):
    """Deterministic rotational gate logits: token ``s`` ranks the experts ``(e + s) % E``."""
    import jax.numpy as jnp

    batch_size, seq_len, num_experts = gate_logits.shape
    # Base weights for experts (increasing values)
    base_weights = jnp.linspace(0.1, 0.1 * num_experts, num_experts, dtype=gate_logits.dtype)
    # Position-based indices [seq_len, num_experts]: which base weight each position uses after rotation
    indices = (jnp.arange(num_experts)[None, :] + jnp.arange(seq_len)[:, None]) % num_experts
    rotated_weights = base_weights[indices]
    return jnp.broadcast_to(rotated_weights[None, :, :], (batch_size, seq_len, num_experts))


@register_patch(
    patch_id="maxtext.moe.expert_balance",
    backend="maxtext",
    phase="setup",
    description="expert_balance: force a uniform, deterministic expert assignment in RoutedMoE.dense_matmul",
    condition=lambda ctx: True,
)
def patch_expert_balance(ctx: PatchContext) -> None:
    del ctx
    prefix = "[Patch:maxtext.moe.expert_balance]"
    moe = _import_moe(prefix)
    if moe is None:
        return
    if any_in_source(moe.RoutedMoE, "dense_matmul", ["expert_balance"]):
        log_rank_0(f"{prefix} RoutedMoE.dense_matmul already handles expert_balance; not patched.")
        return

    orig = moe.RoutedMoE.dense_matmul
    signature = inspect.signature(orig)

    @functools.wraps(orig)
    def dense_matmul(self, *args, **kwargs):
        if not self.config.expert_balance:
            return orig(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.arguments["gate_logits"] = balanced_gate_logits(bound.arguments["gate_logits"])
        return orig(*bound.args, **bound.kwargs)

    moe.RoutedMoE.dense_matmul = dense_matmul
    log_rank_0(f"{prefix} RoutedMoE.dense_matmul honours expert_balance.")


# ---------------------------------------------------------------------------
# moe_routing_maps remat
# ---------------------------------------------------------------------------

_TOPK_ANCHOR = "        top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)\n"
_TOPK_NEW = (
    _TOPK_ANCHOR
    + "        # moe_routing_maps_topk: tagged so that the minimal* remat policies SAVE the ids instead of recomputing them (a recomputed bf16 top-k can differ from the forward's);\n"
    "        # the weights are re-gathered with the saved ids so ids and weights stay consistent\n"
    '        top_k_indices = adc.checkpoint_name(top_k_indices, "moe_routing_maps")\n'
    "        top_k_weights = jnp.take_along_axis(gate_logits, top_k_indices, axis=-1)\n"
)

_POLICY_ANCHOR = '        "mlpwo",\n    ]\n'
_POLICY_NEW = '        "mlpwo",\n        "moe_routing_maps",\n    ]\n'


@register_patch(
    patch_id="maxtext.moe.routing_maps_remat",
    backend="maxtext",
    phase="setup",
    description="Save the MoE router top-k expert ids (moe_routing_maps) across remat in the minimal* policies",
    condition=lambda ctx: True,
)
def patch_routing_maps_remat(ctx: PatchContext) -> None:
    del ctx
    prefix = "[Patch:maxtext.moe.routing_maps_remat]"
    moe = _import_moe(prefix)
    if moe is None:
        return
    from maxtext.layers import nnx_decoders

    if any_in_source(moe.RoutedMoE, "get_topk", ['checkpoint_name(top_k_indices, "moe_routing_maps")']):
        log_rank_0(f"{prefix} RoutedMoE.get_topk already tags moe_routing_maps; skipped.")
    else:
        rewrite_source(moe.RoutedMoE, "get_topk", [(_TOPK_ANCHOR, _TOPK_NEW)])
        log_rank_0(f"{prefix} RoutedMoE.get_topk tags the top-k ids as moe_routing_maps.")

    if any_in_source(nnx_decoders.NNXDecoder, "minimal_policy", ['"moe_routing_maps"']):
        log_rank_0(f"{prefix} NNXDecoder.minimal_policy already saves moe_routing_maps; skipped.")
    else:
        rewrite_source(nnx_decoders.NNXDecoder, "minimal_policy", [(_POLICY_ANCHOR, _POLICY_NEW)])
        log_rank_0(f"{prefix} NNXDecoder.minimal_policy saves moe_routing_maps.")


# ---------------------------------------------------------------------------
# Primus-Turbo grouped GEMM (+ ragged_dot group_offset window)
# ---------------------------------------------------------------------------

_GMM_ANCHOR = """\
      else:
        # jax.lax.ragged_dot
        output = jax_ragged_dot_gmm(
            inputs,
            kernel,
            tiling,
            group_sizes,
            expert_assignments,
            padding_amount,
        )
"""

_GMM_NEW = """\
      elif self.config.use_turbo_grouped_gemm:
        if not (isinstance(group_offset, int) and group_offset == 0):
          # ring-of-experts + ragged sort: rhs holds the local experts, group_sizes are global, the local rows start at sum(group_sizes[:group_offset])
          output = turbo_grouped_gemm.gmm_window(
              inputs, kernel, group_sizes, group_offset, backend=self.config.turbo_grouped_gemm_backend, quant=self.config.turbo_grouped_gemm_quant
          )
        else:
          output = turbo_grouped_gemm.gmm(
              inputs,
              kernel,
              group_sizes,
              backend=self.config.turbo_grouped_gemm_backend,
              exact_rows=self.get_expert_parallelism_size() == 1,
              quant=self.config.turbo_grouped_gemm_quant,
          )
      else:
        # jax.lax.ragged_dot
        if not (isinstance(group_offset, int) and group_offset == 0):
          # ring-of-experts + ragged sort: kernel holds the local experts only and group_sizes are global (same window as gmm_window above)
          output = turbo_grouped_gemm.window_apply(
              inputs,
              kernel.shape[0],
              group_sizes,
              group_offset,
              lambda lhs, gs: jax_ragged_dot_gmm(lhs, kernel, tiling, gs, expert_assignments, padding_amount),
          )
        else:
          output = jax_ragged_dot_gmm(
              inputs,
              kernel,
              tiling,
              group_sizes,
              expert_assignments,
              padding_amount,
          )
"""


@register_patch(
    patch_id="maxtext.moe.turbo_grouped_gemm",
    backend="maxtext",
    phase="setup",
    description="Primus-Turbo grouped GEMM (use_turbo_grouped_gemm) and ring-of-experts group_offset support in RoutedMoE.sparse_matmul",
    condition=lambda ctx: True,
)
def patch_turbo_grouped_gemm(ctx: PatchContext) -> None:
    del ctx
    prefix = "[Patch:maxtext.moe.turbo_grouped_gemm]"
    moe = _import_moe(prefix)
    if moe is None:
        return
    if any_in_source(moe.RoutedMoE, "sparse_matmul", ["use_turbo_grouped_gemm"]):
        log_rank_0(f"{prefix} RoutedMoE.sparse_matmul already supports use_turbo_grouped_gemm; not patched.")
        return

    from primus.backends.maxtext.kernels import turbo_grouped_gemm

    moe.turbo_grouped_gemm = turbo_grouped_gemm
    rewrite_source(moe.RoutedMoE, "sparse_matmul", [(_GMM_ANCHOR, _GMM_NEW)])
    log_rank_0(f"{prefix} RoutedMoE.sparse_matmul supports use_turbo_grouped_gemm.")
