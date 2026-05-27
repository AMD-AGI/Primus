###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton kernels backing :mod:`primus...transformer.v4_attention_kernels`.

Submodules:

* :mod:`v4_attention_fwd` — forward kernel for the dense
  (``compress_ratio == 0``) and HCA (``compress_ratio == 128``) paths.
* :mod:`v4_attention_bwd` — backward kernel matching the FWD's saved
  LSE.
* :mod:`v4_csa_attention_fwd` — forward kernel for the CSA
  (``compress_ratio == 4``) fused local-SWA + per-query top-K + sink
  path.
* :mod:`v4_csa_attention_bwd` — backward kernel for the CSA path
  (``dq, dk_local, dv_local, dgathered, dsink``).
* :mod:`rope_interleaved_partial` — plan-6 P35 fused interleaved partial
  RoPE FWD/BWD (replaces the 9-op eager chain in
  :func:`primus.backends.megatron.core.transformer.dual_rope.apply_interleaved_partial_rope`).
* :mod:`sinkhorn` — plan-6 P36 fused Sinkhorn-Knopp FWD/BWD (replaces
  the plan-5 P29 ``torch.compile`` fast path in
  :func:`primus.backends.megatron.core.transformer.hyper_connection.sinkhorn_normalize`).
  Runs the full alternating row/col normalize trajectory in registers
  per row of the leading axis; BWD recomputes the trajectory and walks
  the analytic VJP backward step by step.
"""
