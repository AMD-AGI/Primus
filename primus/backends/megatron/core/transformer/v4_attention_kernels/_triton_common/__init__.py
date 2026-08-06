###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared (non-attention) Triton kernels for the DeepSeek-V4 pipeline.

These back the surrounding V4 components rather than the attention core, and
are imported by full submodule path (no package-level re-export):

* :mod:`indexer_score` / :mod:`indexer_score_post` — lightning indexer scores
* :mod:`compressor_pool` — compressed-pool builder
* :mod:`hc_expand` / :mod:`hc_glue` — hierarchical-compression expand / glue
* :mod:`sinkhorn` — fused Sinkhorn-Knopp normalize (FWD/BWD)
* :mod:`rope_interleaved_partial` — fused interleaved partial RoPE (FWD/BWD)
* :mod:`indexer_distill_target` / :mod:`indexer_distill_kl` /
  :mod:`indexer_distill_window_lse` — the three halves of the indexer
  distillation loss: the KL target, the KL tail, and the sliding-window log
  mass the target's denominator needs. Only reached when
  ``v4_indexer_distill_loss_coeff > 0``.

The attention kernels live in :mod:`.._triton_v0_deprecated` / :mod:`.._triton_v1` /
:mod:`.._triton_v2`.
"""
