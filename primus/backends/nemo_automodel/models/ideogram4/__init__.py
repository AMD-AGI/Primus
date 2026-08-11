###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Ideogram-4 hooks for the AutoModel diffusion backend.

Single-stream rectified-flow T2I. All of it is Primus-side: the Automodel
submodule and diffusers stay pristine.

    adapter.py         flow-matching adapter: packs [left-pad][text][image],
                       maps sigma -> t, negates the DiT velocity
    attention.py       var-len (cu_seqlens) flash attention processor, the exact
                       replacement for the dense-mask SDPA path
    packing_buffer.py  per-step transport carrying packing metadata from the
                       adapter to the attention processor
    parallelize.py     parallelization strategy: real AC + FSDP2 sharding
    zero1.py           DDP + ZeRO-1 sharded-optimizer path
    profile.py         torch.profiler wrapper around the recipe train loop
    processor.py       offline encoder (VAE + text encoder) producing the cache
    data/              dataloaders: synthetic.py (smoke) and cache.py (real)

``attention.py`` and ``packing_buffer.py`` are mutually dependent by design (the
attention processor reads what the adapter published); everything else in here
depends only on the Automodel/diffusers API.
"""
