###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Ideogram-4 dataloaders.

Both emit the same batch contract the adapter and the flow-matching pipeline
consume (``image_latents``, ``llm_features``, ``text_lengths``, ``data_type``),
so a config can swap one for the other:

    synthetic.py  fixed per-index synthetic tensors for an overfit smoke, no
                  encoder weights and no dataset required
    cache.py      the real pre-encoded cache produced by ``processor.py``

They are referenced from YAML by ``_target_``, not imported here.
"""
