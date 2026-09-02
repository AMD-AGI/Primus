###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Ideogram-4 dataloaders.

Two of them, emitting the SAME batch contract, so a config can swap one for the
other without anything downstream noticing:

    synthetic.py  fixed per-index synthetic tensors. No encoder weights and no
                  dataset needed, which is what makes it usable as a smoke test
                  anywhere.
    cache.py      the real pre-encoded cache.

The contract both produce, and which the adapter and the flow-matching pipeline
consume:

    image_latents  [B, C, grid_h, grid_w]  clean packed latents, the x0 the
                                           pipeline adds noise to
    llm_features   [B, T, F]               LEFT-padded text features
    text_lengths   [B]                     real, non-pad token count per sample
    data_type      "image"

The loaders emit only these raw tensors. Everything about the packed
``[pad][text][image]`` layout -- the position, segment and indicator ids, and the
var-len packing -- is built by the adapter, so the two loaders have nothing to
keep in step with it.

LEFT-padding is the part that is easy to get backwards and silent when wrong. The
adapter marks the text region as the LAST ``n`` positions of the text width, so
features padded on the right would put real tokens where the adapter expects
padding and vice versa. Nothing would error; the model would simply train on
conditioning that does not line up with its own position ids.

Neither module is imported here. They are named from YAML by dotted path, and
``cache.py`` in particular should not be imported by a run that only wants the
synthetic loader.
"""
