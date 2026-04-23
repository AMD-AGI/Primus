###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-LM Source Patches

This package ships patches that are applied directly to the
``third_party/Megatron-LM`` source tree bundled with Primus. The patches
cover behaviour that cannot be cleanly expressed via Python-level
monkey-patching (e.g. inserting ``torch.cuda.synchronize()`` at multiple
points inside ``get_model``) and behaviour that was previously maintained
downstream in the MLPerf-training workspace.

The patches are ``.patch`` files (unified diff format, ``a/...`` vs
``b/...`` prefixes relative to the Megatron-LM repo root). See
``megatron_lm_source_patches.py`` for the runtime applier.
"""
