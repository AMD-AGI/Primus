###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus-managed, bridge-free native HuggingFace -> Megatron-Core checkpoint
converters.

These modules move the previously in-submodule ``tools/checkpoint/loader_*.py``
logic into Primus so the Megatron-LM submodule stays pristine. Each converter
runs entirely in a *single process*: it builds the mcore ``GPTModel`` with the
same builder Primus training uses (``get_model_provider`` ->
``model_provider(gpt_builder)``), copies HF ``safetensors`` weights directly onto
those parameters, and saves a ``torch_dist`` checkpoint that Megatron reshards
to any TP / PP / EP on load (or a legacy TP1/PP1 ``torch`` checkpoint if asked).
No ``megatron.bridge`` / ``AutoBridge`` / ``modelopt`` is ever imported.
"""
