###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers shared by the native HyLo runtime, its replay dataset and its patches."""

from __future__ import annotations

from types import ModuleType


def unset(value) -> bool:
    """True for a config value left empty: ``None`` or a blank/``null``/``none`` string."""
    return value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none"))


def hybrid_pretrain_module() -> ModuleType:
    """Megatron-LM's hybrid-model training script, the one ``MegatronPretrainTrainer`` takes its
    dataset provider and ``get_batch`` from.

    Megatron-core 0.19 moved them to ``pretrain_hybrid``; its ``pretrain_mamba`` only forwards there.
    """
    import pretrain_mamba

    if hasattr(pretrain_mamba, "train_valid_test_datasets_provider"):
        return pretrain_mamba
    import pretrain_hybrid

    return pretrain_hybrid


# get_batch's order before Megatron-core 0.19; from 0.19 it returns its module's BATCH_KEYS order.
_LEGACY_BATCH_KEYS = ("tokens", "labels", "loss_mask", "attention_mask", "position_ids")


def hybrid_batch_getter():
    """``data_iterator -> (tokens, labels, loss_mask, attention_mask, position_ids)`` from the hybrid
    script's ``get_batch``."""
    module = hybrid_pretrain_module()
    keys = tuple(getattr(module, "BATCH_KEYS", _LEGACY_BATCH_KEYS))

    def get(data_iterator):
        batch = dict(zip(keys, module.get_batch(data_iterator)))
        return tuple(batch[key] for key in _LEGACY_BATCH_KEYS)

    return get
