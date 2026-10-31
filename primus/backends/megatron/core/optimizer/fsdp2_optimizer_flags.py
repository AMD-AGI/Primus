###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which Primus FSDP2 custom optimizers replace get_megatron_optimizer for a run.

skip_embedding_weight_decay needs this (those optimizers do not take Megatron's
config_overrides). The flag names have one definition: ``_FSDP2_OPTIMIZER_FLAGS`` in
``patches/args/rocm_arg_validation.py`` (the exclusivity check), imported on use. The
activation rule (flag set, with ``use_torch_fsdp2`` and ``bf16``) must match the
conditions in ``patches/optimizer_patches.py``;
``test_fsdp2_optimizer_rule_agrees_with_its_users`` checks that they agree.
"""


def _flags():
    # Imported on use: importing primus.backends.megatron.patches runs its auto-import of
    # every *_patches module, and embedding_weight_decay_patches imports this module.
    from primus.backends.megatron.patches.args.rocm_arg_validation import (
        _FSDP2_OPTIMIZER_FLAGS,
    )

    return tuple(_FSDP2_OPTIMIZER_FLAGS)


def active_fsdp2_optimizers(args):
    """The FSDP2 optimizer flags whose optimizer replaces get_megatron_optimizer for
    ``args``: the flag set, with ``use_torch_fsdp2`` and ``bf16``."""
    if not (getattr(args, "use_torch_fsdp2", False) and getattr(args, "bf16", False)):
        return []
    return [flag for flag in _flags() if getattr(args, flag, False)]
