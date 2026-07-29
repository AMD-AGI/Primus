###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Mamba FLA-Order Dataset Patch
===============================

Wraps ``pretrain_mamba.train_valid_test_datasets_provider`` so that, when
enabled, training reads tokens through ``tools/hybrid/fla_order_dataset.py``
(``FLAOrderGPTDataset``) instead of Megatron's ``GPTDataset``. This replays
the exact same token order flash-linear-attention's (FLA) reference trainer
used, which is required for bit-for-bit loss-curve parity checks against FLA.

Toggle: ``args.use_fla_data`` + ``args.fla_cache_dir`` (resolved by
``fla_runtime_patches.py`` from ``PRIMUS_FLA_DATA`` / ``PRIMUS_FLA_CACHE_DIR``
or YAML ``use_fla_data`` / ``fla_cache_dir``).

``primus.backends.megatron.megatron_pretrain_trainer`` imports
``train_valid_test_datasets_provider`` by name from ``pretrain_mamba`` at
``train()`` time (well after ``before_train`` patches run), so replacing the
attribute on the ``pretrain_mamba`` module here is sufficient -- no need to
touch call sites.
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mamba.fla_order_dataset"


def _install_mamba_fla_data_patch() -> None:
    import pretrain_mamba

    if is_patched(pretrain_mamba, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] pretrain_mamba already patched; skipping.")
        return

    original_provider = pretrain_mamba.train_valid_test_datasets_provider

    def patched_provider(train_val_test_num_samples, vp_stage=None):
        from megatron.training import get_args as _get_args
        from megatron.training import print_rank_0

        args = _get_args()
        fla_data_flag = getattr(args, "use_fla_data", False)
        fla_cache = getattr(args, "fla_cache_dir", "")
        print_rank_0(f"> [FLA-check] use_fla_data={fla_data_flag!r}, fla_cache_dir={fla_cache!r}")
        if fla_data_flag and fla_cache:
            import importlib.util
            import os

            from megatron.core import parallel_state
            from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer

            _spec = importlib.util.spec_from_file_location(
                "fla_order_dataset",
                os.path.join(os.environ.get("PRIMUS_PATH", os.getcwd()), "tools", "hybrid", "fla_order_dataset.py"),
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            FLAOrderGPTDataset = _mod.FLAOrderGPTDataset

            dp_size = parallel_state.get_data_parallel_world_size()
            tokenizer = build_tokenizer(args)
            print_rank_0(f"> building FLA-order dataset from {fla_cache} ...")
            train_ds = FLAOrderGPTDataset(
                cache_dir=fla_cache,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                data_parallel_size=dp_size,
                seed=args.seed,
                pad_token_id=0,
                eod_token=tokenizer.eod,
                eod_mask_loss=args.eod_mask_loss,
            )
            print_rank_0(f"> FLA-order dataset: {len(train_ds)} samples")
            return train_ds, None, None
        return original_provider(train_val_test_num_samples, vp_stage=vp_stage)

    pretrain_mamba.train_valid_test_datasets_provider = patched_provider
    mark_patched(pretrain_mamba, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] Patched pretrain_mamba.train_valid_test_datasets_provider "
        "to serve FLAOrderGPTDataset when use_fla_data is set."
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Serve tokens through FLAOrderGPTDataset (tools/hybrid/fla_order_dataset.py) "
        "instead of Megatron's GPTDataset, to replay FLA's exact reference token order."
    ),
    # Runs after fla_runtime_knobs (priority=-100) has resolved args.use_fla_data / fla_cache_dir.
    priority=50,
    condition=lambda ctx: bool(getattr(get_args(ctx), "use_fla_data", False))
    and bool(getattr(get_args(ctx), "fla_cache_dir", "")),
)
def patch_mamba_fla_data(ctx: PatchContext) -> None:
    _install_mamba_fla_data_patch()
