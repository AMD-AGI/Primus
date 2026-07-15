###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Registration-completeness tests for the real backend patches.

Importing the backend patch packages triggers every ``@register_patch``; we then
assert the registry is populated, ids are unique, and anchor patches are present.
Guards against a patch silently not being applied at training time (import
failure, decorator regression, id collision). Pure CPU - no GPU needed.
"""


def _registered_ids(prefix: str):
    from primus.core.patches.patch_registry import PatchRegistry

    return [pid for pid in PatchRegistry.list_ids() if pid.startswith(prefix)]


def test_megatron_patches_auto_register():
    import primus.backends.megatron.patches  # noqa: F401 (import triggers registration)

    ids = _registered_ids("megatron.")
    assert len(ids) >= 30, f"too few megatron patches registered ({len(ids)}); auto-import may have broken"
    assert len(ids) == len(set(ids)), "duplicate megatron patch ids in registry"
    # Long-lived core patches that must always be applied for Megatron training.
    for must in ("megatron.args.mock_data", "megatron.checkpoint.save_checkpoint"):
        assert must in ids, f"expected patch '{must}' is not registered"


def test_torchtitan_patches_auto_register():
    import primus.backends.torchtitan.patches  # noqa: F401

    ids = _registered_ids("torchtitan.")
    assert len(ids) >= 8, f"too few torchtitan patches registered ({len(ids)})"
    assert len(ids) == len(set(ids)), "duplicate torchtitan patch ids in registry"
