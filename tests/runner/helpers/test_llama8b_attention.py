# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU-only checks for launcher gating and atomic Turbo source preparation."""

import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "llama8b_attention", ROOT / "runner/helpers/hooks/train/pretrain/megatron/llama8b_attention.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config():
    return SimpleNamespace(
        framework="megatron", stage="mlperf_pretrain", num_layers=32,
        hidden_size=4096, num_attention_heads=32, num_query_groups=8,
        seq_length=8192, micro_batch_size=4, tensor_model_parallel_size=1,
        enable_primus_turbo=True, use_turbo_attention=True,
    )


class PreparationTests(unittest.TestCase):
    def test_recipe_gate(self):
        self.assertTrue(MODULE.eligible(config()))
        for key in vars(config()):
            with self.subTest(key=key):
                changed = config()
                setattr(changed, key, None)
                self.assertFalse(MODULE.eligible(changed))

    def test_disabled_does_not_locate_or_import_turbo(self):
        with patch.dict(os.environ, {"PRIMUS_LLAMA8B_ATTENTION_DS_LAYOUT": "0"}):
            with patch.object(MODULE.importlib.util, "find_spec") as locate:
                MODULE.prepare_attention(config())
                locate.assert_not_called()

    def test_other_recipe_does_not_locate_turbo(self):
        other = config()
        other.stage = "pretrain"
        with patch.object(MODULE.importlib.util, "find_spec") as locate:
            MODULE.prepare_attention(other)
            locate.assert_not_called()

    def test_missing_turbo_fails_for_selected_recipe(self):
        with patch.dict(os.environ, {"PRIMUS_LLAMA8B_ATTENTION_DS_LAYOUT": "1"}):
            with patch.object(MODULE.importlib.util, "find_spec", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "installed Primus-Turbo"):
                    MODULE.prepare_attention(config())

    def test_atomic_patch_idempotence_and_source_drift(self):
        original, changed = b"value = 1\n", b"value = 2\n"
        patch_text = (
            "diff --git a/primus_turbo/flydsl/attention/flash_attn_bwd.py "
            "b/primus_turbo/flydsl/attention/flash_attn_bwd.py\n"
            "--- a/primus_turbo/flydsl/attention/flash_attn_bwd.py\n"
            "+++ b/primus_turbo/flydsl/attention/flash_attn_bwd.py\n"
            "@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, diff = root / "source.py", root / "change.patch"
            source.write_bytes(original)
            source.chmod(0o640)
            diff.write_text(patch_text)
            with patch.multiple(
                MODULE, BASE_SHA256=hashlib.sha256(original).hexdigest(),
                PATCHED_SHA256=hashlib.sha256(changed).hexdigest(), PATCH=diff,
            ):
                self.assertEqual(MODULE.apply_layout(source), "applied")
                self.assertEqual(source.read_bytes(), changed)
                self.assertEqual(source.stat().st_mode & 0o777, 0o640)
                before = source.stat().st_mtime_ns
                self.assertEqual(MODULE.apply_layout(source), "already applied")
                self.assertEqual(source.stat().st_mtime_ns, before)
                source.write_bytes(b"unknown source\n")
                with self.assertRaisesRegex(RuntimeError, "Unqualified"):
                    MODULE.apply_layout(source)
                self.assertEqual(source.read_bytes(), b"unknown source\n")

    def test_wrong_result_hash_does_not_modify_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            source.write_bytes(b"original\n")
            with patch.multiple(MODULE, BASE_SHA256=hashlib.sha256(b"original\n").hexdigest()):
                with patch.object(MODULE.subprocess, "run"):
                    with self.assertRaisesRegex(RuntimeError, "Unexpected patched"):
                        MODULE.apply_layout(source)
            self.assertEqual(source.read_bytes(), b"original\n")


if __name__ == "__main__":
    unittest.main()
