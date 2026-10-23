# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the mock_data args patch:
- vocab_size comes from the tokenizer_model HF config (minus NullTokenizer's EOD token)
- explicit vocab_size is preserved
- falls back to DEFAULT_NULL_TOKENIZER_VOCAB_SIZE when the config cannot be read
"""

import json
from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches.args import mock_data_patches
from primus.core.patches.context import PatchContext


@pytest.fixture(autouse=True)
def _quiet_logs(monkeypatch):
    for name in ("log_rank_0", "log_kv_rank_0", "warning_rank_0"):
        monkeypatch.setattr(mock_data_patches, name, lambda *a, **k: None)


def _run(tokenizer_model, vocab_size=None):
    args = SimpleNamespace(tokenizer_model=tokenizer_model, vocab_size=vocab_size, data_path=["x"])
    module_config = SimpleNamespace(params=SimpleNamespace(mock_data=True))
    ctx = PatchContext(
        backend="megatron",
        phase="build_args",
        extra={"backend_args": args, "module_config": module_config},
    )
    mock_data_patches.patch_mock_data(ctx)
    return args


def _hf_dir(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_vocab_size_from_tokenizer_config(tmp_path):
    args = _run(_hf_dir(tmp_path, {"vocab_size": 32000}))
    assert args.vocab_size == 31999
    assert args.tokenizer_type == "NullTokenizer"
    assert args.data_path is None


def test_vocab_size_from_nested_text_config(tmp_path):
    args = _run(_hf_dir(tmp_path, {"text_config": {"vocab_size": 152064}}))
    assert args.vocab_size == 152063


def test_explicit_vocab_size_preserved(tmp_path):
    args = _run(_hf_dir(tmp_path, {"vocab_size": 32000}), vocab_size=50000)
    assert args.vocab_size == 50000


@pytest.mark.parametrize("tokenizer_model", [None, "/nonexistent/tokenizer"])
def test_fallback_default_vocab_size(tokenizer_model):
    args = _run(tokenizer_model)
    assert args.vocab_size == mock_data_patches.DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
