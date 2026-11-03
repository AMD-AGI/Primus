# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the mock_data args patch:
- vocab_size comes from the tokenizer_model HF config (minus NullTokenizer's EOD token)
- explicit vocab_size is preserved
- falls back to DEFAULT_NULL_TOKENIZER_VOCAB_SIZE when the config cannot be read
- on Megatron-LM main, whose NullTokenizer counts EOD inside vocab_size, one id is added
  back so both versions build the same tokenizer
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


@pytest.fixture(params=[False, True], ids=["eod_on_top", "eod_inside"])
def eod_inside(request, monkeypatch):
    monkeypatch.setattr(mock_data_patches, "_null_tokenizer_vocab_includes_eod", lambda: request.param)
    return request.param


def _older_megatron_value(vocab_size, eod_inside):
    """The vocab_size older Megatron would have been handed for the same tokenizer."""
    return vocab_size - 1 if eod_inside else vocab_size


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


def test_vocab_size_from_tokenizer_config(tmp_path, eod_inside):
    args = _run(_hf_dir(tmp_path, {"vocab_size": 32000}))
    assert _older_megatron_value(args.vocab_size, eod_inside) == 31999
    assert args.tokenizer_type == "NullTokenizer"
    assert args.data_path is None


def test_vocab_size_from_nested_text_config(tmp_path, eod_inside):
    args = _run(_hf_dir(tmp_path, {"text_config": {"vocab_size": 152064}}))
    assert _older_megatron_value(args.vocab_size, eod_inside) == 152063


def test_explicit_vocab_size_preserved(tmp_path, eod_inside):
    args = _run(_hf_dir(tmp_path, {"vocab_size": 32000}), vocab_size=50000)
    assert _older_megatron_value(args.vocab_size, eod_inside) == 50000


@pytest.mark.parametrize("tokenizer_model", [None, "/nonexistent/tokenizer"])
def test_fallback_default_vocab_size(tokenizer_model, eod_inside):
    args = _run(tokenizer_model)
    assert (
        _older_megatron_value(args.vocab_size, eod_inside)
        == mock_data_patches.DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
    )


def test_detects_the_installed_null_tokenizer():
    pytest.importorskip("megatron.core.tokenizers.text.libraries.null_tokenizer")
    from megatron.core.tokenizers.text.libraries.null_tokenizer import NullTokenizer

    tokenizer = NullTokenizer(1000)
    expected_total = 1000 if mock_data_patches._null_tokenizer_vocab_includes_eod() else 1001
    assert tokenizer.vocab_size == expected_total
    assert tokenizer.eod == expected_total - 1
