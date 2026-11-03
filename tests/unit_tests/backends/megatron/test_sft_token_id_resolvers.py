###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SFT token-id lookups must survive a tokenizer without an EOS / BOS / PAD token.

Megatron's ``HuggingFaceTokenizer.eos_id`` raises ``TypeError`` when the HF tokenizer's
``eos_token`` is ``None`` (thinkingmachines/Inkling). The resolvers must try the next
attribute instead of failing the DataLoader worker.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from primus.backends.megatron.sft import preprocessing  # noqa: E402
from primus.backends.megatron.sft.preprocessing import (  # noqa: E402
    _resolve_bos_token_id,
    _resolve_eos_token_id,
    _resolve_pad_token_id,
)


class _NoSpecialTokens:
    """Megatron's HuggingFaceTokenizer over an HF tokenizer without EOS / BOS: the token strings
    are None and ``eos_id`` (no None guard) would raise TypeError if it were called."""

    eos_token = None
    bos_token = None
    pad = None
    pad_token_id = None

    def _raise(self):
        raise TypeError("'None' is not an instance of 'str'")

    eos_id = property(_raise)
    eos = property(_raise)
    bos = property(_raise)
    bos_id = property(_raise)
    eod = property(lambda self: None)


class _EodOnly:
    """Only an end-of-document id (a tokenizer with no special-token strings, such as Megatron's
    NullTokenizer): the earlier attributes are None, a later one is used."""

    pad = None
    pad_token_id = None
    eos = None
    eos_token_id = None
    eos_id = None
    eod = 200006


def test_resolvers_fall_back_when_token_lookup_raises():
    tok = _NoSpecialTokens()
    assert _resolve_pad_token_id(tok) == 0
    assert _resolve_eos_token_id(tok) is None
    assert _resolve_bos_token_id(tok) is None


def test_resolvers_use_a_later_attribute_after_absent_ones():
    tok = _EodOnly()
    assert _resolve_pad_token_id(tok) == 200006
    assert _resolve_eos_token_id(tok) == 200006


class _LegacyBase:
    """Megatron's legacy tokenizer base: unsupported special tokens raise NotImplementedError."""

    eos_token_id = 5

    @property
    def eos(self):
        raise NotImplementedError("EOS is not provided for this tokenizer")


def test_not_implemented_special_token_falls_through():
    assert _resolve_eos_token_id(_LegacyBase()) == 5


class _BrokenProperty:
    @property
    def eos(self):
        raise KeyError("a tokenizer bug, not a missing token")


def test_a_tokenizer_bug_is_not_swallowed():
    with pytest.raises(KeyError, match="tokenizer bug"):
        _resolve_eos_token_id(_BrokenProperty())


class _Wrapper:
    """Megatron's TextTokenizer: the token strings live on the wrapped library tokenizer."""

    def __init__(self):
        self._tokenizer = _NoSpecialTokens()

    @property
    def eos_id(self):
        return self._tokenizer.eos_id


class _WrapperWithEod:
    """A wrapper with a valid end-of-text id (e.g. a configured <|endoftext|>) over a library
    tokenizer whose eos_token is None: the valid id is used, not discarded."""

    def __init__(self):
        self._tokenizer = _NoSpecialTokens()

    eod = 7
    eos_id = property(lambda self: self._tokenizer.eos_id)


def test_a_valid_id_is_used_even_when_the_token_string_is_none():
    assert _resolve_eos_token_id(_WrapperWithEod()) == 7
    assert _resolve_pad_token_id(_WrapperWithEod()) == 7

    class _EosIdWithNoneString:
        eos_token = None
        eos_id = 9

    assert _resolve_eos_token_id(_EosIdWithNoneString()) == 9


def test_a_wrapped_tokenizer_without_eos_resolves_to_none():
    assert _resolve_eos_token_id(_Wrapper()) is None


class _TypeErrorBug:
    """A TypeError from a bug, while the EOS token string exists."""

    eos_token = "</s>"

    @property
    def eos(self):
        return 1 + "1"


def test_a_type_error_with_a_real_token_is_not_swallowed():
    with pytest.raises(TypeError):
        _resolve_eos_token_id(_TypeErrorBug())


@pytest.fixture
def logs(monkeypatch):
    lines = []
    monkeypatch.setattr(preprocessing, "_LOGGED_ONCE", set())
    monkeypatch.setattr(preprocessing, "log_rank_0", lines.append)
    return lines


def test_a_skipped_attribute_is_logged_once_not_per_sample(logs):
    tok = _NoSpecialTokens()
    for _ in range(100):
        _resolve_pad_token_id(tok)
        _resolve_eos_token_id(tok)
    absent = [line for line in logs if "treated as absent" in line]
    fallback = [line for line in logs if "padding uses token id 0" in line]
    # one line per skipped attribute and one pad-fallback warning, not one per call
    assert len(absent) == len(set(absent)) and 0 < len(absent) <= 5, logs
    assert all("eos_token is None" in line for line in absent)
    assert len(fallback) == 1 and len(logs) == len(absent) + 1


def test_a_tokenizer_class_is_warned_once_across_instances(logs):
    """The once-only log is per tokenizer class: the new tokenizer of each dataset / split does not
    warn again, and another class does."""

    class _Other(_NoSpecialTokens):
        pass

    for tok in (_NoSpecialTokens(), _NoSpecialTokens(), _Other()):
        for _ in range(3):
            _resolve_pad_token_id(tok)
    assert sum("padding uses token id 0" in line for line in logs) == 2


class _Slotted(_NoSpecialTokens):
    """Neither weak-referenceable nor hashable."""

    __slots__ = ()
    __hash__ = None


def test_the_warn_once_state_holds_no_tokenizer(logs):
    """No tokenizer is kept alive by the warn-once state (also one that is not hashable or
    weak-referenceable)."""
    import gc
    import weakref

    class _Weak(_NoSpecialTokens):
        pass

    tok = _Weak()
    ref = weakref.ref(tok)
    _resolve_pad_token_id(tok)
    _resolve_pad_token_id(_Slotted())
    del tok
    gc.collect()
    assert ref() is None
    assert all(
        isinstance(k, tuple) and all(isinstance(p, str) for p in k) for k in preprocessing._LOGGED_ONCE
    )
    assert sum("padding uses token id 0" in line for line in logs) == 2


class _Chars:
    """No EOS, no pad; one id per character (ord)."""

    eos_token = None
    pad_token = None

    def tokenize(self, text):
        return [ord(c) for c in text]


@pytest.mark.parametrize("inline_bos", [False, True])
def test_no_supervised_position_has_a_pad_label(monkeypatch, inline_bos, logs):
    """Without an EOS, the last sample token's label is the first pad (id 0 under the fallback):
    it must be loss-masked, or the model learns to emit the pad id after each response."""
    from primus.backends.megatron.sft.schema import FormattedSFTSample, TextSegment

    monkeypatch.setattr(preprocessing, "tokenize_text", lambda tok, text: [ord(c) for c in text])
    sample = FormattedSFTSample(
        segments=(TextSegment(text="ab", supervise=False), TextSegment(text="cd", supervise=True))
    )
    ids, labels, mask = preprocessing.tokenize_formatted_sft_sample(
        sample, _Chars(), 8, bridge_compat_inline_bos=inline_bos
    )
    assert list(ids) == [97, 98, 99, 100, 0, 0, 0, 0]
    assert list(labels[:4]) == [98, 99, 100, 0]
    assert list(mask) == [0, 0, 1, 0, 0, 0, 0, 0]  # "c" -> "d" is the only target


@pytest.mark.parametrize("inline_bos", [False, True])
def test_with_an_eos_pad_the_stop_label_is_kept(monkeypatch, inline_bos, logs):
    """A tokenizer with an EOS / EOD id pads with it: the last token's label is that stop token
    and stays a target, as before this port (only the pad-0 fallback masks it)."""
    from primus.backends.megatron.sft.schema import FormattedSFTSample, TextSegment

    class _Eod(_Chars):
        eod = 2

    monkeypatch.setattr(preprocessing, "tokenize_text", lambda tok, text: [ord(c) for c in text])
    sample = FormattedSFTSample(
        segments=(TextSegment(text="ab", supervise=False), TextSegment(text="cd", supervise=True))
    )
    ids, labels, mask = preprocessing.tokenize_formatted_sft_sample(
        sample, _Eod(), 8, bridge_compat_inline_bos=inline_bos
    )
    assert list(ids) == [97, 98, 99, 100, 2, 2, 2, 2] and list(labels[:4]) == [98, 99, 100, 2]
    # the same targets as the base commit (labels[3] = the stop token is trained); the inline-BOS
    # mode's appended EOS keeps its own (unshifted) mask slot, as at base
    expected = [0, 0, 1, 1, 1, 0, 0, 0] if inline_bos else [0, 0, 1, 1, 0, 0, 0, 0]
    assert list(mask) == expected


def test_the_packed_dataset_warns_without_eos_also_on_a_warm_cache(monkeypatch, logs):
    from primus.backends.megatron.sft import packing

    monkeypatch.setattr(
        packing, "_build_pack_cache_key", lambda **kw: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    with pytest.raises(RuntimeError, match="stop"):
        packing.PackedSFTDataset("d", _NoSpecialTokens(), 8, bridge_compat_inline_bos=True)
    assert sum("is not trained to stop" in line for line in logs) == 1


def test_no_eos_warning_for_an_empty_sample(monkeypatch, logs):
    from primus.backends.megatron.sft.schema import FormattedSFTSample

    preprocessing.tokenize_formatted_sft_sample(
        FormattedSFTSample(segments=()), _NoSpecialTokens(), 8, bridge_compat_inline_bos=True
    )
    assert not any("is not trained to stop" in line for line in logs)


def test_the_pad_fallback_to_0_is_warned(logs):
    assert _resolve_pad_token_id(_NoSpecialTokens()) == 0
    assert any("WARNING" in line and "padding uses token id 0" in line for line in logs)


class _RaisingWithoutEosString(_NoSpecialTokens):
    """No EOS string, but an EOS-id property that raises a TypeError from a bug (not from the
    missing token): the property is never called, so nothing is swallowed. A pad-id property
    whose token exists raises: that TypeError propagates."""

    pad_token = "<pad>"

    @property
    def pad(self):
        return 1 + "1"


def test_a_type_error_is_never_swallowed_even_without_eos():
    with pytest.raises(TypeError):
        _resolve_pad_token_id(_RaisingWithoutEosString())


def test_packed_inline_bos_without_eos_warns_once(logs, monkeypatch):
    """The packed-SFT twin (packing._tokenize_no_pad): one warning, no EOS appended."""
    from primus.backends.megatron.sft import packing
    from primus.backends.megatron.sft.schema import FormattedSFTSample, TextSegment

    monkeypatch.setattr(packing, "tokenize_text", lambda tok, text: [7] * len(text))
    sample = FormattedSFTSample(segments=(TextSegment(text="ab", supervise=True),))
    tok = _NoSpecialTokens()  # one tokenizer for every sample, as in a dataset
    for _ in range(5):
        out = packing._tokenize_no_pad(sample, tok, 8, bridge_compat_inline_bos=True)
    assert sum("has no EOS token id" in line for line in logs) == 1
    assert list(out["input_ids"]) == [7, 7]
    assert list(out["loss_mask"]) == [1, 1]


def test_inline_bos_without_eos_warns_once(logs, monkeypatch):
    from primus.backends.megatron.sft.schema import FormattedSFTSample, TextSegment

    monkeypatch.setattr(preprocessing, "tokenize_text", lambda tok, text: [7] * len(text))
    sample = FormattedSFTSample(segments=(TextSegment(text="ab", supervise=True),))
    tok = _NoSpecialTokens()  # one tokenizer for every sample, as in a dataset
    for _ in range(5):
        ids, _, mask = preprocessing.tokenize_formatted_sft_sample(
            sample, tok, 8, bridge_compat_inline_bos=True
        )
    warnings = [line for line in logs if "has no EOS token id" in line]
    assert len(warnings) == 1
    assert list(np.asarray(ids)[:3]) == [7, 7, 0]  # no EOS appended; pad falls back to 0 (warned)
    assert sum("padding uses token id 0" in line for line in logs) == 1
    assert list(np.asarray(mask)[:3]) == [1, 0, 0]  # the last token's label is the pad: not a target
