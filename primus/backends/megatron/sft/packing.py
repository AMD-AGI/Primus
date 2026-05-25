###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Sequence packing for Megatron-native SFT.

Concatenates multiple short instruction-tuning samples into one fixed-length
sequence and emits per-segment ``cu_seqlens`` that are consumed by attention
kernels (``PackedSeqParams.qkv_format='thd'``) so attention does not leak across
samples. This dramatically increases real-token throughput on datasets where
the average sample is much shorter than ``max_seq_length`` (e.g. Alpaca ~80 vs
seq_length=4096 -> ~98% of FLOPs are on padding without packing).

The dataset is built offline at construction time:

    raw samples -> tokenize (no padding) -> first-fit-decreasing pack ->
    sequence buckets, each padded to max_seq_length and tagged with its
    sub-segment cu_seqlens.

Each ``__getitem__`` returns a single packed sequence so that PyTorch's
default_collate can stack ``micro_batch_size`` of them in the standard way.
``forward_step`` is responsible for stitching per-sample cu_seqlens into a
batch-global ``PackedSeqParams``.

Persistent on-disk cache
------------------------
The tokenize+bin-pack step is expensive (~71 s tokenize + ~15 s pack for
SQuAD 87599 samples at seq=2048) and used to be repeated on every rank on
every process restart, costing 13+ minutes of wall-clock in the worst case
(observed: ``train/valid/test-data-iterators-setup`` timer min=90 s /
max=783 s on run native_llama2_70b_faststart_20260513_105737).

``PackedSFTDataset.__init__`` now saves the materialized ``self._packed``
to a per-config ``.pt`` file. A hashed cache key covers every input that
affects the pack content (dataset name, split, formatter, max_seq_length,
pad_id, tokenizer identity, plus ``PACK_FORMAT_VERSION`` so a code change
forces a rebuild). Concurrent ranks serialize through a ``filelock``:
the first rank to acquire the lock builds + saves under the lock; the
remaining ranks find the file on disk and load it.

Where the cache lives:
    1. ``$PRIMUS_PACK_CACHE_DIR`` if set
    2. ``$HF_DATASETS_CACHE/primus_packed`` (default in our launcher,
       which routes to ``/workspace/cache_persist/hf_cache/datasets``)
    3. ``$HF_HOME/primus_packed``
    4. ``~/.cache/primus_packed_sft``

Disable entirely with ``PRIMUS_DISABLE_PACK_CACHE=1`` (falls back to
the old per-run rebuild path).
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from primus.backends.megatron.sft.formatters import create_formatter
from primus.backends.megatron.sft.preprocessing import (
    _resolve_eos_token_id,
    _resolve_pad_token_id,
    log_rank_0,
    normalize_sft_sample,
    tokenize_text,
)
from primus.backends.megatron.sft.schema import FormattedSFTSample


# Bump whenever the on-disk pack layout changes (output dict keys, dtype of any
# tensor, or the packing algorithm itself). Old caches with a different version
# tag are ignored because the key hash includes this constant.
#
# v1                  -- Independent padding-terminator segment (legacy).
#                        Worked for GQA models (Llama2/3, Qwen3) because their
#                        RoPE fuses into TE's thd kernel and tolerates one
#                        extra trailing zero-length-effective segment.
#                        BROKE MLA (DeepSeek-V2/-V3) which applies RoPE
#                        outside the fused kernel and demands
#                        len(cu_seqlens) == num_real_segments + 1.
# v2_merge_padding    -- Bridge-aligned: trailing padding is merged into the
#                        last real segment (cu_seqlens[-1] = max_seq_length
#                        in place). Matches NVIDIA upstream Megatron-LM
#                        sft_dataset and works for MLA + GQA uniformly.
PACK_FORMAT_VERSION = "v6_bridge_parity_squad_template_drop_overlength"


# Maximum number of sub-sequences allowed inside a single packed sequence.
# Sized for the worst case: very short Alpaca samples (~30 tokens) inside a
# long sequence window (8K seq_length -> ~270 sub-segments). 256 is a safe
# generous cap that still keeps cu_seqlens tiny per pack (256 * 4B = 1KB).
# A fixed cap is required so default_collate can stack cu_seqlens into
# [batch, MAX_SEGMENTS+1].
#
# With the v2_merge_padding scheme each pack carries num_real_segments
# segments total (the trailing padding is absorbed into the last real
# segment, NOT a separate terminator). The FFD packer caps raw samples at
# (MAX_SEGMENTS_PER_PACK - 1) anyway to stay strictly tighter than the
# legacy layout.
MAX_SEGMENTS_PER_PACK = 256


def _tokenize_no_pad(
    formatted_sample: FormattedSFTSample,
    tokenizer,
    max_seq_length: int,
    bridge_compat_inline_bos: bool = False,
) -> Dict[str, np.ndarray]:
    """Tokenize a single formatted sample WITHOUT padding to max_seq_length.

    When ``bridge_compat_inline_bos=False`` (default) we tokenize the whole
    sample text in one shot and derive loss_mask by prefix-incrementally
    re-tokenizing; this is the clean Native path that yields a stream with
    exactly one BOS at the start (if any) and no inline BOS/EOS noise.

    When ``bridge_compat_inline_bos=True`` we instead tokenize each
    ``FormattedSFTSample`` segment independently. Megatron's
    ``TextTokenizer.tokenize`` delegates to ``HuggingFaceTokenizer.text_to_ids``
    which (with the default ``include_special_tokens=True``) calls
    ``self.tokenizer(text).input_ids`` -- exactly equivalent to
    ``tokenizer.encode(text, add_special_tokens=True)`` -- so the returned ids
    already carry one leading BOS per segment. That is the same per-segment
    inline BOS that NeMo Megatron-Bridge bakes into its packed parquet files
    (see ``fill_packing_strategy`` + ``_separate_template`` upstream), so we
    do NOT prepend another BOS by hand; doing so would yield ``<s><s>`` at
    each segment boundary and inflate iter-1 lm-loss by ~2 nats relative to
    Bridge. We additionally append a sample-final EOS so the supervised tail
    matches Bridge's ``add_eos=True`` default. This mode exists *only* for
    numerical A/B comparisons against Bridge runs; it is not the recommended
    default because every inline BOS is an OOD token that Llama-2 never saw
    inside a prompt during pretraining, which is exactly what inflates
    Bridge's iter-1 loss to ~4.3 vs Native's ~0.15 on the same SQuAD recipe.
    """
    if not bridge_compat_inline_bos:
        text = formatted_sample.text
        token_ids = tokenize_text(tokenizer, text)
        if len(token_ids) > max_seq_length:
            token_ids = token_ids[:max_seq_length]

        loss_mask = np.zeros(len(token_ids), dtype=np.int64)
        prefix_text = ""
        prefix_token_count = 0
        for segment in formatted_sample.segments:
            start = prefix_token_count
            prefix_text += segment.text
            prefix_token_count = len(tokenize_text(tokenizer, prefix_text))
            end = prefix_token_count

            if segment.supervise and start < len(token_ids):
                loss_mask[start:min(end, len(token_ids))] = 1
            if start >= len(token_ids):
                break

        return {
            "input_ids": np.asarray(token_ids, dtype=np.int64),
            "loss_mask": loss_mask,
            "length": len(token_ids),
        }

    eos_id = _resolve_eos_token_id(tokenizer)

    token_ids: List[int] = []
    loss_mask_list: List[int] = []

    # Per-segment tokenize. ``tokenize_text`` routes through
    # ``TextTokenizer.tokenize`` -> ``HuggingFaceTokenizer.text_to_ids`` which
    # already adds a BOS at the start of every standalone segment, so we do
    # NOT manually prepend ``bos_id`` here -- doubling up would emit
    # ``<s><s>`` and shift Native's iter-1 loss ~2 nats above Bridge.
    for segment in formatted_sample.segments:
        seg_ids = list(tokenize_text(tokenizer, segment.text))
        mask_val = 1 if segment.supervise else 0
        token_ids.extend(seg_ids)
        loss_mask_list.extend([mask_val] * len(seg_ids))

    if eos_id is not None and formatted_sample.segments:
        tail_supervised = bool(formatted_sample.segments[-1].supervise)
        token_ids.append(eos_id)
        loss_mask_list.append(1 if tail_supervised else 0)

    # NOTE: do NOT truncate to max_seq_length here in Bridge-compat mode.
    # The caller (``_build_packs_in_memory``) drops over-length samples
    # outright -- the same filter Bridge applies via ``create_hist`` --
    # which requires us to expose the raw length here.

    return {
        "input_ids": np.asarray(token_ids, dtype=np.int64),
        "loss_mask": np.asarray(loss_mask_list, dtype=np.int64),
        "length": len(token_ids),
    }

## 把多个真样本压进一条序列。把多个真样本物理拼接，cu_seqlens 记录每段边界，只有最末尾装不下时才用 pad_id 填满到 seq_length
def _first_fit_pack(
    samples: List[Dict[str, np.ndarray]],
    max_seq_length: int,
    order: List[int],
) -> List[List[int]]:
    """Greedy first-fit bin packing in ``order``.

    Returns a list of bins; each bin is a list of indices into ``samples``.
    Within a bin the total token length is <= max_seq_length. The traversal
    order is supplied by the caller; see ``_first_fit_decreasing_pack`` and
    ``_first_fit_shuffle_pack`` for the two policies we currently expose.
    """
    raw_sample_cap = MAX_SEGMENTS_PER_PACK - 1
    bins: List[Dict[str, Any]] = []  # each: {"indices": [...], "used": int}
    for idx in order:
        length = samples[idx]["length"]
        if length == 0:
            continue
        placed = False
        for b in bins:
            if (
                b["used"] + length <= max_seq_length
                and len(b["indices"]) < raw_sample_cap
            ):
                b["indices"].append(idx)
                b["used"] += length
                placed = True
                break
        if not placed:
            bins.append({"indices": [idx], "used": length})
    return [b["indices"] for b in bins]


def _first_fit_decreasing_pack(
    samples: List[Dict[str, np.ndarray]],
    max_seq_length: int,
) -> List[List[int]]:
    """Native default packer.

    Sort samples by length descending, then first-fit. Yields packs whose
    per-pack length variance is minimised (early packs cluster long samples,
    late packs cluster short samples). Produces tighter total bin count than
    ``first_fit_shuffle`` but skews iter-1 statistics because early packs --
    which feed iter-1 -- only contain the longest samples.
    """
    # Cap real samples per bin at MAX_SEGMENTS_PER_PACK - 1. With the v2
    # merge-padding scheme there is no longer a separate trailing terminator,
    # so MAX_SEGMENTS_PER_PACK itself would be safe; the -1 margin is kept
    # only so this change is strictly tighter than legacy bins and never
    # increases per-pack segment count.
    order = sorted(range(len(samples)), key=lambda i: -samples[i]["length"])
    return _first_fit_pack(samples, max_seq_length, order)


def _first_fit_shuffle_pack(
    samples: List[Dict[str, np.ndarray]],
    max_seq_length: int,
    seed: int = 0,
) -> List[List[int]]:
    """Bridge-parity packer.

    Random-shuffle samples once with ``seed`` and then first-fit. Yields packs
    of mixed-length samples, matching NeMo Megatron-Bridge's default
    ``packing_algorithm="first_fit_shuffle"`` (megatron/bridge/data/datasets/
    packed_sequence.py L88). Use this together with
    ``sft_bridge_compat_inline_bos=True`` to align Native and Bridge iter-1
    sample composition.
    """
    rng = np.random.default_rng(seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    return _first_fit_pack(samples, max_seq_length, order)


def _build_packed_sequence(
    sample_indices: List[int],
    samples: List[Dict[str, np.ndarray]],
    max_seq_length: int,
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """Concatenate the chosen samples into one max_seq_length sequence.

    cu_seqlens layout (Bridge-aligned merge-padding, ``v2_merge_padding``):
      * Real sub-segments: cu_seqlens[0..num_real_segments] cover the actual
        packed samples (cu_seqlens[i+1] - cu_seqlens[i] == sample length).
      * Trailing pad region: merged into the LAST real segment by rewriting
        ``cu_seqlens[-1] = max_seq_length`` IN PLACE (no separate terminator
        segment is appended). This matches NVIDIA's upstream Megatron-LM
        ``sft_dataset.py`` (``cu_seqlens[-1] = len(pack_tokens) - 1``) and
        is required for MLA models (DeepSeek-V2 / -V3): MLA applies RoPE
        outside TE's fused thd kernel and assumes
        ``len(cu_seqlens) == num_real_segments + 1``; an extra terminator
        produces a "size of tensor a (max_seq_length) must match tensor b
        (real_tokens)" runtime error.
      * cu_seqlens is then padded with ``max_seq_length`` (zero-length dummy
        entries) up to a fixed length so default_collate can stack it.
    """
    input_ids = np.full(max_seq_length, pad_id, dtype=np.int64)
    labels = np.full(max_seq_length, pad_id, dtype=np.int64)
    loss_mask = np.zeros(max_seq_length, dtype=np.int64)
    position_ids = np.zeros(max_seq_length, dtype=np.int64)

    cu_seqlens = [0]
    offset = 0
    for idx in sample_indices:
        sample = samples[idx]
        length = sample["length"]
        if offset + length > max_seq_length:
            length = max_seq_length - offset
            if length <= 0:
                break

        input_ids[offset:offset + length] = sample["input_ids"][:length]
        # Next-token prediction: ``labels[i] = input_ids[i+1]`` inside each
        # sub-segment. Megatron's ``compute_language_model_loss`` does NOT
        # internally shift labels (it computes CE between logits[t] and
        # labels[t] directly), so the dataset MUST emit shifted labels.
        # Bridge's packed-SFT collate (sft.py:920) does the same shift via
        # ``input_ids[boundaries[i]+1 : boundaries[i+1]]``. Without this shift
        # the model is asked to predict the current token from its own
        # representation, which yields random-init-level loss (~13.6 instead
        # of ~4.3 for Llama-2 on SQuAD).
        if length >= 2:
            labels[offset:offset + length - 1] = sample["input_ids"][1:length]
        # ``labels[offset + length - 1]`` is undefined (no next token inside
        # this sub-segment); it stays as ``pad_id`` and is masked out below.

        # ``loss_mask`` must also shift to align with the shifted labels:
        # ``loss_mask[i] = sample.loss_mask[i+1]`` so that we compute CE only
        # at positions whose prediction target is a supervised (response)
        # token. Mirrors Bridge ``loss_mask = item["loss_mask"][1:]`` in
        # sft.py:1219.
        if length >= 2:
            loss_mask[offset:offset + length - 1] = sample["loss_mask"][1:length]
        # The last position of every sub-segment has no in-segment next-token
        # target -- mask it out so it never contributes to the loss.
        loss_mask[offset + length - 1] = 0
        # Per-segment positional IDs (each sub-segment starts from 0)
        position_ids[offset:offset + length] = np.arange(length, dtype=np.int64)

        offset += length
        cu_seqlens.append(offset)

    real_tokens = offset  # number of REAL (non-pad) tokens
    num_real_segments = len(cu_seqlens) - 1

    # Merge trailing padding into the LAST real segment instead of appending
    # a separate terminator (Bridge / NVIDIA upstream convention). Padding
    # tokens keep loss_mask=0, so they never contribute to the loss; thd
    # attention is allowed to span the padding tail inside the last segment,
    # exactly as on the Bridge backend.
    if offset < max_seq_length:
        pad_len = max_seq_length - offset
        # Padding region: input_ids/labels already filled with pad_id, loss_mask
        # already zero. Position_ids must be sequential to avoid RoPE oddity.
        position_ids[offset:max_seq_length] = np.arange(pad_len, dtype=np.int64)
        if num_real_segments > 0:
            cu_seqlens[-1] = max_seq_length
        else:
            # Degenerate empty-pack case (not produced by the FFD packer in
            # practice, since zero-length samples are dropped upstream).
            cu_seqlens.append(max_seq_length)
            num_real_segments = 1

    num_segments = len(cu_seqlens) - 1  # bridge-aligned: == num_real_segments
    assert num_segments <= MAX_SEGMENTS_PER_PACK, (
        f"Pack contains {num_segments} segments which exceeds "
        f"MAX_SEGMENTS_PER_PACK={MAX_SEGMENTS_PER_PACK}; raise the cap."
    )
    assert cu_seqlens[-1] == max_seq_length, (
        f"cu_seqlens[-1]={cu_seqlens[-1]} != max_seq_length={max_seq_length}"
    )

    # Pad cu_seqlens to fixed size by repeating ``max_seq_length`` (zero-length
    # dummy entries accepted by TE; repeating the final offset would produce
    # nonzero-but-invalid segments).
    while len(cu_seqlens) < MAX_SEGMENTS_PER_PACK + 1:
        cu_seqlens.append(max_seq_length)

    # Compute max real sub-segment length for this packed sequence (for FlashAttn).
    diffs = [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(num_real_segments)]
    max_sub_seqlen = max(diffs) if diffs else 0

    return {
        "input_ids": torch.from_numpy(input_ids),
        "labels": torch.from_numpy(labels),
        "loss_mask": torch.from_numpy(loss_mask),
        "position_ids": torch.from_numpy(position_ids),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
        "num_segments": torch.tensor(num_segments, dtype=torch.int32),
        "num_real_segments": torch.tensor(num_real_segments, dtype=torch.int32),
        "real_tokens": torch.tensor(real_tokens, dtype=torch.int32),
        "max_sub_seqlen": torch.tensor(max_sub_seqlen, dtype=torch.int32),
    }


def _resolve_pack_cache_dir() -> Path:
    """Return (and create) the directory used to cache packed SFT datasets.

    Resolution order:
        1. ``$PRIMUS_PACK_CACHE_DIR`` (explicit user override)
        2. ``$HF_DATASETS_CACHE/primus_packed`` (default in our launcher,
           which routes to ``/workspace/cache_persist/hf_cache/datasets``
           via ``run_pretrain.sh``)
        3. ``$HF_HOME/primus_packed``
        4. ``~/.cache/primus_packed_sft``
    """
    base = os.environ.get("PRIMUS_PACK_CACHE_DIR")
    if not base:
        for env_var in ("HF_DATASETS_CACHE", "HF_HOME"):
            root = os.environ.get(env_var)
            if root:
                base = os.path.join(root, "primus_packed")
                break
    if not base:
        base = os.path.expanduser("~/.cache/primus_packed_sft")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tokenizer_identity(tokenizer) -> str:
    """Best-effort stable fingerprint of a tokenizer for cache-key hashing.

    Two different tokenizers (e.g. Llama-2 vocab=32000 vs Qwen3 vocab=152064)
    produce completely different token-id streams from the same dataset, so
    the cache key MUST distinguish them. We probe several attribute names in
    decreasing order of specificity:
        * ``name_or_path``  -- HF AutoTokenizer-style, e.g.
          "NousResearch/Llama-2-70b-hf"
        * ``tokenizer_model_name`` -- Megatron's MegatronTokenizer
        * ``vocab_file``    -- sentencepiece-backed tokenizers
        * ``vocab_size`` + class name -- last-resort generic fingerprint

    Users can override with ``$PRIMUS_PACK_TOKENIZER_ID=<str>`` if their
    tokenizer is dynamically built (no stable attribute) but they want to
    pin a specific cache key for a given run.
    """
    override = os.environ.get("PRIMUS_PACK_TOKENIZER_ID")
    if override:
        return override
    for attr in ("name_or_path", "tokenizer_model_name", "vocab_file"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, str) and value:
            return value
    vocab_size = getattr(tokenizer, "vocab_size", None)
    cls_name = type(tokenizer).__name__
    return f"{cls_name}_vocab{vocab_size}"


def _build_pack_cache_key(
    *,
    dataset_name: str,
    split: str,
    formatter: str,
    max_seq_length: int,
    pad_id: int,
    tokenizer_id: str,
    bridge_compat_inline_bos: bool = False,
) -> str:
    """Return a 16-char hex digest that uniquely identifies a pack output."""
    pieces = [
        f"version={PACK_FORMAT_VERSION}",
        f"max_segments={MAX_SEGMENTS_PER_PACK}",
        f"dataset={dataset_name}",
        f"split={split}",
        f"formatter={formatter}",
        f"max_seq_length={max_seq_length}",
        f"pad_id={pad_id}",
        f"tokenizer={tokenizer_id}",
        f"bridge_compat_inline_bos={int(bool(bridge_compat_inline_bos))}",
    ]
    blob = "|".join(pieces).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class PackedSFTDataset(Dataset):
    """SFT dataset whose ``__getitem__`` returns a packed multi-sample sequence.

    Heavy tokenize+bin-pack work is cached to disk; see module docstring for
    cache directory resolution and the ``PRIMUS_DISABLE_PACK_CACHE`` /
    ``PRIMUS_PACK_TOKENIZER_ID`` env-var escape hatches.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_seq_length: int,
        split: str = "train",
        formatter: str = "alpaca",
        seed: int = 1234,
        bridge_compat_inline_bos: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.formatter_name = formatter
        self.formatter = create_formatter(formatter)
        self.pad_id = _resolve_pad_token_id(tokenizer)
        self.bridge_compat_inline_bos = bool(bridge_compat_inline_bos)
        if self.bridge_compat_inline_bos:
            log_rank_0(
                "[Pack] bridge_compat_inline_bos=True: per-segment tokenize "
                "with inline BOS + trailing EOS to mirror NeMo Megatron-Bridge's "
                "packed parquet layout. This intentionally inflates iter-1 loss "
                "(BOS inside the prompt is OOD for Llama-2) and exists only for "
                "Bridge A/B comparisons."
            )

        cache_disabled = os.environ.get("PRIMUS_DISABLE_PACK_CACHE", "0") not in ("0", "", "false", "False")
        if cache_disabled:
            log_rank_0("[Pack] Cache disabled via PRIMUS_DISABLE_PACK_CACHE; rebuilding from scratch.")
            self._packed = self._build_packs_in_memory(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                split=split,
                formatter=formatter,
                seed=seed,
                **kwargs,
            )
            return

        cache_key = _build_pack_cache_key(
            dataset_name=dataset_name,
            split=split,
            formatter=formatter,
            max_seq_length=max_seq_length,
            pad_id=self.pad_id,
            tokenizer_id=_tokenizer_identity(tokenizer),
            bridge_compat_inline_bos=self.bridge_compat_inline_bos,
        )
        cache_dir = _resolve_pack_cache_dir()
        cache_file = cache_dir / f"sft_pack_{cache_key}.pt"
        lock_file = cache_dir / f"sft_pack_{cache_key}.lock"

        # Lazy import keeps filelock optional for non-cache code paths (and out
        # of unit tests that don't exercise PackedSFTDataset).
        try:
            from filelock import FileLock
        except ImportError:
            log_rank_0(
                "[Pack] filelock not installed; falling back to "
                "uncoordinated rebuild on each rank. "
                "`pip install filelock` to enable the shared on-disk cache."
            )
            self._packed = self._build_packs_in_memory(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                split=split,
                formatter=formatter,
                seed=seed,
                **kwargs,
            )
            return

        with FileLock(str(lock_file)):
            if cache_file.exists():
                log_rank_0(
                    f"[Pack] CACHE HIT key={cache_key} ({cache_file.name}); "
                    "loading packed dataset from disk."
                )
                self._packed = torch.load(cache_file, weights_only=False)
                log_rank_0(
                    f"[Pack] Loaded {len(self._packed)} packed sequences from cache."
                )
                return

            log_rank_0(
                f"[Pack] CACHE MISS key={cache_key} ({cache_file}); "
                "building packed dataset under filelock."
            )
            self._packed = self._build_packs_in_memory(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                split=split,
                formatter=formatter,
                seed=seed,
                **kwargs,
            )
            # Atomic rename to avoid leaving a half-written file behind if the
            # process dies mid-save (e.g. user Ctrl-C during the torch.save
            # call). Concurrent ranks waiting on the filelock will observe
            # either the absent file or the complete file -- never a partial.
            tmp_file = cache_file.with_suffix(".pt.tmp")
            torch.save(self._packed, tmp_file)
            os.replace(tmp_file, cache_file)
            log_rank_0(f"[Pack] Cached packed dataset to {cache_file}")

    def _build_packs_in_memory(
        self,
        dataset_name: str,
        tokenizer,
        max_seq_length: int,
        split: str,
        formatter: str,
        seed: int,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """Tokenize + first-fit-decreasing bin-pack + materialize to tensors.

        Pulled out of ``__init__`` so the cache hit / cache miss / cache
        disabled branches can all reuse exactly the same construction path.
        """
        # Reuse SFTDataset's data loading path (HF hub or local file).
        from primus.backends.megatron.sft.dataset import SFTDataset

        base = SFTDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split=split,
            formatter=formatter,
            seed=seed,
            **kwargs,
        )
        # Bypass SFTDataset.__getitem__ (which pads) and read the underlying HF
        # dataset directly to tokenize without padding.
        raw_dataset = base.dataset
        log_rank_0(f"[Pack] Tokenizing {len(raw_dataset)} samples for packing...")

        tokenized: List[Dict[str, np.ndarray]] = []
        dropped_too_long = 0
        for i in range(len(raw_dataset)):
            sample = normalize_sft_sample(raw_dataset[i])
            formatted = self.formatter.format_sample(sample)
            tok = _tokenize_no_pad(
                formatted,
                tokenizer,
                max_seq_length,
                bridge_compat_inline_bos=self.bridge_compat_inline_bos,
            )
            if tok["length"] <= 0:
                continue
            # Bridge-parity dataset filter: NeMo Megatron-Bridge's
            # ``create_hist`` (packing_utils.py L122-139) silently drops any
            # sample whose tokenized length exceeds ``truncate_seq_len`` --
            # the histogram loop only iterates ``range(truncate_seq_len+1)``
            # so longer samples never enter ``create_packing_strategy``.
            # Native's default behaviour is to *truncate* long samples
            # rather than drop them, which leaves long-context QA in iter-1
            # batches that Bridge would never see, inflating Native iter-1
            # lm-loss vs Bridge. Mirror Bridge's drop policy here when
            # bridge-compat mode is on.
            if self.bridge_compat_inline_bos and tok["length"] > max_seq_length:
                dropped_too_long += 1
                continue
            tokenized.append(tok)
        if self.bridge_compat_inline_bos and dropped_too_long > 0:
            log_rank_0(
                f"[Pack] Dropped {dropped_too_long} over-length samples "
                f"(>{max_seq_length} tokens) to match Bridge's create_hist filter."
            )

        # When emulating NeMo Megatron-Bridge byte-for-byte, also align the
        # packing algorithm: Bridge defaults to ``first_fit_shuffle``
        # (megatron/bridge/data/datasets/packed_sequence.py L88) whose random
        # traversal makes every pack contain length-mixed samples. Native's
        # default ``first_fit_decreasing`` instead clusters long samples into
        # the first few packs and short samples into the tail, which makes
        # iter-1 (first 16 packs of the global epoch) operate on a very
        # different sample distribution -- one of the residual sources of
        # iter-1 lm-loss skew vs Bridge.
        if self.bridge_compat_inline_bos:
            log_rank_0(
                f"[Pack] Bin-packing {len(tokenized)} samples with "
                f"first_fit_shuffle (Bridge-parity, seed={seed}); "
                f"max_seq_length={max_seq_length}..."
            )
            bins = _first_fit_shuffle_pack(tokenized, max_seq_length, seed=seed)
        else:
            log_rank_0(
                f"[Pack] Bin-packing {len(tokenized)} samples with "
                f"first_fit_decreasing (Native default); "
                f"max_seq_length={max_seq_length}..."
            )
            bins = _first_fit_decreasing_pack(tokenized, max_seq_length)

        # Materialize each pack now so __getitem__ is just a list index.
        # Memory cost is small (packed_samples * max_seq_length * 4 bytes).
        packed: List[Dict[str, torch.Tensor]] = [
            _build_packed_sequence(b, tokenized, max_seq_length, self.pad_id)
            for b in bins
        ]

        avg_per_pack = (
            sum(int(t["num_real_segments"].item()) for t in packed)
            / max(len(packed), 1)
        )
        utilization = (
            sum(int(t["real_tokens"].item()) for t in packed)
            / max(len(packed) * max_seq_length, 1)
        )
        log_rank_0(
            f"[Pack] {len(tokenized)} raw samples -> {len(packed)} packed sequences. "
            f"avg samples/pack={avg_per_pack:.1f}, "
            f"token utilization={utilization * 100:.1f}% (real / max_seq_length)."
        )
        return packed

    def __len__(self) -> int:
        return len(self._packed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._packed[idx]


__all__ = [
    "MAX_SEGMENTS_PER_PACK",
    "PACK_FORMAT_VERSION",
    "PackedSFTDataset",
]
