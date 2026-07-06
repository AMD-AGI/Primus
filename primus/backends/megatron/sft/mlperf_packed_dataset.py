###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mlperf-style packed SFT dataset for Megatron-native SFT.

This module lets the Megatron-native SFT backend consume the exact same
``train.npy`` / ``validation.npy`` / ``packed_metadata.jsonl`` artefacts that
the upstream mlperf LLama-2-70B PEFT recipe ships through Megatron-Bridge
(``examples/megatron_bridge/configs/MI355X/llama2_70b_lora_mlperf_posttrain.yaml``).

Why a separate dataset class?
-----------------------------
Native ``SFTDataset`` / ``PackedSFTDataset`` only know how to consume
``HuggingFace Hub`` paths or local ``.jsonl/.json`` files and rebuilds the
tokenize + bin-pack offline pipeline. The mlperf branch already ships a
pre-tokenised + pre-packed dataset in NumPy ``object`` arrays:

    train.npy           : shape=(N,), dtype=object; each element is a dict
                            { input_ids : int32[seq_len],
                              loss_mask : list[int]  of length seq_len,
                              seq_start_id : list[int] of segment starts }
    validation.npy      : same schema, smaller N
    packed_metadata.jsonl :
        [{"max_samples_per_bin": 1,
          "dataset_max_seqlen": 8192,
          "min_packed_seqlen": 8192}]

So Native cannot just point ``sft_dataset_name`` at this directory and reuse
``PackedSFTDataset`` -- the tokenize + pack pipeline would clobber what mlperf
already computed. Instead we add this minimal loader that materialises one
pre-built pack per ``__getitem__`` and produces the *exact* batch dict the
Native forward step (``primus/backends/megatron/sft/forward_step.py``)
expects.

Shift semantics
---------------
Megatron-LM's ``compute_language_model_loss`` does NOT internally shift
labels (it computes CE between ``logits[t]`` and ``labels[t]`` directly), so
the dataset MUST emit shifted labels and a correspondingly shifted
``loss_mask``. This mirrors:
  * Native ``PackedSFTDataset._build_packed_sequence`` (packing.py:319-333)
  * Bridge ``GPTSFTPackedDataset.collate_fn`` (third_party/Megatron-Bridge/
    src/megatron/bridge/data/datasets/sft.py:869-877) where the per-segment
    ``input_ids[boundaries[i]+1 : boundaries[i+1]]`` slice IS the label.

For mlperf samples (``max_samples_per_bin=1`` => exactly one supervised
segment of length ``dataset_max_seqlen`` per packed bin, already padded),
this means:
    labels[0:L-1]    = input_ids[1:L]
    labels[L-1]      = pad_id              (no next-token target inside this segment)
    shifted_mask[i]  = loss_mask[i+1]       for i in [0, L-2]
    shifted_mask[L-1]= 0

cu_seqlens layout
-----------------
mlperf packs are single-segment (``max_samples_per_bin=1``). We emit
``cu_seqlens = [0, L]`` with the trailing entries padded up to
``MAX_SEGMENTS_PER_PACK + 1`` by repeating ``L`` so the tensor can be
``default_collate``-stacked. This matches the layout produced by
``PackedSFTDataset._build_packed_sequence`` so ``forward_step`` does not need
to branch on dataset source.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from primus.backends.megatron.sft.packing import MAX_SEGMENTS_PER_PACK
from primus.backends.megatron.sft.preprocessing import _resolve_pad_token_id, log_rank_0

_MLPERF_TRAIN_FILENAME = "train.npy"
_MLPERF_VAL_FILENAME = "validation.npy"
_MLPERF_METADATA_FILENAME = "packed_metadata.jsonl"


def is_mlperf_packed_dir(path: str) -> bool:
    """Return True iff ``path`` looks like a directory of mlperf packed npy files.

    The recipe-level dispatch in ``build_train_valid_test_datasets`` calls
    this to decide whether to route to ``MlperfPackedDataset`` or fall
    through to the standard ``SFTDataset`` / ``PackedSFTDataset`` path.
    """
    if not path or not isinstance(path, str):
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    return (p / _MLPERF_TRAIN_FILENAME).is_file() and (p / _MLPERF_METADATA_FILENAME).is_file()


def _load_packed_metadata(metadata_path: Path) -> dict:
    """Read the single-record packed metadata jsonl produced by mlperf.

    The file is a json *list* with one dict, e.g.
        [{"max_samples_per_bin": 1,
          "dataset_max_seqlen": 8192,
          "min_packed_seqlen": 8192}]
    """
    with open(metadata_path, "r") as f:
        records = json.load(f)
    assert (
        isinstance(records, list) and len(records) >= 1
    ), f"packed_metadata.jsonl must be a non-empty json list, got: {records!r}"
    return records[0]


class MlperfPackedDataset(Dataset):
    """Single-segment packed SFT dataset from mlperf-style ``.npy``.

    Parameters
    ----------
    npy_path :
        Absolute path to ``train.npy`` or ``validation.npy``.
    max_seq_length :
        Expected packed length per sample. Asserted against the on-disk
        ``dataset_max_seqlen`` and against every loaded array so a
        misconfigured ``seq_length`` cannot silently train at the wrong
        size.
    pad_id :
        Token id used for the synthetic labels[L-1] padding. Resolved from
        the tokenizer via ``_resolve_pad_token_id`` at builder time so the
        Llama-2 tokenizer's pad token (0) is picked up automatically.
    """

    def __init__(
        self,
        *,
        npy_path: Path,
        max_seq_length: int,
        pad_id: int,
        split_name: str = "train",
    ):
        super().__init__()
        self._npy_path = Path(npy_path)
        self._max_seq_length = int(max_seq_length)
        self._pad_id = int(pad_id)
        self._split_name = split_name

        log_rank_0(f"[MlperfPacked] Loading {split_name} dataset from {self._npy_path} ...")
        # ``allow_pickle`` is required because mlperf packs each sample as a
        # Python dict and stores them in an ``object`` ndarray.
        self._data = np.load(str(self._npy_path), allow_pickle=True)
        log_rank_0(
            f"[MlperfPacked] Loaded {len(self._data)} packed samples "
            f"({split_name}) from {self._npy_path.name}."
        )

    def __len__(self) -> int:
        return int(len(self._data))

    def __getitem__(self, idx: int):
        sample = self._data[int(idx)]
        if not isinstance(sample, dict):
            raise TypeError(
                f"mlperf packed sample at index {idx} is not a dict; "
                f"got {type(sample).__name__}. The .npy file must be the "
                f"output of upstream convert_dataset.py + create_metadata.py "
                f"(object array of {{input_ids, loss_mask, seq_start_id}})."
            )

        input_ids_np = np.asarray(sample["input_ids"], dtype=np.int64)
        loss_mask_np = np.asarray(sample["loss_mask"], dtype=np.int64)

        L = int(input_ids_np.shape[0])
        if L != self._max_seq_length:
            raise ValueError(
                f"mlperf packed sample length {L} != max_seq_length "
                f"{self._max_seq_length}. Set seq_length and "
                f"max_position_embeddings to {L} in the yaml to match "
                f"dataset_max_seqlen on disk."
            )
        if loss_mask_np.shape[0] != L:
            raise ValueError(
                f"mlperf packed sample loss_mask length {loss_mask_np.shape[0]} " f"!= input_ids length {L}."
            )

        # Shift labels and loss_mask by 1 to match Megatron's
        # compute_language_model_loss contract (no internal shift). The last
        # position has no in-segment next-token target -- labels[L-1] gets
        # pad_id and shifted_loss_mask[L-1] stays 0 so it never contributes.
        labels_np = np.full(L, self._pad_id, dtype=np.int64)
        if L >= 2:
            labels_np[: L - 1] = input_ids_np[1:L]
        shifted_loss_mask_np = np.zeros(L, dtype=np.int64)
        if L >= 2:
            shifted_loss_mask_np[: L - 1] = loss_mask_np[1:L]

        position_ids_np = np.arange(L, dtype=np.int64)

        # cu_seqlens layout: single segment spanning the full packed sample.
        # We pad cu_seqlens to ``MAX_SEGMENTS_PER_PACK + 1`` so the tensor
        # has the same fixed shape as the one produced by
        # ``PackedSFTDataset._build_packed_sequence``; downstream collate
        # and ``_build_packed_seq_params`` only consume the prefix marked
        # by ``num_segments`` / ``num_real_segments`` anyway.
        cu_seqlens = [0, L]
        num_segments = 1
        while len(cu_seqlens) < MAX_SEGMENTS_PER_PACK + 1:
            cu_seqlens.append(L)

        return {
            "input_ids": torch.from_numpy(input_ids_np),
            "labels": torch.from_numpy(labels_np),
            "loss_mask": torch.from_numpy(shifted_loss_mask_np),
            "position_ids": torch.from_numpy(position_ids_np),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "num_segments": torch.tensor(num_segments, dtype=torch.int32),
            "num_real_segments": torch.tensor(num_segments, dtype=torch.int32),
            "real_tokens": torch.tensor(L, dtype=torch.int32),
            "max_sub_seqlen": torch.tensor(L, dtype=torch.int32),
        }


def build_mlperf_packed_datasets(
    *,
    data_dir: str,
    tokenizer,
    max_seq_length: int,
    train_val_test_num_samples: list[int],
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """Build train/val/test datasets from an mlperf-style packed directory.

    The signature mirrors ``dataset.build_train_valid_test_datasets`` so
    ``runtime.create_sft_datasets_provider`` can call this function
    interchangeably.

    Parameters
    ----------
    data_dir :
        Directory containing ``train.npy`` / ``validation.npy`` /
        ``packed_metadata.jsonl`` -- e.g. ``/data/mlperf_llama2``.
    tokenizer :
        Used only to derive ``pad_id``. The dataset itself is already
        tokenised so no encode/decode happens here.
    max_seq_length :
        Asserted against ``dataset_max_seqlen`` in ``packed_metadata.jsonl``
        AND against every sample's ``input_ids`` length.
    train_val_test_num_samples :
        Megatron pretrain-style triple ``[train_samples, val_samples,
        test_samples]``. We do not slice the dataset to these numbers
        (Megatron's cyclic dataloader handles oversampling), we only use
        them as enable/disable gates: a 0 means "skip building this
        split's dataset".
    """
    data_path = Path(data_dir)
    metadata_path = data_path / _MLPERF_METADATA_FILENAME
    metadata = _load_packed_metadata(metadata_path)
    dataset_max_seqlen = int(metadata.get("dataset_max_seqlen", -1))
    min_packed_seqlen = int(metadata.get("min_packed_seqlen", dataset_max_seqlen))
    max_samples_per_bin = int(metadata.get("max_samples_per_bin", 1))

    log_rank_0(
        f"[MlperfPacked] Found mlperf packed dataset directory: {data_path}. "
        f"dataset_max_seqlen={dataset_max_seqlen}, "
        f"min_packed_seqlen={min_packed_seqlen}, "
        f"max_samples_per_bin={max_samples_per_bin}, "
        f"yaml seq_length={max_seq_length}."
    )

    if dataset_max_seqlen != max_seq_length:
        raise ValueError(
            f"mlperf packed dataset seq length mismatch: dataset_max_seqlen="
            f"{dataset_max_seqlen} (from {metadata_path.name}) vs yaml "
            f"seq_length={max_seq_length}. Adjust seq_length / "
            f"max_position_embeddings to {dataset_max_seqlen} OR regenerate "
            f"the packed dataset with convert_dataset.py at the desired "
            f"sequence length."
        )

    pad_id = _resolve_pad_token_id(tokenizer)
    log_rank_0(f"[MlperfPacked] Using pad_id={pad_id} for label tail.")

    train_samples, valid_samples, test_samples = train_val_test_num_samples

    train_ds: Optional[Dataset] = None
    if train_samples > 0:
        train_npy = data_path / _MLPERF_TRAIN_FILENAME
        if not train_npy.is_file():
            raise FileNotFoundError(
                f"mlperf train npy not found at {train_npy}. Run "
                f"upstream download_dataset.py + convert_dataset.py first."
            )
        train_ds = MlperfPackedDataset(
            npy_path=train_npy,
            max_seq_length=max_seq_length,
            pad_id=pad_id,
            split_name="train",
        )

    valid_ds: Optional[Dataset] = None
    val_npy = data_path / _MLPERF_VAL_FILENAME
    if valid_samples > 0 and val_npy.is_file():
        valid_ds = MlperfPackedDataset(
            npy_path=val_npy,
            max_seq_length=max_seq_length,
            pad_id=pad_id,
            split_name="validation",
        )
    elif valid_samples > 0:
        # mlperf ships validation.npy alongside train.npy; if it is missing
        # the recipe is misconfigured. Surface this loudly instead of
        # silently disabling eval the way the squad path does.
        raise FileNotFoundError(
            f"mlperf validation npy not found at {val_npy} but "
            f"valid_samples={valid_samples}. Either set eval_iters=0 in the "
            f"yaml or regenerate the validation split with "
            f"convert_dataset.py."
        )

    test_ds: Optional[Dataset] = None
    if test_samples > 0:
        # mlperf llama2 recipe has no test split; mirror the squad path
        # by silently returning None.
        log_rank_0(
            "[MlperfPacked] test split not supported by mlperf llama2 " "dataset; returning None for test_ds."
        )

    return train_ds, valid_ds, test_ds


__all__ = [
    "MlperfPackedDataset",
    "build_mlperf_packed_datasets",
    "is_mlperf_packed_dir",
]
