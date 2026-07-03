###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Variable-length SFT sample store + LB-Mini data iterator (ODC path).

This is the data layer for LB-Mini. Unlike ``PackedSFTDataset`` (which bin-packs
everything into fixed ``max_seq_length`` blocks and hands each DP rank the SAME
number of blocks), LB-Mini keeps samples VARIABLE length and, for every global
minibatch, uses Karmarkar-Karp (see ``lb_mini_packing.plan_minibatch``) to:

  * balance the TOTAL token workload across DP ranks, and
  * let each rank own a DIFFERENT number of micro-batches.

The "different micro-batch count per rank" is exactly what a collective-comm
backend (NCCL all-gather/reduce-scatter) cannot tolerate, which is why this path
is gated behind ``ODC_ENABLE=1`` -- ODC's point-to-point comm lets ranks run
out of lockstep without deadlocking.

Two pieces live here:
  1. ``build_varlen_samples`` -- tokenize the dataset ONCE into variable-length
     ``{input_ids, labels, loss_mask, length}`` dicts (cached to disk, reusing
     packing.py's tokenizer-identity cache-key machinery), and
  2. ``LBMiniDataIterator`` -- the iterator Megatron's schedule pulls from. It
     refills one global minibatch at a time, plans this rank's micro-batches via
     KK, and exposes ``current_num_microbatches`` so the schedule patch can run
     the right (rank-local) number of forward/backward steps.
"""

import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from primus.backends.megatron.sft.lb_mini_packing import plan_minibatch, resolve_cost_func
from primus.modules.module_utils import log_rank_0


def _shift_labels_loss_mask(input_ids: np.ndarray, loss_mask: np.ndarray):
    """Next-token shift, mirroring packing._build_packed_sequence semantics.

    Megatron's ``compute_language_model_loss`` does NOT shift internally, so the
    dataset MUST emit shifted labels: labels[i]=input_ids[i+1], and loss_mask
    shifted the same way; the final position has no target and is masked.
    """
    n = len(input_ids)
    labels = input_ids.copy()
    if n >= 2:
        labels[:-1] = input_ids[1:]
    shifted_mask = np.zeros_like(loss_mask)
    if n >= 2:
        shifted_mask[:-1] = loss_mask[1:]
    shifted_mask[-1] = 0
    return labels, shifted_mask


def build_varlen_samples(
    dataset_name: str,
    tokenizer,
    max_seq_length: int,
    split: str,
    formatter: str,
    seed: int,
    bridge_compat_inline_bos: bool = False,
    **kwargs,
) -> List[Dict[str, np.ndarray]]:
    """Tokenize the dataset into variable-length (un-padded) samples, cached.

    Reuses ``PackedSFTDataset``'s tokenize step (``_tokenize_no_pad``) but does
    NOT bin-pack: every raw sample becomes one variable-length record. Samples
    whose tokenized length is 0 are dropped; samples longer than max_seq_length
    are truncated (same as the non-bridge packing path).
    """
    from primus.backends.megatron.sft import packing as pack_mod
    from primus.backends.megatron.sft.dataset import SFTDataset

    cache_disabled = os.environ.get("PRIMUS_DISABLE_PACK_CACHE", "0") not in ("0", "", "false", "False")
    pad_id = pack_mod._resolve_pad_token_id(tokenizer)
    cache_file = lock_file = None
    if not cache_disabled:
        # Distinct cache namespace from packed (different layout): prefix "varlen".
        key = pack_mod._build_pack_cache_key(
            dataset_name=dataset_name,
            split=split,
            formatter=formatter,
            max_seq_length=max_seq_length,
            pad_id=pad_id,
            tokenizer_id=pack_mod._tokenizer_identity(tokenizer),
            bridge_compat_inline_bos=bridge_compat_inline_bos,
        )
        cache_dir = pack_mod._resolve_pack_cache_dir()
        cache_file = cache_dir / f"sft_varlen_{key}.pt"
        # Node-local filelock (PRIMUS_PACK_LOCK_DIR): avoid NFS "Stale file handle".
        _lock_dir = os.environ.get("PRIMUS_PACK_LOCK_DIR")
        if _lock_dir:
            os.makedirs(_lock_dir, exist_ok=True)
            lock_file = os.path.join(_lock_dir, f"sft_varlen_{key}.lock")
        else:
            lock_file = cache_dir / f"sft_varlen_{key}.lock"

    def _build() -> List[Dict[str, np.ndarray]]:
        base = SFTDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split=split,
            formatter=formatter,
            seed=seed,
            **kwargs,
        )
        raw = base.dataset
        log_rank_0(f"[LB-Mini] Tokenizing {len(raw)} samples (variable length, no packing)...")
        out: List[Dict[str, np.ndarray]] = []
        for i in range(len(raw)):
            sample = pack_mod.normalize_sft_sample(raw[i])
            formatted = base.formatter.format_sample(sample)
            tok = pack_mod._tokenize_no_pad(
                formatted, tokenizer, max_seq_length, bridge_compat_inline_bos=bridge_compat_inline_bos
            )
            if tok["length"] <= 0:
                continue
            labels, shifted_mask = _shift_labels_loss_mask(tok["input_ids"], tok["loss_mask"])
            out.append(
                {
                    "input_ids": tok["input_ids"].astype(np.int64),
                    "labels": labels.astype(np.int64),
                    "loss_mask": shifted_mask.astype(np.int64),
                    "length": int(tok["length"]),
                }
            )
        sup = sum(int(s["loss_mask"].sum()) for s in out)
        log_rank_0(
            f"[LB-Mini] Built {len(out)} variable-length samples; "
            f"supervised tokens={sup} (sanity: must be > 0)."
        )
        return out

    if cache_disabled:
        return _build()

    try:
        from filelock import FileLock
    except ImportError:
        return _build()

    with FileLock(str(lock_file)):
        if cache_file.exists():
            log_rank_0(f"[LB-Mini] varlen CACHE HIT ({cache_file.name}); loading.")
            samples = torch.load(cache_file, weights_only=False)
            log_rank_0(f"[LB-Mini] Loaded {len(samples)} variable-length samples from cache.")
            return samples
        samples = _build()
        tmp = cache_file.with_suffix(".pt.tmp")
        torch.save(samples, tmp)
        os.replace(tmp, cache_file)
        log_rank_0(f"[LB-Mini] Cached variable-length samples to {cache_file}")
        return samples


class LBMiniDataIterator:
    """Iterator that yields THIS rank's variable-length micro-batches.

    Every ``global_batch_size`` raw samples form one *global minibatch*. For each
    global minibatch we:
      1. read the per-sample effective lengths (identical view on all ranks),
      2. ``plan_minibatch`` -> KK-balance across DP ranks + split this rank into
         micro-batches each <= ``max_token_len``,
      3. push this rank's micro-batches onto a queue and record
         ``current_num_microbatches`` (may differ across ranks).

    The schedule patch calls ``begin_minibatch()`` once at the top of each
    train_step to materialize the plan and read ``current_num_microbatches``;
    then Megatron's ``forward_step`` pulls each micro-batch via ``__next__``.

    A micro-batch here is ONE packed 1-D sequence (the KK-chosen samples for that
    micro-batch concatenated). With ``micro_batch_size`` typically 1 and short
    samples concatenated up to ``max_token_len``, attention runs causal over the
    concatenation (same implicit-multi-turn regime as Primus packed SFT with
    ``use_packed_attention=false``); ``loss_mask`` keeps loss on response tokens.
    """

    def __init__(
        self,
        samples: List[Dict[str, np.ndarray]],
        global_batch_size: int,
        max_token_len: int,
        dp_rank: int,
        dp_size: int,
        pad_id: int,
        cost_model: str = "linear",
        seed: int = 1234,
        shuffle: bool = True,
        same_micro_num: bool = False,
        packing_method: str = "kk",
    ):
        self.samples = samples
        self.global_batch_size = int(global_batch_size)
        self.max_token_len = int(max_token_len)
        self.dp_rank = int(dp_rank)
        self.dp_size = int(dp_size)
        self.pad_id = int(pad_id)
        self.cost_func = resolve_cost_func(cost_model)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        # same_micro_num=True forces all ranks to the SAME micro-batch count
        # (all_reduce MAX): this is the no-LB-Mini baseline (ranks aligned, the
        # short-workload ranks pad/idle). False = LB-Mini (ranks may differ).
        self.same_micro_num = bool(same_micro_num)
        # packing_method: "kk" = Karmarkar-Karp load balance (LB-Mini);
        # "round_robin" = ODC-example-style "None" packing (rank takes
        # idx[rank::dp], NO balancing -> uneven workload, for A/B of ODC comm).
        self.packing_method = str(packing_method)

        self._order: List[int] = []
        self._cursor = 0
        self._epoch = 0
        self._queue: List[Dict[str, torch.Tensor]] = []
        self.current_num_microbatches = 0
        self._dbg_count = 0
        self._reshuffle()

    def _reshuffle(self):
        n = len(self.samples)
        if self.shuffle:
            g = np.random.default_rng(self.seed + self._epoch)
            self._order = g.permutation(n).tolist()
        else:
            self._order = list(range(n))
        self._cursor = 0
        self._epoch += 1

    def _next_global_indices(self) -> List[int]:
        """Pull the next ``global_batch_size`` sample indices (wraps epochs)."""
        if self._cursor + self.global_batch_size > len(self._order):
            self._reshuffle()
        idx = self._order[self._cursor : self._cursor + self.global_batch_size]
        self._cursor += self.global_batch_size
        return idx

    def _pack_microbatch(self, sample_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Concatenate chosen samples into one variable-length micro-batch.

        Emits thd-compatible fields (cu_seqlens padded to [1, MAX_SEGMENTS+1],
        per-segment position_ids, max_sub_seqlen) so that with
        ``use_packed_attention=true`` the forward path runs SEGMENTED (thd)
        attention -- O(sum seqlen^2) instead of O(total^2). This is what makes a
        large max_token_len (packed multi-sample micro-batches) fit in memory.
        """
        from primus.backends.megatron.sft.packing import MAX_SEGMENTS_PER_PACK

        ids = np.concatenate([self.samples[i]["input_ids"] for i in sample_indices])
        lbl = np.concatenate([self.samples[i]["labels"] for i in sample_indices])
        msk = np.concatenate([self.samples[i]["loss_mask"] for i in sample_indices])
        seglens = [int(self.samples[i]["length"]) for i in sample_indices]
        total = int(sum(seglens))
        # Per-segment position ids (each sub-sample restarts at 0).
        pos = np.concatenate([np.arange(s, dtype=np.int64) for s in seglens])
        # cu_seqlens padded to a fixed width so default_collate can stack.
        cu = [0]
        off = 0
        for s in seglens:
            off += s
            cu.append(off)
        n_seg = len(seglens)
        while len(cu) < MAX_SEGMENTS_PER_PACK + 1:
            cu.append(total)
        cu_np = np.asarray(cu, dtype=np.int32)
        max_sub = max(seglens) if seglens else 0
        return {
            "input_ids": torch.from_numpy(ids).long().unsqueeze(0),  # [1, T]
            "labels": torch.from_numpy(lbl).long().unsqueeze(0),
            "loss_mask": torch.from_numpy(msk).long().unsqueeze(0),
            "position_ids": torch.from_numpy(pos).long().unsqueeze(0),
            "cu_seqlens": torch.from_numpy(cu_np).unsqueeze(0),  # [1, MAX_SEG+1]
            "num_segments": torch.tensor([n_seg], dtype=torch.int32),  # [1]
            "max_sub_seqlen": torch.tensor([max_sub], dtype=torch.int32),
        }

    def begin_minibatch(self) -> int:
        """Plan ONE global minibatch; fill this rank's queue. Returns rank count."""
        global_idx = self._next_global_indices()
        lengths = [int(self.samples[i]["length"]) for i in global_idx]
        if self.packing_method == "round_robin":
            # No load balancing (ODC example's default "None"): rank takes
            # global_idx[rank::dp], one sample per micro-batch. Per-rank workload
            # is UNEVEN (variable lengths) -> this is where ODC's on-demand comm
            # is supposed to win by overlapping ranks instead of bulk-syncing.
            local_positions = list(range(self.dp_rank, len(global_idx), self.dp_size))
            local_micro = [[p] for p in local_positions]
        else:
            # KK-balance across DP ranks + split this rank into micro-batches.
            local_micro = plan_minibatch(
                lengths,
                rank=self.dp_rank,
                world_size=self.dp_size,
                max_token_len=self.max_token_len,
                same_micro_num=self.same_micro_num,  # False=LB-Mini, True=aligned baseline
                get_seq_costs_func=self.cost_func,
            )
        # local_micro entries index into ``lengths`` (== position in global_idx);
        # map back to absolute sample indices.
        self._queue = [self._pack_microbatch([global_idx[p] for p in micro]) for micro in local_micro]
        self.current_num_microbatches = len(self._queue)
        # Observability: show per-rank micro-batch count for the first few global
        # minibatches -- this is where "different count per rank" becomes visible.
        if self._dbg_count < 4:
            tot = sum(int(mb["input_ids"].numel()) for mb in self._queue)
            print(
                f"[LB-Mini] rank{self.dp_rank}: num_microbatches={self.current_num_microbatches} "
                f"total_tokens={tot}",
                flush=True,
            )
            self._dbg_count += 1
        return self.current_num_microbatches

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if not self._queue:
            # Schedule patch should call begin_minibatch() first; as a fallback
            # (e.g. eval loops) we refill transparently.
            self.begin_minibatch()
        return self._queue.pop(0)


__all__ = ["build_varlen_samples", "LBMiniDataIterator"]
