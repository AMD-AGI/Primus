###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""InklingModel: the thinkingmachines/Inkling text backbone on Megatron-Core.

normed embedding -> InklingBlock -> final norm -> (h / logits_mup_width_multiplier) @ unembed,
plus the multi-token-prediction depths (``mtp_num_layers``; ``inkling_mtp.py``) on the last
pipeline stage, trained with Megatron's MTP loss. Audio / vision towers are not part of this port.
"""

import contextlib
import weakref

import torch
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    mtp_on_this_rank,
    process_mtp_loss,
)

from primus.backends.megatron.core.models.inkling.inkling_block import InklingBlock
from primus.backends.megatron.core.transformer.inkling.inkling_layers import (
    InklingRMSNorm,
)
from primus.backends.megatron.core.transformer.inkling.inkling_mtp import (
    get_inkling_mtp_block_spec,
)


def _padded_label_error(count, n):
    return ValueError(
        f"Inkling: {count} label(s) >= inkling_unpadded_vocab_size={n} are not masked by loss_mask; "
        "those ids do not exist in the HF vocab (tokenizer / vocab mismatch?). Mask them with loss_mask"
    )


_LIVE_CHECKS = weakref.WeakSet()
_DEFER_DEPTH = 0  # > 0 inside deferring_padded_label_checks() (the patched forward-backward pass)


@contextlib.contextmanager
def deferring_padded_label_checks():
    """Inside it, padded-label counts are queued (CPU or GPU) for ``flush_padded_label_checks``,
    which every rank calls at the same point (the ``megatron.inkling.padded_label_check`` patch
    wraps each forward-backward pass in it). Outside it, a model call checks its labels at once,
    on its own rank: nothing is queued and no collective is needed (eval scripts, converters, a
    teacher model, any forward outside the training / eval schedule)."""
    global _DEFER_DEPTH
    _DEFER_DEPTH += 1
    try:
        yield
    finally:
        _DEFER_DEPTH -= 1


def _collective_device(dist, pending):
    """The device of the all-reduced count: the CPU on a gloo group (CPU runs, the unit tests),
    else the device of a queued count, else the current GPU."""
    try:
        backend = dist.get_backend()
    except (RuntimeError, ValueError):  # no default group (only the initialized flag is set)
        backend = None
    if backend == "gloo":
        return torch.device("cpu")
    if pending:
        return pending[0].device
    return torch.device("cuda", torch.cuda.current_device())


def flush_padded_label_checks(unpadded_vocab_size):
    """Check every live model's pending padded-label counts, on every rank together.

    ``unpadded_vocab_size`` is the run's ``inkling_unpadded_vocab_size`` (from the shared config,
    the same on every rank). When it is ``None`` the run checks no padded labels: queued counts
    are dropped, with no collective and no host read. Otherwise the counts queued since the last
    flush are summed (one device read) and, with torch.distributed initialized, all-reduced over
    the world group, so every rank raises the same ValueError (not only the last-stage rank that
    saw the bad labels, which would leave the others waiting in their next collective). Whether a
    rank joins the collective depends only on that argument, never on which models are alive on
    the rank. Every rank must call it at the same point: the ``megatron.inkling.padded_label_check``
    patch calls it after each forward-backward pass of a train step or eval step, before the
    optimizer step."""
    pending = [p for check in list(_LIVE_CHECKS) for p in check.take()]
    if unpadded_vocab_size is None:
        return
    dist = torch.distributed
    distributed = dist.is_available() and dist.is_initialized()
    if not pending and not distributed:
        return
    device = _collective_device(dist, pending) if distributed else pending[0].device
    total = torch.zeros((), dtype=torch.int64, device=device)
    for count in pending:
        total += count.to(device)
    if distributed:
        dist.all_reduce(total)
    total = int(total)
    if total > 0:
        raise _padded_label_error(total, unpadded_vocab_size)


def discard_padded_label_checks():
    """Drop every live model's queued padded-label counts without reading them (the
    forward-backward pass that queued them failed). Never raises: it only empties host-side
    lists, so the caller's original exception is what propagates."""
    for check in list(_LIVE_CHECKS):
        check.pending = []


class _PaddedLabelCheck:
    """Refuses labels past the HF vocab that the loss scores.

    In the patched forward-backward pass (``deferring_padded_label_checks``) the count is queued
    without a host read, on any device: ``flush_padded_label_checks`` reads the queued counts,
    summed over all ranks, once after the pass (one scalar all-reduce and one host read per step,
    not per micro-batch), so a bad micro-batch fails its step on every rank together, before the
    optimizer applies its update. A rank never raises alone inside the pass (it would leave the
    others waiting in a collective), also on the CPU / gloo. Outside the pass the count is read at
    once and the call raises on its own rank (ValueError)."""

    def __init__(self):
        self.pending = []  # [count tensor]: labels past the HF vocab, per call
        _LIVE_CHECKS.add(self)

    def __call__(self, scored, n):
        if _DEFER_DEPTH > 0:
            self.pending.append(scored.sum())
            return
        if bool(scored.any()):
            raise _padded_label_error(int(scored.sum()), n)

    def take(self):
        """The queued counts, removed from the queue."""
        out, self.pending = self.pending, []
        return out


class InklingModel(LanguageModule):
    """The Inkling text backbone. ``fp16_lm_cross_entropy`` is accepted for the GPT builder
    signature and refused if set (the loss is computed by LanguageModule in fp32)."""

    def __init__(
        self,
        config,
        vocab_size,
        max_sequence_length,
        pre_process=True,
        post_process=True,
        fp16_lm_cross_entropy=False,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        pg_collection=None,
        vp_stage=None,
        **_kwargs,
    ):
        super().__init__(config=config, pg_collection=pg_collection)
        self.model_type = ModelType.encoder_or_decoder
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process, self.post_process = pre_process, post_process
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vp_stage = vp_stage
        if fp16_lm_cross_entropy:
            raise NotImplementedError("Inkling: fp16_lm_cross_entropy is not supported")
        if config.sequence_parallel or config.context_parallel_size > 1:
            raise NotImplementedError(
                "Inkling: sequence parallel and context parallel are not supported "
                "(set sequence_parallel: false, context_parallel_size: 1)"
            )
        n = config.inkling_unpadded_vocab_size
        if n is not None and not 0 < n <= vocab_size:
            raise ValueError(
                f"Inkling: inkling_unpadded_vocab_size={n} must be in (0, vocab_size={vocab_size}]"
            )
        # Created after the checks above, so a refused model never registers a check.
        self._padded_label_check = _PaddedLabelCheck() if n is not None else None
        # MTP runs on the last pipeline stage; that stage also needs the token embedding and the
        # trunk embed_norm (copies when it is not the first stage, kept in sync by
        # setup_embeddings_and_output_layer and allreduce_mtp_embed_norm_grads).
        self.mtp_process = False
        self.mtp_block_spec = get_inkling_mtp_block_spec(config)
        if self.mtp_block_spec is not None:
            self.mtp_process = mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
        if pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=config,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type="none",
                scatter_to_sequence_parallel=False,
                tp_group=self.pg_collection.tp,
            )
            self.embed_norm = (
                InklingRMSNorm(config.hidden_size, config.layernorm_epsilon)
                if config.inkling_use_embed_norm
                else None
            )
        self.decoder = InklingBlock(
            config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )
        # Not pre-assigned None: Megatron probes hasattr(model, "mtp") (as for GPTModel).
        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(
                config=config, spec=self.mtp_block_spec, vp_stage=vp_stage, pg_collection=self.pg_collection
            )
        if post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not parallel_output,
                skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights,
                tp_group=self.pg_collection.tp,
            )
        if pre_process or post_process or self.mtp_process:
            self.setup_embeddings_and_output_layer()
        pp_mtp = bool(config.mtp_num_layers) and config.pipeline_model_parallel_size > 1
        if pp_mtp and pre_process and not self.mtp_process and self.embed_norm is not None:
            # the first stage's embed_norm: its own DDP bucket, as its MTP-stage copy below
            self.embed_norm.weight.shared_embedding = True
        if self.mtp_process and not pre_process and self.embed_norm is not None:
            # The MTP stage's copy of the trunk embed_norm: its gradient is summed with the first
            # stage's (allreduce_mtp_embed_norm_grads); both start at ones (or load the same
            # checkpoint key). ``shared``: counted once in the grad norm. ``shared_embedding``: its
            # own DDP bucket, so the distributed optimizer shards it the same way on both stages.
            self.embed_norm.weight.shared = True
            self.embed_norm.weight.shared_embedding = True

    def _embed(self, input_ids, position_ids=None):
        """Token embedding, then the trunk embed_norm (the decoder input; the MTP depths'
        embedding, before their own embed_norm)."""
        del position_ids  # Inkling has no position embedding
        out = self.embedding(input_ids=input_ids, position_ids=None)  # [s, b, h]
        return self.embed_norm(out) if self.embed_norm is not None else out

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Inkling"
        self.decoder.set_input_tensor(input_tensor[0])

    def _mask_padded_vocab(self, logits):
        """Logits of ids >= unpadded_vocab_size (HF drops them) set to the dtype minimum.

        ``logits`` is this TP rank's vocab shard when ``parallel_output`` (columns
        ``[rank * per, (rank + 1) * per)``), else the full vocab."""
        n = self.config.inkling_unpadded_vocab_size
        if not n:
            return logits
        per = logits.shape[-1]
        start = per * self.pg_collection.tp.rank() if self.parallel_output else 0
        if start + per <= n:
            return logits
        # in place: no second logits-sized buffer. Safe for autograd: the output layer's
        # backward needs its input and weight, not these logits, and masked_fill_ zeroes the
        # gradient of the filled columns.
        cols = torch.arange(start, start + per, device=logits.device)
        return logits.masked_fill_(cols >= n, torch.finfo(logits.dtype).min)

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        decoder_input=None,
        labels=None,
        loss_mask=None,
        packed_seq_params=None,
        inference_context=None,
        **kwargs,
    ):
        """Logits ``[b, s, V]`` (this TP rank's vocab shard ``[b, s, V / tp]`` when
        ``parallel_output``) when ``labels`` is None, else the per-token loss ``[b, s]``.

        ``position_ids`` is unused (Inkling has a relative bias, no position embedding).
        ``attention_mask`` (True = masked) is applied on top of the causal / window mask.
        ``loss_mask`` is applied by the training loss function, as for GPTModel.

        Padded vocab ids (``>= inkling_unpadded_vocab_size``; HF does not have them): their
        logits are masked, so they have probability zero and cannot be a training target. A
        label with such an id that ``loss_mask`` masks scores 0. One that it does not mask (or
        any, when ``loss_mask`` is None) is refused (a tokenizer / vocab mismatch), unless it is
        ``inkling_ignored_label_id`` (the NullTokenizer EOD id under mock data, set by the
        builder): that one scores 0 and stays in the caller's token count. The refusal is a
        ``ValueError``. Inside the patched forward-backward pass (training / eval schedule, any
        device) it is raised on every rank after the pass, with no host sync per micro-batch;
        outside it (eval scripts, converters) it is raised at once on the calling rank (see
        ``_PaddedLabelCheck``). ``loss_mask`` is read, never changed.
        """
        if packed_seq_params is not None:
            raise NotImplementedError("Inkling: packed sequences are not supported")
        if inference_context is not None or kwargs.get("inference_params") is not None:
            raise NotImplementedError("Inkling: KV-cache inference is not supported")
        if decoder_input is None and self.pre_process:
            decoder_input = self._embed(input_ids)
        mtp = getattr(self, "mtp", None) if self.mtp_process else None
        if mtp is None:
            hidden = self.decoder(decoder_input, attention_mask=attention_mask)
        else:
            # The MTP depths read the trunk's last-layer output before the final norm
            # (SGLang hidden_states_before_norm): the MTP stage holds the last decoder layer.
            hidden, (pre_norm,) = self.decoder(
                decoder_input,
                attention_mask=attention_mask,
                extract_layer_indices={self.config.num_layers - 1},
            )
            if input_ids is None:
                raise ValueError("Inkling MTP needs input_ids on the MTP stage")
            if position_ids is None:
                b, s = input_ids.shape
                position_ids = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, -1)
            mtp_hidden = mtp(
                input_ids=input_ids,
                position_ids=position_ids,  # only rolled along (Inkling has no position embedding)
                hidden_states=pre_norm,
                attention_mask=attention_mask,
                embedding=self._embed,
            )
            # [(1 + depths) * s, b, h]: the trunk output (normed), then each depth (raw)
            hidden = torch.cat([hidden, mtp_hidden[pre_norm.shape[0] :]], dim=0)
        if not self.post_process:
            return hidden
        hidden = hidden / self.config.inkling_logits_mup_width_multiplier
        weight = (
            self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else None
        )
        if mtp is not None:
            # Megatron's MTP loss: each depth's logits (padded vocab masked) against the labels
            # rolled d + 1 positions; scaled by mtp_loss_scaling_factor / depths and added to the
            # gradient by MTPLossAutoScaler. Returns the trunk chunk.
            hidden = process_mtp_loss(
                hidden_states=hidden,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=self.output_layer,
                output_weight=weight,
                runtime_gather_output=None,
                is_training=self.training,
                compute_language_model_loss=self._masked_language_model_loss,
                config=self.config,
                cp_group=self.pg_collection.cp,
                scale_logits_fn=self._mask_padded_vocab,
            )
        logits, _ = self.output_layer(hidden, weight=weight)
        # Ids >= unpadded_vocab_size do not exist in HF (it slices them off): they get no
        # probability mass, in scoring and in the training loss alike.
        logits = self._mask_padded_vocab(logits)
        if labels is None:
            return logits.transpose(0, 1).contiguous()
        n = self.config.inkling_unpadded_vocab_size
        if n:
            # Padded ids cannot be targets (see the docstring): masked ones score 0, unmasked
            # ones are refused. (The MTP labels are these labels, rolled: checked here too.)
            invalid = labels >= n
            scored = invalid if loss_mask is None else invalid & (loss_mask != 0)
            if self.config.inkling_ignored_label_id is not None:
                scored = scored & (labels != self.config.inkling_ignored_label_id)
            self._padded_label_check(scored, n)
        return self._masked_language_model_loss(labels, logits)

    def _masked_language_model_loss(self, labels, logits):
        """Per-token loss ``[b, s]``; labels past the HF vocab score 0 (they are scored against
        id 0 first, so the loss kernel sees valid ids)."""
        n = self.config.inkling_unpadded_vocab_size
        if not n:
            return self.compute_language_model_loss(labels, logits)
        invalid = labels >= n
        loss = self.compute_language_model_loss(labels.masked_fill(invalid, 0), logits)
        return loss.masked_fill(invalid, 0.0)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """LanguageModule's (it ties the MTP stage's word-embedding copy), plus the MTP stage's
        copy of the trunk ``embed_norm``: the first stage's key, as a secondary replica
        (replica id ``(1, tp, dp)``, as ``tie_word_embeddings_state_dict``), so it is loaded
        from the first stage's tensor and never saved twice."""
        sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        key = f"{prefix}embed_norm.weight"
        if self.mtp_process and not self.pre_process and key in sd:
            from megatron.core.dist_checkpointing.mapping import ShardedTensor
            from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
            from megatron.core.utils import get_pg_rank

            metadata = ensure_metadata_has_dp_cp_group(metadata)
            sd[key] = ShardedTensor.from_rank_offsets(
                key,
                self.embed_norm.weight,
                replica_id=(1, get_pg_rank(self.pg_collection.tp), get_pg_rank(metadata["dp_cp_group"])),
            )
        return sd


def allreduce_mtp_embed_norm_grads(model, config):
    """Sum the trunk ``embed_norm`` gradient of the first pipeline stage and of the MTP stage's
    copy (over the embedding group, as Megatron does for the word embedding). A no-op without
    MTP, without PP or without embed_norm."""
    if not getattr(config, "mtp_num_layers", None) or config.pipeline_model_parallel_size == 1:
        return
    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import _allreduce_embedding_grad

    def weight(module):
        norm = getattr(module, "embed_norm", None)
        if norm is None or not (module.pre_process or getattr(module, "mtp_process", False)):
            return None
        return norm.weight

    _allreduce_embedding_grad(
        model,
        parallel_state.get_embedding_group(check_initialized=False),
        parallel_state.get_pipeline_model_parallel_group(),
        weight,
        config=config,
    )
