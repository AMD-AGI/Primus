###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 decoder block.

A :class:`TransformerBlock` (layers are Megatron ``TransformerLayer`` s built from this
stage's specs, ``vp_stage`` aware) with the state Mistral Large 4 shares across layers:

* one YaRN + LeRoPE rope table (``rotary``; the checkpoint's ``lerope.alpha``) and the query
  positions SSMax needs, computed once per forward;
* the DSA top-k of each full-indexer layer, reused by the following ``shared`` layers;
* the normalized KV latent of a source layer (35 in the release), read by its consumer layers.
  Inside a stage it is passed between layers as a tensor, so gradients from the consumers reach
  the source layer. Under full recompute the checkpoint unit is a chunk (``uniform``:
  ``recompute_num_layers`` consecutive layers; ``block``: each of the stage's first
  ``recompute_num_layers`` layers alone; see ``_chunks``): a chunk takes
  the latent as an input only when one of its layers consumes it, and returns it only when one
  of its layers stores it, so it crosses chunk boundaries as a checkpointed tensor. Across pipeline stages it travels in the P2P tensor: ``[s + n, b, h]`` = hidden states,
  then each sample's latent ``[s, kv_lora_rank]`` packed densely into ``n = ceil(s * kv_lora_rank / h)``
  rows (``kv_carrier_rows``; 171 rows for s 2048 at the release's 512 / 6144; see
  ``mistral_large4_pp_shape_patches``).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import (
    get_pg_rank,
    get_pg_size,
    log_single_rank,
    make_viewless_tensor,
)

from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    kv_carrier_rows,
)
from primus.backends.megatron.core.models.mistral_large4.mistral_large4_transformer_config import (
    kv_carrier_needed,
)
from primus.backends.megatron.core.transformer.mistral_large4.rotary import (
    MistralLarge4RotaryEmbedding,
    positions_for,
)

__all__ = ["MistralLarge4TransformerBlock"]

logger = logging.getLogger(__name__)


class MistralLarge4TransformerBlock(TransformerBlock):
    """Decoder block of Mistral Large 4 (see the module docstring)."""

    def __init__(
        self,
        config,
        spec,
        post_layer_norm=True,
        pre_process=True,
        post_process=True,
        pg_collection=None,
        vp_stage=None,
    ):
        if config.fp8 or config.fp4:
            raise NotImplementedError("Mistral Large 4: FP8 / FP4 training is not supported yet (BF16 only)")
        if config.cpu_offloading:
            raise NotImplementedError("Mistral Large 4: CPU activation offloading is not supported")
        if config.recompute_granularity == "full" and config.recompute_method not in ("uniform", "block"):
            raise ValueError(
                f"recompute_granularity full needs recompute_method uniform or block, got {config.recompute_method!r}"
            )
        if config.recompute_granularity == "full" and not int(config.recompute_num_layers or 0):
            raise ValueError("recompute_granularity full needs recompute_num_layers >= 1")
        super().__init__(
            config,
            spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        self.rotary = MistralLarge4RotaryEmbedding(config)
        self.layer_indices = [layer.layer_number - 1 for layer in self.layers]
        self.carrier = kv_carrier_needed(
            config.kv_source_layers,
            config.num_layers,
            config.pipeline_model_parallel_size,
            config.no_rope_layers,
        )
        self.kv_source = config.kv_source_layer  # the one layer whose latent is reused, or None
        for layer in self.layers:  # fixed at build: checked once, not per forward
            attn = layer.self_attention
            if attn.consumes_kv and attn.kv_source_idx != self.kv_source:
                raise RuntimeError(
                    f"layer {attn.layer_idx} reuses layer {attn.kv_source_idx}, not {self.kv_source}"
                )
        first = self.layer_indices[0] if self.layer_indices else None
        self._stage_starts_on_shared = first is not None and config.indexer_types[first] == "shared"

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None):
        """TransformerBlock's sharded state dict, with the LeRoPE ``rotary.alpha`` marked as one
        tensor replicated over the pipeline (and virtual pipeline) stages.

        Every stage's block holds the same ``alpha`` (one checkpoint tensor for the whole model).
        Megatron's default makes it a main replica on each stage (replica id ``(0, tp, dp)``), so a
        torch_dist save with PP > 1 fails validation ("Invalid access pattern"). Only the first
        stage's copy is the main replica; the others are replicas (first replica-id element = the
        stage index ``pp_rank + pp_size * vp_stage``)."""
        sharded = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        key = f"{prefix}rotary.alpha"
        sh = sharded.get(key)
        if sh is not None and hasattr(sh, "replica_id"):
            pp = getattr(getattr(self, "pg_collection", None), "pp", None)
            pp_rank, pp_size = get_pg_rank(pp), get_pg_size(pp)
            stage = pp_rank + pp_size * int(self.vp_stage or 0)
            rid = tuple(sh.replica_id) if isinstance(sh.replica_id, tuple) else (sh.replica_id,)
            sh.replica_id = (stage,) + rid[1:]
        return sharded

    # ---- pipeline carrier of the shared KV latent ------------------------------
    def _source_position(self):
        """Where this stage's layers sit around the KV source layer: ``(ran before the stage, ran by
        its end)``, both False without sharing or layers (kv_source / layer_indices are fixed)."""
        if self.kv_source is None or not self.layer_indices:
            return False, False
        return self.layer_indices[0] > self.kv_source, self.layer_indices[-1] >= self.kv_source

    @property
    def _source_ran_before_stage(self) -> bool:
        return self._source_position()[0]

    @property
    def _source_ran_by_stage_end(self) -> bool:
        return self._source_position()[1]

    def _pack_latent(self, latent, s, b, h):
        """``[s, b, r]`` latent -> ``[n, b, h]`` rows, each sample's ``s * r`` values dense, the
        tail of the last row zero (``n = kv_carrier_rows(s, r, h)``)."""
        if latent.size(0) != s or latent.size(1) != b:
            raise RuntimeError(
                f"Mistral Large 4 PP carrier: latent {tuple(latent.shape)} does not match hidden [{s}, {b}, {h}]"
            )
        n = kv_carrier_rows(s, latent.size(-1), h)
        flat = latent.transpose(0, 1).reshape(b, -1)  # [b, s * r]
        pad = n * h - flat.size(1)
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        return flat.view(b, n, h).transpose(0, 1)

    def _unpack_latent(self, rows, s):
        """Inverse of ``_pack_latent``: ``[n, b, h]`` rows -> ``[s, b, r]`` latent."""
        r = int(self.config.kv_lora_rank)
        b = rows.size(1)
        flat = rows.transpose(0, 1).reshape(b, -1)[:, : s * r]
        return flat.view(b, s, r).transpose(0, 1)

    def _expected_local_seq(self):
        """The stage-local sequence rows of the P2P tensor: ``seq_length // cp`` (``// tp`` with
        sequence parallelism), or None when the length is not fixed (``variable_seq_lengths``) or
        not known (``pp_carrier_seq_length`` unset)."""
        cfg = self.config
        seq = getattr(cfg, "pp_carrier_seq_length", None)
        if not seq or getattr(cfg, "variable_seq_lengths", False):
            return None
        pgs = getattr(self, "pg_collection", None)
        cp = pgs.cp.size() if pgs is not None else 1
        tp = pgs.tp.size() if pgs is not None and getattr(cfg, "sequence_parallel", False) else 1
        return int(seq) // (cp * tp)

    def _split_input(self, x):
        """(hidden, latent) from the received P2P tensor; latent None when the source layer has
        not run before this stage (the carrier's rows are then zeros). A stage that builds no
        layers forwards the received latent unchanged (zeros or the source's)."""
        if not self.carrier or self.pre_process:
            return x, None
        total, h = x.size(0), x.size(-1)
        r = int(self.config.kv_lora_rank)
        s = self._expected_local_seq()
        if s is None:
            # no fixed sequence length (variable_seq_lengths, or a config built without args):
            # s + ceil(s * r / h) == total: s is the largest s with that sum (unique, as rows grow with s)
            s = (total * h) // (h + r)
            while s + kv_carrier_rows(s, r, h) > total:
                s -= 1
        if s <= 0 or s + kv_carrier_rows(s, r, h) != total:
            raise RuntimeError(
                f"Mistral Large 4 PP carrier: received first dim {total}, expected s + ceil(s * {r} / {h}) "
                f"with stage-local s={s} (hidden rows followed by the shared KV latent): the "
                "megatron.mistral_large4.pp_tensor_shape patch did not size this P2P tensor"
            )
        if not self.layer_indices or self._source_ran_before_stage:
            return x[:s], self._unpack_latent(x[s:], s)
        return x[:s], None

    def _join_output(self, h, latent):
        """The P2P tensor to send: hidden states, then the densely packed latent; zero rows while
        the source layer has not run yet; an error if it ran and the latent is missing."""
        if not self.carrier or self.post_process:
            return h
        if self._source_ran_by_stage_end and latent is None:
            raise RuntimeError(
                f"Mistral Large 4 PP carrier: layer {self.kv_source}'s latent should reach later stages "
                "but this stage has none to send"
            )
        s, b, width = h.size(0), h.size(1), h.size(-1)
        if latent is None:
            rows = h.new_zeros((kv_carrier_rows(s, self.config.kv_lora_rank, width), b, width))
        else:
            rows = self._pack_latent(latent.to(h.dtype), s, b, width)
        return torch.cat((h, rows), dim=0)

    # ---- forward ---------------------------------------------------------------
    _POSITION_IDS_ERROR = (
        "Mistral Large 4 derives rope / SSMax positions itself (0..s-1 per sequence, or per "
        "document with packed THD sequences); position_ids that differ from them (reset_position_ids "
        "without packing) are not supported"
    )
    _SPARSE_MASK_ERROR = (
        "Mistral Large 4 sparse DSA attention (sequence > index_topk) takes only a causal mask; "
        "reset_attention_mask / custom masks are not supported there"
    )

    # check_position_layout guarantees that positions and masks are this block's own: it
    # refuses reset_position_ids / reset_attention_mask without packed sequences when the model
    # is built (the only Megatron settings that make them differ). The checks below are a one-time host sanity check per
    # block (first micro-batch that has the input): a Python error with the reason, and no
    # per-micro-batch device-to-host sync.
    @staticmethod
    def _assert(ok: torch.Tensor, message: str) -> None:
        """``ok`` (a bool tensor) must be True; checked on the host, a Python error otherwise."""
        if not bool(ok):
            raise NotImplementedError(message)

    @classmethod
    def _check_position_ids(cls, position_ids, pos):
        """The rope / SSMax positions are this block's own (``positions_for``: arange, its CP
        slice, or per document for THD). Refuse position ids that differ from them (e.g.
        ``reset_position_ids`` without THD packing) instead of silently using other phases.
        Every sample of the given batch is checked."""
        if position_ids is None:
            return
        ids = position_ids.reshape(-1, position_ids.size(-1))
        if ids.size(-1) != pos.numel():
            raise NotImplementedError(cls._POSITION_IDS_ERROR)
        cls._assert((ids == pos.to(device=ids.device, dtype=ids.dtype)).all(), cls._POSITION_IDS_ERROR)

    @classmethod
    def _check_sparse_mask(cls, attention_mask, s):
        """The sparse DSA path applies the causal top-k mask itself: only a plain causal
        ``attention_mask`` (Megatron's: True = masked) is accepted there (no reset_attention_mask).
        Every sample of the given batch is checked.

        No [s, s] causal tensor is built: row ``i`` is causal iff it masks exactly ``s - 1 - i``
        keys and its first masked key is ``i + 1`` (then keys ``i+1..s-1`` are masked and
        ``0..i`` are not). Both are row reductions over a zero-copy uint8 view of a bool mask, so
        the check adds O(b * s) memory, not O(s^2). A non-bool mask is accepted as Megatron's
        "nonzero = masked" and needs one bool copy of it (Megatron's masks are bool)."""
        if attention_mask is None:
            return
        if attention_mask.size(-1) != s or attention_mask.size(-2) != s:
            raise NotImplementedError(
                f"Mistral Large 4 sparse DSA attention: attention_mask of shape {tuple(attention_mask.shape)}, "
                f"sequence {s}"
            )
        m = attention_mask.bool()
        u = m.view(torch.uint8)
        rows = torch.arange(s, device=m.device)
        count_ok = (u.sum(-1, dtype=torch.int32) == (s - 1 - rows).to(torch.int32)).all()
        first = u.argmax(-1)  # first masked key (0 for a row with none)
        first_ok = (first == torch.where(rows < s - 1, rows + 1, 0)).all()
        cls._assert(count_ok & first_ok, cls._SPARSE_MASK_ERROR)

    def _context(self, hidden_states, packed_seq_params, position_ids=None, attention_mask=None):
        cfg = self.config
        s = hidden_states.size(0)
        tp = self.pg_collection.tp.size() if cfg.sequence_parallel else 1
        full = s * tp  # this CP rank's rows, after the SP gather
        cp = self.pg_collection.cp
        pos = positions_for(full, hidden_states.device, cp if cp.size() > 1 else None, packed_seq_params)
        if (
            packed_seq_params is None
            and cp.size() == 1
            and position_ids is not None
            and not getattr(self, "_checked_positions", False)
        ):
            # one-time sanity check (see _assert): check_position_layout is the guarantee
            self._check_position_ids(position_ids, pos)
            self._checked_positions = True
        freqs = self.rotary.freqs(pos) if self.rotary.dim else None
        total = full * cp.size()
        if packed_seq_params is not None and getattr(packed_seq_params, "max_seqlen_q", None):
            # THD: each document attends only to itself, so the longest document decides
            total = int(packed_seq_params.max_seqlen_q)
        sparse = total > int(cfg.index_topk)
        if sparse:
            if cp.size() > 1:
                raise NotImplementedError(
                    "Mistral Large 4 DSA sparse attention (sequence > index_topk) with context parallelism is "
                    "not supported; use context_parallel_size 1 or seq_length <= index_topk"
                )
            # one-time sanity check on the first sparse micro-batch (see _assert)
            if attention_mask is not None and not getattr(self, "_checked_sparse_mask", False):
                # the first sparse micro-batch that carries a mask
                self._check_sparse_mask(attention_mask, full)
                self._checked_sparse_mask = True
            if not getattr(self, "_logged_sparse", False):
                self._logged_sparse = True
                log_single_rank(
                    logger,
                    logging.INFO,
                    f"[Mistral-Large-4] DSA sparse attention: sequence {total} > index_topk {cfg.index_topk}",
                )
            if self._stage_starts_on_shared:
                raise RuntimeError(
                    "Mistral Large 4: a pipeline stage starting on a 'shared' indexer layer cannot "
                    "run sparse attention; start each stage on a full-indexer layer"
                )
        # log(pos + 1) in fp32 once per forward: every layer's and indexer's SSMax reads it
        log_pos = (pos.float() + 1.0).log()
        return {
            "positions": pos,
            "log_pos": log_pos,
            "freqs": freqs,
            # created here so that every layer (also through _local_ctx's copy on another device,
            # transformer/mistral_large4/attention.py) writes into the same dicts
            "topk": {},
            "keep": {},
            "sparse": sparse,
        }

    def _run_layer(self, layer, h, latent, ctx, attention_mask, packed_seq_params, padding_mask):
        attn = layer.self_attention
        attn.set_forward_context(ctx, latent if attn.consumes_kv else None)
        out, _ = layer(
            hidden_states=h,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        new_latent = attn.latent_out if attn.stores_kv else None
        attn.latent_out = None
        attn.set_forward_context(None)
        return out, new_latent

    def _chunks(self):
        """``(local layer indices, recompute)`` pairs, each run as one unit: recompute chunks
        (Megatron's uniform / block methods) or single layers; ``recompute`` says whether the unit
        is checkpointed."""
        n = len(self.layers)
        cfg = self.config
        if not (cfg.recompute_granularity == "full" and self.training):
            return [([i], False) for i in range(n)]
        num = int(cfg.recompute_num_layers)
        if cfg.recompute_method == "uniform":
            return [(list(range(i, min(i + num, n))), True) for i in range(0, n, num)]
        return [([i], i < num) for i in range(n)]  # block: the first num layers of the stage

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        padding_mask=None,
        extract_layer_indices=None,
        position_ids=None,
        *,
        inference_params=None,
        dynamic_inference_decode_only=None,
    ):
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("Mistral Large 4: KV-cache inference is not supported")
        if context is not None or context_mask is not None:
            raise NotImplementedError("Mistral Large 4 is decoder-only: no cross-attention context")
        if any(t is not None for t in (rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin)):
            raise ValueError("Mistral Large 4 computes its own YaRN + LeRoPE rope: no rotary_pos_* inputs")
        if attention_bias is not None or sequence_len_offset is not None:
            raise NotImplementedError(
                "Mistral Large 4: attention_bias / sequence_len_offset are not supported"
            )
        if extract_layer_indices:
            raise NotImplementedError("Mistral Large 4: extract_layer_indices is not supported")
        if not self.pre_process:
            hidden_states = self.input_tensor
        hidden_states, latent = self._split_input(hidden_states)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        ctx = self._context(hidden_states, packed_seq_params, position_ids, attention_mask)
        rng = (
            tensor_parallel.get_cuda_rng_tracker().fork() if self.config.sequence_parallel else nullcontext()
        )
        with rng:
            for idxs, recompute in self._chunks():
                layers = [self.layers[i] for i in idxs]
                consumes = any(layer.self_attention.consumes_kv for layer in layers)
                stores = any(layer.self_attention.stores_kv for layer in layers)

                # top-k of earlier chunks that this chunk's shared layers reuse: captured now, and
                # restored when run() starts, so a full-recompute backward (chunks recomputed last
                # first, after later full layers replaced ctx's one entry) sees the same top-k
                own = {layer.self_attention.layer_idx for layer in layers}
                incoming = {
                    k: ctx["topk"][k]
                    for k in {layer.self_attention.index_source_idx for layer in layers} - own
                    if k in ctx["topk"]
                }

                def run(h, *lat, _layers=layers, _stores=stores, _incoming=incoming):
                    lat = lat[0] if lat else None
                    if any(ctx["topk"].get(k) is not t for k, t in _incoming.items()):
                        ctx["topk"].clear()
                        ctx["keep"].clear()
                        ctx["topk"].update(_incoming)  # the keep mask is rebuilt from it
                    for layer in _layers:
                        h, new_lat = self._run_layer(
                            layer, h, lat, ctx, attention_mask, packed_seq_params, padding_mask
                        )
                        if layer.self_attention.stores_kv:
                            lat = new_lat
                    return (h, lat) if _stores else h

                args = (hidden_states, latent) if consumes and latent is not None else (hidden_states,)
                if recompute:
                    out = tensor_parallel.checkpoint(run, self.config.distribute_saved_activations, *args)
                else:
                    out = run(*args)
                if stores:
                    hidden_states, latent = out
                else:
                    hidden_states = out
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
        out = self._join_output(hidden_states, latent)
        if not self.pre_process and len(self.layers) == 0 and self.final_layernorm is None:
            # empty pass-through stage: do not return the received P2P buffer itself (the schedule
            # reuses / frees it)
            out = out.clone()
        return make_viewless_tensor(inp=out, requires_grad=True, keep_graph=True)
