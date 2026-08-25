###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Ideogram-4 flow-matching adapter + no-fork registration for the AutoModel
diffusion recipe.

WHY (no Automodel/diffusers fork):
  AutoModel selects a flow-matching model adapter by name via
  ``nemo_automodel.components.flow_matching.pipeline.create_adapter`` (a closed
  dict of {"flux","flux2","wan","hunyuan","qwen_image","simple"}). Ideogram-4 is
  not in that dict. Rather than edit the submodule, ``install()`` wraps
  ``create_adapter`` at runtime so ``flow_matching.adapter_type: ideogram4``
  resolves to :class:`Ideogram4Adapter`. The recipe imports ``create_adapter`` by
  name (``from ...pipeline import create_adapter``), so we patch it in BOTH the
  pipeline module and the already-imported recipe module namespace.

WHAT the adapter does:
  Ideogram-4 is a single-stream flow-matching DiT: text-conditioning tokens and
  patchified image latents live in ONE packed sequence, distinguished by a
  per-token ``indicator`` and joined by a block-diagonal mask (``segment_ids``),
  with 3-axis MRoPE (``position_ids``). This adapter maps AutoModel's latent-space
  flow-matching convention (noise ``x_t=(1-sigma)x0+sigma eps``; target
  ``v=eps-x0`` in the 128-dim packed-latent space) onto that packed contract:

  - ``prepare_inputs`` folds the micro-batch's ``N`` samples into ``B = N // pack_size``
    packed rows. It packs image latents ``[N,128,H_p,W_p] -> [N,n_img,128]``, scatters them
    and the left-padded text features into ``hidden_states [B,S,128]`` /
    ``encoder_hidden_states [B,S,53248]``, and takes the
    ``position_ids/segment_ids/indicator`` from :func:`build_packed_layout`. Ideogram model
    time is ``t = 1 - sigma``. It ALSO builds the var-len ``cu_seqlens`` packing on the host
    (see ``packing.py``), so the processor never derives it from the mask inside the
    compiled region -- that derivation is data-dependent, host-syncing, and under FSDP2
    its graph break desyncs the per-layer collectives. Disable with
    ``PRIMUS_IDEOGRAM_PRECOMPUTE_CU_SEQLENS=0``.
  - ``forward`` publishes that packing into the attention modules' shared buffer
    (``ideogram4_packing_buffer``, which documents why that route and not diffusers'
    ``attention_kwargs``), runs the DiT, GATHERS each sample's image-token velocity out of
    its row, unpacks to ``[N,128,H_p,W_p]`` and NEGATES it: the DiT predicts ``x0 - eps``
    (inference feeds ``-v`` to the scheduler) while AutoModel's target is ``eps - x0``.

MULTI-SAMPLE PACKING (``pack_size`` > 1):
  ``N`` stays the leading dimension on BOTH sides of the adapter, so ``FlowMatchingPipeline``
  never learns that packing happened: its ``sigma``, target, loss and loss weighting are all
  still per-sample and need no change. Only the transformer sees rows. Two consequences worth
  knowing:

    * ``timestep`` becomes per-TOKEN ``(B,S)`` once a row holds samples at different
      flow-matching times. At ``pack_size == 1`` it stays per-sample ``(B,)``, both because
      the model then does the adaln projection once instead of ``S`` times and because the
      context-parallel plan asserts that shape.
    * ``PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE`` is refused outright for a packed batch. Dense
      attention over a packed row lets neighbouring SAMPLES attend to each other, which is
      the archetypal packing corruption: no error, descending loss, ruined model.

  ``pack_size`` and ``text_budget`` are read off the BATCH, not from adapter kwargs -- see
  :meth:`_resolve_packing` for why that is the only arrangement in which the loader and the
  adapter cannot disagree.

  Batch keys consumed (from the Ideogram-4 preprocessor / cache):
    - ``image_latents``: ``[N,128,H_p,W_p]`` patchified + BN-normalized latents.
    - ``llm_features``: ``[N,text_capacity,53248]`` left-padded Qwen3-VL 13-layer feats.
    - ``text_lengths``: ``[N]`` int, real (non-pad) text token count per sample.
    - ``pack_size`` / ``text_budget``: optional ints; absent means the unpacked layout.

  ``install()`` is additive and safe: it only adds the ``ideogram4`` route, so a
  FLUX/Wan run is unaffected. Idempotent.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

from primus.backends.nemo_automodel.models.ideogram4.packing_buffer import publish_packing
from primus.backends.nemo_automodel.models.ideogram4.attention import (
    assume_dense_enabled,
    precompute_cu_seqlens_active,
)

# The layout constants and both packing builders live in ``packing.py`` now that a row can
# hold more than one sample. Re-exported here because they are part of this module's public
# surface (tests and the runbook import them from the adapter).
from primus.backends.nemo_automodel.models.ideogram4.packing import (  # noqa: F401
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    PackedLayout,
    build_cu_seqlens,
    build_packed_layout,
    derive_text_budget,
)

logger = logging.getLogger(__name__)

_logged: set = set()


def _log_once(key: str, level: int, msg: str, *args) -> None:
    """Log once per process. Called from the per-step path, so it must not repeat."""
    if key not in _logged:
        _logged.add(key)
        logger.log(level, msg, *args)


def _base_adapter_cls():
    """Import AutoModel's ModelAdapter base lazily (submodule must be importable)."""
    from nemo_automodel.components.flow_matching.adapters.base import ModelAdapter

    return ModelAdapter


# Defined as a factory so importing this module never requires nemo_automodel until
# install()/first use — keeps the module import-safe in any context.
def _build_ideogram4_adapter_class():
    ModelAdapter = _base_adapter_cls()

    class Ideogram4Adapter(ModelAdapter):
        """Model adapter for Ideogram-4 single-stream flow-matching T2I."""

        def __init__(self, in_channels: int = 128, predict_negative_velocity: bool = True):
            self.in_channels = in_channels
            self.predict_negative_velocity = predict_negative_velocity

        @staticmethod
        def _prepare_ids(
            text_lengths: List[int],
            grid_h: int,
            grid_w: int,
            max_text_tokens: int,
            device: torch.device,
        ):
            """Build ``[left-pad][text][image]`` position/segment/indicator tensors.

            Mirrors ``Ideogram4Pipeline._prepare_ids`` so training == inference layout.
            """
            batch_size = len(text_lengths)
            num_image_tokens = grid_h * grid_w
            total_seq_len = max_text_tokens + num_image_tokens

            h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
            w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
            t_idx = torch.zeros_like(h_idx)
            image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

            position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
            segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
            indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

            for b, num_text in enumerate(text_lengths):
                num_text = int(num_text)
                offset = max_text_tokens - num_text
                text_pos = torch.arange(num_text)
                text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
                position_ids[b, offset : offset + num_text] = text_pos_3d
                position_ids[b, offset + num_text :] = image_pos
                indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
                indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR
                segment_ids[b, offset : offset + num_text + num_image_tokens] = 1

            return position_ids.to(device), segment_ids.to(device), indicator.to(device)

        def _pack_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
            b, c, h, w = latents.shape
            return latents.reshape(b, c, h * w).permute(0, 2, 1).contiguous()

        def _unpack_image_latents(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
            b, _, c = tokens.shape
            return tokens.permute(0, 2, 1).contiguous().reshape(b, c, h, w)

        def _resolve_packing(self, batch, text_capacity: int, num_samples: int):
            """Decide ``(pack_size, text_budget)`` for this micro-batch.

            The BATCH is authoritative, not adapter kwargs. ``pack_size`` determines which
            samples end up adjacent in a row, which is a decision only the batch sampler can
            make (it is the one component that sees enough of the dataset to keep every row
            inside the budget). Reading it back off the batch makes it impossible for the
            loader and the adapter to disagree -- a disagreement would not raise, it would
            just mislabel every token.

            A batch without these keys is the unpacked layout: one sample per row and a
            budget of ``text_capacity + 1``, the reserved-slack convention that the
            one-sample path already uses. So every existing dataloader config keeps working
            untouched.
            """
            pack_size = int(batch.get("pack_size", 1) or 1)
            text_budget = int(batch.get("text_budget", 0) or 0)
            if text_budget <= 0:
                # +1 is the reserved slack slot: a caption filling the full width must still
                # leave one, or the row collapses to K segments instead of K+1 and cu_seqlens
                # changes shape.
                text_budget = pack_size * text_capacity + 1
                if pack_size > 1:
                    _log_once(
                        "derived_budget",
                        logging.WARNING,
                        "[PrimusIdeogram4] pack_size=%d but the batch carries no text_budget, so "
                        "it defaults to pack_size*text_capacity+1=%d. That is always feasible but "
                        "saves nothing -- it is K copies of the unpacked row. Set text_budget on "
                        "the dataloader config to actually reclaim the caption padding.",
                        pack_size,
                        text_budget,
                    )
            if num_samples % pack_size:
                raise ValueError(
                    f"pack_size={pack_size} does not divide the micro-batch size {num_samples}. "
                    "The row count would then vary between steps and recompile the graph; make "
                    "local_batch_size a multiple of pack_size and keep drop_last=true."
                )
            return pack_size, text_budget

        def prepare_inputs(self, context) -> Dict[str, Any]:
            batch = context.batch
            device = context.device
            dtype = context.dtype

            noisy = context.noisy_latents
            if noisy.ndim != 4:
                raise ValueError(
                    f"Ideogram4Adapter expects 4D patchified latents [B, C, H_p, W_p], got {noisy.ndim}D"
                )
            num_samples, C, H_p, W_p = noisy.shape
            if C != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} packed channels, got {C}")

            img_tokens = self._pack_image_latents(noisy)  # [N, n_img, 128]

            llm_features = batch["llm_features"].to(device, dtype=dtype, non_blocking=True)
            if llm_features.ndim == 2:
                llm_features = llm_features.unsqueeze(0)
            # Text capacity as produced by the dataloader: the width of the left-padded text
            # axis, NOT the row's text budget.
            text_capacity = llm_features.shape[1]
            feature_dim = llm_features.shape[-1]

            text_lengths = batch.get("text_lengths")
            if text_lengths is None:
                text_lengths = [text_capacity] * num_samples
            elif torch.is_tensor(text_lengths):
                text_lengths = text_lengths.tolist()
            text_lengths = [int(t) for t in text_lengths]

            pack_size, text_budget = self._resolve_packing(batch, text_capacity, num_samples)

            # ASSUME_DENSE tells the attention processor to skip the block-diagonal analysis
            # and run dense flash over the whole row. That is exact only when a row is a
            # single segment. Two ways it is not: a ragged batch (padding would attend), and
            # ANY packed row, where dense flash would let neighbouring SAMPLES attend to each
            # other -- the archetypal packing corruption, silent and loss-descending. The
            # lengths are already on the host here, so refusing costs nothing.
            if assume_dense_enabled():
                if pack_size > 1:
                    raise ValueError(
                        "PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE=1 cannot be combined with pack_size="
                        f"{pack_size}. A packed row holds {pack_size} independent samples; dense "
                        "flash ignores the segment boundaries and would let them attend to each "
                        "other, mixing unrelated images into every prediction with no error "
                        "raised. Unset the flag, or set pack_size=1."
                    )
                distinct = sorted(set(text_lengths))
                if len(distinct) > 1:
                    raise ValueError(
                        "PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE=1 requires every sample in the batch to "
                        f"have the same text length, but this batch has lengths {distinct}. Dense "
                        "flash would let padding tokens attend and silently corrupt training. Unset "
                        "the flag to use the exact var-len path, or pin "
                        "min_text_tokens == max_text_tokens."
                    )

            # One host-side pass produces every boundary tensor and the three index tensors
            # that fold N samples into B rows. Pure integer arithmetic -- no tensor value is
            # read -- which is what keeps all of this a graph INPUT.
            layout = build_packed_layout(
                text_lengths,
                pack_size=pack_size,
                text_budget=text_budget,
                grid_h=H_p,
                grid_w=W_p,
                text_capacity=text_capacity,
                device=device,
            )
            rows, seq_len = layout.num_rows, layout.seq_len

            if context.cfg_dropout_prob > 0.0:
                # Per SAMPLE, not per row: with pack_size > 1 a row holds several independent
                # samples and dropping a whole row would correlate their conditioning.
                drop = torch.rand(num_samples, 1, 1, device=device) < context.cfg_dropout_prob
                llm_features = llm_features.masked_fill(drop, 0.0)

            # Scatter the image latents to their slots. ``image_dst`` is a bijection onto the
            # image positions (every image token is real), so no dustbin row is needed here.
            hidden_states = torch.zeros(rows * seq_len, C, device=device, dtype=dtype)
            hidden_states.index_add_(0, layout.image_dst, img_tokens.reshape(-1, C))
            hidden_states = hidden_states.view(rows, seq_len, C)

            # Same for the text features, except the incoming tensor is LEFT-PADDED, so its
            # pad slots have no destination. They are aimed at one extra dustbin row which is
            # then sliced off. ``index_add_`` rather than ``index_copy_`` because several pad
            # slots share that dustbin index, and index_copy_ with duplicate indices is
            # undefined; accumulating zeros into a row we discard is well defined and free.
            encoder = torch.zeros(rows * seq_len + 1, feature_dim, device=device, dtype=dtype)
            encoder.index_add_(0, layout.text_dst, llm_features.reshape(-1, feature_dim))
            encoder_hidden_states = encoder[: rows * seq_len].view(rows, seq_len, feature_dim)

            # Ideogram model time: 0=noise, 1=data => t = 1 - sigma.
            sample_time = (1.0 - context.sigma).to(dtype)
            if pack_size == 1:
                # Per-SAMPLE (B,), which the model unsqueezes to (B,1,...) so the adaln
                # projections run on one position instead of S identical copies. Keeping this
                # shape at pack_size=1 is not just cosmetic: the per-token form below makes
                # adaln_proj do S times the work for the same numbers, and it is what the
                # context-parallel plan's unsplit-timestep assumption is checked against.
                timestep = sample_time
            else:
                # Per-TOKEN (B,S). A packed row holds samples at DIFFERENT flow-matching
                # times, so a per-row timestep cannot express it. The transformer supports
                # this shape natively (it only unsqueezes when timestep.dim() == 1); slack
                # tokens get 0 and are masked out of every segment anyway.
                padded_time = torch.cat([sample_time, sample_time.new_zeros(1)])
                timestep = padded_time[layout.token_sample].view(rows, seq_len)

            return {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "position_ids": layout.position_ids,
                "segment_ids": layout.segment_ids,
                "indicator": layout.indicator,
                "_layout": layout,
                "_h_p": H_p,
                "_w_p": W_p,
                # Built on the HOST and published by ``forward`` into the attention modules'
                # shared buffer, so that single copy_ is the only host->device transfer.
                "_cu_seqlens": layout.cu_seqlens if precompute_cu_seqlens_active() else None,
                "_max_seqlen": layout.max_seqlen,
            }

        def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
            layout: PackedLayout = inputs.pop("_layout")
            h_p = inputs.pop("_h_p")
            w_p = inputs.pop("_w_p")
            cu_seqlens = inputs.pop("_cu_seqlens", None)
            max_seqlen = inputs.pop("_max_seqlen", None)

            # Publish the packing onto the attention modules, which is where the processor
            # reads it (``ideogram4_packing_buffer``). Diffusers' ``attention_kwargs`` cannot
            # carry it: the model never forwards it to the blocks in 0.39.0, so passing it was
            # a silent no-op. Two ordering rules, both silent when broken: this must happen
            # BEFORE the model call, and nothing may republish between the forward and its
            # backward -- per-layer compile lives inside the checkpoint wrapper, so the block
            # re-reads this buffer during the backward recompute.
            if cu_seqlens is not None:
                publish_packing(
                    model,
                    cu_seqlens,
                    max_seqlen,
                    device=inputs["hidden_states"].device,
                    # Segments per row is K+1, so the processor can tell a stale packing from
                    # a current one by shape alone -- a static comparison, no host sync.
                    segments_per_row=layout.segments_per_row,
                    # Having built a packing, a model that cannot read it is a misconfiguration,
                    # not a fallback: every layer would derive its own from the mask, and on a
                    # subset of ranks that quietly averages two attention paths into one
                    # gradient. Non-zero ranks log nothing after init, so raising is the only
                    # way this becomes visible.
                    required=True,
                )

            out = model(
                hidden_states=inputs["hidden_states"],
                timestep=inputs["timestep"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
                return_dict=False,
            )
            pred = self.post_process_prediction(out)  # [B, S, 128]

            # Gather each sample's image tokens back out of its row. With one sample per row
            # this is the old ``pred[:, max_text:]`` slice; with several it is the only way to
            # collect them, since each sample's image block starts at a different offset. The
            # result restores the leading dim to N, which is what makes the flow-matching
            # pipeline's target, sigma, and loss weighting line up with no change on its side.
            channels = pred.shape[-1]
            img_pred = pred.reshape(-1, channels).index_select(0, layout.image_dst)
            img_pred = img_pred.view(layout.num_samples, layout.num_image_tokens, channels)
            unpacked = self._unpack_image_latents(img_pred, h_p, w_p)  # [N, 128, H_p, W_p]

            # DiT predicts x0 - eps; AutoModel target is eps - x0.
            return -unpacked if self.predict_negative_velocity else unpacked

    return Ideogram4Adapter


# Cache the built class so identity is stable across calls.
_IDEOGRAM4_ADAPTER_CLS = None


def get_ideogram4_adapter_class():
    global _IDEOGRAM4_ADAPTER_CLS
    if _IDEOGRAM4_ADAPTER_CLS is None:
        _IDEOGRAM4_ADAPTER_CLS = _build_ideogram4_adapter_class()
    return _IDEOGRAM4_ADAPTER_CLS


def install() -> bool:
    """Route ``adapter_type == "ideogram4"`` to the Ideogram-4 adapter via a no-fork
    wrapper around AutoModel's ``create_adapter``.

    Additive and idempotent; never changes existing adapter behavior. Returns True.
    """
    import nemo_automodel.components.flow_matching.pipeline as P

    orig = P.create_adapter
    if getattr(orig, "_ideogram4_patched", False):
        return True

    def create_adapter_patched(adapter_type: str, **kwargs):
        if adapter_type == "ideogram4":
            return get_ideogram4_adapter_class()(**kwargs)
        return orig(adapter_type, **kwargs)

    create_adapter_patched._ideogram4_patched = True
    P.create_adapter = create_adapter_patched

    # The recipe does ``from ...pipeline import create_adapter`` (bound by name), so
    # patch its module namespace too if it is already imported.
    try:
        import nemo_automodel.recipes.diffusion.train as T

        if getattr(T, "create_adapter", None) is orig:
            T.create_adapter = create_adapter_patched
    except Exception as exc:  # pragma: no cover
        logger.debug("[PrimusIdeogram] recipe namespace patch skipped: %s", exc)

    logger.info("[PrimusIdeogram] Registered 'ideogram4' flow-matching adapter (no-fork).")
    return True
