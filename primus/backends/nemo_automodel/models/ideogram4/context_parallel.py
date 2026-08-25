###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Context parallelism (Ulysses) for Ideogram-4 in the AutoModel diffusion recipe.

WHY:
  AutoModel's diffusion CP support delegates to diffusers' ContextParallelConfig
  and refuses to enable CP unless the transformer declares a ``_cp_plan``
  (``_diffusers/auto_diffusion_pipeline.py``). Every other piece CP needs -- the
  mesh with a ``cp`` axis, FSDP2 grad sync over ``dp_shard_cp``, the per-DP-rank
  RNG reseed, the ``loss * cp_size`` correction -- already exists upstream.
  ``Ideogram4Transformer2DModel`` simply has no plan, so CP is unreachable for it
  and for no other reason. This module supplies one, Primus-side: diffusers stays
  pristine.

THE PLAN, and why each entry is what it is:
  Ideogram-4 is a SINGLE-STREAM model. ``hidden_states``, ``encoder_hidden_states``,
  ``position_ids``, ``segment_ids`` and ``indicator`` are all length-L along the
  *same* packed ``[left-pad][text][image]`` sequence axis -- there is no separate
  text stream to leave alone (contrast QwenImage, which splits an encoder stream
  of its own). So every per-token input that is combined elementwise with
  ``hidden_states`` has to be split with it, or the shapes stop matching the moment
  the root forward multiplies by the indicator masks.

  * ``hidden_states`` / ``encoder_hidden_states`` / ``indicator`` -- split at the
    ROOT. The root forward derives ``llm_token_mask`` / ``output_image_mask`` from
    ``indicator`` and applies them to the other two before the first block, so all
    three must be sliced at the same point.
  * ``segment_ids`` -- deliberately NOT split. See the mask note below; this is the
    one entry whose absence is load-bearing.
  * ``position_ids`` -- also not split. Instead the MRoPE module's two OUTPUTS are
    split (``split_output=True``), which is the Wan/QwenImage pattern: let the layer
    see the full positions, then slice what it produced. RoPE is applied to q/k
    *before* the Ulysses all-to-all, so cos/sin must end up LOCAL and aligned with
    the local tokens. Splitting the output rather than the input also keeps MRoPE's
    autocast-disabled fp32 arithmetic (needed because image positions start at
    65536) operating on exactly the tensor it was written for.
  * ``timestep`` -- not split, and must not be. Our adapter passes it per-sample,
    shape ``(B,)``, which the root forward unsqueezes to ``(B, 1, ...)`` and
    broadcasts. Wan splits its timestep only because Wan's is per-token. ``install``
    below is a no-op for that difference, so ``_assert_timestep_is_per_sample``
    guards it instead of letting a per-token timestep silently mis-slice.
  * ``final_layer`` -- the gather point (Ideogram's equivalent of the ``proj_out``
    that Flux/Wan/QwenImage gather at). Nothing happens after it but the output
    wrapper, so gathering there restores full-length ``(B, L, out_channels)``.

WHY ``segment_ids`` STAYS WHOLE:
  ``TemplatedUlyssesAttention`` computes ``H_LOCAL = H // world_size`` and then does
  the all-to-all, so after the exchange each rank holds the FULL sequence with a
  SUBSET of heads. Attention therefore wants a full-length mask. The root forward
  builds ``(segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)``,
  which is full-length ``(B, 1, L, L)`` exactly when ``segment_ids`` is left unsplit
  while ``hidden_states`` is split. Splitting it too would produce an
  ``(B, 1, L_local, L_local)`` mask that silently drops all cross-shard attention.
  diffusers anticipates the other convention as well (a mask arriving at
  ``S_KV_LOCAL`` gets all-gathered back), so both shapes are survivable upstream --
  which is precisely why getting it wrong would not raise. Verify by A/B-ing CP vs
  non-CP logits on a RAGGED batch before trusting any loss curve.

WHY THE VAR-LEN PROCESSOR IS REFUSED (``_assert_varlen_not_installed``):
  ``Ideogram4VarlenAttnProcessor`` calls ``aiter.flash_attn_varlen_func`` directly
  and never reaches diffusers' ``dispatch_attention_fn`` -- which is where the
  Ulysses all-to-all lives. Combined with CP the result is not an error: the plan's
  hooks still split the inputs, the processor then attends only WITHIN each rank's
  shard, the output is gathered, and training proceeds on a plausible-looking loss
  curve. Its precomputed ``cu_seqlens`` describes global segment boundaries on top
  of that, so the segment map is wrong as well. diffusers' own CP-capability check
  cannot catch this, because we bypassed the dispatch it guards. Hence the explicit
  refusal here. The two are an either/or until an aiter CP forward/backward op pair
  exists.

CONSTRAINTS worth knowing before configuring a run:
  * Ulysses only -- ring degree > 1 is rejected upstream (broken backward in
    diffusers <= 0.39).
  * FSDP2 only -- DDP + CP is rejected upstream, so this and the DDP/ZeRO-1 path
    are mutually exclusive.
  * Ulysses shards HEADS, so ``cp_size`` must divide both the world size and the
    head count. Ideogram-4 has 18 heads, so on one 8-GPU node the only usable
    degrees are 1 and 2: ``cp_size=2, dp_size=4``.
  * The post-patchify sequence length must be divisible by ``cp_size``
    (equipartition sharding).

Installed unconditionally: attaching a class attribute costs nothing and is inert
until a config asks for ``cp_size > 1``, so this needs no environment flag.
"""

import logging

logger = logging.getLogger(__name__)


def _build_cp_plan():
    """Build the diffusers CP plan. Imported lazily so this module stays importable
    without diffusers present (the trainer installs hooks defensively)."""
    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelOutput,
    )

    return {
        # Root forward inputs. All three are per-token on the same packed axis and
        # are combined with each other before the first block, so they split together.
        # segment_ids and position_ids are absent on purpose -- see the module docstring.
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "indicator": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        # MRoPE returns (cos, sin), each (B, L, head_dim). Split what it produced,
        # not what it consumed, so RoPE is applied to the local tokens pre-all-to-all.
        "rotary_emb": {
            0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
        },
        # Gather back to full length; nothing follows this but the output wrapper.
        "final_layer": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }


def _assert_varlen_not_installed() -> None:
    """Refuse CP + the var-len flash processor. Silent corruption otherwise."""
    from primus.backends.nemo_automodel.models.ideogram4.attention import (
        is_varlen_attn_enabled,
    )

    if is_varlen_attn_enabled():
        raise ValueError(
            "Context parallelism (cp_size > 1) and PRIMUS_IDEOGRAM_VARLEN_ATTN=1 cannot be "
            "used together. The var-len processor calls aiter directly and never reaches "
            "diffusers' dispatch_attention_fn, which is where the Ulysses all-to-all happens, "
            "so each rank would attend only within its own sequence shard and training would "
            "produce wrong gradients WITHOUT raising. Unset PRIMUS_IDEOGRAM_VARLEN_ATTN to run "
            "CP on the stock SDPA path, or set cp_size=1 to keep the var-len path. Note this "
            "also rules out CP + multi-sample packing (pack_size > 1), which depends on the "
            "var-len path to isolate the samples sharing a row."
        )


def _timestep_pre_hook(module, args, kwargs):
    """Assert the plan's unsplit-``timestep`` assumption still holds, once per forward.

    Registered on the INSTANCE when CP is switched on, deliberately not by wrapping
    ``forward``: diffusers resolves the root plan's named inputs against the forward
    signature, so a ``*args, **kwargs`` wrapper would hide ``encoder_hidden_states``
    and ``indicator`` from it and quietly leave them unsplit.
    """
    timestep = kwargs.get("timestep")
    if timestep is None and len(args) >= 2:
        timestep = args[1]
    _assert_timestep_is_per_sample(timestep)
    return None


def _assert_timestep_is_per_sample(timestep) -> None:
    """The plan leaves ``timestep`` unsplit, which is only correct while it is ``(B,)``.

    This is also what catches CP + multi-sample packing: a packed row holds samples at
    different flow-matching times, so the adapter necessarily switches ``timestep`` to the
    per-token ``(B, S)`` form, and that arrives here.
    """
    if timestep is not None and hasattr(timestep, "dim") and timestep.dim() != 1:
        raise ValueError(
            f"Context parallelism expects a per-sample timestep of shape (B,), got "
            f"{tuple(timestep.shape)}. A per-token timestep is broadcast against the "
            "sequence axis and would need splitting in the CP plan (as Wan does); it is "
            "left unsplit here because the Ideogram adapter passes t = 1 - sigma per sample. "
            "If this came from pack_size > 1: multi-sample packing and context parallelism "
            "are not usable together yet. Packing gives each sample in a row its own "
            "timestep, which this plan cannot split, and it needs the var-len processor that "
            "CP already refuses. Set pack_size=1 to run CP, or cp_size=1 to run packing."
        )


def install(model=None) -> bool:
    """Attach a diffusers context-parallel plan to Ideogram-4 (no-fork).

    Sets the ``_cp_plan`` class attribute so AutoModel's CP path accepts the model,
    and wraps ``enable_parallelism`` -- which upstream calls only when
    ``cp_size > 1`` -- so the incompatible var-len combination and a per-token
    timestep are rejected at the moment CP is actually switched on rather than
    silently miscomputing. Idempotent. Modifies NO Automodel/diffusers source.
    """
    try:
        from diffusers.models.transformers.transformer_ideogram4 import (
            Ideogram4Transformer2DModel,
        )
    except ImportError:
        return False

    if getattr(Ideogram4Transformer2DModel, "_primus_cp_installed", False):
        return False

    Ideogram4Transformer2DModel._cp_plan = _build_cp_plan()

    original_enable_parallelism = Ideogram4Transformer2DModel.enable_parallelism

    def enable_parallelism(self, *args, **kwargs):
        _assert_varlen_not_installed()
        result = original_enable_parallelism(self, *args, **kwargs)
        self.register_forward_pre_hook(_timestep_pre_hook, with_kwargs=True)
        return result

    Ideogram4Transformer2DModel.enable_parallelism = enable_parallelism
    Ideogram4Transformer2DModel._primus_cp_installed = True

    logger.info(
        "[PrimusIdeogramCP] Attached diffusers context-parallel plan to "
        "Ideogram4Transformer2DModel (Ulysses; cp_size must divide both the world size "
        "and the 18 attention heads, so cp_size=2 is the only non-trivial degree on an "
        "8-GPU node)."
    )
    return True
