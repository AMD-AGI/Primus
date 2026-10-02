###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Context parallelism (Ulysses) for Ideogram-4.

WHY THIS IS ALL IT TAKES:
  AutoModel's diffusion context-parallel support delegates to diffusers' parallel
  config, which refuses to enable CP unless the transformer declares a ``_cp_plan``.
  Everything else CP needs already exists upstream: the device mesh with a ``cp``
  axis, FSDP2 gradient synchronization over the combined shard-and-CP dimension,
  the per-data-parallel-rank generator reseed, and the loss correction for the
  extra ranks now sharing a sample. Ideogram-4 simply has no plan, and that -- not
  anything missing in the machinery -- is the only reason CP is unreachable for it.

  So this module supplies a plan, from Primus, leaving diffusers untouched.

THE PLAN, ENTRY BY ENTRY. Ideogram-4 is a SINGLE-STREAM model: hidden states, the
encoder features, the position ids, the segment ids and the indicator are all
length-L along the SAME packed sequence axis. There is no separate text stream to
leave alone, unlike the dual-stream models whose plans this otherwise resembles.
Every per-token input combined elementwise with the hidden states therefore has to
be split with them, or shapes stop matching the moment the root forward applies
the indicator-derived masks.

  * hidden_states, encoder_hidden_states, indicator -- split at the ROOT. The root
    forward derives the token masks from the indicator and applies them to the
    other two before the first block, so all three have to be sliced at the same
    point.

  * segment_ids -- deliberately NOT split. This is the one entry whose ABSENCE is
    load-bearing; see below.

  * position_ids -- also not split. Instead the two OUTPUTS of the rotary embedding
    are split. Rotary embeddings are applied to queries and keys BEFORE the Ulysses
    exchange, so the cosines and sines have to end up local and aligned with the
    local tokens. Splitting the output rather than the input also leaves the
    module's deliberately float32, autocast-disabled arithmetic operating on
    exactly the tensor it was written for -- it needs that precision because image
    positions start at a large offset, far enough out that float16 cannot represent
    adjacent positions distinctly.

  * timestep -- not split, and must not be. The adapter passes it per sample, shape
    (B,), which the root forward unsqueezes and broadcasts. Models whose plans DO
    split the timestep do so because theirs is per token. Nothing about installing
    the plan would notice the difference, so a forward pre-hook asserts it rather
    than letting a per-token timestep be mis-sliced.

  * final_layer -- the gather point, this model's equivalent of the output
    projection the other plans gather at. Nothing follows it but the output
    wrapper, so gathering there restores the full-length result.

WHY segment_ids STAYS WHOLE, since this is the subtle one:
  Ulysses attention computes a local head count and then does the all-to-all, so
  AFTER the exchange each rank holds the FULL sequence with a SUBSET of the heads.
  Attention therefore wants a full-length mask. The root forward builds the mask by
  comparing segment ids against themselves, which comes out full-length exactly
  when segment_ids is left unsplit while the hidden states are split. Splitting it
  too would produce a mask of local-by-local size. diffusers all-gathers a mask that
  arrives at local key-value length, which turns that into a local-by-full mask
  against full-length queries. With diffusers 0.39 and the native backend, the first
  forward then fails with a shape error. That error comes from diffusers' current
  mask handling, not from anything this module guarantees, so the convention is
  stated here and pinned by a test instead.

WHY THE VARIABLE-LENGTH PROCESSOR IS REFUSED:
  That processor calls the flash-attention kernel directly and never reaches
  diffusers' attention dispatch -- which is where the Ulysses all-to-all lives.
  Combined with CP the result is not an error: the plan's hooks still split the
  inputs, the processor then attends only WITHIN each rank's shard, the output is
  gathered, and training proceeds on a loss curve that looks plausible. Its
  precomputed sequence boundaries describe global segments on top of that, so the
  segment map is wrong as well. diffusers' own capability check cannot catch any of
  this, because the dispatch it guards was bypassed. Hence the explicit refusal
  here. The two remain an either/or until a context-parallel forward and backward
  pair exists for that kernel.

CONSTRAINTS WORTH KNOWING BEFORE CONFIGURING A RUN, none of them imposed here:
  * Ulysses only. A ring degree above one is rejected upstream.
  * FSDP2 only. DDP with CP is rejected upstream, so this and the DDP/ZeRO-1 path
    are mutually exclusive.
  * Ulysses shards HEADS, so the CP degree has to divide both the world size and
    the attention head count. The head count is not a power of two for this model,
    which is the binding constraint: on a single eight-GPU node it leaves 2 as the
    only usable degree above one, alongside a data-parallel degree of 4.
  * The packed sequence is equipartitioned, so its length has to be divisible by the
    CP degree. This module left-pads the adapter's packed batch to that multiple
    once CP is switched on; nothing in the config has to be sized by hand.

Installed whenever the run is an Ideogram-4 one, with no setting of its own:
attaching a class attribute costs nothing and stays inert until a config asks
for a CP degree above one.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusIdeogramCP]"

# 1 until enable_parallelism runs, which upstream calls only for a degree above
# one. The adapter reads this to left-pad the packed sequence; the plan itself
# cannot, because the tensors are already built by the time its hooks run.
_SEQUENCE_MULTIPLE = 1


def sequence_multiple() -> int:
    """Packed-sequence multiple required by the enabled context-parallel degree."""
    return _SEQUENCE_MULTIPLE


def _degree_of(config) -> int:
    """Ring times Ulysses, which is the equipartition the plan will apply."""
    if config is None:
        return 1
    ring = int(getattr(config, "ring_degree", 1) or 1)
    ulysses = int(getattr(config, "ulysses_degree", 1) or 1)
    return max(1, ring * ulysses)


_pad_logged = False


def pad_prepared_inputs(inputs):
    """Left-pad a prepared Ideogram batch so its packed length is divisible by the CP degree.

    The adapter has already right-aligned the caption inside the text region, so
    columns added on the left are padding and the image tokens stay at the tail,
    which is where the loss slices them. A degree of one, and a length that is
    already divisible, are returned unchanged.
    """
    multiple = sequence_multiple()
    if multiple <= 1:
        return inputs
    max_text = int(inputs["_max_text"])
    num_image_tokens = int(inputs["_grid_h"]) * int(inputs["_grid_w"])
    remainder = (max_text + num_image_tokens) % multiple
    if remainder == 0:
        return inputs
    extra = multiple - remainder

    import torch

    from primus.backends.nemo_automodel.models.ideogram4.adapter import (
        _layout_constants,
    )

    _, _, _, padding_id = _layout_constants()

    def _left(name, fill):
        tensor = inputs[name]
        shape = list(tensor.shape)
        shape[1] = extra
        return torch.cat([tensor.new_full(shape, fill), tensor], dim=1)

    padded = dict(inputs)
    padded["hidden_states"] = _left("hidden_states", 0)
    padded["encoder_hidden_states"] = _left("encoder_hidden_states", 0)
    padded["position_ids"] = _left("position_ids", 0)
    padded["indicator"] = _left("indicator", 0)
    padded["segment_ids"] = _left("segment_ids", padding_id)
    padded["_max_text"] = max_text + extra

    global _pad_logged
    if not _pad_logged:
        _pad_logged = True
        logger.info(
            "%s left-padded the packed sequence by %d so its length %d is divisible "
            "by the context-parallel degree (text %d, image %d).",
            _LOG_PREFIX,
            extra,
            max_text + extra + num_image_tokens,
            max_text + extra,
            num_image_tokens,
        )
    return padded


def _install_input_padding() -> None:
    """Wrap the adapter once. The degree is read on each batch, so the wrap is
    inert until enable_parallelism records one above 1."""
    from primus.backends.nemo_automodel.models.ideogram4.adapter import (
        get_ideogram4_adapter_class,
    )

    cls = get_ideogram4_adapter_class()
    if getattr(cls, "_primus_cp_pad_wrapped", False):
        return
    original = cls.prepare_inputs

    def prepare_inputs(self, context):
        return pad_prepared_inputs(original(self, context))

    cls.prepare_inputs = prepare_inputs
    cls._primus_cp_pad_wrapped = True


def _build_cp_plan():
    """Build the diffusers context-parallel plan.

    Imported lazily so this module stays importable without diffusers, which is
    what lets the patch condition be answered during discovery.
    """
    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelOutput,
    )

    return {
        # Root forward inputs. All three are per-token on the same packed axis and
        # are combined with each other before the first block, so they split
        # together. segment_ids and position_ids are absent on purpose; the module
        # docstring says why for each.
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "indicator": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        # The rotary embedding returns a cosine and a sine. Split what it produced
        # rather than what it consumed, so the embedding is applied to the local
        # tokens before the all-to-all.
        "rotary_emb": {
            0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
        },
        # Gather back to full length; nothing follows this but the output wrapper.
        "final_layer": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }


def _varlen_attention_in_use() -> bool:
    """Whether the variable-length attention processor is actually in effect.

    Checks the installed state and not only the request, so that the refusal is
    about what is TRUE rather than what was asked for: the processor patches a
    class default, which outlives the setting that caused it.
    """
    from primus.backends.nemo_automodel.models.ideogram4 import _varlen_common

    if _varlen_common.is_varlen_attn_enabled():
        return True

    try:
        from diffusers.models.transformers.transformer_ideogram4 import (
            Ideogram4Attention,
        )
    except ImportError:
        return False
    return getattr(Ideogram4Attention, "_primus_varlen_installed", False)


def _refuse_varlen_combination() -> None:
    """Refuse CP together with the variable-length processor.

    Raised at the moment CP is switched on rather than at install time, because
    installing the plan is harmless -- it is enabling CP that makes the combination
    wrong.
    """
    if not _varlen_attention_in_use():
        return
    raise ValueError(
        "Context parallelism and the Ideogram-4 variable-length attention processor "
        "cannot be used together. The processor calls the flash-attention kernel "
        "directly and never reaches diffusers' attention dispatch, which is where the "
        "Ulysses all-to-all happens, so each rank would attend only within its own "
        "shard of the sequence and training would produce wrong gradients WITHOUT "
        "raising anywhere. Turn primus_ideogram4.varlen_attention off to run CP on the stock "
        "attention path, or set the context-parallel degree to 1 to keep the "
        "variable-length path."
    )


def _assert_timestep_is_per_sample(timestep) -> None:
    """The plan leaves the timestep unsplit, which is correct only while it is (B,)."""
    if timestep is None or not hasattr(timestep, "dim") or timestep.dim() == 1:
        return
    raise ValueError(
        f"Context parallelism expects a per-sample timestep of shape (B,), got "
        f"{tuple(timestep.shape)}. A per-token timestep is broadcast against the "
        "sequence axis and would have to be split in the context-parallel plan, the "
        "way the plans for per-token-timestep models do it. It is left unsplit here "
        "because the Ideogram-4 adapter passes one value per sample."
    )


def _timestep_pre_hook(module, args, kwargs):
    """Check the plan's unsplit-timestep assumption on every forward.

    Registered as a pre-hook on the instance, deliberately NOT by wrapping the
    forward: diffusers resolves the root plan's named inputs against the forward
    SIGNATURE, so a wrapper taking ``*args, **kwargs`` would hide
    ``encoder_hidden_states`` and ``indicator`` from it and quietly leave them
    unsplit -- turning a guard into the very class of bug it guards against.
    """
    timestep = kwargs.get("timestep")
    if timestep is None and len(args) >= 2:
        timestep = args[1]
    _assert_timestep_is_per_sample(timestep)
    return None


def install() -> bool:
    """Attach a context-parallel plan to Ideogram-4.

    Sets the ``_cp_plan`` class attribute, which is what makes AutoModel's CP path
    accept this model at all, and wraps the enable-parallelism entry point --
    called upstream only for a CP degree above one -- so that the two silently
    wrong configurations are rejected at the moment CP is switched on.

    Returns False when diffusers is absent, so a run without it is unaffected, and
    when diffusers ships its own plan for the model, which then takes precedence.
    Idempotent, and edits no diffusers or AutoModel source.
    """
    try:
        from diffusers.models.transformers.transformer_ideogram4 import (
            Ideogram4Transformer2DModel,
        )
    except ImportError:
        return False

    if getattr(Ideogram4Transformer2DModel, "_primus_cp_installed", False):
        return True

    # Installed with the plan, not only when CP is enabled: the degree is 1 until
    # then, so the wrap adds no padding to a run that never asks for CP.
    _install_input_padding()

    plan = _build_cp_plan()
    shipped = getattr(Ideogram4Transformer2DModel, "_cp_plan", None)
    if shipped is not None and shipped != plan:
        logger.warning(
            "%s diffusers now ships a context-parallel plan for Ideogram-4; keeping "
            "it and installing nothing. Re-check this module against the diffusers pin.",
            _LOG_PREFIX,
        )
        return False

    Ideogram4Transformer2DModel._cp_plan = plan

    original_enable_parallelism = Ideogram4Transformer2DModel.enable_parallelism

    def enable_parallelism(self, *args, **kwargs):
        _refuse_varlen_combination()
        result = original_enable_parallelism(self, *args, **kwargs)
        # Recorded only after the original accepts the config, so a refused
        # enable does not make the adapter pad a run that is not actually split.
        config = kwargs.get("config", args[0] if args else None)
        global _SEQUENCE_MULTIPLE
        _SEQUENCE_MULTIPLE = _degree_of(config)
        self.register_forward_pre_hook(_timestep_pre_hook, with_kwargs=True)
        logger.info(
            "%s context parallelism is on (degree %d). The adapter left-pads the "
            "packed sequence to a multiple of that degree.",
            _LOG_PREFIX,
            _SEQUENCE_MULTIPLE,
        )
        return result

    Ideogram4Transformer2DModel.enable_parallelism = enable_parallelism
    Ideogram4Transformer2DModel._primus_cp_installed = True

    logger.info(
        "%s attached a context-parallel plan to Ideogram-4 (Ulysses). The CP degree "
        "has to divide both the world size and the attention head count; it is inert "
        "until a config asks for a degree above one.",
        _LOG_PREFIX,
    )
    return True
