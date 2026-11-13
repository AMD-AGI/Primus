###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 (``model_type: mistral_large4``) argument patches: the model_type check, the
per-layer lists from the checkpoint or the preset, the post-training checkpoint check, the
``moe_layer_freq`` derivation, the DeepEP / Turbo attention / Turbo grouped-GEMM fallbacks and the
TE GEMM workspace.

The MoE layout is "the first ``first_k_dense_replace`` layers dense, the rest MoE". Megatron
takes it as a per-layer ``moe_layer_freq`` list whose length must equal ``num_layers``, so a
fixed list in the preset breaks every depth-reduced run (``--num_layers 4``). This patch
derives the list from ``first_k_dense_replace`` and the run's ``num_layers``; the model's
layer specs and the converter call the same function (``moe_layer_freq_list``).
"""

import ast
import os
from types import SimpleNamespace

from primus.backends.megatron.checkpoint.native_convert_common import read_hf_config
from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    DEFAULT_FIRST_K_DENSE_REPLACE,
    LAYER_LISTS,
    MEGATRON_TRACKER,
    MODEL_TYPE,
    experts_on_turbo_grouped_gemm,
    hf_layer_lists,
    holds_megatron_checkpoint,
    is_mistral_large4_args,
    moe_layer_freq_list,
    text_config,
)
from primus.core.config.preset_loader import PresetLoader
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_kv_rank_0, log_rank_0


def _param(ctx: PatchContext, name):
    """``name`` from the Megatron args or, at build_args, the module params."""
    args = ctx.extra.get("backend_args", None)
    params = getattr(ctx.extra.get("module_config", None), "params", None)
    v = getattr(args, name, None)
    return v if v not in (None, "") else getattr(params, name, None)


def _is_mistral_large4(ctx: PatchContext) -> bool:
    """``model_type`` is a Primus key: at build_args it is in the module params, not yet in the
    Megatron args (``_param`` reads both); the later phases use ``is_mistral_large4_args`` on
    ``get_args(ctx)``, the same params."""
    return is_mistral_large4_args(SimpleNamespace(model_type=_param(ctx, "model_type")))


# build_args order: the checkpoint's lists and first_k_dense_replace are read first, then
# moe_layer_freq is derived from the resulting first_k_dense_replace
# (test_layer_lists_order_and_hf_defaults)
PRIORITY_MODEL_TYPE_CHECK = 30  # before anything reads this model's settings
PRIORITY_NEEDS_CHECKPOINT = (
    35  # before the layer-list / moe_layer_freq patches: nothing is read or set for a refused run
)
PRIORITY_LAYER_LISTS = 40
PRIORITY_MOE_LAYER_FREQ = 45


def _has_mistral_large4_layout(ctx: PatchContext) -> bool:
    """The config carries this model's per-layer lists (``LAYER_LISTS``: no_rope_layers,
    indexer_types, kv_source_layers), keys only its preset sets."""
    return any(_param(ctx, k) not in (None, "", []) for k in LAYER_LISTS)


@register_patch(
    "megatron.mistral_large4.model_type_check",
    backend="megatron",
    phase="build_args",
    priority=PRIORITY_MODEL_TYPE_CHECK,
    description="Mistral Large 4: refuse its per-layer layout under another model_type (it would build GPT)",
    condition=lambda ctx: not _is_mistral_large4(ctx) and _has_mistral_large4_layout(ctx),
)
def patch_mistral_large4_model_type_check(ctx: PatchContext):
    """A run whose config has this model's per-layer lists (from mistral_Large_4_1T_A52B.yaml) but
    whose ``model_type`` is not ``mistral_large4`` (a typo, the Hugging Face name
    ``mistral_large4_text``, an override) would get the plain GPT provider: no MLA, indexer or
    KV sharing, and a failure only at checkpoint load. Refused here, from what the config holds,
    whatever the wrong value is."""
    given = {k for k in LAYER_LISTS if _param(ctx, k) not in (None, "", [])}
    raise ValueError(
        f"model_type {_param(ctx, 'model_type')!r} with Mistral Large 4's per-layer lists ({', '.join(sorted(given))}): "
        f"this model is built with model_type '{MODEL_TYPE}' (the preset mistral_Large_4_1T_A52B.yaml sets it); "
        "another model_type would build a plain GPT model"
    )


def _has_local_config(path) -> bool:
    """Whether ``path`` is a local checkpoint directory with a ``config.json``."""
    return bool(path) and os.path.isfile(os.path.join(str(path), "config.json"))


def _hf_path(ctx: PatchContext):
    """``hf_path`` from the Megatron args or, at build_args, the module params (a Primus key)."""
    return _param(ctx, "hf_path")


@register_patch(
    "megatron.mistral_large4.layer_lists_from_checkpoint",
    backend="megatron",
    phase="build_args",
    priority=PRIORITY_LAYER_LISTS,  # before moe_layer_freq: first_k_dense_replace may come from the checkpoint
    description="Mistral Large 4: per-layer lists (NoPE / indexer sharing / KV sharing) from the checkpoint's "
    "config.json when post-training from it (stage set, local hf_path)",
    condition=lambda ctx: _is_mistral_large4(ctx) and _is_posttrain(ctx) and _has_local_config(_hf_path(ctx)),
)
def patch_mistral_large4_layer_lists_from_checkpoint(ctx: PatchContext):
    """A checkpoint's layers decide the per-layer layout, not the preset's: a depth-reduced cut of
    the release keeps chosen layers (e.g. 0..15 with layer 15 a KV consumer of 11), so its lists
    are not the release's first entries. In a post-training run (``stage`` set: the run starts
    from the checkpoint the posttrain hook converts from ``hf_path``), the lists and
    ``first_k_dense_replace`` are taken from its ``config.json``; ``num_layers`` may be smaller
    (the checkpoint's first layers), not larger. Pretraining keeps the preset's layout even with a
    local ``hf_path`` (e.g. for the tokenizer): it does not load that checkpoint."""
    args = ctx.extra["backend_args"]
    params = getattr(ctx.extra.get("module_config", None), "params", None)
    path = _hf_path(ctx)
    t = text_config(read_hf_config(str(path)))
    n = int(args.num_layers)
    hf_n = int(t.get("num_hidden_layers", n))
    if n > hf_n:
        raise ValueError(
            f"Mistral Large 4: num_layers {n} exceeds the checkpoint's num_hidden_layers {hf_n} ({path})"
        )
    if n < hf_n:
        # a depth-reduced run of the checkpoint (its first n layers, as the converter maps them);
        # the lists are cut to n by the model config. Said loudly: the head trains on layer-n states
        log_rank_0(
            f"[Patch:megatron.mistral_large4.layer_lists_from_checkpoint] WARNING: depth-reduced fine-tuning: "
            f"num_layers {n} of the checkpoint's {hf_n}; the final norm and head see layer-{n} states, so "
            f"the loss is the cut's, not the full model's (set num_layers {hf_n} for the full model)"
        )
    # omitted lists: the HF default (no NoPE layers, no sharing), not the preset's release lists;
    # the converter uses the same function, so both sides build the same layout
    values, defaulted = hf_layer_lists(dict(t, num_hidden_layers=hf_n))
    for k in defaulted:
        log_rank_0(
            f"[Patch:megatron.mistral_large4.layer_lists_from_checkpoint] WARNING: {path}/config.json has no {k}; "
            f"using the HF default {values[k][:n]}"
        )
    if t.get("first_k_dense_replace") is not None:
        values["first_k_dense_replace"] = int(t["first_k_dense_replace"])
    else:
        # the converter uses the model's default for it (DEFAULT_FIRST_K_DENSE_REPLACE): so does
        # training, rather than the preset's or the user's value
        values["first_k_dense_replace"] = DEFAULT_FIRST_K_DENSE_REPLACE
        log_rank_0(
            f"[Patch:megatron.mistral_large4.layer_lists_from_checkpoint] WARNING: {path}/config.json has no "
            f"first_k_dense_replace; using the default {DEFAULT_FIRST_K_DENSE_REPLACE}, as the converter does"
        )
    preset = _preset_values(params) or {}
    for k, v in values.items():
        given = _param(ctx, k)
        given_v, ckpt_v, preset_v = (_comparable(x, k, n) for x in (given, v, preset.get(k)))
        if given_v is not None and given_v != ckpt_v:
            if preset_v is not None and given_v == preset_v:
                log_rank_0(
                    f"[Patch:megatron.mistral_large4.layer_lists_from_checkpoint] {k}: the preset's {given_v} "
                    f"replaced by the checkpoint's {ckpt_v}"
                )
            else:
                # set by the user (experiment override or CLI), not the preset's list; or the
                # model is not a Primus preset, so a user value cannot be told from a default:
                # refused either way (no silent replacement)
                raise ValueError(
                    f"Mistral Large 4: {k}={given_v} disagrees with the checkpoint's {ckpt_v} ({path}/config.json); "
                    "the layout of a checkpoint is its own: drop the override, or set hf_path to a checkpoint "
                    "with that layout"
                )
        for target in (args, params):
            if target is not None:
                setattr(target, k, v)
        log_kv_rank_0(f"[Patch:megatron.mistral_large4.layer_lists_from_checkpoint] -{k}", f"{v}")


def _comparable(value, name, n):
    """``value`` as a list cut to ``n`` layers (or an int for first_k_dense_replace); None if unset."""
    if value is None:
        return None
    if name == "first_k_dense_replace":
        return int(value)
    try:
        lst = value if isinstance(value, (list, tuple)) else ast.literal_eval(str(value))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as e:
        raise ValueError(f"Mistral Large 4: {name} {value!r} is not a list literal") from e
    if not isinstance(lst, (list, tuple)):
        raise ValueError(f"Mistral Large 4: {name} {value!r} is not a list literal")
    return list(lst)[:n]


def _preset_values(params):
    """The model preset's own values of the per-layer keys (to tell a user override from them);
    None when there is no model preset (no ``model``, or not a Primus preset file). Other load
    errors (bad YAML, bad ``extends``, path traversal) are raised."""
    model = getattr(params, "model", None)
    if not model:
        return None
    try:
        preset = PresetLoader.load(str(model), framework="megatron")
    except FileNotFoundError:
        # a model that is not a Primus preset: no defaults to tell a user value from, so the caller
        # refuses any value that disagrees with the checkpoint; any other preset error is raised
        return None
    return {k: preset.get(k) for k in LAYER_LISTS + ("first_k_dense_replace",)}


@register_patch(
    "megatron.mistral_large4.layer_lists_from_preset",
    backend="megatron",
    phase="build_args",
    priority=PRIORITY_LAYER_LISTS,
    description="Mistral Large 4: no local checkpoint config.json (hf_path unset or not local); say that the "
    "preset's per-layer lists are used",
    condition=lambda ctx: _is_mistral_large4(ctx)
    and _is_posttrain(ctx)
    and not _has_local_config(_hf_path(ctx)),
)
def patch_mistral_large4_layer_lists_from_preset(ctx: PatchContext):
    """A post-training run whose ``hf_path`` is unset (e.g. started from an already converted
    ``pretrained_checkpoint``), a Hub id or a directory without ``config.json``: the per-layer
    lists cannot be read from the checkpoint here, so the preset's are used. Logged as a warning,
    because a depth-reduced cut keeps chosen layers whose lists differ from the release's first
    entries; the posttrain hook passes the local snapshot when it converts."""
    log_rank_0(
        "[Patch:megatron.mistral_large4.layer_lists_from_preset] WARNING: hf_path "
        f"{_hf_path(ctx)!r} is not a local directory with a config.json; using the preset's no_rope_layers / indexer_types / "
        "kv_source_layers / first_k_dense_replace. Set hf_path to the local checkpoint directory to "
        "take them from the checkpoint."
    )


# the stage Primus assumes when none is set (core/runtime/train_runtime.py)
PRETRAIN_STAGE = "pretrain"


def _is_posttrain(ctx: PatchContext) -> bool:
    """A post-training run: ``stage`` set to a post-training stage (``sft_trainer.yaml``'s ``sft``).
    No ``stage`` or ``stage: pretrain`` is pretraining."""
    stage = _param(ctx, "stage")
    return bool(stage) and str(stage).strip().lower() != PRETRAIN_STAGE


def _posttrain_without_weights(ctx: PatchContext) -> bool:
    """A post-training run (``stage`` set: ``sft_trainer.yaml``'s ``sft``) of this model whose
    arguments name no checkpoint to start from (a ``pretrained_checkpoint`` or ``load`` directory
    that holds a Megatron checkpoint: ``holds_megatron_checkpoint``). Pretraining has no ``stage`` or
    ``stage: pretrain`` (``finetune`` is no signal: ``pre_trainer.yaml`` defaults it to true).

    ``hf_path`` is deliberately not part of the condition: build_args runs after the posttrain
    hook (``runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py``), whose
    successful conversion of ``hf_path`` reaches the arguments as ``pretrained_checkpoint``
    (its ``extra.pretrained_checkpoint=...`` line). So with ``stage`` set, no
    ``pretrained_checkpoint`` / ``load`` means no conversion reached this run, whatever
    ``hf_path`` holds; the message says why for each kind of ``hf_path``
    (``_no_checkpoint_reason``)."""
    return _is_posttrain(ctx) and not (
        holds_megatron_checkpoint(_param(ctx, "pretrained_checkpoint"))
        or holds_megatron_checkpoint(_param(ctx, "load"))
    )


def _no_checkpoint_reason(hf_path) -> str:
    """What to do, given the run's ``hf_path``, when no converted checkpoint reached it."""
    if not hf_path:
        return (
            "no hf_path is set. Pass --hf_path <local snapshot of mistralai/Mistral-Large-4-1T-A52B> (a "
            "directory with its config.json, e.g. from `hf download`) and run `primus-cli ... train "
            "posttrain`: its posttrain hook converts it with the native converter"
        )
    if not _has_local_config(hf_path):
        return (
            f"hf_path {str(hf_path)!r} is not a local directory with a config.json (a Hub id, or a "
            "directory without the release's config.json): the posttrain hook cannot detect this model's "
            "native family there and does not convert it. Set hf_path to a local snapshot of "
            "mistralai/Mistral-Large-4-1T-A52B (e.g. from `hf download`)"
        )
    return (
        f"hf_path {str(hf_path)!r} is a local checkpoint, but no conversion of it reached this run: the "
        "posttrain hook converts it only when the run goes through `primus-cli ... train posttrain` (the "
        "hook passes the result as pretrained_checkpoint). Run it that way"
    )


@register_patch(
    "megatron.mistral_large4.posttrain_needs_checkpoint",
    backend="megatron",
    phase="build_args",
    priority=PRIORITY_NEEDS_CHECKPOINT,
    description="Mistral Large 4: refuse fine-tuning with no checkpoint to start from",
    condition=lambda ctx: _is_mistral_large4(ctx) and _posttrain_without_weights(ctx),
)
def patch_mistral_large4_posttrain_needs_checkpoint(ctx: PatchContext):
    """Fine-tuning starts from the converted release: the posttrain hook, which runs before the
    trainer builds its arguments, converts a *local* ``hf_path`` (a directory with the release's
    ``config.json``) and passes the result as ``pretrained_checkpoint``. With no ``hf_path`` the
    hook falls back to ``tokenizer_model`` (a Hub id); with a Hub id or a directory without
    ``config.json`` it cannot detect this model's native family and takes the Megatron-Bridge
    path, which does not support it; and a trainer started without the hook converts nothing. In
    each case a run would start from random weights: refused here, at argument time, with the
    reason for its ``hf_path`` and what to set (or set ``pretrained_checkpoint`` to an already
    converted checkpoint)."""
    raise ValueError(
        "Mistral Large 4 fine-tuning has no checkpoint to start from (no pretrained_checkpoint or load "
        f"directory with {MEGATRON_TRACKER}: point it at the directory that holds that file, not at "
        f"release/ or iter_N/; pretrained_checkpoint={_param(ctx, 'pretrained_checkpoint')!r}, "
        f"load={_param(ctx, 'load')!r}): "
        f"{_no_checkpoint_reason(_hf_path(ctx))}. Or set pretrained_checkpoint to an already converted checkpoint."
    )


@register_patch(
    "megatron.mistral_large4.moe_layer_freq",
    backend="megatron",
    phase="build_args",
    priority=PRIORITY_MOE_LAYER_FREQ,  # after layer_lists_from_checkpoint (first_k_dense_replace)
    description="Mistral Large 4: moe_layer_freq = [0]*first_k_dense_replace + [1]*(num_layers - k)",
    condition=_is_mistral_large4,
)
def patch_mistral_large4_moe_layer_freq(ctx: PatchContext):
    args = ctx.extra["backend_args"]
    k = _param(ctx, "first_k_dense_replace")  # None: the config's default
    experts = getattr(args, "num_experts", None)
    derived = moe_layer_freq_list(k, int(args.num_layers), experts)
    given = _as_freq_list(getattr(args, "moe_layer_freq", None))
    if given is not None and given != derived:
        raise ValueError(
            f"Mistral Large 4: moe_layer_freq {given} disagrees with first_k_dense_replace={k} "
            f"({derived}); the dense / MoE layout is set by first_k_dense_replace only"
        )
    if int(experts or 0) and 1 not in derived:
        # moe_layer_freq_list caps first_k_dense_replace at num_layers: every layer is dense, the
        # MoE settings build nothing (a depth-reduced run shorter than the dense prefix)
        log_rank_0(
            "[Patch:megatron.mistral_large4.moe_layer_freq] WARNING: first_k_dense_replace="
            f"{k if k is not None else 'default'} >= num_layers={int(args.num_layers)}: every layer is dense, "
            f"no MoE layer is built although num_experts={experts}"
        )
    args.moe_layer_freq = derived
    log_kv_rank_0("[Patch:megatron.mistral_large4.moe_layer_freq] -moe_layer_freq", f"{args.moe_layer_freq}")


def _as_freq_list(value):
    """A user's moe_layer_freq as a per-layer list, or None for 1 (the integer or the string).

    1 is Megatron's default and the presets' value ("not set"; the layout comes from
    first_k_dense_replace), so it is not compared. A list of 0 / 1 integers, or a string that
    Megatron's ``moe_freq_type`` reads (a list literal, ``([0]*k+[1]*m)``, ``[1]*4``,
    ``([0]*1+[1]*3)*2``) is returned as a list; anything else raises ValueError."""
    if value is None:
        return None
    if isinstance(value, str):
        # Megatron's own parser of the expression form (as minimax_m3 uses it); imported here, the
        # patch modules load before Megatron is on sys.path
        from megatron.training.arguments import moe_freq_type

        text = value.strip()
        try:
            # Megatron's pattern check allows no whitespace; "[0, 1, 1]" is a common YAML spelling
            value = moe_freq_type("".join(text.split()))
        except (ValueError, SyntaxError, TypeError) as e:
            raise ValueError(
                f"Mistral Large 4: moe_layer_freq {text!r} is not 1, a list of 0 / 1, or a Megatron "
                "list expression such as ([0]*1+[1]*3)"
            ) from e
    if isinstance(value, int) and not isinstance(value, bool):
        if value == 1:
            return None  # Megatron's default (every layer MoE): not set by the user for this model
        raise ValueError(
            f"Mistral Large 4: moe_layer_freq={value} is not supported; the dense / MoE layout is "
            "first_k_dense_replace (leave moe_layer_freq at 1, or give the per-layer list)"
        )
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, int) and not isinstance(v, bool) and v in (0, 1) for v in value
    ):
        return [int(v) for v in value]
    raise ValueError(f"Mistral Large 4: moe_layer_freq {value!r} is not a list of 0 / 1")


def _expert_tp(args) -> int:
    """The expert tensor-parallel size: ``expert_tensor_parallel_size``, which defaults to TP."""
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    return int(getattr(args, "expert_tensor_parallel_size", None) or tp)


def _deepep_unusable(ctx: PatchContext) -> bool:
    """Primus-Turbo DeepEP checks ``expert-TP x EP > 1`` (its dispatcher's TP group is the
    expert tensor-parallel group, ``expert_tensor_parallel_size``, which defaults to TP)."""
    args = get_args(ctx)
    # this model only: args.model_type (is_mistral_large4_args)
    if not is_mistral_large4_args(args) or not getattr(args, "use_turbo_deepep", False):
        return False
    ep = int(getattr(args, "expert_model_parallel_size", 1) or 1)
    return _expert_tp(args) * ep == 1


@register_patch(
    "megatron.mistral_large4.deepep_single_rank",
    backend="megatron",
    phase="setup",  # before the before_train phase selects megatron.turbo.moe_dispatcher
    description="Mistral Large 4: Primus-Turbo DeepEP needs expert-TP x EP > 1; use Megatron's alltoall otherwise",
    condition=_deepep_unusable,
)
def patch_mistral_large4_deepep_single_rank(ctx: PatchContext):
    """The experiment enables Primus-Turbo DeepEP for its expert-parallel layouts; a one-rank
    expert layout (expert-TP x EP == 1, e.g. TP2 with ETP1 and EP1) cannot use it, so it gets
    Megatron's alltoall dispatcher, which places tokens the same way."""
    args = get_args(ctx)
    args.use_turbo_deepep = False
    args.moe_token_dispatcher_type = "alltoall"
    log_kv_rank_0(
        "[Patch:megatron.mistral_large4.deepep_single_rank] -use_turbo_deepep", "False (expert-TP x EP == 1)"
    )


def _turbo_attention_with_cp(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    # this model only: args.model_type (is_mistral_large4_args)
    return (
        is_mistral_large4_args(args)
        and bool(getattr(args, "use_turbo_attention", False))
        and int(getattr(args, "context_parallel_size", 1) or 1) > 1
    )


@register_patch(
    "megatron.mistral_large4.turbo_attention_cp",
    backend="megatron",
    phase="setup",  # before the spec provider reads use_turbo_attention
    description="Mistral Large 4: TE core attention instead of Primus-Turbo attention with context parallelism",
    condition=_turbo_attention_with_cp,
)
def patch_mistral_large4_turbo_attention_cp(ctx: PatchContext):
    """``PrimusTurboAttention`` runs context parallelism as Ulysses (all-to-all) with a one-rank
    ring group; Megatron's MLA splits the sequence for a p2p ring, so the two disagree and the
    CP > 1 losses diverge from the CP1 run. TE's core attention runs the p2p ring. CP1 keeps the
    Turbo kernel."""
    args = get_args(ctx)
    args.use_turbo_attention = False
    log_kv_rank_0(
        "[Patch:megatron.mistral_large4.turbo_attention_cp] -use_turbo_attention",
        f"False (context_parallel_size {args.context_parallel_size} > 1: TE core attention)",
    )


def _turbo_grouped_gemm_unusable(ctx: PatchContext) -> bool:
    """Layouts Primus-Turbo's grouped linear does not support yet (three cases).

    ``PrimusTurboGroupedLinear`` keeps all local experts in one ``weights`` parameter and
    registers per-expert ``weight{i}`` views lazily (at the first forward), and asserts
    expert-TP 1:

    * Megatron-FSDP (``use_megatron_fsdp``): its post-forward release walks
      ``module.parameters()`` and finds the views in no FSDP bucket (KeyError);
    * expert tensor parallelism > 1: the assertion;
    * the non-distributed optimizer's sharded checkpoint (``torch_dist`` / ``fsdp_dtensor``
      save or load without ``use_distributed_optimizer``): the optimizer holds ``weights``, the
      model's sharded state dict names the views (KeyError in ``sharded_state_dict``).

    In those layouts the experts run through Megatron's ``SequentialMLP`` (one TE linear per
    expert): the same math and the same checkpoint keys (``experts.linear_fc*.weight`` with the
    expert axis), not TE's own grouped GEMM (see te_gemm_workspace below, which enlarges the TE
    GEMM workspace for both and warns for TE's grouped GEMM, untested on ROCm for this model).
    """
    args = get_args(ctx)
    # this model only: args.model_type (is_mistral_large4_args)
    if not is_mistral_large4_args(args) or not getattr(args, "use_turbo_grouped_gemm", False):
        return False
    if getattr(args, "use_megatron_fsdp", False):
        return True
    if _expert_tp(args) > 1:
        return True  # PrimusTurboGroupedLinear asserts expert-TP 1
    # only when a sharded checkpoint is saved or loaded (a run that does neither keeps Turbo);
    # an unset ckpt_format is Megatron's default, torch_dist
    sharded_ckpt = str(getattr(args, "ckpt_format", None) or "torch_dist") != "torch"
    uses_ckpt = bool(
        getattr(args, "save", None)
        or getattr(args, "load", None)
        or getattr(args, "pretrained_checkpoint", None)
    )
    return not getattr(args, "use_distributed_optimizer", False) and sharded_ckpt and uses_ckpt


@register_patch(
    "megatron.mistral_large4.turbo_grouped_gemm_fallback",
    backend="megatron",
    phase="setup",  # before the model specs read use_turbo_grouped_gemm
    description="Mistral Large 4: SequentialMLP experts where Primus-Turbo's grouped linear is unsupported "
    "(Megatron-FSDP; expert-TP > 1; sharded checkpoint without the distributed optimizer)",
    condition=_turbo_grouped_gemm_unusable,
)
def patch_mistral_large4_turbo_grouped_gemm_fallback(ctx: PatchContext):
    args = get_args(ctx)
    args.use_turbo_grouped_gemm = False
    args.moe_grouped_gemm = False
    log_kv_rank_0(
        "[Patch:megatron.mistral_large4.turbo_grouped_gemm_fallback] -moe_grouped_gemm",
        "False (Megatron-FSDP, expert-TP > 1, or sharded checkpoint without the distributed optimizer: "
        "SequentialMLP experts)",
    )


def _experts_on_te_gemms(args) -> bool:
    """A Mistral Large 4 run (args.model_type) whose routed experts are not on Primus-Turbo's
    grouped GEMM (``experts_on_turbo_grouped_gemm``, the layer specs' rule)."""
    return is_mistral_large4_args(args) and not experts_on_turbo_grouped_gemm(args)


@register_patch(
    "megatron.mistral_large4.te_gemm_workspace",
    backend="megatron",
    phase="before_train",
    description="Mistral Large 4: enlarge the TE hipBLASLt GEMM workspace when the experts run on TE GEMMs "
    "(not on Primus-Turbo's grouped GEMM)",
    condition=lambda ctx: _experts_on_te_gemms(get_args(ctx)),
)
def patch_mistral_large4_te_gemm_workspace(ctx: PatchContext):
    """Runs for every Mistral Large 4 run whose experts are not on Primus-Turbo's grouped GEMM
    (``PrimusGroupedMLP`` is built only with Primus-Turbo enabled and installed and both
    ``use_turbo_grouped_gemm`` and ``moe_grouped_gemm``; if any of them is off, including after
    the setup-phase turbo_grouped_gemm_fallback, the experts run on TE GEMMs).

    * ``moe_grouped_gemm`` off (``SequentialMLP``, the fallback's path): the per-expert TE GEMMs
      (6144 -> 12288 / 6144, a few hundred tokens) make hipBLASLt pick split-K algorithms whose
      workspace exceeds TE's 64 MiB default ("HIPBLASLT Error: 6" on gfx950); the larger
      workspace avoids that failure.
    * ``moe_grouped_gemm`` on without Primus-Turbo (TE's own grouped GEMM): failed on ROCm with
      TE's default workspace and is not tested with the larger one; a warning says so.

    Uses ``te_patches/gemm_workspace_size_patches.py`` (``PRIMUS_TE_GEMM_WORKSPACE_MB``, default 512; nothing on
    CUDA or when the workspace is already large enough; handles TE 2.17's ``cpp_extensions.gemm``
    and TE 2.8's ``module.base`` layouts, an error on a TE with neither)."""
    if getattr(get_args(ctx), "moe_grouped_gemm", False):  # here: without Primus-Turbo
        log_rank_0(
            "[Patch:megatron.mistral_large4.te_gemm_workspace] WARNING: TE grouped GEMM experts "
            "(moe_grouped_gemm without Primus-Turbo's grouped GEMM) are not tested for Mistral Large 4 on ROCm "
            "(hipBLASLt error 6 with TE's default workspace); use_turbo_grouped_gemm, or moe_grouped_gemm "
            "false (SequentialMLP), are the tested paths"
        )
    # deferred import, as in the GLM-5.3 patch: the helper imports torch at module level
    from primus.backends.megatron.patches.te_patches.gemm_workspace_size_patches import (
        enlarge_te_gemm_workspace,
    )

    enlarge_te_gemm_workspace("megatron.mistral_large4.te_gemm_workspace")
