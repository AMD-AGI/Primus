###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Skip weight decay on the token-embedding weight (Megatron optimizer path).

Megatron's standard optimizer overrides (``get_standard_config_overrides``) set
``wd_mult = 0`` only for 1-D parameters and biases, so the 2-D word-embedding
matrix gets weight decay. Some recipes (OLMo 2, SmolLM3) remove weight decay
from the embedding. ``embedding_no_wd_overrides`` returns the extra
``ParamKey -> ParamGroupOverride`` entry that does this. Megatron's own
``_get_param_groups`` resolves it together with the standard overrides, so it
applies wherever Megatron's ``get_megatron_optimizer`` builds the optimizer
(DDP or the distributed optimizer, any TP/PP/CP layout). The Primus FSDP2
custom optimizers do not take ``config_overrides``; the patch rejects them.

Matched parameters:
  * ``*embedding.word_embeddings.weight``: the input embedding (first stage).
  * any parameter with ``shared_embedding`` set: with tied embeddings and PP>1,
    the last stage holds a copy of the embedding as ``output_layer.weight``;
    Megatron keeps both copies equal by all-reducing their grads, so both copies
    must also get the same (zero) weight decay.
An untied ``output_layer.weight`` (the LM head) keeps weight decay.
"""

import fnmatch
from typing import Dict, Iterable, List, Optional, Tuple

EMBEDDING_PARAM_NAME = "*embedding.word_embeddings.weight"
EMBEDDING_PARAM_ATTR = "shared_embedding"


def embedding_no_wd_overrides() -> Dict:
    """Return ``{ParamKey: ParamGroupOverride(wd_mult=0.0)}`` for the embedding weight."""
    from megatron.core.optimizer.optimizer_config import ParamKey
    from megatron.core.optimizer_param_scheduler import ParamGroupOverride

    key = ParamKey(name=EMBEDDING_PARAM_NAME, attr=EMBEDDING_PARAM_ATTR)
    return {key: ParamGroupOverride(wd_mult=0.0)}


def add_embedding_no_wd(config_overrides: Optional[Dict], config=None) -> Dict:
    """Return ``config_overrides`` plus the embedding no-weight-decay entry.

    Follows ``_get_param_groups``'s contract: ``None`` means "Megatron's standard
    overrides" (no weight decay on biases and 1-D params), so they are added
    explicitly here (this needs ``config``); a dict, including ``{}``, is the
    caller's complete choice and is kept as is (``{}`` is Megatron's documented way
    to turn the standard overrides off). Megatron's ``get_megatron_optimizer_config``
    always returns the standard overrides. If the caller already has an entry with
    the same ``ParamKey``, the two are merged (``wd_mult`` set to 0, other fields kept).
    """
    if config_overrides is None:
        if config is None:
            raise ValueError(
                "add_embedding_no_wd(None) needs the OptimizerConfig to add the standard overrides"
            )
        from megatron.core.optimizer import get_standard_config_overrides

        config_overrides = get_standard_config_overrides(config=config)
    merged = dict(config_overrides)
    for key, override in embedding_no_wd_overrides().items():
        merged[key] = {**dict(merged.get(key) or {}), **override}
    return merged


def is_embedding_param(name: str, param) -> bool:
    """True for the parameters the embedding no-wd override is meant to match."""
    return fnmatch.fnmatch(name, EMBEDDING_PARAM_NAME) or bool(getattr(param, EMBEDDING_PARAM_ATTR, False))


def wd_mult_by_param_id(param_groups: List[Dict]) -> Dict[int, float]:
    """``id(param) -> wd_mult`` for every parameter in ``param_groups`` (1.0 when a group
    has no ``wd_mult``, Megatron's default)."""
    by_id = {}
    for group in param_groups:
        for p in group["params"]:
            by_id[id(p)] = group.get("wd_mult", 1.0)
    return by_id


def embedding_wd_mults(
    model_chunks: Iterable, param_groups: List[Dict], by_id=None
) -> List[Tuple[str, object]]:
    """For every trainable embedding parameter in ``model_chunks``: the ``wd_mult`` of
    the param group that holds it (1.0 when the group has no ``wd_mult``, as in
    Megatron's default), or None when no param group holds it. Frozen parameters
    (``requires_grad=False``) are skipped: no weight decay applies to them.
    ``by_id`` is ``wd_mult_by_param_id(param_groups)`` when the caller already has it."""
    if by_id is None:
        by_id = wd_mult_by_param_id(param_groups)
    found = []
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if param.requires_grad and is_embedding_param(name, param):
                found.append((name, by_id.get(id(param))))
    return found


def _is_first_stage(chunk) -> bool:
    """``pre_process`` of a (possibly DDP/Float16Module-wrapped) model chunk."""
    from megatron.core.utils import get_attr_wrapped_model

    try:
        return bool(get_attr_wrapped_model(chunk, "pre_process"))
    except RuntimeError:  # no pre_process anywhere: not a Megatron model chunk
        return False


def check_embedding_no_wd(model_chunks: Iterable, param_groups: List[Dict], by_id=None) -> List[str]:
    """Raise unless every trainable embedding parameter is in a param group with
    ``wd_mult == 0``, and unless every chunk that holds the input embedding
    (``pre_process``) has a parameter matching ``EMBEDDING_PARAM_NAME`` (trainable or
    frozen; a frozen one needs no override). Returns the names of the trainable
    embedding parameters checked (empty on a middle pipeline stage, which holds none,
    or when the embedding is frozen)."""
    model_chunks = list(model_chunks)
    found = embedding_wd_mults(model_chunks, param_groups, by_id)
    ungrouped = [n for n, wd in found if wd is None]
    if ungrouped:
        raise RuntimeError(f"embedding parameter(s) not in any param group: {ungrouped}")
    bad = [(n, wd) for n, wd in found if wd != 0.0]
    if bad:
        raise RuntimeError(f"embedding parameter(s) kept weight decay: {bad}")
    for i, chunk in enumerate(model_chunks):
        if not _is_first_stage(chunk):
            continue
        if not any(fnmatch.fnmatch(n, EMBEDDING_PARAM_NAME) for n, _ in chunk.named_parameters()):
            raise RuntimeError(
                f"first-stage model chunk {i} has no parameter matching {EMBEDDING_PARAM_NAME!r}; "
                "the embedding no-weight-decay override would not apply to it"
            )
    return [n for n, _ in found]


def weight_decay_summary(model_chunks: Iterable, param_groups: List[Dict], by_id=None) -> Dict[str, object]:
    """Run-time facts for the log: how many trainable 2-D non-embedding weights keep
    weight decay (``wd_mult > 0``) and how many do not, and whether the chunks hold a
    separate (untied) ``output_layer.weight``."""
    if by_id is None:
        by_id = wd_mult_by_param_id(param_groups)
    decayed = not_decayed = 0
    untied_head = False
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if name.endswith("output_layer.weight") and not getattr(param, EMBEDDING_PARAM_ATTR, False):
                untied_head = True
            if not param.requires_grad or param.dim() < 2 or is_embedding_param(name, param):
                continue
            wd = by_id.get(id(param))
            if wd is None:
                continue
            if wd > 0:
                decayed += 1
            else:
                not_decayed += 1
    return {"decayed_2d": decayed, "not_decayed_2d": not_decayed, "untied_output_layer": untied_head}
