###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-layer q/k/v split shapes for Megatron's Muon on Inkling.

Muon orthogonalises q, k and v separately: ``TensorParallelMuon.orthogonalize``
views each ``linear_qkv`` gradient as ``[groups, q + k + v, hidden]`` using one
``qkv_split_shapes`` that ``get_megatron_muon_optimizer`` computes from the
model config, i.e. from the global attention layers. Inkling's sliding and
global layers can have different head geometry (Inkling-Large: 16 KV groups on
sliding layers, 8 on global ones), so the shared shapes fail on the sliding
layers' ``linear_qkv``.

This patch stores each layer's own split shapes on its ``linear_qkv`` weight,
copies them to the fp32 master weight that the bf16 optimizer actually steps,
and has ``orthogonalize`` use them for that parameter.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_LOG_PREFIX = "[Patch:megatron.optimizer.inkling_muon_qkv]"

QKV_SPLIT_SHAPES_ATTR = "_primus_qkv_split_shapes"


def layer_qkv_split_shapes(config) -> tuple:
    """``(q, k, v)`` rows per KV group for an attention layer built from ``config``."""
    heads_per_group = config.num_attention_heads // config.num_query_groups
    return (heads_per_group * config.kv_channels, config.kv_channels, config.kv_channels)


def tag_qkv_split_shapes(model_chunk) -> dict:
    """Store each attention layer's split shapes on its ``linear_qkv.weight``.

    Returns ``{shapes: number of weights tagged}``.
    """
    tagged: dict = {}
    for name, module in model_chunk.named_modules():
        weight = getattr(getattr(module, "linear_qkv", None), "weight", None)
        config = getattr(module, "config", None)
        if weight is None or weight.dim() != 2 or getattr(config, "num_query_groups", None) is None:
            continue
        shapes = layer_qkv_split_shapes(config)
        if weight.shape[0] % sum(shapes):
            raise ValueError(
                f"{_LOG_PREFIX} {name}.linear_qkv.weight has {weight.shape[0]} rows, "
                f"not a multiple of the per-group size {sum(shapes)} from {shapes}"
            )
        setattr(weight, QKV_SPLIT_SHAPES_ATTR, shapes)
        tagged[shapes] = tagged.get(shapes, 0) + 1
    return tagged


def propagate_to_master_weights(model_chunk) -> int:
    """Copy the tag onto each fp32 master weight.

    ``Float16OptimizerWithFloat16Params`` copies only the tensor-parallel
    attributes (``is_qkv`` among them) onto its fp32 clone, and that clone is
    what ``orthogonalize`` receives.
    """
    count = 0
    for param in model_chunk.parameters():
        shapes = getattr(param, QKV_SPLIT_SHAPES_ATTR, None)
        main_param = getattr(param, "main_param", None)
        if shapes is not None and main_param is not None:
            setattr(main_param, QKV_SPLIT_SHAPES_ATTR, shapes)
            count += 1
    return count


def make_orthogonalize(original_orthogonalize):
    """Run ``original_orthogonalize`` with the parameter's own split shapes, if it has them."""

    def orthogonalize(self, p, grad, **kwargs):
        shapes = getattr(p, QKV_SPLIT_SHAPES_ATTR, None)
        if shapes is None or not (self.split_qkv and self.is_qkv_fn(p)):
            return original_orthogonalize(self, p, grad, **kwargs)
        model_shapes = self.qkv_split_shapes
        self.qkv_split_shapes = list(shapes)
        try:
            return original_orthogonalize(self, p, grad, **kwargs)
        finally:
            self.qkv_split_shapes = model_shapes

    orthogonalize._primus_inkling_qkv = True
    return orthogonalize


def make_muon_factory(original_factory):
    """Wrap ``get_megatron_muon_optimizer`` to tag parameters before and master weights after."""

    def get_megatron_muon_optimizer(config, model_chunks, *args, **kwargs):
        tagged: dict = {}
        for model_chunk in model_chunks:
            for shapes, count in tag_qkv_split_shapes(model_chunk).items():
                tagged[shapes] = tagged.get(shapes, 0) + count
        optimizer = original_factory(config, model_chunks, *args, **kwargs)
        masters = sum(propagate_to_master_weights(model_chunk) for model_chunk in model_chunks)
        # dist_muon shards the dense weights over DP ranks, so each rank tags only its own.
        import torch

        if torch.distributed.is_initialized():
            count = torch.tensor([masters], dtype=torch.int64, device=torch.cuda.current_device())
            torch.distributed.all_reduce(count)
            masters = int(count.item())
        log_rank_0(
            f"{_LOG_PREFIX} per-layer q/k/v split shapes on {sum(tagged.values())} linear_qkv "
            f"weight(s): {dict(sorted(tagged.items()))}; {masters} fp32 master weight(s) tagged "
            "across all ranks"
        )
        return optimizer

    get_megatron_muon_optimizer._primus_original = original_factory
    return get_megatron_muon_optimizer


@register_patch(
    "megatron.optimizer.inkling_muon_qkv",
    backend="megatron",
    phase="before_train",
    description="Muon: split each Inkling linear_qkv by its own layer's head geometry.",
    priority=46,
    condition=lambda ctx: (
        getattr(get_args(ctx), "model_type", None) == "inkling"
        and "muon" in str(getattr(get_args(ctx), "optimizer", "") or "")
    ),
)
def patch_inkling_muon_qkv(ctx: PatchContext):
    import megatron.core.optimizer.muon as muon_module
    from megatron.core.optimizer.muon import TensorParallelMuon

    if getattr(TensorParallelMuon.orthogonalize, "_primus_inkling_qkv", False):
        return
    TensorParallelMuon.orthogonalize = make_orthogonalize(TensorParallelMuon.orthogonalize)

    factory = make_muon_factory(muon_module.get_megatron_muon_optimizer)
    muon_module.get_megatron_muon_optimizer = factory
    # training.py binds the factory at import time, so rebind it there as well.
    from megatron.training import training as megatron_training

    if hasattr(megatron_training, "get_megatron_muon_optimizer"):
        megatron_training.get_megatron_muon_optimizer = factory
    log_rank_0(f"{_LOG_PREFIX} installed")
