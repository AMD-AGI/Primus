###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build a TorchTitan model without starting a training loop.

``torchtitan.train.Trainer.__init__`` builds far more than a model: a tokenizer,
a dataloader, a loss function, a metrics processor, a checkpoint manager, and
optionally a validator.  Projection's layer benchmark needs none of that -- it
needs the model and its optimizer, built exactly the way training builds them,
because that is what makes the measurement representative.  Asking for the
tokenizer and dataloader would also make the benchmark depend on dataset assets
being present on the bench node, which they need not be.

So this reproduces just the model-shaped subset of that constructor, in the same
order and with the same calls, and stops:

1. initialize the process group and derive :class:`ParallelDims`
2. build the model on the meta device under the training dtype
3. apply model converters (this is what installs Float8 linears)
4. apply parallelism -- TP, activation checkpointing, compile, FSDP
5. materialize on device and initialize weights
6. build the optimizer over the parallelized model

Keeping the order matters. Converters must see an unparallelized model, and the
optimizer must be built after parallelism so it sees the sharded parameters.
"""

from typing import Any, List, Tuple

_PIPELINE_UNSUPPORTED = (
    "Projection benchmarks build a single pipeline stage, but this config asks "
    "for pipeline_parallel_degree={pp}. The projection driver normally flattens "
    "pipeline parallelism onto the bench node before building the model, so "
    "reaching here means that step was skipped."
)


def build_model_only(job_config: Any) -> Tuple[List[Any], Any, Any]:
    """Build TorchTitan's model and optimizer for *job_config*, no train loop.

    Returns:
        ``(model_parts, optimizers, parallel_dims)``.  ``model_parts`` is a list
        for consistency with TorchTitan's own pipeline-parallel shape, even
        though the benchmark always builds a single stage.
    """
    import torch
    from torchtitan.config import TORCH_DTYPE_MAP
    from torchtitan.distributed import ParallelDims
    from torchtitan.distributed import utils as dist_utils
    from torchtitan.protocols import train_spec as train_spec_module
    from torchtitan.protocols.model_converter import build_model_converters
    from torchtitan.tools import utils

    parallelism = job_config.parallelism
    if getattr(parallelism, "pipeline_parallel_degree", 1) > 1:
        raise NotImplementedError(_PIPELINE_UNSUPPORTED.format(pp=parallelism.pipeline_parallel_degree))

    world_size = dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.job.dump_folder,
    )
    parallel_dims = ParallelDims(
        dp_shard=parallelism.data_parallel_shard_degree,
        dp_replicate=parallelism.data_parallel_replicate_degree,
        cp=parallelism.context_parallel_degree,
        tp=parallelism.tensor_parallel_degree,
        pp=parallelism.pipeline_parallel_degree,
        ep=parallelism.expert_parallel_degree,
        etp=parallelism.expert_tensor_parallel_degree,
        world_size=world_size,
    )

    train_spec = train_spec_module.get_train_spec(job_config.model.name)

    model_args = train_spec.model_args[job_config.model.flavor]
    model_args.update_from_config(job_config)
    _apply_model_args_overrides(model_args, job_config)

    # Meta init: the model is described but not allocated, so parallelism can
    # shard it before any real memory is touched.
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
    ):
        model = train_spec.model_cls(model_args)

    # No-op unless `model.converters` is set; this is where Float8 swaps in.
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    model = train_spec.parallelize_fn(model, parallel_dims, job_config)

    device_type = _device_type()
    init_device = "cpu" if job_config.training.enable_cpu_offload else device_type
    buffer_device = torch.device(device_type) if job_config.training.enable_cpu_offload else None

    model.to_empty(device=init_device)
    with torch.no_grad():
        model.init_weights(buffer_device=buffer_device)
    # Train mode, not eval: dropout and any train-only branches have to be on
    # for the measurement to reflect a training step.
    model.train()

    model_parts = [model]
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config.optimizer, parallel_dims)
    optimizers.register_step_post_hook(
        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
    )

    return model_parts, optimizers, parallel_dims


def _device_type() -> str:
    from torchtitan.tools.utils import device_type

    return device_type


def _apply_model_args_overrides(model_args: Any, job_config: Any) -> List[str]:
    """Apply projection's ``ModelArgs`` overrides, ignoring ones that do not apply.

    A flavor's layer and expert counts come from TorchTitan's own model registry,
    not from the job config, so ``update_from_config`` cannot reach them.  The
    projection needs to: its benchmark profiles one or two representative layers
    rather than the whole stack, and shrinks the expert count to match the
    expert parallelism that fits on the bench node.

    Overrides are staged by
    :func:`primus.core.projection.frameworks.torchtitan.torchtitan_apply_bench_overrides`
    under ``primus_projection.model_args_overrides``, keyed by dotted path.  Each
    model family gates its MoE layers differently -- DeepSeek V3 counts dense
    layers from the front, Llama 4 interleaves on a stride, Qwen 3 flips a flag
    for the whole model -- so the projection stages all the spellings and this
    drops the ones the resolved flavor does not have.

    Returns the paths actually applied, for logging.
    """
    overrides = getattr(getattr(job_config, "primus_projection", None), "model_args_overrides", None)
    if not overrides:
        return []

    applied = []
    for path, value in dict(overrides).items():
        target = model_args
        *parents, leaf = path.split(".")
        for part in parents:
            target = getattr(target, part, None)
            if target is None:
                break
        # Only override attributes this flavor actually declares. Setting an
        # absent one would be accepted silently by the dataclass and then never
        # read, which reads as success while changing nothing.
        if target is not None and hasattr(target, leaf):
            setattr(target, leaf, value)
            applied.append(path)

    if applied:
        from torchtitan.tools.logging import logger

        logger.info(
            "[Primus:Projection] Benchmark ModelArgs overrides applied: "
            + ", ".join(f"{p}={dict(overrides)[p]}" for p in applied)
        )
    return applied
