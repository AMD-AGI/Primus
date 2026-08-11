###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Install Per-Head Muon (Kimi K3 report §2.5) onto Megatron's Muon path.

See :mod:`primus.backends.megatron.core.optimizer.per_head_muon` for the design, the
parameter-selection rule and the resolved ambiguities. This module only wires it in.

Three things are patched, all inside one registered patch:

1. ``TensorParallelMuon.orthogonalize`` — the per-head branch itself
   (``muon.py:110-162`` is upstream's documented override point).
2. ``TensorParallelMuon.__init__`` — records the Newton-Schulz kwargs on the instance.
   They are otherwise captured only in the ``scaled_orthogonalize_fn`` closure
   (``muon.py:67-90``) and never stored, and ``coefficient_type`` in particular is not
   exposed on ``OptimizerConfig`` at all, so the batched implementation has no other
   honest source for them.
3. ``get_megatron_muon_optimizer`` — wrapped to tag the selected parameters *before*
   the optimizer is built and to copy the tags onto the fp32 master weights *after*.
   It is patched in **both** binding locations, because
   ``megatron/training/training.py:130`` does
   ``from megatron.core.optimizer.muon import get_megatron_muon_optimizer`` at module
   scope, so rebinding only the defining module would have no effect on the call at
   ``training.py:1538``. Same two-location treatment as
   ``optimizer_patches.py:72-98``.

Why patch rather than replace ``get_megatron_muon_optimizer`` wholesale: everything
else in that function — master weights, the chained AdamW for non-2-D parameters, the
expert-parallel split, ``LayerWiseDistributedOptimizer`` — is orthogonal to per-head
blocking and must not fork.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_LOG_PREFIX = "[Patch:megatron.optimizer.per_head_muon]"


def _muon_selected(args) -> bool:
    """Whether this job actually runs Muon.

    ``args.optimizer`` and not ``config.optimizer``: ``get_megatron_muon_optimizer``
    overwrites the latter with ``'adam'`` on entry (``muon.py:186``).
    """
    return "muon" in str(getattr(args, "optimizer", "") or "")


def _per_head_requested(args) -> bool:
    """``muon_per_head``, resolved the same way the config resolves it.

    Going through ``PerHeadMuonConfig`` rather than a bare ``bool(...)`` matters: a raw
    string value would make ``bool("false")`` true and fire the patch for a job that
    explicitly turned it off.
    """
    from primus.backends.megatron.core.optimizer.per_head_muon import PerHeadMuonConfig

    try:
        return PerHeadMuonConfig.from_args(args).enabled
    except ValueError:
        # An uninterpretable flag must surface when the patch runs, with a real
        # traceback, rather than silently disabling the feature from a predicate.
        return True


@register_patch(
    "megatron.optimizer.per_head_muon",
    backend="megatron",
    phase="before_train",
    description=(
        "Per-Head Muon (Kimi K3 report §2.5): partition Q/K/V momentum along the head "
        "dimension and run Newton-Schulz per head block."
    ),
    priority=45,
    # Deliberately gated on the OPT-IN ARGS, not on model_type. Per-Head Muon is
    # a Muon-optimizer feature (any head-structured model can request it); it only
    # installs when a run explicitly selects optimizer=muon AND muon_per_head=true.
    # No non-K3 recipe sets those, so shared Muon code stays untouched for other
    # models, while a future non-K3 model that wants per-head blocking can still
    # opt in. (Contrast the model-type gate on the K3 flops/pp/probe/QB patches,
    # which wrap functions shared by ALL models and key off generic args.)
    condition=lambda ctx: (_muon_selected(get_args(ctx)) and _per_head_requested(get_args(ctx))),
)
def patch_per_head_muon(ctx: PatchContext):
    """Install the per-head branch. Idempotent."""
    from primus.backends.megatron.core.optimizer.per_head_muon import (
        PerHeadMuonConfig,
        make_per_head_orthogonalize,
        propagate_specs_to_master_weights,
        tag_per_head_params,
    )

    args = get_args(ctx)
    per_head_config = PerHeadMuonConfig.from_args(args)
    if not per_head_config.enabled:
        # Belt and braces: the condition already gates on this, but `enabled` is meant
        # to be the single source of truth and `tag_per_head_params` enforces it too.
        log_rank_0(f"{_LOG_PREFIX} muon_per_head is off; not installing")
        return

    import megatron.core.optimizer.muon as muon_module
    from megatron.core.optimizer.muon import TensorParallelMuon

    if per_head_config.impl == "batched":
        # Fail here rather than on the first optimizer step: the batched path needs a
        # private symbol from emerging_optimizers.
        from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: F401
            _COEFFICIENT_SETS,
        )

    # ---- 1. the orthogonalize override --------------------------------------
    if not getattr(TensorParallelMuon.orthogonalize, "_primus_per_head_wrapped", False):
        TensorParallelMuon.orthogonalize = make_per_head_orthogonalize(
            TensorParallelMuon.orthogonalize, per_head_config
        )
        log_rank_0(f"{_LOG_PREFIX} wrapped TensorParallelMuon.orthogonalize")

    # ---- 2. record the Newton-Schulz kwargs on each instance ----------------
    if not getattr(TensorParallelMuon.__init__, "_primus_per_head_wrapped", False):
        import inspect

        original_init = TensorParallelMuon.__init__
        init_signature = inspect.signature(original_init)

        def patched_init(self, *init_args, **init_kwargs):
            original_init(self, *init_args, **init_kwargs)
            # bind_partial + apply_defaults so a positional call or an omitted kwarg
            # still yields the same value the closure in muon.py:67-90 captured.
            bound = init_signature.bind_partial(self, *init_args, **init_kwargs)
            bound.apply_defaults()
            self._primus_muon_ns_kwargs = {
                "steps": bound.arguments.get("num_ns_steps", 5),
                "coefficient_type": bound.arguments.get("coefficient_type", "quintic"),
                "scale_mode": bound.arguments.get("scale_mode", "spectral"),
                "extra_scale_factor": bound.arguments.get("extra_scale_factor", 1.0),
            }

        patched_init._primus_per_head_wrapped = True
        patched_init._primus_original = original_init
        TensorParallelMuon.__init__ = patched_init

    # ---- 3. tag the parameters around the factory ---------------------------
    original_factory = muon_module.get_megatron_muon_optimizer
    if getattr(original_factory, "_primus_per_head_wrapped", False):
        log_rank_0(f"{_LOG_PREFIX} already installed; nothing to do")
        return

    def patched_get_megatron_muon_optimizer(config, model_chunks, *f_args, **f_kwargs):
        total_selected = 0
        rules: dict = {}
        skipped: list = []
        for model_chunk in model_chunks:
            summary = tag_per_head_params(model_chunk.named_parameters(), model_chunk.config, per_head_config)
            total_selected += summary.num_selected
            for rule, count in summary.by_rule().items():
                rules[rule] = rules.get(rule, 0) + count
            skipped.extend(summary.skipped_head_structured)

        log_rank_0(
            f"{_LOG_PREFIX} per-head blocking selected {total_selected} parameter(s); "
            f"by rule: {dict(sorted(rules.items()))}; impl={per_head_config.impl}, "
            f"split_kv={per_head_config.split_kv}, "
            f"include_output_proj={per_head_config.include_output_proj}, "
            f"include_gates={per_head_config.include_gates}"
        )
        if skipped:
            log_rank_0(
                f"{_LOG_PREFIX} {len(skipped)} head-structured parameter(s) left on the "
                f"whole-matrix path by config (e.g. {sorted(set(skipped))[:3]})"
            )
        if total_selected == 0:
            message = (
                f"{_LOG_PREFIX} muon_per_head is enabled but no parameter matched the "
                "per-head Q/K/V rule. Either the model is not Kimi K3 or its head dims "
                "are not set on the transformer config."
            )
            if per_head_config.strict:
                raise RuntimeError(message + " Set muon_per_head_strict=false to downgrade.")
            log_rank_0("WARNING " + message)

        optimizer = original_factory(config, model_chunks, *f_args, **f_kwargs)

        tagged_masters = 0
        for model_chunk in model_chunks:
            tagged_masters += propagate_specs_to_master_weights(model_chunk.named_parameters())
        log_rank_0(f"{_LOG_PREFIX} propagated the per-head spec to {tagged_masters} fp32 master " "weight(s)")
        if total_selected > 0 and tagged_masters == 0 and getattr(config, "bf16", False):
            # bf16 means Float16OptimizerWithFloat16Params, whose fp32 clone is what
            # orthogonalize() actually sees (optimizer.py:675-684).
            log_rank_0(
                f"WARNING {_LOG_PREFIX} bf16 is on but no master weight was tagged; "
                "per-head blocking would silently not apply."
            )
        return optimizer

    patched_get_megatron_muon_optimizer._primus_per_head_wrapped = True
    patched_get_megatron_muon_optimizer._primus_original = original_factory

    patched = 0
    muon_module.get_megatron_muon_optimizer = patched_get_megatron_muon_optimizer
    patched += 1
    try:
        from megatron.training import training as megatron_training
    except ImportError as exc:
        # megatron.training is expected to import at before_train; a failure means a
        # broken install, not an optional dependency. Fail loudly rather than leave the
        # module-scope binding at training.py:1538 unpatched (per-head Muon would then
        # silently not apply there).
        raise RuntimeError(
            f"{_LOG_PREFIX} could not import megatron.training.training to patch the "
            "module-scope get_megatron_muon_optimizer binding. This indicates a broken "
            "megatron installation."
        ) from exc

    if hasattr(megatron_training, "get_megatron_muon_optimizer"):
        megatron_training.get_megatron_muon_optimizer = patched_get_megatron_muon_optimizer
        patched += 1

    log_rank_0(
        f"{_LOG_PREFIX} wrapped get_megatron_muon_optimizer in {patched} location(s); "
        f"config={per_head_config}"
    )
