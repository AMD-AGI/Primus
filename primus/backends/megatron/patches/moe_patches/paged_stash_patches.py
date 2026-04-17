###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE Paged Stashing Patches

Patches for enabling paged-stashing of MoE expert activations. Ports the feature
implemented in https://github.com/NVIDIA/Megatron-LM/pull/2690 onto a stock
Megatron-LM checkout (tested against commit d3528a2 and the ``core_r0.16.0``
branch).

Design:
    - Installs ``megatron.core.transformer.moe.paged_stash`` into ``sys.modules``
      by loading the primus-owned copy at
      ``primus.backends.megatron.core.transformer.moe.paged_stash``. That module
      is a verbatim copy of the upstream ``paged_stash.py`` and exposes the core
      runtime (``PagedStashManager``, ``PagedStashRunner``,
      ``paged_stash_reset``, ``paged_stash_group_start/commit``,
      ``paged_stash_init_chunk_handler``, ``get_paged_stash_context``).
    - Adds the new ``TransformerConfig`` fields required by the feature.
    - Wires the MoE activation stashing context into ``TEGroupedMLP.forward``.
    - Adds over-budget tracking on the flex/hybrid EP token dispatcher.
    - Calls ``paged_stash_reset`` from the pipeline schedules.
    - Calls ``preprocess_for_paged_stash`` from ``GPTModel.forward``.
    - Extends ``FullCudaGraphWrapper`` with ``reset_cuda_graph`` and
      ``speculative_cuda_graph_check`` and accepts the new
      ``moe_paged_stash`` / ``moe_expert_rank_capacity_factor`` constructor
      arguments.
    - Wraps ``forward_backward_func`` with ``PagedStashRunner`` inside training
      / evaluation to handle over-budget detection.

The condition gates this patch on ``--moe_paged_stash`` (Megatron's canonical
flag). When the flag is not set, no mutation is performed.
"""

import dataclasses
import importlib
import sys
from contextlib import nullcontext
from typing import Optional

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _install_paged_stash_module():
    """Install primus-owned paged_stash.py into megatron's namespace.

    Any future ``from megatron.core.transformer.moe.paged_stash import ...``
    will resolve to the primus copy.
    """
    module_path = "megatron.core.transformer.moe.paged_stash"
    if module_path in sys.modules:
        return sys.modules[module_path]

    primus_mod = importlib.import_module(
        "primus.backends.megatron.core.transformer.moe.paged_stash"
    )
    sys.modules[module_path] = primus_mod

    try:
        moe_pkg = importlib.import_module("megatron.core.transformer.moe")
        setattr(moe_pkg, "paged_stash", primus_mod)
    except ImportError:
        pass

    log_rank_0(
        f"[Patch:megatron.moe.paged_stash]   Installed {module_path} "
        f"-> primus.backends.megatron.core.transformer.moe.paged_stash"
    )
    return primus_mod


def _add_transformer_config_fields():
    """Add paged-stash related ``TransformerConfig`` fields if missing.

    The upstream PR adds:
        - ``moe_expert_rank_capacity_factor: Optional[float] = None``
        - ``moe_paged_stash: bool = False``
        - ``moe_paged_stash_page_size: int = 64``
        - ``moe_paged_stash_buffer_size_factor_cuda: float = 1.10``
        - ``moe_paged_stash_buffer_size_factor_cpu: float = 0.0``
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    new_fields = [
        ("moe_expert_rank_capacity_factor", Optional[float], None),
        ("moe_paged_stash", bool, False),
        ("moe_paged_stash_page_size", int, 64),
        ("moe_paged_stash_buffer_size_factor_cuda", float, 1.10),
        ("moe_paged_stash_buffer_size_factor_cpu", float, 0.0),
    ]

    existing_fields = TransformerConfig.__dataclass_fields__
    added = []
    for name, typ, default in new_fields:
        if name in existing_fields:
            continue
        field = dataclasses.field(default=default)
        field.name = name
        field.type = typ
        field._field_type = dataclasses._FIELD  # type: ignore[attr-defined]
        existing_fields[name] = field
        TransformerConfig.__annotations__[name] = typ
        setattr(TransformerConfig, name, default)
        added.append(name)

    if added:
        log_rank_0(
            f"[Patch:megatron.moe.paged_stash]   Added TransformerConfig fields: {added}"
        )


def _patch_token_dispatcher():
    """Add paged-stash hooks onto the flex/hybrid EP token dispatcher.

    The upstream changes:
        - ``_HybridEPManager.__init__`` tracks per-manager ``over_budget`` flag.
        - ``_HybridEPManager.setup_metadata`` derives a static
          ``num_permuted_tokens`` when ``moe_expert_rank_capacity_factor`` is
          set.
        - ``_HybridEPManager.combine`` keeps ``num_permuted_tokens`` stable for
          CUDA-graph replay when the capacity factor is in use.
        - ``MoEFlexTokenDispatcher`` exposes ``check_over_budget`` /
          ``reset_over_budget``.
    """
    try:
        from megatron.core.transformer.moe import token_dispatcher
    except ImportError:
        return

    if getattr(token_dispatcher.MoEFlexTokenDispatcher, "_primus_paged_stash_patched", False):
        return

    try:
        HybridEPManager = token_dispatcher._HybridEPManager
    except AttributeError:
        HybridEPManager = None

    if HybridEPManager is not None:
        orig_hybrid_init = HybridEPManager.__init__
        orig_hybrid_setup = HybridEPManager.setup_metadata
        orig_hybrid_combine = HybridEPManager.combine

        def patched_hybrid_init(self, *args, **kwargs):
            orig_hybrid_init(self, *args, **kwargs)
            self.moe_expert_rank_capacity_factor = getattr(
                self.config, "moe_expert_rank_capacity_factor", None
            )
            self.over_budget = torch.zeros(1, dtype=torch.bool, device="cuda")

        def patched_hybrid_setup(self, routing_map, probs):
            if getattr(self, "moe_expert_rank_capacity_factor", None) is not None:
                from megatron.core.transformer.moe.moe_utils import (
                    get_align_size_for_quantization,
                )

                pad_multiple = get_align_size_for_quantization(self.config)
                budget = int(
                    routing_map.shape[0]
                    * self.config.moe_router_topk
                    * self.moe_expert_rank_capacity_factor
                )
                budget += -budget % pad_multiple
                self.num_permuted_tokens = budget
            return orig_hybrid_setup(self, routing_map, probs)

        def patched_hybrid_combine(self, *args, **kwargs):
            result = orig_hybrid_combine(self, *args, **kwargs)
            if getattr(self, "moe_expert_rank_capacity_factor", None) is not None:
                self.num_permuted_tokens = getattr(self, "num_permuted_tokens", None)
            return result

        HybridEPManager.__init__ = patched_hybrid_init
        HybridEPManager.setup_metadata = patched_hybrid_setup
        HybridEPManager.combine = patched_hybrid_combine

    def check_over_budget(self):
        if hasattr(self._comm_manager, "over_budget"):
            return self._comm_manager.over_budget
        return None

    def reset_over_budget(self):
        if hasattr(self._comm_manager, "over_budget"):
            self._comm_manager.over_budget.fill_(0)

    if not hasattr(token_dispatcher.MoEFlexTokenDispatcher, "check_over_budget"):
        token_dispatcher.MoEFlexTokenDispatcher.check_over_budget = check_over_budget
    if not hasattr(token_dispatcher.MoEFlexTokenDispatcher, "reset_over_budget"):
        token_dispatcher.MoEFlexTokenDispatcher.reset_over_budget = reset_over_budget

    token_dispatcher.MoEFlexTokenDispatcher._primus_paged_stash_patched = True

    log_rank_0(
        "[Patch:megatron.moe.paged_stash]   Patched token_dispatcher with over-budget hooks"
    )


def _patch_te_grouped_mlp():
    """Wrap ``TEGroupedMLP.forward`` with the paged-stash activation context.

    The upstream PR wraps the fused TEGroupedMLP forward call with
    ``paged_stash_group_start`` / ``paged_stash_group_commit`` and a
    ``get_paged_stash_context`` range when ``moe_paged_stash`` is enabled.
    """
    try:
        from megatron.core.transformer.moe import experts
    except ImportError:
        return

    TEGroupedMLP = getattr(experts, "TEGroupedMLP", None)
    if TEGroupedMLP is None or getattr(TEGroupedMLP, "_primus_paged_stash_patched", False):
        return

    from megatron.core.transformer.moe.paged_stash import (
        get_paged_stash_context,
        paged_stash_group_commit,
        paged_stash_group_start,
    )

    orig_forward = TEGroupedMLP.forward

    def patched_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs=None):
        if not getattr(self.config, "moe_paged_stash", False):
            return orig_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs)

        permuted_local_hidden_states = paged_stash_group_start(permuted_local_hidden_states)
        max_num_tokens = permuted_local_hidden_states.shape[0]
        cap_factor = getattr(self.config, "moe_expert_rank_capacity_factor", None)
        avg_num_tokens = (
            int(max_num_tokens // cap_factor)
            if cap_factor is not None and cap_factor > 0
            else None
        )
        stash_context = get_paged_stash_context(
            name="grouped_mlp",
            max_num_tokens=max_num_tokens,
            num_tokens_tensor=tokens_per_expert.sum(),
            avg_num_tokens=avg_num_tokens,
        )
        with stash_context:
            output = orig_forward(
                self, permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )
        if isinstance(output, tuple):
            main, *rest = output
            main = paged_stash_group_commit(main, name="grouped_mlp")
            return (main, *rest)
        return paged_stash_group_commit(output, name="grouped_mlp")

    TEGroupedMLP.forward = patched_forward
    TEGroupedMLP._primus_paged_stash_patched = True

    log_rank_0(
        "[Patch:megatron.moe.paged_stash]   Patched TEGroupedMLP.forward to apply paged stash"
    )


def _patch_schedules():
    """Call ``paged_stash_reset`` at the start of each pipeline schedule."""
    try:
        from megatron.core.pipeline_parallel import schedules
    except ImportError:
        return

    from megatron.core.transformer.moe.paged_stash import paged_stash_reset

    for name in (
        "forward_backward_no_pipelining",
        "forward_backward_pipelining_without_interleaving",
        "forward_backward_pipelining_with_interleaving",
    ):
        func = getattr(schedules, name, None)
        if func is None or getattr(func, "_primus_paged_stash_patched", False):
            continue

        original_func = func

        def _make_wrapped(orig):
            def wrapped(*args, **kwargs):
                config = kwargs.get("config", None)
                forward_only = kwargs.get("forward_only", False)
                if config is None:
                    import inspect

                    sig = inspect.signature(orig)
                    bound = sig.bind_partial(*args, **kwargs)
                    config = bound.arguments.get("config", None)
                    forward_only = bound.arguments.get("forward_only", False)
                if getattr(config, "moe_paged_stash", False):
                    paged_stash_reset(enabled=not forward_only, config=config)
                return orig(*args, **kwargs)

            wrapped._primus_paged_stash_patched = True
            return wrapped

        setattr(schedules, name, _make_wrapped(original_func))

    log_rank_0(
        "[Patch:megatron.moe.paged_stash]   Patched pipeline schedules to call paged_stash_reset"
    )


def _patch_gpt_model():
    """Call ``preprocess_for_paged_stash`` at the top of ``GPTModel.forward``.

    The upstream PR adds ``preprocess_for_paged_stash`` on ``GPTModel`` and
    invokes it from ``forward`` when ``moe_paged_stash`` is enabled.
    """
    try:
        from megatron.core.models.gpt.gpt_model import GPTModel
    except ImportError:
        return

    if getattr(GPTModel, "_primus_paged_stash_patched", False):
        return

    from megatron.core.transformer.moe.paged_stash import (
        paged_stash_init_chunk_handler,
    )

    def preprocess_for_paged_stash(self):
        return paged_stash_init_chunk_handler(
            vp_size=self.config.virtual_pipeline_model_parallel_size,
            vp_stage=getattr(self, "vp_stage", None),
        )

    orig_forward = GPTModel.forward

    def patched_forward(self, *args, **kwargs):
        if getattr(self.config, "moe_paged_stash", False):
            self.preprocess_for_paged_stash()
        return orig_forward(self, *args, **kwargs)

    GPTModel.preprocess_for_paged_stash = preprocess_for_paged_stash
    GPTModel.forward = patched_forward
    GPTModel._primus_paged_stash_patched = True

    log_rank_0(
        "[Patch:megatron.moe.paged_stash]   Patched GPTModel.forward with preprocess_for_paged_stash"
    )


def _patch_full_cuda_graph_wrapper():
    """Extend ``FullCudaGraphWrapper`` with paged-stash awareness.

    Adds:
        - ``reset_cuda_graph`` instance method
        - ``speculative_cuda_graph_check`` instance method
        - ``moe_paged_stash`` and ``moe_expert_rank_capacity_factor`` kwargs in
          ``__init__``
    """
    try:
        from megatron.core import full_cuda_graph
    except ImportError:
        return

    Wrapper = getattr(full_cuda_graph, "FullCudaGraphWrapper", None)
    if Wrapper is None or getattr(Wrapper, "_primus_paged_stash_patched", False):
        return

    import gc

    orig_init = Wrapper.__init__

    def patched_init(
        self,
        forward_backward_func,
        cuda_graph_warmup_steps=1,
        moe_paged_stash=False,
        moe_expert_rank_capacity_factor=None,
        **kwargs,
    ):
        orig_init(
            self,
            forward_backward_func,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
            **kwargs,
        )
        self.moe_paged_stash = moe_paged_stash
        self.moe_expert_rank_capacity_factor = moe_expert_rank_capacity_factor

    def reset_cuda_graph(self, stage=None):
        if stage is None or stage == "training":
            if Wrapper.cuda_graph["training"] is not None:
                del Wrapper.cuda_graph["training"]
                Wrapper.cuda_graph["training"] = None
            Wrapper.result["training"] = None
            Wrapper.curr_iteration["training"] = 0
        if stage is None or stage == "validation":
            if Wrapper.cuda_graph["validation"] is not None:
                del Wrapper.cuda_graph["validation"]
                Wrapper.cuda_graph["validation"] = None
            Wrapper.result["validation"] = None
            Wrapper.curr_iteration["validation"] = 0
        gc.collect()

    def speculative_cuda_graph_check(self, model):
        if getattr(self, "moe_expert_rank_capacity_factor", None) is None:
            return
        over_budget = torch.zeros(1, dtype=torch.bool, device="cuda")
        for model_chunk in model:
            module = model_chunk
            for attr in ("module", "module", "decoder"):
                module = getattr(module, attr, module)
            layers = getattr(module, "layers", None)
            if layers is None:
                continue
            for layer in layers:
                mlp = getattr(layer, "mlp", None)
                if mlp is None:
                    continue
                if hasattr(mlp, "token_dispatcher") and hasattr(
                    mlp.token_dispatcher, "check_over_budget"
                ):
                    overflow = mlp.token_dispatcher.check_over_budget()
                    if overflow is not None:
                        over_budget |= overflow
        if over_budget.item():
            raise Exception(f"Rank {torch.distributed.get_rank()} overbudget")

    Wrapper.__init__ = patched_init
    if not hasattr(Wrapper, "reset_cuda_graph"):
        Wrapper.reset_cuda_graph = reset_cuda_graph
    if not hasattr(Wrapper, "speculative_cuda_graph_check"):
        Wrapper.speculative_cuda_graph_check = speculative_cuda_graph_check

    Wrapper._primus_paged_stash_patched = True

    log_rank_0(
        "[Patch:megatron.moe.paged_stash]   Extended FullCudaGraphWrapper with paged-stash helpers"
    )


def _patch_training_loop():
    """Wrap ``forward_backward_func`` with ``PagedStashRunner`` in training.

    Mirrors the upstream PR which wraps the function in both ``train()`` and
    ``evaluate()``. Because those functions are large we patch by replacing
    ``get_forward_backward_func`` at its usage site via a thin indirection.

    Implementation strategy: monkey-patch
    ``megatron.core.pipeline_parallel.get_forward_backward_func`` so that when
    called we return a function that, on first invocation for a given model,
    wraps itself with ``PagedStashRunner``. This avoids editing the training
    loop source while preserving behaviour when ``moe_paged_stash`` is off.
    """
    try:
        from megatron.core import pipeline_parallel as pp_pkg
    except ImportError:
        return

    if getattr(pp_pkg, "_primus_paged_stash_patched", False):
        return

    from megatron.core.transformer.moe.paged_stash import PagedStashRunner

    orig_get = pp_pkg.get_forward_backward_func

    def patched_get():
        fbf = orig_get()

        def wrapped(*args, **kwargs):
            config = kwargs.get("config")
            model = kwargs.get("model")
            forward_only = kwargs.get("forward_only", False)
            if (
                config is not None
                and getattr(config, "moe_expert_rank_capacity_factor", None) is not None
                and model is not None
                and not getattr(wrapped, "_paged_stash_runner", None)
            ):
                try:
                    from megatron.training import get_args as _mg_get_args

                    mg_args = _mg_get_args()
                    copy_main_params = getattr(
                        mg_args, "reuse_grad_buf_for_mxfp8_param_ag", False
                    ) and getattr(mg_args, "overlap_param_gather", False)
                except Exception:
                    copy_main_params = False

                wrapped._paged_stash_runner = PagedStashRunner(
                    config,
                    copy_main_params,
                    model if isinstance(model, list) else [model],
                    None,
                    fbf,
                )
            runner = getattr(wrapped, "_paged_stash_runner", None)
            if runner is not None:
                if forward_only:
                    runner.optimizer = None
                return runner(*args, **kwargs)
            return fbf(*args, **kwargs)

        wrapped._paged_stash_runner = None
        return wrapped

    pp_pkg.get_forward_backward_func = patched_get
    pp_pkg._primus_paged_stash_patched = True

    log_rank_0(
        "[Patch:megatron.moe.paged_stash]   Patched pipeline_parallel.get_forward_backward_func "
        "with PagedStashRunner injection"
    )


def _paged_stash_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return bool(getattr(args, "moe_paged_stash", False))


@register_patch(
    "megatron.moe.paged_stash",
    backend="megatron",
    phase="before_train",
    description="Enable MoE paged-stashing on top of stock Megatron-LM (PR #2690 backport)",
    condition=_paged_stash_enabled,
)
def patch_paged_stash(ctx: PatchContext):
    """Wire up the paged-stash feature at runtime.

    The patch is a no-op when ``--moe_paged_stash`` is not set.
    """
    _install_paged_stash_module()
    _add_transformer_config_fields()
    _patch_full_cuda_graph_wrapper()
    _patch_token_dispatcher()
    _patch_te_grouped_mlp()
    _patch_schedules()
    _patch_gpt_model()
    _patch_training_loop()

    log_rank_0("[Patch:megatron.moe.paged_stash]   Paged-stashing patches applied")
