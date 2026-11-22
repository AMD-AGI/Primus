# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Move a prefix of the third forward into the initial two-stage pipeline bubble."""

from contextvars import ContextVar
from dataclasses import dataclass
from random import getstate, setstate

import numpy as np
import torch

_pending_prefix = ContextVar("primus_pipeline_forward_prefix", default=None)


@dataclass
class _RNGState:
    cpu: torch.Tensor
    gpu: torch.Tensor
    named: dict
    python: tuple
    numpy: tuple

    @classmethod
    def capture(cls, tracker):
        states = tracker.get_states()
        if any(not isinstance(value, torch.Tensor) for value in states.values()):
            raise ValueError("Pipeline forward prefix requires eager Megatron RNG tracking.")
        return cls(
            torch.get_rng_state(),
            torch.cuda.get_rng_state(),
            {k: v.clone() for k, v in states.items()},
            getstate(),
            np.random.get_state(),
        )

    def restore(self, tracker):
        torch.set_rng_state(self.cpu)
        torch.cuda.set_rng_state(self.gpu)
        tracker.set_states({k: v.clone() for k, v in self.named.items()})
        setstate(self.python)
        np.random.set_state(self.numpy)

    def matches(self, other):
        return (
            torch.equal(self.cpu, other.cpu)
            and torch.equal(self.gpu, other.gpu)
            and self.named.keys() == other.named.keys()
            and all(torch.equal(value, other.named[k]) for k, value in self.named.items())
            and self.python == other.python
            and self.numpy[0] == other.numpy[0]
            and np.array_equal(self.numpy[1], other.numpy[1])
            and self.numpy[2:] == other.numpy[2:]
        )


def _ambient_state():
    from primus.backends.megatron.core.extensions.primus_turbo import (
        FP8GlobalStateManager,
        PrimusTurboLowPrecisionGlobalStateManager,
    )

    return (
        torch.is_grad_enabled(),
        torch.is_autocast_enabled("cuda"),
        torch.is_autocast_enabled("cpu"),
        torch.cuda.current_stream(),
        getattr(FP8GlobalStateManager, "FP8_AUTOCAST_DEPTH", None),
        PrimusTurboLowPrecisionGlobalStateManager.get_fp8_autocast_state(),
    )


class ForwardPrefix:
    def __init__(self, model, depth, forward_step, timers):
        from megatron.core.tensor_parallel import get_cuda_rng_tracker
        from megatron.core.utils import get_attr_wrapped_model

        self.model = model
        self.decoder = get_attr_wrapped_model(model, "decoder", allow_none=False)
        if not 0 < depth < len(self.decoder.layers):
            raise ValueError("pipeline_forward_prefix_layers must leave at least one layer for the suffix.")
        ranks = torch.distributed.get_process_group_ranks
        stage = set(ranks(model.dp_cp_group))
        if any(not stage.issuperset(ranks(group)) for group in (model.ep_group, model.expt_dp_group)):
            raise ValueError(
                "Forward prefix requires all expert collectives to stay on the same pipeline stage."
            )
        self.depth = depth
        self.forward_step = forward_step
        self.timers = timers
        self.tracker = get_cuda_rng_tracker()
        self.handoff = None

    def defer(self, handoff):
        self.handoff = handoff
        self.baseline_rng = _RNGState.capture(self.tracker)
        self.baseline_context = _ambient_state()
        grad_enabled, cuda_autocast, cpu_autocast, _, _, precision = self.baseline_context
        if not grad_enabled or any(
            (cuda_autocast, cpu_autocast, precision[0], precision[1], precision[6], precision[7])
        ):
            raise RuntimeError(
                "Forward prefix requires the ordinary training context outside FP8 and autocast."
            )

    def forward(self, *args, **kwargs):
        token = _pending_prefix.set(self)
        try:
            result = self.forward_step(*args, **kwargs)
        finally:
            _pending_prefix.reset(token)
        if self.handoff is not None:
            raise RuntimeError("The requested forward prefix was not reached in the decoder.")
        return result

    def run_handoff(self):
        from megatron.core.transformer.cuda_graphs import set_current_microbatch

        # The preceding layer's precision context has exited. Preserve F2's random
        # routing draws and run B0 under its original (post-F1) RNG and microbatch ID.
        if _ambient_state() != self.baseline_context or any(
            layer.current_microbatch != 2 for layer in self.decoder.layers
        ):
            raise RuntimeError("Forward prefix reached a different precision, stream or microbatch context.")
        prefix_rng = _RNGState.capture(self.tracker)
        handoff, self.handoff = self.handoff, None
        self.baseline_rng.restore(self.tracker)
        set_current_microbatch(self.model, 1)
        # Suspend the native F2 compute timer while the original P2P and B0
        # timers run. Below log level 2 Megatron supplies a no-op timer.
        forward_timer = self.timers("forward-compute", log_level=2) if self.timers is not None else None
        if forward_timer is not None:
            forward_timer.stop()
        try:
            following_input = handoff()
            if following_input is not None and not (
                isinstance(following_input, (tuple, list))
                and len(following_input) == 1
                and following_input[0] is None
            ):
                raise RuntimeError("Forward prefix expected the empty input of the first pipeline stage.")
            if not self.baseline_rng.matches(_RNGState.capture(self.tracker)):
                raise RuntimeError(
                    "Forward prefix cannot reorder a backward pass that consumes random numbers."
                )
        finally:
            if forward_timer is not None:
                forward_timer.start()
            set_current_microbatch(self.model, 2)
            prefix_rng.restore(self.tracker)
        if _ambient_state() != self.baseline_context:
            raise RuntimeError("Backward changed the precision or stream context of the suspended forward.")


def after_decoder_layer(decoder, completed):
    prefix = _pending_prefix.get()
    if prefix is not None and decoder is prefix.decoder and completed == prefix.depth:
        prefix.run_handoff()


def prefetch_expert_parameters(model):
    """Let the native first-use DDP hook finish the last stage's expert all-gather."""
    if not model.remove_forward_pre_hook_handles:
        return  # Megatron disables parameter hooks on the first optimizer step and during eval.
    groups = model.expert_parallel_bucket_groups
    if len(groups) != 1 or groups[0].next_param_gather_bucket_group is not None:
        raise ValueError(
            "early_expert_param_gather requires one unchained expert bucket group on the last stage."
        )
    group = groups[0]
    if group.intra_distributed_optimizer_instance_group is not model.intra_expt_dp_group:
        raise ValueError("Early expert gather expects the native expert data parallel optimizer group.")
    if not group.param_gather_dispatched:
        model._start_bucket_group_param_sync(group, force_sync=False)
