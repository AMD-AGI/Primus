###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PRIMUS_OVERLAP_*: join the side stream of the overlapped wgrads after every microbatch backward.

Primus-Turbo's ``primus_turbo.pytorch.core.overlap.defer`` runs weight-gradient GEMMs (whose only consumer is the
optimizer) on a side stream next to the dgrad chain (PRIMUS_OVERLAP_WGRAD / PRIMUS_OVERLAP_MOE_WGRAD). This patch
wraps Megatron's ``schedules.backward_step`` so that the current stream waits for that side stream right after each
microbatch's backward: before the next microbatch, ``finalize_model_grads``, the grad norm and the optimizer. Under
``--cuda_graph_impl full_iteration`` the fork and the join are edges of the same capture.

EP > 1 / TP > 1 / DP > 1 need no special handling for correctness (the join precedes every grad reduction), but the
overlap has only been measured on the 1-GPU proxy.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_SWITCHES = ("WGRAD", "MOE_WGRAD", "SHARED", "INDEXER")


def _is_enabled(ctx=None) -> bool:
    return any(os.environ.get(f"PRIMUS_OVERLAP_{k}", "0") == "1" for k in _SWITCHES)


@register_patch(
    "megatron.overlap.join_after_backward",
    backend="megatron",
    phase="before_train",
    description="Join the PRIMUS_OVERLAP side stream after every microbatch backward",
    priority=72,
    condition=_is_enabled,
)
def patch_overlap_join(ctx: PatchContext) -> None:
    del ctx
    import megatron.core.pipeline_parallel.schedules as sched
    from primus_turbo.pytorch.core import overlap

    orig = sched.backward_step
    if getattr(orig, "_primus_overlap_join", False):
        return

    def backward_step(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        finally:
            overlap.join()

    backward_step._primus_overlap_join = True
    sched.backward_step = backward_step
    on = [k for k in _SWITCHES if os.environ.get(f"PRIMUS_OVERLAP_{k}", "0") == "1"]
    log_rank_0(f"[Patch:overlap] side-stream join after backward_step installed; switches on: {on}")


def _checksum_file():
    return os.environ.get("PRIMUS_OVERLAP_CHECKSUM_FILE", "")


def _fingerprint(t) -> str:
    """Bit-sensitive fingerprint of a CUDA tensor: position-weighted sum of its raw words (chunked, int64)."""
    import torch

    flat = t.detach().reshape(-1)
    es = flat.element_size()
    words = flat.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[es])
    acc = torch.zeros((), dtype=torch.int64, device=t.device)
    step = 1 << 26
    for s in range(0, words.numel(), step):
        w = words[s:s + step].to(torch.int64)
        idx = torch.arange(s, s + w.numel(), device=t.device, dtype=torch.int64) % 1000003 + 1
        acc += (w * idx).sum()
    return f"{tuple(t.shape)}:{t.dtype}:{int(acc.item())}"


@register_patch(
    "megatron.overlap.checksum",
    backend="megatron",
    phase="before_train",
    description="PRIMUS_OVERLAP_CHECKSUM_FILE: bit fingerprints of params / main_grads / optimizer state after the last step",
    priority=73,
    condition=lambda ctx: bool(_checksum_file()),
)
def patch_overlap_checksum(ctx: PatchContext) -> None:
    del ctx
    import megatron.training.training as mt
    import torch
    from megatron.training import get_args

    orig = mt.train_step
    if getattr(orig, "_primus_overlap_ck", False):
        return
    state = {"n": 0}

    def train_step(*args, **kwargs):
        out = orig(*args, **kwargs)
        state["n"] += 1
        if state["n"] == get_args().train_iters:
            model = kwargs.get("model", args[2] if len(args) > 2 else None)
            optimizer = kwargs.get("optimizer", args[3] if len(args) > 3 else None)
            torch.cuda.synchronize()
            lines = []
            for ci, chunk in enumerate(model if isinstance(model, list) else [model]):
                for n, p in chunk.named_parameters():
                    lines.append(f"param {ci}.{n} {_fingerprint(p)}")
                    mg = getattr(p, "main_grad", None)
                    if isinstance(mg, torch.Tensor):
                        lines.append(f"main_grad {ci}.{n} {_fingerprint(mg)}")
            opts = list(getattr(optimizer, "chained_optimizers", None) or [optimizer])
            for oi, o in enumerate(opts):
                inner = getattr(o, "optimizer", None)
                if not isinstance(inner, torch.optim.Optimizer):
                    continue
                for gi, g in enumerate(inner.param_groups):
                    for pi, p in enumerate(g["params"]):
                        lines.append(f"opt{oi}.g{gi}.p{pi} {_fingerprint(p)}")
                        for k, v in inner.state.get(p, {}).items():
                            if isinstance(v, torch.Tensor) and v.is_cuda and v.numel() > 1:
                                lines.append(f"opt{oi}.g{gi}.p{pi}.{k} {_fingerprint(v)}")
            with open(_checksum_file(), "w") as f:
                f.write("\n".join(lines) + "\n")
            log_rank_0(f"[Patch:overlap] wrote {len(lines)} fingerprints to {_checksum_file()}")
        return out

    train_step._primus_overlap_ck = True
    mt.train_step = train_step
    log_rank_0("[Patch:overlap] checksum after the last train_step installed")
