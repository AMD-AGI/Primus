###############################################################################
# Diagnostic only, off by default: per-iteration CUDA-event timing of the MoE all-to-all
# and of the token-dedup kernels / dispatcher phases. Enabled with PRIMUS_MOE_A2A_TIMING=1.
# Prints "[moe-timing] iteration N | cat=mean/max ms ..." (mean and max over all ranks).
###############################################################################
import functools
import os
from collections import defaultdict

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_EVENTS = defaultdict(list)  # category -> [(start_event, end_event)]
_ITER = [0]


def _timed(category, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import torch

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(*args, **kwargs)
        e.record()
        _EVENTS[category].append((s, e))
        return out

    return wrapper


def _wrap_autograd_fn(cls, cat):
    fwd, bwd = cls.forward, cls.backward
    cls.forward = staticmethod(_timed(cat + "_fwd", fwd.__func__ if hasattr(fwd, "__func__") else fwd))
    cls.backward = staticmethod(_timed(cat + "_bwd", bwd.__func__ if hasattr(bwd, "__func__") else bwd))


def _wrap_methods(cls, names, prefix):
    for n in names:
        if n in cls.__dict__:
            setattr(cls, n, _timed(f"{prefix}{n}", cls.__dict__[n]))


def _flush_and_log():
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize()
    cats = sorted(_EVENTS)
    vals = torch.tensor([sum(s.elapsed_time(e) for s, e in _EVENTS[c]) for c in cats], dtype=torch.float64, device="cuda")
    _EVENTS.clear()
    if not cats:
        return
    mx = vals.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mean = vals.clone()
    dist.all_reduce(mean, op=dist.ReduceOp.SUM)
    mean /= dist.get_world_size()
    if dist.get_rank() == 0:
        parts = [f"{c}={mean[i].item():.1f}/{mx[i].item():.1f}" for i, c in enumerate(cats)]
        print(f"[moe-timing] iteration {_ITER[0]} | mean/max ms over ranks | " + " | ".join(parts), flush=True)


def _enabled(ctx: PatchContext) -> bool:
    return os.environ.get("PRIMUS_MOE_A2A_TIMING", "0") == "1"


@register_patch(
    "megatron.moe.a2a_timing",
    backend="megatron",
    phase="before_train",
    description="Diagnostic: CUDA-event timing of MoE all-to-all, dedup kernels and dispatcher phases per iteration",
    condition=_enabled,
)
def patch_moe_a2a_timing(ctx: PatchContext):
    import megatron.core.tensor_parallel.mappings as mappings
    import megatron.training.training as training
    from megatron.core.transformer.moe import token_dispatcher as core_td

    from primus.backends.megatron.core.transformer.moe import token_dedup, token_dedup_dispatcher as ptd

    phases = ["dispatch_preprocess", "token_dispatch", "dispatch_postprocess", "combine_preprocess", "token_combine", "combine_postprocess"]
    classes = [core_td.MoEAlltoAllTokenDispatcher] + [getattr(ptd, n) for n in dir(ptd) if n.endswith("TokenDispatcher") and isinstance(getattr(ptd, n), type)]
    _wrap_autograd_fn(mappings._AllToAll, "a2a")
    fine = os.environ.get("PRIMUS_MOE_A2A_TIMING_FINE", "0") == "1"
    for name in ["_PackTokenDedup", "_ExpandTokenDedup", "_PermuteTokenDedup", "_ReduceTokenDedup",
                 "_RestoreTokenDedup", "_WeightedReduceTokenDedup", "_UnpackTokenDedupPayload"]:
        if hasattr(token_dedup, name):
            _wrap_autograd_fn(getattr(token_dedup, name), "dedup" if not fine else "dd" + name.replace("TokenDedup", "").replace("Payload", ""))
    if fine:
        # forward-only index-build helpers (called by name from the dispatcher module)
        for fn in ["routing_counts", "build_wire_send_index", "build_wire_recv_index", "column_prefix_sum", "segment_ranks"]:
            if hasattr(ptd, fn):
                setattr(ptd, fn, _timed("ix_" + fn, getattr(ptd, fn)))
        # baseline dispatcher permutation kernels (what dedup's pack/unpack/reduce/restore replace)
        try:
            from megatron.core.transformer.moe import moe_utils
            for fn in ["permute", "unpermute", "sort_chunks_by_idxs"]:
                if hasattr(moe_utils, fn):
                    timed = _timed("ix_" + fn, getattr(moe_utils, fn))
                    setattr(moe_utils, fn, timed)
                    if hasattr(core_td, fn):
                        setattr(core_td, fn, timed)
        except Exception as e:  # pragma: no cover
            log_rank_0(f"[Patch:megatron.moe.a2a_timing] permute wrap skipped: {e}")
        for cls in classes:
            for m in ["_metadata_gather_launch", "_metadata_finish", "_to_slot_space"]:
                if m in cls.__dict__:
                    setattr(cls, m, _timed("ix_" + m, cls.__dict__[m]))
    for cls in classes:
        _wrap_methods(cls, phases, "phase_")

    orig_training_log = training.training_log

    @functools.wraps(orig_training_log)
    def training_log_timed(*args, **kwargs):
        _ITER[0] += 1
        _flush_and_log()
        return orig_training_log(*args, **kwargs)

    training.training_log = training_log_timed
    log_rank_0(f"[Patch:megatron.moe.a2a_timing] wrapped _AllToAll, {len(classes)} dispatcher classes, dedup autograd fns; logging per iteration")
