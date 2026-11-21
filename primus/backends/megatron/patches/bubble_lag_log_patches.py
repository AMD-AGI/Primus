###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Lag the per-iteration ``training_log`` by one step so the GPU never drains between steps
(opt-in: ``PRIMUS_BUBBLE_LAG_LOG=1``; default off).

Problem (full-iteration CUDA graph, 1 GPU): the step-end logging reads GPU values (``.item()``,
``torch.cuda.synchronize`` in the ``interval-time`` timer, barriers, pageable H2D of fresh
``torch.tensor(..., device='cuda')``), so the host waits for the optimizer to finish and then
spends ~1.6 ms of pure host work (loss/aux-loss reduction, MoE metrics, string formatting, print,
memory report) plus the next step's zero_grad / data copy / graph launch while the GPU is idle.
Measured: ~2 ms of GPU idle per step.

Fix: at ``training_log(k)`` the host only queues, on the main stream right after step k, pinned
non-blocking D2H copies of every GPU value the log reads (loss dict, grad norm, MoE metrics
tracker buffers; the tracker is then cleared on the main stream exactly where the original
``report()`` cleared it) and records an event E_k. The real ``training_log(k-1)`` then runs on
the host behind E_{k-1} while step k (graph + optimizer) is already queued, so the GPU stays
busy. The lagged log touches no GPU at all: tensor factories asked for 'cuda' build CPU tensors
(a side stream does not help: HIP serialises a side-stream H2D behind a running graph, measured
~40 ms wait), ``torch.cuda.synchronize`` in the timer is a no-op (the completion event was waited
for), and the 1-rank collectives (``barrier``/``all_reduce``) are no-ops (identities on one
rank). The loop's ``optimizer.get_loss_scale()`` returns a cached CPU copy of the static BF16
unity scale. Training numerics are untouched: no op of the training step changes or is
reordered; only a few tiny D2H copies are added after the optimizer.

Logged values are the same as without the patch (they are snapshots of step k taken in stream
order right after step k). ``elapsed time per iteration`` is the host time between consecutive
lagged logs, each gated by the completion event of its step, i.e. the GPU step period.

Scope / safety:
  - active only when the world size is 1 (the logging collectives on a multi-rank world would
    share the NCCL streams of the training step; that needs a dedicated logging communicator),
  - only after the full-iteration CUDA graph has been captured,
  - only when every optimizer uses a static scale (no grad scaler, i.e. BF16),
  - the pending log is flushed synchronously at the last iteration and at any iteration that
    evaluates, saves or exits.
"""

import contextlib
import inspect
import os
import time

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled() -> bool:
    return os.environ.get("PRIMUS_BUBBLE_LAG_LOG", "0") == "1"


@register_patch(
    "megatron.bubble_lag_log",
    backend="megatron",
    phase="before_train",
    description="Lag training_log by one step so the GPU does not drain between steps (PRIMUS_BUBBLE_LAG_LOG=1)",
    priority=98,
    condition=lambda ctx: _enabled(),
)
def patch_bubble_lag_log(ctx: PatchContext) -> None:
    del ctx
    import torch
    import megatron.training.training as mt
    from megatron.core.full_cuda_graph import FullCudaGraphWrapper
    from megatron.training import get_args

    try:
        from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
    except Exception:  # older megatron-core
        get_moe_metrics_tracker = None

    orig_log = mt.training_log
    orig_train_step = mt.train_step
    _names = ["loss_dict", "total_loss_dict", "learning_rate", "iteration", "loss_scale",
              "report_memory_flag", "skipped_iter", "grad_norm", "params_norm", "num_zeros_in_grad",
              "max_attention_logit"]
    try:  # other patches may wrap training_log with (*args, **kwargs)
        _params = list(inspect.signature(inspect.unwrap(orig_log)).parameters)
        if "iteration" in _params and "loss_dict" in _params:
            _names = [n for n in _params if n not in ("args", "kwargs")]
    except (TypeError, ValueError):
        pass

    def _bind(a, k):
        if len(a) > len(_names):
            raise TypeError("training_log: unexpected positional arguments")
        kw = dict(zip(_names, a))
        kw.update(k)
        return kw

    st = {"active": False, "disabled": False, "pending": None, "in_step": False, "cpu_scale": None, "opt": None}

    def _static_scale_ok(opt):
        children = getattr(opt, "chained_optimizers", None) or [opt]
        return all(getattr(c, "grad_scaler", None) is None for c in children)

    def train_step(*a, **k):
        opt = a[3] if len(a) > 3 else k.get("optimizer")
        if opt is not None and st["opt"] is None:
            st["opt"] = opt
            orig_gls = opt.get_loss_scale

            def get_loss_scale():
                if st["active"] and not st["in_step"] and st["cpu_scale"] is not None:
                    return st["cpu_scale"]
                return orig_gls()

            opt.get_loss_scale = get_loss_scale
        st["in_step"] = True
        try:
            return orig_train_step(*a, **k)
        finally:
            st["in_step"] = False

    def _flush_iteration(args, iteration):
        if iteration >= args.train_iters:
            return True
        if args.eval_interval and iteration % args.eval_interval == 0 and getattr(args, "do_valid", False):
            return True
        if args.save and args.save_interval and iteration % args.save_interval == 0:
            return True
        if getattr(args, "exit_interval", None) and iteration % args.exit_interval == 0:
            return True
        if getattr(args, "exit_duration_in_mins", None) or getattr(args, "exit_signal_handler", False):
            return True
        return False

    def _try_activate(args):
        if st["disabled"]:
            return False
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            st["disabled"] = True
            log_rank_0("[Patch:bubble_lag_log] world size != 1: lagged logging disabled")
            return False
        if FullCudaGraphWrapper.cuda_graph.get("training") is None:
            return False
        opt = st["opt"]
        if opt is None or not _static_scale_ok(opt):
            st["disabled"] = True
            log_rank_0("[Patch:bubble_lag_log] optimizer has a grad scaler: lagged logging disabled")
            return False
        torch.cuda.synchronize()
        st["cpu_scale"] = opt.get_loss_scale().detach().to("cpu").reshape(-1)[:1].clone()
        st["active"] = True
        log_rank_0("[Patch:bubble_lag_log] active: training_log runs one step late, host-only")
        return True

    def _d2h(v):
        """Async copy of a device tensor into pinned host memory (ordered on the current stream)."""
        if isinstance(v, torch.Tensor) and v.is_cuda:
            c = torch.empty(v.shape, dtype=v.dtype, device="cpu", pin_memory=True)
            c.copy_(v.detach(), non_blocking=True)
            return c
        return v

    def _snapshot(kw, args):
        """Runs on the main stream right after step k was queued: pinned D2H copies of every GPU
        value the log reads, then the MoE tracker is cleared where report() -> clear() did it."""
        kw = dict(kw)
        kw["loss_dict"] = {k: _d2h(v) for k, v in kw["loss_dict"].items()}
        for name in ("grad_norm", "params_norm", "num_zeros_in_grad", "max_attention_logit", "loss_scale"):
            if name in kw:
                kw[name] = _d2h(kw[name])
        tracker = {}
        if get_moe_metrics_tracker is not None:
            tr = get_moe_metrics_tracker()
            for name, entry in getattr(tr, "_metrics", {}).items():
                tracker[name] = _d2h(entry.values)
                entry.values.zero_()
        ev = torch.cuda.Event()
        ev.record()
        gl = {a: getattr(args, a) for a in ("consumed_train_samples", "skipped_train_samples",
                                           "consumed_train_bins", "curr_iteration") if hasattr(args, a)}
        return {"kw": kw, "tracker": tracker, "event": ev, "args": gl}

    def _dict_to(d, device):
        for k, v in list(d.items()):
            if isinstance(v, torch.Tensor):
                d[k] = v.to(device)

    def _cpu_dev(d):
        if isinstance(d, int) and not isinstance(d, bool):
            return "cpu"
        if isinstance(d, str) and d.startswith("cuda"):
            return "cpu"
        if isinstance(d, torch.device) and d.type == "cuda":
            return "cpu"
        return d

    @contextlib.contextmanager
    def _cpu_runtime():
        """The lagged log runs on host tensors only: any GPU op would queue behind the step that
        is executing (HIP serialises side-stream copies behind a running graph), so tensor
        factories asked for 'cuda' build CPU tensors, the timer's device synchronize is a no-op
        (the step's completion event was already waited for) and 1-rank collectives are no-ops."""
        saved = {}
        for fn in ("tensor", "zeros", "ones", "full", "empty"):
            orig = getattr(torch, fn)
            saved[fn] = orig

            def make(orig=orig):
                def f(*a, **k):
                    if "device" in k:
                        k["device"] = _cpu_dev(k["device"])
                    return orig(*a, **k)

                return f

            setattr(torch, fn, make())
        cuda_sync = torch.cuda.synchronize
        barrier = torch.distributed.barrier
        all_reduce = torch.distributed.all_reduce

        def noop(*a, **k):
            return None

        torch.cuda.synchronize = noop
        torch.distributed.barrier = noop
        torch.distributed.all_reduce = noop
        try:
            yield
        finally:
            for fn, orig in saved.items():
                setattr(torch, fn, orig)
            torch.cuda.synchronize = cuda_sync
            torch.distributed.barrier = barrier
            torch.distributed.all_reduce = all_reduce

    dbg = os.environ.get("PRIMUS_BUBBLE_LAG_LOG_DEBUG", "0") == "1"
    prof_iter = int(os.environ.get("PRIMUS_BUBBLE_LAG_LOG_PROFILE_ITER", "-1"))

    def _run(snap, args, report_memory_flag):
        t0 = time.perf_counter()
        snap["event"].synchronize()  # host waits for step k-1 only; step k is queued on the GPU
        t1 = time.perf_counter()
        saved_tr, saved_args = {}, {}
        tr = get_moe_metrics_tracker() if (get_moe_metrics_tracker is not None and snap["tracker"]) else None
        kw = dict(snap["kw"])
        kw["report_memory_flag"] = report_memory_flag
        with _cpu_runtime():
            try:
                if tr is not None:
                    for name, v in snap["tracker"].items():
                        if name in tr._metrics:
                            saved_tr[name] = tr._metrics[name].values
                            tr._metrics[name].values = v
                for a, v in snap["args"].items():
                    saved_args[a] = getattr(args, a)
                    setattr(args, a, v)
                if kw["iteration"] == prof_iter:
                    import sys as _sys

                    stack, slow = [], []

                    def _prof(frame, event, arg):
                        if event in ("call", "c_call"):
                            nm = (f"{frame.f_code.co_filename.split('/')[-1]}:{frame.f_code.co_name}" if event == "call"
                                  else f"C:{getattr(arg, '__qualname__', getattr(arg, '__name__', str(arg)))}")
                            stack.append((nm, time.perf_counter()))
                        elif event in ("return", "c_return", "c_exception") and stack:
                            nm, t = stack.pop()
                            d = time.perf_counter() - t
                            if d > 0.002:
                                slow.append((d * 1e3, " < ".join([nm] + [x[0] for x in stack[-4:]][::-1])))

                    _sys.setprofile(_prof)
                    try:
                        flag = orig_log(**kw)
                    finally:
                        _sys.setprofile(None)
                    out = os.environ.get("PRIMUS_BUBBLE_LAG_LOG_PROFILE_OUT", "/tmp/bubble_lag_log.prof.txt")
                    with open(out, "w") as fh:
                        for d, nm in sorted(slow, key=lambda x: -x[0])[:80]:
                            fh.write(f"{d:9.3f} ms  {nm}\n")
                else:
                    flag = orig_log(**kw)
            finally:
                for a, v in saved_args.items():
                    setattr(args, a, v)
                if tr is not None:
                    for name, v in saved_tr.items():
                        tr._metrics[name].values = v
        if dbg:
            t2 = time.perf_counter()
            log_rank_0(f"[bubble_lag_log] it {kw['iteration']} wait_ms={(t1 - t0) * 1e3:.3f} "
                       f"log_ms={(t2 - t1) * 1e3:.3f}")
        return flag

    def training_log(*a, **k):
        args = get_args()
        if not st["active"] and not _try_activate(args):
            return orig_log(*a, **k)
        kw = _bind(a, k)
        if st["pending"] is None:
            _dict_to(kw["total_loss_dict"], "cpu")  # first lagged step (just synchronised)
        iteration = kw["iteration"]
        flag = kw["report_memory_flag"]
        snap = _snapshot(kw, args)
        if st["pending"] is not None:
            flag = _run(st["pending"], args, flag)
        st["pending"] = snap
        if _flush_iteration(args, iteration):
            flag = _run(st["pending"], args, flag)
            st["pending"] = None
            st["active"] = False
            torch.cuda.synchronize()
            _dict_to(kw["total_loss_dict"], torch.cuda.current_device())
            # stay inactive until the next call re-activates (graph is still captured)
        return flag

    mt.train_step = train_step
    mt.training_log = training_log
    log_rank_0("[Patch:bubble_lag_log] installed (PRIMUS_BUBBLE_LAG_LOG=1)")
