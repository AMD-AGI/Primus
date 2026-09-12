###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import statistics
import threading
import time

import torch
from megatron.core import parallel_state
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.training import ft_integration, get_args, get_timers
from megatron.training.utils import is_last_rank

from primus.backends.megatron.training.eval_budget import (
    get_eval_global_batch_size,
    get_eval_micro_batch_size,
    get_eval_num_microbatches,
)
from primus.backends.megatron.training.eval_session import begin_eval_session
from primus.backends.megatron.training.global_vars import get_train_start_time
from primus.backends.megatron.training.utils import is_pipeline_stage_containing_loss
from primus.core.utils.module_utils import debug_rank_0, log_rank_0

# The key under which the diffusion validation path reports
# (summed per-sample loss, sample count). Its denominator is the only one that
# is a sample count rather than a microbatch count.
VAL_LOSS_KEY = "loss"


def _make_eval_profiler():
    """Trace the evaluation loop, off unless MXFP6_EVAL_PROFILE=<iterations>.

    PROBE ONLY. Megatron builds its profiler inside training.train(), which
    skip_train bypasses, so an evaluation-only arm cannot be traced through the
    ordinary profile / profile_step_start path -- those keys are read but nothing
    ever creates the profiler. That left the eval loop the one part of the step
    budget never profiled, which is what this exists to fix.

    Profiles rank 0 only, for the first MXFP6_EVAL_PROFILE iterations after one
    wait and one warmup, and writes a chrome trace to MXFP6_EVAL_PROFILE_DIR
    (default /tmp). Stacks are deliberately not collected: this campaign already
    established they add phantom overhead of the same order as the eval anomaly
    being chased.
    """
    active = int(os.environ.get("MXFP6_EVAL_PROFILE", "0"))
    if active <= 0 or torch.distributed.get_rank() != 0:
        return None

    out_dir = os.environ.get("MXFP6_EVAL_PROFILE_DIR", "/tmp")
    os.makedirs(out_dir, exist_ok=True)

    def _export(prof):
        path = os.path.join(out_dir, "eval_profile.pt.trace.json")
        prof.export_chrome_trace(path)
        log_rank_0(f"[MXFP6_EVAL_PROFILE] wrote {path}")

    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=active, repeat=1),
        record_shapes=True,
        with_stack=False,
        on_trace_ready=_export,
    )


_VAL_PREFETCH = os.environ.get("MXFP6_VAL_PREFETCH", "1") == "1"

# Keyed by id() of the validation iterator, so a recipe with more than one
# validation set cannot hand one set's batch to another.
_val_prefetch_state = {}


class _FirstBatchFrom:
    """Yield one already-fetched batch, then delegate to the real iterator."""

    def __init__(self, batch, inner):
        self._batch = batch
        self._inner = inner

    def __iter__(self):
        return self

    def __next__(self):
        if self._batch is not None:
            batch, self._batch = self._batch, None
            return batch
        return next(self._inner)


def _start_val_prefetch(data_iterator):
    """Fetch the next evaluation's first batch on a background thread.

    The first fetch of every evaluation is orders of magnitude slower than the rest,
    which the loader hides: the loop consumes the split exactly, so every evaluation
    ends on an epoch boundary and the next one's first fetch pays the restart with
    nothing queued across it. Doing that fetch at the end of an evaluation would only
    move the cost inside the same timed region, so it is issued here and the training
    steps that follow absorb it.

    The batch this pulls is the one the next evaluation would have read first: the
    validation stream repeats deterministically, and the eval noise is keyed to the
    eval step index rather than to the fetch, so ``val_loss`` is unchanged. That
    bit-identity is the gate; ``MXFP6_VAL_PREFETCH=0`` turns this off.
    """
    if not _VAL_PREFETCH or data_iterator is None:
        return
    key = id(data_iterator)
    prev = _val_prefetch_state.get(key)
    if prev is not None and prev["thread"].is_alive():
        # Nothing consumed the last prefetch, so the iterator's position is not
        # what this would assume. Leave it alone rather than fetch twice.
        return
    state = {"thread": None, "batch": None, "error": None}

    def _fetch():
        try:
            state["batch"] = next(data_iterator)
        except BaseException as exc:  # surfaced on the consuming side
            state["error"] = exc

    # Daemon: a fetch blocked on storage must not keep a finished run alive.
    state["thread"] = threading.Thread(target=_fetch, name="val-prefetch", daemon=True)
    _val_prefetch_state[key] = state
    state["thread"].start()


def _take_val_prefetch(data_iterator):
    """Return the iterator, with its first batch already in hand if we have one.

    Joins the background fetch first: a DataLoader iterator is not safe to advance
    from two threads, so the loop must not touch it until that thread is done.
    """
    state = _val_prefetch_state.pop(id(data_iterator), None)
    if state is None:
        return data_iterator, None
    t0 = time.perf_counter()
    state["thread"].join()
    join_ms = (time.perf_counter() - t0) * 1000.0
    if state["error"] is not None or state["batch"] is None:
        # Fall back to fetching in the loop. The error resurfaces there if it is
        # real, with the traceback the loop would have produced anyway.
        debug_rank_0(f"[MXFP6_VAL_PREFETCH] discarded: {state['error']!r}")
        return data_iterator, None
    return _FirstBatchFrom(state["batch"], data_iterator), join_ms


class _EvalIterTimer:
    """Iteration-to-iteration wall clock for the evaluation loop, off unless
    MXFP6_EVAL_ITER_TIMING=1.

    PROBE ONLY, and it exists because pricing evaluation off a window *inside* the
    iteration reports a much faster iteration, and a much higher GPU-busy fraction,
    than the same arm's end-to-end rate over the split implies. The difference is host
    time that was never in any profiled window, and a within-iteration probe cannot see
    it by construction.

    So this measures the period, not the activity: wall clock from the top of one
    iteration to the top of the next, the wall clock of the forward-backward call inside
    it, and CUDA-event GPU time for that same call. The three split the iteration into
    GPU work, host time inside the model call (which includes the dataloader, since the
    forward step is what consumes the iterator), and loop overhead outside it.

    It is deliberately sync-free in the hot path. The CUDA events are recorded and read
    only at report time, so the probe cannot create the serialization it is looking for.
    That failure mode is real and was hit once already on this model, where an apparent
    end-of-step "host stall" turned out to be the profiler's own overhead.
    """

    def __init__(self):
        self.iter_wall = []
        self.fbf_wall = []
        self.events = []
        self._t_iter = None
        self._t_fbf = None
        self._pair = None
        self.losses = []

    @classmethod
    def maybe_create(cls):
        if os.environ.get("MXFP6_EVAL_ITER_TIMING", "0") not in ("1", "true", "True"):
            return None
        return cls()

    def iteration_start(self):
        self._t_iter = time.perf_counter()

    def fbf_start(self):
        self._t_fbf = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        self._pair = (start, stop)

    def fbf_stop(self):
        self._pair[1].record()
        self.events.append(self._pair)
        self.fbf_wall.append((time.perf_counter() - self._t_fbf) * 1000.0)

    def iteration_stop(self):
        self.iter_wall.append((time.perf_counter() - self._t_iter) * 1000.0)

    def record_loss(self, numerator, denominator):
        """Per-iteration loss, kept to localise a whole-eval loss difference.

        When two arms report different whole-eval losses, each reproducibly, the
        aggregate cannot say whether every iteration moved a little or one moved a
        lot, and those have different causes: a uniform shift is a kernel or reduction
        difference applying to all batches, while a single differing iteration is a
        data or first-batch state difference. Stored as float, compared offline.
        """
        num = numerator.item() if isinstance(numerator, torch.Tensor) else float(numerator)
        den = denominator.item() if isinstance(denominator, torch.Tensor) else float(denominator)
        self.losses.append(num / den if den else float("nan"))

    def report(self):
        torch.cuda.synchronize()
        gpu = [a.elapsed_time(b) for a, b in self.events]
        n = len(self.iter_wall)
        if n == 0:
            return

        def stats(xs):
            s = sorted(xs)
            return (
                statistics.median(s),
                s[len(s) // 10],
                s[min(len(s) - 1, 9 * len(s) // 10)],
                min(s),
                max(s),
                sum(s),
            )

        # The CUDA event pair measures the *span* of the device timeline between the two
        # records, not GPU busy time: if the host blocks on the dataloader mid-call the
        # GPU sits idle inside that span and the span still counts it. Read it as an
        # upper bound on GPU work, and take the compute floor from a standalone
        # forward-only microbenchmark at the same shape instead.
        rows = [
            ("iteration wall", self.iter_wall),
            ("  fwd/bwd call wall", self.fbf_wall),
            ("  fwd/bwd device span", gpu),
        ]
        lines = [
            f"[MXFP6_EVAL_ITER_TIMING] {n} iterations",
            f"{'':22}{'median':>9}{'p10':>9}{'p90':>9}{'min':>9}{'max':>9}{'total s':>10}",
        ]
        for label, xs in rows:
            med, p10, p90, lo, hi, tot = stats(xs)
            lines.append(
                f"{label:22}{med:>9.2f}{p10:>9.2f}{p90:>9.2f}{lo:>9.2f}{hi:>9.2f}{tot/1000.0:>10.2f}"
            )
        med_iter = statistics.median(self.iter_wall)
        med_fbf = statistics.median(self.fbf_wall)
        med_gpu = statistics.median(gpu)
        lines.append(
            f"{'  host outside span':22}{med_fbf - med_gpu:>9.2f}"
            f"   ({100*(med_fbf-med_gpu)/med_iter:.1f}% of iteration)"
        )
        lines.append(
            f"{'  loop overhead':22}{med_iter - med_fbf:>9.2f}"
            f"   ({100*(med_iter-med_fbf)/med_iter:.1f}% of iteration)"
        )
        # Optional: anything above the model's standalone forward cost at this shape is
        # not compute. No default, because the figure is specific to a shape and a
        # machine and does not scale across microbatch sizes -- measure it with a
        # forward-only microbenchmark and pass it in. Two runs of the same such harness
        # on the same tree have been seen to disagree by several percent while each was
        # internally tight to a fraction of a percent, so treat a small residual as
        # noise rather than signal.
        floor_env = os.environ.get("MXFP6_EVAL_COMPUTE_FLOOR_MS")
        if floor_env:
            floor = float(floor_env)
            lines.append(
                f"{'  non-compute':22}{med_iter - floor:>9.2f}"
                f"   ({100*(med_iter-floor)/med_iter:.1f}% of iteration, against a "
                f"{floor:.2f} ms compute floor)"
            )
        # The scored eval time is a total, not a median, so a single slow iteration is
        # worth as much as a far smaller per-iteration regression. Name them.
        outliers = [(i + 1, x) for i, x in enumerate(self.iter_wall) if x > 2 * med_iter]
        excess = sum(x - med_iter for _, x in outliers) / 1000.0
        lines.append("  first 3 iterations:  " + ", ".join(f"{x:.1f}" for x in self.iter_wall[:3]) + " ms")
        if self.losses:
            path = os.environ.get("MXFP6_EVAL_LOSS_DUMP", "")
            if path:
                with open(path, "w") as fh:
                    fh.write("\n".join(f"{i}\t{x!r}" for i, x in enumerate(self.losses)) + "\n")
                lines.append(f"  per-iteration losses dumped to {path}")
            lines.append("  first 4 losses:      " + ", ".join(f"{x:.9f}" for x in self.losses[:4]))

        # Separates "the first eval iteration waits on the val loader" from "the
        # first eval iteration is slow for some other reason". The probe above
        # cannot: it brackets the whole forward_backward call, and the fetch is
        # inside it, which is why it reports almost nothing outside the span.
        try:
            from primus.backends.megatron.training.diffusion.forward_step import (
                take_batch_fetch_timings,
            )

            fetches = take_batch_fetch_timings()
        except Exception:
            fetches = []
        if fetches:
            med_f = statistics.median(fetches)
            lines.append(
                f"  batch fetch (n={len(fetches)}): median {med_f:.2f} ms, "
                f"max {max(fetches):.1f} ms, first {fetches[0]:.1f} ms, "
                f"total {sum(fetches)/1000:.2f} s"
            )
            slow = [(i + 1, x) for i, x in enumerate(fetches) if x > 2 * med_f]
            lines.append(
                "  slow fetches:        "
                + (", ".join(f"#{i} {x:.0f}ms" for i, x in slow[:16]) if slow else "none")
            )
        lines.append(
            "  above 2x median:     "
            + (", ".join(f"#{i} {x/1000:.2f}s" for i, x in outliers) if outliers else "none")
            + (
                f"  -> {excess:.2f} s of excess, {100*excess/(sum(self.iter_wall)/1000):.1f}% of the eval"
                if outliers
                else ""
            )
        )
        log_rank_0("\n".join(lines))


def _report_eval(args, message):
    """Report evaluation progress or a result on rank 0, at debug under MLPerf mode.

    MLPerf mode keeps the run's output to a single voice: it stubs out
    Megatron's print_rank_last and the tensorboard and wandb writers, and
    reports the loss itself, both as an mllog eval_accuracy event and as its own
    [MLPerf] line. Repeating any of that alongside a submission log is noise, so
    under MLPerf mode it drops to debug and stays in debug.log for a post-mortem.

    Only progress and results that confirm things went as configured come
    through here. A coverage shortfall raises, and the one mismatch that does
    not raise still reports at info, so nothing this quietens can turn a bad run
    into a silent one.
    """
    if getattr(args, "mlperf_mode", False):
        debug_rank_0(message)
    else:
        log_rank_0(message)


def _record_consumed_valid_samples(args, observed_samples, eval_iters, eval_batch_size):
    """Account for the samples the evaluation actually read, and say so.

    The previous behaviour added ``eval_batch_size`` per iteration regardless
    of how wide the batches really were, so a short final batch on any worker
    was counted as a full one and the logged sample count could exceed the
    count evaluated. Reporting the intended number is worse than reporting
    nothing, because it is indistinguishable from a correct run.
    """
    expected = eval_iters * eval_batch_size

    if observed_samples is None:
        # No loss-bearing stage on this rank, or a metric shape that carries
        # microbatch counts rather than sample counts. Fall back to the
        # configured budget rather than skipping the accounting entirely.
        args.consumed_valid_samples += expected
        return

    args.consumed_valid_samples += observed_samples

    if observed_samples == expected:
        _report_eval(
            args,
            f"[eval] covered {observed_samples} samples "
            f"({eval_iters} iterations x {eval_batch_size} per iteration)",
        )
        return

    # Context parallelism duplicates the per-sample loss across CP ranks, which
    # inflates the reduced denominator; only assert when it cannot.
    cp_size = parallel_state.get_context_parallel_world_size()
    detail = (
        f"Evaluation read {observed_samples} samples but the configuration "
        f"implies {expected} ({eval_iters} iterations x {eval_batch_size}). "
        f"Difference: {expected - observed_samples}."
    )
    if cp_size == 1:
        raise RuntimeError(
            f"{detail}\nThis is the silent under-read described in eval_budget: "
            f"Energon workers whose batch quota is short leave the tail of their "
            f"slice unread. Check val_num_workers against eval_samples."
        )
    log_rank_0(f"[eval] {detail} (context_parallel_size={cp_size}, not asserting)")


def _reduction_device(numerators):
    """Where to build the packed buffer: wherever the accumulators already live.

    Packing onto the accumulators' own device avoids a transfer and keeps the
    buffer on the device the process group can reduce over.
    """
    for value in numerators.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cuda")


def reduce_eval_losses(numerators, denominators, dp_group):
    """Reduce every accumulated (numerator, denominator) pair across data parallelism.

    One reduction, one group, one host sync.

    Every numerator and denominator across every key is packed into a single
    fp64 buffer and reduced once. The reduction must produce a value identical
    on every rank: the target-eval-loss early stop compares it against a
    threshold, and if ranks disagree near the target one can exit train() alone
    while the others keep training, desyncing collectives (grad-norm
    all-reduce) into an NCCL hang.

    The previous implementation reduced num/den over DP-with-CP and then
    reduced the result again over DP-without-CP. That left both sides
    multiplied by data_parallel_size, so the ratio was right but the
    denominator could not be read as a sample count.

    Returns:
        ``(total_loss_dict, observed_samples)``, where ``observed_samples`` is
        the globally reduced ``VAL_LOSS_KEY`` denominator -- a true sample
        count -- or None when no such key was reported.
    """
    keys = sorted(numerators.keys())
    packed = torch.tensor(
        [float(value) for key in keys for value in (numerators[key], denominators[key])],
        dtype=torch.float64,
        device=_reduction_device(numerators),
    )
    torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM, group=dp_group)

    # Single host sync for every key at once.
    reduced = packed.tolist()
    total_loss_dict = {}
    observed_samples = None
    for index, key in enumerate(keys):
        numerator, denominator = reduced[2 * index], reduced[2 * index + 1]
        # Keep the result as a 0-dim tensor: downstream Megatron code
        # (evaluate_and_print_results) and mlperf logging call .item().
        if denominator > 0:
            total_loss_dict[key] = torch.tensor(
                numerator / denominator, dtype=torch.float32, device=packed.device
            )
        else:
            total_loss_dict[key] = torch.zeros((), dtype=torch.float32, device=packed.device)
        if key == VAL_LOSS_KEY:
            observed_samples = int(round(denominator))

    return total_loss_dict, observed_samples


def primus_evaluate(
    forward_step_func,
    data_iterator,
    model,
    process_non_loss_data_func,
    config,
    verbose=True,
    non_loss_data_func=None,
    eval_iters=None,
):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    # Before any forward step, so consumers deriving per-microbatch state can
    # tell this evaluation apart from the previous one even at the same step.
    begin_eval_session()

    timers("evaluate", log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        from megatron.legacy.model.vision.knn_monitor import compute_feature_bank

        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    # Disable result validation during evaluation
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    # Accumulate numerator and denominator separately across all eval iterations
    total_loss_numerators = {}
    total_loss_denominators = {}

    # Validation batch size, independent of the training one where the recipe
    # says so: the training global batch has a floor of dp_size x
    # micro_batch_size and cannot always divide a validation split, and a split
    # it cannot divide would otherwise be evaluated only in part.
    eval_batch_size = get_eval_global_batch_size(args)
    eval_micro_batch_size = get_eval_micro_batch_size(args)
    # Shared with the dataloader provider so the loop and the dataset it reads
    # from cannot disagree about how large an evaluation is.
    eval_num_microbatches = get_eval_num_microbatches(args)
    forward_backward_func = get_forward_backward_func()
    if args.enable_cuda_graph and args.cuda_graph_scope == "full_iteration":
        forward_backward_func = FullCudaGraphWrapper(
            forward_backward_func, cuda_graph_warmup_steps=args.cuda_graph_warmup_steps
        )

    if eval_iters is None:
        eval_iters = args.eval_iters

    eval_prof = _make_eval_profiler()
    if eval_prof is not None:
        eval_prof.start()

    iter_timer = _EvalIterTimer.maybe_create()

    # Keep the real iterator: the wrapper is per-evaluation, and the next
    # evaluation is handed the same underlying object this one was.
    val_iterator = data_iterator
    data_iterator, prefetch_join_ms = _take_val_prefetch(data_iterator)
    if prefetch_join_ms is not None:
        debug_rank_0(f"[MXFP6_VAL_PREFETCH] first batch ready, waited {prefetch_join_ms:.1f} ms")

    with torch.no_grad():
        iteration = 0
        if verbose:
            _report_eval(args, f"Evaluating on {eval_iters * eval_batch_size} samples")
        while iteration < eval_iters:
            iteration += 1
            if iter_timer is not None:
                iter_timer.iteration_start()
            if verbose:
                # One line per iteration, so 58 per evaluation at the MLPerf
                # shape and 580 over a ten-evaluation run. Progress is worth
                # watching on an ordinary run and worth nothing in a submission
                # log, which reports its own eval_start and eval_stop.
                _report_eval(args, f"Evaluating iter {iteration}/{eval_iters}")

            # Don't care about timing during evaluation
            config.timers = None
            ft_integration.on_eval_step_start()
            if iter_timer is not None:
                iter_timer.fbf_start()
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=eval_micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
            )
            if iter_timer is not None:
                iter_timer.fbf_stop()
            ft_integration.on_eval_step_end()
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if is_pipeline_stage_containing_loss():
                # Accumulate loss across microbatches for this iteration.
                for key in loss_dicts[0].keys():
                    numerator = 0
                    denominator = 0
                    for x in loss_dicts:
                        val = x[key]
                        # there is one dict per microbatch. in new reporting, we average
                        # over the total number of tokens across the global batch.
                        if isinstance(val, tuple) or isinstance(val, list):
                            numerator += val[0]
                            denominator += val[1]
                        elif isinstance(val, torch.Tensor) and val.numel() == 2:
                            # [loss, num_tokens] from pretrain_gpt loss_func (Megatron default)
                            numerator += val[0]
                            denominator += val[1]
                        else:
                            # legacy behavior. we average over the number of microbatches,
                            # and so the denominator is 1.
                            numerator += val
                            denominator += 1
                    # diffusion_trainer.py:309 reports Flux under "loss"; the GPT paths
                    # use "lm loss". Take whichever this model produces.
                    if iter_timer is not None and key in ("loss", "lm loss"):
                        iter_timer.record_loss(numerator, denominator)

                    # Accumulate across all eval iterations
                    if key not in total_loss_numerators:
                        total_loss_numerators[key] = 0
                        total_loss_denominators[key] = 0
                    total_loss_numerators[key] += numerator
                    total_loss_denominators[key] += denominator

            if args.exit_duration_in_mins:
                train_time = (time.time() - get_train_start_time()) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins], dtype=torch.int, device="cuda"
                )
                torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    rerun_state_machine.set_mode(rerun_mode)
                    log_rank_0("Exiting during evaluation, timelimit reached")
                    return None, None, True

            if eval_prof is not None:
                eval_prof.step()
            if iter_timer is not None:
                iter_timer.iteration_stop()

        if eval_prof is not None:
            eval_prof.stop()

        # Before the loss reduction below, so the fetch overlaps that too.
        _start_val_prefetch(val_iterator)

        if iter_timer is not None:
            iter_timer.report()

        total_loss_dict = {}
        observed_samples = None
        if is_pipeline_stage_containing_loss():
            from megatron.core import mpu

            total_loss_dict, observed_samples = reduce_eval_losses(
                total_loss_numerators,
                total_loss_denominators,
                mpu.get_data_parallel_group(with_context_parallel=True),
            )

        _record_consumed_valid_samples(args, observed_samples, eval_iters, eval_batch_size)

        # Megatron reports the losses with print_rank_last, and torchrun does
        # not forward the last rank's stdout, so on a multi-GPU job the number
        # the evaluation exists to produce never reaches the console. Repeat it
        # through the Primus logger, which does.
        if total_loss_dict:
            summary = ", ".join(f"{key}={value.item():.6f}" for key, value in sorted(total_loss_dict.items()))
            _report_eval(args, f"[eval] {summary}")

        collected_non_loss_data = None
        if non_loss_data_func is not None:
            collected_non_loss_data = non_loss_data_func(model)
        elif process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True,
            )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    timers("evaluate").stop()
    timers.log(["evaluate"])

    rerun_state_machine.set_mode(rerun_mode)

    return total_loss_dict, collected_non_loss_data, False
