###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pipeline-parallel visualization helpers.

Records per-iteration forward/backward/weight-grad CUDA events so the offline
``dump_pp_data`` step can emit a JSON timeline consumed by the PP visualizer.
These wrappers are installed by the ``megatron.pp.dump_pp_data`` patch (and by
the Primus-Pipe / ZeroBubble handlers) when ``--dump_pp_data`` is enabled.
"""

import json
import os

import torch
from megatron.core import parallel_state

_GLOBAL_PP_VIS_EVENTS = []
_GLOBAL_PP_VIS_EVENTS_PER_ITER = None


def schedule_wrapper(func):
    def wrapper(*args, **kwargs):
        global _GLOBAL_PP_VIS_EVENTS_PER_ITER
        _GLOBAL_PP_VIS_EVENTS_PER_ITER = {
            "start": None,
            "end": None,
            "memory": None,
            "fwd_start": [],
            "fwd_end": [],
            "fwd_minibatch": [],
            "fwd_chunk": [],
            "bwd_start": [],
            "bwd_end": [],
            "bwd_minibatch": [],
            "bwd_chunk": [],
            "wgrad_start": [],
            "wgrad_end": [],
            "wgrad_minibatch": [],
            "wgrad_chunk": [],
        }

        _GLOBAL_PP_VIS_EVENTS_PER_ITER["start"] = torch.cuda.Event(enable_timing=True)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["start"].record()
        res = func(*args, **kwargs)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["end"] = torch.cuda.Event(enable_timing=True)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["end"].record()

        _GLOBAL_PP_VIS_EVENTS_PER_ITER["memory"] = torch.cuda.max_memory_reserved() / 1024**3

        global _GLOBAL_PP_VIS_EVENTS
        _GLOBAL_PP_VIS_EVENTS.append(_GLOBAL_PP_VIS_EVENTS_PER_ITER)

        return res

    return wrapper


def fwd_bwd_wrapper(func, mode, minibatch=None, chunk=None):
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        res = func(*args, **kwargs)
        end.record()

        global _GLOBAL_PP_VIS_EVENTS_PER_ITER
        _GLOBAL_PP_VIS_EVENTS_PER_ITER[mode + "_start"].append(start)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER[mode + "_end"].append(end)

        if minibatch is not None:
            _GLOBAL_PP_VIS_EVENTS_PER_ITER[mode + "_minibatch"].append(minibatch)
        if chunk is not None:
            _GLOBAL_PP_VIS_EVENTS_PER_ITER[mode + "_chunk"].append(chunk)
        return res

    return wrapper


def combined_fwd_bwd_wrapper(func, fwd_minibatch, fwd_chunk, bwd_minibatch, bwd_chunk):
    """Record a single combined forward+backward call as both an ``fwd`` event
    and a ``bwd`` event sharing the same ``[start, end]`` interval.

    Used by ``megatron_combined_fwd_bkwd_handler`` so that nodes collapsed into
    a combined FB group still appear in the dump_pp_data output. Without this
    the visualizer's per-rank F/B/W totals are heavily under-counted on ranks
    that hit the steady state (the F and B halves are interleaved inside
    ``combined_forward_backward_step`` and cannot be timed separately).
    """

    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        res = func(*args, **kwargs)
        end.record()

        global _GLOBAL_PP_VIS_EVENTS_PER_ITER
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["fwd_start"].append(start)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["fwd_end"].append(end)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["fwd_minibatch"].append(fwd_minibatch)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["fwd_chunk"].append(fwd_chunk)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["bwd_start"].append(start)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["bwd_end"].append(end)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["bwd_minibatch"].append(bwd_minibatch)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["bwd_chunk"].append(bwd_chunk)
        return res

    return wrapper


def set_dump_pp_data_patch():
    from megatron.core.pipeline_parallel import schedules

    schedules.forward_step = fwd_bwd_wrapper(schedules.forward_step, "fwd")
    schedules.backward_step = fwd_bwd_wrapper(schedules.backward_step, "bwd")


def dump_pp_data(args, num_mbs, pp_data_dir):
    torch.cuda.synchronize()

    global _GLOBAL_PP_VIS_EVENTS
    all_iter_data = {}
    for iter_idx, iter_events in enumerate(_GLOBAL_PP_VIS_EVENTS):
        iter_data = {
            "total": None,
            "memory": None,
            "fwd_start": [],
            "fwd_end": [],
            "fwd_minibatch": [],
            "fwd_chunk": [],
            "bwd_start": [],
            "bwd_end": [],
            "bwd_minibatch": [],
            "bwd_chunk": [],
            "wgrad_start": [],
            "wgrad_end": [],
            "wgrad_minibatch": [],
            "wgrad_chunk": [],
        }
        iter_data["total"] = iter_events["start"].elapsed_time(iter_events["end"])
        iter_data["memory"] = iter_events["memory"]

        for i in range(len(iter_events["fwd_start"])):
            for key in ["fwd_start", "fwd_end", "bwd_start", "bwd_end", "wgrad_start", "wgrad_end"]:
                if i >= len(iter_events[key]):
                    continue
                event_time = iter_events["start"].elapsed_time(iter_events[key][i])
                iter_data[key].append(event_time)
            for key in [
                "fwd_minibatch",
                "fwd_chunk",
                "bwd_minibatch",
                "bwd_chunk",
                "wgrad_minibatch",
                "wgrad_chunk",
            ]:
                if i >= len(iter_events[key]):
                    continue
                iter_data[key].append(iter_events[key][i])

        all_iter_data[iter_idx + 1] = iter_data

    rank = torch.distributed.get_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    os.makedirs(pp_data_dir, exist_ok=True)
    if dp_rank == 0:
        log_path = os.path.join(pp_data_dir, f"pp_rank_{pp_rank}.json")
        with open(log_path, "w") as f:
            json.dump(all_iter_data, f, indent=2)

    if rank == 0:
        vp_size = args.virtual_pipeline_model_parallel_size
        vp_size = 1 if vp_size is None else vp_size
        config_dict = {
            "world_size": args.world_size,
            "dp_size": args.data_parallel_size,
            "tp_size": args.tensor_model_parallel_size,
            "ep_size": args.expert_model_parallel_size,
            "pp_size": args.pipeline_model_parallel_size,
            "vp_size": vp_size,
            "num_mbs": num_mbs,
            "train_iters": args.train_iters,
        }
        log_path = os.path.join(pp_data_dir, f"config.json")
        with open(log_path, "w") as f:
            json.dump(config_dict, f, indent=2)
