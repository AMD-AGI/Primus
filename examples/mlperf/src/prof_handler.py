import torch
import os
import pathlib
from dataclasses import dataclass
from torch.profiler import ProfilerActivity
import pandas as pd


TORCHPROF_OUTPUT_DIR = os.getenv("TORCHPROF_OUTPUT_DIR", "/results/artifacts")
TORCHPROF_OUTPUT = os.getenv("TORCHPROF_OUTPUT", "csv_handler")
TORCHPROF_VERBOSE = os.getenv("TORCHPROF_VERBOSE", 1)
TORCHPROF_DEVICES = os.getenv("TORCHPROF_DEVICES", "GPU")
TORCHPROF_MAXROWS = os.getenv("TORCHPROF_MAXROWS", 100)
TORCHPROF_PROFILE_MEMORY = bool(os.getenv("TORCHPROF_PROFILE_MEMORY", 1))
TORCHPROF_WITH_STACK = bool(os.getenv("TORCHPROF_WITH_STACK", 0))
TORCHPROF_RECORD_SHAPES = bool(os.getenv("TORCHPROF_RECORD_SHAPES", 1))
TORCHPROF_WITH_FLOPS = bool(os.getenv("TORCHPROF_WITH_FLOPS", 1))
PROF_WARMUP_STEPS = int(os.getenv("PROF_WARMUP_STEPS", 3))
PROF_ACTIVE_STEPS = int(os.getenv("PROF_ACTIVE_STEPS", 2))
PROF_REPITIONS = int(os.getenv("PROF_REPITIONS", 1))

def get_devices() -> list:
    devices = TORCHPROF_DEVICES.split(",")
    devices = [x.lower().strip() for x in devices]
    devices_set = set()
    for device in devices:
        if device.lower() not in ['cpu', 'gpu']:
            raise ValueError(f"Invalid Device :{device}")
        if device.lower() == "gpu":
            devices_set.add(torch.profiler.ProfilerActivity.CUDA)
        elif device.lower() == "cpu":
            devices_set.add(torch.profiler.ProfilerActivity.CPU)
        
    devices_list = list(devices_set)

    return devices_list

@dataclass
class TorchProfConfig:
    skip_first = 1
    wait = 0
    warmup = PROF_WARMUP_STEPS
    active = PROF_ACTIVE_STEPS
    repeat = PROF_REPITIONS

TOTAL_WARMUP_STEPS = TorchProfConfig.skip_first + \
    TorchProfConfig.wait + \
    TorchProfConfig.warmup

TOTAL_ACTIVE_STEPS = TorchProfConfig.active    


def kernel_table_from_prof(prof, top_k=200):
    rows = []
    for evt in prof.events():
        # evt has fields like name, device_type, cuda_time_total, cpu_time_total, etc.
        # Kernel events show up as CUDA device events in the trace.
        # We keep entries that have some CUDA time.
        cuda_us = getattr(evt, "cuda_time_total", 0) or 0
        if cuda_us <= 0:
            continue

        name = getattr(evt, "name", "<?>")

        # Heuristic: keep kernel-ish entries.
        # Depending on PyTorch version, true kernels often include patterns like:
        #  - "void " (C++ kernel signature)
        #  - "ampere_" / "sm" / "cutlass" / "cudnn" / "gemm" / "flash" etc.
        # You can loosen/tighten this filter.
        #kernelish = (
        #    name.startswith("void ")
        #    or "kernel" in name.lower()
        #    or "cudnn" in name.lower()
        #    or "cutlass" in name.lower()
        #    or "gemm" in name.lower()
        #)
        #if not kernelish:
        #    continue

        rows.append((name, cuda_us))

    if not rows:
        return pd.DataFrame(columns=["kernel", "calls", "total_us", "avg_us", "pct_total"])

    df = pd.DataFrame(rows, columns=["kernel", "cuda_us"])

    agg = (df.groupby("kernel", as_index=False)
             .agg(calls=("cuda_us", "size"), total_us=("cuda_us", "sum")))
    agg["avg_us"] = agg["total_us"] / agg["calls"]
    total = agg["total_us"].sum()
    agg["pct_total"] = 100.0 * agg["total_us"] / total
    agg = agg.sort_values("total_us", ascending=False).head(top_k)

    # Pretty formatting helpers
    agg["total_ms"] = agg["total_us"] / 1000.0
    agg["avg_ms"] = agg["avg_us"] / 1000.0

    return agg[["kernel", "calls", "avg_ms", "total_ms", "pct_total"]]


def trace_handler(prof):
    save_path = f"{TORCHPROF_OUTPUT_DIR}/key_avg_{prof.step_num}_{torch.distributed.get_rank()}.txt"
    print(f"Saving torchprof results at: {save_path}")
    with open(save_path, 'w') as f:
        #output = prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=TORCHPROF_MAXROWS)
        output = prof.key_averages().table(row_limit=TORCHPROF_MAXROWS)
        f.write(output)
        if TORCHPROF_VERBOSE:
            print(output)

    save_path_pandas = f"{TORCHPROF_OUTPUT_DIR}/pandas_key_avg_{prof.step_num}_{torch.distributed.get_rank()}.txt"
    with open(save_path_pandas, 'w') as f:
        kt = kernel_table_from_prof(prof, top_k=200)
        kt.to_csv(save_path_pandas, index=False)
        


    prof.export_chrome_trace(f"{TORCHPROF_OUTPUT_DIR}/trace_{prof.step_num}_{torch.distributed.get_rank()}.json")


def _get_torchprof():
    if TORCHPROF_OUTPUT == "csv_handler":
        output_handler = trace_handler
    elif TORCHPROF_OUTPUT == "tensorboard":
        output_handler = torch.profiler.tensorboard_trace_handler(TORCHPROF_OUTPUT_DIR)
    else:
        raise ValueError("Invalid Output Handler for TorchProf.")
    pathlib.Path(TORCHPROF_OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], on_trace_ready=output_handler)   

    #prof = torch.profiler.profile(
    #    activities=get_devices(), 
    #    schedule=torch.profiler.schedule(
    #        skip_first=TorchProfConfig.skip_first,
    #        wait=TorchProfConfig.wait,
    #        warmup=TorchProfConfig.warmup,
    #        active=TorchProfConfig.active, 
    #        repeat=TorchProfConfig.repeat
    #   ),
        # called each time the trace is ready at the end of each cycle.  
    #    on_trace_ready=output_handler,
    #    profile_memory=TORCHPROF_PROFILE_MEMORY, # adds extra overhead if True !!
    #    with_flops=TORCHPROF_WITH_FLOPS,
    #    with_stack=TORCHPROF_WITH_STACK, # adds extra overhead if True !! 
    #    record_shapes=TORCHPROF_RECORD_SHAPES
    #)

    return prof

def get_profiler():
    if os.getenv("PROFILER", '') == 'torchprof':
        return _get_torchprof()
    return None

def _get_rpd():
    from rpdTracerControl import rpdTracerControl
    rpdTracerControl.setFilename(name=f"trace.rpd", append=True)
    prof = rpdTracerControl()
    print(f"RPD profiler initialized")
    return prof

def get_profiler_rpd():
    if os.getenv("PROFILER", '') == 'rpd':
        return _get_rpd()
    return None
