import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity

from primus.tools.preflight.global_vars import (
    ITERATION,
    LOCAL_WORLD_SIZE,
    RANK,
    WARMUP,
    WORLD_SIZE,
)
from primus.tools.preflight.utility import log

# profile parameters
_ENABLE_PROFILE = False
_PROFILE_STEPS = min(ITERATION, 10)

_NODE_RANK = RANK // LOCAL_WORLD_SIZE
_GLOBAL_PIPELINE_GROUP = None


def init_pipeline_group():
    global _GLOBAL_PIPELINE_GROUP
    assert WORLD_SIZE % LOCAL_WORLD_SIZE == 0
    num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    if num_nodes <= 1:
        return False

    pipeline_ranks = [[(i + LOCAL_WORLD_SIZE * j) for j in range(num_nodes)] for i in range(LOCAL_WORLD_SIZE)]

    for ranks in pipeline_ranks:
        pp_group = torch.distributed.new_group(ranks, backend="NCCL")
        if RANK in ranks:
            assert _GLOBAL_PIPELINE_GROUP is None
            _GLOBAL_PIPELINE_GROUP = pp_group
    return True


def send_recv_once(x, y, use_batched_p2p=True):
    pp_group = _GLOBAL_PIPELINE_GROUP

    if use_batched_p2p:
        ops = []
        op_send = dist.P2POp(
            dist.isend, tensor=x, peer=(RANK + LOCAL_WORLD_SIZE) % WORLD_SIZE, group=pp_group
        )
        ops.append(op_send)
        op_recv = dist.P2POp(
            dist.irecv, tensor=y, peer=(RANK - LOCAL_WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE, group=pp_group
        )
        ops.append(op_recv)
        reqs = dist.batch_isend_irecv(ops)
        return reqs

    # non-batched p2p
    if _NODE_RANK % 2 == 0:
        req_send = dist.isend(tensor=x, dst=(RANK + LOCAL_WORLD_SIZE) % WORLD_SIZE, group=pp_group)
        req_recv = dist.irecv(
            tensor=y, src=(RANK - LOCAL_WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE, group=pp_group
        )
    else:
        req_recv = dist.irecv(
            tensor=y, src=(RANK - LOCAL_WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE, group=pp_group
        )
        req_send = dist.isend(tensor=x, dst=(RANK + LOCAL_WORLD_SIZE) % WORLD_SIZE, group=pp_group)
    return [req_send, req_recv]


def run_ring_p2p(num_bytes: int):
    # tensor send to next node
    x = torch.zeros((num_bytes,), dtype=torch.uint8, device="cuda")

    # tenosr recv from previous node
    y = torch.empty((num_bytes,), dtype=torch.uint8, device="cuda")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for it in range(WARMUP):
        reqs = send_recv_once(x, y)

    if WARMUP > 0:
        # TODO(limou)
        # check, does this create a cuda event
        # making default stream waiting for NCCL stream ?
        for req in reqs:
            req.wait()

    if _ENABLE_PROFILE and RANK == 0:
        assert ITERATION >= _PROFILE_STEPS
        prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=ITERATION - _PROFILE_STEPS,
                warmup=0,
                active=_PROFILE_STEPS,
            ),
        )
        prof.start()

    start_event.record()
    for it in range(ITERATION):
        reqs = send_recv_once(x, y)
        if _ENABLE_PROFILE and RANK == 0:
            prof.step()

    for req in reqs:
        req.wait()

    end_event.record()
    end_event.synchronize()
    avg_time_elapsed = start_event.elapsed_time(end_event) / ITERATION
    if _ENABLE_PROFILE and RANK == 0:
        prof.stop()
        prof.export_chrome_trace(f"inter-node_ring_p2p_trace_{num_bytes}.json")

    return avg_time_elapsed


def write_markdown(args, sizes_in_mb, time_statistics):
    if RANK != 0:
        return
    num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    keys = [f"{size}MB" for size in sizes_in_mb]
    formatted_keys = [f"{key:<6}" for key in keys]
    with open(args.markdown_file, "a", encoding="utf-8") as f:
        # latency
        log(f"=======InterNodeRingP2P latency - ringsize={num_nodes} - (ms)=======")
        f.write(f"=======InterNodeRingP2P latency - ringsize={num_nodes} - (ms)=======\n")
        log(f"{'ring':<{4}} {' '.join(formatted_keys)}")
        f.write(f"| ring | {' | '.join(keys)}|\n")
        f.write(f"|----------{'|----------' * len(keys)}|\n")
        for r_idx in range(LOCAL_WORLD_SIZE):
            times_elapsed = [time_statistics[s_idx][r_idx] for s_idx in range(len(time_statistics))]
            formatted_values = [f"{time_elapsed:<6.3f}" for time_elapsed in times_elapsed]

            log(f"{r_idx:<{4}} {' '.join(formatted_values)}")
            f.write(f"| {r_idx} | {' | '.join(formatted_values)}|\n")
        f.write("\n")

        # bandwidth
        log(f"=======InterNodeRingP2P bidirectional bandwidth - ringsize={num_nodes} - (GB/s)=======")
        f.write(f"=======InterNodeRingP2P bidirectional bandwidth - ringsize={num_nodes} - (GB/s)=======\n")
        log(f"{'ring':<{4}} {' '.join(formatted_keys)}")
        f.write(f"| ring | {' | '.join(keys)}|\n")
        f.write(f"|----------{'|----------' * len(keys)}|\n")
        for r_idx in range(LOCAL_WORLD_SIZE):
            times_elapsed = [time_statistics[s_idx][r_idx] for s_idx in range(len(time_statistics))]
            ring_transmission_sizes = [size * num_nodes for size in sizes_in_mb]
            formatted_values = [
                "{:<6.2f}".format(ring_transmission_sizes[s_idx] / 1024.0 * 1000 / times_elapsed[s_idx])
                for s_idx in range(len(ring_transmission_sizes))
            ]

            log(f"{r_idx:<{4}} {' '.join(formatted_values)}")
            f.write(f"| {r_idx} | {' | '.join(formatted_values)}|\n")
        f.write("\n")


"""
This test simulates P2P transmission between nodes in pipeline parallelism
  without any intra-node transmission.

The test connects GPUs with the same local_rank from each node in a ring topology.

for example (num_nodes=4, num_gpus_per_node=8, ringsize=4):

ring0 : gpu0-->gpu8-->gpu16-->gpu24-->gpu0
ring1 : gpu1-->gpu9-->gpu17-->gpu25-->gpu1
...
ring7 : gpu7-->gpu15-->gpu23-->gpu31-->gpu7

Each ring operates independently.
Every GPU sends data to the next GPU in the ring
  while simultaneously receiving data from the previous GPU.
"""


def run_inter_node_ring_p2p(args):
    assert WORLD_SIZE % LOCAL_WORLD_SIZE == 0
    num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE

    if num_nodes <= 1:
        log(f"Skip inter node comm benchmark, {num_nodes=}")
        return

    status = init_pipeline_group()
    if not status:
        log("Skip inter node ring p2p benchmark")
        return

    SIZES_IN_MB_TO_BENCH = [10, 20, 40, 80, 160]
    time_statistics = []
    for size_in_mb in SIZES_IN_MB_TO_BENCH:
        avg_time_elapsed = run_ring_p2p(size_in_mb * (2**20))

        all_latency_results = [-1.0 for _ in range(WORLD_SIZE)]
        dist.gather_object(avg_time_elapsed, all_latency_results if RANK == 0 else None, dst=0)
        time_statistics.append(all_latency_results[:LOCAL_WORLD_SIZE])

    write_markdown(args, SIZES_IN_MB_TO_BENCH, time_statistics)

    # TODO (limou)
    # support plot
