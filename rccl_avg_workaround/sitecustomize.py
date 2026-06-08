"""RCCL workaround for gfx1250: torch.distributed.all_reduce(op=AVG) HANGS on
this build for (at least) single-rank process groups, while SUM works fine
(verified by collective microbench). Megatron's MoE aux-loss metric reduction
(moe_utils.reduce_aux_losses_tracker_across_ranks) uses op=AVG and deadlocks.

Replace AVG with SUM + divide-by-world-size, which is mathematically identical
for any world size and avoids the broken AVG reduction kernel. Auto-imported in
every Python worker via sitecustomize (this dir is on PYTHONPATH).
"""
import sys


def _install():
    import torch.distributed as dist

    orig_all_reduce = dist.all_reduce
    avg_op = dist.ReduceOp.AVG

    def all_reduce_avg_safe(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        is_avg = False
        try:
            is_avg = (op == avg_op)
        except Exception:
            is_avg = str(op) == str(avg_op)
        if is_avg:
            work = orig_all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
            try:
                ws = dist.get_world_size(group)
            except Exception:
                ws = 1
            if ws and ws > 1:
                # Enqueued on the same stream after the all_reduce, so ordering holds.
                tensor.div_(ws)
            return work
        return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)

    dist.all_reduce = all_reduce_avg_safe
    print("[rccl_avg_workaround] patched torch.distributed.all_reduce (AVG -> SUM/ws)",
          file=sys.stderr, flush=True)


# Only patch the actual training worker. Importing torch here is heavy, so we
# must NOT do it for pip/offload-arch/build helper invocations (they stall and
# block setup). Gate on the training entrypoint appearing in argv.
def _is_training_worker():
    argv = " ".join(sys.argv)
    return ("primus/cli/main.py" in argv) or ("run_pretrain" in argv) or ("pretrain" in argv)


if _is_training_worker():
    try:
        _install()
    except Exception as e:  # noqa: BLE001
        print(f"[rccl_avg_workaround] install FAILED: {e}", file=sys.stderr, flush=True)
