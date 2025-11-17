###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
import socket
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

try:

    import torch
    import torch.distributed as dist

    def init_distributed():
        """Init only when launched with >1 processes via torchrun."""
        if dist.is_initialized():
            return
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                device_id=torch.device(f"cuda:{local_rank}"),
                init_method="env://",
                timeout=timedelta(minutes=5),
            )

    def finalize_distributed():
        """Destroy process group if initialized."""
        if dist.is_initialized():
            try:
                dist.barrier()  # optional: ensure all ranks reach this point
            except Exception:
                pass  # ignore barrier errors on exit
            dist.destroy_process_group()

    def gather_all_results(obj):
        """
        Gather arbitrary Python object(s) from all ranks.
        - If `obj` is a list, returns a single flattened list (concatenate).
        - If `obj` is a dict (your per-rank summary), returns a list[dict].
        - If `obj` is None, it's skipped.
        """
        # Single-process fallback
        if not dist.is_initialized():
            if obj is None:
                return []
            # keep list as-is; wrap non-list as list
            return obj if isinstance(obj, list) else [obj]

        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, obj)

        # Filter out None from any rank that returned nothing
        gathered = [g for g in gathered if g is not None]

        # If ranks each returned a list, concatenate
        if len(gathered) > 0 and isinstance(gathered[0], list):
            out = []
            for g in gathered:
                out.extend(g)
            return out

        # Otherwise, assume ranks returned single objects (e.g., dict summaries)
        return gathered

    def is_rank_0() -> bool:
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized, cannot determine rank.")
        return dist.get_rank() == 0

    def get_local_rank() -> int:
        """Return local rank for this node (torchrun/Slurm/fallback)."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() % max(1, torch.cuda.device_count())
        if "LOCAL_RANK" in os.environ:  # torchrun
            return int(os.environ["LOCAL_RANK"])
        return 0

    def get_current_device() -> torch.device:
        """Set and return the current CUDA device for this rank."""
        lr = get_local_rank()
        torch.cuda.set_device(lr)
        return torch.device(f"cuda:{lr}")

    # @torch.no_grad()
    # def allreduce_once(tensor):
    #     dist.all_reduce(tensor)

    def get_rank_world() -> Tuple[int, int]:
        """Best-effort rank/world detection (dist first, then env)."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        # Fallback to environment variables
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world

    def gather_times(times_local: List[float]) -> List[List[float]]:
        """
        Gather per-rank lists to rank0 only (so we don't perturb measured section).
        """
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world == 1:
            return [times_local]
        if rank == 0:
            gather_list: List[List[float]] = [None for _ in range(world)]
            dist.gather_object(times_local, object_gather_list=gather_list, dst=0)
            return gather_list
        else:
            dist.gather_object(times_local, dst=0)
            return []

    def gather_hostnames() -> List[str]:
        """
        Gather per-rank hostnames to rank0.
        """
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        name = socket.gethostname()
        if world == 1:
            return [name]
        if rank == 0:
            lst: List[str] = [None for _ in range(world)]  # type: ignore
            dist.gather_object(name, lst, dst=0)  # positional arg for compatibility
            return lst
        else:
            dist.gather_object(name, None, dst=0)
            return []

    def gather_records(record: Dict[str, Any], dst: int = 0) -> Optional[List[Dict[str, Any]]]:
        """
        Gather per-rank dict records to dst (root) rank only.
        - Automatically injects 'host', 'rank', and 'world'.
        - Returns:
            * On root (rank==dst): List[Dict[str, Any]] of all records.
            * On non-root: None  (change to [] if you prefer old behavior).
        - Works even if torch.distributed is not initialized (single-process).
        """
        rank, world = get_rank_world()

        # Inject host/rank/world into a shallow copy to avoid mutating caller's dict
        local = dict(record)
        local["host"] = get_hostname()
        local["rank"] = rank
        local["world"] = world

        if world == 1 or not (dist.is_available() and dist.is_initialized()):
            # Single process: just return a singleton list on "root"
            return [local]

        if rank == dst:
            # Root gathers a list of objects from all ranks
            gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world)]
            dist.gather_object(local, object_gather_list=gathered, dst=dst)
            # Type narrowing: all entries must be dicts here
            return [g for g in gathered if g is not None]
        else:
            # Non-root sends and returns None to avoid extra work/printing
            dist.gather_object(local, dst=dst)
            return None

    def pick_dtype(name: str):
        name = name.lower()
        if name in ("bf16", "bfloat16"):
            return torch.bfloat16
        if name in ("fp16", "half", "float16"):
            return torch.float16
        if name in ("fp32", "float", "float32"):
            return torch.float32
        raise ValueError(f"Unsupported dtype: {name}")

except ImportError:
    print("[WARNING] dist tools depend on torch, which does not exist in current environment!")


def parse_bytes(s: str) -> int:
    _SUFFIX = {
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
    }
    s = s.strip().upper()
    if s[-1] in _SUFFIX:
        return int(float(s[:-1]) * _SUFFIX[s[-1]])
    return int(s)  # plain integer


def get_hostname() -> str:
    return socket.gethostname()


def round_up_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def derive_path(base: str, suffix: str, default_ext: str = ".csv") -> str:
    """
    Derive a sibling file path by inserting a suffix before extension.
    If base is empty or stdout ('-'), return '' (caller can decide).
    """
    if not base or base == "-":
        return ""
    root, ext = os.path.splitext(base)
    if ext.lower() in (".md", ".markdown", ".csv", ".tsv", ".jsonl", ".gz"):
        # handle double ext like .csv.gz
        if ext.lower() == ".gz":
            root2, ext2 = os.path.splitext(root)
            return f"{root2}{suffix}{ext2}.gz"
        return f"{root}{suffix}{ext}"
    # unknown -> attach default
    return f"{base}{suffix}{default_ext}"


def parse_sizes_list(s: str) -> set[int]:
    """
    Parse a comma-separated size list like '1M,16M,4096' into bytes.
    Reuse your parse_bytes() helper.
    """
    if not s:
        return set()
    return {parse_bytes(tok.strip()) for tok in s.split(",") if tok.strip()}
