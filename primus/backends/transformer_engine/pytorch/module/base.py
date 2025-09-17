###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from typing import Optional, Union

import torch
from megatron.core.utils import is_te_min_version
from transformer_engine.pytorch.module import base

import primus.backends.transformer_engine.transformer_engine_torch as ptex
from primus.modules.module_utils import log_rank_0, warning_rank_0


def get_cublas_workspace_size_bytes() -> None:
    """Return workspace size needed for current architecture"""
    return 0


# NOTE(zhenhuang12): call gemm by torch.mm, blas workspace is useless for now
def get_workspace() -> torch.Tensor:
    """Returns workspace for cublas."""
    if base._cublas_workspace is None:
        base._cublas_workspace = torch.Tensor()
    return base._cublas_workspace


def initialize_ub(
    shape: list,
    tp_size: int,
    use_fp8: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    ub_cfgs: Optional[dict] = None,
    bootstrap_backend: Union[str, torch.distributed.Backend] = None,
) -> None:
    r"""
        Initialize the Userbuffers communicator for overlapping tensor-parallel communications with
        GEMM compute in te.Linear, te.LayerNormLinear and te.LayerNormMLP modules.

        Parameters
        ----------
    shape : list
                shape of the communication buffer, typically set to be the same as the global shape of
                the input tensor to a te.TransformerLayer forward pass, with the sequence and batch
                dimensions collapsed together -- i.e.: `(sequence_length * batch_size, hidden_size)`
        tp_size : int
                  number of GPUs in the tensor-parallel process group
        use_fp8 : bool = False
                  allocate the communication buffer for FP8 GEMM inputs/outputs
        dtype : torch.dtype = torch.bfloat16
                non-FP8 data type of the communication buffer when `use_fp8 = False`
        ub_cfgs: dict = None
                 Configuration dictionary with the structure
                 ```
                 {
                    <gemm_name> : {
                        "method": <"ring_exchange" or "pipeline">,
                        "is_reduce_scatter": bool,
                        "num_sm": int,
                        "cga_size": int,
                        "set_sm_margin": bool,
                        "num_splits": int,
                        "aggregate": bool,
                        "atomic_gemm": bool,
                        "use_ce": bool,
                        "fp8_buf": bool,
                    }
                 }
                 ```
                 for `te.TransformerLayer` GEMM layers in `["qkv_fprop", "qkv_dgrad", "qkv_wgrad",
                 "proj_fprop", "proj_dgrad", "proj_wgrad", "fc1_fprop", "fc1_dgrad", "fc2_dgrad",
                 "fc2_fprop", "fc2_dgrad"]`.
        bootstrap_backend : str = None
                            `torch.distributed` communication backend for the all-gather, broadcast and
                            barrier collectives during Userbuffers initialization. Not all backends are
                            valid for every cluster configuration and distributed launch method even if
                            they are available in PyTorch. When left unset, the initialization prefers
                            to use the MPI backend, falling back first on Gloo and then NCCL if MPI is
                            not available. Setting `NVTE_UB_WITH_MPI=1` when building TE overrides this
                            option and always initializes Userbuffers with direct MPI calls in C++,
                            which also requires `MPI_HOME=/path/to/mpi/root` to be set at compile time.
    """
    assert base._ub_communicators is None, "UB communicators are already initialized."
    base._ub_communicators = {}

    # Bootstrapping with torch.distributed API, so check backend and construct
    # intra/inter-node process groups...
    assert torch.distributed.is_initialized(), "torch.distributed must be initialized before Userbuffers"
    if bootstrap_backend is None:
        bootstrap_backend = "nccl"
    else:
        assert bootstrap_backend in [
            "gloo",
            "mpi",
            "nccl",
        ], "Invalid torch.distributed backend for bootstrapping Userbuffers!"
        assert torch.distributed.is_backend_available(bootstrap_backend), (
            f"PyTorch must be compiled with '{bootstrap_backend}' support in order to "
            f"bootstrap Userbuffers with '{bootstrap_backend}' collectives."
        )
    world_group = torch.distributed.new_group(backend=bootstrap_backend)
    world_rank = torch.distributed.get_rank(world_group)
    world_size = torch.distributed.get_world_size(world_group)

    num_domains = world_size // tp_size
    mydomain_idx = world_rank // tp_size
    if num_domains > 1:
        ranks_per_domain_list = [[i * tp_size + t for t in range(tp_size)] for i in range(num_domains)]
        tp_domain_group, _ = torch.distributed.new_subgroups_by_enumeration(
            ranks_per_domain_list, backend=bootstrap_backend
        )
        local_rank = torch.distributed.get_rank(tp_domain_group)
        tp_domain_ranks = torch.distributed.get_process_group_ranks(tp_domain_group)

        group_name = tp_domain_group.group_name
    else:
        # TP model on single NVLink domain, no replication, no data-parallelism
        mydomain_idx = 0
        local_rank = world_rank
        tp_domain_ranks = list(range(world_size))

        group_name = world_group.group_name

    if world_rank == 0:
        log_rank_0(f"[UB] Number of TP domains: {num_domains}\n")
    if local_rank == 0:
        log_rank_0(
            f"[UB] Global ranks on TP domain {mydomain_idx}: {tp_domain_ranks}\n",
            end="",
            flush=True,
        )

    # Increase the workspace by the number of maximum concurrent streams
    base._cublas_workspace = get_workspace().repeat(base._NUM_MAX_UB_STREAMS)

    # Default buffer precision: AllGather buffers use fp8 when using fp8 recipe
    layers_all_gather_overlap = [
        "qkv_fprop",
        "qkv_dgrad",
        "proj_dgrad",
        "fc1_fprop",
        "fc1_dgrad",
        "fc2_dgrad",
    ]
    layers_reduce_scatter_overlap = ["proj_fprop", "fc2_fprop", "qkv_wgrad", "fc1_wgrad"]
    dgrad_reduce_scatter_overlap = ["qkv_dgrad", "fc1_dgrad"]
    # Default overlap methods for layers
    methods = {
        "ring_exchange": ["qkv_fprop", "fc1_fprop", "proj_dgrad", "fc2_dgrad"],
        "pipeline": ["proj_fprop", "fc2_fprop"],
        "bulk": ["qkv_dgrad", "qkv_wgrad", "fc1_dgrad", "fc1_wgrad"],
    }

    # AG-RS overlap pairs of layers forming a tensor-parallel block
    ag_rs_pairs = {"qkv_fprop": "proj_fprop", "fc1_fprop": "fc2_fprop"}
    rs_ag_pairs = {v: k for k, v in ag_rs_pairs.items()}
    global layers_atomic_ring_exchange
    layers_atomic_ring_exchange = []

    def get_method(name):
        for method, names in methods.items():
            if name in names:
                return method
        raise KeyError(f"Given layer name {name} does not exist.")

    def get_default_config(name):
        method = get_method(name)
        is_reduce_scatter = name in layers_reduce_scatter_overlap
        if is_te_min_version("2.0"):
            if base._MIN_STREAM_PRIORITY is None or base._MAX_STREAM_PRIORITY is None:
                base._MIN_STREAM_PRIORITY, base._MAX_STREAM_PRIORITY = (
                    ptex.comm_overlap.get_stream_priority_range()
                )
            default_cfg = {
                "method": method,
                "is_reduce_scatter": is_reduce_scatter,
                "num_sm": 1 if method == "ring_exchange" else 16,
                "cga_size": 1 if method == "ring_exchange" else 2,
                "set_sm_margin": not method == "ring_exchange",
                "num_splits": tp_size if method == "ring_exchange" else 4,
                "aggregate": False,
                "atomic_gemm": False,
                "use_ce": True,
                "fp8_buf": name in layers_all_gather_overlap,
                "comm_priority": base._MAX_STREAM_PRIORITY,
                "gemm_priority": base._MIN_STREAM_PRIORITY,
                "pipeline_rs_overlap_first_gemm": False,
            }
        else:
            default_cfg = {
                "method": method,
                "is_reduce_scatter": is_reduce_scatter,
                "num_sm": 1 if method == "ring_exchange" else 16,
                "cga_size": 1 if method == "ring_exchange" else 2,
                "set_sm_margin": False,
                "num_splits": 4 if method == "pipeline" else tp_size,
                "aggregate": False,
                "atomic_gemm": False,
                "use_ce": True,
                "fp8_buf": name in layers_all_gather_overlap,
            }
        return default_cfg

    def add_ub(
        name: str,
        method: str,
        is_reduce_scatter: int,
        num_sm: int = 16,
        cga_size: int = 2,
        set_sm_margin: int = 0,
        num_splits: int = 0,
        aggregate: int = 0,
        atomic_gemm: int = 0,
        use_ce: bool = True,
        fp8_buf: bool = False,
        comm_priority: int = 0,
        gemm_priority: int = 0,
        pipeline_rs_overlap_first_gemm: bool = False,
    ) -> None:

        # force method to use 'pipeline', atomic_gemm=0
        method = "pipeline"
        atomic_gemm = 0

        # Check if both AG and RS overlaps use `atomic GEMM`` + `p2p ring-exchange`.
        # Using atomic GEMM + p2p ring-exchange in only one of the pair breaks functionality.
        global layers_atomic_ring_exchange
        if atomic_gemm and method == "ring_exchange" and name in ag_rs_pairs:
            layers_atomic_ring_exchange += [name, ag_rs_pairs[name]]
        if name in rs_ag_pairs:
            assert_message = (
                f"At {name}, atomic AG-GEMM overlap with `ring_exchange` shuffles GEMM chunk "
                "outputs, and  RS-GEMM overlap un-suffle them. When one of the GEMM-AG and "
                "GEMM-RS overlaps forming a TP block (e.g., qkv_fprop and proj_fprop) uses "
                "`atomic gemm` and `ring_exhcnage`, its pair must use the same overlap config "
                "for functionality."
            )
            if name in layers_atomic_ring_exchange:
                assert atomic_gemm and method == "ring_exchange", assert_message
            else:
                if atomic_gemm and method == "ring_exchange":
                    assert rs_ag_pairs[name] in layers_atomic_ring_exchange, assert_message

        buffer_dtype = torch.uint8 if (use_fp8 and fp8_buf) else dtype
        if method == "ring_exchange":
            ub_obj = ptex.CommOverlapP2P(
                shape,  # Communication buffer shape
                buffer_dtype,  # Communication buffer data type
                group_name,
                # Tensor-parallel group size (may be different than local_size)
                tp_size,
                ptex.CommOverlapType.RS if is_reduce_scatter else ptex.CommOverlapType.AG,
                num_max_streams=base._NUM_MAX_UB_STREAMS,
                comm_cga_size=cga_size,
                num_comm_sm=num_sm,
                set_sm_margin=set_sm_margin,
                atomic_gemm=atomic_gemm,
                use_ce=use_ce,
                aggregate=aggregate,
            )
        else:
            ub_obj = ptex.CommOverlap(
                shape,  # Communication buffer shape
                buffer_dtype,  # Communication buffer data type
                group_name,
                # Tensor-parallel group size (may be different than local_size)
                tp_size,
                num_splits=num_splits,
                num_max_streams=base._NUM_MAX_UB_STREAMS,
                comm_cga_size=cga_size,
                num_comm_sm=num_sm,
                set_sm_margin=set_sm_margin,
                atomic_gemm=atomic_gemm,
            )
        base._ub_communicators[name] = ub_obj

    if ub_cfgs is not None:
        for name in dgrad_reduce_scatter_overlap:
            if name in ub_cfgs and "method" in ub_cfgs[name] and ub_cfgs[name]["method"] != "bulk":
                wgrad_name = name.replace("dgrad", "wgrad")
                assert wgrad_name not in ub_cfgs
                layers_reduce_scatter_overlap.remove(wgrad_name)
                layers_all_gather_overlap.remove(name)
                layers_reduce_scatter_overlap.append(name)
                methods["bulk"].remove(name)
                new_method = ub_cfgs[name]["method"]
                methods[new_method].append(name)

    warning_rank_0(f"tp_comm_overlap only support pipeline algo for now! Defaulting to method=pipeline.")
    warning_rank_0(f"tp_comm_overlap not support atomic_gemm! Defaulting to atomic_gemm=False.")

    for name in methods["ring_exchange"] + methods["pipeline"] + methods["bulk"]:
        ub_cfg = get_default_config(name)
        if ub_cfgs is not None and name in ub_cfgs:
            fp8_buf = (name in layers_all_gather_overlap) or (
                ub_cfgs[name].get("fp8_buf", False) and name in methods["pipeline"]
            )
            ub_cfg.update(ub_cfgs[name])
            ub_cfg["fp8_buf"] = fp8_buf
        add_ub(name, **ub_cfg)
