# SDMA and MORI AllGather for FSDP

Primus provides two custom PyTorch FSDP2 communication paths:

| `FSDP_ALL_GATHER_BACKEND` | FSDP all-gather implementation | Intra-node path | Cross-node path |
|---|---|---|---|
| `rccl_sdma` | PyTorch symmetric memory over RCCL | RCCL with SDMA | RDMA without SDMA |
| `mori` | MORI `HierAllGather` | MORI SDMA | RDMA & SDMA |

Both paths move all-gather traffic away from CU-resident RCCL kernels so that
FSDP communication can overlap with GEMM-heavy forward compute. Leave
`FSDP_ALL_GATHER_BACKEND` unset to use the framework default.

## RCCL symmetric-memory SDMA

### Enablement

Set one user-facing switch before launching Primus:

```bash
export FSDP_ALL_GATHER_BACKEND=rccl_sdma

runner/primus-cli direct -- train pretrain --config <experiment.yaml>
```

When `FSDP_ALL_GATHER_BACKEND=rccl_sdma` is set, the Primus hook
`runner/helpers/hooks/06_enable_sdma_all_gather.sh` emits the runtime
environment needed by the training container and torchrun children.

The hook sets:

```bash
NCCL_CTA_POLICY=2
NCCL_CUMEM_ENABLE=1
NCCL_LOCAL_REGISTER=0
TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
FSDP_ALL_GATHER_BACKEND=rccl_sdma
LD_PRELOAD=/tmp/libhip_attr_drain.so
```

It also rebuilds `runner/helpers/hooks/sdma/hip_attr_drain_preload.c` into
`/tmp/libhip_attr_drain.so`. The interposer drains a stale HIP TLS error from
RCCL's cuMem capability probe on ROCm builds that do not have the upstream
fix. It does not change RCCL return values.

### What the Primus patch does

The Python backend patch is gated by
`FSDP_ALL_GATHER_BACKEND=rccl_sdma`. It wires PyTorch FSDP2 modules to use
symmetric-memory collectives:

```python
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    SymmMemAllGather,
    SymmMemReduceScatter,
)

module.set_custom_all_gather(SymmMemAllGather(group))
module.set_custom_reduce_scatter(SymmMemReduceScatter(group))
```

`SymmMemAllGather` allocates all-gather buffers from PyTorch symmetric memory.
Those buffers are cuMem-backed and rendezvoused across ranks. With zero-CTA
policy enabled, RCCL can dispatch the all-gather through the ROCm copy-engine
path (`__amd_rocclr_batchMemOp.kd` / `hsa_amd_memory_async_batch_copy`) instead
of running the data movement inside `ncclDevKernel_Generic_2` on CUs.

The important discriminator is the buffer provenance:

| Buffer source | Expected data path |
|---|---|
| `symm_mem.empty` / FSDP `SymmMemAllGather` | SDMA / copy engine |
| regular `torch.empty` / default FSDP all-gather | CU-resident RCCL kernel |

The environment variables make the copy-engine path legal and observable, but
they do not by themselves turn regular `torch.empty` FSDP buffers into SDMA
buffers. The FSDP custom all-gather hook is the key.

### Validation

For low-level validation, use a symmetric-memory probe and verify that
`symm_mem.rendezvous()` completes and that a symmetric-memory all-gather runs.
For profiling validation, count HSA API calls or inspect traces for:

```text
hsa_amd_memory_async_batch_copy
__amd_rocclr_batchMemOp.kd
```

Non-zero counts during all-gather indicate the SDMA copy-engine path. A trace
showing only `ncclDevKernel_Generic_2` for the data movement is not the SDMA
path, even if the communicator reports cuMem transport setup.

### Driver and runtime compatibility

The SDMA/FSDP path depends on PyTorch symmetric-memory rendezvous, which uses
ROCr virtual-memory APIs under the hood. The ROCm runtime and loaded amdgpu
driver must be compatible.

One known failure mode was observed with a ROCm 7.15 nightly image where
`symm_mem.rendezvous()` hung inside:

```text
hsa_amd_vmem_set_access
  -> hsaKmtMemoryVaMap
  -> driver ioctl / timeline wait
```

The userspace change involved:

```text
b58362f60ff4f0b2b31a32a2a368db6bffdd5883
ROCM-21775 Use DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT ioctl in hsaKmt map/unmap ops
```

With an older loaded amdgpu driver, the relevant ioctl did not return, causing
`torch.distributed._symmetric_memory.rendezvous()` to hang. Updating and
reloading the amdgpu driver fixed the hang on the affected MI300X system:

```text
$ sudo dkms status
amdgpu/7.1.3-2377367.22.04, 6.5.0-45-generic, x86_64: installed
$ uname -r
uname -r: 6.5.0-45-generic
```

If an SDMA run hangs before any FSDP forward progress, please try to dump stack and see where it hangs.

## MORI hierarchical all-gather

MORI replaces FSDP2 all-gather with a hierarchical intra-node and cross-node
collective. The default device-driven mode uses SDMA inside each node and
vendor direct verbs for cross-node RDMA. A host-proxy mode is also available.

### Enablement

Set the single user-facing switch before launching Primus:

```bash
export FSDP_ALL_GATHER_BACKEND=mori

runner/primus-cli direct -- train pretrain --config <experiment.yaml>
```

TorchTitan applies MORI to each compatible FSDP2 module. Megatron applies it to
FSDP2 transformer layers and additionally requires:

```bash
--use_torch_fsdp2 true
```


### What the Primus patch does

The TorchTitan and Megatron patches wrap `fully_shard()` and attach one shared
adapter to compatible modules:

```python
from primus.backends.common.mori_allgather import MoriAllGather

mori_all_gather = MoriAllGather()
module.set_custom_all_gather(mori_all_gather)
```

The adapter:

1. Initializes MORI SHMEM once from torchrun's default c10d `TCPStore`. This
   avoids creating an eager cross-node RCCL transport solely for MORI
   bootstrap.
2. Derives ranks per node from `LOCAL_WORLD_SIZE`.
3. Inspects every compatible FSDP parameter group before training, computes the
   largest padded per-rank shard using its effective communication dtype, and
   builds `HierAllGather` once at that capacity.
4. Launches MORI on the current CUDA stream and returns a Work-like object when
   FSDP requests asynchronous completion.


### Inter-node execution modes

Primus supports two MORI implementations for the cross-node leg.

#### Device-driven RDMA (default)

When `MORI_FSDP_HOST_PROXY` is unset or false, Primus constructs
`mori.ccl.HierAllGather`. GPU kernels post cross-node RDMA operations directly
through MORI's device-verbs/IBGDA path, while intra-node traffic uses MORI SDMA.
The CPU is not in the per-collective data path.

This mode provides direct GPU/NIC overlap, but requires a compatible live NIC
stack and working device-side queue support. For Ionic devices, that includes
the effective CCQE capability checked by MORI preflight. Driver, firmware,
vendor-library, GID, or CCQE mismatches can prevent initialization or cause a
device-side collective failure.

#### CPU host proxy

Set the following to use MORI's persistent host-proxy implementation:

```bash
export MORI_FSDP_HOST_PROXY=1
```

Primus then constructs `mori.ccl.HostProxyHierAllGather`. A CPU proxy posts
RDMA work requests and polls completion queues on behalf of the GPU. Collective
payloads remain in registered GPU memory; host proxy does not bounce the data
through CPU memory. This path can maintain deeper NIC send queues and avoids
depending on GPU-posted RDMA, but it adds CPU progress and synchronization to
the hot path.

By default, host proxy uses PyTorch/RCCL for its intra-node gather legs. Enable
MORI SDMA for those legs with:

```bash
export MORI_HOSTPROXY_SDMA_INTRA=1
```

The
[currently pinned host-proxy implementation](https://github.com/ROCm/mori/blob/12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4/python/mori/ccl/host_proxy_ag.py#L177-L185)
has a single-node degenerate path and a two-node cross-node path; more than two
nodes raises `NotImplementedError`. This limitation applies only to host proxy,
not the default device-driven `HierAllGather`. Host proxy allocates a
persistent full-output GPU staging buffer. Primus automatically derives the
maximum per-rank shard from the attached FSDP groups; no manual size calculation
is required. The host-proxy-specific `MORI_FSDP_HOSTPROXY_CAP_MB` remains
available as an additional minimum.


### Runtime preflight

MORI is sensitive to the live NIC driver, firmware, direct-verbs library, GID,
and capabilities such as Ionic CCQE.  Any slight misalignment / misconfig will likly cause MORI to fail. To mitigate this issue, we provide an unified Primus preflight command, which can detect all the known critical configs and do a test all-gather on every target node:

```bash
runner/primus-cli direct -- preflight --mori
```

For multi-node validation:

```bash
runner/primus-cli direct -- preflight --mori \
  --mori-nodes node1,node2 \
  --mori-socket-ifname <bootstrap-interface> \
  --mori-gid-index <rocev2-gid-index>
```

The CLI invokes `primus/tools/preflight/mori_preflight.py`, which runs
`mori_preflight.sh` on every selected node. The shell worker:

1. Prints host identity, GPU, IP, RDMA links, valid GIDs, NIC
   driver/firmware, vendor-library hash, and required DV symbols.
2. Starts a privileged temporary container from the pinned Primus CI image.
3. Mounts the detected host vendor library into that container.
4. Calls `runner/helpers/mori/install_mori.sh` to install dependencies, clone
   the pinned source/submodules, and build MORI with live RDMA visibility.
5. Calls MORI's runtime detector and records `ccqe_runtime` in the node
   fingerprint.
6. Runs an 8-GPU bit-exact all-gather smoke.
7. When `--mori-nodes` is set, keeps the temporary containers for this same
   information/build/local smoke on
   every node, verifies matching node fingerprints, then launches one
   all-gather over all `8 × N` ranks before removing them.

The mori source version pinning is required for now, till this PR is stablized in our base rocm docker: https://github.com/ROCm/mori/pull/441

Logs and phase timing are written under
`/tmp/primus-mori-preflight-<node>-<timestamp>/`.

### Vendor library names

The library names used by preflight come directly from MORI:

| NIC | MORI runtime loader names |
|---|---|
| Ionic / AINIC | `libionic.so` |
| Broadcom BNXT | `libbnxt_re.so`, then `libbnxt_re-rdmav59.so`, then `libbnxt_re-rdmav34.so` |
| Mellanox mlx5 | `libmlx5.so` |

MORI's
[`dv_loader.hpp`](https://github.com/ROCm/mori/blob/dc4bc75a8ae63cb79a3ce17e55f2be3d8aa692c2/include/mori/application/transport/rdma/providers/dv_loader.hpp#L133)
uses these exact `dlopen()` names. Its
[`MoriDetectDevice.cmake`](https://github.com/ROCm/mori/blob/dc4bc75a8ae63cb79a3ce17e55f2be3d8aa692c2/cmake/MoriDetectDevice.cmake#L140)
uses the same names for build-time `find_library()` detection. Preflight mounts
the host's detected vendor library under these aliases so build-time detection
and runtime loading use the same library.

### Automatic MORI installation

When `FSDP_ALL_GATHER_BACKEND=mori` is set, the launcher hook checks whether
`mori.ccl.HierAllGather` is available. If it is missing, the hook calls
`runner/helpers/mori/install_mori.sh` before torchrun starts, so no separate
user installation command is needed. Automatic installation requires root
inside the training container.

Useful overrides are `MORI_REPO`, `MORI_REF`, `MORI_SOURCE_DIR`, `MAX_JOBS`,
and `ROCM_PATH`. The installer clears `MORI_DEVICE_NIC` so MORI detects the
live NIC and mounted vendor library.


### Troubleshooting

- `ccqe_candidate=true` on one node and `mixed` or `false` on another: choose
  nodes with matching Ionic firmware and vendor-library capabilities.
- `ccqe_runtime=true` on one node and `false` on another: the training image
  sees different effective vendor-library or NIC support; do not launch the
  cross-node collective.
- `local GID N/A`: inspect `/sys/class/infiniband/ionic_*/ports/1/gids/`;
  this pair uses `NCCL_IB_GID_INDEX=1`, not `3`.
- BNXT `231.x`: unsupported for MORI IBGDA; use supported firmware/userspace or
  a validated mlx5/ionic pair.
