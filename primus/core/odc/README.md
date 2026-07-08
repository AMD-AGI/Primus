# On-demand Communication (ODC)

ODC is a patch to FSDP that adapts Parameter Server (PS) into Fully Sharded Data Parallel (FSDP) by replacing collective all-gather and
reduce-scatter with on-demand point-to-point communication.

![Original-FSDP](./docs/readme/FSDP-ODC.jpg)

With ODC, the synchronization frequency is reduced from per-iteration to per-minibatch, which fundamentally reduces the workload-imbalance bubbles in FSDP.

ODC is accepted in ICLR 2026! Check out the [paper](https://openreview.net/pdf?id=iIEEgI6WsF) for more details.

## Attribution / Provenance

This ODC code is ported from the upstream open-source project
[sail-sg/odc](https://github.com/sail-sg/odc) (Sea AI Lab, ICLR 2026). Per its
package metadata the upstream project is released under the MIT License
(copyright held by the original authors, Sea AI Lab); the upstream repository
does not ship a standalone `LICENSE` file or per-file license headers.

This version has been adapted for the AMD ROCm / MI300X platform, including
migrating the communication backend from nvshmem/CUDA to rocSHMEM/MORI + HIP,
device-side reduce, and torchrun-based launching.

## ODC Primitives

The key idea is to replace the collective all-gather and reduce-scatter with a on-demand point-to-point communication.

![ODC Primitives](./docs/readme/ag_rs.png)

To support transparent on-demand communications, we implement RDMA based primitives on ROCm using HIP GPU IPC (intra-node) and MORI-SHMEM / rocSHMEM (inter-node). Details are in [odc/primitives](./odc/primitives).

## Support for FSDP
- FSDP1
- FSDP2
  - HSDP
  - `reshard_after_forward=int`

## Usage

### Prerequisites

- PyTorch (ROCm build)
- ROCm 7.x
- Python >= 3.8
- `amd_mori` (symmetric-memory / RDMA backend)

We highly recommend using a ROCm PyTorch base image (e.g. the
`tasimage/primus-odc` images, ROCm 7.2.0 based).

### Enable ODC
ODC ships as an **in-tree module of Primus** (no separate `pip install`; it has no
compiled extension). Put its parent directory on `PYTHONPATH` so `import odc`
resolves, plus the `odc_early` shim dir so the MORI/TE load-order fix runs at
interpreter startup:
```
export PYTHONPATH=<PRIMUS_ROOT>/primus/core:<PRIMUS_ROOT>/primus/core/odc/odc_early:$PYTHONPATH
```
(The launcher `rocshmem_runtime/scripts/run_odc.sh` sets this up automatically.)
ODC is pure Python. The rocSHMEM backend (single-node XGMI IPC host
API and multi-node GPU-direct GDA) is provided by Primus-Turbo as the
`primus_turbo.pytorch._C.odc_rocshmem_host` / `odc_rocshmem_gda` pybind
submodules and consumed by `odc/primitives/_rocshmem_backend.py`; select it with
`ODC_P2P_BACKEND=rocshmem` (and `ODC_ROCSHMEM_GDA=1` for the multi-node GDA
path). Ensure a Primus-Turbo build with the ODC rocSHMEM ops is importable
(installed, or on `PYTHONPATH`).

> TODO(primus-turbo): pin the exact Primus-Turbo merge commit (`PRIMUS_TURBO_COMMIT`)
> once the PR adding the ODC rocSHMEM ops is merged.

## Quick Start

A complete example is provided in `examples/llm_training/`:
```shell
pip install -r examples/llm_training/requirements.txt

bash examples/llm_training/run.sh
```

## Memory
> User may need to tune `MORI_SHMEM_HEAP_SIZE` for better memory usage.
When using ODC in FSDP, symmetric buffers are allocated for sharded parameters, sharded gradient accumulation buffer, miscellaneous buffers for `gather` and `scatter-accumulate`.
To achieve smallest memory footprint,
`MORI_SHMEM_HEAP_SIZE` should be set to be slightly higher than the size of sharded parameters and gradient: params.element_size() * params.numel() + grad_reduce_buf.element_size() * grad_reduce_buf.numel().

### Basic Usage with FSDP1

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import odc
from odc.fsdp import fsdp1


fsdp1.patch_fsdp1()

torch.distributed.init_process_group(backend="nccl", device_id=device)
odc.init_shmem()


fsdp_model = FSDP(
    model,
    # ...
)

for epoch in range(10):
    for minibatch in dataset:
        fsdp1.pre_minibatch_start(fsdp_model)
        loss = loss_fn(model)
        loss.backward()
        fsdp1.pre_optimizer_step(model)
        optimizer.step()
        optimizer.zero_grad()

fsdp1.stop()
```

### Basic Usage with FSDP2

```python
import torch
import odc
from odc.fsdp import fsdp2


torch.distributed.init_process_group(backend="nccl", device_id=device)
odc.init_shmem()

fsdp2.patch_fsdp2()

for layer in model.layers:
    fully_shard(layer, **fsdp_kwargs)
fsdp_model = fully_shard(model, **fsdp_kwargs)

# Call patch_lazy_init just as how we call fully_shard above.
for layer in fsdp_model.layers:
    fsdp2.patch_lazy_init(layer)
fsdp2.patch_lazy_init(fsdp_model)

for epoch in range(10):
    for minibatch in dataset:
        fsdp2.pre_minibatch_start(fsdp_model)
        loss = loss_fn(model)
        loss.backward()
        fsdp2.pre_optimizer_step(model)
        optimizer.step()
        optimizer.zero_grad()

fsdp2.stop()
```


## Development

### Running Linter
```
make lint
```

### Running Tests
```bash
make test
```
