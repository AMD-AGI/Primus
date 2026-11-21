# RCCL SDMA AllGather for TorchTitan FSDP2

This guide shows how to run TorchTitan (FSDP2) training on MI355X with FSDP's parameter AllGather
on RCCL's copy-engine (SDMA) path, and the same run with the AllGather on RCCL kernels for
comparison. The examples use Llama 3.1 70B; the setup is the same for any TorchTitan model trained
with FSDP2. Only the AllGather moves to the copy engine; ReduceScatter stays on RCCL kernels.

## How it works

`FSDP_ALL_GATHER_BACKEND=rccl_sdma` (older name: `SDMA_ALL_GATHER=1`) turns on two pieces:

- **Hook** `runner/helpers/hooks/06_enable_sdma_all_gather.sh` exports `NCCL_CTA_POLICY=2`
  (zero-CTA policy, required for any RCCL copy-engine collective), cuMem
  (`NCCL_CUMEM_ENABLE=1`, `NCCL_LOCAL_REGISTER=0`), the PyTorch allocator registration hook, and
  `FSDP_ALL_GATHER_BACKEND` itself into the training processes.
- **Patch** `torchtitan.fsdp.sdma_symm_mem_collectives` attaches PyTorch's `SymmMemAllGather` to
  every `fully_shard`-ed module. FSDP then gathers into symmetric-memory buffers that RCCL has
  registered as windows, which is what makes the AllGather eligible for the copy-engine path.

RCCL then picks:

- **One node:** the CE AllGather. Peer copies run through `hipMemcpyBatchAsync`, synchronized with
  `hipStreamBatchMemOp`.
- **Several nodes:** the hierarchical CE AllGather. Intra-node copies run on the copy engine and
  inter-node puts go through RCCL's RMA proxy.

## Requirements

| Item | Tested with |
|---|---|
| GPU | MI355X, 8 per node |
| Host kernel | Linux 6.8 or newer (RCCL cuMem); the hook refuses older kernels |
| Image | `unifiedtrainingdockers.azurecr.io/utd/nightly:primus_rocm10.2_20261021` (RCCL 2.32.3, hierarchical CE). Older RCCL builds silently stay on kernels for multi-node runs |
| PyTorch | 2.12 or newer (`SymmMemAllGather`) |
| Primus | A version that includes the `fully_shard` rebinding fix in `sdma_symm_mem_collectives.py` (see [Getting the code](#getting-the-code)). Without it, the patch logs `installed` but RCCL reports `CE-registered disqualified ... ceRegWindow=0` and AllGather stays on kernels |
| Network (multi-node) | RoCE with routing between rails, and `NCCL_CROSS_NIC` not 0 (see [Settings](#settings)) |
| Tokenizer | The model's tokenizer files in a local directory, passed as `--model.hf_assets_path` (mock data still loads the tokenizer); the example uses Llama 3.1's |

## Getting the code

If your image's Primus predates the fix, mount a Primus checkout that includes it over the
image's copy (`-v /path/to/Primus:/workspace/Primus` in the launch command below). Primus runs
TorchTitan from `third_party/torchtitan`, so initialize that submodule
(`git submodule update --init third_party/torchtitan`).

Alternatively, keep the image's Primus and copy the changed files over it inside the container
before launching:

- `primus/backends/torchtitan/patches/sdma_symm_mem_collectives.py`
- `runner/helpers/hooks/06_enable_sdma_all_gather.sh`
- `runner/.primus.yaml`, needed only when launching with `primus-cli slurm` or `primus-cli container`

Run each comparison only on nodes that are completely idle. Check `rocm-smi --showpids`,
`--showuse` and `--showmemuse` before starting: Slurm can report a node as idle while a stray
process from another job still holds a GPU.

## Settings

The examples use the stock recipe `examples/torchtitan/configs/MI355X/llama3.1_70B-BF16-pretrain.yaml`;
other TorchTitan recipes take the same SDMA and network settings.
The settings below apply to both modes:

- **Model:** full Llama 3.1 70B (80 layers).
- **Training:** BF16, full activation checkpointing, `local_batch_size=6`, `seq_len=8192`, mock
  data. The global batch is `6 x GPUs`, with no gradient accumulation.
- **Recipe default:** `GPU_MAX_HW_QUEUES=4`.

**SDMA mode** sets one variable:

```bash
export FSDP_ALL_GATHER_BACKEND=rccl_sdma
```

**Baseline mode** leaves `FSDP_ALL_GATHER_BACKEND`, `SDMA_ALL_GATHER` and `NCCL_CTA_POLICY` unset.

**Multi-node network settings, both modes.** `runner/helpers/envs/base_env.sh` defaults are wrong
for some clusters:

| Variable | Set to | Why |
|---|---|---|
| `NCCL_IB_GID_INDEX` | First routable RoCE v2 GID index of the NIC (1 on our cluster) | The default of 3 was not routable here |
| `NCCL_CROSS_NIC` | `2` | The default of `0` disables RCCL's RMA proxy, so multi-node AllGather silently falls back to kernels |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | The out-of-band Ethernet interface | Otherwise `IP_INTERFACE` comes from the first `hostname -I` address, which can be an RDMA rail address that other nodes can't reach, and Gloo hangs |
| `NCCL_IB_HCA` | The RDMA NICs, e.g. `ionic_0,...,ionic_7` | |

To find the GID index, look for the first RoCE v2 entry that isn't link-local (`fe80:`):

```bash
dev=ionic_0
for t in $(ls -v /sys/class/infiniband/$dev/ports/1/gid_attrs/types/*); do
  i=${t##*/}; g=$(cat /sys/class/infiniband/$dev/ports/1/gids/$i)
  grep -qi v2 $t && [[ $g != fe80:* && $g != 0000:0000:0000:0000:0000:0000:0000:0000 ]] && { echo $i; break; }
done
```

## Launching

Our runs started one container per node with `docker run` and ran `primus-cli direct` inside
it. On each node, with `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT` set (for example
from Slurm):

```bash
IMAGE=unifiedtrainingdockers.azurecr.io/utd/nightly:primus_rocm10.2_20261021
MODE_ENV=(-e FSDP_ALL_GATHER_BACKEND=rccl_sdma)   # baseline: MODE_ENV=()
NET_ENV=(-e NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
         -e NCCL_IB_GID_INDEX=1 -e NCCL_CROSS_NIC=2
         -e NCCL_SOCKET_IFNAME=<oob-if> -e GLOO_SOCKET_IFNAME=<oob-if> -e IP_INTERFACE=<oob-if>)

docker run --rm --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video \
  --ulimit memlock=-1:-1 --shm-size 64g \
  -v /path/to/Primus:/workspace/Primus -v /path/to/llama3.1-tokenizer:/model:ro \
  "${MODE_ENV[@]}" "${NET_ENV[@]}" \
  -e GPUS_PER_NODE=8 -e NNODES=$NNODES -e NODE_RANK=$NODE_RANK \
  -e MASTER_ADDR=$MASTER_ADDR -e MASTER_PORT=$MASTER_PORT -e HF_HUB_OFFLINE=1 \
  $IMAGE bash -c 'cd /workspace/Primus && ./primus-cli direct -- train pretrain \
    --config examples/torchtitan/configs/MI355X/llama3.1_70B-BF16-pretrain.yaml \
    --model.hf_assets_path /model --training.local_batch_size 6 \
    --training.steps 100 --training.mock_data True --metrics.log_freq 1'
```

On a single node, drop `NET_ENV` and set `NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1`.

With `primus-cli slurm`, export the same variables before launching. `runner/.primus.yaml`
forwards `FSDP_ALL_GATHER_BACKEND`, `SDMA_ALL_GATHER`, `NCCL_*` and `GLOO_SOCKET_IFNAME` into the
container. We haven't timed runs through this launcher.

```bash
export FSDP_ALL_GATHER_BACKEND=rccl_sdma NCCL_CROSS_NIC=2 NCCL_IB_GID_INDEX=1 \
       NCCL_SOCKET_IFNAME=<oob-if> GLOO_SOCKET_IFNAME=<oob-if>
./runner/primus-cli slurm srun -N 2 -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_70B-BF16-pretrain.yaml \
  --model.hf_assets_path /path/to/llama3.1-tokenizer --training.local_batch_size 6 \
  --training.steps 100 --training.mock_data True
```

## Checking that SDMA is active

A run can finish normally with the AllGather still on RCCL kernels. Check the SDMA run:

1. **Primus log.** Look for these lines:
   - `[Hooks] exported env from hook: NCCL_CTA_POLICY=2`
   - `[Patch:torchtitan.fsdp.sdma_symm_mem_collectives] installed: ... rebound already-imported aliases in [...]`
2. **RCCL log.** Add `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING`.
   - Multi-node runs should log `AllGather [Hierarchical CE]: ... RMA proxy + CE`.
   - `CE-registered disqualified ... ceRegWindow=0` means the FSDP buffers were not registered, so
     the patch did not attach `SymmMemAllGather`.
3. **Profiler trace.** In SDMA mode, AllGather contains no `ncclDevKernel` kernels. Instead it
   shows `Memcpy` records and `__amd_rocclr_batchMemOp` kernels.
   - The profiler labels the peer copies `Memcpy DtoH`. They are actually GPU-to-GPU copies.
   - `__amd_rocclr_batchMemOp` is the blit kernel HIP uses for `hipStreamBatchMemOp`.
     `GPU_STREAMOPS_CP_WAIT=1` does not remove it: that flag only affects single
     `hipStreamWaitValue` calls on signal memory.
4. **Copy-engine counters.** Per-process copy-engine busy time is in
   `/sys/class/kfd/kfd/proc/<pid>/sdma_<gpu_id>`, in microseconds. It grows steadily in SDMA mode
   and stays near zero in baseline mode.

## Measurement method

- **Same nodes:** run both modes on the same nodes, one after the other, SDMA first.
- **Warm-up:** discard the first 10 minutes of each mode.
- **Measurement:** average step time over the next 10 minutes, about 20 steps for the 70B example.
- **Profiling:** exclude any profiled steps from the throughput average.
- **Cleanup:** before starting the second mode, wait until no process holds GPU memory.
