# Full DeepSeek-V4-Flash 4k SFT — multi-node on gfx942 (MI308X / CDNA3)

One-click launcher: [`run_full_4k_multinode.sh`](./run_full_4k_multinode.sh).

This trains the **complete** DeepSeek-V4-Flash model — 43 decoder layers + 1 MTP,
256 experts (top-6), `compress_ratios` cycling dense / CSA(4) / HCA(128) — at 4k
sequence length, across 3 nodes (24× MI308X). It is the full model, not the 4-layer
cut used by the single-node 128k smoke.

## TL;DR

```bash
# In each node's container, from the Primus repo root. Only two vars differ per node.
# node 0 (master):
MASTER_ADDR=<node0-ip> NODE_RANK=0 bash examples/deepseek-v4/gfx942/run_full_4k_multinode.sh
# node 1:
MASTER_ADDR=<node0-ip> NODE_RANK=1 bash examples/deepseek-v4/gfx942/run_full_4k_multinode.sh
# node 2:
MASTER_ADDR=<node0-ip> NODE_RANK=2 bash examples/deepseek-v4/gfx942/run_full_4k_multinode.sh
```

`MASTER_ADDR` is the master node's IP on the socket NIC (see Networking). Node
addresses are **not** hardcoded in the script.

## Why 3 nodes

The model does not fit on 1 or 2 nodes on this GPU, and the binding term is the
**expert optimizer state**:

- Experts are sharded `ETP × EP × PP` ways. Their optimizer state (fp32 master +
  Adam m/v) **cannot** be sharded across data parallelism, because at EP=8 the
  expert data-parallel size is 1 (EP consumes the parallelism DP would use).
- On 2 nodes (16 GPUs) `EP × PP ≤ 16`, giving ~18 B experts/card and an
  unshardable optimizer state that fits neither GPU nor host.
- On 3 nodes (24 GPUs) `EP × PP = 24` → ~12 B experts/card, and with the optimizer
  offloaded to the host it fits. This is the minimum viable configuration.

Parallelism used:

| domain    | layout                         |
|-----------|--------------------------------|
| attention | TP=1 · PP=3 · CP=1 → DP=8       |
| expert    | ETP=1 · EP=8 · PP=3 (24-way)   |
| PP layout | `Et*14｜t*14｜t*15mL`           |

## The two fixes that make it work

Both were diagnosed from measurement (py-spy stacks + dmesg), not guessed.

### 1. `NCCL_ALGO=Ring` — cross-node collective deadlock

Symptom: the first step hangs with GPUs at 0%, stuck in a 1-int `all_reduce` on the
cross-node model-parallel group (`logical_and_across_model_parallel_group`).

Cause: NCCL defaults a **small** `all_reduce` to the **Tree** algorithm, and
Tree-over-TCP-socket across nodes **deadlocks** on this fabric.

Fix: force `NCCL_ALGO=Ring` (plus `RCCL_USE_AMD_SMI_LIB=1` for fabric-topology
probing). This matches a known-working 6-node run on the same fabric, which passed
all-reduce / all-gather / all-to-all over the management NIC in ~10 s.

### 2. `optimizer_offload_fraction=0.75` — host OOM

Symptom: a single rank is SIGKILLed by the OOM-killer (dmesg: `python invoked
oom-killer`, anon-rss ~160 GB); the surviving ranks then hang forever in the
collective above, waiting on the dead peer. (This is why fix #1's effect was
masked at first — the "deadlock" was really a dead peer.)

Cause: the optimizer state is **151 GB/rank** (of which the expert part is 147 GB,
unshardable). At `fraction=1.0` that is ~1208 GB/node, and pinned-memory allocation
peaks push it to ~1570 GB — which overran the node that started with the least free
RAM.

Fix (calculated, not a seesaw): `fraction=0.75` puts

- **113 GB/rank on the host** → ~1178 GB/node peak, comfortably under a 3 TB host
  (~460 GB headroom on all three nodes), and
- **38 GB/rank back on the GPU** → ~109 GB of 192 GB used (83 GB headroom).

Both sides are safe, and it keeps `exp_avg_sq` in fp32 (the config's NaN guard) so
numerics are unchanged.

## Networking

RDMA on this fabric is unusable (PFC unconfigured → `IBV_WC_RETRY_EXC_ERR`, and an
ionic GDA hang), so NCCL runs over **TCP on the management NIC**. The script pins
the interface for all three transports — set `NCCL_SOCKET_IFNAME` if your NIC is not
`ens50f0`:

```bash
NCCL_SOCKET_IFNAME=<nic> MASTER_ADDR=<ip> NODE_RANK=<r> bash .../run_full_4k_multinode.sh
```

`GLOO_SOCKET_IFNAME` and `TP_SOCKET_IFNAME` default to the same NIC. All three are
required — miss one and rendezvous or the TP group binds the wrong interface and
hangs.

## gfx942 kernel settings

- `PRIMUS_DSA_BWD_NUM_STAGES=1` — the V4 sparse-MLA backward Triton kernel is tuned
  for gfx950/CDNA4 (160 KB LDS); gfx942 has 64 KB, so LDS multi-buffering is turned
  off or the kernel fails to compile.
- `PRIMUS_INDEXER_TRITON_FULL=1` — use the fused indexer path instead of an eager
  einsum that materialises `[B,S,H,P]`.
- attention backend `triton_v2` for both the dense/HCA and CSA paths (the turbo
  sparse-MLA backend is CDNA4-only).

## Verified result

10-step and 100-step runs both completed cleanly on 3× MI308X nodes over the socket
network, from random init:

| metric              | 10-step        | 100-step       |
|---------------------|----------------|----------------|
| lm loss             | 11.90 → 11.21  | 11.90 → 9.19   |
| grad norm           | 20.0 → 14.9    | 20.0 → 2.8     |
| nan iterations      | 0 (all steps)  | 0 (all steps)  |
| GPU peak mem        | ~150 GB / 192  | ~150 GB / 192  |
| exit                | 0              | 0              |

Loss decreases monotonically and grad norm is stable — no NaN, no backward errors.

## Requirements

- 3 nodes × 8 MI308X (gfx942), 192 GB/GPU, ~3 TB host RAM each.
- Container image `rocm/primus:v26.5-pytorch2.12-te2.15` (or equivalent), started
  with `--network host --privileged --device /dev/kfd --device /dev/dri`, `/apps`
  (or wherever the repo lives) mounted.
- A local DeepSeek-V4 tokenizer directory (default `/apps/DeepSeek-V4-Flash`); only
  `tokenizer.json` / `tokenizer_config.json` are read. Weights are **not** loaded —
  this trains from random init (no V4 Megatron checkpoint exists). Nothing is saved.

## Overridable knobs

| var                                 | default          | meaning                          |
|-------------------------------------|------------------|----------------------------------|
| `MASTER_ADDR`                       | (required)       | master node IP on the socket NIC |
| `NODE_RANK`                         | (required)       | this node's rank, 0..NNODES-1    |
| `NNODES`                            | 3                | node count                       |
| `MASTER_PORT`                       | 29710            | rendezvous port                  |
| `NCCL_SOCKET_IFNAME`                | ens50f0          | socket NIC                       |
| `V4_TOKENIZER`                      | /apps/DeepSeek-V4-Flash | tokenizer dir             |
| `TRAIN_ITERS`                       | 10               | training steps                   |
| `GBS`                               | 24               | global batch size                |
| `PRIMUS_LR`                         | 1.0e-6           | learning rate                    |
| `PRIMUS_OPTIMIZER_OFFLOAD_FRACTION` | 0.75             | optimizer host-offload fraction  |
