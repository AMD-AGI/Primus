# MoRI EP-Bench (Slurm, no torchrun)

This directory provides a Slurm-based launcher for the **MoRI** expert-parallel
dispatch/combine microbenchmarks on AMD Instinct GPUs. It mirrors the style of
[`benchmark/kernel/rccl/run_slurm.sh`](../rccl/run_slurm.sh) but drives the
MoRI tests **without `torchrun`**.

## Why no torchrun?

Both MoRI bench scripts use `torch.multiprocessing.spawn` internally to spawn
one process per GPU on each node:

- Intranode: [`tests/python/ops/bench_dispatch_combine.py`](https://github.com/ROCm/mori) spawns `world_size` (=GPUs) local processes itself.
- Internode: [`examples/ops/dispatch_combine/test_dispatch_combine_internode.py`](https://github.com/ROCm/mori) reads
  `RANK` as the **node rank**, `WORLD_SIZE` as the **number of nodes**, and
  spawns `GPU_PER_NODE` local processes.

So we only need **one task per node** from Slurm. `srun -N $NNODES
--ntasks-per-node=1` already provides `SLURM_NODEID` / `SLURM_NNODES` /
`SLURM_JOB_NODELIST`, and the launcher forwards those as `NODE_RANK` /
`NNODES` / `MASTER_ADDR` / `MASTER_PORT` to the inner script.

## Benchmarks covered

| Scope     | Mode        | Script                                                          |
|-----------|-------------|------------------------------------------------------------------|
| Intranode | Normal (bf16)        | `tests/python/ops/bench_dispatch_combine.py`             |
| Intranode | Low latency (fp8)    | `tests/python/ops/bench_dispatch_combine.py --dtype fp8_e4m3_fnuz` |
| Internode | Normal (v1, bf16)    | `examples/ops/dispatch_combine/test_dispatch_combine_internode.py --cmd bench` |
| Internode | Low latency (v1_ll)  | `examples/ops/dispatch_combine/test_dispatch_combine_internode.py --cmd bench --kernel-type v1_ll` |

Intranode runs trigger automatically when `NNODES == 1`; internode runs trigger
when `NNODES > 1`. (DeepEP tests are out of scope for this launcher.)

## Prerequisites

- Slurm cluster with AMD Instinct nodes.
- InfiniBand HCA(s) available on each node (default `mlx5_0`).
- Docker available on compute nodes.
- A Docker image that ships MoRI under `/app/mori`. Two options:

  ### Option A (recommended): build the slim MoRI image shipped here

  Built on the latest stable upstream
  [`vllm/vllm-openai-rocm`](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags)
  image (which already provides ROCm, PyTorch, and vLLM) with **only MoRI**
  added on top. AMD's `rocm/vllm` / `rocm/vllm-dev` images are deprecated in
  favor of `vllm/vllm-openai-rocm`
  (see [vLLM docker docs](https://docs.vllm.ai/en/stable/deployment/docker/)).

  ```bash
  cd Primus/benchmark/kernel/ep_bench/docker

  # Defaults: BASE_IMAGE=vllm/vllm-openai-rocm:v0.21.0, GPU_ARCH=gfx942,
  #           MORI_REPO=https://github.com/ROCm/mori.git, MORI_REF=main
  docker build -f Dockerfile.mori -t primus-mori-ep:latest .

  # MI355 / gfx950 + pinned MoRI ref:
  docker build -f Dockerfile.mori \
      --build-arg GPU_ARCH=gfx950 \
      --build-arg MORI_REF=v0.x.y \
      -t primus-mori-ep:mi355 .

  # Precompile MoRI JIT kernels at build time (needs a GPU exposed to the build):
  DOCKER_BUILDKIT=0 docker build -f Dockerfile.mori \
      --build-arg MORI_PRECOMPILE=1 \
      -t primus-mori-ep:precompiled .
  ```

  Build args supported by [`docker/Dockerfile.mori`](./docker/Dockerfile.mori):

  | Build arg          | Default                                  | Purpose                                            |
  |--------------------|------------------------------------------|----------------------------------------------------|
  | `BASE_IMAGE`       | `vllm/vllm-openai-rocm:v0.21.0`          | Override to pin a different upstream vLLM ROCm tag |
  | `GPU_ARCH`         | `gfx942`                                 | Sets `PYTORCH_ROCM_ARCH` for the MoRI install      |
  | `MORI_REPO`        | `https://github.com/ROCm/mori.git`       | MoRI git remote                                    |
  | `MORI_REF`         | `main`                                   | Branch / tag / SHA to check out                    |
  | `MORI_PRECOMPILE`  | `0`                                      | If `1`, JIT-precompile MoRI kernels into the image |

  ### Option B: reuse the MAD `large_ep_benchmark` image

  Heavier image that also includes rocSHMEM + DeepEP + perftest. Build from
  the MAD repository root:

  ```bash
  # From the MAD repository root
  docker build \
      -f docker/large_ep_benchmark.ubuntu.amd.Dockerfile \
      -t ep-benchmarking:latest .
  ```

  See
  [`MAD/scripts/large-ep-benchmark/README.md`](https://github.com/ROCm/MAD/blob/main/scripts/large-ep-benchmark/README.md)
  for cluster/Dockerfile customization (GPU arch, NIC arch).

## Environment variables

| Variable            | Required | Default                                | Description                                                  |
|---------------------|----------|----------------------------------------|--------------------------------------------------------------|
| `DOCKER_IMAGE`      | yes      | —                                      | Docker image to launch (e.g. `primus-mori-ep:latest` or `ep-benchmarking:latest`). |
| `NNODES`            | no       | `$SLURM_NNODES`, then `1`              | Number of nodes; drives intranode vs internode branch.       |
| `PARTITION`         | no       | unset (prefer `-p` on `sbatch`)        | Slurm partition for child `srun` calls.                      |
| `GPUS_PER_NODE`     | no       | `8`                                    | GPUs per node; forwarded as `GPU_PER_NODE` to MoRI.          |
| `MASTER_PORT`       | no       | `2373`                                 | Rendezvous port for `dist.init_process_group`.               |
| `IBDEVICES`         | no       | `mlx5_0`                               | IB HCA(s) for `ROCSHMEM_USE_IB_HCA`.                         |
| `LOG_DIR`           | no       | `${SLURM_SUBMIT_DIR}/logs`             | Host directory mounted as `/run_logs` inside the container.  |
| `EXTRA_DOCKER_ARGS` | no       | unset                                  | Extra args appended to `docker run`.                         |

## Usage

### Single node (intranode only)

```bash
cd Primus/benchmark/kernel/ep_bench

DOCKER_IMAGE=primus-mori-ep:latest \
    sbatch -N 1 -p <partition> run_slurm.sh
```

### Multi-node (internode)

```bash
cd Primus/benchmark/kernel/ep_bench

DOCKER_IMAGE=primus-mori-ep:latest \
IBDEVICES=mlx5_0 \
NNODES=4 \
    sbatch -N 4 -p <partition> -w <node-list> run_slurm.sh
```

### Overriding the rendezvous port or log directory

```bash
DOCKER_IMAGE=primus-mori-ep:latest \
MASTER_PORT=39566 \
LOG_DIR=/shared/logs/mori-$(date +%Y%m%d) \
    sbatch -N 4 -p <partition> run_slurm.sh
```

## Output

- Slurm job stdout/stderr go to the default `slurm-<jobid>.out` (or whatever
  you configure with `-o` / `-e` on `sbatch`).
- Per-test logs are written under `${LOG_DIR}` (default
  `${SLURM_SUBMIT_DIR}/logs`):

  | File                                              | Produced when |
  |---------------------------------------------------|---------------|
  | `mori_intranode_bf16.log`                         | `NNODES == 1` |
  | `mori_intranode_fp8_ll.log`                       | `NNODES == 1` |
  | `mori_internode_v1_rank<NODE_RANK>.log`           | `NNODES > 1`  |
  | `mori_internode_v1_ll_rank<NODE_RANK>.log`        | `NNODES > 1`  |

## Files

- [`run_slurm.sh`](./run_slurm.sh) — sbatch entrypoint; pre-pulls the image
  and `docker run`s the inner script on each node with one task per node.
- [`run_mori_bench.sh`](./run_mori_bench.sh) — inner script executed inside
  the container; branches on `NNODES` and invokes MoRI bench scripts directly
  with plain `python` (no torchrun).
- [`docker/Dockerfile.mori`](./docker/Dockerfile.mori) — slim image recipe
  layered on the latest stable `vllm/vllm-openai-rocm`, installing only MoRI
  (mirrors the MoRI portion of the MAD `large_ep_benchmark` Dockerfile).
