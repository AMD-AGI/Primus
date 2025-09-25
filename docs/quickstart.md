# Quickstart

This guide helps you run your **first Primus job** on AMD ROCm GPUs.

---

## 1. Environment Setup

Primus requires **Python 3.10+** and ROCm (**6.3+**, recommended 6.4).
The easiest way is to use AMD’s pre-built ROCm container images:

```bash
# Pull ROCm + PyTorch image
docker pull rocm/megatron-lm:v25.8_py310
```

Run the container:

```bash
docker run -it --rm \
    --ipc=host \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add video \
    --privileged \
    --device=/dev/infiniband \
    -v /mnt/data:/data  \
    rocm/megatron-lm:v25.8_py310 bash
```

Inside the container:

```bash
git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git
cd primus
pip install -r requirements.txt
```

---

## 2. Run Your First Job

Example: pretrain Llama3.1-8B inside the container:

```bash
primus-cli direct -- train pretrain --config ./examples/configs/llama3_8B-pretrain.yaml
```

---

## 3. What’s Next

- [Slurm Usage](./usage/slurm_container.md) — multi-node training
- [Container Options](./usage/container.md) — mounts & envs
- [Benchmarking](./benchmark/overview.md) — GEMM/RCCL tests
- [FAQ](./faq.md) — troubleshooting and tips

---

🎉 Congratulations! You’ve run your first Primus job.

---

_Last updated: 2025-09-23_
