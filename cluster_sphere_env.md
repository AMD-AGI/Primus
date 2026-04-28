## Cluster Sphere — RDMA environment recommendations

### Host `useocpm2m-097-078`

**Warnings:**

- Multiple firmware versions detected — standardization recommended.

| RDMA | PCI | NETDEV | Firmware | GID idx | GID | Vendor |
|------|-----|--------|----------|---------|-----|--------|
| mlx5_0 | 0000:0c:00.0 | rdma0 | 28.47.1900 | 3 | ::ffff:10.224.0.73 | MLNX |
| mlx5_1 | 0000:1f:00.0 | eth0 | 22.47.1088 | 3 | ::ffff:10.158.212.73 | MLNX |
| mlx5_2 | 0000:2a:00.0 | rdma1 | 28.47.1900 | 3 | ::ffff:10.224.4.73 | MLNX |
| mlx5_3 | 0000:41:00.0 | rdma2 | 28.47.1900 | 3 | ::ffff:10.224.8.73 | MLNX |
| mlx5_4 | 0000:58:00.0 | rdma3 | 28.47.1900 | 3 | ::ffff:10.224.12.73 | MLNX |
| mlx5_5 | 0000:86:00.0 | rdma4 | 28.47.1900 | 3 | ::ffff:10.224.16.73 | MLNX |
| mlx5_6 | 0000:9a:00.0 | eth1 | 22.47.1088 | - | N/A | MLNX |
| mlx5_7 | 0000:a5:00.0 | rdma5 | 28.47.1900 | 3 | ::ffff:10.224.20.73 | MLNX |
| mlx5_8 | 0000:bd:00.0 | rdma6 | 28.47.1900 | 3 | ::ffff:10.224.24.73 | MLNX |
| mlx5_9 | 0000:d5:00.0 | rdma7 | 28.47.1900 | 3 | ::ffff:10.224.28.73 | MLNX |

**Suggested NCCL / socket exports:**

```bash
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

**Suggested rocSHMEM exports:**

```bash
export ROCSHMEM_HEAP_SIZE=7524589824
export ROCSHMEM_MAX_NUM_CONTEXTS=256
```

**Example Docker launch (vendor-specific template):**

```bash
docker run --rm -it \
    --device /dev/dri \
    --device /dev/infiniband \
    --device /dev/kfd \
    --network host \
    --ipc host \
    --privileged \
    --ulimit memlock=-1:-1 \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 64G \
    -v /sys:/sys \
    -v $HOME/.ssh:/root/.ssh \
    -v $HOME:$HOME \
    -v /dev/infiniband:/dev/infiniband \
    -v /sys/class/infiniband:/sys/class/infiniband:ro \
    -v /sys/class/net:/sys/class/net:ro \
    -v /sys/bus/pci:/sys/bus/pci:ro \
    <image>
```

