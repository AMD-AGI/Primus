

podman rm xiaoming-dev
podman run -d \
  --name=xiaoming-dev \
  --network=host \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri  \
  --device=/dev/infiniband \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  -v /shared/amdgpu/home/xiaoming_peng_qle/workspace:/shared/amdgpu/home/xiaoming_peng_qle/workspace \
  -w /shared/amdgpu/home/xiaoming_peng_qle/workspace/dev/Primus \
  rocm/primus:v25.9_gfx942 sleep infinity

#rocm/pyt-megatron-lm-jax-nightly-private:pytorch_rocm7.0_20251024 sleep infinity

#tasimage/primus:pr-255 sleep infinity

#rocm/pytorch-training:v25.6 sleep infinity
#rocm/megatron-lm:v25.7_py310 sleep infinity
#rocm/pyt-megatron-lm-jax-nightly-private:pytorch_latest
