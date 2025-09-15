# How to run Aiter attention kernel benchmark on AMD MI355/350


## 1. Pull the Docker image for MI355

```bash
docker pull rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
```
## 2. Start the container.

```bash
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME \
    -w /workspace/torchtitan     --name MI355_bench 
    docker.io/rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
```

## 3. Run Aiter attention benchmark

```bash
cd benchmark/kernel/attention
python test_aiter.py 
```
