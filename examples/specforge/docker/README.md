# SpecForge ROCm runtime image

Build and run instructions are in
[docs/04-technical-guides/specforge/runtime-image.md](../../../docs/04-technical-guides/specforge/runtime-image.md).

Build from the **Primus repository root**:

```bash
docker build -f examples/specforge/docker/Dockerfile \
  -t primus-specforge:v0.5.23-rocm720-mi35x-20261103 .
```

For the Muse Spark base and SpecForge EAGLE4 PR, use
[`Dockerfile.muse-spark-eagle4`](Dockerfile.muse-spark-eagle4).
Its [build instructions and capture limitation](../../../docs/04-technical-guides/specforge/muse-spark-eagle4-image.md)
describe the separate GPU-free validation scope.
