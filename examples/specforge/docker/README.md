# SpecForge ROCm runtime image

Build and run instructions are in
[docs/04-technical-guides/specforge/runtime-image.md](../../../docs/04-technical-guides/specforge/runtime-image.md).

Build from the **Primus repository root**:

```bash
docker build -f examples/specforge/docker/Dockerfile \
  -t primus-specforge:v0.5.14-rocm700-mi35x .
```
