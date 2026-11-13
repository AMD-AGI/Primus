# SpecForge ROCm runtime image

Build and run instructions are in
[docs/04-technical-guides/specforge/runtime-image.md](../../../docs/04-technical-guides/specforge/runtime-image.md).

Build from the **Primus repository root**:

```bash
uv run --no-project python examples/specforge/docker/build.py mi35x --build \
  --tag primus-specforge:local-mi35x
```
