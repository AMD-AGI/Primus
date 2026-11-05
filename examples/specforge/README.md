# SpecForge on Primus

Official documentation is in
[`docs/04-technical-guides/specforge/`](../../docs/04-technical-guides/specforge/introduction.md):

- [Introduction](../../docs/04-technical-guides/specforge/introduction.md)
- [Run SpecForge with Primus](../../docs/04-technical-guides/specforge/running.md)
- [Configuration](../../docs/04-technical-guides/specforge/configuration.md)
- [Runtime image](../../docs/04-technical-guides/specforge/runtime-image.md)

Example YAMLs stay in [`configs/`](configs/). Build the image from the
repository root:

```bash
docker build -f examples/specforge/docker/Dockerfile \
  -t primus-specforge:v0.5.14-rocm700-mi35x .
```
