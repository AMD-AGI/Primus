# Vision backend

`framework: vision` trains multi-view geometry models with a Primus-owned loop.
`primus/backends/vision/registry.py` maps `model.name` to a model builder, a
dataset builder, and a trainer builder. Adding a model registers those builders
and adds a preset under `primus/configs/models/vision/`.

`VisionFSDP2Trainer` is the loop. `parallelism.mode` is `ddp` or `fsdp2`.
The model implements `VisionModelForTraining`: `forward` returns the loss,
`block_classes` names the transformer blocks, and `param_groups` returns the
optimizer groups.

VGGT lives in `primus/backends/vision/vggt_model/`. WorldMirror lives in
`primus/backends/vision/worldmirror/`. Both are part of this package, the same
way the diffusion models live under `primus/backends/diffusion/`. The trainer
does not import `third_party/vggt` or `third_party/HunyuanWorld-Mirror-rocm`.

The existing `worldmirror` backend still launches WorldMirror's Lightning
trainer from `third_party/HunyuanWorld-Mirror-rocm`. That backend is unchanged.

Stage 2's Gaussian renderer needs the `gsplat` package. Set
`VISION_BUILD_GSPLAT=1` and the install hook builds it. Stage 1 does not import
it.

Launch recipes live in `examples/vision/`.
