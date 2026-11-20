# Vision backend

Primus owns the training loop for multi-view geometry models. A model is a
registered builder, selected by `model.name`, the same way Flux and Wan are
selected in the diffusion backend. VGGT and WorldMirror are the first two.
`synthetic` is a tiny in-tree model for checking the loop.

The trainer supports DDP and FSDP2, bf16 autocast, AdamW with parameter groups,
a warmup-cosine schedule, gradient accumulation, and safetensors export. It
does not call WorldMirror's Lightning trainer or VGGT's trainer.

VGGT and WorldMirror are in the vision package
(`primus/backends/vision/vggt_model/` and `primus/backends/vision/worldmirror/`).
Published checkpoints still load: `facebook/VGGT-1B` and
`tencent/HunyuanWorld-Mirror`.

Stage 2's Gaussian renderer needs the `gsplat` package. Set
`VISION_BUILD_GSPLAT=1` so the install hook builds gsplat v1.5.3 from a
temporary copy, with the ROCm fixes and GLM. Stage 1 does not import it. The
first build takes several minutes. Set `PYTORCH_ROCM_ARCH` (for example
`gfx950` on MI355X) to compile for one GPU architecture.

```bash
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
```

VGGT is post-training. It loads `facebook/VGGT-1B` and continues on Co3D.
`CO3D_ANNOTATION_DIR` holds VGGT's `{category}_{split}.jgz` files.

```bash
CO3D_DIR=/shared_nfs/3dwm/hywm-3dwm-sub20/co3d \
CO3D_ANNOTATION_DIR=/path/to/vggt_co3d_annotations \
MAX_STEPS=20 \
./primus-cli direct -- train posttrain \
  --config examples/vision/configs/MI355X/vggt-posttrain.yaml
```

WorldMirror is post-training. Both stages use `train posttrain`. Stage 1 trains
the geometry transformer and the camera, point, depth, and normal heads on the
Hypersim `train` split. The Gaussian head stays off.

The paper ([arXiv 2510.10726](https://arxiv.org/abs/2510.10726), "Decoupled
Sequential Training") starts stage 1 from pretrained VGGT weights, not from a
WorldMirror checkpoint. The loaders copy tensors by exact name and VGGT uses
different module names, so rename the VGGT file first. The new modules (depth,
pose and ray embeds, normal head, Gaussian head) keep their random
initialisation.

```bash
python3 examples/vision/convert_vggt_to_worldmirror.py \
  /path/to/VGGT-1B/model.safetensors /path/to/vggt_as_worldmirror.safetensors
```

Point `PRETRAINED_PATH` at the converted file. The trainer prints
`loaded N/M WorldMirror tensors` at start-up. Without `PRETRAINED_PATH`, stage 1
starts from the published `tencent/HunyuanWorld-Mirror` weights instead, which
is a fine-tune of the final model rather than the paper recipe.

```bash
PRETRAINED_PATH=/path/to/vggt_as_worldmirror.safetensors \
MAX_STEPS=20 \
./primus-cli direct -- train posttrain \
  --config examples/vision/configs/MI355X/worldmirror-stage1-posttrain.yaml
```

Stage 2 loads the stage-1 weights, freezes the geometry backbone, and trains
the Gaussian head. `PRETRAINED_CKPT` is required. `VISION_BUILD_GSPLAT=1`
builds the renderer.

```bash
VISION_BUILD_GSPLAT=1 \
PRETRAINED_CKPT=./output/vision-worldmirror-stage1/model.safetensors \
MAX_STEPS=20 \
./primus-cli direct -- train posttrain \
  --config examples/vision/configs/MI355X/worldmirror-stage2-posttrain.yaml
```

`PARALLELISM_MODE=fsdp2` shards each transformer block, then the rest of the
model. On one process the trainer leaves the model unsharded.
