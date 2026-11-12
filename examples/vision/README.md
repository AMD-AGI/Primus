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
`VISION_BUILD_GSPLAT=1` so the install hook builds it. Stage 1 does not import
it. The first build takes several minutes. Set `PYTORCH_ROCM_ARCH` (for example
`gfx950` on MI355X) to compile for one GPU architecture.

```bash
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
```

Synthetic smoke run:

```bash
MAX_STEPS=2 \
./primus-cli direct -- train pretrain \
  --config examples/vision/configs/MI355X/synthetic-pretrain.yaml
```

VGGT on Co3D. `CO3D_ANNOTATION_DIR` holds VGGT's `{category}_{split}.jgz` files.

```bash
CO3D_DIR=/shared_nfs/3dwm/hywm-3dwm-sub20/co3d \
CO3D_ANNOTATION_DIR=/path/to/vggt_co3d_annotations \
MAX_STEPS=20 \
./primus-cli direct -- train pretrain \
  --config examples/vision/configs/MI355X/vggt-pretrain.yaml
```

WorldMirror stage 1 on Hypersim:

```bash
MAX_STEPS=20 \
./primus-cli direct -- train pretrain \
  --config examples/vision/configs/MI355X/worldmirror-stage1-pretrain.yaml
```

WorldMirror stage 2 loads the stage-1 weights and trains the Gaussian head.
`PRETRAINED_CKPT` is required.

```bash
VISION_BUILD_GSPLAT=1 \
PRETRAINED_CKPT=./output/vision-worldmirror-stage1/model.safetensors \
MAX_STEPS=20 \
./primus-cli direct -- train pretrain \
  --config examples/vision/configs/MI355X/worldmirror-stage2-pretrain.yaml
```

`PARALLELISM_MODE=fsdp2` shards each transformer block, then the rest of the
model. On one process the trainer leaves the model unsharded.
