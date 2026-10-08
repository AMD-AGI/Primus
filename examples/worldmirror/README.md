# HunyuanWorld-Mirror training

The model code is the Primus submodule `third_party/HunyuanWorld-Mirror-rocm`, tracking `main` from https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror. Primus does not modify that checkout. The directory name is unchanged so existing launch paths keep working.

```bash
git submodule update --init third_party/HunyuanWorld-Mirror-rocm
```

Primus launches that checkout's Lightning trainer with `train posttrain` for both stages. Stage 1 starts from the public Hugging Face weights and trains the geometry heads. Stage 2 loads the stage-1 checkpoint, freezes the geometry backbone, and trains the Gaussian head.

These pieces stay in Primus and are applied in memory when training starts:

- Official stage configs in `primus/backends/worldmirror/hydra_configs/train/stage1.yaml` and `stage2.yaml`, plus the Hypersim-only configs next to them. Hydra's search path selects the Primus copy.
- aiter flash attention on the geometry transformer, with SDPA as the fallback
- a float32 cast before `gsplat` rasterization
- a boolean depth-confidence mask in the render depth loss
- a scalar aspect-ratio sample, so the batch sampler works on NumPy 2

Before training, the World Mirror install script installs the packages in `runner/helpers/hooks/train/posttrain/worldmirror/requirements-worldmirror.txt` and builds `gsplat` when it is not already installed. The build copies `third_party/HunyuanWorld-Mirror-rocm/submodules/gsplat` to a temporary directory, applies the ROCm compile fixes there, and fetches the GLM headers. The submodule checkout is left unchanged. The first launch compiles `gsplat`, which takes several minutes. Set `PYTORCH_ROCM_ARCH` (for example `gfx950` on MI355X) to build for one GPU architecture.

Data parallel and bf16 stay as World Mirror defines them.

```bash
export HYPERSIM_DIR=/shared_nfs/3dwm/hywm-3dwm-sub20/hypersim
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
```

Official stage 1 and stage 2 use the upstream recipe (`stage1_wogs`, `stage2_wgs`). Stage 1 validation expects NRGBD, DTU, 7Scenes, Re10K, NYUv2, ScanNet, Sintel, and KITTI. iBims is left out because its normals are EXR files and this image's OpenCV cannot read them. Stage 2 validation is Hypersim only. Re10K novel-view chunks and the DL3DV 10K test scenes are not on the mounted data. Primus sets `paths.hypersim_dir` only. The other roots stay at the placeholders in the submodule's `training/configs/paths/default.yaml` until those datasets are mounted.

```bash
MAX_STEPS=20 \
./primus-cli direct -- train posttrain \
  --config examples/worldmirror/configs/MI355X/stage1-posttrain.yaml
```

```bash
PRETRAINED_CKPT=./output/worldmirror-stage1/logs/stage1_wogs/checkpoints/last.ckpt \
MAX_STEPS=20 \
./primus-cli direct -- train posttrain \
  --config examples/worldmirror/configs/MI355X/stage2-posttrain.yaml
```

Hypersim-only stage 1, which skips the extra validation sets:

```bash
MAX_STEPS=20 \
./primus-cli direct -- train posttrain \
  --config examples/worldmirror/configs/MI355X/stage1-hypersim-posttrain.yaml
```

Stage 2, after stage 1 has written `last.ckpt`:

```bash
PRETRAINED_CKPT=./output/worldmirror-stage1/logs/stage1_hypersim/checkpoints/last.ckpt \
MAX_STEPS=20 \
./primus-cli direct -- train posttrain \
  --config examples/worldmirror/configs/MI355X/stage2-hypersim-posttrain.yaml
```

`PRETRAINED_CKPT` loads model weights only. Resuming the stage-1 optimizer fails because stage 2 freezes the backbone and the optimizer groups no longer match.
