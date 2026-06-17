# Wan Examples

Wan examples exercise the independent PyTorch Diffusion backend under
`primus/backends/diffusion`. For backend details, data/checkpoint layout, and minimal
configs, see `primus/backends/diffusion/README.md`.

## Data

The default smoke-test dataset is `zirui3/tiny-video-samples` on Hugging Face:

```bash
huggingface-cli download zirui3/tiny-video-samples \
  --repo-type dataset \
  --local-dir /data/tiny-video-samples
```

Expected layout:

```text
/data/tiny-video-samples/
  meta.jsonl
  data/*.mp4
```

## Run

Launch an MI355X example config with 8 GPUs:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-pretrain.yaml
```

Override common paths and runtime options with environment variables:

```bash
DATASET_PATH=/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/data/tiny-video-samples/data \
ATTENTION_BACKEND=sdpa \
SP_SIZE=1 \
MAX_STEPS=3 \
torchrun --standalone --nproc_per_node=8 \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-pretrain.yaml
```

Use `SP_SIZE=4` or `SP_SIZE=8` to enable Ulysses sequence parallelism
when the model head count supports it.

Post-train examples use the same public override shape under `modules.post_trainer`:

```bash
INIT_CHECKPOINT=/models/Wan2.2-TI2V-5B \
DATASET_PATH=/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/data/tiny-video-samples/data \
MAX_STEPS=3 \
torchrun --standalone --nproc_per_node=8 \
  -m primus.cli.main train posttrain \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-posttrain.yaml
```

The MI355X configs use Primus-style override sections such as `training`,
`data`, `parallelism`, `optimizer`, `runtime`, and `metrics`. The diffusion
adapter normalizes those sections into the Wan model/dataset/trainer
arguments at runtime.
