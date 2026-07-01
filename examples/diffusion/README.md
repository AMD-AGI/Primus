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

Set the shared `torchrun` environment first:

```bash
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
```

### Pretrain

```bash
DATASET_PATH=/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/data/tiny-video-samples/data \
ATTENTION_BACKEND=sdpa \
SP_SIZE=1 \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-pretrain.yaml
```

Use `SP_SIZE=4` or `SP_SIZE=8` to enable Ulysses sequence parallelism
when the model head count supports it.

### Posttrain

```bash
INIT_CHECKPOINT=/models/Wan2.2-TI2V-5B \
DATASET_PATH=/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/data/tiny-video-samples/data \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train posttrain \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-posttrain.yaml
```

The MI355X configs use Primus-style override sections such as `training`,
`data`, `parallelism`, `optimizer`, `runtime`, and `metrics`. The diffusion
adapter normalizes those sections into the Wan model/dataset/trainer
arguments at runtime.