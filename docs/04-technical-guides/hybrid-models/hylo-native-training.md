# Native HyLo 1B training

Primus can reproduce the public HyLo two-stage training procedure without the
reference TRL trainer:

1. Enhanced-ILD trains an all-GDN student and an all-MLA student from
   `meta-llama/Llama-3.2-1B-Instruct` at 2K context.
2. ILD weights are assembled into the exact published block layout (MLA blocks
   1, 5, 10, and 14).
3. The assembled 4MLA12GDN student is trained at 8K against
   `meta-llama/Llama-3.1-8B-Instruct` with pure
   `KL(teacher || student)` (`KL=1`, `CE=0`) and YaRN factor 4.

The runtime currently supports TP=1 and PP=1. It is intended for eight-way data
parallel training on MI300X with `rocm/primus:v26.8`.

## Prepare public assets

Accept access to both gated Meta Llama repositories, then keep the token in the
environment only:

```bash
export HF_TOKEN=...
docker run --rm --entrypoint bash \
  --device /dev/kfd --device /dev/dri --group-add video \
  -e HF_TOKEN \
  -v "$PWD:/workspace/src" -w /workspace/src \
  rocm/primus:v26.8 -lc \
  'python tools/hybrid/prepare_hylo_assets.py --root /workspace/src/data/hylo'
```

This downloads and converts the 1B backbone and 8B teacher to TP=PP=1 Megatron
checkpoints and exports `JunxiongWang/sftdatasetv3` as JSONL. The token is not
written to disk.

The public HyLo repository notes that the released stage-2 run also used four
AMD-preprocessed dataset variants that are not published there. Consequently,
training with `sftdatasetv3` alone reproduces the code path and public-data
objective, but cannot be expected to reproduce the released weights bit for
bit.

## Create from-teacher initial checkpoints

For each all-GDN, all-MLA, and target 4MLA12GDN layout, first save a
constructor checkpoint from `hylo_gdn_1B_hybrid.yaml` with `lr: 0`,
`train_iters: 1`, `save_interval: 1`, and the corresponding
`hybrid_override_pattern`. Upcycle it from the converted 1B checkpoint:

```bash
python tools/hybrid/upcycle_dense_to_hybrid.py \
  --dense-checkpoint data/hylo/megatron/llama32-1b \
  --hybrid-init-checkpoint /path/to/constructor-checkpoint \
  --output-dir /path/to/upcycled-checkpoint \
  --hybrid-pattern 'M-*-M-M-M-*-M-M-M-M-*-M-M-M-*-M-'
```

Use `M-M-M-M-M-M-M-M-M-M-M-M-M-M-M-M-` for all-GDN and
`*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-` for all-MLA.

## Run both stages

```bash
export HYLO_LLAMA32_1B_MEGATRON_CKPT="$PWD/data/hylo/megatron/llama32-1b"
export HYLO_LLAMA31_8B_MEGATRON_CKPT="$PWD/data/hylo/megatron/llama31-8b"
export HYLO_SFT_DATASET="$PWD/data/hylo/hylo-sft.jsonl"
export HYLO_GDN_UPCYCLED_CKPT=/path/to/all-gdn-upcycled
export HYLO_MLA_UPCYCLED_CKPT=/path/to/all-mla-upcycled
export HYLO_STAGE2_BASE_CKPT=/path/to/4mla12gdn-upcycled

tools/hybrid/run_hylo_1b_4mla12gdn.sh
```

The stage recipes are under `examples/megatron/configs/MI300X/`. Set
`HYLO_ILD_TRAIN_ITERS=1` and `HYLO_SFT_TRAIN_ITERS=1` for a smoke run.

File-by-file notes for this change are in
[hylo-native-training-changes.md](hylo-native-training-changes.md).
