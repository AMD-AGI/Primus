# Native HyLo 1B training

Primus can reproduce the public HyLo two-stage training procedure without the
reference TRL trainer:

1. Enhanced-ILD trains an all-GDN student and an all-MLA student from
   `meta-llama/Llama-3.2-1B-Instruct` at 4K context. The public reference
   config uses 2K; this recipe starts at 4K.
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
`*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-` for all-MLA. The all-MLA constructor is
`hylo_mla_1B-constructor.yaml`, which has the grouped key/value heads that
section [All-MLA Enhanced-ILD](#all-mla-enhanced-ild) describes.

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

## Enhanced-ILD loss

The loss follows the reference `kd_trainer.py`. For each of the 16 blocks it
adds the mean L2 distance between student and teacher block outputs and
between their token-mixer outputs. The last block's outputs are compared after
each model's final norm, as in Hugging Face's `hidden_states[-1]`, so the
student's final norm trains in this stage.

The loss is already a per-token mean, so each micro-batch counts once and
Megatron does not divide it by the token count again. The logged `hylo ild`
is the mean over data-parallel ranks, the value the reference logs.

## All-MLA Enhanced-ILD

`hylo_mla_1B-enhanced-ild.yaml` follows Zebra-Llama's
`configs/llama3.2_1B/zebra_MLA_ILD.yaml` in AMD-Hybrid-Models (`feat/HyLo`):

- The MLA keeps Llama's 8 key/value heads (`use_full_kv_head: false`).
  `hylo_grouped_mla_hybrid_stack_spec` uses `GroupedKVUpProjection`: its
  `linear_kv_up_proj.weight` has one 96-row block per key/value head, 768 rows
  in place of 3072. Each block's output feeds 4 consecutive query heads, as
  `repeat_kv` does. `hylo_mla_kv_heads` sets the count. The upcycler reads the
  head count from that weight, so `re_init_kv` factors the 8 teacher heads
  without repeating them, like the reference.
- The student runs YaRN with factor 32 over 8192 positions, and the teacher
  keeps its own Llama-3 rope scaling (`hylo_teacher_use_rope_scaling: true`):
  Zebra-Llama's `train_distill.py` builds the teacher from its own config.
  HyLo's newer copy gives the teacher the student's YaRN
  (`hylo_teacher_yarn_factor`), which the reference stage-1 checkpoints were
  not trained with: against that teacher the reference all-MLA model's
  Enhanced-ILD loss rises from 29 to 70.
- `seed: 42`, `weight_decay: 0.0`, and `adam_beta2: 0.999` are the Hugging Face
  Trainer values, as in the all-Mamba-2 recipe.

Build the init checkpoint with `hylo_mla_1B-constructor.yaml`, upcycle it with
`--hybrid-pattern '*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-'`, then train with
`HYLO_MLA_UPCYCLED_CKPT` set.

The released 4MLA12GDN model also keeps 8 key/value heads, but
`hylo_4MLA12GDN_1B-8K-sft.yaml` and `hylo_1B-constructor.yaml` still use
Megatron's per-head MLA. Assembly rejects the grouped all-MLA weights for that
target until stage 2 moves to the grouped spec.

## All-Mamba-2 Enhanced-ILD

`hylo_m2_1B-enhanced-ild.yaml` is the Primus form of Zebra-Llama's
`configs/llama3.2_1B/zebra_M2_ILD.yaml` in AMD-Hybrid-Models (`feat/HyLo`).
All 16 Llama-3.2-1B attention blocks become Mamba-2 and train with the same
Enhanced-ILD objective as the GDN and MLA recipes. It is a stage-1 recipe on
its own; the 4MLA12GDN pipeline above does not use it.

The mixer is MambaInLlama's grouped Mamba-2 (`Mamba2DxbMixer` in
`hylo_m2_hybrid_stack_spec`, widths in `hylo_m2_1B_hybrid.yaml`). x and B are
projected to the 8 key/value groups and each group feeds 4 consecutive heads,
the way GQA shares keys and values:

| Part | Width |
| --- | --- |
| `in_proj` rows for z, x, B, C, dt | 2048, 512, 512, 2048, 32 |
| Causal convolution over x, B, C | 3072 channels, kernel 4, SiLU |
| SSM heads | 32 heads of 64, state 64 |
| Gated RMSNorm | 32 groups of 64, gate applied before the norm |
| Scan chunk | 256 |

The upcycler applies HyLo's `init_with_kqvo`: V goes into x, K into B, Q into
C, `o_proj` into `out_proj`, and the attention input norm into the mixer's
fused input norm. z, dt, the convolution, `A_log`, `D`, and `dt_bias` keep the
constructor's init, which follows HyLo's: `nn.Linear` and `nn.Conv1d` defaults,
`A = U(1, 16)`, `D = 1`, and `softplus(dt_bias)` log-uniform in [0.001, 0.1].

The teacher keeps its own Llama-3 rope scaling, as in the GDN and MLA recipes.
`seed: 42`, `weight_decay: 0.0`, and `adam_beta2: 0.999` differ from the GDN
recipe: they are the Hugging Face Trainer values the reference trained with,
in place of the SFT trainer defaults (1234, 0.1, and 0.95).

Build the init checkpoint, then train:

```bash
PRIMUS_EXP_NAME=hylo-m2-1b-constructor ./primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/hylo_m2_1B-constructor.yaml
python tools/hybrid/upcycle_dense_to_hybrid.py \
  --dense-checkpoint data/hylo/megatron/llama32-1b \
  --hybrid-init-checkpoint output/amd/root/hylo-m2-1b-constructor/checkpoints \
  --output-dir /path/to/all-m2-upcycled \
  --hybrid-pattern 'M-M-M-M-M-M-M-M-M-M-M-M-M-M-M-M-'
HYLO_M2_UPCYCLED_CKPT=/path/to/all-m2-upcycled ./primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI300X/hylo_m2_1B-enhanced-ild.yaml
```

The mixer needs `mamba-ssm`, uses `causal-conv1d` when it is installed, and
supports training with TP=1 and CP=1 only.

File-by-file notes for this change are in
[hylo-native-training-changes.md](hylo-native-training-changes.md).
The artifact flow and execution order are summarized separately in
[hylo-stage-workflow.md](hylo-stage-workflow.md).
