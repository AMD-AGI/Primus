# Native HyLo stage workflow

This page describes the artifacts and execution order for the native Primus
reproduction of `amd/HyLo-Llama-4MLA12GDN-8K-SFT`. For setup and public-data
limitations, see [Native HyLo 1B training](hylo-native-training.md).

## Inputs

- `meta-llama/Llama-3.2-1B-Instruct`, converted to a TP=PP=1 Megatron
  checkpoint, is the stage-1 teacher and the source of the student weights.
- `meta-llama/Llama-3.1-8B-Instruct`, also converted to TP=PP=1, is the
  stage-2 teacher.
- `JunxiongWang/sftdatasetv3` is the public training set.
- `rocm/primus:v26.8` is the validated runtime image.

Run `tools/hybrid/prepare_hylo_assets.py` once to prepare those inputs. Pass the
Hugging Face token through `HF_TOKEN`; the script does not write it to disk.

## Initialization

Before either training stage, save three zero-learning-rate constructor
checkpoints and overlay the Llama-3.2-1B weights:

1. all-GDN: `M-M-M-M-M-M-M-M-M-M-M-M-M-M-M-M-`
2. all-MLA: `*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-`
3. target 4MLA12GDN: `M-*-M-M-M-*-M-M-M-M-*-M-M-M-*-M-`

The constructor supplies parameters that do not have a dense-teacher mapping.
The upcycler copies the embedding, MLP, norm, MLA SVD factors, and HyLo GDN
Q/K/V/O slices. The resulting all-GDN and all-MLA checkpoints initialize
stage 1. The target checkpoint is the base used when stage-1 outputs are
assembled.

## Stage 1: Enhanced-ILD

Stage 1 runs two independent students against the frozen 1B teacher:

- `hylo_gdn_1B-enhanced-ild.yaml` trains an all-GDN student.
- `hylo_mla_1B-enhanced-ild.yaml` trains an all-MLA student.

For each dense layer, the student mixer receives the matching teacher layer
input. The loss is the sum of the layer-output L2 distance and mixer-output L2
distance. These recipes use 4K context. The public reference uses 2K. Both use
BF16, learning rate `2e-4`, cosine decay, 1% warmup, and micro-batch 12 per GPU.

After both runs, `tools/hybrid/assemble_hylo_ild.py` starts from the target
constructor checkpoint, copies GDN blocks from the all-GDN result, and copies
MLA blocks 1, 5, 10, and 14 from the all-MLA result. This assembled checkpoint
is the stage-2 student initialization.

## Stage 2: teacher-guided SFT

`hylo_4MLA12GDN_1B-8K-sft.yaml` trains the assembled student against the frozen
Llama-3.1-8B teacher. Its objective is pure
`KL(teacher || student)`: KL weight 1 and CE weight 0. The implementation
computes masked KL in vocabulary chunks to limit memory.

The reference recipe uses 8K context, BF16, learning rate `6e-5`, cosine decay,
1% warmup, micro-batch 4 per GPU, and YaRN factor 4 over the original 2K
window.

## Orchestration

Set the converted checkpoints and invoke the orchestration script:

```bash
export HYLO_LLAMA32_1B_MEGATRON_CKPT=/path/to/llama32-1b
export HYLO_LLAMA31_8B_MEGATRON_CKPT=/path/to/llama31-8b
export HYLO_SFT_DATASET=JunxiongWang/sftdatasetv3
export HYLO_OUTPUT_ROOT=/path/to/hylo-output
export PRIMUS_IMAGE=rocm/primus:v26.8

tools/hybrid/run_hylo_1b_4mla12gdn.sh
```

The execution order is:

```text
construct and upcycle all-GDN ──> all-GDN Enhanced-ILD ──┐
                                                         ├─> assemble 4MLA12GDN ─> pure-KL SFT
construct and upcycle all-MLA ──> all-MLA Enhanced-ILD ──┘
construct and upcycle target 4MLA12GDN ──────────────────┘
```

The native distillation runtime currently requires TP=1 and PP=1. Data
parallelism may span the eight GPUs on one MI300X node.

## Short validation run

For a bounded integration test, use the one-example fixture, one optimizer step,
4K context for Enhanced-ILD, and 8K context for pure KL:

```bash
export HYLO_SFT_DATASET="$PWD/tests/fixtures/hylo_smoke.jsonl"
export HYLO_ILD_TRAIN_ITERS=1
export HYLO_SFT_TRAIN_ITERS=1
export HYLO_ILD_SEQ_LENGTH=4096
export HYLO_SFT_SEQ_LENGTH=8192
export HYLO_MICRO_BATCH_SIZE=1
export HYLO_GLOBAL_BATCH_SIZE=1
```

The full script still runs both Enhanced-ILD branches, assembles the target
checkpoint, and then runs pure KL. A one-GPU check on `useocpm2m-097-017` in
`rocm/primus:v26.8` completed both 4K ILD steps, assembled them, and finished
one 8K pure-KL step from that checkpoint. `tw044` was not used: SSH to the
jump host timed out.

| Step | Result |
| --- | --- |
| all-GDN Enhanced-ILD, 4K | 360.000000 (hidden 168.000000, mixer 192.000000); iteration 1, 243 tensors, all finite |
| all-MLA Enhanced-ILD, 4K | 169.000000 (hidden 66.000000, mixer 103.000000); iteration 1, 275 tensors, all finite |
| assemble | every mixer/MLP pair matches its ILD source; 251 tensors, all finite |
| pure KL from the assembly, 8K | summed masked KL 72.599213; iteration 1, reload finite |

