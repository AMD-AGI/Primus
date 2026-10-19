# Inkling on the Megatron backend

Pretraining support for the text backbone of Thinking Machines' Inkling
(`thinkingmachines/Inkling-Small` and `Inkling`, the Large model) in Primus,
as `model_type: inkling`. The numerical reference is the official
transformers `models/inkling` code (commit `263c5eb9`); Miles is the design
reference. Multi-token prediction, vision and audio are not included.

How the single-GPU parity tests are run, and their last results, are in
[tests/unit_tests/backends/megatron/inkling/README.md](../../../../../../tests/unit_tests/backends/megatron/inkling/README.md).

## The model

| | Inkling-Small | Inkling (Large) |
| --- | --- | --- |
| Text parameters (total / active) | 263.7B / 12.0B | 947.0B / 41.1B |
| Layers (global attention) | 42 (7) | 66 (11) |
| Hidden size | 4096 | 6144 |
| Attention heads; KV groups sliding / global | 32; 8 / 8 | 64; 16 / 8 |
| Dense layers, then MoE layers | 2, 40 | 2, 64 |
| Routed experts (top-k), shared experts | 256 (6), 2 | 256 (6), 2 |

Every layer is attention then MLP, each behind an RMSNorm. What differs from
a Megatron GPT layer:

- **Attention.** Five sliding-window layers (512 tokens) for every global
  one. Each layer adds a learned relative-position bias, computed from
  per-head relative states `r` and a learned `rel_proj`, over a fixed extent
  (512 sliding, 1024 global). Global layers also scale queries
  logarithmically past `log_scaling_n_floor` tokens. q and k are RMS-normed.
  There is no RoPE.
- **ShortConv.** Four depthwise causal convolutions per layer (kernel 4, run in
  fp32): on K, on V, on the attention output and on the MLP output.
- **MoE.** Sigmoid router with an expert-score correction bias (no auxiliary
  loss). Two shared experts are routed by their own "sink" gate rows rather
  than always on, and a learned `global_scale` scales the output.
- **Embedding and head.** RMSNorm on the embeddings. The LM head is untied
  and its input is divided by the muP width multiplier.

## Code layout

| Path | Contents |
| --- | --- |
| `primus/backends/megatron/core/models/inkling/` | `InklingModel`, the config class (`InklingTransformerConfig`, per-layer geometry), layer specs, builders, and the release ↔ Megatron weight mapping |
| `primus/backends/megatron/core/transformer/inkling/` | Attention (eager and FlexAttention backends), ShortConv, router, MoE layer and shared experts, dense MLP, decoder layer |
| `primus/backends/megatron/patches/inkling_muon_qkv_patches.py` | Per-layer q/k/v split for Megatron's Muon (see [Muon](#muon)) |
| `primus/configs/models/megatron/inkling_*.yaml` | Presets: `inkling_small`, `inkling_large`, the proxies `inkling_small_12L` and `inkling_large_6L` (released widths, fewer layers), and `inkling_small_tiny` for tests |
| `examples/megatron/configs/MI355X/inkling_small_12L-BF16-pretrain.yaml` | Single-node pretraining recipe (EP 8, AdamW, mock data) |
| `tools/checkpoint_conversion/convert_inkling_hf_megatron.py` | Hugging Face → Megatron checkpoint converter |
| `tests/unit_tests/backends/megatron/inkling/` | Unit tests, the transformers reference harness, and the 8-GPU parallel check |

The model is wired into `get_model_provider` (`primus/core/utils/import_utils.py`)
and `megatron_pretrain_trainer.py`, which reuses `pretrain_gpt`'s forward step
and datasets.

### Parameter layout

Megatron names map one-to-one onto the released checkpoint, apart from these
merges and splits:

| Megatron | Release |
| --- | --- |
| `self_attention.linear_qkv.weight` | `attn.wq_du`, `attn.wk_dv`, `attn.wv_dv`, in Megatron's grouped `[q heads of group, k, v]` row order |
| `self_attention.linear_rel.weight` | `attn.wr_du` (the relative states `r`, kept out of `linear_qkv` so Muon's q/k/v split applies) |
| `mlp.router.weight`, `mlp.router.shared_weight` | `mlp.gate.weight`, split into routed rows and shared-expert sink rows |
| `mlp.experts.linear_fc{1,2}.weight{e}` | `mlp.experts.w13_weight`, `w2_weight` (grouped GEMM, gate and up de-interleaved) |
| `mlp.router.expert_bias` (buffer) | `mlp.gate.bias` |

`inkling_hf_mapping.py` owns these conversions in both directions; the
converter and the tests both use it.

## Running

Pretraining on one MI355X node with the 12-layer Small proxy, mock data and
AdamW:

```bash
export PYTHONDONTWRITEBYTECODE=1
./runner/primus-cli direct -- train pretrain \
    --config examples/megatron/configs/MI355X/inkling_small_12L-BF16-pretrain.yaml
```

The recipe takes these environment overrides:

| Variable | Default | Effect |
| --- | --- | --- |
| `PRIMUS_MODEL` | `inkling_small_12L` | Model preset, e.g. `inkling_large_6L` |
| `PRIMUS_INKLING_ATTN` | `flex` | `flex` or `eager` (dense bias; slower, about 2x the step time and more memory) |
| `PRIMUS_TRAIN_ITERS` | `50` | Training steps |
| `PRIMUS_SEQ_LENGTH`, `PRIMUS_GBS`, `PRIMUS_MBS`, `PRIMUS_EP` | 4096, 64, 1, 8 | Sequence length, global and micro batch, expert parallelism |

Other Megatron arguments can be appended after `--config` (for example
`--optimizer muon`); later ones win.

Mock data uses `NullTokenizer` with `vocab_size: 200057`, whose end-of-document
token lands on id 200057, giving the 200,058 real Inkling tokens. The HF
tokenizer has no eos/bos/pad token, so real data needs an explicit
end-of-document id first.

### Parallelism

- **Expert and data parallelism** are used by the recipe and are covered by the
  8-GPU parallel check.
- **Pipeline parallelism**, with or without virtual (interleaved) stages, is
  covered by the 8-GPU parallel check. It has not been used for a training
  run yet.
- **Tensor parallelism** has not been run. **Sequence parallelism** and
  **context parallelism** are rejected: the output ShortConvs would need a
  3-token halo exchange across ranks, which isn't written.

### Converting a released checkpoint

```bash
python tools/checkpoint_conversion/convert_inkling_hf_megatron.py \
    --hf-dir /data/Inkling-Small \
    --model-yaml primus/configs/models/megatron/inkling_small.yaml \
    --output /data/inkling_small_megatron
```

The embedding and LM head are resized to the vocabulary that training builds:
the 200,058 real tokens rounded up to a multiple of
`make_vocab_size_divisible_by` × TP, which is 200,064 with the defaults (128 × 1).
If the training run uses other values, pass the same ones with
`--make-vocab-size-divisible-by` and `--tensor-model-parallel-size`, or set
`--target-vocab-size` directly; otherwise loading fails on the embedding and
LM-head shapes.

`--dry-run` maps every tensor from the safetensors headers alone and writes
nothing. The output is a single-rank (TP = PP = EP = 1) `torch` checkpoint; load
it with `--load <output> --ckpt-format torch --finetune --no-load-optim
--no-load-rng`. A resharded `torch_dist` checkpoint for multi-node runs is not
written yet.

### Muon

Megatron's Muon (Newton–Schulz on 2-D weights, Adam for the rest) runs on
both sizes:

```bash
export PRIMUS_INSTALL_EMERGING_OPTIMIZERS=1   # the v26.8 image doesn't ship Emerging-Optimizers
./runner/primus-cli direct -- train pretrain \
    --config examples/megatron/configs/MI355X/inkling_small_12L-BF16-pretrain.yaml \
    --optimizer dist_muon --muon_extra_scale_factor 1.0 \
    --overlap_grad_reduce true --overlap_param_gather false
```

- Megatron splits `linear_qkv` into q, k and v using one shape for the whole
  model, but the sliding and global layers of Inkling Large have different KV
  group counts (16 and 8), so the stock split fails. The
  `megatron.optimizer.inkling_muon_qkv` patch gives each `linear_qkv` weight
  its own layer's split. Small has 8 groups on both, so it is unaffected.
- On Muon: `linear_qkv`, `linear_rel`, `linear_proj`, `rel_proj`, dense and
  expert MLP weights, shared experts, and the router's `weight` and
  `shared_weight`. On Adam: embedding, LM head, ShortConvs (3-D), norms and
  `global_scale`.
- `dist_muon` with `--overlap_param_gather true` fails at the second step when
  every expert lives on one rank (EP = number of GPUs): Megatron's layer-wise
  optimizer leaves the expert buckets' sizes unset. Keep parameter-gather
  overlap off for now.

## Testing

### Single GPU

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/eo:/tmp/inkling_ref/site \
    python -m pytest -q tests/unit_tests/backends/megatron/inkling
```

The tests that compare against transformers skip unless the pinned
transformers build is on `PYTHONPATH`, and the Muon tests skip without
Emerging-Optimizers. Installing both, and fetching the real Inkling-Small
layers for the real-weight test, are described in the tests README. With the reference, the fp32 tiny model must match its logits and loss
to 1e-5, and every gradient to 1e-5 of that tensor's largest value. That is
checked on both attention backends, and on the released layers 0–5 of
Inkling-Small. bf16 must stay as close to the fp32 reference as transformers'
own bf16 run is.

### 8 GPUs: parallel layouts against the reference

```bash
PYTHONPATH=/tmp/inkling_ref/site torchrun --nproc-per-node 8 \
    tests/unit_tests/backends/megatron/inkling/parallel_reference_check.py \
    [--layouts dp8 ep8 pp2_ep4 ...] [--backends eager flex]
```

An 8-layer fp32 Inkling (dense layer 0, MoE layers 1–7, global attention on
layers 3 and 7) takes one step through Megatron's pipeline schedules on a fixed
16 × 256-token batch. Each rank loads its slice of the reference weights. The
global loss and every gradient (summed over data-parallel ranks) are compared
with the reference run in fp64 on the full batch. Each gradient must be within
1e-5 of fp64, relative to that tensor's largest value. The exception is the
one-element `global_scale` gradients, which must be within 1e-4. Each is a sum
of thousands of cancelling terms, so its fp32 rounding depends on the data and
on the number of microbatches; the reference's own fp32 run is off by 1e-6 to
4e-5 on them.

| Layout | PP | Virtual stages | EP | DP |
| --- | --- | --- | --- | --- |
| `dp8` | 1 | 1 | 1 | 8 |
| `ep8` | 1 | 1 | 8 | 8 |
| `pp2_ep4` | 2 | 1 | 4 | 4 |
| `pp4_ep2` | 4 | 1 | 2 | 2 |
| `pp8` | 8 | 1 | 1 | 1 |
| `pp2_vpp2_ep4` | 2 | 2 | 4 | 4 |
| `pp4_vpp2_ep2` | 4 | 2 | 2 | 2 |

TP stays at 1 in every layout. The script exits non-zero if any layout fails.

## Known limitations

- FlexAttention global layers are about 10× slower than causal attention
  without a bias, which makes them the main step-time cost.
- On torch 2.14, flex compiled with `dynamic=False` can pick a decode kernel
  for sequences of 128 tokens or fewer, and that kernel rejects the block mask.
  It hits the 96-token unit tests in some orders. Training lengths are
  unaffected.
- No `sharded_state_dict` yet, so distributed checkpoint save and restore are
  untested. No Inkling FLOPs formula, so logged TFLOP/s use the GPT formula.
- Expert load balance and the bias update have not been verified with
  per-expert load logging.
