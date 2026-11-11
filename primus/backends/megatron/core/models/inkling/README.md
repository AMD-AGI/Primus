# Inkling on the Megatron backend

Pretraining support for the text backbone of Thinking Machines' Inkling
(`thinkingmachines/Inkling-Small` and `Inkling`, the Large model) in Primus,
as `model_type: inkling`. The numerical reference is the official
transformers `models/inkling` code (commit `263c5eb9`); Miles is the design
reference. Multi-token prediction, vision and audio are not included.

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
| `primus/configs/models/megatron/inkling_*.yaml` | Presets: `inkling_small`, `inkling_large`, the proxies `inkling_small_12L`, `inkling_large_6L` and `inkling_large_20L` (released widths, fewer layers), and `inkling_small_tiny` for tests |
| `examples/megatron/configs/MI355X/inkling_small_12L-BF16-pretrain.yaml`, `inkling_large_6L-BF16-pretrain.yaml`, `inkling_large_6L-FP8-pretrain.yaml` | Single-node recipes (EP 8, AdamW, mock data) |
| `examples/megatron/configs/MI355X/inkling_small-BF16-pretrain.yaml`, `inkling_large_20L-BF16-pretrain.yaml` | Four-node recipes (PP 4 × EP 8) for the full Small model and the 20-layer Large proxy; not yet run on this code |
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

## Reproducing the single-node results

Needs one node with 8 MI355X GPUs, Docker and internet access, and the
`rocm/primus:v26.9` image (torch 2.16.0a0+rocm10.2, Transformer Engine
2.20.0.dev0, Primus-Turbo 0.6.1.dev4). Mock data needs no Hugging Face token.

**1. Clone Primus** on the host, with its Megatron-LM submodule. The image's
own `/workspace/Primus` is an older copy, so mount your clone instead.

```bash
cd $HOME
git clone https://github.com/AMD-AGI/Primus.git Primus-inkling
cd Primus-inkling
git submodule update --init third_party/Megatron-LM
```

**2. Start the container** on the GPU node and open a shell in your clone.
Stay root: as a normal user, aiter's GPU probe fails.

```bash
docker run -d --name primus_v26.9 \
    --device /dev/dri --device /dev/kfd --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    --shm-size 64G -v $HOME:$HOME \
    rocm/primus:v26.9 sleep infinity
docker exec -it -w $HOME/Primus-inkling primus_v26.9 bash
```

**3. Run the recipes** inside the container:

```bash
export PYTHONDONTWRITEBYTECODE=1            # no root-owned __pycache__ in your clone
export PRIMUS_WORKSPACE=/tmp/inkling_out
export PRIMUS_INSTALL_EMERGING_OPTIMIZERS=1 # Muon needs Emerging-Optimizers; the image lacks it
CFG=examples/megatron/configs/MI355X
SMALL=$CFG/inkling_small_12L-BF16-pretrain.yaml
LARGE=$CFG/inkling_large_6L-BF16-pretrain.yaml
LARGE_FP8=$CFG/inkling_large_6L-FP8-pretrain.yaml
LOGS=/tmp/inkling_logs; mkdir -p $LOGS
MUON="--use_distributed_optimizer false --overlap_param_gather false --muon_extra_scale_factor 1.0"
train() { log=$1; shift; ./runner/primus-cli direct --log_file $LOGS/$log.log -- train pretrain "$@"; }

train small_adamw --config $SMALL
train small_muon  --config $SMALL --optimizer muon --overlap_grad_reduce false $MUON
train large_bf16  --config $LARGE
train large_fp8   --config $LARGE_FP8
PRIMUS_TURBO=false train large_bf16_te   --config $LARGE
PRIMUS_TURBO=false train large_muon      --config $LARGE --optimizer muon --overlap_grad_reduce false $MUON
PRIMUS_TURBO=false train large_dist_muon --config $LARGE --optimizer dist_muon --overlap_grad_reduce true $MUON
```

Each run is 50 steps of 64 × 4096 tokens (EP 8 × DP 8, flex attention, random
initialisation) and takes 4–5 minutes. The first run in a fresh container
spends a few more minutes compiling, and the first FP8 run several more
autotuning Turbo's FP8 kernels.

**4. Read the results.** Step 50 carries the loss, the NaN and skipped-step
counts and `hip mem usage`; the step time is the median over steps 20–50:

```bash
for log in $LOGS/*.log; do
  last=$(grep -a 'iteration  *50/' $log | tail -1)
  loss=$(grep -oE 'lm loss: [0-9.E+]+' <<<"$last")
  nan=$(grep -oE 'nan iterations: +[0-9]+' <<<"$last")
  mem=$(grep -oE 'usage_ratio: [0-9.]+GB' <<<"$last" | cut -d' ' -f2)
  ms=$(grep -aoE 'iteration +([2-4][0-9]|50)/.*per iteration \(ms\): [0-9.]+' $log \
    | grep -oE '[0-9.]+$' | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  echo "$(basename $log .log): $loss, $nan, median $ms ms, hip mem $mem"
done
```

Results on Nov 4, 2026 (no NaN or skipped steps in any run):

| Run | Recipe | Loss at step 50 | Step time | `hip mem usage` |
| --- | --- | --- | --- | --- |
| `small_adamw` | Small 12L (67.5B), Turbo off | 5.762 | 3.59 s | 208 GB |
| `small_muon` | Small 12L, `muon` | 5.846 | 4.49 s | 203 GB |
| `large_bf16` | Large 6L (62.6B), Turbo on | 5.747 | 3.26 s | 201 GB |
| `large_fp8` | Large 6L, Turbo on, FP8 tensorwise | 5.702 | 2.99 s | 222 GB |
| `large_bf16_te` | Large 6L, Turbo off | 5.747 | 3.31 s | 204 GB |
| `large_muon` | Large 6L, Turbo off, `muon` | 5.901 | 4.53 s | 216 GB |
| `large_dist_muon` | Large 6L, Turbo off, `dist_muon` | 5.902 | 4.26 s | 183 GB |

Step times vary by up to 4% between repeats of the same run. Mock data is
random tokens, so these numbers measure throughput and stability, not quality.

## Recipe options

The single-node recipes take these environment overrides; other Megatron
arguments can be appended after `--config` (later ones win).

| Variable | Default | Effect |
| --- | --- | --- |
| `PRIMUS_TURBO` | `true` (Large recipes) | `false` turns off every Turbo kernel in the Large recipes; the Small recipe always runs without Turbo |
| `PRIMUS_MODEL` | `inkling_small_12L` or `inkling_large_6L` | Model preset |
| `PRIMUS_INKLING_ATTN` | `flex` | `flex` or `eager` (dense bias; about 2× the step time and more memory) |
| `PRIMUS_TRAIN_ITERS` | `50` | Training steps; must be more than the 5 warmup steps |
| `PRIMUS_SEQ_LENGTH`, `PRIMUS_GBS`, `PRIMUS_MBS`, `PRIMUS_EP` | 4096, 64, 1, 8 | Sequence length, global and micro batch, expert parallelism |

### Primus-Turbo

The Large recipes turn on Turbo grouped GEMM, RMSNorm, DeepEP and GEMM.
Inkling keeps its own attention core, so `use_turbo_attention` stays off.

- In BF16, Turbo's GEMM and TE's linears call the same hipBLASLt kernels, so
  the 1.5% gain over Turbo off comes from DeepEP. The LM head stays Megatron's
  `ColumnParallelLinear`.
- DeepEP needs fp32 router probabilities (`moe_router_dtype: fp32`, set by the
  presets) and switches the dispatcher to `flex` itself.
  `turbo_deepep_num_cu: 64` is about 1% faster than the default 32.
  `moe_permute_fusion` and `turbo_sync_free_moe_stage: 2` change nothing;
  `turbo_sync_free_moe_stage: 3` and `use_turbo_fused_act_with_probs` are
  slower, so the recipe leaves them off. `moe_use_fused_router_with_aux_score`
  has no effect, because Inkling's router replaces Megatron's routing.
- Turbo grouped GEMM trains each expert linear as one stacked 3-D `weights`
  tensor, exposing the per-expert `weight{e}` as views of it. Muon does not
  orthogonalize 3-D weights, so Primus rejects Muon with Turbo grouped GEMM
  (unless `primus_turbo` isn't installed or the model has no MoE layers);
  run Muon with `PRIMUS_TURBO=false`. Checkpoints hold only the per-expert
  `weight{e}` (`torch_dist` always; `torch` ones written without Turbo), and
  loading them fills the stacked tensor, so checkpoints move freely between
  Turbo and non-Turbo runs.

### FP8

`inkling_large_6L-FP8-pretrain.yaml` is the Large recipe with FP8 tensorwise
GEMMs. It needs Primus-Turbo 0.6 or later (`rocm/primus:v26.9`): the FP8
grouped GEMM in Turbo 0.5 (`rocm/primus:v26.8`) fails on Inkling's expert
shapes. The LM head stays in BF16, as in Megatron's GPT model.

Other recipes, measured the same way on `rocm/primus:v26.9`:

- TE's FP8 without Turbo (`PRIMUS_TURBO=false`): 3.23 s per step.
- FP8 blockwise: 3.29 s, no faster than BF16.
- MXFP8 (`fp8: e4m3`, `fp8_recipe: mxfp8`, `NVTE_ROCM_ENABLE_MXFP8=1`): 3.00 s,
  as fast as tensorwise but with about 17 minutes of first-step autotuning.
- Adding `turbo_fused_grouped_gemm`, `use_turbo_fused_act_with_probs` and
  `moe_permute_fusion`: about 5 s.

Every FP8 run ends about 0.045 (tensorwise) or 0.03 (blockwise, MXFP8) below
BF16 at step 50, starting from step 3, whether the FP8 GEMMs are TE's or
Turbo's. Qwen3-30B-A3B with the same settings ends at the same loss in BF16
and FP8, so the gap comes from Inkling's layers in FP8; its cause is not known
yet.

### Muon

Megatron's Muon (Newton–Schulz on 2-D weights, Adam for the rest) runs on both
sizes, with the flags in `$MUON` above.

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

## Training on real data

Mock data replaces the tokenizer with `NullTokenizer`; `vocab_size: 200057`
plus its end-of-document token gives the 200,058 real Inkling tokens. Real
data needs the real tokenizer with an end-of-document token. The released
tokenizer defines no eos, bos or pad token, so tokenize with `<|endoftext|>`
(id 199999) as the separator; this also saves a copy of the tokenizer with that
token as its `eos_token`:

```bash
python tools/convergence_test/prepare_dataset.py --model inkling_small.yaml \
    --eod-token '<|endoftext|>' --out-dir /data/inkling_fineweb
```

Then train from that directory instead of mock data:

```yaml
      mock_data: false
      tokenizer_type: HuggingFaceTokenizer
      tokenizer_model: /data/inkling_fineweb/tokenizer
      train_data_path: ["/data/inkling_fineweb/train_text_document"]
      valid_data_path: ["/data/inkling_fineweb/valid_text_document"]
```

Megatron then reports the 200,058-token vocabulary (padded to 200,064) and
end-of-document id 199999.

## Converting a released checkpoint

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

The output is a `torch_dist` checkpoint in `<output>/release`. Megatron
reshards it on load, so any layout (PP, virtual stages, EP, DP, with or without
Turbo) can start from it:

```bash
--load /data/inkling_small_megatron --ckpt_format torch_dist --finetune --no_load_optim --no_load_rng
```

`--ckpt-format torch` writes the older single-rank (TP = PP = EP = 1) file
instead, which only a single-rank run can load. `--dry-run` maps every tensor
from the safetensors headers alone and writes nothing. Every mode needs one
visible GPU (Transformer Engine layers will not build without one), but the
model is built on the meta device, so no GPU memory is allocated. The converted
weights sit in host memory until they are written: 2 bytes per parameter in
bf16.

## Saving and resuming

The recipes save `torch_dist` checkpoints (`ckpt_format: torch_dist`); set
`save` and `save_interval` to write them and `load` to resume. Primus writes
them under `<workspace>/<work_group>/<user_name>/<exp_name>/checkpoints`
whatever path `save` names. Megatron keys
every tensor by its global layer and expert number, so a run can resume under a
different PP, EP or DP size, or with Turbo switched on or off. Changing PP or EP
also needs the optimizer state saved fully reshardable:

```yaml
      dist_ckpt_optim_fully_reshardable: true
```

With the default (`dp_reshardable`), only the DP size may change between save
and resume.

On Nov 9, 2026 (`rocm/primus:v26.9`, one node), the 6L recipe on a 4-layer
model (`PRIMUS_MODEL=inkling_small_tiny --num_layers 4 --moe_layer_freq
"([0]*1+[1]*3)"`, Turbo on, EP 8) saved at step 10 of 20 and was resumed three
ways:

| Resume | Steps 13–20 vs. the uninterrupted run |
| --- | --- |
| Same layout | identical |
| PP 2 × EP 4, Turbo on | within 6e-4 |
| PP 2 × EP 4, Turbo off | within 6e-4 |

The PP 2 runs differ only by summation order. With the same layout, step 11 is
identical too. Step 12's logged loss differs in every resume (by 3e-3 with the
same layout), yet steps 13–20 then match exactly, so the training state is
unchanged and only that logged value differs.

## Parallelism

- **Expert and data parallelism** are used by the single-node recipes and are
  covered by the 8-GPU parallel check.
- **Pipeline parallelism**, with or without virtual (interleaved) stages, is
  covered by the 8-GPU parallel check. The four-node recipes use PP 4 but have
  not been run on this code yet.
- **Tensor parallelism** has not been run. **Sequence parallelism** and
  **context parallelism** are rejected: the output ShortConvs would need a
  3-token halo exchange across ranks, which isn't written.

## Testing

### Single GPU

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/eo:/tmp/inkling_ref/site \
    python -m pytest -q tests/unit_tests/backends/megatron/inkling
```

The tests that compare against transformers skip unless the pinned
transformers build is on `PYTHONPATH`, and the Muon tests skip without
Emerging-Optimizers. Installing both, and fetching the real Inkling-Small
layers for the real-weight test, are described in
[the tests README](../../../../../../tests/unit_tests/backends/megatron/inkling/README.md).
With the reference, the fp32 tiny model must match its logits and loss to
1e-5, and every gradient to 1e-5 of that tensor's largest value, on both
attention backends and on the released layers 0–5 of Inkling-Small. bf16 must
stay as close to the fp32 reference as transformers' own bf16 run is.

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
- No Inkling FLOPs formula, so logged TFLOP/s use the GPT formula.
- Checkpoint save and resume have been run on one node only, on a 4-layer
  model; the four-node recipes have not saved or resumed yet. Checkpoints
  assume TP 1: the K/V short convs are saved as replicated tensors, not TP
  shards.
- Expert load balance and the bias update have not been verified with
  per-expert load logging.
- FP8 lowers the loss by about 0.045 at 50 steps for reasons not yet known
  (see [FP8](#fp8)).
