# Dense-to-hybrid upcycling — change notes

This note records what was added for non-MoE hybrid upcycling and why. The user-facing procedure is in [README.md](README.md#upcycling-a-dense-checkpoint).

## Why this exists

Continued pretraining of a Zebra hybrid (attention + Mamba, GDN, or KDA, plus MLP) should be able to start from a dense Transformer checkpoint. Megatron already has `--moe-use-upcycling`, which duplicates one dense MLP into experts. That path does not apply here: a hybrid stack is a different module layout, and a new recurrent mixer does not have a dense QKV tensor to copy.

Primus hybrid stacks are a flat list of sublayers. `HybridStack.allocate_layers` pairs every sequence mixer with its own MLP, so a dense model of `N` blocks becomes `2N` sublayers:

```text
dense layer i  ->  hybrid sublayers (2i, 2i + 1) = (mixer, MLP)
```

Pattern symbols are the ones Megatron already uses: `*` attention, `M` recurrent mixer, `-` MLP.

## Files

| File | Why it changed |
| --- | --- |
| `primus/backends/megatron/checkpoint/hybrid_upcycle.py` | Library that plans the pattern, copies matching tensors onto a hybrid init checkpoint, and writes a legacy Megatron checkpoint. Lives next to the other native checkpoint converters so tests can import it without a `sys.path` hack. |
| `tools/hybrid/upcycle_dense_to_hybrid.py` | CLI. Inserts the repo root on `sys.path` and calls the library. Same entry-point style as the other `tools/hybrid` converters. |
| `tests/unit_tests/backends/megatron/test_hybrid_upcycle.py` | CPU tests for pattern allocation, MoE rejection, MLP remapping, attention shape mismatches, fused-norm aliases, and the checkpoint envelope. |
| `docs/04-technical-guides/hybrid-models/README.md` | Short procedure for producing the hybrid init checkpoint, running the converter, and loading the result. |
| `docs/04-technical-guides/hybrid-models/upcycling-changes.md` | This note. |

No Megatron-LM submodule files were edited. The converter is an offline checkpoint rewrite, so it does not hook `pretrain()`.

## Copy rules, and why

The hybrid init checkpoint is the base. The converter overlays dense tensors and leaves everything else alone. That is required because Mamba, GDN, and KDA parameters (`A_log`, `dt_bias`, `conv1d`, fused `in_proj`) have architecture-specific initializers. Inventing those shapes in the converter would drift from the real module.

What is copied:

- `embedding.word_embeddings.weight`
- `output_layer.weight`, when the hybrid model unties it
- the final norm, accepting either `decoder.final_layernorm.weight` (GPT) or `decoder.final_norm.weight` (`HybridStack`)
- every MLP (`linear_fc1`, `linear_fc2`, and the pre-MLP norm)

The pre-MLP norm has two layouts in this repo. Transformer Engine folds it into `mlp.linear_fc1.layer_norm_weight`. The no-TE hybrid spec stores `pre_mlp_layernorm.weight`. The converter treats those names as aliases so a TE dense checkpoint can initialize a no-TE hybrid MLP. The same alias exists for attention input norm versus `self_attention.linear_qkv.layer_norm_weight`.

Attention (`*` slots) is copied only when the relative name exists on both sides and the shapes match. MLA projections (`linear_q_down_proj`, `linear_kv_up_proj`, and so on) are not present in a dense GQA checkpoint, so they stay at the hybrid initialization and are listed in `upcycle_report.json`. A same-named tensor with a different shape, typically `linear_proj` when the MLA head layout differs from GQA, is also left alone. Copying QKV into a Mamba or GDN `in_proj` would be a silent shape or layout bug, so `M` slots are never filled from attention weights.

MLP, embedding, and final-norm shape mismatches raise. Those tensors are the transplant; a vocab-padding or FFN mismatch should fail before training. Attention mismatches do not raise, because GQA-to-MLA is the expected Zebra case.

`E`, `|`, and `/` are rejected. Expert duplication stays on `--moe-use-upcycling`. Pipeline and MTP markers are a different checkpoint layout than this paired map.

The written checkpoint is iteration 0. Optimizer, RNG, and rerun state are dropped so a resumed Adam state cannot be applied to rewritten weights. Load it with:

```yaml
load: /path/to/upcycled
finetune: true
no_load_optim: true
no_load_rng: true
auto_continue_train: false
```

`load_checkpoint` puts `third_party/Megatron-LM` on `sys.path` before `torch.load`. Training does that itself; a standalone converter does not, and the pickled `args` namespace references Megatron classes.

Both inputs must be legacy `ckpt_format: torch` checkpoints with a single `mp_rank_00`. Distributed checkpoints are rejected with a pointer at `tools/hybrid/consolidate_distcp_to_torch.py`.

## Smoke run

Ran on one MI300X with mock data and `NullTokenizer` (`vocab_size: 512`, padded to 640).

| Run | Config idea | Result |
| --- | --- | --- |
| Dense source | 2-layer Llama-style GPT, hidden 128, FFN 256, SwiGLU, TE fused norms, 1 step | Saved `iter_0000001` |
| Hybrid init | Pattern `*-M-`, same hidden and FFN, no-TE Mamba+MLA spec, `lr: 0` for 1 step | Saved constructor weights at `iter_0000001` |
| Convert | `tools/hybrid/upcycle_dense_to_hybrid.py --hybrid-pattern '*-M-'` | 10 tensors copied. Mixer sublayer 2 left at init. Six MLA-only tensors left at init. `linear_proj` left at init because the shape did not match. |
| Continued train | Same hybrid config, `finetune: true`, `lr: 1e-3`, 3 steps, load the upcycled directory | `pretrain()` completed. Losses `6.383`, `6.424`, `6.115`. Grad norms finite. Zero NaN iterations. |

Weight check after conversion: embeddings, output layer, final norm, both MLP `fc1`/`fc2` weights, and both pre-MLP norms matched the dense tensors exactly, including the TE fused-norm to `pre_mlp_layernorm.weight` alias. `decoder.layers.2.mixer.in_proj.weight` was bitwise identical to the hybrid init checkpoint.

Weight check after the 3 training steps: that MLP, the mixer `in_proj`, the embedding, and the MLA `linear_q_down_proj` had all moved (max abs delta about `3e-3`). The loaded checkpoint was iteration 3.

Unit tests: `pytest tests/unit_tests/backends/megatron/test_hybrid_upcycle.py` — 6 passed.

## What this does not do

- It does not duplicate an MLP into MoE experts.
- It does not invent a QKV-to-Mamba or QKV-to-GDN mapping.
- It does not reshard TP/PP. Re-save both sides at TP=PP=1 first.
- It does not run logit distillation. The training YAML still has to pick the continued-pretrain learning rate and warmup.
