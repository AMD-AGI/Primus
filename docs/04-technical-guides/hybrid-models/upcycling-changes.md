# Dense-to-hybrid upcycling — change notes

This note records what was added for non-MoE hybrid upcycling and why. The user-facing procedure is in [README.md](README.md#upcycling-a-dense-checkpoint).

## Why this exists

Continued pretraining of a Zebra hybrid (attention + Mamba, GDN, or KDA, plus MLP) should be able to start from a dense Transformer checkpoint. Megatron already has `--moe-use-upcycling`, which duplicates one dense MLP into experts. That path does not apply here: a hybrid stack is a different module layout. HyLo already defines how to seed MLA and GDN or Mamba from the dense Q, K, V, and O projections, and this converter follows that recipe.

Primus hybrid stacks are a flat list of sublayers. `HybridStack.allocate_layers` pairs every sequence mixer with its own MLP, so a dense model of `N` blocks becomes `2N` sublayers:

```text
dense layer i  ->  hybrid sublayers (2i, 2i + 1) = (mixer, MLP)
```

Pattern symbols are the ones Megatron already uses: `*` attention, `M` recurrent mixer, `-` MLP.

## Files

| File | Why it changed |
| --- | --- |
| `primus/backends/megatron/checkpoint/hybrid_upcycle.py` | Library that plans the pattern, applies the HyLo from-teacher init on top of a hybrid init checkpoint, and writes a legacy Megatron checkpoint. Lives next to the other native checkpoint converters so tests can import it without a `sys.path` hack. |
| `tools/hybrid/upcycle_dense_to_hybrid.py` | CLI. Inserts the repo root on `sys.path` and calls the library. Same entry-point style as the other `tools/hybrid` converters. |
| `tests/unit_tests/backends/megatron/test_hybrid_upcycle.py` | CPU tests for pattern allocation, MoE rejection, MLP remapping, attention shape mismatches, fused-norm aliases, and the checkpoint envelope. |
| `docs/04-technical-guides/hybrid-models/README.md` | Short procedure for producing the hybrid init checkpoint, running the converter, and loading the result. |
| `docs/04-technical-guides/hybrid-models/upcycling-changes.md` | This note. |

No Megatron-LM submodule files were edited. The converter is an offline checkpoint rewrite, so it does not hook `pretrain()`.

## What changed

The first version of this converter copied a tensor only when the dense and hybrid names and shapes matched. MLA projections stayed at hybrid init, and every `M` slot (Mamba, GDN, and KDA) was left untouched on purpose. That avoided inventing a QKV-to-mixer map.

This version uses the from-teacher initialization in [HyLo](https://github.com/AMD-AGI/AMD-Hybrid-Models/tree/feat/HyLo), `hybrid/hybrid_wrapper.py` on `feat/HyLo`. HyLo builds a Hugging Face hybrid module and copies teacher weights in `HybridModelWrapper.__init__` (`init_with_svd` for MLA, `init_with_kqvo` for GDN and Mamba). Primus checkpoints are Megatron state dicts, so the same math is applied to Megatron tensor names instead of importing that module.

| HyLo (`feat/HyLo`) | Primus checkpoint |
| --- | --- |
| `DeepseekV3Attention.re_init_q` writes `q_a_proj` and the nope rows of `q_b_proj` | `self_attention.linear_q_down_proj.weight` and the nope rows of `linear_q_up_proj.weight` |
| `re_init_kv` writes `kv_a_proj_with_mqa` and `kv_b_proj` | the leading `kv_lora_rank` rows of `linear_kv_down_proj.weight`, and `linear_kv_up_proj.weight` |
| `out_proj` copies `o_proj[:, :mla_o_in]` | overlapping columns of `self_attention.linear_proj.weight` |
| `_copy_llama_attn_to_gdn` writes `gdn.q_proj` / `k_proj` / `v_proj` / `o_proj` after repeating GQA K and V | the `q`, `k`, `v` slices of fused `mixer.in_proj.weight` (the order in `convert_gdn_hybrid_to_fla_hf.py`) and `mixer.out_proj.weight` |
| Mamba `in_proj[d_inner:d_inner+d_xb] <- V`, then K, then Q into `C`, and `out_proj <- o_proj` | the same slices when `mixer.in_proj` has HyLo's width `2 * d_inner + 2 * d_xb + nheads` |

RoPE rows in `linear_q_up_proj` and the rows of `linear_kv_down_proj` after `kv_lora_rank` stay at hybrid init. HyLo's SVD does not fill those channels. They are listed in `hylo_partial`.

GDN `g_proj`, `a_proj`, `b_proj`, `A_log`, `dt_bias`, and `conv1d` stay at hybrid init, matching the comment in `_copy_llama_attn_to_gdn`. HyLo's `post_attention_layernorm` is the norm in front of the MLP; on a Primus stack that tensor lives on the following `-` sublayer and is copied with the MLP.

KDA is unchanged from the first version: HyLo has no KDA recipe, so a KDA `M` slot stays at hybrid init and is listed in `mixer_layers_left_initialized`. A Mamba `in_proj` is also left alone unless its row count is HyLo's. Primus Mamba2 usually stores `[z | x | B | C | dt]` with `x` width `d_inner` and `B`/`C` width `n_groups * d_state`, which matches HyLo only when `d_xb == d_inner == n_groups * d_state`.

HyLo can also load a finished stage-1 ILD checkpoint (`linear_ILD_path` / `mla_model`). This converter does not. It only does the from-teacher init.

## Copy rules, and why

The hybrid init checkpoint is the base. The converter overlays dense tensors and leaves everything else alone. Recurrent parameters that HyLo does not fill (`A_log`, `dt_bias`, `conv1d`, the GDN gate) keep the constructor initialization from the hybrid init checkpoint.

What is copied in full:

- `embedding.word_embeddings.weight`
- `output_layer.weight`, when the hybrid model unties it
- the final norm, accepting either `decoder.final_layernorm.weight` (GPT) or `decoder.final_norm.weight` (`HybridStack`)
- every MLP (`linear_fc1`, `linear_fc2`, and the pre-MLP norm)

The pre-MLP norm has two layouts in this repo. Transformer Engine folds it into `mlp.linear_fc1.layer_norm_weight`. The no-TE hybrid spec stores `pre_mlp_layernorm.weight`. The converter treats those names as aliases so a TE dense checkpoint can initialize a no-TE hybrid MLP. The same alias exists for attention input norm versus `self_attention.linear_qkv.layer_norm_weight`, and for `mixer.in_proj.layer_norm_weight`.

MLP, embedding, and final-norm shape mismatches raise. Those tensors are the transplant; a vocab-padding or FFN mismatch should fail before training. HyLo slice copies that clip a projection do not raise; they are recorded in `hylo_partial`.

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

Recorded against the earlier exact-shape converter, before the HyLo recipes. It is not a result for the SVD or QKVO init.

Ran on one MI300X with mock data and `NullTokenizer` (`vocab_size: 512`, padded to 640).

| Run | Config idea | Result |
| --- | --- | --- |
| Dense source | 2-layer Llama-style GPT, hidden 128, FFN 256, SwiGLU, TE fused norms, 1 step | Saved `iter_0000001` |
| Hybrid init | Pattern `*-M-`, same hidden and FFN, no-TE Mamba+MLA spec, `lr: 0` for 1 step | Saved constructor weights at `iter_0000001` |
| Convert | `tools/hybrid/upcycle_dense_to_hybrid.py --hybrid-pattern '*-M-'` | 10 tensors copied. Mixer sublayer 2 left at init. Six MLA-only tensors left at init. `linear_proj` left at init because the shape did not match. |
| Continued train | Same hybrid config, `finetune: true`, `lr: 1e-3`, 3 steps, load the upcycled directory | `pretrain()` completed. Losses `6.383`, `6.424`, `6.115`. Grad norms finite. Zero NaN iterations. |

Weight check after conversion: embeddings, output layer, final norm, both MLP `fc1`/`fc2` weights, and both pre-MLP norms matched the dense tensors exactly, including the TE fused-norm to `pre_mlp_layernorm.weight` alias. `decoder.layers.2.mixer.in_proj.weight` was bitwise identical to the hybrid init checkpoint.

Weight check after the 3 training steps: that MLP, the mixer `in_proj`, the embedding, and the MLA `linear_q_down_proj` had all moved (max abs delta about `3e-3`). The loaded checkpoint was iteration 3.

Unit tests at the time of that smoke run: 6 passed. After the HyLo port the same file has 8 tests, including the GDN slice copy and the MLA SVD.

## What this does not do

- It does not duplicate an MLP into MoE experts.
- It does not initialize KDA from dense QKV. HyLo has no KDA path.
- It does not load a HyLo stage-1 ILD checkpoint. Only the from-teacher SVD and QKVO copy are ported.
- It does not reshard TP/PP. Re-save both sides at TP=PP=1 first.
- It does not run logit distillation. The training YAML still has to pick the continued-pretrain learning rate and warmup.
