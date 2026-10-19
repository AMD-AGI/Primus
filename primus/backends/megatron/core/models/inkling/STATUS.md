# Inkling on Megatron (Primus): P0–P2 status

Status as of Monday Oct 19, 2026, on branch `feat/inkling-megatron` (merged
with `main` at `8e7a2406`; draft PR
[#1256](https://github.com/AMD-AGI/Primus/pull/1256)). Phases follow the plan
(P0 scaffolding, P1 components on one GPU, P2 training loop and recipe). The
numerical reference is the official transformers `models/inkling` code; Miles
is the design reference. Usage is in [README.md](README.md).

**Summary.** P0 and P1 are complete: every Inkling layer matches the
reference, and real checkpoint layers 0–5 import and match. P2 is partly
done: an 8-GPU AdamW pretraining run on mock data works, at Small widths
(12 layers) and at large-model widths (6 layers). Still missing from P2:
checkpoint save and restore, the real-tokenizer data path, Inkling FLOPs
reporting, the loss-curve comparison against the official implementation,
the Muon recipe details (router and `rel_proj` on Adam, lr²-coupled weight
decay), and a check that expert load stays balanced.

Status values: **Done**, **Partial** (works but not fully verified or not
fully built), **Missing** (not started).

## P0 — Scaffolding and reference harness

| Item | Status | Evidence / notes |
| --- | --- | --- |
| `inkling` model type, config class with every Inkling setting | Done | `InklingTransformerConfig`; `model_type: inkling` in `import_utils.get_model_provider` and in `megatron_pretrain_trainer.py` (reuses `pretrain_gpt` forward step and datasets) |
| Model presets | Done | `inkling_small_tiny` (2 layers), `inkling_small_12L`, `inkling_small`; added later: `inkling_large` (66 layers) and `inkling_large_6L` |
| Comparison harness against the official model | Done | `tests/.../inkling/inkling_reference.py`, `test_inkling_reference.py`; pinned transformers 5.19.0.dev0 at `263c5eb9` |
| Hugging Face → Megatron weight importer | Done | `tools/checkpoint_conversion/convert_inkling_hf_megatron.py`: router split into routed + shared gates, fused q/k/v (`linear_qkv`) with r kept separate (`linear_rel`), ShortConv, grouped-GEMM experts; embedding and LM head trimmed or zero-padded to the vocabulary training builds (200,058 rounded up to `make_vocab_size_divisible_by` × TP, 200,064 by default; the release has 201,024 rows); MTP, vision and audio skipped |
| Done when: full model builds without memory; every parameter maps 1:1 onto the release | Done | meta-device builds in `test_inkling_model.py`; `test_full_checkpoint_maps_one_to_one` and `test_release_schema_matches_checkpoint_headers` against the real Inkling-Small headers |

## P1 — Model components on one GPU

| Item | Status | Evidence / notes |
| --- | --- | --- |
| Attention with the learned relative-position bias | Done | `attention.py`, two backends: `eager` (dense bias, the reference) and `flex` (FlexAttention `score_mod` + cached block mask, trainable `rel_proj`). A query with no allowed key gets an all-zero output in both |
| Sliding (512) and global layers; log scaling on global layers | Done | Per-layer head counts and masks via `attention_config(layer)`; log scaling covered by the tiny model (floor 64 tokens) |
| ShortConv (4 per layer, fp32, plain PyTorch) | Done | `short_conv.py`; on K, V, attention output and MLP output |
| MoE router with the shared-expert sink, shared experts, dense layers | Done | `router.py`, `moe.py`, `mlp.py`; dense → MoE via `moe_layer_freq` |
| Embedding RMSNorm, muP-scaled untied head, init matching the reference | Done | `inkling_model.py`; `test_init_matches_reference_statistics` |
| fp32 SwiGLU / combine / `global_scale` (Miles numerics) | Done, opt-in | `inkling_fp32_activations` (default off = transformers numerics); needs TE grouped experts, not Primus-Turbo |
| Done when: tiny fp32 outputs and every gradient match to 1e-5 | Done | `test_tiny_fp32_forward_backward_matches_reference`, both backends, flag on and off |
| Done when: real checkpoint layers give the reference outputs | Done | Extended from 2 to 6 layers (0–5: sliding 0–4, global 5, dense 0–1, MoE 2–5); max logits error 1.49e-6 (eager) and 1.53e-6 (flex), top token identical everywhere (Oct 19) |
| bf16 numerics | Done | Megatron bf16 stays as close to fp32 as transformers' own bf16 (median token error 6.2–7.0e-3 vs 7.0–8.0e-3) |

Test totals: 64 tests, all passing on the merged branch (Oct 19, job 225454,
`rocm/primus:v26.8`). Three things must be in place, or those tests skip:
- the pinned transformers reference on `PYTHONPATH`, for 10 tests;
- Inkling-Small layers 0–5 fetched with `fetch_release_layers.py` (56.6 GB),
  for the real-layer test;
- Emerging-Optimizers at `93d9eb3a`, for the 5 Muon tests.

After r was split out of `linear_qkv`, the tiny fp32 model's worst errors were
4.6e-7 (eager logits), 5.2e-7 (flex logits) and 1.6e-6 (gradients).

Known P1 performance gap, not a correctness issue: flex global layers are
about 10× slower than plain causal attention (8k tokens: 61 ms vs 5.6 ms per
layer), mostly from the atomic-add gradient for the bias. A near/far split
measured 2.4× faster at 8k but is not implemented. See O1.

## P2 — Training loop and pretraining recipe

| Item | Status | Evidence / notes |
| --- | --- | --- |
| Pretraining config and single-node run | Done | `examples/megatron/configs/MI355X/inkling_small_12L-BF16-pretrain.yaml` (EP=8, DP=8, AdamW, mock data). 50 steps on `inkling_small_12L` and `inkling_large_6L`; see "Runs" below |
| Launch fix: `position_embedding_type` | Done | Megatron rejects `none`; `inkling_base.yaml` now uses `rope` at the args level (the model hard-codes none) |
| Masking: fixed causal + 512 window, packed rows without document boundaries | Done | Megatron default (`reset_attention_mask: false`); packed sequences raise `NotImplementedError` and `reset_attention_mask: true` is rejected by the builder, because the ShortConvs ignore the attention mask and would mix tokens across documents (O8). Padded rows are unsupported for the same reason |
| Data loading and tokenizer | Partial | Mock data works via `NullTokenizer` with `vocab_size: 200057` (+ eod = the 200,058 real tokens). The real HF tokenizer leaves `eod` as None (no eos/bos/pad token), so real data and mock data with the HF tokenizer would break. Need an explicit eod (`<\|endoftext\|>` = 199999, or 200006 = config `eos_token_id`). A real tokenized dataset has not been run |
| Checkpoint save and restore | Missing | No `sharded_state_dict` for `linear_qkv` / `linear_rel`, the TP-split K/V ShortConv weights, `shared_weight`, `global_scale`. Save → restore → same next-step loss is not tested. The converter writes only a single-rank `torch` checkpoint |
| Activation recompute | Missing (untested) | Megatron's recompute hooks are inherited (MoE and shared-expert recompute in `moe.py`); full or selective recompute with Inkling layers has never been run. Runs so far used `recompute_granularity: null` |
| FLOPs reporting | Missing | No Inkling FLOPs patch; Megatron's GPT formula is used. Logged 188 TFLOP/s/GPU vs about 210 by hand (Small 12L); 259 vs about 285 (large 6L) |
| Comparison with the official implementation (200-step loss curve) | Missing | Only proxies so far: step-1 loss = ln(200058); flex vs eager curves within about 0.15% (a few × run-to-run noise, consistent with bf16 rounding in eager); fp32 parity tests |
| First performance profile | Partial | Step times and the flex attention breakdown; no profiler trace of a training step |
| Muon + Adam hybrid optimizer | Partial | Stock Megatron `muon` runs on Small: 12L, 8 GPUs, 50 steps, no NaN; with `muon_extra_scale_factor: 1.0` loss 5.85 vs AdamW 5.76 at step 50, at 4.48 vs 3.55 s/step. Needs `PRIMUS_INSTALL_EMERGING_OPTIMIZERS=1`, no distributed optimizer and no grad/param overlap. Large: Megatron takes one q/k/v split shape from the global layers (8 KV groups) and sliding layers have 16, so the split crashed; `inkling_muon_qkv_patches.py` gives each `linear_qkv` its own layer's shapes (tests in `test_inkling_muon.py`). The Large 6L proxy then runs 50 steps with no NaN: loss 5.90 vs AdamW 5.75, at 4.52 vs 3.45 s/step. `dist_muon` (layer-wise distributed optimizer) with gradient overlap: same loss (5.902 at step 50), 4.28 s/step, 167 vs 212 GB per GPU. `dist_muon` with `overlap_param_gather` fails at step 2 when expert DP = 1 (EP 8 on 8 GPUs): Megatron leaves the expert buckets' layer-wise sizes unset; not patched yet. Missing: (1) the router weights and `rel_proj` are 2-D and go to Muon (the plan keeps them on Adam; ShortConvs are 3-D and already on Adam); (2) weight decay is not coupled to lr² |
| Expert load balancing (no auxiliary loss) | Partial | Wired: `expert_bias` (= `e_score_correction_bias`) is added in top-k selection, and Megatron's bias update runs via `_apply_expert_bias` and `moe_router_bias_update_rate`. Not verified: per-expert load and bias movement were never logged |
| Done when: AdamW 200-step loss curve tracks the official implementation | Missing | See above |
| Done when: save and restore give the same next-step loss | Missing | See checkpoint row |
| Done when: Muon vs AdamW at matched tokens; expert load balanced | Missing | Needs Muon and load metrics |

### Runs (8× MI355X, job 197602, container `primus_v26.8rc_rocm10.1`)

Both runs used AdamW (fp32 states, distributed optimizer), random init, mock
data, sequence length 4096, global batch 64, flex attention and no
Primus-Turbo. Logs are in `~/inkling_runs/`.

| Model | Params | Step-1 loss | Loss at 50 | Step time | Tokens/s/GPU | Memory/GPU |
| --- | --- | --- | --- | --- | --- | --- |
| `inkling_small_12L` | 67.5B | 12.210 | 5.76 | 3.59 s | 9,130 | 206 GB (71%) |
| `inkling_large_6L` | 62.6B | 12.209 | 5.75 | 3.32 s | 9,870 | 202 GB (70%) |

No skipped or NaN steps. Gradient norms steady at about 0.1. With eager
attention, steps take about 2× as long and use about 240 GB.

On the merged branch (Oct 14, job 216262, `rocm/primus:v26.8`), with the same
settings unless noted:

| Model | Optimizer | Loss at 50 | Step time | Memory/GPU |
| --- | --- | --- | --- | --- |
| `inkling_small_12L` | AdamW | 5.762 (5.7615 before the merge) | 3.54 s | about 206 GB |
| `inkling_small_12L` | `muon`, scale 1.0 | 5.85 | 4.48 s | about 200 GB |
| `inkling_large_6L` | AdamW | 5.75 | 3.45 s | about 202 GB |
| `inkling_large_6L` | `muon` + per-layer q/k/v split | 5.90 | 4.52 s | 212 GB |
| `inkling_large_6L` | `dist_muon`, gradient overlap | 5.902 | 4.28 s | 167 GB |

Muon runs set `--use_distributed_optimizer false` (plain `muon` only) and
`--muon_extra_scale_factor 1.0`; scale 0.2 gave loss 7.10 on Small.

## Outside P0–P2, for reference

- Parallel layouts on 8 GPUs (Oct 14, rerun Oct 19 after the review fixes and
  the merge with the same result):
  `tests/.../inkling/parallel_reference_check.py` trains one step of an
  8-layer fp32 Inkling under DP 8, EP 8, PP 2 / 4 / 8 (with EP 4 / 2 / 1) and
  interleaved PP 2 and PP 4 (2 virtual stages each). It compares the global
  loss and every gradient with the transformers reference run in fp64. All
  14 runs (7 layouts × eager and flex) pass:
  - The loss equals fp64 to 7 digits.
  - The worst tensor gradient error is 3.4e-6, the same as the reference's
    own fp32 error.
  - The worst `global_scale` (scalar) error is 1.4e-5, against a limit
    of 1e-4.
- P3 (multi-node): TP and EP above 8 have never run, and PP has not been
  used in a training run. Sequence parallelism is rejected (the output ShortConvs need a 3-token halo exchange). Context
  parallelism is rejected. Streaming per-rank import of the full checkpoint
  is not written.
- Large model: `inkling_large.yaml` matches the released config (947.0B
  text parameters). Its shapes were not checked against the real large
  checkpoint headers.

## Suggested order for the rest of P2

1. eod fix for the HF tokenizer, then a run on a real tokenized dataset.
2. `sharded_state_dict` and a save → restore → same-loss test.
3. Inkling FLOPs patch.
4. Expert-load logging, to confirm balancing works.
5. 200-step loss comparison against the official implementation.
6. Muon recipe details (router and `rel_proj` on Adam, lr²-coupled weight
   decay) and the `dist_muon` parameter-gather fix, then Muon vs AdamW at
   matched tokens.
7. A recompute test, and a profile of a training step.
