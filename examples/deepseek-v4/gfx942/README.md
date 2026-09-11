# DeepSeek-V4 SFT on gfx942 (MI308X / CDNA3)

Runs a 4-layer DeepSeek-V4 SFT at **128k context** exercising **all three** V4 attention
branches — dense+SWA, CSA, HCA — on a single 8×MI308X node.

Upstream's V4 support targets gfx950 / CDNA4 and, as shipped, only ever ran at
`seq_length=4096`. Several things that are invisible at 4k become hard blockers at 128k.
This directory carries the fixes and a one-click launcher.

```bash
# inside a rocm/primus:v26.5-pytorch2.12-te2.15 container, from the repo root
bash examples/deepseek-v4/gfx942/run_128k_dense_hca_csa.sh
```

The script is self-contained: it copies Megatron-LM out of the image (or fetches the
submodule), builds the SFT dataset on first run, and resolves every path relative to
itself. The only external requirement is a **local DeepSeek-V4 tokenizer directory** —
point `V4_TOKENIZER` at it, or leave it at `/apps/DeepSeek-V4-Flash`. Only
`tokenizer.json` / `tokenizer_config.json` are read.

## Scope — read this before quoting any number

* **Weights are randomly initialised.** No DeepSeek-V4 Megatron checkpoint exists, and
  Megatron-Bridge registers importers only for V2-Lite/V2/V3. This is *SFT-shaped training
  from scratch*, not fine-tuning. The tokenizer and the prompt/response loss mask are real.
* **4 layers, not 43.** `compress_ratios [0, 0, 4, 128]` was chosen to cover one layer of
  each branch type. The released model is 3 dense / 21 CSA / 20 HCA.
* **1M does not fit on one node.** See "Long context" below for how far it gets and why.

## Results

8×MI308X, 4 layers, `[0, 0, 4, 128]`, `triton_v2`, 10 steps:

| | TP=8 / CP=1 (first working recipe) | **CP=8 / TP=1 / EP=8 (current)** |
|---|---|---|
| peak memory | 188.63 GB (98.25%) | **42.30 GB (22%)** |
| step time | 9.4 s | **3.83 s** |
| loss | 11.935 → 11.777, monotone | 11.886 → 11.686, monotone |
| nan iterations | 0 | 0 |

### Why CP, not TP

This is the single most important finding here, and it is not the obvious choice.

TP shards along the head axis. But the tensors that actually dominate at long context
**have no head axis**, so TP cannot touch them:

* the indexer's `scores` is `[B, S, P]` — heads are already summed out
* V4's KV is a **single MQA latent** `[B, S, 1, head_dim]`, broadcast to H heads by a
  stride-0 view
* MoE / MLP / residual activations scale with `S` alone
* the LM head's logits are `[S, B, vocab]`

CP shards the sequence and therefore shards every one of them. Measured: 4.5× less memory
and 2.5× faster than the TP=8 recipe, same model, same math.

Both can be 8 on 8 GPUs because the expert side decomposes as `ETP × EP × PP`, which does
not include CP (`parallel_state.py:794`), while the attention side needs
`TP × PP × CP ≤ world_size`.

## What had to change

### 1. P14 — shard attention heads across TP

`deepseek_v4_layer_specs.py`, `deepseek_v4_attention.py`, `indexer.py`,
`deepseek_v4_transformer_config.py`

V4 built `linear_q_up_proj` with `gather_output=True` and set
`num_attention_heads_per_partition = num_heads` (no `divide()`). TP therefore sharded
**weights only** — the `[B, S, H, head_dim]` query was replicated on every rank. The source
called head-sharded attention "tracked in P14"; it was never implemented.

Enabled by `v4_shard_attention_heads: true` (default **false**). The grouped-O projection
makes this clean: it is block-diagonal over `o_groups`, and `linear_o_b` is already
row-parallel, so at TP=8 with `o_groups=8` each rank owns one group and the row-parallel
all-reduce sums them — verified in float64, residual 2.5e-15.

Superseded in practice by the CP recipe above, but kept: it is correct, and it is what
makes TP>1 meaningful at all.

### 2. Fused indexer scoring at the real head count

`v4_attention_kernels/_triton_common/indexer_score.py`, `indexer_score_post.py`

Both Triton indexer kernels gated on `_SUPPORTED_H = (1, 2, 4, 8, 16)`, while the released
config and Primus's own `deepseek_v4_base.yaml` set `index_n_heads=64`. **At the real width
these kernels were unreachable** and the indexer silently fell back to an eager einsum that
materialises `[B, S, H, P]`: 0.5 GiB at 4k, **512 GiB at 128k**.

* `_SUPPORTED_H` extended to 32 and 64 (fwd matches eager to 3e-7, bwd to bf16 noise)
* `k_tile` hoisted out of the unrolled head loop — it never depended on `h`, so H=64 was
  doing 63 redundant tile loads per block

### 3. int64 offsets in the indexer kernels

Every offset was int32. `s_offs * P` wraps negative once `S*P` exceeds 2³¹, which for CSA
(`P = S/4`) happens at `S ≈ 92682` — the store then lands out of bounds *silently*.
Bisected: 64k clean, 96k NaN, 128k NaN. 15 offsets promoted.

### 4. A dead 16 GiB allocation in the CSA path

The CSA branch built a dense `[S, S]` sliding-window mask and passed it to `_csa_forward`,
whose own docstring says it is *"retained in the signature for back-compat but unused"* —
the function `del`s it on entry. 16 GiB at 128k, allocated to be thrown away.

### 5. gfx942 LDS budget

`v4_attention_kernels/_triton_v2/dsa_bwd_v4_triton.py`

The sparse-MLA backward is tuned for gfx950's 160 KB LDS; gfx942 has 64 KB and the stock
staging asks for 73728 B, so the kernel fails to **compile**. `PRIMUS_DSA_BWD_NUM_STAGES=1`
disables Triton's LDS multi-buffering. Measured: this is the *only* kernel switch needed.

Also added `PRIMUS_DSA_BWD_R_CHUNK`. The backward's rank-chunk width is hard-coded to 256,
tuned for short sequences where the per-chunk buffers are small and dq reload traffic
dominates. At long context that inverts: the buffers scale with `total_tokens * R_CHUNK`,
so at `S_local = 131072` the `interm` buffer alone is 36 GiB.

### 6. Context parallelism for all three branches

`deepseek_v4_cp.py` (new), `deepseek_v4_attention.py`, `indexer.py`,
`v4_sparse_mla_adapter.py`, `sft/forward_step.py`

A CP rank needs the `d_window` post-RoPE KV rows left of its shard plus its global row
offset; the sparse-MLA adapter then validates the window against **global** positions while
indexing the local `[boundary ++ local]` buffer.

Three bugs were found getting CSA and HCA correct, all of which produce a *correct forward*
and a wrong gradient:

1. **The local sliding window was not exchanged.** CP was wired only into the dense branch,
   but CSA and HCA each run a sliding window over raw tokens too, and that window straddles
   the shard edge. Factored into `_cp_prepend_boundary`, used by all three.
2. **`_AllGatherPool.backward` sliced instead of reduce-scattering.** Rank r's pool rows are
   read by the queries of every rank at or after r, so each holds a partial gradient;
   taking only this rank's block dropped all the downstream contributions. Invisible in the
   forward, compounding over steps (loss drift 2e-5 → 4.5e-4 over three steps).
3. **The indexer built its key pool from local hidden only.** `indexer_compressor(hidden)`
   produced `P_local` while the attention pool was already `P_global`, so the top-K indices
   named the wrong columns and a query could never select history from an earlier rank.

Verified at 8k with a uniform dataset, TP=1, EP=1 (so weight init and data are identical
across runs). Spread across CP=1/2/4, against a measured CP noise floor of 5e-5:

| config | CP=1 | CP=2 | CP=4 | spread |
|---|---|---|---|---|
| dense only (control) | 11.88296 | — | 11.88293 | 5e-5 |
| dense + HCA | 11.88916 | 11.88918 | 11.88914 | 5e-5 |
| dense + CSA + HCA | 11.86170 | 11.86179 | 11.86169 | 1e-4 |

CSA sits slightly above the floor and does not grow with steps — top-K is a discrete
selection, so a bf16-level perturbation occasionally swaps the 512th and 513th column.

### 7. Memory: measure, don't guess

Every attempt to reason about where the memory went was wrong until it was profiled. A
`sitecustomize.py` probe (injected via `PYTHONPATH`, so no code change) recording allocation
stacks gave the actual composition of a 179.67 GiB peak, and the top entry was not in
attention at all:

| GiB | site | |
|---:|---|---|
| 31.56 | `deepseek_v4_model.py` LM head logits | `S × vocab`, fixed below |
| 24.24 | `v4_sparse_mla_adapter.py` topk index matrix | int64, fixed below |
| 18.00 | `_rope_pad_q` | real |
| 16.00 | sparse-MLA fwd kernel | real |
| 16.00 | RMSNorm | real |
| 12.00 | `hc_expand` | real |

Fixes that followed:

* **Chunked linear + cross-entropy for V4.** Primus already ships this
  (`patches/fused_linear_ce_patches.py`) but it hooks `GPTModel._postprocess`, and
  `DeepseekV4Model` derives from `LanguageModule` — so it never applied. Wired in behind
  `FUSED_LINEAR_CE=1`, restricted to TP=1 (the chunked path matmuls against the full output
  weight, which is only equivalent when the vocab is not sharded).
* **int32 topk index matrix.** `torch.full((B,S,P), -1)` materialised an 8.6 GiB constant
  just to be a `torch.where` else-branch, and every intermediate was int64 although
  `_pad_topk_64` casts to int32 immediately. ~24 GiB at 1M.
* **HCA concatenated on the head-broadcast views.** `torch.cat` on a stride-0 expanded view
  materialises the H-fold copy the broadcast exists to avoid — 8.51 GiB for K and another
  8.51 GiB for V, per HCA layer, of which the consumer reads 136 MiB (it takes `k_bh[:, 0]`
  and never reads `v_bh` at all).
* **Output BSHD→BHSD→BSHD round trip.** The adapter returned a `.contiguous()` BHSD copy
  and every caller immediately made a BSHD copy of it. Both are now views.
* **`dk = zeros(B, H, Skv, D)` in both backwards**, 63/64 of it zeros, allocated only to
  match the broadcast view's shape. The adapter now takes the un-broadcast latent.

### 8. SFT plumbing

* `mock_data` is fatal under `stage: sft` — the mock-data patch force-installs
  `NullTokenizer`, whose `text_to_ids` is `int(x)` per whitespace token.
* `train_data_path` must be non-null or the pretrain data-prep hook demands `HF_TOKEN`.
* `rope_type: rope` — the base preset says `yarn`, but common-attention asserts `rope`.
* `moe_router_enable_expert_bias: false` — expert bias requires `sigmoid` scoring, which
  conflicts with V4's `sqrtsoftplus`.
* `create_attention_mask_in_dataloader: false` — otherwise a `[S, S]` bool tensor.
* Contiguous CP sharding of the batch and a CP-aware loss reduction.

## A pattern worth naming

`DeepseekV4Model` derives from `LanguageModule` rather than `GPTModel`;
`DeepseekV4TransformerBlock` deliberately bypasses `TransformerBlock.__init__`;
`DeepseekV4Attention` fully overrides `MultiLatentAttention.forward`. Each is defensible on
its own. The cumulative effect is that **whole classes of upstream optimisation silently do
nothing on V4**, with no error and no warning:

| optimisation | upstream hook point | why it missed V4 |
|---|---|---|
| chunked linear+CE | `GPTModel._postprocess` | V4 derives from `LanguageModule` |
| TE CPU activation offload | `TransformerBlock.__init__` / `.forward` | V4 bypasses both |
| fine-grained activation offload | `GPTModel.forward`, `attention.py` | both bypassed |
| head-sharded TP | `Attention.__init__`'s `divide()` | V4 set the full head count |

The first two are fixed here. When adding anything to V4, check whether the upstream
version of it is reachable before concluding it does not help.

## Known limits

* **`turbo` backend does not run on gfx942.** Its FlyDSL kernels emit `permlane16/32_swap`
  and `mfma_f32_16x16x32_bf16`, both CDNA4-only. The permlane butterfly has a `ds_bpermute`
  equivalent, but the MFMA tile shape does not — CDNA3's `mfma_f32_16x16x16bf16_1k` has half
  the K depth, so 25 call sites plus an inline-asm block would need re-tiling. `gluon*` and
  `flydsl_v1` hard-assert gfx950. **`triton_v2` is the fastest usable backend.**
* **turbo and the fused indexer want opposite TP.** turbo asserts `num_heads % 32 == 0`
  (TP ≤ 2 at H=64) while the fused indexer needs H_local ≤ 16 (TP ≥ 4).
* **`FUSED_LINEAR_CE=1` requires `overlap_param_gather: false`.** The chunked backward
  issues one `autograd.grad` per chunk, changing each parameter's backward-hook firing
  count, which desyncs the distributed optimizer's overlapped all-gather.
* **Streaming indexer top-K (`PRIMUS_INDEXER_TOPK_CHUNK`) is not a general memory switch.**
  It is bit-exact and costs nothing at chunk=8192, but it only helps when
  `S_local × P_global` makes `scores` a dominant term. At 128k with CP=8 that tensor is
  1 GiB and chunking it saves exactly zero.

## Long context

| seq | S_local (CP=8) | result |
|---|---|---|
| 128k | 16384 | 42.30 GB, 3.83 s/step |
| 512k | 65536 | one step at 175.7 GB and **432 s/step**, then OOM |
| 1M | 131072 | OOM |

**1M does not fit on one node, and narrowing the model does not fix it.** Measured:

| narrowing | outcome |
|---|---|
| `num_attention_heads` 64 → 8 | OOM at 182.81 GiB — peak unchanged |
| `hc_mult` 4 → 2 | OOM at 180.69 GiB |
| `kv_channels` 512 → 256 | OOM at 182.84 GiB (steady state only 56 GB) |

Steady-state memory drops a lot in each case, but the failure watermark does not move: the
binding terms are the ones that do not scale with model width. After the section-7 fixes the
watermark is 167.38 GiB with 5.63 GiB free, and what remains is genuine computation state,
not waste. Two nodes at CP=16 halves `S_local` and is the clean path.

One open item: at `PRIMUS_DSA_BWD_R_CHUNK=64` — the only setting whose workspace fits — the
8-GPU run dies with `HSA_STATUS_ERROR_MEMORY_FAULT`. The kernel is *not* at fault: a
standalone harness at the identical shapes (both CSA `topk=640` and HCA `topk=8320`, with
30% `-1` sentinels, under 130 GiB of ballast) completes correctly, and an exhaustive audit
found no int32 overflow. The remaining difference is the distributed context. Unresolved.
