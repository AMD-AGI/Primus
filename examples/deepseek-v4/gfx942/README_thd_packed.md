# DeepSeek-V4 SFT on packed sequences (THD), gfx942

Sequence packing for DeepSeek-V4: many real SFT samples concatenated into one window and
delimited by `cu_seqlens`, with attention strictly isolated so no query sees another
sample. Three recipes, sharing one implementation:

| recipe | script | model | context | nodes |
|---|---|---|---|---|
| single-node | [`run_128k_thd_packed.sh`](run_128k_thd_packed.sh) | 3 layers (dense / CSA / HCA) | 128k | 1 |
| full, short | [`run_full_4k_thd_multinode.sh`](run_full_4k_thd_multinode.sh) | 43 layers + MTP, 256 experts | 4k | 3 |
| full, long | [`run_full_128k_thd_multinode.sh`](run_full_128k_thd_multinode.sh) | 43 layers + MTP, 256 experts | 128k | 3 |

```bash
# single node, from the repo root inside a rocm/primus:v26.5-pytorch2.12-te2.15 container
bash examples/deepseek-v4/gfx942/run_128k_thd_packed.sh

# three nodes: run on each, NODE_RANK 0 / 1 / 2
MASTER_ADDR=<node0-ip> NODE_RANK=<r> bash examples/deepseek-v4/gfx942/run_full_4k_thd_multinode.sh
MASTER_ADDR=<node0-ip> NODE_RANK=<r> bash examples/deepseek-v4/gfx942/run_full_128k_thd_multinode.sh
```

The multi-node scripts delegate to [`run_full_4k_multinode.sh`](run_full_4k_multinode.sh)
for networking, parallelism and offload, and the 128k one delegates further to the 4k one,
so the BSHD and THD recipes cannot drift apart. See
[README.md](README.md) for the gfx942 kernel settings and the CP-not-TP argument, and
[README_full_4k_multinode.md](README_full_4k_multinode.md) for the BSHD multi-node recipe.

## Why packing is not just a data change

Alpaca's runtime segments have a median of 84 tokens, so an unpacked 4096-token row is
almost entirely padding and the run measures the hardware rather than the model.

But packing puts sequence boundaries at arbitrary offsets, and V4's compressor pools fixed
windows of `ratio` rows **anchored at each sequence's start**. Under context parallelism a
window can then straddle a shard boundary, with its leading rows owned by the left
neighbour. Two mechanisms make that work:

* **Left-boundary exchange** (`deepseek_v4_cp.exchange_boundary_hidden`) ships the
  neighbour's trailing rows, so no alignment padding is needed.
* **Strict window ownership** — a window belongs to the rank holding its *last* row — plus
  a fixed per-rank capacity. The capacity has to be fixed because `_AllGatherPool` sizes
  its receive buffers with `torch.empty_like(pool_local)`: if two ranks disagreed on the
  pool width, the all-gather's shapes would not match.

## Results

### Single node, 3 layers, 128k

Three independent runs, and one from a pristine container with empty data / output / cache
directories:

| | |
|---|---|
| loss @ step 10 | 11.71954 / 11.71952 / 11.71954, pristine container 11.71954 |
| grad norm | 18.27 → 15.323 |
| peak memory | 119.04 GB (62.01%) |
| step time | ~4.3 s |
| nan iterations | 0 |

Reproducible to five significant figures; step time and memory are not, and depend on what
else shares the node.

### Full model, three nodes

Ten steps, natural-length Alpaca packed at runtime:

| | 4k | 128k |
|---|---|---|
| parallelism | TP=1 PP=3 **CP=1** EP=8, DP=8 | TP=1 PP=3 **CP=8** EP=8, DP=1 |
| GBS | 24 | 3 |
| optimizer offload | 0.75 | **0.90** |
| loss | 11.889 → 10.896 | 11.879 → 10.949 |
| grad norm | 30.7 → 20.6 | 31.2 → 19.6 |
| nan iterations | 0 | 0 |
| peak memory | 149.84 GB (78%) | 163.60 GB (85%) |
| step time | ~31 s | ~165 s |

The two loss curves have the same shape, which is the point of running the long one: 128k
introduces no divergence or gradient blow-up that 4k does not have.

### Correctness

| check | how | result |
|---|---|---|
| CP=1 vs CP=2 and CP=4 | manual, see below | dense / CSA / HCA all match; max 3.9e-3 |
| packed vs unpacked, three branches | `pytest test_deepseek_v4_thd_packing.py` | pass |
| window ownership vs upstream's rule | `pytest test_thd_compact_plan.py` | pass |
| streaming top-K vs one-shot scoring | `pytest test_thd_indexer_streaming_topk.py` | pass |
| empty pool keeps its gradient | `pytest test_compressor_empty_pool_grad.py` | pass |

pytest needs `PRIMUS_V4_UT_ALLOW_NON_MI355X=1` on gfx942. `thd_cp_equivalence.py` is **not**
a pytest module and is **forward-only**:

```bash
D=tests/unit_tests/megatron/transformer/deepseek_v4
for n in 1 2 4; do
  torchrun --nproc_per_node=$n $D/thd_cp_equivalence.py --out /tmp/cp$n.pt \
      --seq-lens 333 191 277 223
done
python $D/thd_cp_equivalence.py --compare /tmp/cp1.pt /tmp/cp2.pt
```

Pass `--seq-lens` explicitly: the default `[512, 256, 128, 128]` puts every shard boundary
on a sequence start, the one layout that does *not* exercise straddling windows. CP=4
matters more than CP=2 — at CP=2 this layout gives both ranks equal window counts and never
reaches the fixed-capacity path, a trap `test_thd_compact_plan.py` pins shut deliberately.

## Parallelism at 128k, and why there is no second option

24 GPUs. PP=3 is forced by parameter memory, leaving `TP × CP × DP = 8`. TP cannot help:
V4's dominant tensors have no head axis to shard — the indexer's `scores` is `[B, S, P]`
with heads already summed out, the KV is a single MQA latent, and the MoE / residual
activations scale with `S` alone. CP shards all of them, so CP=8, and DP falls out as 1.
The expert side decomposes independently: `ETP=1 × EP=8 × PP=3 = 24`.

Local rows per rank are `131072 / 8 = 16384`, which stays divisible by the HCA ratio 128,
so every compressed branch is exercised rather than silently skipped.

## Memory: measured, not extrapolated

Peak GPU memory for the full model, walking the sequence length:

| seq | peak | of 192 GB |
|---|---|---|
| 4k | 149.84 GB | 78.05% |
| 16k | 150.41 GB | 78.35% |
| 32k | 156.03 GB | 81.27% |
| 64k | 166.06 GB | 86.50% |
| **128k** | **186.32 GB** | **97.05%** (offload 0.75) |
| **128k** | **163.60 GB** | **85.21%** (offload **0.90**, the default) |

4k → 16k costs almost nothing: at that scale params + optimizer dominate and the sequence
is not yet the constraint. From 32k on it is roughly linear in local rows, ~+10 GB per
doubling.

**offload 0.90 is the default deliberately.** 97% is not a configuration to rely on — the
reading is device-wide, so another process taking 6 GB kills the run. Moving more optimizer
state to the host buys 22.7 GB of headroom and costs nothing measurable; ~2.9 TB of host
memory is free.

Levers, in the order worth trying: raise `PRIMUS_OPTIMIZER_OFFLOAD_FRACTION`, drop `GBS` to
1 (removes two microbatches of pipeline-resident activation, at the cost of a worse bubble),
shrink `FUSED_CE_CHUNK`.

### `FUSED_LINEAR_CE` is required at 128k, not an optimisation

The LM head's logits are `[S, B, vocab]` = **31.6 GB** at 128k with vocab 129280, before the
loss upcasts. The chunked linear+CE path is what makes that fit.

It carries a hard constraint: `PRIMUS_OVERLAP_PARAM_GATHER` must be `false`. The chunked
backward issues one `autograd.grad` per chunk, which changes each parameter's backward-hook
firing count and desyncs the distributed optimizer's overlapped gather. `DeepseekV4Model`
raises if you forget, so this fails loudly rather than silently.

## What packing buys, measured on the same corpus

Both layouts consume identical total tokens (`GBS × seq`), so step time is directly
comparable and the difference is entirely in how much of each window is real content.
Measured over every pack the runtime packer produces, at 4k on natural-length Alpaca:

| | total | real | supervised |
|---|---|---|---|
| BSHD | 163,840 | 2.8% | 1.9% |
| THD | 163,840 | 100.0% | 86.1% |

and the step times, four alternating runs on three nodes:

| | ms/step | run-to-run spread |
|---|---|---|
| BSHD | 30,445 | 4.8% |
| THD | 30,930 | 3.2% |

The 485 ms difference is smaller than either configuration's own run-to-run spread, so the
honest reading is that **step time is indistinguishable**. Useful throughput therefore
tracks the token budget almost exactly: **35.2× real tokens/s, 44.6× supervised tokens/s**.

Two things must travel with those numbers:

* **This is not a criticism of the existing BSHD recipe.** `run_full_4k_multinode.sh`
  defaults to `sft_4096.jsonl`, whose rows are deliberately built LONGER than the window so
  truncation rather than padding sets the length — measured at **99.3% supervised**. The
  comparison above forces both layouts onto the same natural-length corpus, which is the
  only way to isolate what packing does. Quoting 44.6× as "the current recipe wastes 97% of
  its compute" would be wrong.
* **Packing also changes what is learned, not just how much.** Those 99.3% supervised
  tokens come from a synthetic row concatenating unrelated samples with no boundary between
  them, so attention flows freely across them. THD isolates samples strictly. That is not in
  the throughput number.

## Things that bit, and what they looked like

Each produced a *plausible-looking run*. None announced itself as an error.

### An exit-0 run that trained nothing

Feeding the training pipeline the output of `prepare_packed_data.py` — a pre-packed
`{input_ids, labels, cu_seqlens}` jsonl — completes 10/10 with exit 0, zero nan and a
sensible step time. It also has **grad norm 0.000 and no loss line**, because there is no
loader for that schema: the alpaca formatter finds no `instruction`/`output` fields, every
sample tokenizes to nothing, and the loss mask is all zero.

`prepare_packed_data.py` is an **offline measurement aid**, not a training input. The
packing that matters happens at runtime in `PackedSFTDataset`, from an alpaca-format corpus,
and is what `sft_packing_segment_align` applies to.

### MTP running dense-causal across the whole pack

`DeepseekV4MTPLayer` declared `packed_seq_params` and did not pass it on. The inner layer
then received `None`, `_thd_seq_starts` returned `None`, and that **also switched off the
packing guard** — so the MTP depth attended across every sample boundary in the pack, with
no error, no NaN and a finite loss. It only appears with `mtp_num_layers=1`, which the
full-model recipe enables by default, and a PP=2 probe cannot see it because MTP lives on
the last stage.

### A segment cap sized for 8k

`MAX_SEGMENTS_PER_PACK = 256` is documented as generous — for an 8k window. At 128k, Alpaca
wants ~1300 segments per pack, so 256 becomes the binding constraint instead of
`max_seq_length`: a pack holds ~19% real tokens while reporting itself full. That looks like
packing working and is really packing giving up. The cap now scales with the window,
unchanged at ≤8k.

### An empty compressed pool that detached the graph

A sequence contributes `len // ratio` windows, so HCA (ratio=128) has **none** on a pack of
short samples. Returning a bare `new_zeros` there is correct in value and severs
`wkv_gate` / `ape` / `kv_norm` from the graph — those parameters simply never train.
Megatron's DDP catches it one iteration later via an assert that names no parameter; under a
DDP without that check it is completely silent. The pool is now built *through* the
projections and multiplied by zero. This is not an edge case: at 4k it is 22 of 39 packs.

### A cache key missing an input that changes the data

`segment_align` changes pack content but was absent from the pack cache digest, so flipping
`PRIMUS_PACK_ALIGN` silently reloaded the other setting's packs — an aligned-vs-unaligned
A/B on a warm cache compares identical data and reports no difference, which reads as a
finding.

## Alignment: why it was dropped

Segments used to be padded up to a multiple of the largest compress ratio (128), so every
pooling window fell inside one shard. Measured over every pack:

| | supervised tokens | packs with zero HCA windows |
|---|---|---|
| `PRIMUS_PACK_ALIGN=128` | 34.2% | 0 / 64 |
| **unaligned (default)** | **56.1%** | 22 / 39 |

Dropping alignment is worth 1.64× the supervised tokens. But the second column is what
alignment silently bought: padding every segment to a multiple of 128 makes every segment
≥128, so HCA always has a window. Unaligned, HCA has none on 22 of 39 packs — a property of
the data, and what the empty-pool path exists to handle.

## Known limits

* **4k cannot evaluate HCA.** ~40% of packs contain no complete ratio-128 window, so the 20
  HCA layers carry almost no signal. 128k is where that branch becomes meaningful.
* **Step time is not a performance figure** at 128k: `num_microbatches = GBS = 3 = PP`
  leaves the pipeline idle roughly two thirds of the time by construction.
* **Random init**, no DeepSeek-V4 Megatron checkpoint exists — SFT-*shaped* training, and no
  number here says anything about model quality.
* **`micro_batch_size` must be 1**; both the attention module and the fused gather raise
  otherwise.
* **The fused dsv4_cp gather is off by default** and needs the out-of-tree `dsv4_cp_layout`
  package on `PYTHONPATH` (`PRIMUS_DSV4_CP_DIR`) — it is not in the image. It enumerates
  windows by upstream's reach-back rule rather than this repo's strict ownership, so it
  refuses loudly on cross-boundary shards instead of silently pairing rows with the wrong
  identity.

## Image compatibility

`main` has moved ahead of the `primus_turbo` shipped in `rocm/primus:v26.5` in two
independent ways, so these scripts default the affected paths off:

| symptom | switch |
|---|---|
| `grouped_gemm() got an unexpected keyword argument 'fuse_bgrad_accum_pattern'` | `TURBO_USE_GROUPED_MLP=False` |
| `No module named 'primus_turbo.pytorch.ops.moe.fused_mega_moe'` (the image has `mega_moe_fused`) | `PRIMUS_OPT_MEGA_MOE=0` |

**Which switch is the effective one depends on how the recipe is launched**, and the two
chains here differ:

* **Multi-node** (`run_full_*_thd_multinode.sh` → `run_deepseek_v4_flash.sh` →
  `run_deepseek_v4.sh`): `TURBO_USE_GROUPED_MLP` is what matters, *not* the yaml key.
  `run_deepseek_v4.sh` passes `--use_turbo_grouped_gemm "$TURBO_USE_GROUPED_MLP"` on the
  command line, which overrides whatever the config says. That is why
  `run_full_4k_multinode.sh` sets the env var rather than editing the yaml.
* **Single-node** (`run_128k_thd_packed.sh` → `primus-cli direct`): the opposite. That chain
  never enters `run_deepseek_v4.sh`, so nothing overrides the config and the yaml's
  `use_turbo_grouped_gemm: ${PRIMUS_USE_TURBO_GROUPED_GEMM:true}` decides. The single-node
  recipe therefore runs *with* the turbo grouped GEMM, which is deliberate — it is the
  configuration the reproducible step-10 loss reported above (11.71954) was measured under.
  Setting `PRIMUS_USE_TURBO_GROUPED_GEMM=false` would change it.

Set both back on an image with a newer `primus_turbo`; CI builds it from source rather than
using the one in the image.

## Operational notes (multi-node)

* **Pack cache is node-local** (`PRIMUS_PACK_CACHE_DIR`), one build per node. Sharing it over
  NFS looks tidier and is worse: the writer stages to a fixed `<final>.tmp` name and renames,
  so two nodes building concurrently collide and one dies with `ENOENT`. The build is
  deterministic — no seed, hostname, clock or directory order feeds it — so the nodes end up
  with identical packs. Verify it: every node must log the same `[Pack] CACHE HIT key=…` and
  the same pack count.
* **The lock goes on node-local disk** (`PRIMUS_PACK_LOCK_DIR`). With 24 ranks on shared
  storage, one rank builds under the lock while 23 block with filelock's default
  `timeout=-1`; exceeding Megatron's 10-minute collective timeout takes the job down.
* **Clean up with `docker restart`, not `pkill -9`.** Killed ranks leave processes holding
  ~120 GB/GPU; the next run's optimizer setup then cannot allocate and aborts in a way that
  looks exactly like a training bug. Confirm every node is back at its idle baseline (~2 GiB)
  before starting, rather than sleeping a fixed interval.

## Knobs

| variable | default | effect |
|---|---|---|
| `PRIMUS_PACK_ALIGN` | 1 | segment padding multiple; 128 reproduces the old aligned baseline |
| `PRIMUS_PACK_MIN_TOKENS_PER_SEGMENT` | 32 | divisor setting the per-pack segment cap |
| `PRIMUS_INDEXER_TOPK_CHUNK` | 0 (packed path uses 512) | pool-column chunk width |
| `PRIMUS_OPTIMIZER_OFFLOAD_FRACTION` | 0.75 (4k) / 0.9 (128k) | optimizer state moved to host |
| `FUSED_CE_CHUNK` | 4096 | LM-head chunk; required at 128k |
| `PRIMUS_THD_COMPACT_BACKEND` | unset | `torch_native` / `flydsl`; needs `dsv4_cp_layout` |
| `PRIMUS_DSV4_CP_DIR` | unset | checkout of the out-of-tree `dsv4_cp` package |
| `V4_TOKENIZER` | probed | tokenizer directory |
| `ALPACA_JSONL` | `${DATA_DIR}/alpaca_natural.jsonl` | pre-placed corpus, to avoid the Hub |
| `TRAIN_ITERS` | 10 | steps |
