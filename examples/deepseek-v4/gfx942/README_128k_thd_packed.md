# DeepSeek-V4 SFT on packed sequences (THD) at 128k, gfx942

Runs a 3-layer DeepSeek-V4 SFT at **128k context** on **packed** sequences — many real SFT
samples concatenated into one 128k window and delimited by `cu_seqlens` — exercising all
three V4 attention branches (dense+SWA, CSA, HCA) on a single 8×MI308X node.

```bash
# inside a rocm/primus:v26.5-pytorch2.12-te2.15 container, from the repo root
bash examples/deepseek-v4/gfx942/run_128k_thd_packed.sh
```

See [README.md](README.md) for the gfx942 kernel settings and the CP-not-TP argument, which
this recipe inherits unchanged.

## Prerequisites

* A local **DeepSeek-V4 tokenizer directory** (`V4_TOKENIZER`, default
  `/apps/DeepSeek-V4-Flash`). Only `tokenizer.json` / `tokenizer_config.json` are read.
* **Exactly 8 GPUs.** `PRIMUS_CP=8` and `PRIMUS_EP=8` are assigned unconditionally in the
  script, so `GPUS_PER_NODE` cannot usefully be changed.
* **Network access to the HF Hub** on first run, to fetch Alpaca. Pre-place the corpus and
  point `ALPACA_JSONL` at it to run offline.
* `third_party/Megatron-LM` present, or a reachable git remote (the script copies it out of
  the image when the image has it, else fetches the submodule).
* **Time and disk**: the first run tokenizes and bin-packs 52002 samples (several minutes,
  cached afterwards). Even on a cache hit, iteration 1 takes ~70 s including Triton warm-up;
  the full 10 steps take about 3 minutes wall-clock.

Verified end to end from a **pristine `rocm/primus:v26.5-pytorch2.12-te2.15` container**
with empty data / output / cache directories: exit 0, loss bit-identical to the development
container.

## Why packing is not just a data change

Alpaca's runtime segments have a median of **84 tokens**. An *unpacked* 128k sequence is
therefore almost entirely padding, so a 128k BSHD run measures the hardware and little else.
Packing is what makes the context length mean something.

But packing puts sequence boundaries at arbitrary offsets, and V4's compressor pools fixed
windows of `ratio` rows **anchored at each sequence's start**. Under context parallelism a
window can then straddle a shard boundary, with its leading rows owned by the left
neighbour. Two mechanisms make that work:

* **Left-boundary exchange** (`deepseek_v4_cp.exchange_boundary_hidden`) ships the
  neighbour's trailing rows, so no alignment padding is needed.
* **Strict window ownership** — a window belongs to the rank holding its *last* row — plus a
  fixed per-rank capacity. The capacity has to be fixed because `_AllGatherPool` sizes its
  receive buffers with `torch.empty_like(pool_local)`: if two ranks disagreed on the pool
  width, the all-gather's shapes would not match.

Attention isolation is **STRICT**: no query may attend across a packed boundary.

## Results

8×MI308X, 3 layers, `compress_ratios [0, 4, 128, 0]` (the 4th entry is the MTP slot),
CP=8 / TP=1 / EP=8, MBS=1, 10 steps, at the default indexer chunk of 512:

| | run A | run B | pristine container |
|---|---|---|---|
| loss @ step 10 | 11.71951 | 11.71954 | **11.71954** |
| grad norm | 15.323 | 15.323 | 15.323 |
| step time | 4356 ms | 4332 ms | 4311 ms |
| nan iterations | 0 | 0 | 0 |

Loss starts at 11.899 and is reproducible **to five significant figures**; step time and
memory are not, and depend on what else shares the node.

Peak memory reads **119.04 GB (62.01%)**, but note what that number is: the logger reports
`torch.cuda.mem_get_info()`, i.e. **device-wide** usage including any other process on the
GPU. It was taken on an otherwise-idle node (0.3 GiB/GPU baseline), so it is close to this
job's own footprint — but it is not the allocator's `max_memory_allocated`.

### Correctness

| check | how to run | result |
|---|---|---|
| CP=1 vs CP=2 and CP=4 | manual, see below | dense / CSA / HCA all match; max 3.9e-3, mean 2.5e-5 |
| packed vs unpacked, three branches | `pytest test_deepseek_v4_thd_packing.py` | pass |
| window ownership vs upstream's rule | `pytest test_thd_compact_plan.py` | pass |
| streaming top-K vs one-shot scoring | `pytest test_thd_indexer_streaming_topk.py` | pass |
| empty pool keeps its gradient | `pytest test_compressor_empty_pool_grad.py` | pass |

The pytest files need `PRIMUS_V4_UT_ALLOW_NON_MI355X=1` on gfx942. `thd_cp_equivalence.py` is
**not** a pytest module and is **forward-only**; run it by hand:

```bash
D=tests/unit_tests/megatron/transformer/deepseek_v4
for n in 1 2 4; do
  torchrun --nproc_per_node=$n $D/thd_cp_equivalence.py --out /tmp/cp$n.pt \
      --seq-lens 333 191 277 223
done
python $D/thd_cp_equivalence.py --compare /tmp/cp1.pt /tmp/cp2.pt
python $D/thd_cp_equivalence.py --compare /tmp/cp1.pt /tmp/cp4.pt
```

Pass `--seq-lens` explicitly: the default `[512, 256, 128, 128]` puts every shard boundary on
a sequence start, which is the one layout that does *not* exercise straddling windows. The
script's own verdict uses `rtol=atol=2e-2`; the 3.9e-3 above is the observed `max|diff|`
inside that tolerance. CP=4 matters more than CP=2 — at CP=2 this layout gives both ranks
equal window counts and never reaches the fixed-capacity path, a trap
`test_thd_compact_plan.py` pins shut deliberately.

## Alignment: why it was dropped

Segments used to be padded up to a multiple of the largest compress ratio (128), so every
pooling window fell inside one shard. Measured over every pack `PackedSFTDataset` produces:

| | supervised tokens | packs with zero HCA windows |
|---|---|---|
| `PRIMUS_PACK_ALIGN=128` | 34.2% | 0 / 64 |
| **unaligned (default)** | **56.1%** | 22 / 39 |

Dropping alignment is worth 1.64× the supervised tokens. But the second column is the thing
alignment silently bought: padding every segment to a multiple of 128 makes every segment
≥128, so HCA always has a window. Unaligned, **HCA has no window at all on 22 of 39 packs**
— a property of the data, and what the empty-pool path below exists to handle.

## Things that bit, and what they looked like

Each produced a *plausible-looking run*. None announced itself as an error.

### An exit-0 run that trained nothing

Feeding the training pipeline the output of `prepare_packed_data.py` — a pre-packed
`{input_ids, labels, cu_seqlens}` jsonl — completes 10/10 with exit 0, zero nan, and a
sensible step time. It also has **grad norm 0.000 and no loss line**, because there is no
loader for that schema: the alpaca formatter finds no `instruction`/`output` fields, every
sample tokenizes to nothing, and the loss mask is all zero.

`prepare_packed_data.py` is an **offline measurement aid**, not a training input. The packing
that matters happens at runtime in `PackedSFTDataset`, from an alpaca-format corpus, and is
what `sft_packing_segment_align` applies to.

### A segment cap sized for 8k

`MAX_SEGMENTS_PER_PACK = 256` is documented as generous — for an 8k window. At 128k, Alpaca
wants ~1300 segments per pack on average (up to ~2800), so 256 becomes the binding
constraint instead of `max_seq_length`: a pack holds ~19% real tokens (10.7% supervised)
while reporting itself full. That looks like packing working and is really packing giving up.
The cap now scales with the window, unchanged at ≤8k.

(Adding `max_segments` and `segment_align` to the cache digest does invalidate every
*existing* pack cache once. That is a one-off rebuild, not a correctness issue.)

### An empty compressed pool that detached the graph

A sequence contributes `len // ratio` windows, so HCA (ratio=128) has **none** on a pack of
short samples. Returning a bare `new_zeros` there is correct in value and severs
`wkv_gate` / `ape` / `kv_norm` from the graph — those parameters simply never train.
Megatron's DDP catches it one iteration later via an assert that names no parameter and
points at its own grad buffer; under a DDP without that check it is completely silent. The
pool is now built *through* the projections and multiplied by zero: same value, parameters
still in the graph.

This is not an edge case — it is 22 of 39 packs.

### A cache key missing an input that changes the data

`segment_align` changes pack content but was absent from the pack cache digest, so flipping
`PRIMUS_PACK_ALIGN` silently reloaded the other setting's packs. This fails in the worst
direction: an aligned-vs-unaligned A/B on a warm cache compares identical data and reports
no difference, which reads as a finding. (It is how the aligned column above was nearly
mismeasured.)

### 4 GiB hiding behind a reduced axis

The packed indexer scores in chunks over the pool. The transient that sizes the step is the
4-D `dot_c` = `[B, S, H, chunk]` *before* the head axis is reduced away: at 128k / CP=8 /
64 heads that is 4.0 GiB at `chunk=2048` against 1.0 GiB at 512. The packed default is now
512, which drops the device-wide peak from 99.90% to 62.01% with the loss unchanged to five
significant figures and no measurable step-time cost.

Runs at `chunk=2048` did also OOM, but that evidence is confounded: those failures happened
while other jobs shared the node (one OOM'd with only 35.8 GiB attributable to this job on a
full card). The 4 GiB → 1 GiB arithmetic stands on its own; the OOMs do not prove it alone.

## Known limits

* **HCA is largely inert on Alpaca.** 23% of runtime segments reach 128 tokens, but
  first-fit-decreasing packing concentrates them: HCA has no window at all on 22 of 39 packs.
  It no longer breaks anything, but making HCA *learn* needs a corpus with longer samples.
* **`micro_batch_size` must be 1.** Both the attention module and the fused gather raise
  otherwise.
* **Single node only.** Multi-node THD is untested.
* **Random init**, 3 layers not 43 — this is SFT-*shaped* training from scratch, not
  fine-tuning, and no number here says anything about model quality.
* **The fused dsv4_cp gather is off by default**, and requires the out-of-tree
  `dsv4_cp_layout` package on `PYTHONPATH` — it is not in the image. Setting
  `PRIMUS_THD_COMPACT_BACKEND` without it raises rather than falling back. It also enumerates
  windows by upstream's reach-back rule rather than this repo's strict ownership, so it
  refuses loudly on cross-boundary shards instead of silently pairing rows with the wrong
  identity. Its measured saving was a small fraction of a step; it is not worth the risk here.

## Knobs

| variable | default | effect |
|---|---|---|
| `PRIMUS_PACK_ALIGN` | 1 | segment padding multiple; 128 reproduces the old aligned baseline |
| `PRIMUS_PACK_MIN_TOKENS_PER_SEGMENT` | 32 | divisor setting the per-pack segment cap |
| `PRIMUS_INDEXER_TOPK_CHUNK` | 0 (packed path then uses 512) | pool-column chunk width |
| `PRIMUS_THD_COMPACT_BACKEND` | unset | `torch_native` / `flydsl`; needs `dsv4_cp_layout` |
| `PRIMUS_DSV4_CP_DIR` | unset | checkout of the out-of-tree `dsv4_cp` package; its tests skip without it |
| `V4_TOKENIZER` | `/apps/DeepSeek-V4-Flash` | tokenizer directory |
| `ALPACA_JSONL` | `${DATA_DIR}/alpaca_natural.jsonl` | pre-placed corpus, to avoid the Hub |
| `TRAIN_ITERS` | 10 | steps |
