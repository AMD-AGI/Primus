# Primus Projection Skill

## Purpose
Provide a quick, opinionated guide for using Primus Projection to:
- Choose parallelism and pipeline schedules
- Validate memory fit on target nodes
- Understand communication collectives/algorithms
- Explore optimization trade-offs with minimal compute
- Run the right commands fast

## Scope
- Docs: `docs/tech_blogs/projection/projection.md`, `docs/projection.md`
- **Projection code**: `primus/core/projection/`
- **Projection performance entrypoints**: `primus/core/projection/performance_projection/`
- **Projection memory entrypoints**: `primus/core/projection/memory_projection/`
- Pipeline schedulers: `primus/core/pipeline_parallel/scheduler/`
- Example configs: `examples/megatron/configs/`
- Hardware configs: `examples/hardware_configs/`

## Projection Advisor (What users want to know)

1) Best parallelism strategy
- Evaluate TP/PP/EP/CP/DP for the target cluster using performance projection.
- Guidance:
  - Prefer keeping TP intra-node (for faster AR); scale EP and DP across nodes.
  - For MoE, use EP that avoids inter-node A2A when possible; otherwise enable DeepEP.
  - Use CP only when sequence length forces it; in MoE, CP may be folded into EP.
  - Increase DP last (weak scaling) once a minimal viable PP×EP×TP is established.

2) Best pipeline schedule
- Compare (for the chosen PP/VPP):
  - 1F1B
  - Interleaved 1F1B (VPP > 1)
  - Zero-Bubble (B/W split, VPP = 1)
  - ZBV Formatted (VPP = 2)
  - ZBV Greedy (VPP = 2, memory modes: min/half)
  - Megatron ILP (VPP = 1, where available)
- Recommendation pattern:
  - Dense models: interleaved or ZB depending on VPP feasibility and memory.
  - MoE (e.g., Mixtral): VPP=2 with ZBV Formatted often wins, especially with DeepEP ON.

3) Memory fit at target nodes
- Run memory projection first. Review:
  - Parameters + optimizer (sharded by DP)
  - Activations (attention vs MoE MLP; MoE often dominates)
  - Pipeline scaling: peak activations scale with PP and VPP schedule
- If not fitting:
  - Enable recomputation (full or partial)
  - Adjust microbatch size and PP/VPP
  - Reduce top-k or expert sizes for MoE if acceptable

4) Communication collectives and algorithms
- In use (typical):
  - AllReduce: TP and DP gradients (best-of Ring/Hypercube/Bruck/Single-shot)
  - All-to-All: EP token dispatch/combine (pairwise, topology-aware)
  - Reduce-Scatter / All-Gather: optimizer/activation sharding (if configured)
  - P2P Send/Recv: PP stage activations/gradients
- Distinguish intra-node (xGMI/NVLink/UALink) vs inter-node (IB/RoCE). Provide `--hardware-config` to model your network accurately.

5) Sample command lines

```bash
# Memory projection (check fit)
bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection memory \
  --config <model_config.yaml>

# Performance projection (benchmark mode)
bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection performance \
  --config <model_config.yaml> \
  --target-nodes <N>

# Performance projection (simulation-only, no GPU)
bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection performance \
  --config <model_config.yaml> \
  --profiling-mode simulate \
  --gpu-arch <mi300x|mi325x|mi355x> \
  --target-nodes <N>

# Sub-node benchmarking (e.g., on 1 GPU), project to many nodes
bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection performance \
  --config <model_config.yaml> \
  --benchmark-gpus 1 \
  --target-nodes <N>

# Compare benchmark vs simulation (accuracy check)
bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection performance \
  --config <model_config.yaml> \
  --profiling-mode both \
  --target-nodes <N>

# Override parallelism from environment (optional)
export PRIMUS_TP=1
export PRIMUS_PP=3
export PRIMUS_EP=8

bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection performance \
  --config <model_config.yaml> \
  --target-nodes <N>

# Provide cluster network topology
bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection performance \
  --config <model_config.yaml> \
  --hardware-config examples/hardware_configs/mi355x.yml \
  --target-nodes <N>
```

## Optimization Space Exploration

Primus can project throughput and memory across dozens of optimization
combinations from a single-node benchmark (8 GPUs), eliminating the need
for expensive multi-node runs. This was validated on DeepSeek V2 (236B MoE)
across 17 configurations on 4-node MI355X — the projection tracked measured
results across all optimization combinations and correctly identified the
highest-impact optimizations.

### Category 1: MoE Communication Optimizations (CLI flags)

| Optimization | CLI Flag / Config Key | What It Does | Projection Impact |
|---|---|---|---|
| **DeepEP** | `--enable-deepep` / `use_turbo_deepep: true` | Async A2A overlap with compute for MoE. Pipelining A2A dispatch/combine behind expert GEMM. | **High** — largest single-optimization throughput gain (35%+ for DeepSeek V2). A2A critical-path reduced by 65–85%. |
| **SyncFree MoE** | `--sync-free-stage {1,2,3}` | Progressive fusion: 1=fused router, 2=+DeepEP+grouped GEMM, 3=+fused activations. Auto-enables DeepEP. | **Medium–High** — stage 3 achieves 85% A2A overlap (vs. 65% baseline). Each stage adds more fusion. |
| **Target EP size** | `--target-ep-size N` | Override EP for projection (test EP=4 vs EP=8 without changing config). | **High** — EP determines A2A volume and whether A2A stays intra-node or crosses nodes. |

### Category 2: Pipeline Schedule & Layout (CLI flags + config)

| Optimization | CLI Flag / Config Key | What It Does | Projection Impact |
|---|---|---|---|
| **Pipeline schedule** | `--pipeline-schedule {auto,zerobubble,zerobubble-heuristic,zbv-formatted,zbv-greedy-half,zbv-greedy-min,seaailab-ilp,all}` | Select pipeline scheduling algorithm. Use `all` to compare every algorithm in one run. | **Medium** — +8–13% for dense models (ZB vs 1F1B); less for MoE with small PP. |
| **Zero-Bubble enable** | `--enable-zero-bubble` / `enable_zero_bubble: true` | Enable B/W split zero-bubble scheduling. | **Medium** — near-zero pipeline idle time, at higher peak memory. |
| **VPP (interleaving)** | `--num-virtual-stages-per-pipeline-rank N` | Virtual pipeline stages per rank, reduces bubble ratio. | **Medium** — VPP=2–4 reduces bubble; higher VPP adds P2P overhead and memory. |
| **Pipeline layout (uneven)** | `pipeline_model_parallel_layout: "Et*4\|t*4\|..."` (config) | String encoding of per-stage layer assignments. E.g., put embedding on first stage with fewer transformer layers. | **Medium** — balances stage compute times; reduces bubble from stage imbalance. |
| **First/last stage layers** | `decoder_first_pipeline_num_layers: N` / `decoder_last_pipeline_num_layers: N` (config) | Override layer count for first/last PP stages (simpler than full layout string). | **Medium** — simpler way to handle embedding/output imbalance. |

### Category 3: Memory Optimizations (config)

| Optimization | Config Key | What It Does | Projection Impact |
|---|---|---|---|
| **Full recomputation** | `recompute_granularity: full` + `recompute_num_layers: N` | Recompute all activations for the first N layers per virtual chunk. Saves peak activation memory at the cost of extra forward pass. | **Memory: High** — dramatic HBM reduction (up to 60%+ activation savings). Performance: slight decrease from recompute overhead. |
| **Selective recomputation** | `recompute_granularity: selective` | Recompute only expensive activations (e.g., attention softmax) while keeping cheap ones. **Note:** memory projection currently models `full` only; `selective` is a config hint for the runtime. | **Memory: Medium** — partial savings without full recompute cost. |
| **Loss fusion** | `cross_entropy_loss_fusion: true` | Fuses cross-entropy into output GEMM pipeline; eliminates full logits materialisation. | **Memory: Medium** — logits activation → 0 bytes. Compute: removes separate softmax/CE pass. |
| **FSDP2** | `use_torch_fsdp2: true` | Fully Sharded DP: replaces gradient AllReduce with ReduceScatter/AllGather. Enables per-layer overlap model for FSDP comm. | **Memory: High** — shards optimizer state + gradients across DP ranks. Performance: overlap model hides most FSDP comm behind compute. |
| **Distributed optimizer** | `use_distributed_optimizer: true` | ZeRO-1-style optimizer sharding across DP ranks. | **Memory: Medium** — shards optimizer state (Adam moments). Less aggressive than FSDP2. |

### Category 4: Parallelism Configuration (config + env)

| Optimization | Config Key / Env Var | What It Does | Projection Impact |
|---|---|---|---|
| **Tensor parallelism** | `tensor_model_parallel_size: N` | Shard attention/MLP across N GPUs within a node. Adds TP AllReduce per layer. | **Perf** — reduces per-GPU compute; adds intra-node comm. Keep TP ≤ node_size. |
| **Pipeline parallelism** | `pipeline_model_parallel_size: N` | Distribute layers across N stages. Adds pipeline bubbles + P2P comm. | **Perf + Memory** — reduces per-GPU layers/activations; adds bubble overhead. |
| **Expert parallelism** | `expert_model_parallel_size: N` / `--target-ep-size N` | Distribute MoE experts across N ranks. Adds A2A dispatch/combine. | **Perf** — EP intra-node is fast; inter-node EP is expensive (use DeepEP). |
| **Context parallelism** | `context_parallel_size: N` (or `context_model_parallel_size`) | Split sequence length across N ranks for long-context training. | **Perf + Memory** — reduces per-GPU sequence length and activation memory. Use only when seq_len forces it. |
| **Data parallelism** | `data_parallel_size: N` (derived from world_size / (TP×PP×CP)) | Replicate model across N groups. Adds gradient AllReduce. | **Perf** — weak scaling; increase last after TP/PP/EP are set. |
| **Env overrides** | `PRIMUS_TP`, `PRIMUS_PP`, `PRIMUS_EP`, `NNODES`, `GPUS_PER_NODE` | Override parallelism and cluster size from environment. | — |

### Category 5: Precision & Attention (config)

| Optimization | Config Key | What It Does | Projection Impact |
|---|---|---|---|
| **FP8 precision** | `fp8: hybrid` (or any non-null value) | FP8-hybrid GEMMs for linear layers. Halves compute time; changes comm-to-compute ratio. | **High** — ~2× compute speedup for matmuls; projection correctly models FP8 GEMM throughput. |
| **MLA (Multi-head Latent Attention)** | `multi_latent_attention: true` + `q_lora_rank`, `kv_lora_rank`, `qk_head_dim`, `v_head_dim`, `qk_pos_emb_head_dim` | LoRA-factored Q + compressed KV projections (DeepSeek V2/V3). Changes attention GEMM count (6 fwd + 12 bwd) and SDPA dimensions. | **Medium** — changes attention compute profile and KV cache memory. Auto-enables turbo parallel linear. |
| **GQA (Grouped Query Attention)** | `group_query_attention: true` + `num_query_groups: N` | Fewer KV heads than Q heads; reduces KV projection GEMMs and memory. | **Memory + Perf** — less KV activation memory; smaller TP AllReduce for KV. |
| **Flash Attention** | `use_flash_attn: true` | Tiled, fused attention kernel. Changes activation memory profile (no full softmax buffer). | **Memory** — eliminates O(seq²) softmax buffer in activation memory estimate. |
| **SwiGLU** | `swiglu: true` | 3 FFN projections instead of 2 (gate + up + down). Changes MLP GEMM count. | **Perf + Memory** — more MLP compute but often better model quality per FLOP. |

### Category 6: Batch & Gradient Configuration (CLI + config)

| Optimization | CLI Flag / Config Key | What It Does | Projection Impact |
|---|---|---|---|
| **Micro-batch size** | `--micro-batch-size N` / `micro_batch_size: N` | Tokens per GPU per microbatch step. Larger → better GPU utilisation but more activation memory. | **Memory + Perf** — directly scales activation memory; affects GEMM tile efficiency. |
| **Global batch size** | `--global-batch-size N` / `global_batch_size: N` | Total batch across all DP ranks. Determines gradient accumulation steps. | **Perf** — more GA steps amortise pipeline bubbles and comm overhead. Memory: GA-like activation scaling factor. |
| **Gradient overlap** | `overlap_grad_reduce: true/false` | Whether gradient AllReduce overlaps with backward compute. MoE with EP>1 may force no overlap for expert gradients. | **Perf** — when true, gradient comm is hidden; when false, full AllReduce is on critical path. |

### Category 7: Network / Hardware Tuning (hardware config YAML)

| Parameter | Key in `hardware_config:` | What It Tunes |
|---|---|---|
| **Intra-node bandwidth** | `node_bw` | xGMI/NVLink/UALink bandwidth (GB/s). Affects TP AllReduce, intra-node A2A. |
| **Inter-node bandwidth** | `pod_bw` | Per-NIC IB/RoCE bandwidth (GB/s). Affects DP AllReduce, inter-node A2A, P2P. |
| **Inter-pod bandwidth** | `cluster_bw` | Cross-pod bandwidth (GB/s). For multi-pod clusters. |
| **Latencies** | `node_lat`, `pod_lat`, `cluster_lat` | Per-domain latency (µs). Matters for small messages and P2P. |
| **NICs per node** | `nics_per_node` | Aggregate inter-node bandwidth = `pod_bw × nics_per_node`. |
| **Bandwidth efficiency** | `bw_eff` | Global efficiency factor applied to all bandwidths (default ~0.91). |
| **P2P efficiency** | `p2p_bw_eff` | Separate efficiency for point-to-point (pipeline) transfers. |
| **A2A tuning** | `a2a_peer_lat`, `a2a_intra_node_peer_lat`, `a2a_mesh_contention`, `a2a_remote_contention` | Fine-grained All-to-All latency and contention parameters. |
| **AllReduce tuning** | `ar_overlap_factor`, `ar_warmup_chunk_bytes`, `rccl_overhead_us`, `nic_rdma_setup_us` | AllReduce pipelining and RDMA setup overhead. |
| **Node/pod topology** | `node_size`, `pod_size`, `switch_topology`, `node_topology` | Topology structure for collective algorithm selection. |

### Optimization Exploration Guidelines

**Step 1: Establish baseline.** Run projection with no optimization flags to
get the baseline throughput and memory.

```bash
bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection performance \
  --config <model_config.yaml> \
  --target-nodes <N> \
  --hardware-config examples/hardware_configs/mi355x.yml
```

**Step 2: Parallelism search.** Sweep TP/PP/EP/CP combinations:
- Keep TP intra-node (TP ≤ 8 for 8-GPU nodes)
- For MoE: try EP=8 (intra-node) vs EP=16+ (inter-node + DeepEP)
- Use CP only when sequence length requires it
- Compare PP=1 (no bubble) vs PP=2/4 (less memory, more bubble)

**Step 3: Pipeline schedule comparison.** Use `--pipeline-schedule all` to
compare every algorithm in a single run:

```bash
... --pipeline-schedule all
```

Try uneven pipeline layouts when first/last stages have embedding overhead:

```yaml
# In config:
decoder_first_pipeline_num_layers: 6   # fewer layers on first stage (has embedding)
decoder_last_pipeline_num_layers: 6    # fewer layers on last stage (has output layer)
```

**Step 4: Sweep MoE optimizations (for MoE models).** Recommended order:

1. **DeepEP** — single largest gain: `--enable-deepep`
2. **SyncFree stages** — progressive fusion: `--sync-free-stage 2`, then `3`
3. **EP size** — compare: `--target-ep-size 4` vs `--target-ep-size 8`

**Step 5: Precision and compute optimizations.**

```yaml
# In config — compare BF16 vs FP8:
fp8: hybrid          # ~2× compute speedup for linear layers
fp8: null            # BF16 baseline

# Loss fusion (MoE + large vocab):
cross_entropy_loss_fusion: true

# SwiGLU (if model supports it):
swiglu: true
```

**Step 6: Memory-side sweep.** Find the best memory/throughput trade-off:

```yaml
# Full recomputation — minimum memory, maximum recompute overhead:
recompute_granularity: full
recompute_num_layers: 80   # all layers

# Selective recomputation — partial savings, less overhead:
recompute_granularity: selective

# No recomputation — maximum throughput, highest memory:
recompute_granularity: null

# FSDP2 — shard optimizer + gradients across DP ranks:
use_torch_fsdp2: true

# Distributed optimizer — shard optimizer state only:
use_distributed_optimizer: true
```

Compare micro-batch sizes to find the sweet spot:

```bash
... --micro-batch-size 1    # low memory, low GPU util
... --micro-batch-size 2    # balanced
... --micro-batch-size 4    # high GPU util, high memory
```

**Step 7: Network tuning.** If projections show large comm overhead,
tune the hardware config to match your actual cluster:

```yaml
hardware_config:
  node_bw: 896.0          # Measured intra-node BW (GB/s)
  pod_bw: 48.0            # Measured per-NIC BW (GB/s)
  bw_eff: 0.88            # Measured efficiency
  nics_per_node: 8
  a2a_peer_lat: 0.5       # Tune for your fabric
```

**Step 8: Combine best options.** Once you know which optimizations help
most individually, combine them and project the final configuration.

### What the Projection Captures vs. Doesn't

| Captured by Projection | Not Fully Captured |
|---|---|
| DeepEP A2A-compute overlap (configurable efficiency per sync-free stage) | SmartFuse kernel fusion speedups |
| SyncFree stage overlap tiers (65%→75%→80%→85%) | Non-Blocking dispatch latency hiding |
| Loss fusion (compute + memory elimination) | Lazy Free memory reclaim timing |
| Pipeline schedule bubble time (7 algorithms including ILP) | `sequence_parallel` (Megatron flag; projection uses CP instead) |
| Pipeline layout (even, uneven, string-encoded) | `gradient_accumulation_fusion` (runtime only) |
| VPP interleaving overhead and P2P cost | `tp_comm_overlap` (runtime only) |
| MLA attention GEMM profile (6 fwd + 12 bwd GEMMs) | `optimizer_cpu_offload` (runtime only) |
| Full recomputation memory + compute trade-off | Selective recompute (config hint; memory models `full` only) |
| FSDP2 per-layer overlap model | Runtime memory fragmentation |
| FP8 vs BF16 GEMM throughput | OS/driver scheduling jitter |
| GQA KV head reduction | `ddp_bucket_size` tuning |
| Flash Attention activation memory savings | `async_tensor_model_parallel_allreduce` |
| Hardware topology (3-domain bandwidth/latency model) | — |
| Batch size effects on utilisation and memory | — |

**Key insight from validation:** The projection correctly identifies the
*relative ranking* of optimizations — which matters most for decision-making.
Even when absolute error is 10–18%, the ordering (DeepEP >> SyncFree > VPP
> LF) is preserved, so you can confidently narrow the search space on the
simulator before running expensive full-scale validation.

## Quick Interpretation Tips
- If tokens/s improves most when DP increases: you're compute-bound; check comm overlap.
- If comm breakdown shows large EP A2A: enable DeepEP or reduce inter-node EP links.
- If pipeline bubble ratio is high: increase VPP or switch to ZB/ZBV schedules.
- If memory is tight: recomputation + smaller microbatch + redistribute PP.
- If DeepEP shows big gains: try `--sync-free-stage 2` or `3` for additional overlap.
- If MoE A2A dominates comm: check whether EP fits intra-node; if not, DeepEP is essential.
- If absolute error seems high: check measurement variance first (re-run measured baseline).

## References
- Tech blog: `docs/tech_blogs/projection/projection.md`
- User docs: `docs/projection.md`
- Schedulers: `primus/core/pipeline_parallel/scheduler/`
- Simulation backends: Origami (GEMM), SDPA simulator (attention)
