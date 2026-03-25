# Parallelism Strategies for Distributed Training

This guide explains the parallelism dimensions used when training large foundation models on AMD GPUs with Primus. It moves from basic data parallelism to advanced combinations of tensor, pipeline, context, and expert parallelism, including how Primus exposes these options through Megatron-LM and TorchTitan.

For Megatron YAML flags and environment tuning, see [Megatron parameters](../03-configuration-reference/megatron-parameters.md) and [Environment variables](../03-configuration-reference/environment-variables.md).

---

## 1. Introduction

### Why parallelism is needed

Modern foundation models often exceed the memory of a single accelerator: parameters, activations, optimizer states, and KV caches cannot all reside on one device at useful batch sizes. Even when a model *fits*, training throughput may be too low without scaling across many GPUs. Parallelism splits the problem along several independent **dimensions** so that:

- **Memory** is shared across devices (sharding, pipeline stages, sequence splits).
- **Compute** is scaled by processing more data in parallel or by overlapping communication with computation.

### Overview of parallelism dimensions

| Dimension | What is split | Primary goal |
|-----------|----------------|--------------|
| **Data parallelism (DP)** | Input batches | Throughput; same model on each GPU |
| **FSDP / ZeRO** | Parameters, gradients, optimizer (by stage) | Memory; keep DP semantics |
| **Tensor parallelism (TP)** | Individual weight matrices / matmuls | Memory per layer; needs fast links |
| **Sequence parallelism (SP)** | Sequence in non-TP regions | Activation memory with TP |
| **Pipeline parallelism (PP)** | Layer groups across stages | Memory; depth-wise split |
| **Context parallelism (CP)** | Sequence for attention (e.g. ring) | Very long contexts |
| **Expert parallelism (EP)** | MoE experts across devices | Memory and compute for MoE |

These can be **combined**. The product of parallel degrees must match how processes are laid out on the cluster (see [Section 9](#9-combining-parallelism-strategies)).

---

## 2. Data Parallelism (DP)

In **classic data parallelism**, every GPU holds a **full copy** of the model. Each rank receives a **different mini-batch** of data. After the backward pass, **gradients are synchronized** so that all ranks apply the same update.

```
  Batch shard 0     Batch shard 1     Batch shard 2     Batch shard 3
       |                  |                  |                  |
       v                  v                  v                  v
   +--------+        +--------+        +--------+        +--------+
   | GPU 0  |        | GPU 1  |        | GPU 2  |        | GPU 3  |
   | full   |        | full   |        | full   |        | full   |
   | model  |        | model  |        | model  |        | model  |
   +--------+        +--------+        +--------+        +--------+
       |                  |                  |                  |
       +------------------+------------------+------------------+
                          |
                    AllReduce(gradients)
                          |
                          v
              Same weights on all ranks after optimizer step
```

**Properties**

- Simple to reason about and widely supported.
- Requires the **full model, activations for one micro-batch, and optimizer state** to fit in **one GPU’s memory** (unless combined with other strategies).

**Effective batch size**

For a single update that aggregates over data-parallel ranks and gradient accumulation:

\[
\text{effective\_batch\_size} = \text{micro\_batch\_size} \times \text{num\_GPUs}_{\text{DP}} \times \text{gradient\_accumulation\_steps}
\]

Here `num_GPUs_DP` is the **data-parallel group size** (not always the same as `world_size` when TP/PP/EP are also used).

---

## 3. Fully Sharded Data Parallel (FSDP / ZeRO)

**ZeRO** (Zero Redundancy Optimizer) reduces redundant storage by **sharding** optimizer states, gradients, and/or parameters across data-parallel ranks.

| ZeRO stage | Sharded | Idea |
|------------|---------|------|
| **Stage 1** | Optimizer states | Each rank keeps only \(1/N\) of optimizer tensors |
| **Stage 2** | + Gradients | Gradients are sharded; reduced where needed |
| **Stage 3** | + Parameters | Each rank holds \(1/N\) of parameters; gather before use |

**FSDP** (Fully Sharded Data Parallel) in PyTorch is the common **implementation** of sharded data parallel training; in the Megatron ecosystem, **ZeRO-3-style** behavior is often discussed alongside **FSDP** for full parameter sharding.

**Typical execution pattern (conceptual)**

1. **Forward:** **AllGather** (or equivalent) to materialize parameters needed for the current layer/batch on each rank.
2. **Backward:** **ReduceScatter** (or equivalent) to write shard-sized gradient pieces back to ranks.

**Memory intuition**

If replicated training used \(M\) memory per rank for parameters+gradients+optimizer, **ideal** full sharding across \(N\) ranks approaches **\(M/N\)** for the sharded pieces (plus buffers and fragmentation). Moving from **full replication** to **\(1/N\)** sharding for those tensors saves roughly **\((N-1)/N\)** of that component—for **8 GPUs**, about **87.5%** of the replicated footprint for the sharded tensors.

### In Primus

| Backend | Configuration |
|---------|----------------|
| **Megatron-LM** | `use_distributed_optimizer: true` enables the **distributed optimizer** (ZeRO-1–style optimizer sharding in Megatron). For **PyTorch FSDP2**, set `use_torch_fsdp2: true` (see Megatron constraints: FSDP2 and distributed optimizer are not used together). |
| **TorchTitan** | `data_parallel_shard_degree` controls how ranks participate in **FSDP-style** sharding (see TorchTitan job config; `-1` often means auto). |

Exact interactions with checkpoint formats and DDP are documented in [Megatron parameters](../03-configuration-reference/megatron-parameters.md).

---

## 4. Tensor Parallelism (TP)

**Tensor parallelism** splits **individual layers** (usually linear / attention projections) across GPUs so **no single GPU stores the full weight matrix** for that layer.

### Column-parallel vs row-parallel

Consider a linear layer \(Y = X W\) with weight matrix \(W\). **Column-parallel** splits \(W\) **along the output dimension** (columns). **Row-parallel** splits \(W\) **along the input dimension** (rows) and **splits \(X\)** so each rank’s matmul dimensions match.

**Column-parallel linear** — each rank holds **disjoint columns** of \(W\); partial outputs have **half** the width per rank (for 2-way TP), then combine:

```
                    SAME full X replicated on each TP rank
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
      Rank 0: X @ W[:,0:h/2]              Rank 1: X @ W[:,h/2:h]
            |                                   |
            v                                   v
      partial Y_0                         partial Y_1
      (narrow)                            (narrow)
            |                                   |
            +-----------------+-----------------+
                              |
                    AllReduce on output dim
                              |
                              v
                    full-width Y on each rank
```

**Row-parallel linear** — each rank holds **disjoint rows** of \(W\); **input \(X\)** is **split** along the **input feature** dimension so each rank computes part of the reduction:

```
      Rank 0: X_0 @ W[0:r/2,:] ----+
                                   +-- AllReduce --> Y
      Rank 1: X_1 @ W[r/2:r,:] ----+
      (X split along features)     (partial sums add to full Y)
```

Typical **transformer block** pattern: **column-parallel** for one projection, then **row-parallel** for the next so that the **AllReduce** cost is balanced across the stack.

**Communication**

- Often **AllReduce** of partial outputs, or **ReduceScatter** + **AllGather** sequences depending on implementation and **sequence parallelism** (see next section).

**When to use**

- Best **within a node** (NVLink / high-bandwidth GPU–GPU paths). Multi-node TP is possible but latency-sensitive.

### In Primus

| Backend | Parameter |
|---------|-----------|
| Megatron-LM | `tensor_model_parallel_size` |
| TorchTitan | `parallelism.tensor_parallel_degree` |

---

## 5. Sequence Parallelism (SP)

**Sequence parallelism** extends TP by splitting the **sequence dimension** in regions that are not covered by tensor-parallel matmuls—commonly **LayerNorm**, **dropout**, and sometimes **residual** paths—so **activation memory** scales better when **TP > 1**.

**Interaction with TP**

- After a **column-parallel** region, partial activations can be **ReduceScatter**d along the sequence.
- Before a **row-parallel** region, activations may be **AllGather**d along the sequence.

So SP trades **extra collectives** for **lower per-rank activation footprint** on long sequences.

### In Primus

| Backend | Parameter |
|---------|-----------|
| Megatron-LM | `sequence_parallel: true` (used with TP) |
| TorchTitan | Sequence-parallel behavior is integrated with TP/parallelization pipelines in supported models |

---

## 6. Pipeline Parallelism (PP)

**Pipeline parallelism** assigns **disjoint subsets of layers** to **stages** on different devices. Activations (and gradients) move **between stages** with **point-to-point** communication.

```
  Microbatch 1:  Stage0 -> Stage1 -> Stage2 -> Stage3
  Microbatch 2:       Stage0 -> Stage1 -> Stage2 -> Stage3
  ...
```

### Pipeline bubbles

If a stage waits for input while other stages compute, **idle time** appears (**pipeline bubble**). Schedulers reduce bubbles by overlapping forwards and backwards across microbatches.

**Common schedules**

| Schedule | Idea |
|----------|------|
| **1F1B** | One forward, one backward; classic **warmup / steady / cooldown** phases |
| **1F1B interleaved (VPP)** | **Virtual pipeline** stages: multiple chunks per device to improve utilization |
| **Zero-bubble (ZB)** | Reorders / splits backward so **forward and backward** hide each other better; may separate **input-gradient** vs **weight-gradient** phases |
| **V-Schedule / V-Half / V-Min** | Variants reducing bubbles further (names vary by codebase) |
| **DualPipe** | Bidirectional pipeline scheduling (e.g. DeepSeek-style) to overlap forward/backward paths |

**Bubble rate**

\[
\text{bubble\_rate} = \frac{\text{idle time}}{\text{total time}}
\]

Lower is better; large **microbatch counts** and better schedules reduce bubble overhead.

### In Primus (Megatron)

| Parameter | Role |
|-----------|------|
| `pipeline_model_parallel_size` | Number of pipeline stages |
| `patch_zero_bubble` | Enable Primus/Megatron **zero-bubble** pipeline patches |
| `patch_primus_pipeline` | Use Primus pipeline implementation for schedule logic |
| `pp_algorithm` | e.g. `1f1b`, `1f1b-interleaved`, `zero-bubble`, `zbv-formatted`, `v-half`, `v-min` |

See `primus/configs/modules/megatron/primus_pipeline.yaml` and `zero_bubble.yaml` in the repo for defaults.

### In Primus (TorchTitan)

| Parameter | Role |
|-----------|------|
| `parallelism.pipeline_parallel_degree` | Pipeline depth |
| `parallelism.pipeline_parallel_schedule` | e.g. `1F1B`, `Interleaved1F1B`, `GPipe`, zero-bubble variants where supported |

---

## 7. Context Parallelism (CP)

**Context parallelism** splits the **sequence length** across devices for long-context training. A common pattern is **ring attention**: each rank holds a **chunk** of queries/keys/values and participates in a **ring** of message passing so attention covers the full sequence without centralizing all activations on one GPU.

**Use cases**

- Long documents, 32K–128K+ tokens, where **per-layer activation memory** and **attention compute** must be distributed.

### In Primus

| Backend | Parameter |
|---------|-----------|
| Megatron-LM | `context_parallel_size` |
| TorchTitan | `parallelism.context_parallel_degree` |

---

## 8. Expert Parallelism (EP)

**Mixture-of-Experts (MoE)** models route each token to a small subset of **experts**. **Expert parallelism** assigns **different experts** to **different GPUs** so expert weights are not duplicated on every device.

**Communication**

- **AllToAll** (or equivalent) is typical: **dispatch** tokens to expert ranks and **combine** expert outputs back.

**Expert tensor parallelism (ETP)**

- Experts can be further **tensor-parallel** within a subset of GPUs, analogous to TP for dense layers.

### In Primus

| Backend | Parameter |
|---------|-----------|
| Megatron-LM | `expert_model_parallel_size` |
| TorchTitan | `parallelism.expert_parallel_degree`, `parallelism.expert_tensor_parallel_degree` |

---

## 9. Combining Parallelism Strategies

### Common pattern

- **TP within a node** (fast interconnect).
- **PP across nodes or across groups** when layers do not fit on one device.
- **DP / FSDP** for scaling batch size and sharding optimizer state or parameters.

### GPU count (simplified)

For dense models (ignoring CP and detailed MoE layout):

\[
\text{world\_size} \approx \text{TP} \times \text{PP} \times \text{DP}
\]

For MoE-heavy setups, you often see:

\[
\text{world\_size} \approx \text{TP} \times \text{PP} \times \text{EP} \times \text{DP}
\]

**Context parallelism** introduces another multiplicative factor in layouts where CP ranks are part of the global mesh (exact rank ordering is implementation-specific).

### Memory vs communication

- **More TP** → smaller matrices per GPU but **more frequent** collectives within layers.
- **More PP** → less memory per stage but **pipeline bubbles** and **latency** between stages.
- **More DP/FSDP** → better throughput scaling if communication is not saturated.

### Example configurations (illustrative)

| Scenario | TP | PP | DP / notes |
|----------|----|----|------------|
| ~7B on 8 GPUs | 1 | 1 | 8-way DP (or FSDP) |
| ~70B on 64 GPUs (8 nodes × 8) | 8 | 2 | 4-way DP |
| Large MoE (e.g. 671B-class) on 256 GPUs | 8 | 4 | EP 8 (example; real jobs vary widely) |

Always validate against **memory profiling**, **checkpoint sharding**, and **network** on your cluster.

---

## 10. Batch Size Relationships

Let:

- \(B_{\text{micro}}\) = micro-batch size per forward/backward **per data-parallel rank** (per step inside accumulation),
- \(D\) = **data parallel size** (ranks that share the same model split for DP),
- \(G\) = **gradient accumulation** steps,
- \(B_{\text{global}}\) = **global batch size** across all DP ranks for one optimizer update.

Then:

\[
B_{\text{global}} = B_{\text{micro}} \times D \times G
\]

**Data parallel size** from world size (when using TP, PP, EP):

\[
D = \frac{\text{world\_size}}{\text{TP} \times \text{PP} \times \text{EP}}
\]

(If **context parallelism** is present, the denominator must include **CP** in the same way your trainer defines the mesh.)

Solve for accumulation:

\[
G = \frac{B_{\text{global}}}{B_{\text{micro}} \times D}
\]

**Practical notes**

- **Micro batch** drives **per-GPU activation memory** (often linearly in sequence length for attention).
- **Global batch** affects **convergence** and learning dynamics; scaling laws often refer to global batch.
- **Gradient accumulation** increases **time per optimizer step** but **reduces memory** by using smaller \(B_{\text{micro}}\).

Megatron-specific names for batch arguments appear in [Megatron parameters](../03-configuration-reference/megatron-parameters.md).

---

## See also

- [NCCL/RCCL collective operations guide](./collective-operations.md) — which collectives each strategy uses.
- [Megatron parameters](../03-configuration-reference/megatron-parameters.md)
- [Environment variables](../03-configuration-reference/environment-variables.md)
