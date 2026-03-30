# Primus Projection Skill

## Purpose
Provide a quick, opinionated guide for using Primus Projection to:
- Choose parallelism and pipeline schedules
- Validate memory fit on target nodes
- Understand communication collectives/algorithms
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

## Quick Interpretation Tips
- If tokens/s improves most when DP increases: you’re compute-bound; check comm overlap.
- If comm breakdown shows large EP A2A: enable DeepEP or reduce inter-node EP links.
- If pipeline bubble ratio is high: increase VPP or switch to ZB/ZBV schedules.
- If memory is tight: recomputation + smaller microbatch + redistribute PP.

## References
- Tech blog: `docs/tech_blogs/projection/projection.md`
- User docs: `docs/projection.md`
- Schedulers: `primus/core/pipeline_parallel/scheduler/`
- Simulation backends: Origami (GEMM), SDPA simulator (attention)
