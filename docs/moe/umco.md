# Unified MoE Communication Orchestrator (UMCO)

UMCO is a scheduling layer for MoE token communication in Primus.

For Megatron integration, UMCO targets the token dispatch and gather phases and keeps the baseline Megatron behavior as fallback.

## Goals

- Compile-first step planning (`MoEStepPlan`) with deterministic chunking and buffer sizing.
- Runtime dispatch/gather orchestration through a dispatcher interface.
- Topology-aware EP grouping plan for future hierarchical communication.
- Safe disable path (default OFF), no changes to RCCL or `torch.distributed` core.

## Current Phase

- **Phase 1 (implemented):**
  - Build and cache step plans.
  - Hook Megatron dispatcher creation through a minimal wrapper patch.
  - Route dispatch/gather through `MoEDispatcher` abstraction.
  - Keep correctness by delegating communication to baseline Megatron dispatcher calls.
  - Build topology and EP grouping plans for logging/future use.
- **Phase 2 (implemented):**
  - Chunked `all_to_all_single` execution for both dispatch and gather paths.
  - Build per-chunk send/recv slices from Megatron token dispatcher metadata
    (`input_splits` / `output_splits` and permutation order already prepared by Megatron).
  - Stitch chunk outputs into full expert input/output tensors deterministically.
  - Inflight control via `max_inflight` to bound outstanding communication ops.
  - Optional CUDA overlap using:
    - `stream_comm` for communication
    - `stream_compute` for expert compute callback and chunk unpack
    - CUDA events for stream dependency ordering.
  - CPU fallback path without streams.
- **Phase 3 (planned):**
  - Pseudo-sparse A2A and deeper communication schedule optimization.

## Enable Flags

UMCO is off by default.

Environment variables:

- `PRIMUS_UMCO_ENABLE=1`
- `PRIMUS_UMCO_CHUNK_TOKENS=2048` (optional)
- `PRIMUS_UMCO_MAX_INFLIGHT=2` (optional)
- `PRIMUS_UMCO_TOPO_ENABLE=1` (optional)
- `PRIMUS_UMCO_LOG_LEVEL=INFO` (optional)
- `PRIMUS_UMCO_VERIFY=1` (optional; for small tensors run baseline + UMCO and compare outputs)

Primus experiment config:

- `exp.moe.comm_orchestrator.enable: bool`
- `exp.moe.comm_orchestrator.chunk_tokens: int`
- `exp.moe.comm_orchestrator.max_inflight: int`
- `exp.moe.comm_orchestrator.topology.enable: bool`

Environment variables override experiment config.

## How To Enable

Example:

```bash
export PRIMUS_UMCO_ENABLE=1
export PRIMUS_UMCO_CHUNK_TOKENS=2048
export PRIMUS_UMCO_MAX_INFLIGHT=2
export PRIMUS_UMCO_TOPO_ENABLE=1
export PRIMUS_UMCO_LOG_LEVEL=INFO
export PRIMUS_UMCO_VERIFY=1
```

Then run your normal Primus Megatron command. The Megatron backend patch entrypoint will install UMCO dispatcher wrapping only when enabled.

## Logging

UMCO logger: `primus.moe_umco`

It logs:

- world info (`rank/world/ep/tp/pp`)
- chosen chunk size and number of chunks
- topology EP groups when topology mode is enabled
