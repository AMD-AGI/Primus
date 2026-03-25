# Troubleshooting Guide

This guide is the primary reference for diagnosing and resolving common failures when running Primus on AMD GPUs (ROCm, RCCL, Docker). It complements the [CLI reference](../02-user-guide/cli-reference.md), [Preflight](../02-user-guide/preflight.md), [Benchmarking](../02-user-guide/benchmarking.md), and [Configuration system](../02-user-guide/configuration-system.md) documentation.

---

## 1. Diagnostic tools

Use these tools before scaling a job or when a failure is hard to localize.

| Tool | Purpose |
|------|---------|
| `primus-cli --debug` | Enables verbose logging in `primus-cli` for command construction, delegation, and runtime details. |
| `primus-cli --dry-run` | Prints the commands Primus would run without executing them—useful to verify wrappers, paths, and MPI/launcher wiring. |
| `primus-cli --export_config <path>` | Writes the **fully merged** YAML after presets, `extends:`, and overrides—use to confirm effective values. |
| `NCCL_DEBUG=INFO` | Surfaces detailed RCCL/NCCL connection and collective logs (set in the job environment). |
| `primus-cli direct -- preflight --host --gpu --network` | Fast host, GPU, and network validation. See [Preflight](../02-user-guide/preflight.md). |
| `PRIMUS_PATCHES=none` | Disables Primus patches to the selected backend to isolate whether a failure is Primus-specific or upstream. |

**Examples**

```bash
# Verbose CLI + dry run (no training executed)
primus-cli --debug --dry-run train --config path/to/config.yaml

# Export resolved config for inspection
primus-cli train --config path/to/config.yaml --export_config /tmp/resolved.yaml

# Preflight: environment validation only
primus-cli direct -- preflight --host --gpu --network

# RCCL/NCCL verbose logs in the training process environment
export NCCL_DEBUG=INFO

# Isolate backend vs. Primus patch layer
export PRIMUS_PATCHES=none
```

For end-to-end checks including optional performance probes, see [Preflight](../02-user-guide/preflight.md) (`preflight --perf-test`).

---

## 2. Out of memory (OOM) errors

**Symptoms:** HIP/CUDA OOM messages, worker processes killed by the OOM killer, or abrupt exits during forward/backward.

**Primary levers and mitigations**

| Approach | What to change |
|----------|------------------|
| Reduce per-device activation memory | Lower **`micro_batch_size`** (often the first knob). |
| Shard weights/activations across devices | Increase **tensor parallelism** (`tensor_model_parallel_size` or `parallelism.tensor_parallel_degree`, depending on backend). See [Parallelism configuration](../04-technical-guides/parallelism-configuration.md). |
| Trade compute for memory | **Activation recomputation:** `recompute_granularity: full`, `recompute_method: uniform`, `recompute_num_layers: <N>`. |
| Shard optimizer state | **`use_distributed_optimizer: true`** (Megatron-style stacks). |
| FSDP / sharded data parallel | Megatron: **`use_torch_fsdp2: true`**. TorchTitan: increase **`data_parallel_shard_degree`**. |
| CPU offload | **`optimizer_cpu_offload: true`** where supported. |
| Sequence length | Reduce **maximum sequence length** if the workload allows. |
| MoE / scratch memory | Set **`HSA_NO_SCRATCH_RECLAIM=1`** to reduce scratch-memory conflicts on some MoE workloads. |
| Plan before you run | **`primus-cli ... -- projection memory --config <yaml>`**—see [Projection](../02-user-guide/projection.md). |

**Related references:** [Parallelism strategies](../04-technical-guides/parallelism-strategies.md), [Performance tuning](../04-technical-guides/performance-tuning.md), [Megatron parameters](../03-configuration-reference/megatron-parameters.md), [TorchTitan parameters](../03-configuration-reference/torchtitan-parameters.md).

---

## 3. Distributed communication failures

**Symptoms:** Hangs at initialization, NCCL/RCCL timeouts, connection errors, or inconsistent ranks.

| Cause | What to verify / fix |
|-------|----------------------|
| Wrong network interface | Set `NCCL_SOCKET_IFNAME` to the correct interface; exclude virtual interfaces, e.g. `^docker0,lo`. |
| `MASTER_ADDR` unreachable | From every node, resolve DNS/IP consistently; verify firewalls and routing. |
| Port already in use | Change **`MASTER_PORT`** to a free port on all nodes. |
| InfiniBand not used or missing | Confirm `/dev/infiniband` exists where expected; run `ibstat`; ensure IB kernel modules are loaded. See [Multi-node networking](../04-technical-guides/multi-node-networking.md). |
| Timeout too aggressive | Megatron: increase **`distributed_timeout_minutes`**. TorchTitan: increase **`comm.init_timeout_seconds`**. |
| Mismatched world size | Align **`NNODES`**, **`GPUS_PER_NODE`**, and launcher settings across **all** nodes. |

**Debugging**

```bash
export NCCL_DEBUG=INFO
```

**Validation**

```bash
primus-cli direct -- preflight --network
primus-cli direct -- benchmark rccl --op all_reduce
```

Collective behavior and RCCL roles are summarized in [Collective operations](../04-technical-guides/collective-operations.md).

---

## 4. Container issues

**Symptoms:** Permission denied on devices, GPUs not visible inside the container, immediate exit, or RCCL failures only under Docker.

| Symptom | Typical cause | Mitigation |
|---------|-----------------|------------|
| GPU not visible | Devices not passed through | Ensure **`--device /dev/kfd`** and **`--device /dev/dri`** (Primus container mode typically sets these). |
| Permission denied on GPU | Group membership / permissions | Add **`--group-add video`**; ensure the user has **video/render** access on the host. |
| InfiniBand missing in container | Device not mounted | Add **`--device /dev/infiniband`** (and related uverbs devices as required by your site). |
| Debugger/profiler failures | Missing capabilities | **`--cap-add SYS_PTRACE`** and **`--cap-add CAP_SYS_ADMIN`** are required for many ROCm tooling paths. |
| Driver/library mismatch | Image vs. host ROCm | Match **container image ROCm** to **host ROCm driver** version. |
| Data or code not found | Bind mounts | Use **`--volume /host/path:/container/path`** for datasets and workspace. |
| Env vars missing in container | Forwarding | Check **`runner/.primus.yaml`** `container.options.env` for auto-forwarded variables; add extras with **`--env KEY=VALUE`**. |

Installation and container-oriented setup are covered in [Installation](../01-getting-started/installation.md).

---

## 5. Configuration errors

**Symptoms:** YAML parse failures, "unknown key" or type errors, silent wrong behavior after edits.

| Issue | Resolution |
|-------|------------|
| Unset `${VAR}` | `${VAR}` with no default fails if `VAR` is unset. Use **`${VAR:default}`** or **export** the variable before launch. |
| Broken `extends:` chain | Confirm every referenced file exists and paths resolve relative to the expected directory. |
| Wrong parameter name | Cross-check backend docs: [Megatron](../03-configuration-reference/megatron-parameters.md), [TorchTitan](../03-configuration-reference/torchtitan-parameters.md), [MaxText](../03-configuration-reference/maxtext-parameters.md), [Megatron Bridge](../03-configuration-reference/megatron-bridge-parameters.md). |
| Override not applied | Review merge order: **CLI > experiment overrides > model preset > module preset**. See [Configuration system](../02-user-guide/configuration-system.md). |
| Effective config unknown | Use **`--export_config`** and **`--dry-run`** together to inspect merged YAML and planned commands. |

---

## 6. Backend-specific issues

### Megatron

| Issue | Mitigation |
|-------|------------|
| Suspected Primus patch interaction | `export PRIMUS_PATCHES=none` and retry with vanilla Megatron behavior. |
| Custom kernel compile failures | `disable_compile_dependencies: true` skips custom kernel compilation where applicable. |
| Wrong third-party path | Set **`BACKEND_PATH`** to override third-party resolution. |

### TorchTitan

| Issue | Mitigation |
|-------|------------|
| Submodule drift | `git submodule update --recursive` so `third_party/torchtitan` matches the Primus revision you run. |
| `torch.compile` instability | `export TORCH_COMPILE_DISABLE=1` or set **`compile.enable: false`** in config. |

### MaxText (JAX)

| Issue | Mitigation |
|-------|------------|
| JAX / jaxlib vs ROCm | Verify JAX and jaxlib builds match your ROCm stack. |
| XLA memory pressure | Tune **`XLA_PYTHON_CLIENT_MEM_FRACTION`** to cap client-side allocator use. |

---

## 7. Performance issues

**Symptoms:** Low tokens/sec, long iteration time, or poor scaling versus expectations.

**Diagnosis**

| Step | Command / action |
|------|-------------------|
| Compute sanity | `primus-cli direct -- benchmark gemm` |
| Interconnect | `primus-cli direct -- benchmark rccl --op all_reduce` |
| Broader probe | `primus-cli direct -- preflight --perf-test` (see [Preflight](../02-user-guide/preflight.md)) |

**Common fixes**

| Area | Action |
|------|--------|
| GEMM / kernels | Enable **HipBLASLt tuning** (multi-stage workflow—see [Performance tuning](../04-technical-guides/performance-tuning.md)). |
| Primus stack | Enable **`enable_primus_turbo: true`** where supported. |
| Communication overlap | **`overlap_grad_reduce: true`**, **`overlap_param_gather: true`** (when applicable to your backend). |
| MoE / scratch | Confirm **`HSA_NO_SCRATCH_RECLAIM=1`** when recommended for your model class. |
| FP8 | Enable FP8 when hardware and backend support it. |

---

## 8. Data issues

| Symptom | Checks |
|---------|--------|
| Mock data works; real data fails | **`data_path`** format: Megatron typically expects **`.bin` / `.idx`** pairs. See [Data preparation](../04-technical-guides/data-preparation.md). |
| Tokenizer errors | **`tokenizer_type`** and **`tokenizer_model`** must match (e.g., Hugging Face tokenizer ID for `HuggingFaceTokenizer`). |
| Hugging Face download failures | Set **`HF_TOKEN`** for gated models; verify outbound network and cache directories. |

---

## 9. Known limitations

| Area | Note |
|------|------|
| MaxText | Parameter completeness depends on upstream MaxText **`base.yml`**; some keys are inherited from upstream defaults. |
| Megatron Bridge | Recipe parameters may be loaded dynamically; not every key appears in static reference tables. |
| HummingbirdXT | Less mature than other backends; expect sharper edges in configs and tooling. |
| Primus-Turbo | Requires a **separate installation** step; not always present by default. |

For terminology, see the [Glossary](../01-getting-started/glossary.md). For checkpoint-related failures, see [Checkpoint management](../04-technical-guides/checkpoint-management.md).
