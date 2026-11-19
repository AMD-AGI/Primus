---
name: projection-op-bringup
description: Add an analytical performance model for a new op (a new attention variant, linear-attention / state-space layer, gate, convolution, MTP head, ...) to the Primus projection when it reports a PRIMUS_OP_GAP or `[Primus:OpCoverage] GAP` line, or when the projection agent leaves a context pack (op_bringup/<op_id>/context.md). Use when a model's op is unmodeled, priced as a built-in stand-in, or when asked to bring up / model / validate a new op for projection or the tuning agent.
---

# Projection op bring-up

New models ship ops the projection's built-in profilers do not price. The
projection detects them and prints a gap; this skill closes the gap with an
**op model**, validated before it is registered. The flow mirrors the TraceLens
model bring-up perf-model stage: Context, Create, Validate, Register.

## Where things live

- Op-model API: `primus/core/projection/op_models/base.py` (`OpModel`, `OpContext`, `OpCost`)
- Gap detection: `primus/core/projection/op_models/coverage.py`
- Validator: `primus/core/projection/op_models/validate.py`
- Registered ops: `primus/core/projection/op_models/generated/*.py` (auto-discovered)
- Negligible ops: `primus/core/projection/op_models/generated/acknowledged.yaml`
- Driver: the projection agent, `primus/agents/projection_agent/` (`agent.py` per workload, `bringup.py` per op)

Worked examples: `generated/gated_delta_net.py` (fused chunked recurrence on
the 1-CU tile simulator), `generated/attention_output_gate.py` (additive GEMM +
elementwise), `generated/lfm2_short_conv.py` (replacing block, GEMM + streams).

## 1. Context

```bash
python -m primus.agents.projection_agent --workload <pretrain.yaml> --op-out-dir runs/ \
    --op-coder none [--op-paper paper.txt] [--hf-config <repo id | config.json>]
```

This lists the gaps and writes `runs/op_bringup/<op_id>/context.md`. That file
holds the gap keys, the model YAML chain, the HF config, any papers, the API,
and the closest existing implementations. Read it first. Then pull the op's
equations from the paper and its shapes from the HF modeling code.

## 2. Create

Write one module defining one `@register_op_model` class. Pick the slot:

| slot | effect on the layer |
|---|---|
| `attention` | replaces the attention block on `layer_fraction` of layers |
| `attention_extra` | adds to the attention block |
| `mlp` | replaces the MLP block |
| `layer_extra` | adds a term outside attention and MLP |

Price each kernel the way the built-in profilers price theirs:

- **Matmuls and projections:** `ctx.linear(ctx.tokens, n_full, k)` for fwd and bwd. TP and CP are already folded into `ctx.tokens`, so pass the full width.
- **Softmax attention of any shape:** `ctx.sdpa(...)`, the FAv3 tile-level simulator.
- **Other fused kernels (chunked recurrences, scans):** `ctx.fused_tiles(per_workgroup_tiles, n_workgroups)`, which runs a tile program on one CU across waves.
  - Parallelise over the dimensions the real kernel does.
  - Fold sequential loops into a tile dimension, e.g. `(n_chunks * C, n, k)`.
- **Elementwise ops, norms, small convs, state traffic:** `ctx.stream_ms(hbm_bytes)`.
- **Memory-bound fused kernels:** `max(compute_ms, ctx.stream_ms(bytes))`.

Then:

- Every gap key must be in `claims`, `consumes`, or `ignores` (with a reason).
- Write `reference_flops()` from the paper's equations, independently of `cost()`.
- Set `num_params` (full, per layer) and `activation_bytes` (per rank) when they differ from the block the op replaces.

If the op is under about 1% of its layer, acknowledge it instead of modeling
it. Add each key to `acknowledged.yaml` along with the estimate.

## 3. Validate

```bash
python -m primus.core.projection.op_models.validate --config <pretrain.yaml> \
    --op-file my_op.py --gap-keys k1,k2 [--json]
```

The validator checks the workload shape, 2× batch, and ½ sequence. Each must:
- be finite and positive;
- stay at or above the roofline and within 50× of it;
- have a bwd/fwd ratio between 0.8 and 4;
- have a breakdown that sums to the totals;
- agree with the closed-form FLOPs within 25%;
- grow with tokens;
- cover every gap key.

Fix the model, not the check.

## 4. Register and check

- Copy the module to `generated/<name>.py`. The driver does this, and writes `tests/unit_tests/core/projection/op_models/test_generated_<name>.py`.
- Re-run the projection. It prints `[Primus:OpModel] <name> ... on N% of layers`, and the gap is gone.
- A/B against the stand-in with `PRIMUS_OP_MODELS=0`.
- Check the size of the change in Tokens/s/GPU against what the op's arithmetic implies.

## Automated (projection agent)

The projection agent runs this flow for every gap of a workload. It is invoked by:

- `primus projection <suite> --config <yaml> --op-bringup auto`, which then projects with the new models;
- `python -m primus.agents.projection_agent --workload <yaml>`, which only brings the ops up;
- the tuning agent (`--op-bringup auto` by default), before seeding and whenever a trial prints a new `PRIMUS_OP_GAP`.

Who writes the model:

- `--op-coder-cmd '<coding agent> ... {prompt_file} ... {target_file}'` uses an external coding agent.
- Otherwise an LLM: `--op-llm-model` / `$LLM_MODEL`, or the tuning agent's own.
- With no LLM, it falls back to `report` (context packs only), and a human or IDE agent finishes the work with this skill.
