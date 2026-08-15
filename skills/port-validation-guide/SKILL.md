---
name: port-validation-guide
description: Generate a validation plan and test scaffolding to check a Primus optimization that has been ported into a user's own training framework, across correctness (numerical accuracy vs a reference), performance (throughput parity), and integration (no regressions). Use after porting a Primus or Primus-Turbo feature into Megatron-LM, TorchTitan, MaxText, or another framework when the user wants a structured way to verify the port. Produces plans and test skeletons only; it does not run tests, require a GPU, or guarantee a successful port.
---

# Port Validation Guide

After a Primus optimization is ported into a user's own framework, this skill produces a **validation plan** and **test scaffolding** across three tiers: correctness (numerical accuracy), performance (throughput parity), and integration (no regressions).

It generates plans and test skeletons only. It does not execute tests, does not need a GPU, and does not guarantee the port is correct - the user decides which tiers to run, sets the thresholds, and chooses whether to run end-to-end training.

The tests must be driven by what the port actually changed, so start from the facts.

**Use the backend-patch-explorer skill first** to identify, for the target feature: the replaced or wrapped upstream symbol, the Primus-side and Turbo-side implementation, the enable flags, the version and dependency requirements, and the constraints (e.g. TP/EP limits, ROCm-only, Transformer Engine version). Those facts decide what each tier checks and which constraints become test skip conditions.

## Inputs to confirm with the user

- Target framework and version, and where the ported feature is wired in (module / call site).
- Whether they can run on GPU. Op-level correctness checks and all benchmarks need a device; without one, emit the scaffold plus run instructions and stop short of executing.
- The acceptance thresholds they care about. Otherwise use the defaults in [references/test-templates.md](references/test-templates.md) as a starting point.

## Workflow

### Step 1 - Gather the port facts

Run backend-patch-explorer for the feature. Capture the replaced symbol, both-side implementations, enable flags, version/deps, and constraints. These map directly into the tests below.

### Step 2 - Choose the reference (oracle)

Correctness needs something to compare against. Pick per feature:

- Same framework, feature OFF (baseline) vs feature ON (ported), identical inputs and seeds - answers "did I change results".
- Primus's own implementation as the golden reference (dump its tensors) when it can be run - answers "does my port match Primus".

Decide comparison points: op-level (module forward / backward output) and/or step-level (loss, grad norm). Op-level is the strongest signal for a kernel or dispatcher swap.

If the port is a full replacement of a compute-plus-communication path rather than a thin drop-in wrapper (e.g. a token dispatcher swap), op-level outputs may legitimately differ and `assert_close` becomes the wrong oracle. For those, prefer a step-level or end-to-end loss-curve comparison with statistical checks (mean / max relative error, loss within a band) instead of exact op-level equality.

### Step 3 - Correctness (numerical accuracy)

Scaffold a test that builds the module both ways, feeds identical fixed-seed inputs, and compares outputs with dtype-appropriate tolerances (see references/test-templates.md). For fp8 / low precision, compare against a higher-precision reference and assert relative plus max-abs error, not exact equality. Cover backward (gradients) when the patch touches it.

Building the module both ways is the hardest part of the scaffold, so pull the exact construction facts from backend-patch-explorer: the required constructor / config fields and the full forward signature. Attention modules in particular need more than `hidden_states` (e.g. `rotary_pos_emb`, `attention_mask`, `packed_seq_params`), and their config needs feature-specific fields (e.g. MLA needs `q_lora_rank` / `kv_lora_rank` and the head dims). Put these in the scaffold TODOs so the user wires real inputs instead of guessing.

### Step 4 - Performance (throughput parity)

Scaffold an A/B harness: same config, measure tokens/s or step time with the feature OFF vs ON, warmup then timed iters, report speedup and peak memory. Parity means the ported path is at least baseline within a user-set margin - a relative check, not an absolute target. GPU-required: without a device, emit the harness and run instructions only.

### Step 5 - Integration (no regressions)

Select the user's existing tests that touch the changed area to run before and after. Add a smoke test: a few training steps run to completion, loss finite (no NaN / inf), checkpoint save/load roundtrip. Turn the constraints from Step 1 into skip conditions (e.g. skip when TP != 1).

### Step 6 - Assemble

Emit two things:

1. A validation plan (markdown) covering the three tiers, the chosen reference, thresholds, constraints, and which tiers are runnable in the user's environment.
2. Test scaffold files from references/test-templates.md, with TODO markers where the user must wire their framework-specific module construction, config, and data.

Be explicit about what only the user can fill in; this skill cannot know their framework internals.

## Output

Deliver the plan inline. Write scaffold files under a path the user gives, else `./port-validation/`. Offer to save the plan as a document. Do not run any test, benchmark, or training. Write the plan and scaffolds in English by default; use another language only if the user explicitly asks.

## Additional resources

- Test skeletons and default tolerance / benchmark settings: [references/test-templates.md](references/test-templates.md)
