# 03 — Phase 9 Provider A/B Validation Report

Date: 2026-04-30

## Scope

- Validate the new provider-driven DeepSeek-V4 integration after:
  - `DeepSeekV4SpecProvider(PrimusTurboSpecProvider)` wiring,
  - attention projection refactor to `submodules + build_module`,
  - providerized norm/projection and MoE grouped path plumbing.

## Environment

- Host: `uswslocpm2m-106-2371`
- Container: `dev_primus_wenx_691`
- Workspace: `/shared_aig/wenx/workspace/Primus`
- Runtime command style: `ssh` + `docker exec` with repository `PYTHONPATH`

## Test Cases

### Case A — Local provider mode (forward)

- Config mode: `transformer_impl=local`
- Model slice: DeepSeek-V4 runtime decoder spec (2 layers, compress ratios `[0, 4]`)
- Input shape: `hidden_states=(8, 1, 128)`

Result:

- Pass
- Runtime output:
  - `mode=local output_shape=(8, 1, 128) q_b=Linear k_proj=Linear o_proj=Linear`

Interpretation:

- Local baseline path is runnable end-to-end.
- Attention projections are instantiated through submodule specs and resolve to local linears in local mode.

### Case B — TE provider mode (build-only module map)

- Config mode: `transformer_impl=transformer_engine`
- Same decoder slice as Case A.

Result:

- Pass (build-only)
- Runtime output:
  - `mode=transformer_engine build_only q_b=TELinear k_proj=TELinear o_proj=TELinear`

Interpretation:

- Attention projection submodules are correctly provider-resolved to TE modules.
- The `submodules + build_module` path is effective in TE mode.

### Case C — TE provider mode (forward, CUDA path)

- Config mode: `transformer_impl=transformer_engine`
- Same decoder slice with tiny forward input.
- Runtime path: `decoder.cuda()` + `hidden_states` on CUDA.

Result:

- Pass
- Runtime output:
  - `mode=transformer_engine output_shape=(8, 1, 128) q_b=TELinear q_b_weight_device=cuda:0`

Interpretation:

- TE provider path is runnable end-to-end in expected CUDA execution mode.
- Attention projection submodules are both provider-resolved and runtime-functional.

### Case D — TE provider mode (invalid host-input path)

- Config mode: `transformer_impl=transformer_engine`
- Runtime path: decoder/input tensors kept on host side (non-CUDA).

Result:

- Fail (expected invalid path)
- Observed failure signature in early validation:
  - `Memory access fault by GPU ... Reason: Unknown` (exit `134`)

Interpretation:

- TE-backed path must run with CUDA tensors.
- Added runtime guard in decoder forward to fail fast with a clear error when TE/Turbo mode receives non-CUDA hidden states.

## A/B Summary

| Mode | Spec / Module Map | Forward | Verdict |
|---|---|---|---|
| Local | Pass (`Linear`) | Pass | usable baseline |
| TE | Pass (`TELinear`) | Pass (CUDA path) | usable with CUDA-only runtime contract |

## Recommendation

- Keep local fallback available for non-TE environments.
- Use TE path under explicit CUDA runtime contract.
- Retain clear runtime guard/error for invalid host-input invocations.

## Next Debug Steps

1. Extend validation to full trainer path (`DeepseekV4Model`) under TE mode with CUDA inputs.
2. Add a tiny regression check asserting TE mode fails fast on host-input path with the expected guard message.
3. Validate Turbo-mode projection map (`PrimusTurboLinear`) on the same harness when turbo flags are enabled.
