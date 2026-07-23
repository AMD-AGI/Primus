# Port Validation - Test Templates and Defaults

Framework-agnostic skeletons for the three validation tiers. They are starting
points: every `TODO` must be wired to the user's own framework (module
construction, config, data). Tolerances and iteration counts are defaults the
user should tune for their hardware and precision.

## Default tolerances (correctness)

Compare the ported output against the reference with tolerances that match the
compute dtype. These are starting values, not guarantees.

| Compute dtype | rtol | atol | Notes |
|---|---|---|---|
| fp32 | 1e-5 | 1e-6 | near-exact |
| tf32 | 1e-3 | 1e-4 | matmul in tf32 |
| bf16 | 2e-2 | 1e-3 | wide mantissa error |
| fp16 | 1e-2 | 1e-3 | |
| fp8 | compare vs a bf16/fp32 reference | - | assert relative + max-abs error, never exact equality |

For a kernel/dispatcher swap, also check backward: compare input grads (and
weight grads) with the same tolerances.

## Tier 1 - Correctness (op-level, PyTorch)

```python
# port-validation/test_correctness.py
import torch

SEED = 1234

def _fixed_input():
    torch.manual_seed(SEED)
    # TODO: build the exact input tensors the module expects (shapes, dtype, device)
    raise NotImplementedError

def _build_reference():
    # TODO: construct the module WITHOUT the ported feature (baseline),
    #       or load Primus golden tensors, in eval mode with fixed seed.
    raise NotImplementedError

def _build_ported():
    # TODO: construct the module WITH the ported feature enabled.
    raise NotImplementedError

def test_forward_matches_reference():
    x = _fixed_input()
    ref = _build_reference()
    got = _build_ported()
    with torch.no_grad():
        out_ref = ref(x)
        out_got = got(x)
    # TODO: set rtol/atol from the dtype table above
    torch.testing.assert_close(out_got, out_ref, rtol=2e-2, atol=1e-3)

def test_backward_matches_reference():
    x = _fixed_input().requires_grad_(True)
    # TODO: run both paths, backward a scalar (e.g. out.sum()),
    #       compare x.grad (and named weight grads) within tolerance.
    raise NotImplementedError
```

For fp8 or other low precision, replace `assert_close` with an explicit
relative-error check against a higher-precision reference:

```python
err = (out_got.float() - out_ref.float()).abs()
rel = err / (out_ref.float().abs() + 1e-8)
assert rel.mean().item() < 0.05        # TODO: user-set threshold
assert err.max().item() < 1.0          # TODO: user-set threshold
```

## Tier 2 - Performance (throughput parity, PyTorch)

```python
# port-validation/bench_parity.py
import time, torch

WARMUP, ITERS = 10, 50   # TODO: tune

def _run_once(build_fn):
    model = build_fn()                 # TODO: build model/step closure
    step = _make_step(model)           # TODO: one train/forward step closure
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    peak = torch.cuda.max_memory_allocated() / 1e9
    return dt, peak

def main():
    base_dt, base_mem = _run_once(_build_baseline)   # TODO
    port_dt, port_mem = _run_once(_build_ported)     # TODO
    speedup = base_dt / port_dt
    print(f"baseline: {base_dt*1e3:.2f} ms/it, {base_mem:.2f} GB")
    print(f"ported:   {port_dt*1e3:.2f} ms/it, {port_mem:.2f} GB")
    print(f"speedup:  {speedup:.3f}x")
    # Parity check is relative and user-defined:
    MARGIN = 0.95   # TODO: ported must be >= baseline * MARGIN
    assert speedup >= MARGIN, "ported path slower than baseline beyond margin"
```

Requires a GPU. If none is available, emit this harness and the run command
(`python port-validation/bench_parity.py`) without executing it.

## Tier 3 - Integration (no regressions)

```python
# port-validation/test_integration.py
import math, torch, pytest

# Skip conditions come from backend-patch-explorer constraints for the feature.
# TODO: replace with the real constraint checks (parallelism, versions, ROCm).
def _feature_supported():
    return True

pytestmark = pytest.mark.skipif(not _feature_supported(),
                                reason="feature constraints not met")

def test_smoke_train_a_few_steps():
    # TODO: build a tiny model+optimizer+data with the feature ON.
    losses = []
    for _ in range(5):                 # TODO: a few steps
        loss = _train_step()           # TODO
        assert math.isfinite(loss), "loss is NaN/inf"
        losses.append(loss)
    assert losses[-1] <= losses[0] * 2.0   # loose sanity, not convergence

def test_checkpoint_roundtrip(tmp_path):
    # TODO: save then load; assert state_dict keys/shapes match and
    #       a forward pass after load reproduces pre-save output.
    raise NotImplementedError
```

Also list the user's existing tests that touch the changed module/path and run
them before and after the port to confirm no regression. This skill only names
which tests to run; the user runs them.

## MaxText / JAX note

For a JAX (MaxText) port, keep the same three tiers but swap primitives:
`jnp`/`jax.numpy` with `jnp.allclose` (or `chex.assert_trees_all_close`) for
correctness, `block_until_ready()` + `time.perf_counter()` for timing, and the
project's `pytest` suite for integration. Sharding-annotation ports should also
assert the expected sharding on outputs where applicable.
