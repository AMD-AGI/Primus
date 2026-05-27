# Plan-8 P49 — Tilelang infra + dispatcher

> Phase summary written 2026-05-15 at P49 close-out.

**Status: shipped, default OFF.  Infra-only — no kernel landed
yet.  Plan-4..plan-8 ratchet stays green at default-off
(451 passed, 1 pre-existing unrelated failure, 101 skipped in
39.32 s); the dispatcher emits zero behaviour change vs the
plan-7 P48 anchor when `PRIMUS_V4_TILELANG_ATTN` is unset.**

---

## 1. Objective

Land the plan-8 tilelang dispatcher + env knob + cache layout +
build script + lazy-import stubs so plan-8 P50..P55 phases can
register their kernels as they land without further wiring work.

## 2. Design

### 2.1 Dispatcher module

New `primus/backends/megatron/core/transformer/v4_attention_kernels/_tilelang/__init__.py`
exposes:

* `is_tilelang_path_enabled() -> bool` — reads
  `PRIMUS_V4_TILELANG_ATTN == "1"` (default `"0"`).
* `is_tilelang_kernel_available(name: str) -> bool` — checks the
  per-process `_AVAILABLE_KERNELS` registry; empty at P49.
* `register_available_kernel(name: str)` — used by P50..P55
  modules at their import-time to register their kernel name.
* `should_dispatch(name: str) -> bool` — the single predicate the
  `v4_attention` / `v4_csa_attention` wrappers call.  Returns True
  only when env knob is `"1"` AND tilelang importable at the
  pinned version AND the named kernel registered.  Emits a
  one-time rank-0 warning on miss.
* Four stub entry points
  (`v4_attention_fwd_tilelang` / `_bwd_tilelang` /
  `v4_csa_attention_fwd_tilelang` / `_bwd_tilelang`) that raise
  `NotImplementedError` with the phase number (P50/P51/P54/P55) in
  the message.
* `cache_dir() -> str` — returns
  `output/.tilelang_cache/v4` by default; override via
  `PRIMUS_V4_TILELANG_CACHE_DIR`.
* `TILELANG_VERSION_PIN = "0.1.9+cuda.gitbcb2da33"` — matches the
  installed `tilelang/VERSION` at P49 land time.

### 2.2 Dispatch precedence wiring

`v4_attention.py::v4_attention(...)` adds a one-line dispatcher
hook before `V4AttentionFn.apply(...)`:

```python
if _tilelang.should_dispatch("v4_attention_fwd"):
    return _tilelang.v4_attention_fwd_tilelang(...)
return V4AttentionFn.apply(...)
```

`v4_csa_attention.py::v4_csa_attention(...)` adds the same hook
after the `K_topk == 0` short-circuit, before the eager
`V4CSAAttentionFn.apply(...)`.  Effective precedence becomes:

```
cr ∈ {0, 128}:
  use_turbo_attention > PRIMUS_V4_TILELANG_ATTN > use_v4_triton_attention > eager
cr == 4:
  PRIMUS_V4_TILELANG_ATTN > use_v4_triton_csa_attention > eager
```

The dense + HCA precedence already enforced
`use_turbo_attention` higher up in `DeepseekV4Attention.forward`,
so the tilelang hook lives entirely below it.

### 2.3 Cache directory layout

Cache lives at `output/.tilelang_cache/v4/` under the existing
gitignored `output/` directory.  No `.gitignore` change needed
(R6.1 already covers `output/`).  Override via
`PRIMUS_V4_TILELANG_CACHE_DIR=/path/to/dir`.

### 2.4 Build script

`progress/p49/build_tilelang_kernels.sh` is a no-op at P49 (no
kernels to AOT-compile yet); it prints the dispatcher state to
prove the wiring works.  P50..P55 will extend the script with
per-phase AOT compile loops as each kernel ships.

## 3. Code surface

```
primus/backends/megatron/core/transformer/v4_attention_kernels/_tilelang/__init__.py (new)
  + is_tilelang_path_enabled
  + is_tilelang_kernel_available / register_available_kernel
  + should_dispatch (the one-line dispatcher predicate)
  + 4 stub entry points (P50/P51/P54/P55)
  + cache_dir / TILELANG_VERSION_PIN
  + _maybe_warn_fallback (one-time rank-0 warning)

primus/backends/megatron/core/transformer/v4_attention_kernels/v4_attention.py
  M  +1 import line (`from ... import _tilelang`)
  M  +13 lines in `v4_attention(...)` (tilelang dispatcher hook)

primus/backends/megatron/core/transformer/v4_attention_kernels/v4_csa_attention.py
  M  +1 import line
  M  +14 lines in `v4_csa_attention(...)` (tilelang dispatcher hook)

tests/unit_tests/megatron/transformer/deepseek_v4/test_p49_tilelang_dispatch.py (new)
  + 19 G49 tests covering: env knob predicate, kernel availability
    registry, dispatcher fallthrough + one-time warning, stubs
    raise NotImplementedError, version pin / cache dir, wrapper
    source audit.

deepseek-v4/develop/progress/p49/
  + build_tilelang_kernels.sh   (no-op smoke probe at P49)
  + p49-summary.md              (this file)
```

## 4. Performance

No runtime change at default-off (`PRIMUS_V4_TILELANG_ATTN` unset
or `"0"`).  The dispatcher adds two function calls per
`v4_attention` / `v4_csa_attention` invocation:

* `_tilelang.should_dispatch(name)` — checks env (one
  `os.environ.get`), early-exits at default.

The overhead is sub-microsecond — well below any noise floor.

EP=8 proxy A/B at default-off (P49 vs P48 anchor) is bit-equal
by construction; no proxy A/B run needed.  R2.6 (trace + tgz)
skipped at P49 per the rule: it explicitly skips for
documentation-only / infra phases that don't change runtime
behaviour.

## 5. Tests

**G49 — 19 tests, 5 sub-classes, all green:**

* `TestG49EnvKnob` (4 tests): default OFF, `"0"` OFF, `"1"` ON,
  unrelated truthy values (e.g. `"true"`) do NOT enable.
* `TestG49KernelAvailability` (3 tests): P49 empty by default,
  `register_available_kernel(...)` flips to True, unknown kernel
  name raises `ValueError`.
* `TestG49DispatcherFallthrough` (3 tests): default False,
  env=1 + no kernel → False with one-time warning,
  env=1 + registered → True (tilelang-importable).
* `TestG49StubsRaise` (4 tests): each stub raises with the right
  phase number in the message.
* `TestG49Misc` (3 tests): version pin format, default cache
  dir, override cache dir.
* `TestG49WrappersHaveDispatch` (2 tests): source-level audit
  that the dispatcher hook is wired into both wrappers.

**Plan-4..plan-8 ratchet check** (default-off):

```
451 passed, 1 failed (pre-existing unrelated:
  test_v4_mtp.py::test_helper_pulls_norm_and_linear_from_v4_provider),
  101 skipped, 92 warnings in 39.32s
```

Bit-for-bit unchanged from the plan-7 P48 anchor.

## 6. Gating

* `PRIMUS_V4_TILELANG_ATTN` default **`"0"`** (set in
  `is_tilelang_path_enabled()` via
  `os.environ.get("PRIMUS_V4_TILELANG_ATTN", "0")`).
* Per-kernel availability defaults False; P50..P55 flip them
  individually by calling `register_available_kernel(...)` at
  their module-import time.
* `PRIMUS_V4_TILELANG_CACHE_DIR` defaults to
  `output/.tilelang_cache/v4`.

## 7. Failed / negative probes

None — P49 is infra-only.  Two minor wiring fixes during
implementation:

* Initial G49.6 source-audit test imported `v4_attention` as
  `mod` and tried `mod.v4_attention` — Python loaded the
  *function*, not the *module*.  Fixed by using
  `importlib.import_module(...)` to disambiguate.
* Initial build script path-relative `cd` went one level too
  high (`../../../../..` instead of `../../../..` — five hops
  from `progress/p49/` instead of four to reach the repo root).
  Fixed in the same patch.

## 8. Follow-ups + commit pin

* P50 will register `"v4_attention_fwd"` and land the dense FWD
  tilelang kernel.
* P51 will register `"v4_attention_bwd"`.
* P54 will register `"v4_csa_attention_fwd"`.
* P55 will register `"v4_csa_attention_bwd"`.
* Feature commit SHA: 73f763eb.
