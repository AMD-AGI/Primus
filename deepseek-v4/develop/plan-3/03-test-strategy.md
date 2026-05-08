# 03 — Plan-3 Test Strategy

> Plan-3 reuses the test conventions from
> `../plan-2/04-test-strategy.md`. Each gate below maps to one phase
> from `01-roadmap.md` / `02-phase-details.md`.

## Gate matrix

| Gate | Phase | Type | What it checks | Where it lives |
|---|---|---|---|---|
| **G15** | P21 | static (AST) | No `try / except / return nn.Linear` (or `nn.Module`) under `primus/backends/megatron/core/`. Also: no remaining `"fallback to nn.Linear"` / `"submodule init failed"` strings in V4 backend code. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p21_strict_build.py` |
| **G15b** | P21 | runtime (CPU) | TP=2 vs TP=1 forward-equivalence on a 1L V4 toy with `compress_ratio=0`. Output match within 1e-5. Catches sharding mismatches the silent fallback used to mask. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p21_tp_equivalence.py` |
| **G16** | P20 | runtime (CPU) | V4 closed-form FLOPs vs hand-computed reference within 1% on an 8-layer Flash-shape config (`hash_layers=3`, mixed compress-ratios). | `tests/unit_tests/backends/megatron/test_deepseek_v4_flops_patches.py` |
| **G17** | P20 | runtime (CPU) | Non-V4 model types call back to upstream `num_floating_point_operations` byte-for-byte. | same file as G16 |
| **G18** | P22 | runtime (CPU) | Dense V4 attention: forward via `core_attention` vs eager-Python within 1e-3 on a 1L toy. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p22_core_attention.py` |
| **G19** | P22 | runtime (CPU) | HCA + CSA branches still produce identical outputs as before plan-3 (regression guard for the eager-Python paths). | same file as G18 |
| **G20** | P23 | runtime (CPU) | V4 dispatcher selection: turbo on (TP=1) → `PrimusTurboDeepEPTokenDispatcher`; turbo on + TP>1 → `MoEFlexTokenDispatcher` with rank-0 warning; turbo off → unchanged. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p23_dispatcher_pick.py` |

## Banned-warning ratchet

In addition to the existing P19 / P20 gates (`c10d::allreduce_`,
`Routing snapshot diff`), plan-3 adds three banned-string ratchets
that every V4 smoke from P21 onward must keep clean:

1. `"submodule init failed"` — caught by P21 (must raise, not warn).
2. `"fallback to nn.Linear"` — same.
3. `"unsupported dispatcher module"` — caught by P23 (turbo class is now recognised).

Any of these in the smoke log fails the run.

## CPU-toy harness

All CPU-runnable gates (G15 / G15b / G16 / G17 / G18 / G19 / G20) use
the existing `tests/unit_tests/megatron/transformer/deepseek_v4/`
harness (1L / 4L V4 toy with `hidden=128`, `head_dim=32`,
`num_attention_heads=4`, `num_experts=8`, `compress_ratios=[0,4,128,0]`).
No GPU required.

## Reporting hand-off

Plan-3 closes with the P23 entry in `../progress/status.md` plus the
ad-hoc local DeepEP smoke captured under
`../progress/p23/` (gitignored). A `plan-3-summary.md` mirroring
`plan-2-summary.md` can be authored as a follow-up if required.
