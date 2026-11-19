# Description

`DeepEPTokenDispatcher` now runs `moe_permute` / `moe_unpermute` on the HIP (TURBO) backend by default, through a new `permute_backend` argument (TRITON is still selectable).

**Motivation.** #455 made TRITON the generic `moe_permute` default, and the dispatcher inherited it because it never passed a backend. On MI355X the TURBO scatter / gather is faster at every shape measured. The gain is largest in the forward permute and in unpermute's backward, which are the two biggest permute costs in an MoE step.

**Why only the dispatcher.** The generic `moe_permute` default stays TRITON. The TURBO kernel bounds its rows by the routed-token count, so it requires routed tokens to come first. DeepEP's receive buffer always satisfies this: every received token has a local expert, and worst-case buffer rows trail. Arbitrary callers may not satisfy it. The worst-case buffer + `pad_multiple > 0` padding-row bug in the TURBO permute was fixed in #553, so all dispatcher modes are covered.

**Microbenchmark.** Qwen3-30B-A3B EP8 dispatcher call (hidden 2048, 16 local experts, top-8, topk probs, bf16), median µs, TURBO vs TRITON:

| shape | op | TURBO | TRITON |
|---|---|---|---|
| even routing, 32768 tokens → 262144 rows | permute fwd | 255 | 390 |
| | permute bwd | 345 | 442 |
| | unpermute fwd | 253 | 257 |
| | unpermute bwd | 815 | 932 |
| random routing | permute fwd | 452 | 792 |
| hidden 7168, 32 local experts, even | permute fwd | 955 | 1011 |
| hidden 4096, 16 local experts, even | permute fwd | 465 | 579 |

That is about 0.35 ms per MoE layer (fwd + bwd) at the Qwen3 shape.

**End-to-end.** Qwen3-30B-A3B FP8 tensorwise pretrain on 8× MI355X (EP8, MBS 8, GBS 512, seq 4096, even routing), mean of iterations 11–20 of 20-iteration runs:

| run | ms/iter | TFLOP/s/GPU | tokens/s/GPU | loss@20 |
|---|---|---|---|---|
| before (TRITON permute in the dispatcher) | 7774.2 | 767.2 | 33,720 | 11.33431 |
| this change (TURBO permute) | 7643.5 | 780.3 | 34,296 | 11.33423 |

That is −1.68% ms/iter and +1.71% tokens/s/GPU. The loss difference is within run-to-run noise (about 4e-4 at iteration 20). The base of both runs is Turbo 1103b2df with the flat tensorwise FP8 quant kernel and the per-token `get_dispatch_layout` change applied, plus Primus `use_turbo_fused_act_with_probs`.

## Type of change

- [ ] Documentation change (change only to the documentation, either a fix or a new content)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Infra/Build change
- [ ] Code refactoring

## Changes

- `primus_turbo/pytorch/modules/moe/token_dispatcher.py`: `DeepEPTokenDispatcher(..., permute_backend=BackendType.TURBO)`. The backend is passed to both `moe_permute` (`_post_dispatch`) and `moe_unpermute` (`_pre_combine`). The docstring states the routed-prefix requirement.
- `tests/pytorch/modules/test_token_dispatcher.py`: `test_basic` and `test_worst_tokens` run on both permute backends (TURBO and TRITON).

Behavior change: dispatcher users who relied on the implicit TRITON default now get TURBO. Pass `permute_backend=BackendType.TRITON` to keep the old path. With `pad_multiple > 0` the dispatcher already resolved to TURBO, because TRITON does not support padding.

## Tests

- `tests/pytorch/modules/test_token_dispatcher.py` (2 GPUs, MI355X / gfx950): `test_basic` and `test_worst_tokens` on both permute backends, 21 passed.
- Re-run today on a build that also contains #553 and the follow-up FP8 permute branch: 25 passed.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
