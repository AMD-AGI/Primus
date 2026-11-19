# Description

`get_dispatch_layout` (intranode) is about 7.5× faster with a one-thread-per-token kernel.

**Why it was slow.** The existing layout kernel gives each block 4 experts (or 8 ranks) and has it scan every token. So only about num_experts / 4 + 1 blocks run, and the rank block walks all tokens alone.

**New kernel.** It applies to intranode buffers (no RDMA counters) with top-k ≤ 16. It:
- reads each `topk_idx` row once, with one thread per token;
- gathers expert / rank counts with shared-memory integer atomics, which are order-independent, so results stay deterministic;
- adds the counts once per block to zeroed global counters;
- writes `is_token_in_rank` with 64-bit stores when `num_ranks % 8 == 0`.

Internode buffers, top-k > 16 and very large expert counts keep the previous kernel.

**Microbenchmark.** Qwen3-30B-A3B EP8 on 8× MI355X (32768 tokens, top-8, 128 experts), rank-max median:

| routing | before | after |
|---|---|---|
| random | 340 µs | 44 µs |
| even | 340 µs | 45 µs |

The training step calls it 384 times (48 layers × 8 micro-batches), about −114 ms per iteration. Dispatch / combine times are unchanged.

**End-to-end.** Qwen3-30B-A3B FP8 tensorwise pretrain on 8× MI355X (EP8, MBS 8, GBS 512, seq 4096, even routing), mean of iterations 11–20 of 20-iteration runs:

| run | ms/iter | TFLOP/s/GPU | tokens/s/GPU | loss@20 |
|---|---|---|---|---|
| before | 7870.9 | 757.7 | 33,305 | 11.33415 |
| per-token `get_dispatch_layout` | 7774.2 | 767.2 | 33,720 | 11.33431 |

That is −1.23% ms/iter and +1.25% tokens/s/GPU. The loss difference is within run-to-run noise (about 4e-4 at iteration 20). Both runs use Turbo 1103b2df plus the flat tensorwise quant kernel, and Primus `use_turbo_fused_act_with_probs`.

## Type of change

- [ ] Documentation change (change only to the documentation, either a fix or a new content)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Infra/Build change
- [ ] Code refactoring

## Changes

- `csrc/kernels/deep_ep/layout.cu`: new per-token intranode layout kernel, and dispatch between it and the previous kernel.
- `tests/pytorch/deep_ep/test_intranode.py`: new `test_dispatch_layout`.

## Tests

- New `test_dispatch_layout` compares rank / expert counts and `is_token_in_rank` with `get_dispatch_layout_ref` (`torch.equal`):
  - routing: random, masked (-1) and even;
  - 0 / 1 / 777 / 4099 / 32768 tokens;
  - top-k 1 / 8 / 9, and top-k 20 for the fallback path.
- `tests/pytorch/deep_ep/test_intranode.py` on 8× MI355X: 2 passed (`test_dispatch_layout` and the existing intranode dispatch / combine test).

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
