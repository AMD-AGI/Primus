# Opt-in Turbo cross entropy for Megatron training

Install a Primus-Turbo version with `primus_turbo.pytorch.ops.cross_entropy`, then
set `PRIMUS_TURBO_CROSS_ENTROPY=1` before launching training. The patch replaces
Megatron's language-model loss for TP=1, BF16/FP32 logits when
`cross_entropy_loss_fusion` is enabled. TP>1, other dtypes, and an unfused loss
configuration use the existing configured implementation.

The patch preserves Core's layout and reduction contract: sequence-first logits,
batch-first labels and output per-token loss. Training still applies its mask,
normalization and gradient scaling. Do not set `cross_entropy_fusion_impl=turbo`;
the opt-in patch provides the integration while retaining the original Core
configuration for fallback.

`PRIMUS_TURBO_CE_OVERWRITE_INPUT=1` additionally permits backward to reuse
contiguous logits storage. Enable it only when the language-model logits have
no other consumer that needs their original values. Noncontiguous logits use
the preserving path. Both flags default to zero. Remove the CE flag to restore
the configured loss path without changing packages.

This integration changes no installed Transformer Engine files. It requires no
TE native rebuild; Triton JIT compilation occurs on first use. It neither fuses
the LM-head GEMM nor enables full-model graph capture. The operator supports one
backward per forward and no higher-order gradients. Language-model recipes need
numerical and convergence validation because its FP32 gradient arithmetic
differs from older TE's intermediate BF16 rounding.

## GPT-OSS validation on MI355X

The 8-GPU GPT-OSS MLPerf recipe was tested with TP/PP/EP/CP=1, sequence 8192,
GBS 32, MBS 4, MXFP4 linears and BF16 logits (vocabulary 128256).

Eight-rank profiling measured mean accumulated CE kernel time of 9.72 to
5.10 ms per training step, with three main CE launches reduced to two.
These measurements cover CE kernels only. The profile used the existing
`PRIMUS_TURBO_ATTN_SINGLE_STREAM=1` setting in both baseline and candidate;
the CE patch does not set this attention option.

Both full convergence runs passed the validation-loss target:
[baseline](https://github.com/AMD-MLPerf/mlperf-training/actions/runs/38024406682)
and [Turbo](https://github.com/AMD-MLPerf/mlperf-training/actions/runs/38024419787).
This single-seed result establishes an observed convergence pass, not numerical
equivalence. Keep the integration opt-in and validate the intended recipe.
The runs used Primus `7c835227` and Turbo `1c639e25`; subsequent changes to
this document do not change their runtime code.
