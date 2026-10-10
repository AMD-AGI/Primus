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
TE native rebuild; Triton JIT compilation occurs on first use. The operator
supports one backward per forward and no higher-order gradients. Language-model recipes need
numerical and convergence validation because its FP32 gradient arithmetic
differs from older TE's intermediate BF16 rounding.
