# Muse capture API backport

`sglang-muse-1027-capture-api.patch` targets the digest pinned in
`Dockerfile.muse-spark-eagle4`. It preserves the image's Muse ROCm model,
AITER patches, and optimized kernels while adding generic SGLang capture APIs.
The Docker build checks that the patch applies before changing any files.

The patch derives from Apache-2.0-licensed SGLang changes:

- [Auxiliary target capture, PR #43449](https://github.com/sgl-project/sglang/pull/43449),
  merge `e732f5d41e9d279c1be7e5818ce9fd8bd425c0b2`.
- [Forward observers and deferred responses, PR #43450](https://github.com/sgl-project/sglang/pull/43450),
  merge `4a58dc983825e1b150d2f375e6885a9f2c7f52ec`.

Compatibility adjustments preserve the older image's field ordering, imports,
abort-response routing, and encoder replay callers. The Muse EAGLE3 hook reuses
its native post-layer residual capture and final normalization; explicit layer
IDs remain zero-based decoder output indices. EAGLE4 uses the EAGLE3 capture
convention without loading an inference-time draft model.

SpecForge's installed plugin continues to own request parsing, feature
publication to Mooncake, deferred completion, and capture metadata. The patch
contains no replacement SpecForge capture backend or legacy capture flags.

Remove this backport when an upstream Muse-capable image provides these APIs
and its model capture hook. Changing `BASE_IMAGE` requires rechecking both
the source patch and captured feature semantics.
