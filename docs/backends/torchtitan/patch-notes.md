## TorchTitan Patch Notes & Extended Arguments

TorchTitan integration uses the same Primus configuration surface (CLI flags + YAML) but exposes a few extra knobs via patches.
Pair this with [`docs/backends/overview.md`](../overview.md) for shared module parameters.

| New Argument | Default Value | Version | Description | Patched Files | Notes |
| ------------ | ------------- | ------- | ----------- | ------------- | ----- |
| `primus_turbo.enable_embedding_autocast` | `true` | v0.4.0 | Automatically casts `nn.Embedding` outputs to the AMP dtype (e.g., bf16) during training so downstream layers stay in sync. | (Primus TorchTitan patch set) | Disable only if you manage casting manually. |

> ℹ️ Future TorchTitan-specific switches will be appended here so the CLI + docs stay aligned.
