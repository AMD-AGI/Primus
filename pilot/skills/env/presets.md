# Per-Cluster-Class Presets

**Status**: Stub

Validated env baseline combinations per cluster class. Used as:
1. Initial `ClusterProfile.env_baseline` candidate.
2. Re-Plan preference: `prefer_known_env_presets: true` (TargetVector §8.6) biases toward presets here.

## TODO

- [ ] mi300x_8gpu (16-node IB) baseline
- [ ] h100_8gpu (4-node IB) baseline
- [ ] mi325x_8gpu baseline
- [ ] Schema for preset entries (binding + flags + last_validated_at)
