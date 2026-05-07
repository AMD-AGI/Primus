# Axis Taxonomy

**Status**: Stub

Classify tunable axes into `cluster_shared` / `weakly_local` / `strongly_local`. Drives:
- Which axes belong in EnvBaseline vs Plan.env.diff.
- Which Strategy (Champion-Challenger / Per-Plan / Successive Halving) is appropriate.
- Exhausted-neighborhood radius computation per axis.

## TODO

- [ ] Full axis catalog with classification (parallel / runtime / comm / env)
- [ ] Radius rules per type
- [ ] Cross-reference to `execution_strategy.md`
