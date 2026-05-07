# Execution Strategy Selection

**Status**: Stub

Selection logic among **Champion-Challenger / Per-Plan + Pruning / Successive Halving** based on axis taxonomy + model confidence + budget.

## TODO

- [ ] Decision tree (axis_taxonomy x confidence x budget → strategy)
- [ ] Strategy parameter defaults (top_k, halving ratio, ...)
- [ ] Fallback when calibration drift_alarm fires (force Champion-Challenger; see §S1)
