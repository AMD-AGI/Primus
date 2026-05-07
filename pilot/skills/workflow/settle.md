# Settle (Convergence)

**Status**: Stub

Greedy + soft rollback on PlanGraph: champion promotion, shelved bookkeeping, stagnation detection, backtrack triggers. Stop conditions: TargetVector met / gain < ε_stop ×2 / max_rounds / budget reached.

## TODO

- [ ] ε_promote / ε_stop default values + budget-aware adjustment
- [ ] Backtrack rules (subtree dead-rate > 50% × 2 rounds)
- [ ] Periodic explore-round (every K=3 rounds)
- [ ] rounds_since_promotion / rounds_since_explore counters
