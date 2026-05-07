# Correctness (Numerical Gate)

**Status**: Stub

Loss curve / grad norm alignment with reference. BASELINE-time hard gate; OPTIMIZE_LOOP-time periodic LITE gate. Failure → ABORT + escalate.

> Detailed reference-tier design: see `README.cn.supplements.md` §S2.

## TODO

- [ ] T0 / T1 / T2 reference tiers (see §S2)
- [ ] Scale-aware tolerance formula (k×σ + ε_systematic)
- [ ] Tokens-aligned comparison (not step-aligned)
- [ ] Gate hierarchy (smoke / baseline / lite / regression_signal)
