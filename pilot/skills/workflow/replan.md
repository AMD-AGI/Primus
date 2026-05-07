# Re-Plan (Candidate Generation)

**Status**: Stub

Re-Plan emits a CandidatePool (§8.10) with priority = `expected_gain × confidence / cost × novelty_bonus × stability_bonus`. Mixes exploit (from champion) and explore (from shelved) candidates.

## TODO

- [ ] 7-step generation flow (derive → skill-map → score → constrain → dedup → strategy → top-K)
- [ ] Derivation source selection (default champion; on stagnation use shelved)
- [ ] Skill-mapping table per bottleneck (COMM/PIPELINE/MEMORY/COMPUTE → optimization/*)
- [ ] novelty_bonus / stability_bonus rules
