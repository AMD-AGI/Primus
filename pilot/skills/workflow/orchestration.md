# Orchestration Protocol

**Status**: Stub

Orchestrator ↔ Stage Worker contract: spawn rules, SubagentResult validation (§8.11), context hygiene (`state.trim()`), handoff trigger (§13.2 strategy C). This Skill is the contract Orchestrator MUST follow.

## TODO

- [ ] SKILL_SCOPES table (which Skill subtree each Worker may read)
- [ ] SubagentResult schema validation (`summary` ≤ 200 tokens)
- [ ] Context budget rules (§13.4 ledger)
- [ ] Handoff trigger thresholds (0.5×, 0.75× window; periodic K=10)
