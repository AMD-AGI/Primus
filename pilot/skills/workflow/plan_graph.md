# PlanGraph (Search-Space Tree)

**Status**: Stub

Tree-shaped solution-space representation: nodes (champion / shelved / dead / running), `frontier`, `exhausted_neighborhoods`, derivation relations. The fundamental data structure for §7's solution guarantees.

## TODO

- [ ] Field-by-field semantics (parent / status / derived_axis / champion_at)
- [ ] Status transition rules (completed → shelved → dead, etc.)
- [ ] exhausted_neighborhoods radius computation per axis type
- [ ] Schema: `schemas/plan_graph.schema.json`
