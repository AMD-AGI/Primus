# Knowledge (Experience Precipitation)

**Status**: Stub
**Written by**: humans (after curator review of LLM-generated drafts)
**Read by**: Diagnose / Re-Plan Workers

> Per §S4 governance, this directory only accepts curator-merged content. Runtime LLM writes go to `state/knowledge_drafts/`. Direct LLM writes here are FORBIDDEN.

## Reading protocol (followed by Workers)

1. Filter by `applicability` matching current `(cluster_class, model_family, framework_version)`.
2. Drop entries whose `not_applicable_when` clauses hit.
3. Sort by `confirmation_count desc`, take top-3.
4. If top-3 contain mutually conflicting hints on the same axis → discard all, fall back to model-only decision.
5. Skip entries with `status: retired` or `superseded`.

## Files

- `patterns.md` — generalized regularities
- `cases.md` — historical best-config case library
- `anti-patterns.md` — failure cases / known pitfalls
