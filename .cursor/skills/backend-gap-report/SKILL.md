---
name: backend-gap-report
description: Compare a Primus backend against an upstream repository or reference, verify git state, dependencies, directory changes, and integration coupling, then generate comparison reports, dashboard metadata, and a deployable dashboard index. Use when comparing TorchTitan, Megatron, or other Primus backends with upstream branches, tags, or releases.
---

# Backend Gap Report

Use this skill when the user asks to compare a Primus backend with upstream code and wants stable deliverables instead of ad hoc notes.

## Default Outputs

Unless the user explicitly asks otherwise, produce:

- detailed report: `docs/backend-gap/reports/<backend>/<target>/report.md`
- one-page summary: `docs/backend-gap/reports/<backend>/<target>/summary.md`
- publish-time PDF copies of both reports with the same basename
- Dashboard metadata: `docs/backend-gap/dashboard-data/reports/<backend>-<target>.json`
- Refreshed dashboard index: `docs/backend-gap/dashboard-data/index.json`

If legacy report files already exist, update them in place instead of renaming them. For new report series, keep the default artifact names unsuffixed. Only add a language suffix when a non-default variant coexists or when a legacy file pattern already established one.

## Required Inputs

Resolve these inputs before writing:

1. `backend`
2. Local source path or submodule path
3. Upstream repository and comparison ref (`main`, tag, release commit, etc.)
4. Authoritative dependency evidence files
5. Primus integration directories for that backend

If any of these are ambiguous, ask the user before proceeding.

## Workflow

### 1. Establish the Comparison Baseline

Verify:

- local version or pinned commit
- upstream target commit
- commit dates
- commit gap
- merge-base relation
- diff size

Use git facts from the actual local checkout. If the upstream ref might be stale, fetch it first.

### 2. Verify Dependency Facts

Prefer authoritative sources in this order:

1. package metadata such as `pyproject.toml`
2. runtime or CI requirements files
3. workflow install commands
4. release notes or release docs
5. README install examples

Do not treat README examples as stronger evidence than workflow or package metadata.

### 3. Verify Directory and Capability Changes

Check the actual tree or diffs for areas such as:

- model directories
- distributed runtime
- experiments
- components
- docs
- workflows
- tests

Only write facts you can confirm from the repository state.

### 4. Verify Primus Coupling

Identify direct Primus dependencies on upstream internal paths, such as:

- imports from backend internals
- monkey patches
- trainer or adapter coupling
- config object dependencies

If the report discusses upgrade cost or blast radius, ground it in these concrete coupling points.

### 5. Write the Reports

Default report set:

- detailed comparison report
- one-page summary

When continuing an existing report series, preserve the established naming pattern and structure. Keep summary and detailed versions factually consistent.

### 6. Export PDFs

Use the shared stylesheet at `tools/backend_gap_report/templates/pdf-report.css`.

Preferred command pattern:

```bash
pandoc "docs/backend-gap/reports/<backend>/<target>/<report>.md" --from gfm --standalone \
  --css "tools/backend_gap_report/templates/pdf-report.css" \
  --metadata pagetitle="<title>" \
  --pdf-engine=weasyprint \
  -o "/tmp/backend-gap-pdf/<backend>/<target>/<report>.pdf"
```

Use `pagetitle`, not `title`, to avoid duplicate visible titles in the PDF.

Note:

- Markdown reports are the tracked source artifacts in the repository.
- PDF files are generated for publishing and are not tracked in the repository.

### 7. Emit Dashboard Metadata

Create a metadata JSON file under `docs/backend-gap/dashboard-data/reports/`.

The metadata must:

- be relative to the `docs/` root
- reference publish artifact paths that can be generated
- map to markdown source files that exist in the repo
- include backend identity, refs, stats, highlights, and artifact links

Use the schema and examples in [reference.md](reference.md) and [examples.md](examples.md).

### 8. Refresh the Dashboard Index

Run:

```bash
python3 tools/backend_gap_report/build_dashboard_index.py
```

This validates metadata files and rewrites `docs/backend-gap/dashboard-data/index.json`.

### 8.5 Build the Standalone Site Bundle

For publishing or local preview, build the standalone dashboard bundle from the site templates plus generated artifacts:

```bash
python3 tools/backend_gap_report/build_site_bundle.py --output-dir /tmp/backend-gap-site
```

This assembles a publishable site root from:

- `tools/backend_gap_report/site/`
- `docs/backend-gap/dashboard-data/`
- `docs/backend-gap/reports/`

This command rebuilds dashboard index, generates PDF files from markdown report sources, builds the standalone bundle, and validates bundle integrity in one run.

### 9. Final Verification

Before finishing:

- verify the Markdown files exist
- verify the metadata JSON exists
- run the site bundle build successfully
- check lints for edited files when applicable

## Update Semantics

- For the same `backend` + `target`, update the existing files in place.
- Re-running the same report series should overwrite or refresh:
  - `report.md`
  - `summary.md`
  - `docs/backend-gap/dashboard-data/reports/<backend>-<target>.json`
  - `docs/backend-gap/dashboard-data/index.json`
- PDF artifacts are regenerated during standalone site bundle build.
- Create new sibling paths only when the backend or target changes.
- Running the skill does not trigger background automation by itself; updates happen only when the agent is explicitly asked to run the workflow in a task.

## Output Rules

- Keep the default report set as the primary artifact set.
- Do not over-emphasize the default language in filenames, labels, or dashboard copy.
- Add language suffixes or labels only when needed to distinguish a non-default variant or preserve a legacy series.
- Keep facts synchronized across detailed and summary reports.
- Prefer concise factual wording over long explanations.
- Do not invent missing versions or release claims.
- If a file contains comments only, do not call it "empty".

## Dashboard Rules

- Dashboard source templates live under `tools/backend_gap_report/site/`.
- `docs/backend-gap/` stores generated data and report artifacts, not the site templates.
- The deployed site root is a generated standalone bundle.
- Dashboard source data lives under `docs/backend-gap/dashboard-data/reports/`.
- `docs/backend-gap/dashboard-data/index.json` is generated, not hand-maintained.
- Artifact paths in metadata are relative to the standalone published site root.

## Additional Resources

- Schema, metadata fields, and path conventions: [reference.md](reference.md)
- Concrete report and metadata examples: [examples.md](examples.md)
