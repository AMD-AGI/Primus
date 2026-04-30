---
name: backend-gap-report
description: Compare a Primus backend against an upstream repository or reference, verify git state, dependencies, directory changes, and integration coupling, then generate comparison reports, dashboard metadata, and a deployable dashboard index. Also owns the shared Primus engineering dashboard under `tools/backend_gap_report/`, which surfaces both backend-gap reports and weekly engineering reports as first-class sections. Use when comparing TorchTitan, Megatron, or other Primus backends with upstream branches, tags, or releases, or when integrating weekly engineering reports into the shared dashboard.
---

# Backend Gap Report & Shared Dashboard

Use this skill when the user asks to compare a Primus backend with upstream code and wants stable deliverables instead of ad hoc notes, **or** when integrating a different content type (e.g. weekly engineering reports) into the shared Primus engineering dashboard that lives under `tools/backend_gap_report/`.

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
- include `dashboard_summary` with concise homepage-ready facts when the report has enough evidence

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
- Backend deep-dive cards should render structured `dashboard_summary` fields when present. Do not make the frontend infer important conclusions from Markdown prose.

### Backend Dashboard Summary

When generating backend metadata, add a `dashboard_summary` object whenever the comparison has confirmed facts worth surfacing on the homepage:

- `headline`: one sentence explaining why this backend comparison matters
- `recommendation`: concise action label such as `monitor`, `plan sync`, or `urgent sync`
- `why_it_matters`: 3-5 concise facts or risks
- `feature_deltas`: notable upstream capabilities added since the Primus pin
- `dependency_deltas`: dependency, runtime, CUDA, ROCm, or package-channel differences
- `integration_risks`: Primus-specific coupling points or upgrade risks

All `dashboard_summary` content must be derived from the same evidence used in the detailed report. Do not invent features, versions, or risks to make the dashboard look richer.

## Weekly Engineering Report Integration

The shared dashboard under `tools/backend_gap_report/` surfaces two content
types from a single static site:

1. **Backend gap reports** — owned by this skill, stored under `docs/backend-gap/`.
2. **Weekly engineering reports** — the automated weekly Primus reports under
   `docs/weekly_reports/`.

Weekly reports share the same site shell and the same build/publish pipeline.
Routine weekly runs should update weekly data only and should not redesign
the dashboard.

### Weekly-report data plane

- Per-report metadata: `docs/weekly_reports/dashboard-data/reports/{report_id}.json`
- Aggregated index: `docs/weekly_reports/dashboard-data/index.json` (generated)
- The Markdown report itself lives at `docs/weekly_reports/{report_id}-primus-weekly.md`
  and is not bundled into the published site — the dashboard links to the
  GitHub-rendered Markdown via `report_github_url` in each metadata file.
- `report_id` uses ISO week format `YYYY-Www` (e.g. `2026-W17`).

Required fields in each weekly metadata JSON:

- `report_id`, `content_type` (`weekly-report`), `title`
- `report_path`, `report_github_url`
- `time_window` (`timezone`, `start`, `end`)
- `generated_at`, `merged_pr_count`, `category_breakdown`
- `megatron_status`, `torchtitan_status`, `primus_turbo_status`
- `recommendations` (keys: `megatron`, `torchtitan`, `primus_turbo`)
- `key_findings` (non-empty list of short, factual strings)

### Weekly index builder

Run directly to refresh just the weekly index:

```bash
python3 tools/backend_gap_report/build_weekly_reports_index.py
```

Run the full site bundle (which also invokes this script) to validate the
combined bundle:

```bash
python3 tools/backend_gap_report/build_site_bundle.py --output-dir /tmp/primus-dashboard-site
```

### Presentation rules for the shared dashboard

- The homepage should lead with the latest weekly engineering snapshot because
  the weekly report is the broad Primus status view.
- Backend gap reports should appear as backend deep-dive cards, not as a
  separate top-level dashboard module or tab.
- Backend deep-dive cards should be generated from backend metadata records and
  should automatically expand when new backend reports are added.
- The backend card should prioritize `dashboard_summary` content: why it
  matters, new upstream capabilities, dependency shifts, integration risks, and
  PDF links.
- Weekly archive/list sections should stay simple. Avoid unnecessary filters or
  history affordances when there is only one report.
- Visual style must remain calm, editorial, presentation-ready. No emoji,
  no animations, no decorative gradients beyond the existing header treatment.

### When to update weekly-report dashboard shell

- Data-only runs (standard case): update only weekly metadata under
  `docs/weekly_reports/dashboard-data/` and the refreshed Markdown report.
  Do not modify shared tooling, site, or this skill.
- Shell updates are allowed only when there is a concrete structural gap
  (missing weekly-report support, a rendering bug, a schema evolution) or
  when the user explicitly requests a shell change.

## Additional Resources

- Schema, metadata fields, and path conventions: [reference.md](reference.md)
- Concrete report and metadata examples: [examples.md](examples.md)
