#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
BACKEND_GAP_ROOT = DOCS_ROOT / "backend-gap"
DASHBOARD_DATA_ROOT = BACKEND_GAP_ROOT / "dashboard-data"
REPORTS_DIR = DASHBOARD_DATA_ROOT / "reports"
OUTPUT_PATH = DASHBOARD_DATA_ROOT / "index.json"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def get_required(data: dict, path: str):
    current = data
    parts = path.split(".")
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            fail(f"missing required field '{path}'")
        current = current[part]
    return current


def validate_date(value: str, field_name: str) -> None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        fail(f"field '{field_name}' must use YYYY-MM-DD, got {value!r}: {exc}")


def ensure_relative_to_backend_gap(path_value: str, report_id: str) -> None:
    rel_path = Path(path_value)
    if rel_path.is_absolute():
        fail(
            f"artifact path must be relative to docs/backend-gap/: " f"report={report_id}, path={path_value}"
        )

    candidate = (BACKEND_GAP_ROOT / rel_path).resolve()
    backend_gap_root = BACKEND_GAP_ROOT.resolve()

    try:
        candidate.relative_to(backend_gap_root)
    except ValueError as exc:
        fail(
            f"artifact path escapes docs/backend-gap/: " f"report={report_id}, path={path_value}, error={exc}"
        )


def load_report(report_path: Path) -> dict:
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {report_path}: {exc}")

    report_id = get_required(data, "id")
    if not isinstance(report_id, str) or not report_id.strip():
        fail(f"field 'id' must be a non-empty string in {report_path}")

    required_fields = [
        "title",
        "backend.key",
        "backend.label",
        "generated_at",
        "status",
        "scope",
        "local.source_path",
        "local.commit",
        "upstream.repo",
        "upstream.ref",
        "upstream.commit",
        "stats.commit_gap",
        "highlights",
        "artifacts",
    ]
    for field_name in required_fields:
        get_required(data, field_name)

    validate_date(data["generated_at"], "generated_at")

    highlights = data["highlights"]
    if not isinstance(highlights, list) or not highlights:
        fail(f"report '{report_id}' must have a non-empty 'highlights' list")
    if not all(isinstance(item, str) and item.strip() for item in highlights):
        fail(f"report '{report_id}' has an invalid highlight entry")

    artifacts = data["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        fail(f"report '{report_id}' must have a non-empty 'artifacts' list")

    has_pdf = False
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            fail(f"report '{report_id}' has a non-object artifact entry")
        for key in ("label", "path", "format", "language", "kind"):
            value = artifact.get(key)
            if not isinstance(value, str) or not value.strip():
                fail(f"report '{report_id}' artifact missing valid '{key}'")
        ensure_relative_to_backend_gap(artifact["path"], report_id)
        if artifact["format"] != "pdf":
            fail(
                f"report '{report_id}' has unsupported artifact format "
                f"'{artifact['format']}', only 'pdf' is allowed for publish artifacts"
            )

        has_pdf = True

        pdf_rel = Path(artifact["path"])
        source_md = (BACKEND_GAP_ROOT / pdf_rel).with_suffix(".md").resolve()
        backend_gap_root = BACKEND_GAP_ROOT.resolve()
        try:
            source_md.relative_to(backend_gap_root)
        except ValueError as exc:
            fail(
                f"report '{report_id}' source markdown path escapes docs/backend-gap/: "
                f"path={source_md}, error={exc}"
            )

        if not source_md.exists():
            fail(
                f"report '{report_id}' source markdown missing for artifact "
                f"{artifact['path']}: expected {source_md.relative_to(REPO_ROOT)}"
            )

    if not has_pdf:
        fail(f"report '{report_id}' must include at least one PDF artifact")

    return data


def build_index(reports: list[dict]) -> dict:
    backend_counts: dict[str, dict] = {}
    verified_reports = 0

    for report in reports:
        backend_key = report["backend"]["key"]
        backend_label = report["backend"]["label"]
        backend_entry = backend_counts.setdefault(
            backend_key,
            {"key": backend_key, "label": backend_label, "count": 0},
        )
        backend_entry["count"] += 1
        if report["status"] == "verified":
            verified_reports += 1

    latest_report_date = reports[0]["generated_at"] if reports else None

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "total_reports": len(reports),
            "verified_reports": verified_reports,
            "total_backends": len(backend_counts),
            "latest_report_date": latest_report_date,
        },
        "backends": sorted(
            backend_counts.values(),
            key=lambda item: (item["label"].lower(), item["key"]),
        ),
        "reports": reports,
    }


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_files = sorted(REPORTS_DIR.glob("*.json"))
    reports = [load_report(report_path) for report_path in report_files]

    seen_ids: set[str] = set()
    for report in reports:
        if report["id"] in seen_ids:
            fail(f"duplicate report id: {report['id']}")
        seen_ids.add(report["id"])

    reports.sort(
        key=lambda item: (
            item["generated_at"],
            item["backend"]["label"].lower(),
            item["id"],
        ),
        reverse=True,
    )

    output = build_index(reports)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote dashboard index with {len(reports)} report(s) to " f"{OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
