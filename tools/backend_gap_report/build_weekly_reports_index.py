#!/usr/bin/env python3
"""Build the aggregated weekly-report dashboard index.

Scans ``docs/weekly_reports/dashboard-data/reports/*.json`` (per-week metadata
produced by the weekly-report automation), validates the minimum schema
required by the shared dashboard, and writes a single aggregated
``docs/weekly_reports/dashboard-data/index.json`` suitable for a fully
static, backend-free dashboard.

This lives beside ``build_dashboard_index.py`` (which handles backend-gap
reports) so the shared tooling owns both data-plane builders.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
WEEKLY_REPORTS_ROOT = DOCS_ROOT / "weekly_reports"
DASHBOARD_DATA_ROOT = WEEKLY_REPORTS_ROOT / "dashboard-data"
REPORTS_DIR = DASHBOARD_DATA_ROOT / "reports"
OUTPUT_PATH = DASHBOARD_DATA_ROOT / "index.json"

REPORT_ID_RE = re.compile(r"^\d{4}-W\d{2}$")

REQUIRED_TOP_FIELDS = (
    "report_id",
    "content_type",
    "title",
    "report_path",
    "report_github_url",
    "time_window",
    "generated_at",
    "merged_pr_count",
    "category_breakdown",
    "megatron_status",
    "torchtitan_status",
    "primus_turbo_status",
    "recommendations",
    "key_findings",
)

REQUIRED_TIME_WINDOW_FIELDS = ("timezone", "start", "end")
REQUIRED_RECOMMENDATION_KEYS = ("megatron", "torchtitan", "primus_turbo")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing required file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")

    if not isinstance(payload, dict):
        fail(f"JSON payload in {path} must be an object")
    return payload


def _validate_report(path: Path, data: dict) -> dict:
    for field in REQUIRED_TOP_FIELDS:
        if field not in data:
            fail(f"{path}: missing required field '{field}'")

    report_id = data["report_id"]
    if not isinstance(report_id, str) or not REPORT_ID_RE.match(report_id):
        fail(f"{path}: field 'report_id' must match YYYY-Www (got {report_id!r})")

    if data["content_type"] != "weekly-report":
        fail(f"{path}: field 'content_type' must be 'weekly-report'")

    report_rel_path = data["report_path"]
    if not isinstance(report_rel_path, str) or not report_rel_path.strip():
        fail(f"{path}: field 'report_path' must be a non-empty string")
    markdown_path = REPO_ROOT / report_rel_path
    if not markdown_path.exists():
        fail(f"{path}: report_path not found in repo: {report_rel_path}")

    time_window = data["time_window"]
    if not isinstance(time_window, dict):
        fail(f"{path}: field 'time_window' must be an object")
    for tw_field in REQUIRED_TIME_WINDOW_FIELDS:
        if tw_field not in time_window:
            fail(f"{path}: time_window missing '{tw_field}'")

    if not isinstance(data["merged_pr_count"], int) or data["merged_pr_count"] < 0:
        fail(f"{path}: merged_pr_count must be a non-negative integer")

    if not isinstance(data["category_breakdown"], dict):
        fail(f"{path}: category_breakdown must be an object")

    recommendations = data["recommendations"]
    if not isinstance(recommendations, dict):
        fail(f"{path}: recommendations must be an object")
    for key in REQUIRED_RECOMMENDATION_KEYS:
        if key not in recommendations or not isinstance(recommendations[key], str):
            fail(f"{path}: recommendations missing or invalid '{key}'")

    key_findings = data["key_findings"]
    if not isinstance(key_findings, list) or not key_findings:
        fail(f"{path}: key_findings must be a non-empty list")
    for finding in key_findings:
        if not isinstance(finding, str) or not finding.strip():
            fail(f"{path}: key_findings contains invalid entry")

    return data


def _week_sort_key(report_id: str) -> tuple[int, int]:
    year_str, week_str = report_id.split("-W", 1)
    return int(year_str), int(week_str)


def _build_summary(reports: list[dict]) -> dict:
    latest = reports[0] if reports else None
    recommendation_counts: dict[str, int] = {}
    tracked_targets = set()
    for report in reports:
        for key, value in report["recommendations"].items():
            tracked_targets.add(key)
            recommendation_counts[value] = recommendation_counts.get(value, 0) + 1

    return {
        "total_reports": len(reports),
        "latest_report_id": latest["report_id"] if latest else None,
        "latest_generated_at": latest["generated_at"] if latest else None,
        "latest_merged_pr_count": latest["merged_pr_count"] if latest else 0,
        "tracked_drift_targets": sorted(tracked_targets),
        "recommendation_counts": recommendation_counts,
    }


def build_index(reports: list[dict]) -> dict:
    reports_sorted = sorted(
        reports,
        key=lambda item: _week_sort_key(item["report_id"]),
        reverse=True,
    )
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": _build_summary(reports_sorted),
        "reports": reports_sorted,
    }


def main() -> int:
    DASHBOARD_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata_files = sorted(REPORTS_DIR.glob("*.json"))
    reports: list[dict] = []
    seen_ids: set[str] = set()
    for metadata_file in metadata_files:
        payload = _load_json(metadata_file)
        validated = _validate_report(metadata_file, payload)
        report_id = validated["report_id"]
        if report_id in seen_ids:
            fail(f"duplicate weekly-report id: {report_id}")
        seen_ids.add(report_id)
        reports.append(validated)

    output = build_index(reports)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote weekly-report dashboard index with {len(reports)} report(s) to "
        f"{OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
