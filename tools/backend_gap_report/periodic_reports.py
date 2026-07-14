#!/usr/bin/env python3
"""Shared core for the unified *periodic engineering report* data plane.

Weekly and monthly Primus engineering reports share one schema, one set of
validation rules, and one combined dashboard index. This module is the single
source of truth for both cadences so the dashboard renders them through one
generic path instead of two parallel implementations.

Layout per cadence (``<cadence>`` is ``weekly_reports`` or ``monthly_reports``):

- per-report metadata: ``docs/<cadence>/dashboard-data/reports/{report_id}.json``
- per-cadence index:    ``docs/<cadence>/dashboard-data/index.json`` (generated)

The combined index consumed by the dashboard is assembled at bundle-build time
(see ``build_site_bundle.py``) and is *not* committed to the repository.

The thin CLI wrappers ``build_weekly_reports_index.py`` and
``build_monthly_reports_index.py`` simply call into this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Pattern

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"

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


@dataclass(frozen=True)
class Cadence:
    """Configuration for one report cadence."""

    key: str
    label: str
    content_type: str
    id_regex: Pattern[str]
    docs_dirname: str

    @property
    def dashboard_data_root(self) -> Path:
        return DOCS_ROOT / self.docs_dirname / "dashboard-data"

    @property
    def reports_dir(self) -> Path:
        return self.dashboard_data_root / "reports"

    @property
    def index_path(self) -> Path:
        return self.dashboard_data_root / "index.json"


CADENCES: dict[str, Cadence] = {
    "weekly": Cadence(
        key="weekly",
        label="Weekly",
        content_type="weekly-report",
        id_regex=re.compile(r"^\d{4}-W\d{2}$"),
        docs_dirname="weekly_reports",
    ),
    "monthly": Cadence(
        key="monthly",
        label="Monthly",
        content_type="monthly-report",
        id_regex=re.compile(r"^\d{4}-\d{2}$"),
        docs_dirname="monthly_reports",
    ),
}


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


def _parse_timestamp(value: object) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    Unparseable / missing values sort oldest so a malformed entry never
    masquerades as the latest report.
    """

    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def validate_report(path: Path, data: dict, cadence: Cadence) -> dict:
    """Validate one per-report metadata file against the shared schema."""

    for field in REQUIRED_TOP_FIELDS:
        if field not in data:
            fail(f"{path}: missing required field '{field}'")

    report_id = data["report_id"]
    if not isinstance(report_id, str) or not cadence.id_regex.match(report_id):
        fail(
            f"{path}: field 'report_id' must match {cadence.id_regex.pattern} "
            f"for the {cadence.key} cadence (got {report_id!r})"
        )

    if data["content_type"] != cadence.content_type:
        fail(f"{path}: field 'content_type' must be '{cadence.content_type}'")

    report_rel_path = data["report_path"]
    if not isinstance(report_rel_path, str) or not report_rel_path.strip():
        fail(f"{path}: field 'report_path' must be a non-empty string")
    if not (REPO_ROOT / report_rel_path).exists():
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


def load_cadence_reports(cadence: Cadence) -> list[dict]:
    """Load + validate every per-report file for a cadence.

    Each returned record is tagged with a derived ``cadence`` field. A missing
    cadence directory is treated as zero reports (the cadence is not active yet).
    """

    reports_dir = cadence.reports_dir
    if not reports_dir.exists():
        return []

    reports: list[dict] = []
    seen_ids: set[str] = set()
    for metadata_file in sorted(reports_dir.glob("*.json")):
        payload = _load_json(metadata_file)
        validated = validate_report(metadata_file, payload, cadence)
        report_id = validated["report_id"]
        if report_id in seen_ids:
            fail(f"duplicate {cadence.key} report id: {report_id}")
        seen_ids.add(report_id)
        record = dict(validated)
        record["cadence"] = cadence.key
        reports.append(record)
    return reports


def _sort_reports(reports: list[dict]) -> list[dict]:
    return sorted(
        reports,
        key=lambda item: (_parse_timestamp(item.get("generated_at")), item.get("report_id", "")),
        reverse=True,
    )


def _recommendation_rollup(reports: list[dict]) -> tuple[dict[str, int], list[str]]:
    recommendation_counts: dict[str, int] = {}
    tracked_targets: set[str] = set()
    for report in reports:
        for key, value in report.get("recommendations", {}).items():
            tracked_targets.add(key)
            recommendation_counts[value] = recommendation_counts.get(value, 0) + 1
    return recommendation_counts, sorted(tracked_targets)


def build_cadence_index(cadence: Cadence) -> dict:
    """Build the per-cadence index payload (committed by each automation run)."""

    reports = _sort_reports(load_cadence_reports(cadence))
    latest = reports[0] if reports else None
    recommendation_counts, tracked_targets = _recommendation_rollup(reports)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cadence": cadence.key,
        "summary": {
            "total_reports": len(reports),
            "latest_report_id": latest["report_id"] if latest else None,
            "latest_generated_at": latest["generated_at"] if latest else None,
            "latest_merged_pr_count": latest["merged_pr_count"] if latest else 0,
            "tracked_drift_targets": tracked_targets,
            "recommendation_counts": recommendation_counts,
        },
        "reports": reports,
    }


def write_cadence_index(cadence: Cadence) -> int:
    """Validate + (re)write the per-cadence ``index.json``. Returns report count."""

    cadence.dashboard_data_root.mkdir(parents=True, exist_ok=True)
    cadence.reports_dir.mkdir(parents=True, exist_ok=True)
    payload = build_cadence_index(cadence)
    cadence.index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(payload["reports"])


def load_all_reports() -> list[dict]:
    """Load + validate reports across every cadence, newest first."""

    combined: list[dict] = []
    for cadence in CADENCES.values():
        combined.extend(load_cadence_reports(cadence))
    return _sort_reports(combined)


def build_combined_index() -> dict:
    """Assemble the unified index consumed by the dashboard frontend."""

    reports = load_all_reports()
    latest = reports[0] if reports else None
    recommendation_counts, tracked_targets = _recommendation_rollup(reports)
    cadence_counts = {key: 0 for key in CADENCES}
    for report in reports:
        cadence_counts[report["cadence"]] = cadence_counts.get(report["cadence"], 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "total_reports": len(reports),
            "cadence_counts": cadence_counts,
            "latest_report_id": latest["report_id"] if latest else None,
            "latest_cadence": latest["cadence"] if latest else None,
            "latest_generated_at": latest["generated_at"] if latest else None,
            "latest_merged_pr_count": latest["merged_pr_count"] if latest else 0,
            "tracked_drift_targets": tracked_targets,
            "recommendation_counts": recommendation_counts,
        },
        "reports": reports,
    }
