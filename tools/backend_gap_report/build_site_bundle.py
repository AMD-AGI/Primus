#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_GAP_DOCS_ROOT = REPO_ROOT / "docs" / "backend-gap"
SITE_SOURCE_ROOT = REPO_ROOT / "tools" / "backend_gap_report" / "site"
SOURCE_DASHBOARD_DATA_DIR = BACKEND_GAP_DOCS_ROOT / "dashboard-data"
METADATA_REPORTS_DIR = SOURCE_DASHBOARD_DATA_DIR / "reports"
PDF_TEMPLATE = REPO_ROOT / "tools" / "backend_gap_report" / "templates" / "pdf-report.css"
BUILD_INDEX_SCRIPT = Path(__file__).resolve().with_name("build_dashboard_index.py")
REQUIRED_REPORT_FIELDS = (
    "id",
    "title",
    "backend",
    "generated_at",
    "status",
    "scope",
    "highlights",
    "artifacts",
)
REQUIRED_ARTIFACT_FIELDS = ("label", "path", "format", "language", "kind")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing required file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")

    if not isinstance(payload, dict):
        fail(f"JSON payload in {path} must be an object")
    return payload


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        fail(f"missing source path: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def ensure_within(root: Path, candidate: Path, context: str) -> Path:
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        fail(f"{context} escapes {root_resolved}: {candidate_resolved} ({exc})")
    return candidate_resolved


def extract_markdown_title(markdown_path: Path) -> str:
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.startswith("# "):
            return text[2:].strip()
    return markdown_path.stem


def run_build_index() -> None:
    print("[backend-gap] Rebuild dashboard index", flush=True)
    try:
        subprocess.run(["python3", str(BUILD_INDEX_SCRIPT)], check=True)
    except subprocess.CalledProcessError as exc:
        fail(f"failed to rebuild dashboard index (exit={exc.returncode})")


def load_report_metadata() -> list[dict]:
    metadata_files = sorted(METADATA_REPORTS_DIR.glob("*.json"))
    reports: list[dict] = []
    for metadata_file in metadata_files:
        reports.append(load_json(metadata_file))
    return reports


def render_pdf(markdown_path: Path, output_pdf_path: Path) -> None:
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    title = extract_markdown_title(markdown_path)
    command = [
        "pandoc",
        str(markdown_path),
        "--from",
        "gfm",
        "--standalone",
        "--css",
        str(PDF_TEMPLATE),
        "--metadata",
        f"pagetitle={title}",
        "--pdf-engine=weasyprint",
        "-o",
        str(output_pdf_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        fail(f"pandoc is required to build PDF artifacts: {exc}")
    except subprocess.CalledProcessError as exc:
        fail(f"failed to render PDF from {markdown_path}: " f"stdout={exc.stdout}\nstderr={exc.stderr}")


def build_pdf_artifacts(output_dir: Path) -> None:
    reports = load_report_metadata()
    for report in reports:
        report_id = report.get("id", "<unknown>")
        artifacts = report.get("artifacts", [])
        if not isinstance(artifacts, list):
            fail(f"report '{report_id}' has invalid 'artifacts' field")

        for artifact in artifacts:
            if not isinstance(artifact, dict):
                fail(f"report '{report_id}' contains a non-object artifact")
            if artifact.get("format") != "pdf":
                continue

            artifact_rel_path = Path(str(artifact.get("path", "")))
            if artifact_rel_path.is_absolute():
                fail(f"report '{report_id}' artifact path must be relative: " f"{artifact_rel_path}")

            source_markdown = (BACKEND_GAP_DOCS_ROOT / artifact_rel_path).with_suffix(".md")
            ensure_within(BACKEND_GAP_DOCS_ROOT, source_markdown, "source markdown")
            if not source_markdown.exists():
                fail(
                    f"report '{report_id}' source markdown missing for artifact "
                    f"{artifact_rel_path}: expected {source_markdown.relative_to(REPO_ROOT)}"
                )

            output_pdf = output_dir / artifact_rel_path
            ensure_within(output_dir, output_pdf, "output pdf")
            render_pdf(source_markdown, output_pdf)


def validate_report_entry(report: dict, bundle_dir: Path) -> str:
    for field in REQUIRED_REPORT_FIELDS:
        if field not in report:
            fail(f"report entry missing required field: {field}")

    report_id = report["id"]
    if not isinstance(report_id, str) or not report_id.strip():
        fail("report field 'id' must be a non-empty string")

    artifacts = report["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        fail(f"report '{report_id}' must contain a non-empty artifacts list")

    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            fail(f"report '{report_id}' has a non-object artifact at index {index}")

        for field in REQUIRED_ARTIFACT_FIELDS:
            value = artifact.get(field)
            if not isinstance(value, str) or not value.strip():
                fail(f"report '{report_id}' artifact[{index}] missing valid field '{field}'")

        artifact_file = ensure_within(
            bundle_dir,
            bundle_dir / artifact["path"],
            f"report '{report_id}' artifact[{index}]",
        )
        if not artifact_file.exists():
            fail(f"report '{report_id}' artifact[{index}] path not found in bundle: " f"{artifact['path']}")

    return report_id


def validate_bundle(bundle_dir: Path) -> None:
    index_html = bundle_dir / "index.html"
    if not index_html.exists():
        fail(f"bundle missing entrypoint: {index_html}")

    payload = load_json(bundle_dir / "dashboard-data" / "index.json")
    for key in ("summary", "backends", "reports"):
        if key not in payload:
            fail(f"dashboard-data index missing required key: {key}")

    reports = payload["reports"]
    if not isinstance(reports, list):
        fail("dashboard-data index field 'reports' must be a list")

    index_report_ids: set[str] = set()
    for report in reports:
        if not isinstance(report, dict):
            fail("dashboard-data index contains a non-object report entry")
        report_id = validate_report_entry(report, bundle_dir)
        if report_id in index_report_ids:
            fail(f"duplicate report id in index: {report_id}")
        index_report_ids.add(report_id)

    metadata_dir = bundle_dir / "dashboard-data" / "reports"
    if not metadata_dir.exists():
        fail(f"bundle missing report metadata directory: {metadata_dir}")

    metadata_ids: set[str] = set()
    for metadata_file in sorted(metadata_dir.glob("*.json")):
        metadata = load_json(metadata_file)
        metadata_id = metadata.get("id")
        if not isinstance(metadata_id, str) or not metadata_id.strip():
            fail(f"metadata file missing valid 'id': {metadata_file}")
        if metadata_id in metadata_ids:
            fail(f"duplicate metadata id in dashboard-data/reports: {metadata_id}")
        metadata_ids.add(metadata_id)

    if index_report_ids != metadata_ids:
        fail(
            "report ids in dashboard-data/index.json and "
            "dashboard-data/reports/*.json are inconsistent: "
            f"index={sorted(index_report_ids)}, metadata={sorted(metadata_ids)}"
        )


def build_site(output_dir: Path) -> None:
    run_build_index()
    if not PDF_TEMPLATE.exists():
        fail(f"missing PDF template: {PDF_TEMPLATE}")
    if not SOURCE_DASHBOARD_DATA_DIR.exists():
        fail(f"missing dashboard data directory: {SOURCE_DASHBOARD_DATA_DIR}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[backend-gap] Build standalone dashboard bundle", flush=True)
    copy_tree(SITE_SOURCE_ROOT, output_dir)
    copy_tree(SOURCE_DASHBOARD_DATA_DIR, output_dir / "dashboard-data")
    build_pdf_artifacts(output_dir)
    print("[backend-gap] Validate standalone dashboard bundle", flush=True)
    validate_bundle(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone backend-gap dashboard publish bundle.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the standalone site bundle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    build_site(output_dir)
    print(f"[backend-gap] Done: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
