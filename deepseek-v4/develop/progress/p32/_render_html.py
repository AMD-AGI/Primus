"""Convert P32 markdown profile report to HTML with shared profile-report styling."""

from __future__ import annotations

from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = ROOT / "develop" / "profile"
MD_PATH = PROFILE_DIR / "profile-after-p32-ep8-20260511.md"
HTML_PATH = PROFILE_DIR / "profile-after-p32-ep8-20260511.html"

STYLE = """
body { font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif; max-width: 1280px; margin: 1.5rem auto; padding: 0 1rem; color: #1d1f21; line-height: 1.45; }
h1, h2 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.3rem; }
h1 { font-size: 1.5rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; }
table { border-collapse: collapse; margin: 0.8rem 0; font-size: 0.92rem; }
th, td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
code { background: #f6f8fa; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 1.5rem 0; }
blockquote { color: #57606a; border-left: 4px solid #d0d7de; padding-left: 0.8rem; margin: 0.8rem 0; font-size: 0.95em; }
"""


def main() -> None:
    md_content = MD_PATH.read_text()
    body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Plan-5 P32 - V4-Flash EP8 Trace</title>"
        f"<style>{STYLE}</style></head><body>{body}</body></html>"
    )
    HTML_PATH.write_text(html)
    print(f"wrote {HTML_PATH} ({len(html)} chars)")


if __name__ == "__main__":
    main()
