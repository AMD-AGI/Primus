#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD). All rights reserved.
###############################################################################
"""Render a tech-blog markdown file to a single self-contained HTML preview.

Every local image is inlined as a base64 data URI, so the output renders
identically wherever it is opened -- an editor preview pane, a browser on a
laptop, or an email attachment -- with no dependency on the surrounding
directory. This is what makes the preview safe to hand to a reviewer who does
not have the repository checked out.

Usage:
    python tools/blog_preview.py docs/tech_blogs/moe_package_2.0/moe_package.md
    python tools/blog_preview.py <input.md> -o <output.html>
"""

import argparse
import base64
import html
import mimetypes
import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.exit("python-markdown is required: pip install markdown")

MARKDOWN_EXTENSIONS = [
    "tables",
    "fenced_code",
    "attr_list",
    "md_in_html",
    "sane_lists",
    "footnotes",
]

TEMPLATE = """<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    max-width: 900px; margin: 0 auto; padding: 2rem 1.25rem 6rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
                 "PingFang SC", "Microsoft YaHei", Helvetica, Arial, sans-serif;
    font-size: 16px; line-height: 1.7;
  }}
  h1 {{ font-size: 2rem; line-height: 1.25; margin-top: 2.5rem; }}
  h2 {{ font-size: 1.5rem; margin-top: 2.5rem; padding-bottom: .3em;
       border-bottom: 1px solid rgba(128,128,128,.35); }}
  h3 {{ font-size: 1.2rem; margin-top: 2rem; }}
  h4 {{ font-size: 1.02rem; margin-top: 1.6rem; opacity: .9; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 1rem auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1.25rem 0;
          font-size: .9rem; display: block; overflow-x: auto; }}
  th, td {{ border: 1px solid rgba(128,128,128,.4); padding: .45rem .6rem;
           text-align: left; vertical-align: middle; }}
  th {{ background: rgba(128,128,128,.12); }}
  /* Figure-layout tables use borderless cells to place charts side by side. */
  table:has(img) td, table:has(img) th {{ border: none; background: none; }}
  code {{ background: rgba(128,128,128,.16); padding: .12em .35em;
         border-radius: 3px; font-size: .88em; }}
  pre code {{ display: block; padding: .8rem; overflow-x: auto; }}
  blockquote {{ margin: 1.2rem 0; padding: .1rem 1rem;
               border-left: 4px solid rgba(128,128,128,.5); opacity: .92; }}
  hr {{ border: none; border-top: 1px solid rgba(128,128,128,.3); margin: 2.5rem 0; }}
  a {{ color: #0969da; }}
  @media (prefers-color-scheme: dark) {{ a {{ color: #6cb6ff; }} }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def inline_images(html_text: str, base_dir: Path) -> tuple[str, int, list[str]]:
    """Replace every local <img src> with a base64 data URI.

    Returns the rewritten HTML, the number of images inlined, and the list of
    sources that could not be resolved.
    """
    missing: list[str] = []
    inlined = 0

    def replace(match: re.Match) -> str:
        nonlocal inlined
        quote, src = match.group(1), match.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (base_dir / src).resolve()
        if not path.is_file():
            missing.append(src)
            return match.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        inlined += 1
        return f"src={quote}data:{mime};base64,{payload}{quote}"

    return re.sub(r'src=(["\'])(.*?)\1', replace, html_text), inlined, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="markdown file to render")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output HTML file (default: <source>.preview.html)",
    )
    args = parser.parse_args()

    source: Path = args.source
    if not source.is_file():
        sys.exit(f"no such file: {source}")
    output: Path = args.output or source.with_suffix(".preview.html")

    text = source.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=MARKDOWN_EXTENSIONS)
    body, inlined, missing = inline_images(body, source.parent)

    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = html.escape(title_match.group(1).strip()) if title_match else source.stem
    lang = "zh" if source.stem.endswith("_zh") else "en"

    output.write_text(TEMPLATE.format(lang=lang, title=title, body=body), encoding="utf-8")

    print(f"{source} -> {output} ({inlined} images inlined)")
    for src in missing:
        print(f"  WARNING: image not found, left as-is: {src}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
