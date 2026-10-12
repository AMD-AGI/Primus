#!/usr/bin/env python3
"""Render a Spur status report (Markdown) as a standalone HTML page.

Self-contained on purpose: the cluster login nodes have neither `markdown` nor
`pandoc`, and the output has to survive being copied around as a single file, so
the CSS is inlined and no assets are fetched at view time.

Usage:
    python3 md_to_html.py REPORT.md [-o REPORT.html]
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

CSS = """
:root {
  color-scheme: light;
  --bg: #f4f6fb;
  --surface: #ffffff;
  --surface-muted: #f7f9fd;
  --border: #dfe4ee;
  --border-strong: #c9d2e3;
  --text: #121b2b;
  --muted: #596478;
  --accent: #1f4cc7;
  --accent-soft: #e6edff;
  --accent-dark: #102a69;
  --success: #136f43;
  --warning: #8a5a00;
  --warning-soft: #fff1d3;
  --danger: #a51f2d;
  --danger-soft: #fbe4e6;
  --shadow-soft: 0 6px 24px rgba(18, 27, 43, 0.06);
  --radius: 16px;
  --font-display: "Iowan Old Style", "Palatino Linotype", Palatino, "Hoefler Text",
    Georgia, "Times New Roman", serif;
  --font-body: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, "SFMono-Regular", "SF Mono", Menlo, "Cascadia Code",
    Consolas, "Liberation Mono", monospace;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }

body {
  font-family: var(--font-body);
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

.hero {
  background: linear-gradient(135deg, var(--accent-dark), var(--accent));
  color: #fff;
  padding: 44px 32px 40px;
}
.hero__inner { max-width: 1180px; margin: 0 auto; }
.hero .eyebrow {
  margin: 0 0 6px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 0.72rem;
  font-weight: 600;
  opacity: 0.82;
}
.hero h1 {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: clamp(1.9rem, 3.4vw, 2.6rem);
  margin: 0 0 14px;
}
.hero__meta { display: flex; flex-wrap: wrap; gap: 8px; }
.hero__meta span {
  background: rgba(255, 255, 255, 0.16);
  border: 1px solid rgba(255, 255, 255, 0.24);
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 0.82rem;
}
.hero__meta code {
  background: none;
  border: none;
  padding: 0;
  color: inherit;
  font-size: 0.82rem;
}

.layout {
  max-width: 1180px;
  margin: 0 auto;
  padding: 32px;
  display: grid;
  grid-template-columns: 232px minmax(0, 1fr);
  gap: 32px;
  align-items: start;
}

.toc {
  position: sticky;
  top: 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
  padding: 18px 18px 20px;
}
.toc h2 {
  margin: 0 0 10px;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--muted);
}
.toc ol { margin: 0; padding: 0; list-style: none; counter-reset: toc; }
.toc li { margin: 0 0 2px; }
.toc a {
  display: block;
  padding: 5px 8px;
  border-radius: 8px;
  color: var(--text);
  text-decoration: none;
  font-size: 0.87rem;
  border-left: 2px solid transparent;
}
.toc a:hover { background: var(--accent-soft); color: var(--accent-dark); }
.toc a.is-active {
  background: var(--accent-soft);
  color: var(--accent-dark);
  border-left-color: var(--accent);
  font-weight: 600;
}

.doc { min-width: 0; }
.doc section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
  padding: 24px 28px 28px;
  margin: 0 0 24px;
  scroll-margin-top: 20px;
}
.doc h2 {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 1.35rem;
  margin: 0 0 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.doc h3 { font-size: 1.02rem; margin: 22px 0 10px; }
.doc p { margin: 0 0 12px; }
.doc ul { margin: 0 0 12px; padding-left: 20px; }
.doc li { margin: 0 0 5px; }
.doc a { color: var(--accent); }

blockquote {
  margin: 0 0 16px;
  padding: 12px 16px;
  background: var(--surface-muted);
  border-left: 3px solid var(--border-strong);
  border-radius: 0 8px 8px 0;
  color: var(--muted);
  font-size: 0.9rem;
}
blockquote p:last-child { margin-bottom: 0; }

code {
  font-family: var(--font-mono);
  font-size: 0.86em;
  background: var(--surface-muted);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 1px 5px;
  word-break: break-word;
}
pre {
  background: #0f1726;
  color: #dbe4f5;
  border-radius: 12px;
  padding: 16px 18px;
  overflow-x: auto;
  margin: 0 0 16px;
  font-size: 0.83rem;
  line-height: 1.55;
}
pre code { background: none; border: none; padding: 0; color: inherit; font-size: inherit; }
pre .c { color: #7f8ea8; font-style: italic; }

.table-wrap { overflow-x: auto; margin: 0 0 16px; }
table { border-collapse: collapse; width: 100%; font-size: 0.87rem; }
thead th {
  background: var(--surface-muted);
  text-align: left;
  font-weight: 600;
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
  padding: 9px 12px;
  border-bottom: 2px solid var(--border-strong);
  white-space: nowrap;
}
tbody td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: var(--surface-muted); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }

/* Saturation cues on the cap/usage tables: full means blocked, high means close. */
.pct-full { color: var(--danger); background: var(--danger-soft); font-weight: 600; }
.pct-high { color: var(--warning); background: var(--warning-soft); font-weight: 600; }
.zero { color: var(--muted); }

footer.page-foot {
  max-width: 1180px;
  margin: 0 auto;
  padding: 0 32px 40px;
  color: var(--muted);
  font-size: 0.82rem;
}

@media (max-width: 900px) {
  .layout { grid-template-columns: minmax(0, 1fr); padding: 20px; }
  .toc { position: static; }
  .doc section { padding: 20px; }
  .hero { padding: 32px 20px; }
  footer.page-foot { padding: 0 20px 32px; }
}

@media print {
  body { background: #fff; }
  .toc { display: none; }
  .layout { display: block; padding: 0; }
  .doc section { box-shadow: none; border: none; page-break-inside: avoid; padding: 0 0 18px; }
  .hero { background: #fff; color: var(--text); padding: 0 0 16px; }
  .hero__meta span { background: none; border: 1px solid var(--border); color: var(--text); }
}
"""

JS = """
// Highlight the TOC entry for whichever section is currently on screen.
const links = Array.from(document.querySelectorAll('.toc a'));
const byId = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
const seen = new Set();
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => (e.isIntersecting ? seen.add(e.target.id) : seen.delete(e.target.id)));
    const first = links.find((a) => seen.has(a.getAttribute('href').slice(1)));
    links.forEach((a) => a.classList.toggle('is-active', a === first));
  },
  { rootMargin: '-10% 0px -70% 0px', threshold: 0 }
);
document.querySelectorAll('.doc section[id]').forEach((s) => byId.has(s.id) && observer.observe(s));
"""

CODE_SPAN = "\x00CODE{}\x00"


def inline(text: str) -> str:
    """Render inline markdown. Code spans are extracted first so their contents
    are never touched by the emphasis/link passes."""
    spans: list[str] = []

    def stash(m: re.Match[str]) -> str:
        spans.append(html.escape(m.group(1)))
        return CODE_SPAN.format(len(spans) - 1)

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", text)
    for i, span in enumerate(spans):
        text = text.replace(CODE_SPAN.format(i), f"<code>{span}</code>")
    return text


NUM_CELL = re.compile(r"^[-+]?[\d.,]+\s*(%|x)?$")
PCT_CELL = re.compile(r"^(\d+)%$")


def cell_classes(raw: str) -> str:
    """Right-align numbers and flag saturated percentages."""
    classes = []
    if NUM_CELL.match(raw) or raw in {"-", "n/a"}:
        classes.append("num")
    pct = PCT_CELL.match(raw)
    if pct:
        value = int(pct.group(1))
        if value >= 100:
            classes.append("pct-full")
        elif value >= 80:
            classes.append("pct-high")
    elif raw in {"0", "-", "n/a"}:
        classes.append("zero")
    return f' class="{" ".join(classes)}"' if classes else ""


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator(line: str) -> bool:
    return bool(re.match(r"^\|[\s:|-]+\|$", line.strip())) and "-" in line


def render_table(rows: list[str]) -> str:
    header = split_row(rows[0])
    body = [split_row(r) for r in rows[2:]]
    out = ['<div class="table-wrap"><table><thead><tr>']
    out += [f"<th>{inline(c)}</th>" for c in header]
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        out += [f"<td{cell_classes(c)}>{inline(c)}</td>" for c in row]
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def render_code(lang: str, lines: list[str]) -> str:
    body = []
    for line in lines:
        escaped = html.escape(line)
        # Shell comments carry most of the meaning in the command appendix.
        if lang in {"bash", "sh", "shell"} and "#" in line:
            escaped = re.sub(r"(#[^&<>]*)$", r'<span class="c">\1</span>', escaped)
        body.append(escaped)
    return f"<pre><code>{chr(10).join(body)}</code></pre>"


def slugify(text: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", re.sub(r"[`*]", "", text).lower()).strip("-") or "section"
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    used.add(slug)
    return slug


def convert(md: str) -> tuple[str, str, list[str], list[tuple[str, str]]]:
    """Return (body_html, title, meta_items, toc) for the report."""
    lines = md.splitlines()
    title = "Report"
    meta: list[str] = []
    toc: list[tuple[str, str]] = []
    used: set[str] = set()
    out: list[str] = []
    open_section = False
    i = 0

    def close() -> None:
        nonlocal open_section
        if open_section:
            out.append("</section>")
            open_section = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append(render_code(lang, block))
            continue

        if stripped.startswith("# "):
            title = stripped[2:].strip()
            i += 1
            # The leading bullet list under the H1 is report metadata, not content.
            while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("- ")):
                if lines[i].strip():
                    meta.append(inline(lines[i].strip()[2:]))
                i += 1
            continue

        if stripped.startswith("## "):
            close()
            text = stripped[3:].strip()
            slug = slugify(text, used)
            toc.append((slug, text))
            out.append(f'<section id="{slug}"><h2>{inline(text)}</h2>')
            open_section = True
            i += 1
            continue

        if stripped.startswith("### "):
            out.append(f"<h3>{inline(stripped[4:].strip())}</h3>")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(render_table(rows))
            continue

        if stripped.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
            continue

        if stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"<li>{inline(lines[i].strip()[2:])}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue

        if not stripped:
            i += 1
            continue

        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^\s*([#>|-]|```)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
        else:
            out.append(f"<p>{inline(stripped)}</p>")
            i += 1

    close()
    return "".join(out), title, meta, toc


def build_page(md: str, source_name: str) -> str:
    body, title, meta, toc = convert(md)
    meta_html = "".join(f"<span>{m}</span>" for m in meta)
    toc_html = "".join(f'<li><a href="#{s}">{html.escape(t)}</a></li>' for s, t in toc)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<header class="hero">
  <div class="hero__inner">
    <p class="eyebrow">Primus Engineering</p>
    <h1>{html.escape(title)}</h1>
    <div class="hero__meta">{meta_html}</div>
  </div>
</header>
<div class="layout">
  <nav class="toc" aria-label="Contents">
    <h2>Contents</h2>
    <ol>{toc_html}</ol>
  </nav>
  <main class="doc">{body}</main>
</div>
<footer class="page-foot">
  Rendered from <code>{html.escape(source_name)}</code> &mdash; read-only snapshot,
  regenerate with <code>spur_status.py</code> for current numbers.
</footer>
<script>{JS}</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Markdown report to render")
    parser.add_argument("-o", "--output", type=Path, help="Output path (default: source with .html)")
    args = parser.parse_args()

    if not args.source.is_file():
        print(f"error: {args.source} not found", file=sys.stderr)
        return 1

    output = args.output or args.source.with_suffix(".html")
    output.write_text(build_page(args.source.read_text(), args.source.name), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
