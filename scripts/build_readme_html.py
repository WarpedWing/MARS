#!/usr/bin/env python3
"""Convert README.md files to styled HTML for help documentation.

Usage:
    uv run python scripts/build_readme_html.py

This script converts module README files to HTML pages that match the MARS help
documentation styling. Generated HTML files are committed to the repo and linked
from the main help file.
"""

import re
from pathlib import Path

import markdown

# Project root (script is in scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# READMEs to convert: (source_path, output_filename, title)
READMES = [
    (
        "src/mars/report_modules/wifi_report_generator/README.md",
        "wifi_report.html",
        "Wi-Fi Report Generator",
        "#report_modules",
    ),
    (
        "src/mars/report_modules/biome_parser/README.md",
        "biome_parser.html",
        "Biome SEGB Parser",
        "#report_modules",
    ),
    (
        "src/mars/report_modules/firefox_cache_parser/README.md",
        "firefox_cache.html",
        "Firefox Cache Parser",
        "#report_modules",
    ),
    (
        "src/mars/report_modules/firefox_jsonlz4_parser/README.md",
        "firefox_jsonlz4.html",
        "Firefox JSONLZ4 Parser",
        "#report_modules",
    ),
    (
        "src/mars/plotter/README.md",
        "plotter.html",
        "MARS Plotter",
        "#reports",
    ),
    (
        "src/mars/catalog/README.md",
        "catalog.html",
        "MARS Artifact Recovery Catalog (ARC) Manager",
        "#settings",
    ),
    (
        "src/mars/carver/README.md",
        "carver.html",
        "MARS Carver",
        "#candidates-scan",
    ),
    (
        "src/mars/report_modules/README.md",
        "report_modules.html",
        "MARS Report Modules",
        "#report_modules",
    ),
    (
        "src/mars/pipeline/README.md",
        "pipeline.html",
        "MARS Pipeline",
        "#additional_docs",
    ),
    (
        "src/mars/pipeline/lf_processor/README.md",
        "lf_processor.html",
        "Lost & Found (LF) Processor",
        "#additional_docs",
    ),
    (
        "src/mars/pipeline/fingerprinter/README.md",
        "fingerprinter.html",
        "Text Fingerprinter",
        "#additional_docs",
    ),
]

OUTPUT_DIR = PROJECT_ROOT / "src" / "resources" / "help" / "pages"

# HTML template that matches the main help styling
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title} - MARS Documentation</title>
    <link href="../default.css" rel="stylesheet" type="text/css" />
    <style>
      /* Additional styles for markdown content */
      main {{
        max-width: 900px;
        margin: 0 auto;
        padding: 0.5rem 3rem;
      }}
      .back-link {{
        display: inline-block;
        margin-top: 1.0rem;
        color: var(--primary);
        text-decoration: none;
      }}
      .back-link:hover {{
        text-decoration: underline;
      }}
      blockquote {{
        border-left: 3px solid var(--primary);
        margin: 1rem 0;
        padding-left: 1rem;
        color: var(--text-muted);
      }}
    </style>
  </head>
  <body data-theme="dark">
    <main>
      <div style="display: flex; justify-content: space-between; align-items: start;">
        <div>
          <a href="../mars_help.html{return_anchor}" class="back-link">&larr; Back to Help</a>
        </div>
        <div>
          <img src="../../images/WarpedWingLabsLogo_Horizontal_W500.png" height="100" alt="WarpedWing Labs" />
        </div>
      </div>
      {content}
    <footer class="footer">
      <p>MARS (macOS Artifact Recovery Suite) by WarpedWing Labs</p>
    </footer>
    </main>

    <button class="theme-toggle" onclick="toggleTheme()">&#9728;&#65039; Light</button>
    <script>
      (function () {{
        const saved = localStorage.getItem("mars-theme");
        if (saved === "light") {{
          document.body.removeAttribute("data-theme");
          document.querySelector(".theme-toggle").textContent = "\\ud83c\\udf19 Dark";
        }} else {{
          document.body.setAttribute("data-theme", "dark");
        }}
      }})();
      function toggleTheme() {{
        const body = document.body;
        const btn = document.querySelector(".theme-toggle");
        const isDark = body.getAttribute("data-theme") === "dark";
        if (isDark) {{
          body.removeAttribute("data-theme");
          btn.textContent = "\\ud83c\\udf19 Dark";
          localStorage.setItem("mars-theme", "light");
        }} else {{
          body.setAttribute("data-theme", "dark");
          btn.textContent = "\\u2600\\ufe0f Light";
          localStorage.removeItem("mars-theme");
        }}
      }}
    </script>
  </body>
</html>
"""


def strip_relative_links(html: str) -> str:
    """Convert relative links to styled text, keeping external links.

    Relative markdown links like [file.py](file.py) become broken <a href="file.py">
    tags when HTML is displayed outside the file system context (e.g., in help browser).
    This converts them to <code> styled text while preserving external http(s) links.

    Args:
        html: HTML content with <a> tags

    Returns:
        HTML with relative links converted to <code> elements
    """
    # Pattern matches <a href="...">text</a> where href doesn't start with http
    pattern = r'<a href="(?!https?://)([^"]+)">([^<]+)</a>'
    # Replace with code-styled text
    return re.sub(pattern, r"<code>\2</code>", html)


def convert_readme(source_path: Path, output_path: Path, title: str, return_anchor: str) -> None:
    """Convert a single README.md to HTML."""
    if not source_path.exists():
        print(f"  WARNING: {source_path} not found, skipping")
        return

    # Read markdown content
    md_content = source_path.read_text(encoding="utf-8")

    # Convert to HTML with extensions for tables and fenced code blocks
    md = markdown.Markdown(extensions=["tables", "fenced_code"])
    html_content = md.convert(md_content)

    # Strip relative links (they don't work in help browser context)
    html_content = strip_relative_links(html_content)

    # Wrap in template
    full_html = HTML_TEMPLATE.format(title=title, content=html_content, return_anchor=return_anchor)

    # Write output
    output_path.write_text(full_html, encoding="utf-8")
    print(f"  Created: {output_path.name}")


def main() -> None:
    """Convert all READMEs to HTML."""
    print("Converting README files to HTML...")

    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for source_rel, output_name, title, return_anchor in READMES:
        source_path = PROJECT_ROOT / source_rel
        output_path = OUTPUT_DIR / output_name
        convert_readme(source_path, output_path, title, return_anchor)

    print(f"\nDone! HTML files written to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
