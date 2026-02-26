#!/usr/bin/env python3
"""
md_to_site.py — Convert Paper C draft to site-ready format for d-nd.com

Produces:
1. MathJax-compatible markdown (LaTeX inline/display preserved as-is for MathJax)
2. metadata.json with title, description, keywords, authors, sections
3. Copies SVG figures to output directory

The site uses MathJax v3 which renders $...$ inline and $$...$$ display math
natively from markdown — so LaTeX math needs NO conversion, just passthrough.

Author: TM3 (Claude Code)
Date: 2026-02-26
"""

import json
import re
import shutil
from pathlib import Path


INPUT_DIR = Path(__file__).parent.parent / "papers"
OUTPUT_DIR = Path(__file__).parent.parent / "papers" / "site_ready"
FIGURES_SRC = INPUT_DIR / "figures"


def extract_metadata(content: str) -> dict:
    """Extract metadata from the markdown header."""
    metadata = {
        "title": "",
        "subtitle": "",
        "authors": "",
        "date": "",
        "status": "",
        "keywords": [],
        "description": "",
        "paper_id": "C",
        "paper_label": "Paper C",
        "category": "paper",
        "section": "dnd-model",
    }

    # Title: first # heading
    m = re.search(r'^# (.+)$', content, re.MULTILINE)
    if m:
        metadata["title"] = m.group(1).strip()

    # Subtitle: first bold line after title
    m = re.search(r'^\*\*([^*]+)\*\*\s*$', content, re.MULTILINE)
    if m and "Authors" not in m.group(1):
        metadata["subtitle"] = m.group(1).strip()

    # Authors
    m = re.search(r'\*\*Authors?:\*\*\s*(.+)', content)
    if m:
        metadata["authors"] = m.group(1).strip()

    # Date
    m = re.search(r'\*\*Date:\*\*\s*(.+)', content)
    if m:
        metadata["date"] = m.group(1).strip()

    # Status
    m = re.search(r'\*\*Status:\*\*\s*(.+)', content)
    if m:
        metadata["status"] = m.group(1).strip()

    # Keywords
    m = re.search(r'\*\*Keywords?:\*\*\s*(.+)', content)
    if m:
        kw_str = m.group(1).strip()
        metadata["keywords"] = [k.strip() for k in kw_str.split(",")]

    # Description: first sentence of abstract
    m = re.search(r'## Abstract\s+(.+?)(?:\.|$)', content, re.DOTALL)
    if m:
        desc = m.group(1).strip()
        # Clean markdown formatting for description
        desc = re.sub(r'\*\*([^*]+)\*\*', r'\1', desc)
        desc = re.sub(r'\$[^$]+\$', '', desc)
        desc = desc.replace('\n', ' ')[:300]
        metadata["description"] = desc

    return metadata


def extract_sections(content: str) -> list:
    """Extract section structure for navigation."""
    sections = []
    for m in re.finditer(r'^(#{1,3})\s+(.+)$', content, re.MULTILINE):
        level = len(m.group(1))
        title = m.group(2).strip()
        # Create slug
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        sections.append({
            "level": level,
            "title": title,
            "slug": slug,
        })
    return sections


def clean_for_site(content: str) -> str:
    """
    Clean markdown for site display.

    MathJax handles $...$ and $$...$$ natively, so math is left as-is.
    Main changes:
    - Remove metadata header block
    - Add HTML anchors for sections
    - Clean up figure references to point to SVG files
    - Add figure img tags
    """
    lines = content.split('\n')
    output_lines = []
    in_header = False
    header_end = False

    for i, line in enumerate(lines):
        # Skip the metadata block (title through first ---)
        if i == 0 and line.startswith('# '):
            in_header = True
            continue
        if in_header:
            if line.strip() == '---' and i > 3:
                in_header = False
                header_end = True
                continue
            continue

        # Skip second --- after abstract keywords
        if header_end and line.strip() == '---':
            header_end = False
            continue

        # Add HTML id anchors to headings
        m = re.match(r'^(#{1,3})\s+(.+)$', line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
            # Keep the markdown heading but add anchor
            output_lines.append(f'<a id="{slug}"></a>')
            output_lines.append(line)
            continue

        # Convert figure references: (see Figures C1-C4) → linked
        line = re.sub(
            r'\(see Figures? (C\d+)(?:[–-](C\d+))?\)',
            lambda m: f'(see [Figure {m.group(1)}](#{m.group(1).lower()})' +
                      (f'–[{m.group(2)}](#{m.group(2).lower()})' if m.group(2) else '') + ')',
            line
        )

        # Convert (See Figure C1...) patterns
        line = re.sub(
            r'\(See Figures? (C\d+)(?:[–-](C\d+))?\)',
            lambda m: f'(See [Figure {m.group(1)}](#{m.group(1).lower()})' +
                      (f'–[{m.group(2)}](#{m.group(2).lower()})' if m.group(2) else '') + ')',
            line
        )

        output_lines.append(line)

    body = '\n'.join(output_lines)

    # Add figure blocks for each referenced figure
    figure_descriptions = {
        "C1": "Critical curvature |K_c| vs zeta zero positions t_n under three eigenvalue patterns",
        "C2": "K_gen(x, t_n) profiles at selected zeta zeros",
        "C3": "Gap analysis: consecutive differences in critical curvature values",
        "C4": "Critical locations x_c(t_n) as function of zeta zero index",
        "C5": "Nearest-neighbor spacing distributions compared to GUE Wigner surmise",
        "C6": "Eigenvalue staircase functions vs zeta zero staircase",
        "C7": "Topological charge χ_DND evolution through parameter variation",
        "C8": "Gaussian curvature landscape snapshots at different times",
    }

    figures_section = "\n\n---\n\n## Figures\n\n"
    for fig_id, desc in figure_descriptions.items():
        fig_num = fig_id[1:]  # "1", "2", etc.
        figures_section += f'<a id="c{fig_num}"></a>\n\n'
        figures_section += f'### Figure {fig_id}\n\n'
        figures_section += f'![Figure {fig_id}: {desc}](/papers/figures/fig_{fig_id}_*.svg)\n\n'
        figures_section += f'*{desc}.*\n\n'

    body += figures_section

    return body.strip()


def main():
    print("=" * 60)
    print("Paper C → Site-Ready Conversion")
    print("=" * 60)

    # Read source
    source_file = INPUT_DIR / "paper_C_draft2.md"
    content = source_file.read_text()
    print(f"Read {len(content)} chars from {source_file.name}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_out = OUTPUT_DIR / "figures"
    fig_out.mkdir(exist_ok=True)

    # Extract metadata
    metadata = extract_metadata(content)
    sections = extract_sections(content)
    metadata["sections"] = sections
    metadata["figure_count"] = 8
    metadata["page_count_latex"] = 13

    print(f"\nMetadata:")
    print(f"  Title: {metadata['title']}")
    print(f"  Authors: {metadata['authors']}")
    print(f"  Keywords: {len(metadata['keywords'])} items")
    print(f"  Sections: {len(sections)} headings")

    # Save metadata
    meta_file = OUTPUT_DIR / "paper_C_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {meta_file.name}")

    # Clean content for site
    site_content = clean_for_site(content)

    # Save site-ready markdown
    site_file = OUTPUT_DIR / "paper_C.md"
    site_file.write_text(site_content)
    print(f"\nSite-ready markdown: {site_file.name} ({len(site_content)} chars)")

    # Copy SVG figures
    svg_count = 0
    for svg in FIGURES_SRC.glob("*.svg"):
        shutil.copy2(svg, fig_out / svg.name)
        svg_count += 1
    print(f"Copied {svg_count} SVG figures to {fig_out}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(OUTPUT_DIR)} ({size:,} bytes)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
