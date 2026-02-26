#!/usr/bin/env python3
"""
md_to_site_F.py — Convert Paper F draft to site-ready format for d-nd.com

Produces:
1. MathJax-compatible markdown (LaTeX inline/display preserved as-is for MathJax)
2. metadata.json with title, description, keywords, authors, sections

The site uses MathJax v3 which renders $...$ inline and $$...$$ display math
natively from markdown — so LaTeX math needs NO conversion, just passthrough.

Author: TM3 (Claude Code)
Date: 2026-02-26
"""

import json
import re
from pathlib import Path


INPUT_DIR = Path(__file__).parent.parent / "papers"
OUTPUT_DIR = Path(__file__).parent.parent / "papers" / "site_ready"


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
        "paper_id": "F",
        "paper_label": "Paper F",
        "category": "paper",
        "section": "dnd-model",
    }

    # Title: first # heading
    m = re.search(r'^# (.+)$', content, re.MULTILINE)
    if m:
        metadata["title"] = m.group(1).strip()

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
        desc = re.sub(r'\*\*([^*]+)\*\*', r'\1', desc)
        desc = re.sub(r'\$[^$]+\$', '', desc)
        desc = desc.replace('\n', ' ')[:300]
        metadata["description"] = desc

    return metadata


def extract_sections(content: str) -> list:
    """Extract section structure for navigation."""
    sections = []
    for m in re.finditer(r'^(#{1,4})\s+(.+)$', content, re.MULTILINE):
        level = len(m.group(1))
        title = m.group(2).strip()
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
    - Clean up code blocks for display
    """
    lines = content.split('\n')
    output_lines = []
    in_header = False

    for i, line in enumerate(lines):
        # Skip the metadata block (title through first ---)
        if i == 0 and line.startswith('# '):
            in_header = True
            continue
        if in_header:
            if line.strip() == '---' and i > 3:
                in_header = False
                continue
            continue

        # Skip second --- after keywords
        if not in_header and line.strip() == '---' and i < 20:
            continue

        # Add HTML id anchors to headings
        m = re.match(r'^(#{1,4})\s+(.+)$', line)
        if m:
            title = m.group(2).strip()
            slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
            output_lines.append(f'<a id="{slug}"></a>')
            output_lines.append(line)
            continue

        output_lines.append(line)

    body = '\n'.join(output_lines)
    return body.strip()


def main():
    print("=" * 60)
    print("Paper F → Site-Ready Conversion")
    print("=" * 60)

    # Read source
    source_file = INPUT_DIR / "paper_F_draft2.md"
    content = source_file.read_text()
    print(f"Read {len(content)} chars from {source_file.name}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract metadata
    metadata = extract_metadata(content)
    sections = extract_sections(content)
    metadata["sections"] = sections
    metadata["figure_count"] = 0
    metadata["page_count_latex"] = 18

    print(f"\nMetadata:")
    print(f"  Title: {metadata['title']}")
    print(f"  Authors: {metadata['authors']}")
    print(f"  Keywords: {len(metadata['keywords'])} items")
    print(f"  Sections: {len(sections)} headings")

    # Save metadata
    meta_file = OUTPUT_DIR / "paper_F_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {meta_file.name}")

    # Clean content for site
    site_content = clean_for_site(content)

    # Save site-ready markdown
    site_file = OUTPUT_DIR / "paper_F.md"
    site_file.write_text(site_content)
    print(f"\nSite-ready markdown: {site_file.name} ({len(site_content)} chars)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files:")
    for f in sorted(OUTPUT_DIR.rglob("paper_F*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(OUTPUT_DIR)} ({size:,} bytes)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
