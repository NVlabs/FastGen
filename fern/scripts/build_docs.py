#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build Fern pages from an explicit inventory of public FastGen READMEs.

The READMEs remain the source of truth. Generated files are ignored by Git and
rebuilt before validation, preview, or publishing. No FastGen imports are needed.
Only PAGES below are included; internal dev/ docs and vendored READMEs are not
discovered automatically. Authored pages and navigation live in fern/docs/.

Run from any directory: python3 /path/to/fastgen/fern/scripts/build_docs.py
Use --source-ref <public-tag-or-sha> to pin GitHub links (default: main).
Output: fern/docs/generated/*.mdx. See fern/README.md for the full build workflow.
"""

import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote, unquote, urlsplit

REPO = Path(__file__).resolve().parents[2]
GITHUB = "https://github.com/NVlabs/FastGen"
SITE = "/fastgen"


class Page(NamedTuple):
    source: str
    slug: str
    title: str
    sections: tuple[str, ...] = ()


PAGES = (
    Page("README.md", "get-started", "Get started", ("Setup", "Quick Start")),
    Page("README.md", "resources", "Resources", ("Adoption, Collaboration, and Usage", "Reference")),
    Page("scripts/README.md", "inference", "Inference and evaluation"),
    Page("comfyui/README.md", "comfyui", "ComfyUI"),
    Page("fastgen/methods/README.md", "methods", "Choose a method"),
    Page("fastgen/methods/trajectory_matching/README.md", "trajectory-matching", "Trajectory matching"),
    Page("fastgen/methods/distribution_matching/README.md", "distribution-matching", "Distribution matching"),
    Page("fastgen/methods/fine_tuning/README.md", "fine-tuning", "Fine-tuning"),
    Page("fastgen/networks/README.md", "networks", "Networks and pretrained models"),
    Page("fastgen/datasets/README.md", "datasets", "Datasets"),
    Page("fastgen/configs/README.md", "configuration", "Configuration"),
    Page("fastgen/callbacks/README.md", "callbacks", "Callbacks"),
    Page("fastgen/networks/kv_cache/README.md", "kv-cache", "KV cache"),
    Page("fastgen/third_party/README.md", "third-party", "Third-party components"),
    Page("licenses/README.md", "licenses", "Licenses"),
)
PAGE_ROUTES = {page.source: f"{SITE}/{page.slug}" for page in PAGES if page.source != "README.md"}
PAGE_ROUTES["README.md"] = SITE
LINK = re.compile(r"(!?\[[^\]\n]*\])\(([^\s)]+)([^)]*)\)")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def source_url(path: Path, ref: str) -> str:
    relative = path.relative_to(REPO).as_posix()
    kind = "tree" if path.is_dir() else "blob"
    return f"{GITHUB}/{kind}/{quote(ref, safe='')}/{quote(relative, safe='/')}"


def rewrite_link(match: re.Match, source: Path, ref: str) -> str:
    label, target, suffix = match.groups()
    url = urlsplit(target)
    if url.scheme or url.netloc or target.startswith(("#", "/")):
        return match.group(0)
    destination = (source.parent / unquote(url.path)).resolve()
    if not destination.is_relative_to(REPO):
        raise ValueError(f"Link escapes the repository: {source}: {target}")
    if not destination.exists():
        raise ValueError(f"Missing link destination: {source}: {target}")
    relative = destination.relative_to(REPO).as_posix()
    if not label.startswith("!") and relative in PAGE_ROUTES:
        resolved = PAGE_ROUTES[relative]
        if relative == "README.md" and url.fragment:
            # Root README headings other than the overview stay reachable on GitHub.
            resolved = source_url(destination, ref)
    else:
        resolved = source_url(destination, ref)
        if label.startswith("!"):
            resolved = (
                f"https://raw.githubusercontent.com/NVlabs/FastGen/{quote(ref, safe='')}/{quote(relative, safe='/')}"
            )
    if url.query:
        resolved += f"?{url.query}"
    if url.fragment:
        resolved += f"#{url.fragment}"
    return f"{label}({resolved}{suffix})"


def render(page: Page, ref: str) -> str:
    source = REPO / page.source
    lines = source.read_text().splitlines()
    result = []
    fence = ""
    active = not page.sections
    found_sections = set()
    for line in lines:
        boundary = FENCE.match(line)
        if boundary:
            marker = boundary.group(1)
            if not fence:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = ""
            if active:
                result.append(line)
            continue
        if not fence and page.sections and line.startswith("## "):
            title = line[3:].strip()
            active = title in page.sections
            if active:
                found_sections.add(title)
        if not active:
            continue
        if not fence:
            if line.startswith("# "):
                continue  # Fern supplies the page H1.
            line = LINK.sub(lambda match: rewrite_link(match, source, ref), line)
        result.append(line)
    if page.sections and found_sections != set(page.sections):
        raise ValueError(f"Missing sections in {page.source}: {set(page.sections) - found_sections}")
    notice = ""
    if page.slug in ("trajectory-matching", "distribution-matching", "fine-tuning"):
        notice = (
            "> Benchmark results on this page were measured with "
            "[FastGen v0.1.0](https://github.com/NVlabs/FastGen/releases/tag/v0.1.0).\n\n"
        )
    return (
        f"---\ntitle: {json.dumps(page.title)}\n---\n\n"
        "{/* Generated by fern/scripts/build_docs.py. Edit the source README instead. */}\n\n"
        f"[View source on GitHub]({source_url(source, ref)})\n\n"
        f"{notice}" + "\n".join(result).strip() + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", default="main", help="Public GitHub branch, tag, or SHA used in source links")
    args = parser.parse_args()
    if not args.source_ref or any(ord(char) < 32 for char in args.source_ref):
        parser.error("--source-ref must be a nonempty Git ref without control characters")
    generated = REPO / "fern" / "docs" / "generated"
    if generated.is_symlink():
        raise ValueError("fern/docs/generated must not be a symlink")
    # Render before writing so a bad link cannot leave a partially updated build.
    pages = {f"{page.slug}.mdx": render(page, args.source_ref) for page in PAGES}
    generated.mkdir(parents=True, exist_ok=True)
    for name, content in pages.items():
        target = generated / name
        if target.is_symlink():
            raise ValueError(f"Generated page must not be a symlink: {target}")
        target.write_text(content)
    print(f"Built {len(pages)} Fern pages from public READMEs (source ref: {args.source_ref}).")


if __name__ == "__main__":
    main()
