# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test Fern page generation and path validation without FastGen dependencies.

Run from the checkout root with ``python3 -B tests/test_fern.py``.
"""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "fern" / "scripts"
CHECKER = SCRIPTS / "check-docs-paths"
_SPEC = importlib.util.spec_from_file_location("fern_build_docs", SCRIPTS / "build_docs.py")
build_docs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_docs)


class DocsBuildTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="fern-build-")
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        repo_patch = patch.object(build_docs, "REPO", self.repo)
        repo_patch.start()
        self.addCleanup(repo_patch.stop)

    def write(self, path, content):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def build(self, *pages, ref="v1.2.3"):
        if not pages:
            pages = (build_docs.Page("README.md", "guide", "Guide"),)
        with (
            patch.object(build_docs, "PAGES", pages),
            patch.object(sys, "argv", ["build_docs.py", "--source-ref", ref]),
            redirect_stdout(io.StringIO()),
        ):
            build_docs.main()
        return self.repo / "fern" / "docs" / "generated"

    def test_selected_sections_preserve_code_examples(self):
        example = "```markdown\n# Example\n## Other\n[Example](missing.md)\n```"
        self.write(
            "README.md",
            "# FastGen\n\nIntro text.\n\n## Setup\nInstall instructions.\n\n"
            + example
            + "\n\n## Other\nUnselected text.\n\n## Quick Start\nRun instructions.\n",
        )
        output = self.build(build_docs.Page("README.md", "guide", "Guide", ("Setup", "Quick Start")))
        content = (output / "guide.mdx").read_text()
        self.assertTrue(content.startswith('---\ntitle: "Guide"\n---\n'))
        self.assertIn("## Setup\nInstall instructions.", content)
        self.assertIn("## Quick Start\nRun instructions.", content)
        self.assertIn(example, content)
        for omitted in ("# FastGen\n", "Intro text.", "Unselected text."):
            self.assertNotIn(omitted, content)
        self.assertIn("https://github.com/NVlabs/FastGen/blob/v1.2.3/README.md", content)

    def test_links_resolve_to_docs_or_versioned_sources(self):
        self.write("scripts/README.md", "# Inference\n")
        self.write("examples/train demo.py", "# Example\n")
        self.write("assets/plot one.svg", "<svg />\n")
        github = "https://github.com/NVlabs/FastGen"
        cases = (
            (
                '[Guide](scripts/README.md?view=all#usage "Read more")',
                '[Guide](/fastgen/inference?view=all#usage "Read more")',
            ),
            ("[Overview](README.md)", "[Overview](/fastgen)"),
            ("[Setup](README.md#setup)", f"[Setup]({github}/blob/docs%2Fv1.2.3/README.md#setup)"),
            (
                "[Code](examples/train%20demo.py?raw=1#L2)",
                f"[Code]({github}/blob/docs%2Fv1.2.3/examples/train%20demo.py?raw=1#L2)",
            ),
            ("[Examples](examples)", f"[Examples]({github}/tree/docs%2Fv1.2.3/examples)"),
            (
                "![Plot](assets/plot%20one.svg)",
                "![Plot](https://raw.githubusercontent.com/NVlabs/FastGen/docs%2Fv1.2.3/assets/plot%20one.svg)",
            ),
            ("[External](https://example.com/guide)", "[External](https://example.com/guide)"),
            ("[Heading](#setup)", "[Heading](#setup)"),
            ("[Training](/fastgen/training)", "[Training](/fastgen/training)"),
        )
        self.write("README.md", "\n\n".join(source for source, _ in cases))
        content = (self.build(ref="docs/v1.2.3") / "guide.mdx").read_text()
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertIn(expected, content)

    def test_missing_and_escaping_links_are_rejected(self):
        (self.repo / "escape").symlink_to(self.repo.parent, target_is_directory=True)
        cases = (
            ("missing.md", "Missing link destination"),
            ("../private.mdx", "Link escapes the repository"),
            ("escape/private.mdx", "Link escapes the repository"),
        )
        for target, message in cases:
            with self.subTest(target=target):
                self.write("README.md", f"[Example]({target})\n")
                with self.assertRaisesRegex(ValueError, message):
                    self.build()
                self.assertFalse((self.repo / "fern" / "docs" / "generated").exists())

    def test_missing_section_does_not_replace_existing_pages(self):
        original = self.write("fern/docs/generated/guide.mdx", "Existing guide.\n")
        self.write("README.md", "# Guide\nUpdated guide.\n")
        self.write("scripts/README.md", "# Inference\n")
        with self.assertRaisesRegex(ValueError, "Missing sections"):
            self.build(
                build_docs.Page("README.md", "guide", "Guide"),
                build_docs.Page("scripts/README.md", "inference", "Inference", ("Usage",)),
            )
        self.assertEqual(original.read_text(), "Existing guide.\n")
        self.assertFalse((original.parent / "inference.mdx").exists())

    def test_output_symlinks_are_rejected_without_overwriting_their_targets(self):
        self.write("README.md", "# Guide\nGuide content.\n")
        protected = self.write("protected/guide.mdx", "Keep this content.\n")
        generated = self.repo / "fern" / "docs" / "generated"
        generated.parent.mkdir(parents=True)
        generated.symlink_to(protected.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.build()
        self.assertEqual(protected.read_text(), "Keep this content.\n")
        generated.unlink()
        generated.mkdir()
        (generated / "guide.mdx").symlink_to(protected)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.build()
        self.assertEqual(protected.read_text(), "Keep this content.\n")


class DocsPathTests(unittest.TestCase):
    def check_document(self, text: str, *, suffix: str = ".mdx", allowed: str | None = None):
        with TemporaryDirectory(prefix="fern-paths-") as directory:
            path = Path(directory) / f"page{suffix}"
            path.write_text(text)
            command = [sys.executable, str(CHECKER)]
            if allowed is not None:
                command.extend(["--allow", allowed])
            command.append(directory)
            return subprocess.run(command, capture_output=True, text=True, timeout=10)

    def check_value(self, value: str, *, frontmatter: bool, allowed: str | None = None):
        if frontmatter:
            return self.check_document(f"---\nimage: {value}\n---\n", allowed=allowed)
        return self.check_document(f"path: {value}\n", suffix=".yml", allowed=allowed)

    def test_yaml_escapes_are_rejected(self):
        values = (
            r'"./\u002e\u002e/private.mdx"',
            r'"./\x2e\x2e/private.mdx"',
            r'"./\U0000002e\U0000002e/private.mdx"',
            r'".\u002f..\u002fprivate.mdx"',
            r'"./%2e\u002e/private.mdx"',
            r'"\u002fprivate.mdx"',
            r'"fi\x6ce:///private.mdx"',
            r'"./guide\".mdx"',
        )
        for frontmatter in (False, True):
            for value in values:
                with self.subTest(frontmatter=frontmatter, value=value):
                    result = self.check_value(value, frontmatter=frontmatter)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn("escaped double-quoted", result.stdout)

    def test_literal_paths_and_trailing_comments_are_accepted(self):
        values = (
            "./overview.mdx",
            '"./overview.mdx"',
            "'./overview.mdx'",
            r'"./overview.mdx" # Example escape: \u002e\u002e',
            r"'./\u002e\u002e/private.mdx'",  # Backslashes are literal in YAML single quotes.
            "'./operator''s-guide.mdx'",
        )
        for frontmatter in (False, True):
            for value in values:
                with self.subTest(frontmatter=frontmatter, value=value):
                    result = self.check_value(value, frontmatter=frontmatter)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_literal_traversal_requires_an_explicit_allow(self):
        for frontmatter in (False, True):
            with self.subTest(frontmatter=frontmatter):
                result = self.check_value('"../shared.mdx"', frontmatter=frontmatter)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                result = self.check_value('"../shared.mdx"', frontmatter=frontmatter, allowed="../shared.mdx")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_allow_does_not_enable_yaml_escapes(self):
        value = r'"./\u002e\u002e/private.mdx"'
        for frontmatter in (False, True):
            with self.subTest(frontmatter=frontmatter):
                result = self.check_value(value, frontmatter=frontmatter, allowed=value[1:-1])
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("escaped double-quoted", result.stdout)

    def test_unsupported_frontmatter_is_rejected(self):
        values = (
            '"image": /proc/self/environ',
            "'image': /proc/self/environ",
            "image: |\n  /proc/self/environ",
            "image: >-\n  /proc/self/environ",
            "image:\n  /proc/self/environ",
            'image: "\n  /proc/self/environ"',
            "image: !!str /proc/self/environ",
            "image: &secret /proc/self/environ",
            "image: *secret",
            "metadata: {image: /proc/self/environ}",
            "image: [/proc/self/environ]",
            "? image\n: /proc/self/environ",
        )
        for value in values:
            with self.subTest(value=value):
                result = self.check_document(f"---\n{value}\n---\n")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("[frontmatter]", result.stdout)

    def test_bare_and_namespaced_frontmatter_paths_are_checked(self):
        for key in ("image", "og:image", "twitter:image"):
            with self.subTest(key=key):
                result = self.check_document(f"---\n{key}: /proc/self/environ\n---\n")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("absolute path", result.stdout)

    def test_supported_frontmatter_and_markdown_body_are_accepted(self):
        values = (
            "",
            "# Comment\ntitle: FastGen\ndescription: Training and inference.",
            'title: "FastGen" # Comment\nimage: ./assets/logo.svg',
            "og:image: https://example.com/logo.svg\ntwitter:image: ./assets/logo.svg",
            "logo:\n  path: ./assets/logo.svg",
        )
        body = '# Guide\n\n```yaml\n"image": /proc/self/environ\n```\n'
        for value in values:
            with self.subTest(value=value):
                result = self.check_document(f"---\n{value}\n---\n{body}")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.check_document(body)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_frontmatter_errors_use_markdown_line_numbers(self):
        result = self.check_document('---\n# Comment\n"image": /proc/self/environ\n---\n')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("page.mdx:3: [frontmatter]", result.stdout)

    def test_unterminated_frontmatter_is_rejected(self):
        for text in ("---", "---\ntitle: FastGen\n", "---\nimage: /proc/self/environ\n"):
            with self.subTest(text=text):
                result = self.check_document(text)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("unterminated frontmatter", result.stdout)

    def test_component_and_image_paths(self):
        cases = (
            ("./assets/example.svg", 0),
            ("https://example.com/example.svg", 0),
            ("../private.mdx", 1),
            ("%2e%2e/private.mdx", 1),
            ("/proc/self/environ", 1),
            ("file:///proc/self/environ", 1),
        )
        for template in ('<Code src="{}" />', "![Example]({})"):
            for target, expected in cases:
                with self.subTest(template=template, target=target):
                    result = self.check_document(template.format(target))
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_dynamic_component_paths_are_rejected(self):
        result = self.check_document("<Code src={filename} />")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("dynamic src", result.stdout)

    def test_file_and_directory_symlinks_are_rejected(self):
        for is_directory in (False, True):
            with self.subTest(is_directory=is_directory), TemporaryDirectory(prefix="fern-paths-") as directory:
                root = Path(directory)
                docs = root / "docs"
                docs.mkdir()
                target = root / "target"
                if is_directory:
                    target.mkdir()
                else:
                    target.write_text("Example content.\n")
                (docs / "linked").symlink_to(target, target_is_directory=is_directory)
                result = subprocess.run(
                    [sys.executable, str(CHECKER), str(docs)], capture_output=True, text=True, timeout=10
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("[symlink]", result.stdout)


if __name__ == "__main__":
    unittest.main()
