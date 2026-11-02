# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for unsafe paths and unsupported YAML in Fern docs."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

CHECKER = Path(__file__).resolve().parents[1] / "check-docs-paths"


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


if __name__ == "__main__":
    unittest.main()
