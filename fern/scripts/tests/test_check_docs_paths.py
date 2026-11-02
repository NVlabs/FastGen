# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for escaped paths in Fern YAML and MDX frontmatter."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

CHECKER = Path(__file__).resolve().parents[1] / "check-docs-paths"


class DocsPathTests(unittest.TestCase):
    def check_value(self, value: str, *, frontmatter: bool, allowed: str | None = None):
        with TemporaryDirectory(prefix="fern-paths-") as directory:
            if frontmatter:
                path = Path(directory) / "page.mdx"
                path.write_text(f"---\nimage: {value}\n---\n")
            else:
                path = Path(directory) / "index.yml"
                path.write_text(f"path: {value}\n")
            command = [sys.executable, str(CHECKER)]
            if allowed is not None:
                command.extend(["--allow", allowed])
            command.append(directory)
            return subprocess.run(command, capture_output=True, text=True, timeout=10)

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


if __name__ == "__main__":
    unittest.main()
