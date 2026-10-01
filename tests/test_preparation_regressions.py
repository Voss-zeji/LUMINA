from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from lumina import preparation


def fake_marker_modules(text_for: dict):
    """Stand-in marker modules: PdfConverter is the only mocked boundary."""

    class FakeConverter:
        def __init__(self, artifact_dict=None):
            self.calls = []

        def __call__(self, file_path):
            self.calls.append(str(file_path))
            return file_path

    def text_from_rendered(rendered):
        return text_for.get(Path(str(rendered)).stem, "converted body"), None, []

    converters = types.ModuleType("marker.converters.pdf")
    converters.PdfConverter = FakeConverter
    models = types.ModuleType("marker.models")
    models.create_model_dict = lambda *a, **k: {"stub": True}
    output = types.ModuleType("marker.output")
    output.text_from_rendered = text_from_rendered
    marker = types.ModuleType("marker")
    marker.converters = types.ModuleType("marker.converters")
    marker.converters.pdf = converters
    marker.models = models
    marker.output = output
    return {
        "marker": marker,
        "marker.converters": marker.converters,
        "marker.converters.pdf": converters,
        "marker.models": models,
        "marker.output": output,
    }


class PreparationRegressions(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.pdf_dir = self.root / "pdfs"
        self.md_dir = self.root / "mds"
        self.pdf_dir.mkdir()
        self.addCleanup(self._tmp.cleanup)

    def cfg(self):
        return {"pdf_dir": str(self.pdf_dir), "markdown_dir": str(self.md_dir)}

    def add_pdf(self, name):
        (self.pdf_dir / name).write_bytes(b"%PDF-1.4 stub")

    def add_md(self, name, body="existing body"):
        self.md_dir.mkdir(parents=True, exist_ok=True)
        (self.md_dir / name).write_text(body, encoding="utf-8")

    def test_resume_skips_completed_markdown(self):
        self.add_pdf("a.pdf")
        self.add_pdf("b.pdf")
        self.add_md("a.md")
        modules = fake_marker_modules({"b": "b body"})
        with patch.dict(sys.modules, modules):
            written = preparation.convert_pdfs_to_markdown(self.pdf_dir, self.md_dir)
        self.assertEqual([Path(p).name for p in written], ["b.md"])
        self.assertEqual((self.md_dir / "a.md").read_text(encoding="utf-8"), "existing body")
        self.assertEqual((self.md_dir / "b.md").read_text(encoding="utf-8"), "b body")

    def test_empty_markdown_is_reconverted(self):
        self.add_pdf("a.pdf")
        self.add_md("a.md", "")
        with patch.dict(sys.modules, fake_marker_modules({"a": "fresh body"})):
            written = preparation.convert_pdfs_to_markdown(self.pdf_dir, self.md_dir)
        self.assertEqual([Path(p).name for p in written], ["a.md"])
        self.assertEqual((self.md_dir / "a.md").read_text(encoding="utf-8"), "fresh body")

    def test_interrupted_write_publishes_nothing_and_stays_pending(self):
        self.add_pdf("a.pdf")
        self.add_md("a.md", "")
        modules = fake_marker_modules({"a": "new body"})
        with patch.dict(sys.modules, modules):
            with patch("lumina.common.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    preparation.convert_pdfs_to_markdown(self.pdf_dir, self.md_dir)
        self.assertEqual((self.md_dir / "a.md").read_text(encoding="utf-8"), "")
        self.assertEqual([p.name for p in self.md_dir.iterdir()], ["a.md"])
        with patch.dict(sys.modules, fake_marker_modules({"a": "new body"})):
            written = preparation.convert_pdfs_to_markdown(self.pdf_dir, self.md_dir)
        self.assertEqual([Path(p).name for p in written], ["a.md"])
        self.assertEqual((self.md_dir / "a.md").read_text(encoding="utf-8"), "new body")

    def test_no_marker_model_created_when_nothing_pending(self):
        self.add_pdf("a.pdf")
        self.add_md("a.md")
        modules = fake_marker_modules({})
        modules["marker.models"].create_model_dict = lambda *a, **k: self.fail(
            "model dict must not be created when no conversion is pending"
        )
        modules["marker.converters.pdf"].PdfConverter = lambda *a, **k: self.fail(
            "converter must not be built when no conversion is pending"
        )
        with patch.dict(sys.modules, modules):
            written = preparation.convert_pdfs_to_markdown(self.pdf_dir, self.md_dir)
        self.assertEqual(written, [])

    def test_missing_optional_dependency_is_actionable(self):
        self.add_pdf("a.pdf")
        with patch.dict(sys.modules, {"marker": None, "marker.converters.pdf": None}):
            with self.assertRaises(ImportError) as ctx:
                preparation.convert_pdfs_to_markdown(self.pdf_dir, self.md_dir)
        message = str(ctx.exception)
        self.assertIn("marker", message.lower())
        self.assertIn("markdown_dir", message)

    def test_missing_pdf_dir_fails(self):
        with self.assertRaises(FileNotFoundError):
            preparation.convert_pdfs_to_markdown(self.root / "absent", self.md_dir)

    def test_empty_pdf_dir_fails(self):
        with self.assertRaises(ValueError):
            preparation.convert_pdfs_to_markdown(self.pdf_dir, self.md_dir)

    def test_ensure_markdowns_recovers_partial_conversion(self):
        self.add_pdf("a.pdf")
        self.add_pdf("b.pdf")
        self.add_md("a.md")
        with patch.dict(sys.modules, fake_marker_modules({"b": "b body"})):
            mds = preparation.ensure_markdowns(self.cfg())
        self.assertEqual(sorted(Path(m).name for m in mds), ["a.md", "b.md"])

    def test_ensure_markdowns_preserves_markdown_only(self):
        self.add_md("only.md")
        modules = fake_marker_modules({})
        modules["marker.models"].create_model_dict = lambda *a, **k: self.fail(
            "markdown-only use must not initialize marker"
        )
        with patch.dict(sys.modules, modules):
            mds = preparation.ensure_markdowns(self.cfg())
        self.assertEqual([Path(m).name for m in mds], ["only.md"])

    def test_ensure_markdowns_without_any_input_fails(self):
        self.md_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            preparation.ensure_markdowns(self.cfg())

    def test_ensure_markdowns_reports_unmatched_stems_after_recovery(self):
        self.add_pdf("a.pdf")
        modules = fake_marker_modules({"a": ""})
        with patch.dict(sys.modules, modules):
            with self.assertRaises(RuntimeError) as ctx:
                preparation.ensure_markdowns(self.cfg())
        self.assertIn("a", str(ctx.exception))

    def test_whitespace_markdown_is_not_complete(self):
        self.add_pdf('a.pdf')
        self.add_md('a.md', '   \n')
        with patch.dict(sys.modules, fake_marker_modules({'a': 'real body'})):
            preparation.ensure_markdowns(self.cfg())
        self.assertEqual((self.md_dir / 'a.md').read_text(encoding='utf-8'), 'real body')

    def test_empty_markdown_only_input_fails(self):
        self.add_md('empty.md', '')
        with self.assertRaisesRegex(ValueError, 'empty Markdown'):
            preparation.ensure_markdowns(self.cfg())


if __name__ == "__main__":
    unittest.main()
