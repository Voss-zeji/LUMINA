from __future__ import annotations

from pathlib import Path

from .common import ensure_directory_exists, token_calculator, turnIntoPureText
from .utils import get_ext_files


def convert_pdfs_to_markdown(pdf_dir: str | Path, markdown_dir: str | Path) -> list[str]:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(artifact_dict=create_model_dict())
    markdown_dir = Path(markdown_dir)
    ensure_directory_exists(markdown_dir)
    written = []
    for file_path in get_ext_files(pdf_dir, "pdf"):
        rendered = converter(file_path)
        rendered_text, _, _images = text_from_rendered(rendered)
        output_md_file = markdown_dir / Path(file_path).with_suffix(".md").name
        with open(output_md_file, "w", encoding="utf-8") as md_file:
            md_file.write(rendered_text)
        written.append(str(output_md_file))
    return written


def markdown_files(markdown_dir: str | Path) -> list[str]:
    return get_ext_files(markdown_dir, "md")


def token_audit(markdown_paths: list[str]) -> list[dict]:
    rows = []
    for raw_markdown in markdown_paths:
        with open(raw_markdown, encoding="utf-8") as f:
            text = f.read()
        before_refs = turnIntoPureText(raw_markdown)
        tk1 = token_calculator(text)
        tk2 = token_calculator(before_refs)
        rows.append(
            {
                "markdown": raw_markdown,
                "chars_full": len(text),
                "chars_before_refs": len(before_refs),
                "tokens_full": tk1,
                "tokens_before_refs": tk2,
                "kept_token_ratio": (tk2 / tk1) if tk1 else None,
            }
        )
    return rows


def pure_text(markdown_path: str | Path) -> str:
    return turnIntoPureText(markdown_path)


def ensure_markdowns(domain_cfg: dict) -> list[str]:
    mds = markdown_files(domain_cfg["markdown_dir"])
    if mds:
        return mds
    convert_pdfs_to_markdown(domain_cfg["pdf_dir"], domain_cfg["markdown_dir"])
    return markdown_files(domain_cfg["markdown_dir"])
