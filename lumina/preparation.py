from __future__ import annotations

from pathlib import Path

from .common import atomic_output, ensure_directory_exists, token_calculator, turnIntoPureText
from .utils import get_ext_files


def convert_pdfs_to_markdown(pdf_dir: str | Path, markdown_dir: str | Path) -> list[str]:
    pdf_dir = Path(pdf_dir)
    markdown_dir = Path(markdown_dir)
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"pdf_dir does not exist or is not a directory: {pdf_dir}")
    pdfs = get_ext_files(pdf_dir, "pdf")
    if not pdfs:
        raise ValueError(f"no PDF files to convert in pdf_dir: {pdf_dir}")
    pending = [pdf for pdf in pdfs if _needs_conversion(pdf, markdown_dir)]
    if not pending:
        return []
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as exc:
        raise ImportError(
            f"marker is not installed, so PDFs cannot be converted into markdown_dir={markdown_dir}; "
            "install marker (see requirements/optional extras) or put Markdown files directly there"
        ) from exc

    converter = PdfConverter(artifact_dict=create_model_dict())
    ensure_directory_exists(markdown_dir)
    written = []
    for file_path in pending:
        rendered = converter(file_path)
        rendered_text, _, _images = text_from_rendered(rendered)
        if not isinstance(rendered_text, str) or not rendered_text.strip():
            raise RuntimeError(f'PDF conversion produced empty text: {file_path}')
        output_md_file = markdown_dir / Path(file_path).with_suffix(".md").name
        with atomic_output(output_md_file) as temp_md_file:
            temp_md_file.write_text(rendered_text, encoding="utf-8")
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
    pdf_dir = Path(domain_cfg["pdf_dir"])
    pdfs = get_ext_files(pdf_dir, "pdf") if pdf_dir.is_dir() else []
    if not mds and not pdfs:
        raise FileNotFoundError(
            f"no input documents: markdown_dir={domain_cfg['markdown_dir']} holds no Markdown "
            f"and pdf_dir={pdf_dir} holds no PDF"
        )
    if mds and not pdfs:
        if any(not _has_content(Path(md)) for md in mds):
            raise ValueError(f'empty Markdown input in {domain_cfg["markdown_dir"]}')
        return mds
    convert_pdfs_to_markdown(str(pdf_dir), domain_cfg["markdown_dir"])
    mds = markdown_files(domain_cfg["markdown_dir"])
    missing = sorted(
        stem
        for stem in (Path(pdf).stem for pdf in get_ext_files(pdf_dir, "pdf"))
        if not _has_content(Path(domain_cfg["markdown_dir"]) / f"{stem}.md")
    )
    if missing:
        raise RuntimeError(
            f"PDF conversion did not produce Markdown for: {', '.join(missing)} "
            f"in {domain_cfg['markdown_dir']}"
        )
    return mds


def _needs_conversion(pdf_path: str, markdown_dir: Path) -> bool:
    return not _has_content(markdown_dir / Path(pdf_path).with_suffix(".md").name)


def _has_content(markdown_path: Path) -> bool:
    return markdown_path.is_file() and bool(markdown_path.read_text(encoding='utf-8').strip())
