from __future__ import annotations

from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import warnings

from .common import atomic_output, ensure_directory_exists, token_calculator, turnIntoPureText
from .utils import get_ext_files


def pdf_options(raw: dict | None = None) -> dict:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - {'backend', 'fallback'}:
        raise ValueError('pdf settings support only backend and fallback')
    backend = raw.get('backend', 'marker')
    fallback = raw.get('fallback', 'pypdf' if backend == 'marker' else 'none')
    if backend not in ('marker', 'pypdf') or fallback not in ('pypdf', 'none'):
        raise ValueError('pdf.backend must be marker/pypdf; pdf.fallback must be pypdf/none')
    if backend == 'pypdf' and fallback != 'none':
        raise ValueError('pypdf is already the fallback; set pdf.fallback="none"')
    return dict(backend=backend, fallback=fallback)


def _marker_converter():
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(artifact_dict=create_model_dict())
    return lambda path: text_from_rendered(converter(str(path)))[0]


def _pypdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError('encrypted PDF: provide an unlocked input')
    pages = []
    for index, page in enumerate(reader.pages, 1):
        text = (page.extract_text(extraction_mode='layout') or '') if page.get_contents() is not None else ''
        if not text.strip():
            raise ValueError(f'page {index} has no extractable text; OCR/Marker is required')
        pages.append(f'## Page {index}\n\n{text.strip()}')
    return '\n\n'.join(pages)


def _nonempty(text) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError('PDF conversion produced empty text')
    return text


def convert_pdfs_to_markdown(pdf_dir: str | Path, markdown_dir: str | Path, *,
                             pdf_config: dict | None = None, records: dict | None = None,
                             cache: dict | None = None) -> list[str]:
    options = pdf_options(pdf_config)
    cache = {} if cache is None else cache
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
    ensure_directory_exists(markdown_dir)
    written = []
    for file_path in pending:
        backend, primary_error = options['backend'], None
        if backend == 'marker':
            try:
                if 'marker_error' in cache:
                    raise RuntimeError('Marker initialization failed earlier in this preparation pass')
                if 'marker' not in cache:
                    try:
                        cache['marker'] = _marker_converter()
                    except Exception as exc:
                        cache['marker_error'] = type(exc).__name__
                        raise
                rendered_text = _nonempty(cache['marker'](Path(file_path)))
            except Exception as exc:
                primary_error = cache.get('marker_error', type(exc).__name__)
                if options['fallback'] == 'none':
                    raise RuntimeError(f'Marker conversion failed for {Path(file_path).name} ({primary_error}); '
                                       'run install.py or enable pdf.fallback="pypdf"') from exc
                backend = 'pypdf'
        if backend == 'pypdf':
            try:
                rendered_text = _nonempty(_pypdf_text(Path(file_path)))
            except Exception as exc:
                raise RuntimeError(f'PDF conversion failed for {Path(file_path).name}; '
                                   f'Marker error={primary_error}; pypdf: {exc}') from exc
            warnings.warn(f'{Path(file_path).name}: using pypdf text extraction; review tables and reading order',
                          RuntimeWarning, stacklevel=2)
        package = 'marker-pdf' if backend == 'marker' else 'pypdf'
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = 'unknown'
        output_md_file = markdown_dir / Path(file_path).with_suffix(".md").name
        with atomic_output(output_md_file) as temp_md_file:
            temp_md_file.write_text(rendered_text, encoding="utf-8")
        if records is not None:
            records[Path(file_path).stem] = dict(backend=backend, version=package_version,
                                                fallback_used=primary_error is not None,
                                                primary_error=primary_error, text_only=backend == 'pypdf')
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
    options = {'pdf_config': domain_cfg['pdf']} if 'pdf' in domain_cfg else {}
    convert_pdfs_to_markdown(str(pdf_dir), domain_cfg["markdown_dir"], **options)
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
