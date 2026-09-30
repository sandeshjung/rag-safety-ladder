from __future__ import annotations

from pathlib import Path

from refuses_to_lie.corpus import Chunk, chunk_document, load_all_pages, load_body_pages
from refuses_to_lie.intake import parse_cover_sheet


def employer_doc_id(path: Path) -> str:
    """The id an employer policy is indexed under: its cover-sheet reference.

    Public because the provenance register must derive ids exactly the way
    the loader does -- a mismatch would make the register reject a real
    document and quietly delete it from every provenance-checked answer.
    """
    cover_sheet = parse_cover_sheet(path)
    return cover_sheet.doc_ref or path.stem


def statutory_doc_id(path: Path) -> str:
    """Statutory pages have no cover sheet, so the filename is the id."""
    return path.stem


def load_employer_chunks(employer_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf_path in sorted(employer_dir.glob("*.pdf")):
        doc_id = employer_doc_id(pdf_path)
        chunks.extend(chunk_document(doc_id, load_body_pages(pdf_path)))
    return chunks


def load_statutory_chunks(statutory_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf_path in sorted(statutory_dir.glob("*.pdf")):
        chunks.extend(chunk_document(statutory_doc_id(pdf_path), load_all_pages(pdf_path)))
    return chunks


def load_corpus_chunks(employer_dir: Path, statutory_dir: Path) -> list[Chunk]:
    return load_employer_chunks(employer_dir) + load_statutory_chunks(statutory_dir)


def load_injected_chunks(injected_dir: Path) -> list[Chunk]:
    """Load the adversarial corpus.

    These have no cover sheet, so the filename stem is the doc_id and every
    page is body -- which is also what makes an injected document visually
    indistinguishable from a real one once it is chunked.
    """
    chunks: list[Chunk] = []
    for pdf_path in sorted(injected_dir.glob("*.pdf")):
        chunks.extend(chunk_document(pdf_path.stem, load_all_pages(pdf_path)))
    return chunks
