"""PDF loading and text chunking utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import List
from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentChunk:
    """A single text chunk extracted from a PDF with source metadata."""

    chunk_id: str
    text: str
    source: str
    page: int


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph and sentence boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Try to break at a sentence or word boundary near the end of the window
            boundary = max(
                cleaned.rfind(". ", start + chunk_size // 2, end),
                cleaned.rfind(" ", start + chunk_size // 2, end),
            )
            if boundary > start:
                end = boundary + (1 if cleaned[boundary] == "." else 0)

        segment = cleaned[start:end].strip()
        if segment:
            chunks.append(segment)

        if end >= text_len:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def load_pdf_chunks(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[DocumentChunk]:
    """Read a PDF file and return a list of overlapping DocumentChunk objects."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    all_chunks: List[DocumentChunk] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        page_chunks = split_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_idx, chunk_text in enumerate(page_chunks):
            all_chunks.append(
                DocumentChunk(
                    chunk_id=f"{path.stem}_p{page_idx}_c{chunk_idx}",
                    text=chunk_text,
                    source=path.name,
                    page=page_idx,
                )
            )

    return all_chunks

