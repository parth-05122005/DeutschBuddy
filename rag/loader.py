"""Loading documents (Markdown, text, PDF) and splitting them into chunks."""

import hashlib
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str            # file name shown to the user as a citation
    category: str          # grammar / vocabulary / culture / uploads
    section: str = ""      # heading path, e.g. "The Four German Cases > Prepositions"
    page: int | None = None
    id: str = field(init=False)

    def __post_init__(self):
        # Content-based id: re-indexing the same file never creates duplicates
        digest = hashlib.sha1(f"{self.source}|{self.page}|{self.text}".encode("utf-8")).hexdigest()
        self.id = digest[:20]

    @property
    def metadata(self) -> dict:
        meta = {"source": self.source, "category": self.category, "section": self.section}
        if self.page is not None:
            meta["page"] = self.page
        return meta


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into pieces of about `size` characters.

    Splits on paragraph breaks first, then sentences, so a chunk rarely
    ends in the middle of a sentence. Neighbouring chunks share `overlap`
    characters so that context at the boundary is not lost.
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    # Units: paragraphs, and sentences inside paragraphs that are too long
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
        else:
            units.extend(s for s in re.split(r"(?<=[.!?])\s+|\n", para) if s.strip())

    chunks: list[str] = []
    current = ""
    for unit in units:
        while len(unit) > size:  # a single sentence longer than a chunk: hard cut
            chunks.append(unit[:size])
            unit = unit[size - overlap:]
        if current and len(current) + len(unit) + 2 > size:
            chunks.append(current)
            current = current[-overlap:].lstrip() + "\n\n" + unit if overlap else unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def chunk_markdown(text: str, source: str, category: str, size: int, overlap: int) -> list[Chunk]:
    """Chunk Markdown section by section, keeping the heading path as context."""
    chunks: list[Chunk] = []
    headings: list[str] = []
    buffer: list[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if not body:
            return
        section = " > ".join(headings)
        for piece in split_text(body, size, overlap):
            # The heading path is prepended so that each chunk is understandable on its own
            chunks.append(Chunk(text=f"[{section}]\n{piece}" if section else piece,
                                source=source, category=category, section=section))

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            headings = headings[: level - 1] + [match.group(2).strip()]
        else:
            buffer.append(line)
    flush()
    return chunks


def chunk_pdf(data: bytes, source: str, category: str, size: int, overlap: int) -> list[Chunk]:
    reader = PdfReader(BytesIO(data))
    chunks: list[Chunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for piece in split_text(text, size, overlap):
            chunks.append(Chunk(text=piece, source=source, category=category, page=page_no))
    return chunks


def load_file(name: str, data: bytes, category: str, size: int, overlap: int) -> list[Chunk]:
    """Turn one file (as raw bytes) into chunks, based on its extension."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return chunk_pdf(data, name, category, size, overlap)
    if suffix in (".md", ".markdown", ".txt"):
        return chunk_markdown(data.decode("utf-8", errors="ignore"), name, category, size, overlap)
    raise ValueError(f"Unsupported file type: {suffix}. Use PDF, Markdown or TXT.")


def load_directory(root: Path, size: int, overlap: int) -> list[Chunk]:
    """Load the built-in knowledge base. The sub-folder name becomes the category."""
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".md", ".txt", ".pdf"):
            category = path.parent.name if path.parent != root else "general"
            chunks.extend(load_file(path.name, path.read_bytes(), category, size, overlap))
    return chunks
