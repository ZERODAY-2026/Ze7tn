"""Line-aware overlap windows so retrieval can cite exact source lines.

Chunks are built from WHOLE document lines, never from arbitrary character
offsets, so every chunk carries a precise ``start_line`` / ``end_line`` range.
Blank lines are not placed in chunks (they carry no content), but they still
occupy line numbers, so ranges stay faithful to the original text.

A single line longer than ``chunk_size`` (e.g. a pasted paragraph with no line
breaks) is split into several segments that all keep the SAME line number.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion import ExtractedDocument, format_line_ref


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str  # plain text, one row per line segment (what retrieval indexes)
    page: int  # page of the first line
    end_page: int  # page of the last line
    start_line: int
    end_line: int
    document_name: str
    lines: tuple[tuple[int, str], ...] = ()  # ((line_no, text), ...)

    @property
    def numbered_text(self) -> str:
        """Rows labelled exactly as the model must cite them: 'Line N: text'."""
        return "\n".join(f"Line {n}: {t}" for n, t in self.lines)

    @property
    def line_ref(self) -> str:
        return format_line_ref(range(self.start_line, self.end_line + 1))

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "numbered_text": self.numbered_text,
            "page": self.page,
            "end_page": self.end_page,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "line_ref": self.line_ref,
            "document_name": self.document_name,
        }


def split_long_line(text: str, size: int) -> list[str]:
    """Split an over-long line at word boundaries into segments of <= size chars."""
    if len(text) <= size:
        return [text]
    out: list[str] = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size + 1)
        if cut <= 0:
            cut = size
        out.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        out.append(text)
    return out


def chunk_document(
    doc: ExtractedDocument,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Group consecutive non-blank lines into ~chunk_size-character chunks.

    ``overlap`` is measured in characters; the tail lines of a chunk that fit
    inside it are repeated at the start of the next chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    # (line_no, segment_text, page)
    segments: list[tuple[int, str, int]] = []
    for dl in doc.lines:
        stripped = dl.text.strip()
        if not stripped:
            continue
        for seg in split_long_line(stripped, chunk_size):
            segments.append((dl.number, seg, dl.page))

    chunks: list[Chunk] = []
    counter = 1
    i, n = 0, len(segments)
    while i < n:
        size, j = 0, i
        while j < n and (j == i or size + len(segments[j][1]) + 1 <= chunk_size):
            size += len(segments[j][1]) + 1
            j += 1
        window = segments[i:j]
        chunks.append(
            Chunk(
                id=f"C{counter}",
                text="\n".join(seg for _, seg, _ in window),
                page=window[0][2],
                end_page=window[-1][2],
                start_line=window[0][0],
                end_line=window[-1][0],
                document_name=doc.name,
                lines=tuple((ln, seg) for ln, seg, _ in window),
            )
        )
        counter += 1
        if j >= n:
            break
        # step back so ~overlap characters are repeated; always make progress
        k, back = j, 0
        while k - 1 > i and back + len(segments[k - 1][1]) <= overlap:
            k -= 1
            back += len(segments[k][1])
        i = k
    return chunks
