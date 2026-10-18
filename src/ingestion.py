"""Local PDF and plain-text ingestion with explicit line numbering. No cloud OCR.

Both input modes end in the SAME pipeline:

    uploaded .pdf / .txt  ─┐
                           ├─► ingest_input() ─► _finalize() ─► guards ─► number_lines()
    pasted text           ─┘                                              │
                                                                          ▼
                                                              ExtractedDocument.lines
                                                              (Line 1, Line 2, ... N)

Line numbers are 1-based, continuous across the whole document (page breaks do
not reset them) and include blank lines, so "Line 12" is always the 12th line of
the text the user supplied. Every DocLine also remembers which page it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

# Hard limit to prevent OOM on very large documents.
MAX_DOCUMENT_CHARS = 300_000

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

# Strings that suggest prompt-injection attempts inside a document.
_INJECTION_SIGNALS = re.compile(
    r"ignore\s+(all\s+)?previous\s+instructions"
    r"|system\s+prompt"
    r"|you\s+are\s+now\s+a"
    r"|override\s+(all\s+)?instructions"
    r"|disregard\s+(your\s+)?instructions",
    re.IGNORECASE,
)


# ─────────────────────────── data model ───────────────────────────


@dataclass(frozen=True)
class DocLine:
    """One line of the document, with its explicit 1-based line number."""

    number: int
    text: str
    page: int = 1

    @property
    def label(self) -> str:
        return f"Line {self.number}"


@dataclass
class ExtractedDocument:
    name: str
    pages: list[str]
    source_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    lines: list[DocLine] = field(default_factory=list)
    source_kind: str = "file"  # "file" (upload / path) or "paste"

    @property
    def text(self) -> str:
        """Page text joined as before (kept for backwards compatibility)."""
        return "\n\n".join(self.pages)

    @property
    def line_text(self) -> str:
        """All lines joined with single newlines; offsets map 1:1 to line numbers."""
        return "\n".join(dl.text for dl in self.lines)

    @property
    def numbered_text(self) -> str:
        """The whole document as ``Line N: text`` rows."""
        return "\n".join(f"{dl.label}: {dl.text}" for dl in self.lines)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def get_line(self, number: int) -> str:
        """Text of a 1-based line number, or '' if it does not exist."""
        if 1 <= number <= len(self.lines):
            return self.lines[number - 1].text
        return ""


# ─────────────────────────── line numbering ───────────────────────────

_NEWLINES = re.compile(r"\r\n|\r|\n")


def number_lines(pages: list[str]) -> list[DocLine]:
    """Parse pages line-by-line and assign continuous line numbers (Line 1, Line 2, ...)."""
    lines: list[DocLine] = []
    counter = 0
    for page_no, page in enumerate(pages, start=1):
        if not page.strip():
            continue
        for raw in _NEWLINES.split(page.rstrip("\r\n")):
            counter += 1
            lines.append(DocLine(number=counter, text=raw.rstrip(), page=page_no))
    return lines


def number_text(text: str) -> list[DocLine]:
    """Number a plain string (single page)."""
    return number_lines([text or ""])


def compress_ranges(numbers) -> list[tuple[int, int]]:
    """[12, 13, 14, 20] -> [(12, 14), (20, 20)]"""
    out: list[tuple[int, int]] = []
    for n in sorted(set(int(x) for x in numbers)):
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def format_line_ref(numbers) -> str:
    """[12] -> 'Line 12'; [12, 13, 14, 20] -> 'Lines 12-14, 20'; [] -> ''."""
    ranges = compress_ranges(numbers)
    if not ranges:
        return ""
    parts = [str(a) if a == b else f"{a}-{b}" for a, b in ranges]
    single = len(ranges) == 1 and ranges[0][0] == ranges[0][1]
    return ("Line " if single else "Lines ") + ", ".join(parts)


# ─────────────────────────── the single pipeline ───────────────────────────


def ingest_input(
    *,
    data: bytes | None = None,
    filename: str | None = None,
    text: str | None = None,
    name: str | None = None,
) -> ExtractedDocument:
    """ONE entry point for both input modes.

    * file upload  -> ingest_input(data=bytes, filename="contract.pdf")
    * pasted text  -> ingest_input(text="...", name="My contract")

    Either way the result has been guarded (size, control chars, injection scan)
    and line-numbered by the same code.
    """
    if data is not None:
        fname = filename or "upload"
        suffix = Path(fname).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix or '(none)'}. Use a .pdf or .txt file.")
        return extract_bytes(data, fname)
    if text is not None and text.strip():
        return extract_text(text, name=(name or "").strip() or "Pasted text", source_kind="paste")
    raise ValueError("No input provided. Upload a .pdf / .txt file or paste the document text.")


def extract_text(text: str, name: str = "Pasted text", source_kind: str = "paste") -> ExtractedDocument:
    """Plain text (pasted or decoded from a .txt file) -> numbered document."""
    return _finalize(
        ExtractedDocument(name=name, pages=_split_plain_pages(text), source_kind=source_kind)
    )


def extract_file(path: str | Path) -> ExtractedDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".txt", ".md"}:
        return extract_text_file(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use PDF or .txt")


def extract_bytes(data: bytes, filename: str) -> ExtractedDocument:
    name = filename or "upload"
    if name.lower().endswith(".pdf"):
        return extract_pdf_bytes(data, name)
    # .txt, .md and unlabeled bytes are treated as UTF-8 text.
    text = data.decode("utf-8-sig", errors="replace")
    return extract_text(text, name=name, source_kind="file")


def extract_text_file(path: Path) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    doc = extract_text(text, name=path.name, source_kind="file")
    doc.source_path = str(path)
    return doc


def extract_pdf(path: Path) -> ExtractedDocument:
    return extract_pdf_bytes(path.read_bytes(), path.name, source_path=str(path))


def extract_pdf_bytes(
    data: bytes,
    name: str,
    source_path: str | None = None,
) -> ExtractedDocument:
    warnings: list[str] = []
    pages = _extract_with_pypdf(data, warnings)
    if _too_sparse(pages):
        plumber_pages = _extract_with_pdfplumber(data, warnings)
        if not _too_sparse(plumber_pages):
            pages = plumber_pages
        elif plumber_pages and (not pages or len("".join(plumber_pages)) > len("".join(pages))):
            pages = plumber_pages
    if not any(p.strip() for p in pages):
        warnings.append(
            "No extractable text found. This PDF may be image-only / scanned. "
            "Cloud OCR is disabled. To process it locally: export to text using "
            "Tesseract OCR (tesseract file.pdf out -l eng) or paste the text directly."
        )
        pages = [""]
    doc = ExtractedDocument(
        name=name,
        pages=pages,
        source_path=source_path,
        warnings=warnings,
        source_kind="file",
    )
    return _finalize(doc)


def _extract_with_pypdf(data: bytes, warnings: list[str]) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        warnings.append("pypdf is not installed.")
        return []
    try:
        reader = PdfReader(BytesIO(data))
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 — keep triage running
        warnings.append(f"pypdf failed: {exc}")
        return []


def _extract_with_pdfplumber(data: bytes, warnings: list[str]) -> list[str]:
    try:
        import pdfplumber
    except ImportError:
        warnings.append("pdfplumber is not installed.")
        return []
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            return [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pdfplumber failed: {exc}")
        return []


def _too_sparse(pages: list[str]) -> bool:
    joined = "".join(pages).strip()
    return len(joined) < 40


def _split_plain_pages(text: str) -> list[str]:
    """Split on form-feed page breaks. Leading blank lines are preserved so that
    Line 1 is the first line the user actually supplied."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = text.split("\f")
    pages = [p.rstrip("\n") for p in parts if p.strip() or len(parts) == 1]
    return pages or [""]


# ─────────────────────────── guards + numbering ───────────────────────────


def _finalize(doc: ExtractedDocument) -> ExtractedDocument:
    """Shared tail of the pipeline: safety guards, then explicit line numbers."""
    doc = _apply_guards(doc)
    doc.lines = number_lines(doc.pages)
    return doc


def _apply_guards(doc: ExtractedDocument) -> ExtractedDocument:
    """
    Apply safety guards to an already-extracted document:
    1. Truncate oversized documents to MAX_DOCUMENT_CHARS.
    2. Strip null bytes / binary control characters from page text.
    3. Detect prompt-injection signals and attach a warning.
    """
    # --- sanitize control chars (keeps \n, \r and \t) ---
    cleaned_pages = [
        re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", page)
        for page in doc.pages
    ]

    # --- size guard ---
    total = sum(len(p) for p in cleaned_pages)
    if total > MAX_DOCUMENT_CHARS:
        kept: list[str] = []
        budget = MAX_DOCUMENT_CHARS
        for page in cleaned_pages:
            if budget <= 0:
                break
            kept.append(page[:budget])
            budget -= len(page)
        cleaned_pages = kept
        doc.warnings.append(
            f"Document truncated to {MAX_DOCUMENT_CHARS:,} characters to prevent memory issues. "
            "The full document was not analysed. A qualified lawyer should review the complete file."
        )

    # --- injection detection ---
    full_text = "\n".join(cleaned_pages)
    if _INJECTION_SIGNALS.search(full_text):
        doc.warnings.append(
            "Possible prompt-injection text detected inside the document. "
            "The system will ignore any embedded instructions."
        )

    doc.pages[:] = cleaned_pages
    return doc
