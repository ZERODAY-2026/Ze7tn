"""
Structured clause extraction — pure regex, no LLM, no network.

Every field is explicitly marked FOUND / AMBIGUOUS / NOT_FOUND.
Nothing is invented. Silence in the document is reported as NOT_FOUND
("The provided document is silent on this matter."), never guessed.

Every excerpt carries the exact source line range it came from
(``Evidence.start_line`` .. ``Evidence.end_line``).
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from enum import Enum

from src.ingestion import DocLine, ExtractedDocument, format_line_ref, number_text
from src.prompts import SILENT_PHRASE

# ─────────────────────────── status enum ───────────────────────────


class FieldStatus(str, Enum):
    FOUND = "FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"


STATUS_EMOJI = {
    FieldStatus.FOUND: "🟢",
    FieldStatus.AMBIGUOUS: "🟡",
    FieldStatus.NOT_FOUND: "🔴",
}

STATUS_COLOR = {
    FieldStatus.FOUND: "#1a7a3c",
    FieldStatus.AMBIGUOUS: "#92680d",
    FieldStatus.NOT_FOUND: "#8b1a1a",
}


# ─────────────────────────── evidence + field result ───────────────────


@dataclass
class Evidence:
    """One excerpt plus the exact lines it was taken from."""

    text: str
    start_line: int
    end_line: int

    @property
    def lines(self) -> list[int]:
        return list(range(self.start_line, self.end_line + 1))

    @property
    def ref(self) -> str:
        return format_line_ref(self.lines)


@dataclass
class FieldResult:
    label: str
    status: FieldStatus
    evidence: list[Evidence] = field(default_factory=list)
    note: str = ""

    @property
    def emoji(self) -> str:
        return STATUS_EMOJI[self.status]

    @property
    def excerpts(self) -> list[str]:  # backwards-compatible view
        return [e.text for e in self.evidence]

    @property
    def line_numbers(self) -> list[int]:
        return sorted({n for e in self.evidence for n in e.lines})

    @property
    def line_ref(self) -> str:
        return format_line_ref(self.line_numbers)


# ─────────────────────────── clause report ─────────────────────────


@dataclass
class ClauseReport:
    document_name: str
    parties: FieldResult
    effective_dates: FieldResult
    notice_periods: FieldResult
    penalties: FieldResult
    termination: FieldResult
    jurisdiction: FieldResult
    confidentiality: FieldResult
    payment_terms: FieldResult

    def all_fields(self) -> list[FieldResult]:
        return [
            self.parties,
            self.effective_dates,
            self.notice_periods,
            self.penalties,
            self.termination,
            self.jurisdiction,
            self.confidentiality,
            self.payment_terms,
        ]

    @property
    def found_count(self) -> int:
        return sum(1 for f in self.all_fields() if f.status != FieldStatus.NOT_FOUND)

    @property
    def silent_count(self) -> int:
        return sum(1 for f in self.all_fields() if f.status == FieldStatus.NOT_FOUND)

    def risk_summary(self) -> str:
        """One-line plain-English summary, never a legal conclusion."""
        silent = [f.label for f in self.all_fields() if f.status == FieldStatus.NOT_FOUND]
        if not silent:
            return "Every tracked category has at least one cited passage."
        return (
            f"{SILENT_PHRASE} Categories with no matching lines: {', '.join(silent)}. "
            "A qualified lawyer should review the full document."
        )


# ─────────────────────────── line-aware matching ───────────────────────

_WINDOW = 220  # characters of context kept around each match (never beyond the cited lines)


class _LineIndex:
    """All document lines joined by '\\n'; maps any character offset to its line."""

    def __init__(self, lines: list[DocLine]):
        self.lines = lines
        self.starts: list[int] = []
        pos = 0
        parts: list[str] = []
        for dl in lines:
            self.starts.append(pos)
            parts.append(dl.text)
            pos += len(dl.text) + 1
        self.text = "\n".join(parts)

    def index_at(self, offset: int) -> int:
        return max(0, bisect_right(self.starts, offset) - 1)

    def line_end(self, idx: int) -> int:
        return self.starts[idx] + len(self.lines[idx].text)


def _evidence(idx: _LineIndex, m: re.Match) -> Evidence:
    """Excerpt for a regex match, cited to exactly the lines the match touches."""
    st = m.start()
    while st < m.end() - 1 and idx.text[st].isspace():
        st += 1
    en = max(m.end() - 1, st)
    while en > st and idx.text[en].isspace():
        en -= 1
    i0, i1 = idx.index_at(st), idx.index_at(en)
    block_s, block_e = idx.starts[i0], idx.line_end(i1)
    s = max(block_s, m.start() - _WINDOW // 2)
    e = min(block_e, m.end() + _WINDOW // 2)
    snippet = " ".join(idx.text[s:e].split())
    if s > block_s:
        snippet = "…" + snippet
    if e < block_e:
        snippet = snippet + "…"
    return Evidence(snippet, idx.lines[i0].number, idx.lines[i1].number)


def _find(idx: _LineIndex, pattern: str, flags: int = re.IGNORECASE) -> list[Evidence]:
    """Up to 3 deduplicated evidence items for a pattern."""
    seen: set[tuple[int, int, str]] = set()
    results: list[Evidence] = []
    for m in re.finditer(pattern, idx.text, flags):
        ev = _evidence(idx, m)
        key = (ev.start_line, ev.end_line, ev.text)
        if key not in seen:
            seen.add(key)
            results.append(ev)
        if len(results) >= 3:
            break
    return results


def _field(label: str, idx: _LineIndex, patterns: list[str]) -> FieldResult:
    """Run multiple patterns; combine & deduplicate results."""
    found: list[Evidence] = []
    for pat in patterns:
        found.extend(_find(idx, pat))
        if len(found) >= 3:
            break
    unique: list[Evidence] = []
    seen: set[tuple[int, int, str]] = set()
    for ev in found:
        key = (ev.start_line, ev.end_line, ev.text)
        if key not in seen:
            seen.add(key)
            unique.append(ev)
    unique = unique[:3]

    if not unique:
        return FieldResult(label=label, status=FieldStatus.NOT_FOUND, note=SILENT_PHRASE)
    return FieldResult(label=label, status=FieldStatus.FOUND, evidence=unique)


# ─────────────────────────── extractors ────────────────────────────

_PARTY_PATTERNS = [
    # "between [Party A] and [Party B]"
    r"between\s+([A-Z][^\n,;]{3,60})\s+(?:and|&)\s+([A-Z][^\n,;]{3,60})",
    # "Party A:" / "Employer:" / "Client:" labelled blocks
    r"(Employer|Employee|Client|Vendor|Contractor|Licensor|Licensee|Lessor|Lessee"
    r"|Disclosing Party|Receiving Party|Party\s+[AB1-9])\s*[:\-–]\s*([A-Z][^\n]{3,80})",
    # Common intro: "This Agreement is entered into by ABC Corp"
    r"entered\s+into\s+by\s+(?:and\s+between\s+)?([A-Z][^\n,;]{3,60})",
    # Section heading 'Parties' followed by names on the next line
    r"(?:^|\n)\s*Parties\s*\n([^\n]{5,120})",
    # Quoted company names with role: Acme Corp ("Disclosing Party")
    r'([A-Z][A-Za-z0-9\s\.\,]+(?:LLC|Ltd|Corp|Inc|LLP|Co|Limited|GmbH))\s*\(["\u201c]?'
    r"(?:Disclosing|Receiving|Employer|Employee|Client|Vendor|Licensor|Licensee)",
]

_DATE_PATTERNS = [
    r"\b(?:effective|dated?|executed?|signed?|as\s+of)\s+(?:the\s+)?"
    r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July"
    r"|August|September|October|November|December),?\s+\d{4})",
    r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b",
    r"\b(January|February|March|April|May|June|July|August|September|October"
    r"|November|December)\s+\d{1,2},?\s+\d{4}\b",
]

_NOTICE_PATTERNS = [
    r"(\d+)\s*(?:calendar\s+|business\s+)?days[''']?\s+(?:prior\s+|written\s+)?notice",
    r"notice\s+(?:of|period\s+of)\s+(\d+)\s*(?:calendar\s+|business\s+)?days",
    r"written\s+notice\s+(?:of\s+)?(?:at\s+least\s+)?(\d+)\s*days",
    r"notify(?:ing)?\s+.{0,40}within\s+(\d+)\s*(?:business\s+)?days",
]

_PENALTY_PATTERNS = [
    r"(?:liquidat(?:ed)?\s+damages?|penalty|penalties|fine|forfeit(?:ure)?)"
    r"\s+(?:of\s+|up\s+to\s+|equal\s+to\s+)?"
    r"(?:USD?\s*|INR\s*|Rs\.?\s*|£\s*|\$\s*)?[\d,]+",
    r"(?:\$|USD|INR|Rs\.?)\s*[\d,]+(?:\.\d{2})?\s*(?:per\s+day|per\s+breach|as\s+damages?)?",
    r"indemnif(?:y|ication|ies)\s+.{0,80}(?:damages?|losses?|costs?)",
    r"liable\s+for\s+.{0,60}(?:\$|USD|INR|Rs\.?|damages?)",
]

_TERMINATION_PATTERNS = [
    r"(?:terminat(?:e|ion|ed)|cancel(?:lation)?)\s+.{0,120}"
    r"(?:breach|default|notice|immediately|written|upon|without\s+cause)",
    r"(?:either\s+party|company|employer|client)\s+may\s+terminat",
    r"terminat(?:e|ion)\s+(?:for\s+)?(?:cause|convenience|breach|default)",
    r"this\s+(?:agreement|contract)\s+(?:shall\s+)?terminat",
    r"right\s+to\s+terminat",
]

_JURISDICTION_PATTERNS = [
    r"(?:governed\s+by|subject\s+to)\s+the\s+laws?\s+of\s+([A-Z][^\n,;.]{3,60})",
    r"jurisdiction\s+(?:of|shall\s+be)\s+([A-Z][^\n,;.]{3,60})",
    r"courts?\s+of\s+([A-Z][^\n,;.]{3,60})\s+(?:shall\s+have|will\s+have)",
    r"exclusive\s+jurisdiction\s+in\s+([A-Z][^\n,;.]{3,60})",
    r"(?:arbitration|dispute\s+resolution)\s+(?:shall\s+be\s+)?(?:in|at)\s+([A-Z][^\n,;.]{3,60})",
]

_CONFIDENTIALITY_PATTERNS = [
    r"confidential(?:ity)?\s+(?:information|obligation|clause|period)",
    r"(?:shall\s+)?(?:keep|maintain|hold)\s+.{0,40}confidential",
    r"non[‑\-]?disclosure",
    r"proprietary\s+information",
    r"trade\s+secrets?",
]

_PAYMENT_PATTERNS = [
    r"(?:payment|fee|compensation|salary|remuneration)\s+of\s+"
    r"(?:USD?\s*|INR\s*|Rs\.?\s*|£\s*|\$\s*)?[\d,]+",
    r"(?:pay(?:ment)?|invoice|bill)\s+(?:within|due\s+(?:in|within))\s+(\d+)\s+days",
    r"(?:monthly|annual|quarterly)\s+(?:fee|payment|subscription|retainer)\s+of",
    r"(?:\$|USD|INR|Rs\.?)\s*[\d,]+(?:\.\d{2})?\s*(?:per\s+month|per\s+year|per\s+annum)?",
]


# ─────────────────────────── public API ────────────────────────────


def extract_clauses(source, document_name: str = "document") -> ClauseReport:
    """
    Run regex extraction over the whole document and cite exact line numbers.

    ``source`` is an ExtractedDocument (preferred - uses its line numbering) or a
    plain string (numbered on the fly). Never invents content. Offline: no
    network, no LLM.
    """
    if isinstance(source, ExtractedDocument):
        lines, document_name = source.lines, source.name
    else:
        lines = number_text(str(source or ""))

    if not any(dl.text.strip() for dl in lines):
        def not_found(label: str) -> FieldResult:
            return FieldResult(label=label, status=FieldStatus.NOT_FOUND, note=SILENT_PHRASE)

        return ClauseReport(
            document_name=document_name,
            parties=not_found("Parties"),
            effective_dates=not_found("Effective Date / Key Dates"),
            notice_periods=not_found("Notice Periods"),
            penalties=not_found("Penalties / Damages"),
            termination=not_found("Termination Conditions"),
            jurisdiction=not_found("Governing Jurisdiction"),
            confidentiality=not_found("Confidentiality / NDA"),
            payment_terms=not_found("Payment Terms"),
        )

    idx = _LineIndex(lines)
    return ClauseReport(
        document_name=document_name,
        parties=_field("Parties", idx, _PARTY_PATTERNS),
        effective_dates=_field("Effective Date / Key Dates", idx, _DATE_PATTERNS),
        notice_periods=_field("Notice Periods", idx, _NOTICE_PATTERNS),
        penalties=_field("Penalties / Damages", idx, _PENALTY_PATTERNS),
        termination=_field("Termination Conditions", idx, _TERMINATION_PATTERNS),
        jurisdiction=_field("Governing Jurisdiction", idx, _JURISDICTION_PATTERNS),
        confidentiality=_field("Confidentiality / NDA", idx, _CONFIDENTIALITY_PATTERNS),
        payment_terms=_field("Payment Terms", idx, _PAYMENT_PATTERNS),
    )
