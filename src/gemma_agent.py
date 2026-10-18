"""Local Gemma via Ollama. Document content stays on this machine.

Zero-hallucination design: the prompt tells the model to answer only from the
numbered lines and to cite them, and then ``ground_answer`` re-checks the model's
output in code. A statement survives only if

  1. it cites at least one line,
  2. every cited line was actually shown to the model, and
  3. every quotation in it appears verbatim in the cited line(s).

Anything else is removed. If nothing survives, the answer is replaced with
"The provided document is silent on this matter."
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

from src.chunking import chunk_document, split_long_line
from src.ingestion import ExtractedDocument, extract_file, format_line_ref
from src.prompts import (
    EXTRACTION_FIELDS,
    LEGAL_DISCLAIMER,
    REFUSAL_EMPTY_RETRIEVAL,
    REFUSAL_NO_DOCUMENT,
    SILENT_PHRASE,
    SYSTEM_PROMPT,
    TRIAGE_EXTRACTION_PROMPT,
    build_extraction_prompt,
    build_user_prompt,
)
from src.retrieval import LocalStore
from src.triage import ClauseReport, extract_clauses

DEFAULT_MODEL = os.environ.get("LEGAL_TRIAGE_MODEL", "gemma2:2b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Strict mode: a Q&A statement without a verbatim quote is rejected. Set
# LEGAL_TRIAGE_REQUIRE_QUOTE=0 to accept cited statements that carry no quote
# (any quote that IS present must still be verbatim).
REQUIRE_VERBATIM_QUOTE = os.environ.get("LEGAL_TRIAGE_REQUIRE_QUOTE", "1") != "0"

EXTRACTION_WINDOW_CHARS = 6000
MAX_EXTRACTION_WINDOWS = 4


def _lock_localhost() -> None:
    """Refuse non-loopback Ollama hosts so document text cannot leave the device."""
    host = OLLAMA_HOST.lower().replace("http://", "").replace("https://", "")
    allowed = host.startswith("127.0.0.1") or host.startswith("localhost")
    if not allowed:
        raise RuntimeError(
            f"Refusing OLLAMA_HOST={OLLAMA_HOST!r}. Use http://127.0.0.1:11434 only."
        )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


# ─────────────────────────── result types ───────────────────────────


@dataclass
class Statement:
    """One grounded claim: clean text plus the exact lines that support it."""

    text: str
    lines: list[int]

    @property
    def line_ref(self) -> str:
        return format_line_ref(self.lines)


@dataclass
class TriageResult:
    answer: str
    citations: list[dict] = field(default_factory=list)  # retrieved passages
    model: str = DEFAULT_MODEL
    warnings: list[str] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)
    silent: bool = False  # True -> answer is exactly SILENT_PHRASE
    removed: int = 0  # model statements dropped by the grounding check
    elapsed: float | None = None

    @property
    def cited_lines(self) -> list[int]:
        return sorted({n for s in self.statements for n in s.lines})


@dataclass
class ExtractedItem:
    key: str
    label: str
    value: str  # verbatim excerpt, or SILENT_PHRASE
    lines: list[int]
    silent: bool


@dataclass
class ExtractionResult:
    items: list[ExtractedItem]
    covered: tuple[int, int] | None  # first/last line analysed
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────── grounding checks ───────────────────────────

_CITE_RE = re.compile(r"[\[(]\s*Lines?\b\s*([^\])]*)[\])]", re.IGNORECASE)
_NUM_RE = re.compile(r"(\d+)\s*[-–—]\s*(\d+)|(\d+)")
_QUOTE_RE = re.compile(r'"([^"]{3,})"|“([^”]{3,})”')
_BULLET_RE = re.compile(r"^(?:[-*•]+|\d+[.)])\s*")
_LABEL_ONLY_RE = re.compile(r"[#*\s]*[A-Za-z ]{1,30}:?[*\s]*")
_MAX_RANGE = 500


def parse_citations(text: str) -> list[int]:
    """All line numbers cited in ``text`` via [Line 12], [Lines 12-15], [Line 3, Line 9]."""
    nums: list[int] = []
    for m in _CITE_RE.finditer(text):
        for a, b, single in _NUM_RE.findall(m.group(1)):
            if single:
                nums.append(int(single))
            else:
                lo, hi = int(a), int(b)
                if lo <= hi and hi - lo <= _MAX_RANGE:
                    nums.extend(range(lo, hi + 1))
                else:
                    nums.append(-1)  # malformed range -> will fail validation
    return nums


def strip_citations(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _CITE_RE.sub("", text)).strip(" ,;")


def _norm(s: str) -> str:
    s = (
        s.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
    )
    return " ".join(s.lower().split())


def _quote_in(quote: str, haystack: str) -> bool:
    """Quote (ellipses allowed between fragments) appears verbatim in the haystack."""
    parts = [_norm(p) for p in re.split(r"\.{3}|…", quote)]
    parts = [p.strip(" .,;:") for p in parts if len(p.strip(" .,;:")) >= 3]
    return bool(parts) and all(p in haystack for p in parts)


def _quotes(text: str) -> list[str]:
    return [a or b for a, b in _QUOTE_RE.findall(text)]


def _cited_text(doc: ExtractedDocument, numbers: list[int]) -> str:
    return _norm(" ".join(doc.get_line(n) for n in sorted(set(numbers))))


def ground_answer(
    raw: str,
    doc: ExtractedDocument,
    valid_lines: set[int],
    *,
    require_quote: bool = REQUIRE_VERBATIM_QUOTE,
) -> tuple[list[Statement], bool, int]:
    """Keep only statements that are verifiably grounded.

    Returns (statements, model_said_silent, removed_count).
    """
    statements: list[Statement] = []
    said_silent = False
    removed = 0
    for row in (raw or "").splitlines():
        s = row.strip()
        if not s:
            continue
        if SILENT_PHRASE.lower() in s.lower():
            said_silent = True
            continue
        s = _BULLET_RE.sub("", s)
        nums = parse_citations(s)
        text = strip_citations(s)
        if not nums:
            if not _LABEL_ONLY_RE.fullmatch(s):  # ignore bare labels like "Answer:"
                removed += 1
            continue
        if not text or any(n not in valid_lines for n in nums):
            removed += 1
            continue
        quotes = _quotes(text)
        hay = _cited_text(doc, nums)
        if quotes:
            if not all(_quote_in(q, hay) for q in quotes):
                removed += 1
                continue
        elif require_quote:
            removed += 1
            continue
        statements.append(Statement(text=text, lines=sorted(set(nums))))
    return statements, said_silent, removed


def valid_line_numbers(chunks_or_passages) -> set[int]:
    """Every line number inside a retrieved chunk's [start_line, end_line] range."""
    valid: set[int] = set()
    for c in chunks_or_passages:
        d = c if isinstance(c, dict) else c.as_dict()
        valid.update(range(int(d["start_line"]), int(d["end_line"]) + 1))
    return valid


def render_answer(statements: list[Statement]) -> str:
    return "\n".join(f"- {s.text} [{s.line_ref}]" for s in statements)


def _line_windows(
    doc: ExtractedDocument, budget: int
) -> list[list[tuple[int, str]]]:
    """Split the document into windows of whole numbered lines, <= budget chars each."""
    windows: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    size = 0
    for dl in doc.lines:
        t = dl.text.strip()
        if not t:
            continue
        for seg in split_long_line(t, budget // 2):
            cost = len(seg) + 12
            if cur and size + cost > budget:
                windows.append(cur)
                cur, size = [], 0
            cur.append((dl.number, seg))
            size += cost
    if cur:
        windows.append(cur)
    return windows


_EXTRACT_ROW = re.compile(r"^\s*[-*•]?\s*\**\s*([A-Z_]+)\s*\**\s*:\s*(.*)$")


# ─────────────────────────── agent ───────────────────────────


class GemmaTriageAgent:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST):
        _lock_localhost()
        self.model = model
        self.host = host if host.startswith("http") else f"http://{host}"
        if "127.0.0.1" not in self.host and "localhost" not in self.host:
            raise RuntimeError("Ollama host must be localhost.")
        self._store: LocalStore | None = None
        self._doc: ExtractedDocument | None = None
        self._n_chunks = 0

    # ---- loading -------------------------------------------------

    def load_document(self, doc: ExtractedDocument) -> int:
        """Index a document produced by ``ingest_input`` (upload OR paste)."""
        self._doc = doc
        chunks = chunk_document(doc)
        self._n_chunks = len(chunks)
        self._store = LocalStore(chunks)
        return len(chunks)

    def load_path(self, path: str) -> int:
        return self.load_document(extract_file(path))

    @property
    def document(self) -> ExtractedDocument | None:
        return self._doc

    # ---- regex clause extraction (instant, cites exact lines) ----

    def extract_clauses_from_doc(self) -> ClauseReport | None:
        """Run regex-based clause extraction (zero LLM cost, instant)."""
        if self._doc is None:
            return None
        return extract_clauses(self._doc)

    # ---- LLM extraction (verified against the cited lines) -------

    def ask_extraction(self, max_windows: int = MAX_EXTRACTION_WINDOWS) -> ExtractionResult:
        """
        LLM extraction pass over the numbered document. Each value must carry a
        [Line N] citation and a quote that is verbatim in the cited lines, else it
        is reported as silent. Raises RuntimeError if Ollama is unavailable.
        """
        if self._doc is None:
            raise RuntimeError(REFUSAL_NO_DOCUMENT)
        doc = self._doc
        windows = _line_windows(doc, EXTRACTION_WINDOW_CHARS)
        warnings: list[str] = []
        if len(windows) > max_windows:
            last = windows[max_windows - 1][-1][0]
            warnings.append(
                f"AI extraction analysed lines 1-{last} of {doc.line_count}. "
                "Later lines were not analysed by the model (the regex clause cards cover the whole document)."
            )
            windows = windows[:max_windows]

        by_key: dict[str, ExtractedItem] = {
            k: ExtractedItem(k, label, SILENT_PHRASE, [], True) for k, label in EXTRACTION_FIELDS
        }
        for win in windows:
            valid = {n for n, _ in win}
            valid = set(range(min(valid), max(valid) + 1))
            numbered = "\n".join(f"Line {n}: {t}" for n, t in win)
            note = f"(Lines {win[0][0]}-{win[-1][0]} of {doc.line_count} are shown.)"
            raw = self._complete_with_system(
                TRIAGE_EXTRACTION_PROMPT, build_extraction_prompt(numbered, doc.name, note)
            )
            for row in raw.splitlines():
                m = _EXTRACT_ROW.match(row)
                if not m or m.group(1) not in by_key or not by_key[m.group(1)].silent:
                    continue
                item = _verify_extraction(by_key[m.group(1)], m.group(2), doc, valid)
                if item is not None:
                    by_key[m.group(1)] = item
        covered = (windows[0][0][0], windows[-1][-1][0]) if windows else None
        return ExtractionResult([by_key[k] for k, _ in EXTRACTION_FIELDS], covered, warnings)

    # ---- grounded Q&A ---------------------------------------------

    def ask(self, question: str, k: int = 6) -> TriageResult:
        if self._doc is None or self._store is None:
            return TriageResult(answer=REFUSAL_NO_DOCUMENT, warnings=["no_document"])

        hits = self._store.query(question, k=k)
        if not hits:
            return TriageResult(
                answer=REFUSAL_EMPTY_RETRIEVAL,
                silent=True,
                warnings=list(self._doc.warnings) + ["empty_retrieval"],
                model=self.model,
            )

        passages = [h.as_dict() for h in hits]
        user_prompt = build_user_prompt(
            question=question,
            passages=passages,
            document_name=self._doc.name,
        )
        raw = self._complete(user_prompt)
        statements, said_silent, removed = ground_answer(
            raw, self._doc, valid_line_numbers(passages)
        )

        warnings = list(self._doc.warnings)
        if removed:
            warnings.append(
                f"{removed} model statement(s) were removed because they lacked a valid "
                "[Line N] citation or a verbatim quote from the cited lines."
            )
        if statements:
            return TriageResult(
                answer=render_answer(statements),
                citations=passages,
                model=self.model,
                warnings=warnings,
                statements=statements,
                removed=removed,
            )
        return TriageResult(
            answer=SILENT_PHRASE,
            citations=passages,
            model=self.model,
            warnings=warnings,
            silent=True,
            removed=removed,
        )

    # ---- Ollama ----------------------------------------------------

    def _complete(self, user_prompt: str) -> str:
        return self._complete_with_system(SYSTEM_PROMPT, user_prompt)

    def _complete_with_system(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("The ollama package is required. pip install -r requirements.txt") from exc

        client = ollama.Client(host=self.host)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.0,  # deterministic: grounding over creativity
                "num_ctx": 4096,
            },
        )
        message = response.get("message") or {}
        return (message.get("content") or "").strip()


def _verify_extraction(
    item: ExtractedItem, value: str, doc: ExtractedDocument, valid: set[int]
) -> ExtractedItem | None:
    """Return a FOUND item if the value is cited and verbatim in the cited lines."""
    value = value.strip()
    if not value or SILENT_PHRASE.lower() in value.lower() or "NOT_STATED" in value:
        return None
    nums = parse_citations(value)
    if not nums or any(n not in valid for n in nums):
        return None
    text = strip_citations(value)
    quotes = _quotes(text) or [text.strip('"“” ')]
    hay = _cited_text(doc, nums)
    if not text or not all(_quote_in(q, hay) for q in quotes):
        return None
    return ExtractedItem(item.key, item.label, text, sorted(set(nums)), False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="On-device document triage grounded in numbered source lines (not legal advice)."
    )
    parser.add_argument("path", help="Local PDF or .txt")
    parser.add_argument("question", help="Question about the document")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    agent = GemmaTriageAgent(model=args.model)
    agent.load_path(args.path)
    result = agent.ask(args.question)
    sys.stdout.write(result.answer + "\n\n" + LEGAL_DISCLAIMER + "\n")
    if result.warnings:
        sys.stderr.write("warnings: " + "; ".join(result.warnings) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
