"""Streamlit UI - fully on-device. Permanent disclaimer. Data never leaves this machine.

Layout
  sidebar : input method toggle (upload file | paste text) -> one "Analyze" action
  main    : tabs for Triage results, Extracted clauses, Q&A history, Source text

Every extracted item and every answer carries source-line badges (Line 12 /
Lines 12-15) and a numbered-margin excerpt, like pleading paper.
Only system fonts are used so nothing is fetched from the network.
"""

from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import streamlit as st

from src.gemma_agent import DEFAULT_MODEL, ExtractedItem, GemmaTriageAgent, TriageResult
from src.ingestion import ExtractedDocument, compress_ranges, ingest_input
from src.prompts import DISCLAIMER_BLOCK, LEGAL_DISCLAIMER, SILENT_PHRASE
from src.triage import ClauseReport, FieldResult, FieldStatus

MODE_UPLOAD = "Upload file"
MODE_PASTE = "Paste text"
_LONG_LINE = 320  # characters; longer lines are clipped in excerpt strips

# ─────────────────────── styling ──────────────────────────────────


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --lt-blue: #2f4ea2;
            --lt-rule: #b3363f;
            --lt-line: rgba(128,128,128,0.30);
            --lt-tint: rgba(128,128,128,0.07);
            --lt-serif: "Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
        }
        .block-container { padding-top: 1.6rem; max-width: 1320px; }

        .lt-title { font-family: var(--lt-serif); font-size: 2rem; font-weight: 600;
                    letter-spacing: -0.01em; margin: 0; line-height: 1.15; }
        .lt-sub { opacity: .7; margin: .2rem 0 1rem; font-size: .95rem; }

        .lt-brand { border-left: 3px double var(--lt-rule); padding: .1rem 0 .1rem .75rem; margin-bottom: .8rem; }
        .lt-brand b { font-family: var(--lt-serif); font-size: 1.15rem; }
        .lt-brand span { display: block; font-size: .8rem; opacity: .7; }

        .lt-pill { display: inline-block; border: 1px solid var(--lt-line); border-radius: 999px;
                   padding: 2px 11px; font-size: .78rem; margin: 0 6px 6px 0; }

        .lt-card { border: 1px solid var(--lt-line); border-radius: 6px; padding: .8rem 1rem .6rem;
                   margin: .35rem 0 .8rem; background: var(--lt-tint); }
        .lt-card.silent { border-style: dashed; background: transparent; }
        .lt-card.answer { border-left: 3px solid var(--lt-blue); }
        .lt-head { display: flex; justify-content: space-between; align-items: center;
                   gap: .6rem; flex-wrap: wrap; margin-bottom: .45rem; }
        .lt-title-sm { font-weight: 650; font-size: .98rem; }
        .lt-status { font-size: .74rem; padding: 1px 9px; border-radius: 999px; border: 1px solid var(--lt-line); opacity: .8; }
        .lt-status.ok { border-color: var(--lt-blue); color: var(--lt-blue); opacity: 1; font-weight: 600; }

        .lt-badge { display: inline-block; background: var(--lt-blue); color: #fff; border-radius: 3px;
                    padding: 1px 8px; font-size: .74rem; font-weight: 600; margin: 0 5px 5px 0;
                    font-variant-numeric: tabular-nums; white-space: nowrap; }
        .lt-badge.none { background: transparent; color: inherit; opacity: .65; border: 1px dashed var(--lt-line); font-weight: 500; }

        .lt-quote { font-family: var(--lt-serif); font-size: 1rem; line-height: 1.55; margin: .15rem 0 .5rem; }
        .lt-silent-text { font-family: var(--lt-serif); font-style: italic; opacity: .75; margin: .1rem 0 .5rem; }

        .lt-paper { border-left: 3px double var(--lt-rule); margin: .35rem 0 .45rem; }
        .lt-pl { display: flex; gap: .7rem; font-family: var(--lt-serif); font-size: .92rem; line-height: 1.5; }
        .lt-ln { flex: 0 0 2.7rem; text-align: right; color: var(--lt-rule); font-size: .78rem;
                 font-variant-numeric: tabular-nums; padding-top: .12rem; user-select: none; }
        .lt-lt { flex: 1; overflow-wrap: anywhere; }
        .lt-pl.hl { background: rgba(250,204,21,0.24); }
        .lt-more { font-size: .76rem; opacity: .6; padding-left: 3.4rem; }

        .lt-chip { display: inline-block; border: 1px solid var(--lt-line); border-radius: 4px;
                   padding: 3px 9px 0; margin: 0 8px 8px 0; font-size: .84rem; }
        .lt-chip.found { border-left: 3px solid var(--lt-blue); }
        .lt-chip.silent { border-style: dashed; opacity: .65; padding-bottom: 3px; }

        .lt-source { font-size: .9rem; }
        .lt-empty { border: 1px dashed var(--lt-line); border-radius: 6px; padding: 1.6rem 1.8rem; margin-top: .6rem; }
        .lt-empty h3 { font-family: var(--lt-serif); margin: 0 0 .4rem; }
        .lt-foot { border-top: 3px double var(--lt-rule); margin-top: 1.4rem; padding-top: .6rem;
                   font-size: .82rem; opacity: .85; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────── html helpers ─────────────────────────────


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _badges(numbers) -> str:
    """Line badges: 'Line 12', 'Lines 12-15'; one badge per contiguous range."""
    ranges = compress_ranges(numbers)
    if not ranges:
        return '<span class="lt-badge none">No source lines</span>'
    out = []
    for a, b in ranges:
        label = "Line %d" % a if a == b else "Lines %d\u2013%d" % (a, b)
        out.append('<span class="lt-badge">%s</span>' % label)
    return "".join(out)


def _strip(
    doc: ExtractedDocument,
    start: int,
    end: int,
    snippet: str | None = None,
    max_lines: int = 5,
) -> str:
    """Numbered-margin excerpt of lines start..end, straight from the source text."""
    nums = list(range(start, end + 1))
    rows = []
    for n in nums[:max_lines]:
        text = doc.get_line(n)
        if len(text) > _LONG_LINE:
            text = snippet if (snippet and start == end) else text[:_LONG_LINE] + "\u2026"
        rows.append(
            '<div class="lt-pl"><span class="lt-ln">%d</span><span class="lt-lt">%s</span></div>'
            % (n, _esc(text) or "&nbsp;")
        )
    if len(nums) > max_lines:
        rows.append('<div class="lt-more">+%d more line(s)</div>' % (len(nums) - max_lines))
    return '<div class="lt-paper">%s</div>' % "".join(rows)


def _card(title: str, status_html: str, body_html: str, badges_html: str, variant: str = "") -> str:
    head = ""
    if title or status_html:
        head = '<div class="lt-head"><span class="lt-title-sm">%s</span>%s</div>' % (_esc(title), status_html)
    return '<div class="lt-card %s">%s%s<div>%s</div></div>' % (variant, head, body_html, badges_html)


def _clause_card(doc: ExtractedDocument, f: FieldResult) -> str:
    if f.status == FieldStatus.NOT_FOUND:
        return _card(
            f.label,
            '<span class="lt-status">Silent</span>',
            '<div class="lt-silent-text">%s</div>' % _esc(SILENT_PHRASE),
            _badges([]),
            "silent",
        )
    body = "".join(_strip(doc, e.start_line, e.end_line, e.text) for e in f.evidence)
    return _card(f.label, '<span class="lt-status ok">Located</span>', body, _badges(f.line_numbers))


def _item_card(doc: ExtractedDocument, it: ExtractedItem) -> str:
    if it.silent:
        return _card(
            it.label,
            '<span class="lt-status">Silent</span>',
            '<div class="lt-silent-text">%s</div>' % _esc(SILENT_PHRASE),
            _badges([]),
            "silent",
        )
    body = '<div class="lt-quote">%s</div>' % _esc(it.value)
    return _card(it.label, '<span class="lt-status ok">Verified</span>', body, _badges(it.lines))


def _statement_card(doc: ExtractedDocument, text: str, lines: list[int]) -> str:
    strips = "".join(_strip(doc, a, b, max_lines=4) for a, b in compress_ranges(lines))
    return _card(
        "",
        "",
        '<div class="lt-quote">%s</div>%s' % (_esc(text), strips),
        _badges(lines),
        "answer",
    )


def _source_html(doc: ExtractedDocument, highlight: set[int]) -> str:
    rows = []
    for dl in doc.lines:
        cls = "lt-pl hl" if dl.number in highlight else "lt-pl"
        rows.append(
            '<div class="%s"><span class="lt-ln">%d</span><span class="lt-lt">%s</span></div>'
            % (cls, dl.number, _esc(dl.text) or "&nbsp;")
        )
    return '<div class="lt-paper lt-source">%s</div>' % "".join(rows)


# ─────────────────────── state helpers ────────────────────────────


def _agent() -> GemmaTriageAgent:
    if "agent" not in st.session_state:
        st.session_state.agent = GemmaTriageAgent()
    return st.session_state.agent


def _ram_info() -> str:
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        return f"RAM {(mem.total - mem.available) / 1e9:.1f} / {mem.total / 1e9:.1f} GB"
    except ImportError:
        return "RAM: psutil not installed"


def _analyze(mode: str, uploaded, pasted: str, title: str) -> None:
    """Both input modes go through ingest_input(): same guards, same line numbering."""
    try:
        with st.spinner("Reading the document and numbering its lines locally\u2026"):
            if mode == MODE_UPLOAD:
                if uploaded is None:
                    st.warning("Choose a .pdf or .txt file first.")
                    return
                doc = ingest_input(data=uploaded.getvalue(), filename=uploaded.name)
            else:
                doc = ingest_input(text=pasted, name=title)
            if not doc.lines:
                st.error("No text could be read from this input.")
                for w in doc.warnings:
                    st.warning(w)
                return
            agent = _agent()
            n_chunks = agent.load_document(doc)
            report = agent.extract_clauses_from_doc()
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state.update(
        doc=doc, n_chunks=n_chunks, clause_report=report, chat=[], ai=None, highlight=set()
    )


# ─────────────────────── sections ─────────────────────────────────


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="lt-brand"><b>Legal Triage</b>'
            "<span>Runs on this device. No cloud calls, localhost Ollama only, no telemetry.</span></div>",
            unsafe_allow_html=True,
        )
        mode = st.radio("Input method", [MODE_UPLOAD, MODE_PASTE], key="input_mode")
        uploaded, pasted, title = None, "", ""
        if mode == MODE_UPLOAD:
            uploaded = st.file_uploader(
                "Document (.pdf or .txt)",
                type=["pdf", "txt"],
                help="Processed in local memory only. Nothing is sent anywhere.",
            )
        else:
            title = st.text_input("Title (optional)", value="Pasted document")
            pasted = st.text_area(
                "Document text",
                height=280,
                placeholder="Paste the full text here. Each line you paste becomes a numbered line.",
            )
        if st.button("Analyze document", type="primary", use_container_width=True):
            _analyze(mode, uploaded, pasted, title)

        st.divider()
        with st.expander("Local runtime"):
            st.write(f"Model: `{DEFAULT_MODEL}`")
            st.write("Ollama: `http://127.0.0.1:11434`")
            st.write("Retriever: in-memory TF-IDF")
            st.write(_ram_info())
        st.caption("Google Gemma via local Ollama \u00b7 MIT licence")


def _empty_state() -> None:
    st.markdown(
        '<div class="lt-empty"><h3>Add a document to begin</h3>'
        "Choose <b>Upload file</b> or <b>Paste text</b> in the sidebar, then select "
        "<b>Analyze document</b>. Every line is numbered, and every extracted clause and "
        "answer cites the lines it came from. When the text does not say something, "
        "you will see: <i>%s</i></div>" % _esc(SILENT_PHRASE),
        unsafe_allow_html=True,
    )


def _doc_strip(doc: ExtractedDocument) -> None:
    pages = max((dl.page for dl in doc.lines), default=1)
    origin = "pasted text" if doc.source_kind == "paste" else "uploaded file"
    st.markdown(
        '<span class="lt-pill"><b>%s</b></span><span class="lt-pill">%s</span>'
        '<span class="lt-pill">%d lines</span><span class="lt-pill">%d page(s)</span>'
        '<span class="lt-pill">%d chunks</span>'
        % (_esc(doc.name), origin, doc.line_count, pages, st.session_state.get("n_chunks", 0)),
        unsafe_allow_html=True,
    )
    for w in doc.warnings:
        st.warning(w)


def _tab_triage(doc: ExtractedDocument, report: ClauseReport) -> None:
    fields = report.all_fields()
    pages = max((dl.page for dl in doc.lines), default=1)
    cols = st.columns(4)
    for col, (label, value) in zip(
        cols,
        [
            ("Numbered lines", f"{doc.line_count:,}"),
            ("Pages", f"{pages}"),
            ("Categories located", f"{report.found_count} of {len(fields)}"),
            ("Categories silent", f"{report.silent_count}"),
        ],
    ):
        with col, st.container(border=True):
            st.metric(label, value)

    with st.container(border=True):
        st.markdown("**Coverage by category**")
        chips = []
        for f in fields:
            if f.status == FieldStatus.NOT_FOUND:
                chips.append('<span class="lt-chip silent">%s &nbsp;silent</span>' % _esc(f.label))
            else:
                chips.append('<span class="lt-chip found">%s &nbsp;%s</span>' % (_esc(f.label), _badges(f.line_numbers)))
        st.markdown("".join(chips), unsafe_allow_html=True)
        st.caption("Each badge is the exact source line range. Silent means no matching line exists in the text.")

    summary = report.risk_summary()
    (st.warning if report.silent_count else st.success)(summary)


def _tab_clauses(doc: ExtractedDocument, report: ClauseReport) -> None:
    st.caption("Matched with fixed patterns across the whole document. Excerpts are copied from the cited lines.")
    cols = st.columns(2)
    for i, f in enumerate(report.all_fields()):
        with cols[i % 2]:
            st.markdown(_clause_card(doc, f), unsafe_allow_html=True)

    with st.expander("AI cross-check with Gemma", expanded=st.session_state.get("ai") is not None):
        st.caption(
            "Asks the local model to extract the same categories. A value is shown only if it "
            "cites lines and its quote is verbatim in those lines; otherwise it is reported as silent."
        )
        if st.button("Run AI extraction"):
            try:
                with st.spinner(f"Querying local {DEFAULT_MODEL}\u2026"):
                    st.session_state.ai = _agent().ask_extraction()
            except Exception as exc:  # noqa: BLE001
                st.error(
                    "Local Ollama call failed. Confirm `ollama serve` is running "
                    f"and `{DEFAULT_MODEL}` is pulled.\n\n{exc}"
                )
        ai = st.session_state.get("ai")
        if ai is not None:
            for w in ai.warnings:
                st.warning(w)
            c2 = st.columns(2)
            for i, it in enumerate(ai.items):
                with c2[i % 2]:
                    st.markdown(_item_card(doc, it), unsafe_allow_html=True)


def _render_result(doc: ExtractedDocument, res: TriageResult) -> None:
    if res.elapsed is not None:
        st.markdown(
            '<span class="lt-pill">%.1fs</span><span class="lt-pill">%s</span>' % (res.elapsed, _esc(res.model)),
            unsafe_allow_html=True,
        )
    if res.silent or not res.statements:
        st.markdown(
            _card("", "", '<div class="lt-silent-text">%s</div>' % _esc(res.answer), _badges([]), "silent"),
            unsafe_allow_html=True,
        )
    else:
        for s in res.statements:
            st.markdown(_statement_card(doc, s.text, s.lines), unsafe_allow_html=True)
    for w in res.warnings:
        if w not in ("no_document", "empty_retrieval") and w not in doc.warnings:
            st.caption(w)
    if res.citations:
        with st.expander(f"Retrieved passages ({len(res.citations)})"):
            for p in res.citations:
                st.markdown(
                    '<span class="lt-badge">%s</span><span class="lt-pill">%s \u00b7 page %s</span>'
                    % (_esc(p.get("line_ref", "")), _esc(p.get("id", "")), p.get("page", "?")),
                    unsafe_allow_html=True,
                )
                st.code(p.get("numbered_text", p.get("text", "")), language=None)


def _tab_qa(doc: ExtractedDocument) -> None:
    top = st.columns([4, 1])
    with top[0]:
        st.caption(
            "Answers use only the numbered lines of this document. Each statement shows the lines "
            "it relies on. If the text does not say, you get: " + SILENT_PHRASE
        )
    with top[1]:
        if st.button("Clear history", use_container_width=True, disabled=not st.session_state.get("chat")):
            st.session_state.chat = []
            st.session_state.highlight = set()
            st.rerun()

    history = st.container()
    question = st.chat_input("Ask a question about this document")
    if question and question.strip():
        t0 = time.perf_counter()
        try:
            with st.spinner(f"Querying local {DEFAULT_MODEL}\u2026"):
                res = _agent().ask(question.strip())
            res.elapsed = time.perf_counter() - t0
            st.session_state.chat.append({"q": question.strip(), "res": res})
            st.session_state.highlight = set(res.cited_lines)
        except Exception as exc:  # noqa: BLE001
            st.error(
                "Local Ollama call failed. Confirm `ollama serve` is running "
                f"and `{DEFAULT_MODEL}` is pulled.\n\n{exc}"
            )

    with history:
        if not st.session_state.chat:
            st.markdown(
                '<div class="lt-empty">Ask about notice periods, payment terms, who the parties are, '
                "or anything else you need located in the text.</div>",
                unsafe_allow_html=True,
            )
        for turn in st.session_state.chat:
            with st.chat_message("user"):
                st.write(turn["q"])
            with st.chat_message("assistant", avatar="\u2696\ufe0f"):
                _render_result(doc, turn["res"])


def _tab_source(doc: ExtractedDocument) -> None:
    highlight: set[int] = set()
    if st.checkbox("Highlight lines cited in the latest answer", value=True):
        highlight = set(st.session_state.get("highlight") or ())
    st.caption(f"{doc.line_count:,} numbered lines. Numbers match the citations everywhere in this app.")
    with st.container(height=560, border=True):
        st.markdown(_source_html(doc, highlight), unsafe_allow_html=True)


# ─────────────────────── main ─────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Legal Triage \u2014 On-Device",
        layout="wide",
        page_icon="\u2696\ufe0f",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    st.session_state.setdefault("chat", [])
    _sidebar()

    st.markdown(
        '<h1 class="lt-title">Legal Document Triage</h1>'
        '<div class="lt-sub">Every extracted detail and every answer cites the exact source lines.</div>',
        unsafe_allow_html=True,
    )
    st.warning(DISCLAIMER_BLOCK, icon="\u26a0\ufe0f")

    doc: ExtractedDocument | None = st.session_state.get("doc")
    report: ClauseReport | None = st.session_state.get("clause_report")
    if doc is None or report is None:
        _empty_state()
    else:
        _doc_strip(doc)
        t_triage, t_clauses, t_qa, t_source = st.tabs(
            ["Triage results", "Extracted clauses", "Q&A", "Source text"]
        )
        with t_triage:
            _tab_triage(doc, report)
        with t_clauses:
            _tab_clauses(doc, report)
        with t_qa:
            _tab_qa(doc)
        with t_source:
            _tab_source(doc)

    st.markdown(f'<div class="lt-foot">{_esc(LEGAL_DISCLAIMER)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
