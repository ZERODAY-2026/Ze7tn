# Legal Document Triage Agent

**ZeroDay OSS Hacktoberfest 2026 · PS2 — Agentic AI: Private Legal Document Triage Agent**

> **Not legal advice.** This tool helps you *locate and quote* what a document explicitly says. It is not a lawyer, can miss clauses or mis-extract text, and does not create an attorney–client relationship. **Consult a qualified lawyer** before acting on any document.

## Problem statement

Legal documents such as rental agreements, employment contracts, legal notices and service agreements contain complex terminology and clauses that ordinary people find hard to understand, and professional help for every first question is not always practical.

## What it does

Upload a PDF or `.txt` file (or paste text) and the app, running entirely on your machine:

1. **Numbers every line** of the document (`Line 1`, `Line 2`, …) so anything it shows can be traced back to the source.
2. **Locates eight clause categories** across the whole document with fixed patterns (no AI involved): parties, effective/key dates, notice periods, penalties/damages, termination, governing jurisdiction, confidentiality/NDA, and payment terms. Each category shows up to three excerpts with their exact line ranges, or *"The provided document is silent on this matter."*
3. **Answers questions** with a small local Gemma model. Every statement carries a verbatim quote and a `[Line N]` citation, and is shown next to the cited source lines.
4. **Cross-checks with AI (optional):** asks the local model to extract the same eight categories; a value is shown only if it passes the same quote-and-citation check.
5. **Shows the source text** with the lines cited in the latest answer highlighted.

## How answers are checked

The model is told to answer only from the numbered lines it is given, as `- statement "verbatim quote" [Line N]`, or to reply exactly *"The provided document is silent on this matter."* The prompt is not trusted on its own. After the model replies, `ground_answer` (in `src/gemma_agent.py`) keeps a statement only if:

- it cites at least one line,
- every cited line was actually shown to the model, and
- every quotation in it appears verbatim in the cited lines (ignoring case, extra whitespace and quote/dash styles; `...` is allowed between fragments).

Everything else is dropped and a warning reports how many statements were removed. If nothing survives, the answer is the "silent" sentence. Set `LEGAL_TRIAGE_REQUIRE_QUOTE=0` to accept cited statements without a quote (any quote that is present must still be verbatim).

**What this does not guarantee**

- The check verifies the *quote and the citation*, not that the sentence around the quote agrees with it. A statement with a real quote but wrong wording would still be kept. Read the cited lines shown beside each answer.
- "Silent" means *nothing verifiable was found*. It can also mean retrieval missed the right passage or the small model failed to produce a valid answer. It is not proof that the document says nothing on the topic.
- For the pattern-based clause cards, "Silent" means *no pattern matched*.

## Architecture

```
PDF / .txt / pasted text
        │
        ▼
  ingestion.py ── size guard · control-char cleanup · injection-phrase warning · line numbering
        │
        ├──► triage.py ───────────► clause cards (regex, whole document, no LLM)
        │
        ▼
  chunking.py ──► retrieval.py ──► gemma_agent.py ──► ground_answer() ──► ui.py
  (whole lines,    (local top-k)    (Ollama, localhost)   (quote + citation     (Streamlit)
   900 chars)                            ▲                 check)
                                    prompts.py
```

| Module | Role |
| --- | --- |
| `src/ingestion.py` | One entry point (`ingest_input`) for uploads and pasted text. PDF text via `pypdf`, falling back to `pdfplumber`. Truncates at 300,000 characters (with a warning), strips control characters, warns on prompt-injection phrases, assigns continuous 1-based line numbers across pages. |
| `src/chunking.py` | Builds ~900-character chunks (150 overlap) from whole lines, so each chunk knows its exact `start_line`–`end_line`. |
| `src/retrieval.py` | Top-k local retrieval. Default is an in-memory TF-IDF index (no network, no model download). Optional Chroma backend, see [Privacy](#privacy). |
| `src/prompts.py` | System and user prompts, the "silent" sentence, the injection guard, and the disclaimer text. |
| `src/gemma_agent.py` | Talks to Ollama, runs grounded Q&A and the AI extraction pass, and applies `ground_answer`. Also the command-line entry point. |
| `src/triage.py` | Pattern-based clause extraction with exact line citations. |
| `src/ui.py` | Streamlit app: Triage results, Extracted clauses, Q&A and Source text tabs, with a permanent disclaimer. |

## Model and hardware

| Setting | Value |
| --- | --- |
| Runtime | [Ollama](https://ollama.com) on `127.0.0.1` |
| Default model | `gemma2:2b` (Gemma 2 2B class, quantized) |
| Generation settings | temperature 0, context window 4096 tokens |
| Hardware target | 16 GB RAM, Ryzen 7 7735HS, AMD Radeon iGPU |
| Cloud LLMs | None |

The project is intended to stay on a small local Gemma model (Gemma 2 2B / Gemma 3n-E2B class). `LEGAL_TRIAGE_MODEL` can override the tag; the code does not enforce which model you choose.

## Privacy

Document content (file bytes, extracted text, chunks, answers) is meant to stay on the device.

| Channel | What the code does |
| --- | --- |
| LLM | Local Ollama only. On startup the app rejects an `OLLAMA_HOST` that is not `127.0.0.1` / `localhost`. This is a simple string check against misconfiguration, not a firewall. |
| PDF text | Local `pypdf` / `pdfplumber`. No cloud OCR. Scanned, image-only PDFs produce a warning (see [Limitations](#limitations)). |
| Retrieval | In-memory TF-IDF by default. No embeddings, no downloads. |
| Telemetry | The app sets Hugging Face offline flags and disables Chroma telemetry by default. **Streamlit's own usage statistics are on by default**, so launch with `--browser.gatherUsageStats false` (shown below). |
| Remote APIs | None are called by the application code. |

**Optional Chroma backend** (`LEGAL_TRIAGE_RETRIEVER=chroma`, off by default). It uses Chroma's built-in embedding model (`all-MiniLM-L6-v2`, ONNX), not `sentence-transformers`. If that model is not already cached in `~/.cache/chroma/onnx_models/`, Chroma downloads it from the internet on first use, and `HF_HUB_OFFLINE` does not prevent that. Warm the cache while online before going offline, or keep the default retriever.

## Setup and run

You need network access once, to install Python packages and pull the model. After that the app runs with the network disconnected. Use Python 3.10 or newer (the pinned `numpy` requires it).

### 1. Install Ollama and pull the model (once, online)

```
ollama pull gemma2:2b
ollama list
```

Keep Ollama running locally (`ollama serve` if it is not already a service).

### 2. Create the Python environment (once, online)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: `pip install psutil` to show RAM usage in the sidebar.

### 3. Run the app (network can be off)

From the repository root:

```
streamlit run src/ui.py --browser.gatherUsageStats false
```

Open the local URL Streamlit prints (typically `http://localhost:8501`), choose **Upload file** or **Paste text**, and click **Analyze document**. The pattern-based clause cards appear immediately; the **Q&A** tab and the **AI cross-check** call the local model.

### Command-line check (no UI)

```
python -m src.gemma_agent examples/sample_nda.txt "What is the confidentiality period?"
```

Prints the grounded answer followed by the disclaimer. Warnings go to stderr. Use `--model` to pick another local model tag.

### Offline checklist

- [ ] `gemma2:2b` (or another small Gemma tag) appears in `ollama list`
- [ ] `pip install -r requirements.txt` has finished
- [ ] `OLLAMA_HOST` is unset or points to `127.0.0.1` / `localhost`
- [ ] `LEGAL_TRIAGE_RETRIEVER` is unset or `memory`
- [ ] Streamlit was started with `--browser.gatherUsageStats false`
- [ ] No VPN or proxy is rewriting localhost calls

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LEGAL_TRIAGE_MODEL` | `gemma2:2b` | Local Ollama model tag. |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Must be `127.0.0.1` / `localhost`. |
| `LEGAL_TRIAGE_RETRIEVER` | `memory` | `memory` (TF-IDF) or `chroma` (see [Privacy](#privacy)). |
| `LEGAL_TRIAGE_REQUIRE_QUOTE` | `1` | Set to `0` to accept cited statements that carry no quote. |

## Example

`examples/sample_nda.txt` is a short, fictional NDA (38 lines). The pattern-based clause extraction on it reports:

| Category | Result | Source |
| --- | --- | --- |
| Parties | Located | Lines 5–8 |
| Effective Date / Key Dates | Located | Line 5 |
| Confidentiality / NDA | Located | Lines 10–11, 15 |
| Notice Periods | Silent | — |
| Penalties / Damages | Silent | — |
| Termination Conditions | Silent | — |
| Governing Jurisdiction | Silent | — |
| Payment Terms | Silent | — |

Grounded answers use this shape (illustrative; real model output varies):

```
- The obligations last two years after receipt. "continue for two (2) years after the date" [Line 20]
```

## Tests

```
python -m pytest
```

Run from the repository root. The tests do not call Ollama or the network.

## Limitations

- **English-oriented patterns.** The clause cards rely on fixed regular expressions tuned for English contract wording, so unusual phrasing is missed. Each category is either Located or Silent; the code defines an "ambiguous" status but the pattern matcher does not currently produce it.
- **Small model, small context.** Q&A sends the top 6 retrieved chunks to a 2B model with a 4096-token window. Retrieval is lexical (word overlap), so a question phrased with different words than the document can miss the right passage.
- **AI extraction covers the start of long documents.** It analyses at most four windows of about 6,000 characters each and warns when later lines were not analysed. The pattern-based cards always cover the whole document.
- **Document size cap.** Text beyond 300,000 characters is dropped, with a warning.
- **No OCR.** Scanned PDFs without a text layer produce a warning. Convert them locally (for example with Tesseract) or paste the text.
- **Prompt-injection handling is best effort.** The app warns when a document contains phrases like "ignore previous instructions" and the prompts tell the model to ignore embedded instructions, but this is not a guarantee.
- **Quote-level verification only.** See [How answers are checked](#how-answers-are-checked).

## Project layout

```
src/
  ingestion.py   chunking.py   retrieval.py
  prompts.py     triage.py     gemma_agent.py   ui.py
tests/           unit tests (no Ollama or network required)
examples/        fictional sample document
requirements.txt
```

## Legal disclaimer

**This software is not a law firm and does not provide legal advice.** It is a document-grounded triage helper. It may miss clauses, extract text poorly, or mis-quote. Always consult a qualified legal professional licensed in the relevant jurisdiction before relying on a document or taking action.

## License

MIT, see [LICENSE](LICENSE). Files under `examples/` are fictional placeholders, not real contracts and not legal forms.
