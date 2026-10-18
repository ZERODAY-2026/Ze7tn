# PS2 — Agentic AI: Private Legal Document Triage Agent

## Problem Statement

Legal documents such as rental agreements, employment contracts, legal notices and service agreements contain complex terminology and clauses that ordinary people find hard to understand, and professional help for every first question is not always practical.

---

# Legal Document Triage Agent

<!-- your existing content starts here --># Legal Document Triage Agent

Privacy-first, **fully on-device** legal *document triage* for ZeroDay OSS Hacktoberfest 2026 (PS2).

This tool helps you **locate and quote** what an uploaded contract, policy, or notice **explicitly says**. It is **not a lawyer** and does **not** give legal advice.

> **Permanent disclaimer:** This is **not legal advice**. Outputs may be incomplete or wrong. **Consult a qualified lawyer** before acting on any document.

## Model

| Setting | Value |
| --- | --- |
| Runtime | [Ollama](https://ollama.com) on `127.0.0.1` only |
| Default model | `gemma2:2b` (Gemma 2 2B class, quantized) |
| Hardware target | 16 GB RAM, Ryzen 7 7735HS, AMD Radeon iGPU |
| Cloud LLMs | **None** |

Override the model name with `LEGAL_TRIAGE_MODEL` (must still be a **local** Ollama Gemma 2B / Gemma 3n-E2B class tag). Do not point this at a remote host.

## Privacy guarantees

Private document content (**file bytes, extracted text, chunks, embeddings, summaries, answers**) **never leaves the device**.

| Channel | Policy |
| --- | --- |
| LLM | Local Ollama only (`http://127.0.0.1:11434`) |
| OCR / PDF | Local `pypdf` / `pdfplumber` |
| Retrieval | In-memory lexical index by default (no network, no cloud embeddings) |
| Telemetry | Disabled; do not log document text |
| Remote APIs | Forbidden |

**Pass/fail:** if a code path would send document content off-device, it must not exist.

Default retrieval does **not** call Hugging Face or download embedding weights at runtime. Optional Chroma + `sentence-transformers` is off unless you set `LEGAL_TRIAGE_RETRIEVER=chroma` **and** you already have the embedding model cached locally.

## Architecture

```
PDF / .txt  →  ingestion  →  chunking  →  local retrieval
                                              ↓
Streamlit UI  ←  structured answer  ←  Gemma (Ollama)
                         ↑
                    src/prompts.py
```

1. **`src/ingestion.py`** — extract text from PDF or plain text on disk / upload.
2. **`src/chunking.py`** — overlapping passage windows with page + offset ids.
3. **`src/retrieval.py`** — top-k local passages (in-memory lexical; optional Chroma).
4. **`src/prompts.py`** — system prompt: citations, Explicit vs Interpretation, refusals.
5. **`src/gemma_agent.py`** — Ollama chat; answers grounded only in retrieved passages.
6. **`src/ui.py`** — Streamlit app with a permanent legal disclaimer.

Every answer must:

- Quote **exact passages** with citation ids (`[C1]`, page/offset).
- Separate **What the document explicitly says** from **Possible interpretation / concern**.
- Flag **uncertainty / ambiguity**.
- **Refuse** unsupported legal conclusions.
- Repeat: *This is not legal advice – consult a qualified lawyer.*

## How to run fully offline

Do the **one-time** online setup on this machine, then disconnect.

### 1. Install Ollama and pull a small Gemma (once, online)

```bash
ollama pull gemma2:2b
```

Confirm it is local:

```bash
ollama list
```

Keep Ollama running locally (`ollama serve` if it is not already a service). It must listen on **localhost**, not a remote URL.

### 2. Python env (once, online)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run with the network off

Disable Wi-Fi / Ethernet, then:

```bash
# Optional: make Hugging Face / telemetry refuse any accidental call
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set ANONYMIZED_TELEMETRY=False

streamlit run src/ui.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`). Upload a PDF or `.txt`, ask a question, and inspect citations.

CLI smoke check (no UI):

```bash
python -m src.gemma_agent examples/sample_nda.txt "What is the confidentiality period?"
```

### Offline checklist

- [ ] `gemma2:2b` (or smaller Gemma 3n/E2B) already in `ollama list`
- [ ] `pip install` already finished
- [ ] Ollama host is `127.0.0.1`
- [ ] `LEGAL_TRIAGE_RETRIEVER` unset or `memory` (default)
- [ ] No VPN/proxy rewriting localhost LLM calls to the cloud

## Legal disclaimer

**This software is not a law firm and does not provide legal advice.**  
It is a **document-grounded triage** helper. It may miss clauses, OCR poorly, or mis-quote.  
**Always consult a qualified legal professional** licensed in the relevant jurisdiction before relying on a document or taking action.

## License

MIT — see [LICENSE](LICENSE).

Sample files under `examples/` are **fictional public** placeholders, not real contracts and not legal forms.
