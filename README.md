# DocuSense — RAG-Powered Document Q&A API

**Ask questions about your PDFs, get answers with cited sources.**

Upload a PDF → DocuSense extracts the text, splits it into overlapping chunks,
converts each chunk into a vector embedding, and stores it in a FAISS vector
index. When you ask a question, it retrieves the most relevant chunks and an
extractive QA model reads them to produce an answer — always citing the exact
passages it came from.

Everything runs **locally**: no API keys, no paid services, no data leaves your machine.

---

## Features

- 📄 **PDF upload** (`POST /documents/upload`) — text extraction (per page), overlapping chunking, embedding, FAISS indexing
- ❓ **Question answering** (`POST /ask`) — semantic retrieval + extractive QA, with answer, confidence score, and cited source chunks (document name + page + excerpt)
- 🗂️ **Document management** — list (`GET /documents`) and delete (`DELETE /documents/{id}`) indexed documents; deleting also removes their vectors
- 🎛️ **Doc filter** — restrict a question to a single document
- 🖥️ **Demo UI** — minimal web page to upload PDFs and ask questions
- 🛟 **Graceful offline fallback** — if the QA model can't be downloaded, the API still returns the most relevant passages as evidence instead of crashing

## Architecture

```
                        ┌──────────────────────────────┐
                        │        FastAPI (app/)        │
                        └──────────────┬───────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐            ┌───────────────────┐            ┌──────────────────┐
│  INGEST       │            │  RETRIEVE         │            │  READ & ANSWER   │
│               │            │                   │            │                  │
│ PDF bytes     │            │ Question          │            │ Top-k chunks     │
│   → pypdf     │            │   → embed         │            │   → DistilBERT   │
│     (per page)│            │     (MiniLM)      │            │     (SQuAD)      │
│   → chunk     │            │   → FAISS search  │            │   → best answer  │
│     (overlap) │            │     (cosine sim)  │            │     span + score │
│   → embed     │            │                   │            │                  │
│     (MiniLM)  │            └───────────────────┘            └──────────────────┘
│   → FAISS add │                                                       │
└───────────────┘                                          (offline? → return
                                                            chunks as evidence)
```

**Module map**

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI routes, request validation, error mapping |
| `app/store.py` | In-memory doc registry + FAISS index (add/search/delete) |
| `app/embeddings.py` | Lazy `sentence-transformers` wrapper (MiniLM-L6-v2, 384-d, normalized) |
| `app/qa.py` | Lazy `transformers` QA pipeline + `ModelUnavailable` fallback |
| `app/chunking.py` | Sentence-aware overlapping chunker |
| `app/pdfparse.py` | Per-page PDF text extraction via pypdf |
| `app/schemas.py` | Pydantic request/response models |
| `static/index.html` | Demo UI (vanilla HTML/JS) |

## Setup

Requires Python 3.10+.

```bash
# 1. Clone / enter the project
cd docusense

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the CPU-only torch build FIRST (skips the ~2.5 GB CUDA wheel)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 4. Install everything else
pip install -r requirements.txt
```

> First run downloads two models from Hugging Face (~90 MB embeddings +
> ~250 MB QA), cached in `~/.cache` afterwards. An internet connection is only
> needed for that first download.

## Run

```bash
uvicorn app.main:app --reload
```

- Demo UI: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Docker

```bash
docker build -t docusense .
docker run -p 8000:8000 docusense
```

## Try it (with the included sample doc)

A fake "Acme Corp HR Policy" PDF ships in `sample_docs/` (generate it with
`python scripts/make_sample_pdf.py` if it's missing).

```bash
# 1. Upload
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@sample_docs/acme-hr-policy.pdf"

# 2. List documents
curl http://localhost:8000/documents

# 3. Ask (replace DOC_ID from step 1 if you want to filter)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many annual leave days do employees get?"}'

# 4. Ask within one document only
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the core working hours?", "doc_id": "DOC_ID", "top_k": 3}'

# 5. Delete
curl -X DELETE http://localhost:8000/documents/DOC_ID
```

Example answer:

```json
{
  "question": "How many annual leave days do employees get?",
  "answer": "24",
  "confidence": 0.94,
  "mode": "extractive-qa",
  "sources": [
    {
      "doc_id": "a1b2c3d4e5f6",
      "doc_name": "acme-hr-policy.pdf",
      "page": 2,
      "excerpt": "Confirmed employees are entitled to 24 annual leave days per calendar year...",
      "score": 0.81
    }
  ]
}
```

## Running tests

```bash
pytest -v
```

The suite uses tiny deterministic fake models (no downloads, fully offline) but
exercises the **real** FAISS index, the real PDF parser, and the real HTTP
layer via FastAPI's `TestClient`. It covers: upload, non-PDF rejection, empty
PDF, list, delete (including vector removal), ask with answer+sources,
doc-filtered ask, the offline QA fallback, and chunking edge cases.

## Key concepts to learn (interview prep)

**1. Embeddings** — A neural network (here MiniLM) maps each text chunk to a
point in 384-dimensional space, where *meaning* becomes *geometry*: chunks with
similar meaning end up close together. "Annual leave policy" and "vacation
days rules" land near each other even though they share no words — that's what
keyword search can't do.

**2. Vector search (FAISS)** — To find relevant chunks we embed the question
the same way and ask: *which chunk-vectors point in the most similar
direction?* FAISS is Meta's library for exactly this nearest-neighbor search.
We use `IndexFlatIP` (exact brute-force inner product); because our vectors
are L2-normalized, inner product = cosine similarity. Exact search is perfect
recall and plenty fast up to ~1M vectors — beyond that you'd use approximate
indexes (IVF, HNSW).

**3. Chunking** — Models have input limits (MiniLM: 512 tokens), and retrieval
is sharper on focused passages, so documents are split into ~500-char chunks.
We pack whole sentences (never cut mid-thought) and carry a 100-char overlap
into the next chunk so an answer straddling a boundary still appears whole
somewhere.

**4. Extractive QA** — The reader model (DistilBERT fine-tuned on SQuAD)
doesn't *generate* text; it predicts which *span* of the given context answers
the question, plus a confidence score. The answer is always verbatim from
your document — no hallucination possible — but it can only answer what is
literally stated in the text.

**5. RAG (Retrieval-Augmented Generation)** — The overall pattern: *retrieve*
relevant passages, then *read* them to answer. Same idea powers
ChatGPT-with-your-docs products; here the "generator" is an extractive
reader, which makes every answer citable.

## Limitations & next steps

- **In-memory storage** — restart the server and the index is gone. Next step: persist FAISS + metadata in SQLite.
- **No OCR** — scanned/image PDFs yield no text (upload returns a clear 400). Tesseract integration would fix this.
- **Single-file scope** — no auth, no multi-user. Fine for a portfolio/demo; add JWT for production.
- **Extractive only** — answers can't synthesize across chunks. Swapping the reader for a local LLM (e.g. via Ollama) would enable abstractive answers.

## Tech stack

Python · FastAPI · sentence-transformers (all-MiniLM-L6-v2) · FAISS (faiss-cpu) ·
transformers (distilbert-base-cased-distilled-squad) · pypdf · NumPy · pytest · Docker
