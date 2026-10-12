"""DocuSense: RAG-powered document Q&A API.

Upload PDFs -> they get chunked, embedded and indexed in FAISS.
Ask questions -> top chunks are retrieved and an extractive QA model
reads them to produce an answer with cited sources.
"""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import MAX_UPLOAD_MB
from .embeddings import Embedder
from .pdfparse import extract_pages
from .qa import Answerer, ModelUnavailable
from .schemas import (
    AskRequest,
    AskResponse,
    DocumentSummary,
    SourceChunk,
    UploadResponse,
)
from .store import DocumentStore

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="DocuSense",
    description="RAG-powered document Q&A: upload PDFs, ask questions, get cited answers.",
    version="1.0.0",
)

# Singletons, created lazily-safe: constructing these does NOT download
# models (that happens on first embed()/answer() call).
store = DocumentStore(embedder=Embedder())
answerer = Answerer()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(store.list_documents())}


@app.post("/documents/upload", response_model=UploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF, extract text, chunk, embed and index it."""
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File too large (max {MAX_UPLOAD_MB} MB)."
        )
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")

    try:
        pages = extract_pages(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the PDF "
            "(it may be scanned images -- OCR is not supported).",
        )

    try:
        summary = store.add_document(filename, pages)
    except RuntimeError as exc:  # e.g. sentence-transformers not installed
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return UploadResponse(**summary)


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    return [DocumentSummary(**d) for d in store.list_documents()]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    """Delete a document and all of its vectors from the index."""
    if not store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest) -> AskResponse:
    """Retrieve relevant chunks, then extract an answer with citations."""
    if not store.list_documents():
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Upload a PDF first.",
        )
    if req.doc_id and not any(
        d["id"] == req.doc_id for d in store.list_documents()
    ):
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        hits = store.search(req.question, top_k=req.top_k, doc_id=req.doc_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not hits:
        raise HTTPException(
            status_code=404, detail="No relevant passages found for that question."
        )

    contexts = [h["text"] for h in hits]
    try:
        answer_text, confidence = answerer.answer(req.question, contexts)
        mode = "extractive-qa"
    except ModelUnavailable:
        # Graceful degradation: no QA model (offline) -> return the evidence
        # itself instead of crashing.
        answer_text, confidence, mode = None, 0.0, "retrieval-only"

    sources = [
        SourceChunk(
            doc_id=h["doc_id"],
            doc_name=h["doc_name"],
            page=h["page"],
            excerpt=h["text"][:300],
            score=round(h["score"], 4),
        )
        for h in hits
    ]
    return AskResponse(
        question=req.question,
        answer=answer_text or None,
        confidence=round(confidence, 4),
        sources=sources,
        mode=mode,
    )


# --- Demo UI ---------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def demo_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
