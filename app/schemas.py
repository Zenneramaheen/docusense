"""Pydantic request/response schemas for the DocuSense API."""

from pydantic import BaseModel, Field

from .config import DEFAULT_TOP_K, MAX_TOP_K


class UploadResponse(BaseModel):
    id: str
    name: str
    num_pages: int
    num_chunks: int
    uploaded_at: str


class DocumentSummary(BaseModel):
    id: str
    name: str
    num_pages: int
    num_chunks: int
    uploaded_at: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer")
    doc_id: str | None = Field(
        default=None, description="Restrict search to one document"
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Chunks to retrieve"
    )


class SourceChunk(BaseModel):
    doc_id: str
    doc_name: str
    page: int
    excerpt: str
    score: float


class AskResponse(BaseModel):
    question: str
    # None when running in retrieval-only fallback (QA model unavailable).
    answer: str | None
    confidence: float
    sources: list[SourceChunk]
    # "extractive-qa" normally, "retrieval-only" when the QA model failed to load.
    mode: str
