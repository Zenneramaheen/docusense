"""Shared pytest fixtures for the DocuSense test suite.

Heavy ML models (sentence-transformers, transformers) are replaced with
tiny deterministic fakes so the suite is fast and works fully offline.
The real FAISS index and the real PDF parser ARE exercised.
"""

import hashlib
import re
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

import app.main as main
from app.qa import ModelUnavailable
from app.store import DocumentStore


# ---------------------------------------------------------------- fakes ---
class FakeEmbedder:
    """Deterministic word-hash embeddings (dim=64, L2-normalized).

    Words hashing into the same buckets give texts with shared vocabulary
    high cosine similarity -- enough to test retrieval ranking honestly.
    """

    dim = 64

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.stack(vecs).astype(np.float32)


class FakeAnswerer:
    """Canned extractive-QA stand-in."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def answer(self, question: str, contexts: list[str]):
        if self.fail:
            raise ModelUnavailable("simulated offline")
        return "24 days", 0.92


def make_pdf_bytes(pages: list[str]) -> bytes:
    """Build a tiny in-memory PDF with one page per string."""
    pdf = FPDF()
    for body in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, body)
    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


@pytest.fixture()
def client(monkeypatch):
    """TestClient wired to a fresh store with fake (offline) models."""
    monkeypatch.setattr(main, "store", DocumentStore(embedder=FakeEmbedder()))
    monkeypatch.setattr(main, "answerer", FakeAnswerer())
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def client_no_qa(monkeypatch):
    """TestClient whose QA model always fails -> retrieval-only fallback."""
    monkeypatch.setattr(main, "store", DocumentStore(embedder=FakeEmbedder()))
    monkeypatch.setattr(main, "answerer", FakeAnswerer(fail=True))
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def sample_pdf() -> bytes:
    return make_pdf_bytes(
        [
            "Acme Corp leave policy. Confirmed employees get 24 annual leave "
            "days per year, credited quarterly.",
            "Acme Corp probation period is 6 months. Core working hours are "
            "11am to 4pm IST, Monday to Friday.",
        ]
    )
