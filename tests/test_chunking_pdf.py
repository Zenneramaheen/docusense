"""Unit tests for chunking and PDF parsing (no models needed)."""

from app.chunking import chunk_text
from app.pdfparse import extract_pages
from tests.conftest import make_pdf_bytes


def test_chunk_text_basic():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert len(chunks) >= 2
    assert all(len(c) <= 40 for c in chunks)
    # every sentence appears in at least one chunk
    for sent in ["First sentence.", "Second sentence.", "Third sentence.", "Fourth sentence."]:
        assert any(sent in c for c in chunks)


def test_chunk_overlap_shares_text():
    text = " ".join(f"Sentence number {i}." for i in range(20))
    chunks = chunk_text(text, chunk_size=80, overlap=30)
    assert len(chunks) >= 2
    # consecutive chunks share some words (the overlap tail)
    shared = set(chunks[0].split()) & set(chunks[1].split())
    assert len(shared) > 0


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_long_single_sentence():
    # one pathological sentence longer than chunk_size gets hard-split
    text = "word " * 200
    chunks = chunk_text(text.strip(), chunk_size=50, overlap=10)
    assert all(len(c) <= 50 for c in chunks)
    assert len(chunks) > 1


def test_extract_pages_tracks_page_numbers():
    pdf = make_pdf_bytes(["hello page one", "hello page two"])
    pages = extract_pages(pdf)
    assert [p[0] for p in pages] == [1, 2]
    assert "page one" in pages[0][1]
    assert "page two" in pages[1][1]


def test_extract_pages_invalid_bytes():
    import pytest

    with pytest.raises(ValueError):
        extract_pages(b"definitely not a pdf")
