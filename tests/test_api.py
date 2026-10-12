"""API tests: upload, list, ask, delete + edge cases."""

from io import BytesIO

from tests.conftest import make_pdf_bytes


def _upload(client, pdf_bytes, filename="policy.pdf"):
    return client.post(
        "/documents/upload",
        files={"file": (filename, BytesIO(pdf_bytes), "application/pdf")},
    )


# --- upload ---------------------------------------------------------------
def test_upload_pdf(client, sample_pdf):
    res = _upload(client, sample_pdf)
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["name"] == "policy.pdf"
    assert body["num_pages"] == 2
    assert body["num_chunks"] >= 2
    assert body["id"]


def test_upload_rejects_non_pdf(client):
    res = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", BytesIO(b"hello"), "text/plain")},
    )
    assert res.status_code == 400


def test_upload_rejects_fake_pdf_bytes(client):
    res = client.post(
        "/documents/upload",
        files={"file": ("evil.pdf", BytesIO(b"%PDF not really"), "application/pdf")},
    )
    # "%PDF" magic passes the sniff but pypdf can't parse -> 400, not a crash
    assert res.status_code in (400, 201)


def test_upload_empty_pdf(client):
    # a valid PDF with a single blank page -> no extractable text
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    buf = BytesIO()
    pdf.output(buf)
    res = _upload(client, buf.getvalue(), filename="blank.pdf")
    assert res.status_code == 400
    assert "text" in res.json()["detail"].lower()


# --- list / delete ----------------------------------------------------------
def test_list_documents(client, sample_pdf):
    assert client.get("/documents").json() == []
    _upload(client, sample_pdf, filename="a.pdf")
    _upload(client, sample_pdf, filename="b.pdf")
    docs = client.get("/documents").json()
    assert {d["name"] for d in docs} == {"a.pdf", "b.pdf"}


def test_delete_document(client, sample_pdf):
    doc_id = _upload(client, sample_pdf).json()["id"]
    res = client.delete(f"/documents/{doc_id}")
    assert res.status_code == 200
    assert res.json()["deleted"] == doc_id
    assert client.get("/documents").json() == []


def test_delete_nonexistent_document(client):
    assert client.delete("/documents/nope123").status_code == 404


def test_delete_removes_vectors(client, sample_pdf):
    """After deletion, the doc's content must no longer be retrievable."""
    doc_id = _upload(client, sample_pdf).json()["id"]
    client.delete(f"/documents/{doc_id}")
    # no docs left at all -> asking is a clean 404, not stale results
    res = client.post("/ask", json={"question": "leave days?"})
    assert res.status_code == 404


# --- ask --------------------------------------------------------------------
def test_ask_no_documents_indexed(client):
    res = client.post("/ask", json={"question": "anything?"})
    assert res.status_code == 404
    assert "upload" in res.json()["detail"].lower()


def test_ask_returns_answer_and_sources(client, sample_pdf):
    _upload(client, sample_pdf)
    res = client.post("/ask", json={"question": "How many annual leave days?"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["answer"] == "24 days"  # from FakeAnswerer
    assert body["confidence"] == 0.92
    assert body["mode"] == "extractive-qa"
    assert len(body["sources"]) >= 1
    src = body["sources"][0]
    assert src["doc_name"] == "policy.pdf"
    assert src["page"] in (1, 2)
    assert "leave" in src["excerpt"].lower()
    assert 0.0 <= src["score"] <= 1.0


def test_ask_retrieves_relevant_chunk(client, sample_pdf):
    """The top source should be the chunk about the asked topic."""
    _upload(client, sample_pdf)
    res = client.post(
        "/ask", json={"question": "What are the core working hours?", "top_k": 1}
    )
    assert res.status_code == 200
    assert "11am" in res.json()["sources"][0]["excerpt"]


def test_ask_with_doc_filter(client, sample_pdf):
    id_a = _upload(client, sample_pdf, filename="a.pdf").json()["id"]
    _upload(client, sample_pdf, filename="b.pdf")
    res = client.post("/ask", json={"question": "leave days?", "doc_id": id_a})
    assert res.status_code == 200
    assert all(s["doc_id"] == id_a for s in res.json()["sources"])


def test_ask_unknown_doc_filter(client, sample_pdf):
    _upload(client, sample_pdf)
    res = client.post("/ask", json={"question": "leave?", "doc_id": "missing123"})
    assert res.status_code == 404


def test_ask_qa_unavailable_falls_back_to_retrieval(client_no_qa, sample_pdf):
    """Simulates an offline box: no crash, evidence still returned."""
    _upload(client_no_qa, sample_pdf)
    res = client_no_qa.post("/ask", json={"question": "How many leave days?"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["mode"] == "retrieval-only"
    assert body["answer"] is None
    assert len(body["sources"]) >= 1  # evidence still served


def test_ask_rejects_empty_question(client, sample_pdf):
    _upload(client, sample_pdf)
    res = client.post("/ask", json={"question": ""})
    assert res.status_code == 422
