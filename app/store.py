"""In-memory document store backed by a FAISS vector index.

How retrieval works here:
  1. Each chunk is embedded into a normalized 384-d vector.
  2. Vectors live in a FAISS `IndexFlatIP` (inner product == cosine
     similarity for normalized vectors). "Flat" means exact brute-force
     search -- perfect recall, plenty fast up to ~1M vectors.
  3. A parallel Python list `metas` maps each vector position to its
     metadata (document id/name, page, chunk text).

Deletion: FAISS flat indexes don't support cheap single-vector deletes, so
we keep the raw embedding matrix in Python and simply rebuild the index
after a delete. Correct and simple; fine at this scale.
"""

import uuid
from datetime import datetime, timezone

import numpy as np

from .chunking import chunk_text
from .embeddings import Embedder


class DocumentStore:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._docs: dict[str, dict] = {}      # doc_id -> document summary
        self._embeddings: list[np.ndarray] = []  # raw vectors, parallel to _metas
        self._metas: list[dict] = []         # one entry per vector
        self._index = None                   # faiss.IndexFlatIP, rebuilt on change

    # -- index management -------------------------------------------------
    def _rebuild_index(self) -> None:
        import faiss  # local import: faiss only needed when we index/search

        self._index = faiss.IndexFlatIP(self._embedder.dim)
        if self._embeddings:
            matrix = np.stack(self._embeddings).astype(np.float32)
            self._index.add(matrix)

    # -- writes -----------------------------------------------------------
    def add_document(self, name: str, pages: list[tuple[int, str]]) -> dict:
        """Chunk, embed and index a document. Returns its summary."""
        chunked: list[dict] = []
        for page_num, text in pages:
            for piece in chunk_text(text):
                chunked.append({"page": page_num, "text": piece})
        if not chunked:
            raise ValueError("No indexable text found in document.")

        vectors = self._embedder.embed([c["text"] for c in chunked])

        doc_id = uuid.uuid4().hex[:12]
        self._docs[doc_id] = {
            "id": doc_id,
            "name": name,
            "num_pages": len(pages),
            "num_chunks": len(chunked),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        for vec, meta in zip(vectors, chunked):
            self._embeddings.append(vec)
            self._metas.append(
                {
                    "doc_id": doc_id,
                    "doc_name": name,
                    "page": meta["page"],
                    "text": meta["text"],
                }
            )
        self._rebuild_index()
        return self._docs[doc_id]

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all its vectors. Returns False if missing."""
        if doc_id not in self._docs:
            return False
        keep = [m["doc_id"] != doc_id for m in self._metas]
        self._embeddings = [v for v, k in zip(self._embeddings, keep) if k]
        self._metas = [m for m, k in zip(self._metas, keep) if k]
        del self._docs[doc_id]
        self._rebuild_index()
        return True

    # -- reads ------------------------------------------------------------
    def list_documents(self) -> list[dict]:
        return list(self._docs.values())

    def search(
        self, query: str, top_k: int = 5, doc_id: str | None = None
    ) -> list[dict]:
        """Return the top-k most similar chunks (with similarity scores).

        With a doc_id filter we over-fetch then filter, so the top-k are the
        best chunks *within that document*, not just the global top-k.
        """
        if not self._metas:
            return []
        query_vec = self._embedder.embed([query])
        # Over-fetch when filtering so we still fill top_k after the filter.
        fetch_k = min(len(self._metas), top_k if doc_id is None else max(top_k * 10, 50))
        scores, indices = self._index.search(query_vec, fetch_k)
        hits: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metas[int(idx)]
            if doc_id is not None and meta["doc_id"] != doc_id:
                continue
            hits.append({**meta, "score": float(score)})
            if len(hits) >= top_k:
                break
        return hits
