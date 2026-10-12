"""Embeddings: turn text into 384-d vectors capturing meaning.

Uses sentence-transformers (all-MiniLM-L6-v2). The model is loaded LAZILY --
only on the first embed() call -- so that:
  * `import app.main` never needs torch installed,
  * the app can start (and serve the demo page / health check) even offline,
  * tests can inject a lightweight fake embedder instead.

Vectors are L2-normalized, so a FAISS inner-product index ranks chunks by
cosine similarity (1.0 = identical meaning, 0.0 = unrelated).
"""

import numpy as np

from .config import EMBEDDING_DIM, EMBEDDING_MODEL


class Embedder:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        dim: int = EMBEDDING_DIM,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None  # loaded on first use

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with: pip install -r requirements.txt "
                "(install the CPU torch build first, see README)."
            ) from exc
        # First call downloads ~90 MB from Hugging Face; cached afterwards.
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts -> float32 array of shape (n, dim)."""
        self._load()
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
