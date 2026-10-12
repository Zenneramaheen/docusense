"""Extractive question answering over retrieved chunks.

Extractive QA does NOT generate text: it finds the span of the context that
best answers the question (like highlighting with a marker). That makes it
honest -- the answer always comes verbatim from your documents -- but it can
only answer questions whose answer is literally stated in the text.

The model (DistilBERT fine-tuned on SQuAD) is loaded lazily. If it cannot be
downloaded (offline machine, blocked network), we raise ModelUnavailable and
the API falls back to "retrieval-only" mode: no generated answer, but the
most relevant chunks are still returned as evidence. The API never crashes.
"""

from .config import QA_MODEL


class ModelUnavailable(Exception):
    """Raised when the QA model cannot be loaded (e.g. no network)."""


class Answerer:
    def __init__(self, model_name: str = QA_MODEL) -> None:
        self.model_name = model_name
        self._pipe = None
        self._failed = False  # remember failure; don't retry every request

    def _load(self) -> None:
        if self._failed:
            raise ModelUnavailable(
                f"QA model '{self.model_name}' failed to load earlier "
                "(probably no network to download it). Running retrieval-only."
            )
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline

            # ~250 MB download on first use, then cached in ~/.cache.
            self._pipe = pipeline("question-answering", model=self.model_name)
        except Exception as exc:
            self._failed = True
            raise ModelUnavailable(
                f"Could not load QA model '{self.model_name}': {exc}"
            ) from exc

    def answer(self, question: str, contexts: list[str]) -> tuple[str, float]:
        """Run QA over each context; return (best_answer, best_score).

        Score is the model's confidence in [0, 1].
        """
        self._load()
        best_answer, best_score = "", 0.0
        for context in contexts:
            try:
                result = self._pipe(question=question, context=context)
            except Exception:
                continue  # one bad context shouldn't kill the request
            score = float(result.get("score", 0.0))
            answer = str(result.get("answer", "")).strip()
            if answer and score > best_score:
                best_answer, best_score = answer, score
        return best_answer, best_score
