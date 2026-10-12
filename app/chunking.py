"""Text chunking: split documents into overlapping, sentence-aware chunks.

Why chunk at all?
  * Embedding models have a max input length (MiniLM: 512 tokens).
  * Retrieval works best on focused passages: one chunk ~= one idea.
  * The QA model also has a 512-token limit per (question, context) pair.

Strategy: pack whole sentences greedily up to CHUNK_SIZE characters, then
start a new chunk that re-includes an overlap tail of the previous chunk.
Overlap guards against answers being split across a chunk boundary.
"""

import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE

# Split on sentence-ending punctuation followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Hard-split a single over-long sentence on word boundaries."""
    words, parts, current = sentence.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > chunk_size:
            parts.append(current)
            current = ""
        current = f"{current} {word}".strip() if current else word
    if current:
        parts.append(current)
    return parts or [sentence]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks of at most `chunk_size` chars."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        """Emit the current chunk; seed the next one with an overlap tail."""
        nonlocal current
        text_done = current.strip()
        if text_done:
            chunks.append(text_done)
        tail = text_done[-overlap:] if overlap > 0 else ""
        # Snap the tail to a word boundary so we never start mid-word.
        tail = tail.split(" ", 1)[-1] if " " in tail else tail
        current = tail

    for sentence in sentences:
        for part in _split_long_sentence(sentence, chunk_size):
            if current and len(current) + 1 + len(part) > chunk_size:
                flush()
                # If even the overlap tail doesn't fit with this part,
                # drop the tail and start the chunk fresh.
                if len(current) + 1 + len(part) > chunk_size:
                    current = ""
            current = f"{current} {part}".strip() if current else part

    if current.strip():
        chunks.append(current.strip())
    return chunks
