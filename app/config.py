"""Central configuration for DocuSense.

All tunables live here so behaviour can be explained (and tweaked)
from a single place.
"""

# --- Models (downloaded automatically from Hugging Face on first use) ---
# Small, fast sentence embedding model. 384-dimensional output.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Small extractive QA model fine-tuned on SQuAD. ~250 MB download.
QA_MODEL = "distilbert-base-cased-distilled-squad"

# --- Chunking ---
# Target maximum characters per chunk. Small chunks -> precise retrieval,
# but too small loses context. ~500 chars is a common sweet spot.
CHUNK_SIZE = 500
# Characters of overlap carried from the previous chunk so that a sentence
# split across a boundary still appears whole in at least one chunk.
CHUNK_OVERLAP = 100

# --- Retrieval ---
DEFAULT_TOP_K = 5   # chunks retrieved per question
MAX_TOP_K = 20

# --- Upload limits ---
MAX_UPLOAD_MB = 10
