# DocuSense -- RAG document Q&A API (CPU-only, no API keys needed)
FROM python:3.11-slim

WORKDIR /app

# faiss-cpu wheels need libgomp (OpenMP); keep the image otherwise minimal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch first (avoids the multi-GB CUDA wheel), then the rest.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/
COPY sample_docs/ ./sample_docs/

# Cache HF models inside the image layer-friendly location.
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8000

# Models download on first request (~350 MB total), then are cached.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
