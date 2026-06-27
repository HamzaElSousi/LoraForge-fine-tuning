# Tiny CPU-only image for the auth proxy. No torch - inference lives in Ollama.
FROM python:3.12-slim

WORKDIR /app

# Install only the serving deps (fastapi/uvicorn/httpx) for a small image.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY serve/ ./serve/

EXPOSE 8000

# Ollama runs on the host; from inside the container reach it via host.docker.internal.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
    LORAFORGE_MODEL=loraforge-ft

CMD ["uvicorn", "serve.main:app", "--host", "0.0.0.0", "--port", "8000"]
