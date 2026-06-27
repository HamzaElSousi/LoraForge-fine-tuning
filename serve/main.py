"""Thin OpenAI-compatible auth proxy in front of Ollama.

Ollama already exposes ``/v1/chat/completions``, but it has no auth and lets the client set
any system prompt. This proxy adds the two things the PRD actually calls for and nothing
more:

  1. API-key auth   - requests must carry ``Authorization: Bearer <LORAFORGE_API_KEY>``.
  2. Locked system prompt - the server-side system prompt is injected and any client-sent
     system message is dropped, so callers cannot override the assistant's behavior.

Everything else (messages, temperature, streaming) is passed straight through to Ollama,
so any OpenAI SDK client works by pointing ``base_url`` at this service. No torch here:
inference happens in Ollama/llama.cpp, this is just a CPU-light gateway.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

API_KEY = os.environ.get("LORAFORGE_API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
MODEL = os.environ.get("LORAFORGE_MODEL", "loraforge-ft")
SYSTEM_PROMPT = os.environ.get(
    "LORAFORGE_SYSTEM_PROMPT",
    "You are a helpful, concise customer-support assistant. Answer only using information "
    "you are confident about. If a question is outside customer support or you do not know, "
    "say so plainly instead of guessing.",
)
REQUEST_TIMEOUT = float(os.environ.get("LORAFORGE_TIMEOUT", "120"))

app = FastAPI(title="LoRAForge Serve", version="1.0.0")


def _check_auth(authorization: str | None) -> None:
    """401 unless the request carries the configured bearer key."""
    if not API_KEY:
        # Fail closed: refuse to run open if no key is configured.
        raise HTTPException(status_code=503, detail="server missing LORAFORGE_API_KEY")
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _enforce_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Force our model + system prompt; drop any client-supplied system messages."""
    messages = [m for m in payload.get("messages", []) if m.get("role") != "system"]
    payload["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    payload["model"] = MODEL
    return payload


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, authorization: str | None = Header(default=None)
) -> Any:
    _check_auth(authorization)
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="request body must be valid JSON")
    if not isinstance(payload, dict) or "messages" not in payload:
        raise HTTPException(status_code=400, detail="missing 'messages'")

    payload = _enforce_policy(payload)
    upstream = f"{OLLAMA_BASE_URL}/chat/completions"

    if payload.get("stream"):
        return StreamingResponse(_proxy_stream(upstream, payload), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(upstream, json=payload)
    return JSONResponse(status_code=resp.status_code, content=resp.json())


async def _proxy_stream(upstream: str, payload: dict[str, Any]):
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        async with client.stream("POST", upstream, json=payload) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
