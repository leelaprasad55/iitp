"""FastAPI app for the Zepto Support Assistant."""

from __future__ import annotations

from fastapi import FastAPI

try:
    from .embeddings import ingest_to_chroma
    from .graph import ask
    from .models import AskRequest, AssistantResponse
except ImportError:
    from embeddings import ingest_to_chroma
    from graph import ask
    from models import AskRequest, AssistantResponse

app = FastAPI(title="Zepto Support Assistant", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    ingest_to_chroma()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AssistantResponse)
def ask_endpoint(payload: AskRequest) -> AssistantResponse:
    return ask(payload.query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("support_assistant.main:app", host="0.0.0.0", port=7860, reload=False)
