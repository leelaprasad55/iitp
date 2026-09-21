# Zepto Support Assistant (RAG)

Local retrieval-augmented support bot. Default mode is **MOCK_LLM** (no paid API).

## Pipeline (file / node map)

1. **Ingestion** — [`ingest.py`](ingest.py) `load_documents()` reads `docs/doc_01.txt` … `doc_08.txt`.
2. **Chunking** — [`ingest.py`](ingest.py) `chunk_text()` / `load_chunks()`.
3. **Embedding** — [`embeddings.py`](embeddings.py) `all-MiniLM-L6-v2` via sentence-transformers.
4. **ChromaDB** — [`embeddings.py`](embeddings.py) `ingest_to_chroma()` stores chunks with cosine space.
5. **Retrieval** — LangGraph node `retrieve_and_answer` in [`graph.py`](graph.py) calls `retrieve_top_k(..., k=3)`.
6. **Generation** — same node builds `"Based on the retrieved context: ..."` in MOCK_LLM mode; `direct_answer` for general questions.

Intent routing:

```
User query → classify_intent → policy? → retrieve_and_answer
                              → else  → direct_answer
```

## Run independently

From the repository root:

```bash
python -m support_assistant.ingest
python -m support_assistant.embeddings
python -m support_assistant.graph
python -m support_assistant.main
```

FastAPI:

```bash
set MOCK_LLM=1
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

## Docker

Build **from this folder** (`support_assistant`):

```bash
docker build -t zepto-support .
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support
```

## Example questions

Policy (retrieval route): `What is the return policy?`  
General (direct route): `Tell me a joke`

See the root README for captured JSON responses after a live `/ask` call.

## Optional real LLM

`MOCK_LLM=0` with `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`, `OPENAI_MODEL`). Failed Pydantic JSON is retried up to 2 extra times with a schema repair instruction. Keys never live in the repo.
