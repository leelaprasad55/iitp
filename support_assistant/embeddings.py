"""Local embeddings and ChromaDB persistence using all-MiniLM-L6-v2."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import chromadb
import numpy as np

try:
    from .ingest import load_chunks
except ImportError:
    from ingest import load_chunks

PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_tokenizer = None
_st_model = None


def _install_datasets_stub() -> None:
    """
    Some Windows environments block pyarrow DLLs that `datasets` needs.
    Sentence-transformers imports Dataset only for training helpers, not for
    encoding, so a tiny stub lets us still use all-MiniLM-L6-v2.
    """
    if "datasets" in sys.modules:
        return
    stub = types.ModuleType("datasets")

    class Dataset:  # pragma: no cover - import shim
        pass

    stub.Dataset = Dataset
    sys.modules["datasets"] = stub


def _load_sentence_transformer():
    global _st_model
    if _st_model is not None:
        return _st_model
    _install_datasets_stub()
    from sentence_transformers import SentenceTransformer

    _st_model = SentenceTransformer(MODEL_NAME)
    return _st_model


def _load_transformers_backbone():
    """Fallback encoder: same MiniLM weights via Hugging Face transformers."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model
    import torch
    from transformers import AutoModel, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    _model = AutoModel.from_pretrained(HF_MODEL_ID)
    _model.eval()
    return _tokenizer, _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode texts with MiniLM and L2-normalize (cosine-friendly)."""
    try:
        model = _load_sentence_transformer()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in np.asarray(vectors)]
    except Exception:
        import torch
        import torch.nn.functional as F

        tokenizer, model = _load_transformers_backbone()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().tolist()


class MiniLMEmbeddingFunction:
    """Chroma-compatible embedding function wrapping all-MiniLM-L6-v2."""

    def name(self) -> str:
        return MODEL_NAME

    def __call__(self, input: list[str]) -> list[list[float]]:
        return embed_texts(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return embed_texts(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return embed_texts(input)


def embedding_fn():
    return MiniLMEmbeddingFunction()


def get_collection(persist_dir: Path = PERSIST_DIR, rebuild: bool = False):
    """Open (or rebuild) the Chroma collection."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def ingest_to_chroma(persist_dir: Path = PERSIST_DIR) -> int:
    """Chunk documents, embed them, and store vectors in ChromaDB."""
    chunks = load_chunks()
    collection = get_collection(persist_dir, rebuild=True)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return collection.count()


def retrieve_top_k(query: str, k: int = 3, persist_dir: Path = PERSIST_DIR) -> list[dict]:
    """Embed the query and return the top-k most similar chunks (cosine)."""
    collection = get_collection(persist_dir, rebuild=False)
    if collection.count() == 0:
        ingest_to_chroma(persist_dir)
        collection = get_collection(persist_dir, rebuild=False)

    result = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]
    for i, text in enumerate(docs):
        distance = float(dists[i]) if i < len(dists) else 1.0
        similarity = max(0.0, min(1.0, 1.0 - distance))
        hits.append(
            {
                "id": ids[i] if i < len(ids) else f"hit-{i}",
                "text": text,
                "source": metas[i].get("source", "unknown") if i < len(metas) else "unknown",
                "distance": distance,
                "similarity": similarity,
            }
        )
    return hits


if __name__ == "__main__":
    n = ingest_to_chroma()
    print(f"Stored {n} chunks in ChromaDB.")
    for hit in retrieve_top_k("What is the return policy?", k=3):
        print(hit["source"], round(hit["similarity"], 3), hit["text"][:80])
