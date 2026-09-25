"""Load and chunk local policy documents."""

from __future__ import annotations

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent / "docs"


def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Return [{source, text}] for every .txt file in docs/."""
    documents = []
    for path in sorted(docs_dir.glob("doc_*.txt")):
        documents.append({"source": path.name, "text": path.read_text(encoding="utf-8").strip()})
    if len(documents) != 8:
        raise RuntimeError(f"Expected 8 documents, found {len(documents)} in {docs_dir}")
    return documents


def chunk_text(text: str, source: str, max_chars: int = 500, overlap: int = 80) -> list[dict]:
    """Split a document into overlapping character windows, preferring paragraph breaks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict] = []
    buffer = ""
    index = 0

    def emit(piece: str) -> None:
        nonlocal index
        cleaned = " ".join(piece.split())
        if not cleaned:
            return
        chunks.append(
            {
                "id": f"{source}::{index}",
                "source": source,
                "text": cleaned,
            }
        )
        index += 1

    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= max_chars:
            buffer = f"{buffer} {para}".strip()
            continue
        if buffer:
            emit(buffer)
            buffer = (buffer[-overlap:] + " " + para).strip() if overlap else para
        else:
            start = 0
            while start < len(para):
                end = min(start + max_chars, len(para))
                emit(para[start:end])
                if end >= len(para):
                    break
                next_start = max(0, end - overlap)
                if next_start <= start:
                    next_start = min(len(para), start + 1)
                start = next_start
        if len(buffer) > max_chars:
            emit(buffer[:max_chars])
            trim_index = max(0, len(buffer) - overlap) if overlap else len(buffer)
            buffer = buffer[trim_index:]

    if buffer:
        emit(buffer)
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Ingest all documents and return chunks."""
    chunks: list[dict] = []
    for doc in load_documents(docs_dir):
        chunks.extend(chunk_text(doc["text"], doc["source"]))
    return chunks


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from 8 documents.")
    print(chunks[0])
