"""Vector store backends for local cosine-similarity search and optional Pinecone cloud indexing."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import List, Protocol, Tuple
import numpy as np

from src.loader import DocumentChunk


class VectorStore(Protocol):
    """Protocol implemented by local and cloud vector store backends."""

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        ...

    def similarity_search(
        self, query_embedding: List[float], top_k: int = 4
    ) -> List[Tuple[DocumentChunk, float]]:
        ...


class LocalVectorStore:
    """Persistent NumPy-backed vector store using cosine similarity."""

    def __init__(self, store_dir: str | Path, index_name: str = "default") -> None:
        self.store_dir = Path(store_dir)
        self.index_name = index_name
        self.vectors_path = self.store_dir / f"{index_name}_vectors.npz"
        self.meta_path = self.store_dir / f"{index_name}_metadata.json"
        self.chunks: List[DocumentChunk] = []
        self.matrix: np.ndarray | None = None

    def exists(self) -> bool:
        """Return True if a persisted index exists on disk."""
        return self.vectors_path.exists() and self.meta_path.exists()

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Store chunks and their embeddings, normalizing vectors for cosine search."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            raise ValueError("Cannot index an empty chunk list")

        arr = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = arr / norms
        self.chunks = list(chunks)
        self.save()

    def save(self) -> None:
        """Persist vectors and chunk metadata to disk."""
        if self.matrix is None:
            return
        self.store_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.vectors_path, vectors=self.matrix)
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, indent=2)

    def load(self) -> None:
        """Load persisted vectors and chunk metadata from disk."""
        if not self.exists():
            raise FileNotFoundError(f"Index '{self.index_name}' not found in {self.store_dir}")
        data = np.load(self.vectors_path)
        self.matrix = data["vectors"]
        with self.meta_path.open("r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        self.chunks = [DocumentChunk(**item) for item in raw_chunks]

    def similarity_search(
        self, query_embedding: List[float], top_k: int = 4
    ) -> List[Tuple[DocumentChunk, float]]:
        """Return top_k (DocumentChunk, score) pairs ranked by cosine similarity."""
        if self.matrix is None or not self.chunks:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        scores = self.matrix @ q
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.chunks[int(idx)], float(scores[int(idx)])) for idx in top_indices]


class PineconeVectorStore:
    """Optional Pinecone Serverless vector store backend."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 1536,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise ImportError(
                "pinecone package is required for the Pinecone backend. "
                "Install it with: pip install pinecone"
            ) from exc

        if not api_key:
            raise ValueError("PINECONE_API_KEY is required for the Pinecone backend")

        self.client = Pinecone(api_key=api_key)
        self.index_name = index_name

        existing = [idx.name for idx in self.client.list_indexes()]
        if index_name not in existing:
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self.index = self.client.Index(index_name)

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Upsert document chunks and embeddings into Pinecone in batches."""
        vectors = [
            {
                "id": chunk.chunk_id,
                "values": emb,
                "metadata": {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                },
            }
            for chunk, emb in zip(chunks, embeddings)
        ]
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size])

    def similarity_search(
        self, query_embedding: List[float], top_k: int = 4
    ) -> List[Tuple[DocumentChunk, float]]:
        """Query Pinecone for the top_k most similar chunks."""
        response = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        results: List[Tuple[DocumentChunk, float]] = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            chunk = DocumentChunk(
                chunk_id=match["id"],
                text=meta.get("text", ""),
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
            )
            results.append((chunk, float(match.get("score", 0.0))))
        return results

