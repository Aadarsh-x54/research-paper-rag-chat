"""Application configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings for Gemini embeddings, generation, and vector storage."""

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))

    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "research-rag-index")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    vector_db_dir: Path = Path(os.getenv("VECTOR_DB_DIR", "vector_db"))


def get_settings() -> Settings:
    """Return a fresh Settings instance reflecting current environment variables."""
    load_dotenv(override=False)
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash"),
        embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "research-rag-index"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("TOP_K", "4")),
        vector_db_dir=Path(os.getenv("VECTOR_DB_DIR", "vector_db")),
    )
