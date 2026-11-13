"""Unit tests for PDF loading, local vector storage, and Gemini RAG conversation flow."""

from pathlib import Path
from types import SimpleNamespace
import pytest

from src.config import Settings
from src.loader import DocumentChunk, load_pdf_chunks, split_text
from src.rag import ResearchAssistant
from src.vectorstore import LocalVectorStore


class _FakeModelsAPI:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_content(self, model: str, contents: list[str], config=None):
        embeddings = []
        for text in contents:
            lower = text.lower()
            vec = [
                1.0 if "rag" in lower or "retrieval" in lower else 0.1,
                1.0 if "pinecone" in lower or "vector" in lower else 0.1,
                1.0 if "benchmark" in lower or "dataset" in lower else 0.1,
            ]
            embeddings.append(SimpleNamespace(values=vec))
        return SimpleNamespace(embeddings=embeddings)

    def generate_content(self, model: str, contents: str, config=None):
        self.calls.append(contents)
        if "Standalone question:" in contents:
            reply = "What benchmarks were used to evaluate RAG?"
        else:
            reply = "RAG combines parametric and non-parametric memory [p. 1]."
        return SimpleNamespace(text=reply)


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.models = _FakeModelsAPI()


def test_split_text_respects_size_and_overlap():
    text = " ".join(f"sentence{i}." for i in range(60))
    chunks = split_text(text, chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


def test_split_text_validation():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=100, chunk_overlap=100)


def test_load_sample_pdf():
    pdf_path = Path("data/RAGPaper.pdf")
    chunks = load_pdf_chunks(pdf_path, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) > 10
    assert chunks[0].source == "RAGPaper.pdf"
    assert chunks[0].page == 1


def test_local_vector_store_roundtrip(tmp_path: Path):
    chunks = [
        DocumentChunk("c1", "Retrieval augmented generation uses external memory.", "doc.pdf", 1),
        DocumentChunk("c2", "Pinecone is a managed vector database.", "doc.pdf", 2),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store = LocalVectorStore(tmp_path, index_name="test_idx")
    store.add_documents(chunks, embeddings)
    assert store.exists()

    reloaded = LocalVectorStore(tmp_path, index_name="test_idx")
    reloaded.load()
    results = reloaded.similarity_search([0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] > 0.9


def test_research_assistant_qa_and_followup(tmp_path: Path):
    store = LocalVectorStore(tmp_path, index_name="rag_test")
    settings = Settings(gemini_api_key="test-key", vector_db_dir=tmp_path, top_k=2)
    fake_client = _FakeGeminiClient()
    assistant = ResearchAssistant(store=store, settings=settings, client=fake_client)

    chunks = [
        DocumentChunk("c1", "RAG uses retrieval over a non-parametric index.", "paper.pdf", 1),
        DocumentChunk("c2", "Evaluated on Natural Questions benchmark dataset.", "paper.pdf", 4),
    ]
    assistant.index_chunks(chunks)

    answer1, matches1 = assistant.ask("How does RAG work?")
    assert "parametric" in answer1
    assert len(matches1) == 2
    assert len(assistant.history) == 1

    answer2, _ = assistant.ask("What benchmarks did it use?")
    assert answer2
    assert len(assistant.history) == 2

    assistant.clear_history()
    assert assistant.history == []
