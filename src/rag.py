"""Conversational Retrieval-Augmented Generation (RAG) pipeline powered by Gemini."""

from dataclasses import dataclass, field
from typing import Any, List, Tuple

from src.config import Settings
from src.loader import DocumentChunk
from src.vectorstore import VectorStore


FALLBACK_CHAT_MODELS = (
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
)

SYSTEM_PROMPT = (
    "You are a helpful research paper assistant. Answer the user's question using ONLY "
    "the provided document excerpts. Cite page numbers like [p. 3] when referencing specific "
    "claims. If the excerpts do not contain enough information to answer, say clearly that "
    "the provided document does not cover that topic."
)


@dataclass
class ChatTurn:
    """Single user/assistant exchange in conversation memory."""

    question: str
    answer: str


@dataclass
class ResearchAssistant:
    """Handles Gemini embedding, retrieval, query condensation, and grounded generation."""

    store: VectorStore
    settings: Settings
    client: Any | None = None
    history: List[ChatTurn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.client is None:
            if not self.settings.gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Add it to your .env file or environment."
                )
            from google import genai

            self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Generate Gemini embeddings for a list of strings in batches."""
        from google.genai import types

        assert self.client is not None
        embeddings: List[List[float]] = []
        config = types.EmbedContentConfig(output_dimensionality=self.settings.embedding_dim)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.models.embed_content(
                model=self.settings.embedding_model,
                contents=batch,
                config=config,
            )
            embeddings.extend(list(item.values) for item in resp.embeddings)
        return embeddings

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Embed and store document chunks in the configured vector store."""
        embeddings = self.embed_texts([c.text for c in chunks])
        self.store.add_documents(chunks, embeddings)

    def _generate_with_fallback(self, contents: str, config: Any) -> str:
        """Call Gemini generate_content, falling back to other models on transient 503/429/404."""
        assert self.client is not None
        candidates = list(dict.fromkeys([self.settings.chat_model, *FALLBACK_CHAT_MODELS]))
        last_exc: Exception | None = None
        for model_name in candidates:
            try:
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return (resp.text or "").strip()
            except Exception as exc:
                msg = str(exc).upper()
                if any(code in msg for code in ("503", "429", "404", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "NOT_FOUND")):
                    last_exc = exc
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def _condense_question(self, question: str) -> str:
        """Rewrite a follow-up question into a standalone query using recent history."""
        if not self.history:
            return question

        from google.genai import types

        history_text = "\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer}"
            for turn in self.history[-4:]
        )
        prompt = (
            "Given the conversation history and a follow-up question, rephrase the "
            "follow-up into a clear standalone question.\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"Follow-up: {question}\nStandalone question:"
        )
        condensed = self._generate_with_fallback(
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return condensed or question

    def ask(self, question: str) -> Tuple[str, List[Tuple[DocumentChunk, float]]]:
        """Answer a question using retrieved paper chunks and update conversation history."""
        from google.genai import types

        standalone_query = self._condense_question(question)
        query_vector = self.embed_texts([standalone_query])[0]
        matches = self.store.similarity_search(query_vector, top_k=self.settings.top_k)

        context_blocks = [
            f"[{chunk.source} | p. {chunk.page}]\n{chunk.text}"
            for chunk, _ in matches
        ]
        context = "\n\n---\n\n".join(context_blocks)

        user_prompt = (
            f"Document excerpts:\n{context}\n\n"
            f"Question: {standalone_query}"
        )
        answer = self._generate_with_fallback(
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        self.history.append(ChatTurn(question=question, answer=answer))
        return answer, matches

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self.history.clear()
