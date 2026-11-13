"""FastAPI web server and chat UI for Research Paper RAG Chat."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.config import get_settings
from src.loader import load_pdf_chunks
from src.rag import ResearchAssistant
from src.vectorstore import LocalVectorStore


DEFAULT_PDF = Path(os.getenv("RAG_PDF_PATH", "data/RAGPaper.pdf"))
_assistant: ResearchAssistant | None = None


def get_assistant() -> ResearchAssistant:
    """Lazily initialize and return the shared ResearchAssistant instance."""
    global _assistant
    if _assistant is not None:
        return _assistant

    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    index_name = DEFAULT_PDF.stem.lower().replace(" ", "_")
    store = LocalVectorStore(settings.vector_db_dir, index_name=index_name)
    assistant = ResearchAssistant(store=store, settings=settings)

    if store.exists():
        store.load()
    else:
        chunks = load_pdf_chunks(DEFAULT_PDF, settings.chunk_size, settings.chunk_overlap)
        assistant.index_chunks(chunks)

    _assistant = assistant
    return _assistant


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Pre-load index if GEMINI_API_KEY is available at startup
    if os.getenv("GEMINI_API_KEY"):
        try:
            get_assistant()
        except Exception:
            pass
    yield


app = FastAPI(title="Research Paper RAG Chat", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    pages: list[int]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        assistant = get_assistant()
        answer, matches = assistant.ask(question)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc

    pages = sorted({chunk.page for chunk, _ in matches})
    return AskResponse(answer=answer, pages=pages)


@app.post("/clear")
def clear() -> dict[str, str]:
    if _assistant is not None:
        _assistant.clear_history()
    return {"status": "cleared"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Research Paper RAG Chat</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --user-bg: #1d4ed8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      justify-content: center;
      min-height: 100vh;
    }
    .container {
      width: 100%;
      max-width: 820px;
      display: flex;
      flex-direction: column;
      height: 100vh;
      padding: 16px;
      gap: 12px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
    }
    header h1 { margin: 0; font-size: 1.1rem; }
    header span { color: var(--muted); font-size: 0.85rem; }
    button.secondary {
      background: transparent;
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
    }
    button.secondary:hover { color: var(--text); border-color: var(--accent); }
    #chat {
      flex: 1;
      overflow-y: auto;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg {
      max-width: 85%;
      padding: 10px 14px;
      border-radius: 10px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .msg.user {
      align-self: flex-end;
      background: var(--user-bg);
    }
    .msg.bot {
      align-self: flex-start;
      background: #0f172a;
      border: 1px solid var(--border);
    }
    .pages {
      margin-top: 6px;
      font-size: 0.78rem;
      color: var(--accent);
    }
    form {
      display: flex;
      gap: 8px;
    }
    input[type="text"] {
      flex: 1;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      font-size: 0.95rem;
    }
    button.primary {
      padding: 12px 20px;
      border-radius: 10px;
      border: none;
      background: var(--accent);
      color: #0f172a;
      font-weight: 600;
      cursor: pointer;
    }
    button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Research Paper RAG Chat</h1>
        <span>Powered by Gemini API &middot; RAGPaper.pdf</span>
      </div>
      <button class="secondary" id="clearBtn">Clear Chat</button>
    </header>
    <div id="chat">
      <div class="msg bot">Ask any question about the research paper. Answers are grounded in the document with page citations.</div>
    </div>
    <form id="chatForm">
      <input type="text" id="question" placeholder="Ask a question about the paper..." autocomplete="off" required />
      <button type="submit" class="primary" id="sendBtn">Send</button>
    </form>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const form = document.getElementById("chatForm");
    const input = document.getElementById("question");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");

    function addMsg(text, role, pages = []) {
      const div = document.createElement("div");
      div.className = `msg ${role}`;
      div.textContent = text;
      if (pages.length) {
        const p = document.createElement("div");
        p.className = "pages";
        p.textContent = `Sources: pages ${pages.join(", ")}`;
        div.appendChild(p);
      }
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const q = input.value.trim();
      if (!q) return;
      input.value = "";
      addMsg(q, "user");
      const loading = addMsg("Thinking...", "bot");
      sendBtn.disabled = true;
      try {
        const res = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q }),
        });
        const data = await res.json();
        loading.remove();
        if (!res.ok) {
          addMsg(data.detail || "Request failed.", "bot");
        } else {
          addMsg(data.answer, "bot", data.pages || []);
        }
      } catch (err) {
        loading.remove();
        addMsg("Network error: " + err.message, "bot");
      } finally {
        sendBtn.disabled = false;
        input.focus();
      }
    });

    clearBtn.addEventListener("click", async () => {
      await fetch("/clear", { method: "POST" });
      chat.innerHTML = '<div class="msg bot">Conversation cleared. Ask a new question!</div>';
    });
  </script>
</body>
</html>
"""
