"""CLI entry point for chatting with a research paper PDF using RAG."""

import argparse
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.config import get_settings
from src.loader import load_pdf_chunks
from src.rag import ResearchAssistant
from src.vectorstore import LocalVectorStore, PineconeVectorStore


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a research paper PDF using RAG.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/RAGPaper.pdf"),
        help="Path to the PDF file to index and query.",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "pinecone"],
        default="local",
        help="Vector store backend to use (default: local).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector index even if a cached local index already exists.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of document chunks to retrieve per query.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if args.top_k is not None:
        settings = type(settings)(**{**settings.__dict__, "top_k": args.top_k})

    if not settings.gemini_api_key:
        console.print(
            "[bold red]Missing GEMINI_API_KEY.[/bold red] "
            "Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and add your Gemini API key."
        )
        return

    index_name = args.pdf.stem.lower().replace(" ", "_")

    if args.backend == "pinecone":
        store = PineconeVectorStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
        assistant = ResearchAssistant(store=store, settings=settings)
        if args.rebuild:
            console.print(f"[cyan]Indexing {args.pdf} into Pinecone...[/cyan]")
            chunks = load_pdf_chunks(args.pdf, settings.chunk_size, settings.chunk_overlap)
            assistant.index_chunks(chunks)
            console.print(f"[green]Indexed {len(chunks)} chunks.[/green]")
    else:
        store = LocalVectorStore(settings.vector_db_dir, index_name=index_name)
        assistant = ResearchAssistant(store=store, settings=settings)
        if store.exists() and not args.rebuild:
            store.load()
            console.print(f"[green]Loaded cached index for {args.pdf.name} ({len(store.chunks)} chunks).[/green]")
        else:
            console.print(f"[cyan]Loading and indexing {args.pdf}...[/cyan]")
            chunks = load_pdf_chunks(args.pdf, settings.chunk_size, settings.chunk_overlap)
            assistant.index_chunks(chunks)
            console.print(f"[green]Indexed {len(chunks)} chunks into {settings.vector_db_dir}.[/green]")

    console.print(
        Panel(
            f"Ready to chat with [bold]{args.pdf.name}[/bold] ({args.backend} backend).\n"
            "Commands: [bold]clear[/bold] (reset memory), [bold]exit[/bold] / [bold]quit[/bold].",
            title="Research Paper RAG Chat",
            border_style="blue",
        )
    )

    while True:
        try:
            question = console.input("\n[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            console.print("Goodbye!")
            break
        if question.lower() == "clear":
            assistant.clear_history()
            console.print("[yellow]Conversation history cleared.[/yellow]")
            continue

        try:
            with console.status("[cyan]Searching paper and generating answer...[/cyan]"):
                answer, matches = assistant.ask(question)
        except Exception as exc:
            console.print(f"[bold red]API Error:[/bold red] {exc}")
            continue

        console.print("\n[bold blue]Assistant:[/bold blue]")
        console.print(Markdown(answer))
        pages = sorted({chunk.page for chunk, _ in matches})
        console.print(f"[dim]Sources: {args.pdf.name} (pages {', '.join(map(str, pages))})[/dim]")


if __name__ == "__main__":
    main()

