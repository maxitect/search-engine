import argparse
import torch
import pickle
from chromadb import Client
from rich.console import Console
from rich.table import Table

import src.config as config
from src.utils.tokenise import preprocess
from src.utils.model_loader import load_twotowers

console = Console()


def search(query, top_k=5):
    """Search for documents matching a query."""
    # Load models and vocabulary
    console.print("[bold blue]Loading model and vocabulary...[/bold blue]")
    vocab_to_int = pickle.load(open(config.VOCAB_TO_ID_PATH, 'rb'))
    model = load_twotowers(vocab_to_int)
    chroma_client = Client()
    collection = chroma_client.get_collection(name="ms_marco_docs")

    # Process query
    console.print(f"[bold green]Processing query:[/bold green] {query}")
    device = next(model.parameters()).device
    tokens = preprocess(query)
    ids = [vocab_to_int.get(tok, 0) for tok in tokens[:config.MAX_QRY_LEN]]

    # Pad to max length
    if len(ids) < config.MAX_QRY_LEN:
        ids += [0] * (config.MAX_QRY_LEN - len(ids))

    # Convert to tensor and get query embedding
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        query_emb = model.query_tower(x).cpu().numpy().tolist()[0]

    # Query ChromaDB
    console.print("[bold blue]Searching documents...[/bold blue]")
    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k
    )

    # Display results
    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Document ID", style="magenta")
    table.add_column("Similarity", style="green")
    table.add_column("Document", style="white")

    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        similarity = results.get(
            'distances', [[0]*len(results['ids'][0])])[0][i]
        document = results['documents'][0][i]

        # Truncate document text for display
        doc_preview = document[:200] + \
            "..." if len(document) > 200 else document

        table.add_row(
            str(i+1),
            doc_id,
            f"{similarity:.4f}",
            doc_preview
        )

    console.print(table)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Engine CLI")
    parser.add_argument("query", help="Query string to search for")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )

    args = parser.parse_args()
    search(args.query, args.top_k)
