import argparse
from lib.semantic_search import embed_query_text, verify_model, embed_text, SemanticSearch
from search_utils import load_movies

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command")
    verify_parser = subparsers.add_parser("verify", help="Verify the embedding model")
    embed_parser = subparsers.add_parser("embed_text", help="Embed text using the embedding model")
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embedding cache")
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a search query")
    embed_query_parser.add_argument("query", type=str, help="Search query to generate embedding for user query")
    search_parser = subparsers.add_parser("search", help="Search for relevant documents based on a query")
    search_parser.add_argument("query", type=str, help="Search query to find relevant documents")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model(EMBEDDING_MODEL)
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            documents = load_movies()
            semantic_search = SemanticSearch()
            embeddings =  semantic_search.load_or_create_embeddings(documents)
            print(f"Number of docs:   {len(documents)}")
            print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
        
        case "embedquery":
            embed_query_text(args.query)
        
        case "search":
            print(f"Searching for: '{args.query}' with limit {args.limit}")
            documents = load_movies()
            semantic_search = SemanticSearch()
            semantic_search.load_or_create_embeddings(documents)
            print("Performing semantic search...")
            results = semantic_search.search(args.query, limit=args.limit)
            for idx, result in enumerate(results):
                print(f"{idx+1}. {result['title']} (Score: {result['score']:.4f})")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()