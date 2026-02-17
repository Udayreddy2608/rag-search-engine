import argparse
from lib.chunked_semantic import ChunkedSemanticSearch
from inverted_index import InvertedIndex
from hybrid_search import HybridSearch, normalize_scores, hybrid_score
from search_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the dataset")
    normalize_parser.add_argument("scores", nargs="+", help="Path to the input dataset")

    weighted_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor for BM25 scores (0 to 1)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            print(f"Normalizing dataset: {args.scores}")
            scores = [float(score) for score in args.scores]
            if not scores:
                pass
            else:
                if min(scores) == max(scores):
                    print([1.0 for _ in scores])
                else:
                    normalized_scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]
                    print(normalized_scores)
                
        case "weighted-search":
            print(f"Performing weighted hybrid search for query: '{args.query}' with alpha={args.alpha}")
            documents = load_movies()
            hs = HybridSearch(documents=documents)
            results = hs.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']} (Combined Score: {result['combined_score']:.4f})")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()