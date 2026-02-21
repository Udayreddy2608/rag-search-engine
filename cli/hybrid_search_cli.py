import argparse
from hybrid_search import HybridSearch
from search_utils import load_movies
from gemini import generate_response, rewrite_query, expand_query

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the dataset")
    normalize_parser.add_argument("scores", nargs="+", help="Path to the input dataset")

    weighted_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor for BM25 scores (0 to 1)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    rrf_parser = subparsers.add_parser("rrf-search", help= "Reciprocal ranked fusion search")
    rrf_parser.add_argument("query", type=str, help="input query")
    rrf_parser.add_argument("--k",type= int, default=60, help="value of K parameter default = 60")
    rrf_parser.add_argument("--limit", type= int, default= 5, help="number of results")
    rrf_parser.add_argument("--enhance", type= str, choices=['spell','rewrite','expand'], help= "Query Enhance Method")

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

        case "rrf-search":
            try:
                print("Performing RRF search")
                try:
                    documents = load_movies()
                    if not documents:
                        raise ValueError("No documents loaded from load_movies()")
                except Exception as e:
                    print(f"[ERROR] Failed while loading documents: {e}")
                    raise
                try:
                    hs = HybridSearch(documents=documents)
                except Exception as e:
                    print(f"[ERROR] HybridSearch initialization failed: {e}")
                    raise
                try:
                    original_query = args.query

                    if args.enhance == "spell":
                        print("Inside Spell")
                        query = generate_response(query=original_query)

                    elif args.enhance == "rewrite":
                        print("Inside Rewrite")
                        query = rewrite_query(query=original_query)

                    elif args.enhance == "expand":
                        print("Inside Expand")
                        print("Original:", original_query)
                        query = expand_query(query=original_query)
                        print("Expanded:", query)
                        print("Type:", type(query))

                    else:
                        query = original_query

                    if args.enhance:
                        print(
                            f"Enhanced query ({args.enhance}): "
                            f"'{original_query}' -> '{query}'\n"
                        )

                except Exception as e:
                    print(f"[ERROR] Query enhancement failed: {e}")
                    raise
                    
                try:
                    results = hs.rrf_search(query=query)
                    if not results:
                        print("No results found.")
                except Exception as e:
                    print(f"[ERROR] RRF search failed: {e}")
                    raise
                
                print(results)

                for idx, doc in enumerate(results, start=1):
                    try:
                        title = doc.get("title", "N/A")
                        rrf = float(doc.get("rrf_score", 0))
                        bm25_rank = doc.get("bm25_rank", "N/A")
                        semantic_rank = doc.get("semantic_rank", "N/A")
                        overview = doc.get("overview", "")

                        print(f"{idx}. {title}")
                        print(f"   RRF Score: {rrf:.3f}")
                        print(
                            f"   BM25 Rank: {bm25_rank}, "
                            f"Semantic Rank: {semantic_rank}"
                        )
                        print(f"   {overview[:120]}...")
                        print()

                    except Exception as e:
                        print(f"[WARNING] Failed processing document {idx}: {e}")

            except Exception as e:
                print(f"[FATAL] RRF search execution failed: {e}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()