#!/usr/bin/env python3
import argparse
from keyword_search import search_movies_kw
from inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help= "Build Inverted Index")
    search_parser.add_argument("query", type=str, help="Search query")
    inverted_index = InvertedIndex()
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies_list = search_movies_kw(query= args.query)
            if movies_list:
                for ind,name in enumerate(movies_list):
                    print(f"{ind}. {name}")
        case "build":
            inverted_index.save()
            docs = inverted_index.get_documents("merida")
            if docs:
                print(f"First document for token 'merida' = {docs[0]}")
            else:
                print("Index built but verification failed (no 'merida' found)")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()