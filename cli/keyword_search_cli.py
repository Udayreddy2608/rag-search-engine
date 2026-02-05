#!/usr/bin/env python3
import argparse
from inverted_index import InvertedIndex
from keyword_search import search_movies_kw


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str)

    subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    match args.command:

        case "build":
            idx = InvertedIndex()
            idx.build()
            idx.save()

            docs = idx.get_documents("merida")
            if docs:
                print(f"First document for token 'merida' = {docs[0]}")
            else:
                print("Build failed: 'merida' not found")

        case "search":
            print(f"Searching for: {args.query}")
            movies = search_movies_kw(args.query)

            for i, movie in enumerate(movies[:5]):
                print(f"{i}. {movie}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
