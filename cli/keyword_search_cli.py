#!/usr/bin/env python3
import argparse
import math
from inverted_index import InvertedIndex
from keyword_search import search_movies_kw


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str)

    subparsers.add_parser("build", help="Build inverted index")
    tf_parser = subparsers.add_parser(
    "tf",
    help="Show term frequency for a term in a document"
    )
    tf_parser.add_argument("document_id", help="Document ID")
    tf_parser.add_argument("term", help="Term to look up")
    idf_parser = subparsers.add_parser("idf", help="Show inverse document frequency for a term")
    idf_parser.add_argument("term", help="Term to look up")
    tfidf_parser = subparsers.add_parser("tfidf", help="Show TF-IDF score for a term in a document")
    tfidf_parser.add_argument("document_id", help="Document ID")
    tfidf_parser.add_argument("term", help="Term to look up")

    args = parser.parse_args()

    idx = InvertedIndex()

    match args.command:

        case "build":
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

        case "tf":
            idx.load()
            tf = idx.get_tf(args.document_id, args.term)
            print(tf)
        
        case "idf":
            idx.load()
            total_docs = len(idx.docmap)
            term_docs = len(idx.get_documents(args.term))
            idf = math.log((total_docs + 1) / (term_docs + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            idx.load()
            tf = idx.get_tf(args.document_id, args.term)
            total_docs = len(idx.docmap)
            term_docs = len(idx.get_documents(args.term))
            idf = math.log((total_docs + 1) / (term_docs + 1))
            tfidf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.document_id}': {tfidf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
