#!/usr/bin/env python3
import argparse
import math
from inverted_index import BM25_K1, InvertedIndex, BM25_B
from keyword_search import search_movies_kw, get_bm25_idf, get_bm25_tf


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

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

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
        
        case "bm25idf":
            idx.load()
            try:
                bm25idf = get_bm25_idf(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except ValueError as e:
                print(f"Error: {e}")

        case "tfidf":
            idx.load()
            tf = idx.get_tf(args.document_id, args.term)
            total_docs = len(idx.docmap)
            term_docs = len(idx.get_documents(args.term))
            idf = math.log((total_docs + 1) / (term_docs + 1))
            tfidf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.document_id}': {tfidf:.2f}")
        
        case "bm25tf":
            idx.load()
            try:
                bm25tf = get_bm25_tf(args.doc_id, args.term, args.k1,b=args.b)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except ValueError as e:
                print(f"Error: {e}")
        
        case "bm25search":
            idx.load()
            results = idx.bm25_search(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
