import argparse
from lib.semantic_search import embed_query_text, verify_model, embed_text, SemanticSearch
from lib.chunked_semantic import ChunkedSemanticSearch
from search_utils import load_movies
import re

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

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk into smaller pieces")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk in characters")
    chunk_parser.add_argument("--overlap", type=int, default=40, help="Number of characters to overlap between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a document into smaller pieces based on semantic similarity")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk into smaller pieces")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum size of each chunk in characters")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of characters to overlap between chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed chunks of a document")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for relevant documents based on a query using chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query to find relevant documents")
    search_chunked_parser.add_argument("--limit", type=int, default=5,  help="Number of search results to return")

    
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

        case "chunk":
            words = args.text.split()
            chunk_size = args.chunk_size
            overlap = args.overlap
            word_length = len(words)
            if overlap >= chunk_size:
                raise ValueError("Overlap must be smaller than chunk size")
            step = chunk_size - overlap
            chunks = []
            print(f"Chunking {len(args.text)} characters")
            for i in range(0, word_length, step):
                chunk = " ".join(words[i:i+chunk_size])
                chunks.append(chunk)
                if i + chunk_size >= word_length:
                    break
            for idx, chunk in enumerate(chunks):
                print(f"{idx+1}. {chunk}")

        case "semantic_chunk":
            text = args.text.strip()
            if not text:
                print("Input text is empty. No chunks to generate.")
                return []
            sentences = re.split(r"(?<=[.!?])\s+", text)
            sentences = [s.strip() for s in sentences if s.strip()]
            chunks = []
            sentence_length = len(sentences)
            step = args.max_chunk_size - args.overlap
            print(f"Semantically chunking {len(args.text)} characters")
            for i in range(0, sentence_length, step):
                chunk = " ".join(sentences[i:i+args.max_chunk_size])
                chunks.append(chunk)
                if i + args.max_chunk_size >= sentence_length:
                    break
            for idx, chunk in enumerate(chunks):
                print(f"{idx+1}. {chunk}")

            return chunks

        case "embed_chunks":
            documents = load_movies()
            chunked_sem = ChunkedSemanticSearch()
            embeddings = chunked_sem.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            movies = load_movies()
            chunked_sem = ChunkedSemanticSearch()
            chunked_sem.load_or_create_chunk_embeddings(movies)
            results = chunked_sem.search_chunks(args.query, limit=args.limit)
            for result in results:
                print(f"{result['title']} (Score: {result['score']:.4f}")
                print(f"   {result['document']}")
        case _:
            parser.print_help()
            
if __name__ == "__main__":
    main()  