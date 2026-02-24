import argparse
from search_utils import load_movies
from hybrid_search import HybridSearch
from gemini import rag_generation, summarization, summarization_with_citations, question_answer



def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser("summarize", help="Summarize top retrieved documents for a query")
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=5,help="Number of documents to retrieve and summarize (default: 5)")

    citations_parser = subparsers.add_parser("citations", help= "Include citaions for summary")
    citations_parser.add_argument("query", type= str, help= "Query to be searched")
    citations_parser.add_argument("--limit", default = 5, help= "Number of results")

    question_parser = subparsers.add_parser("question", help= "user question")
    question_parser.add_argument("question", type= str, help="User question")
    question_parser.add_argument("--limit", default= 5, help= "Number of results")

    args = parser.parse_args()
    movies = load_movies()
    hs = HybridSearch(movies)
    match args.command:
        case "rag":
            query = args.query
            rrf_results = hs.rrf_search(query= query, limit= 5)
            result = rag_generation(query= query, results= rrf_results)
            print("Search Results: ")
            for result in rrf_results:
                print(f"- {result['title']}")
            print("RAG Response:")
            print(result)
        
        case "summarize":
            query = args.query
            limit = args.limit
            rrf_results = hs.rrf_search(query= query, limit= limit)
            summary = summarization(query= query, results= rrf_results)
            print("Search Results: ")
            for result in rrf_results:
                print(f"- {result['title']}")
            print("LLM Summary:")
            print(summary)
        
        case "citations":
            query = args.query
            limit = args.limit
            rrf_results = hs.rrf_search(query= query, limit= limit)
            summary = summarization_with_citations(query= query, documents= rrf_results)
            print("Search Results: ")
            for result in rrf_results:
                print(f"- {result['title']}")
            print("LLM Answer:")
            print(summary)
        
        case "question":
            query = args.question
            limit = args.limit
            rrf_results = hs.rrf_search(query=query, limit= limit)
            answer = question_answer(question= query, results= rrf_results)
            print("Search Results: ")
            for result in rrf_results:
                print(f"- {result['title']}")
            print("Answer:")
            print(answer)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()