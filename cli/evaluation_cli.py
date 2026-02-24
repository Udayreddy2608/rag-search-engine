import argparse
import json
from hybrid_search import HybridSearch
from search_utils import load_movies

DATA_PATH = "data/golden_dataset.json"

import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    documents = load_movies()
    hs = HybridSearch(documents)

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    test_cases = data["test_cases"]

    for test_case in test_cases:
        query = test_case["query"]

        retrieved = hs.rrf_search(query, limit=limit)
        ret_titles = [movie["title"] for movie in retrieved]

        total_count = len(ret_titles)
        rel_count = sum(
            1 for title in ret_titles
            if title in test_case["relevant_docs"]
        )

        precision = rel_count / total_count if total_count > 0 else 0.0
        recall = rel_count / len(test_case["relevant_docs"]) 
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(ret_titles)}")
        print(f"  - Relevant: {', '.join(test_case['relevant_docs'])}")
        print()
if __name__ == "__main__":
    main()