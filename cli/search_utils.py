import json
from typing import Any

DATA_PATH = "data/movies.json"
STOPWORDS_PATH = "data/stopwords.txt"
SCORE_PRECISION = 3

def load_movies() -> list[dict]:
    with open(DATA_PATH,"r") as f:
        data = json.load(f)
    return data['movies']

def load_stop_words() -> list:
    with open(STOPWORDS_PATH,"r") as f:
        data = f.read()
        words = data.splitlines()
    return words

def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

