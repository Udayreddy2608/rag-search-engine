from search_utils import load_movies, load_stop_words
import string
from typing import List, Optional
from nltk.stem import PorterStemmer

class MovieSearchError(Exception):
    """Base exception for movie search errors."""


def normalize(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator)


def tokenize(text: str) -> List[str]:
    return normalize(text).split()


def token_matching(query_tokens: List[str], title: str, stop_words: list, stemmer: PorterStemmer) -> bool:
    for token in query_tokens:
        if token in stop_words:
            continue
        stemmed_token = stemmer.stem(token)
        if stemmed_token in title:
            return True
    return False


def search_movies_kw(query: str) -> List[str]:
    from inverted_index import InvertedIndex
    idx = InvertedIndex()
    idx.load()

    seen, res = set(), []
    query_tokens = tokenize(query)

    for token in query_tokens:
        matching_doc_ids = idx.get_documents(token)

        for match_id in matching_doc_ids:
            if match_id in seen:
                continue

            seen.add(match_id)
            matching_doc = idx.docmap[match_id]['title']
            res.append(matching_doc)

            if len(res) >= 5:
                return res

    return res


def get_bm25_idf(term: str) -> float:
    from inverted_index import InvertedIndex
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_id(term)

def get_bm25_tf(document_id: str, term: str, k1: float = 1.5, b: float = 0.75) -> float:
    from inverted_index import InvertedIndex
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(document_id, term, k1=k1, b=b)