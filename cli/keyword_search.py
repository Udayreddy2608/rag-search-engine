from search_utils import load_movies, load_stop_words
import string
from typing import List
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex

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


