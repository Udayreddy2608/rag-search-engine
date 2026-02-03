from search_utils import load_movies, load_stop_words
import string
from typing import List
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
    try:
        data = load_movies()
        stop_words = load_stop_words()
        ps = PorterStemmer()
    except Exception as e:
        raise MovieSearchError("Initialization failed") from e

    results = []
    query_tokens = tokenize(query)

    for movie in data:
        try:
            raw_title = movie["title"]
            clean_title = normalize(raw_title)

            if token_matching(query_tokens, clean_title, stop_words, ps):
                results.append(raw_title)
        except Exception as e:
            raise MovieSearchError("Error while processing movie entry") from e
    return results
