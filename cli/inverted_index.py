import math
from keyword_search import tokenize
from search_utils import load_movies, format_search_result
import os
import pickle
import string
from nltk.stem import PorterStemmer
from collections import Counter

CACHE_DIR = "cache"
INDEX_PKL = "index.pkl"
DOCMAP_PKL = "docmap.pkl"
TF_PKL = "term_frequencies.pkl"
DOC_LENS_PKL = "doc_lengths.pkl"
BM25_K1 = 1.5
BM25_B = 0.75




## TF-IDF = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
## length_norm = length_norm = 1 - b + b * (doc_length / avg_doc_length)
## tf_componenent = (tf * (k1 + 1)) / (tf + k1 * length_norm)

"""
-> Inverted Index Structure:

idx = {
    "term1": {"doc1", "doc2"},
    "term2": {"doc2", "doc3"},
}

docmap = {
    "doc1": {"title": "Movie 1", "description": "Description of movie 1"},
    "doc2": {"title": "Movie 2", "description": "Description of movie 2"},
    "doc3": {"title": "Movie 3", "description": "Description of movie 3"},
}

term_frequencies = {
    "doc1": {"term1": 3, "term2": 1},
    "doc2": {"term1": 1, "term2": 2},
    "doc3": {"term1": 0, "term2": 4},
}
"""

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[str]] = {}
        self.docmap: dict[str, dict] = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.stemmer = PorterStemmer()

    def __add_document(self, doc_id: str, text: str):
        translator = str.maketrans("", "", string.punctuation)
        text = text.lower().translate(translator)
        tokens = text.split()
        self.doc_lengths[doc_id] = len(tokens)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        self.term_frequencies[doc_id] = Counter(stemmed_tokens)
        for token in tokens:
            stemmed_token = self.stemmer.stem(token)
            if stemmed_token in self.index:
                self.index[stemmed_token].add(doc_id)
            else:
                self.index[stemmed_token] = {doc_id}
    
    def __get_average_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[str]:
        stemmed_term = self.stemmer.stem(term.lower())
        return sorted(self.index.get(stemmed_term, []))

    def get_tf(self, doc_id: str, term: str) -> int:
        stemmed_term = self.stemmer.stem(term.lower())
        freqs = self.term_frequencies.get(int(doc_id), Counter())
        return freqs.get(stemmed_term, 0)  

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            full_title = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id=doc_id,text= full_title)
            self.docmap[doc_id] = movie
    
    def get_bm25_id(self, term: str) -> float:
        #log((N - df + 0.5) / (df + 0.5) + 1)
        tokens = term.lower().split()
        if len(tokens) > 1:
            raise ValueError("BM25 IDF calculation only supports single terms")
        stemmed_term = self.stemmer.stem(tokens[0])
        N = len(self.docmap)
        df = len(self.index.get(stemmed_term, []))
        bmidf = math.log((N - df + 0.5) / (df + 0.5)+1)
        return bmidf
    
    def get_bm25_tf(self, doc_id: str, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        # BM25_SATURATION_FORMULA = (tf * (k1 + 1)) / (tf + k1)
        tf = self.get_tf(doc_id, term)
        # saturated_score = (tf * (k1 + 1)) / (tf + k1) if tf > 0 else 0
        avg_doc_length = self.__get_average_doc_length()
        if avg_doc_length == 0:
            return 0.0
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component
    
    def bm25(self,doc_id: str, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        #BM25 = bm25_tf * bm25_idf
        idf = self.get_bm25_id(term)
        tf_component = self.get_bm25_tf(doc_id, term, k1=k1, b=b)
        return idf * tf_component
    
    def bm25_search(self, query: str, limit: int = 5) -> list[dict]:
        query_tokens = tokenize(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results
    
    def save(self):
        self.build()
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, INDEX_PKL), "wb") as f:
            pickle.dump(self.index, f)

        with open(os.path.join(CACHE_DIR, DOCMAP_PKL), "wb") as f:
            pickle.dump(self.docmap, f)
        
        with open(os.path.join(CACHE_DIR, TF_PKL), "wb") as f:
            pickle.dump(self.term_frequencies, f)
        
        with open(os.path.join(CACHE_DIR, DOC_LENS_PKL), "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        try:
            with open(os.path.join(CACHE_DIR, INDEX_PKL), "rb") as f:
                self.index = pickle.load(f)

            with open(os.path.join(CACHE_DIR, DOCMAP_PKL), "rb") as f:
                self.docmap = pickle.load(f)
            
            with open(os.path.join(CACHE_DIR, TF_PKL), "rb") as f:
                self.term_frequencies = pickle.load(f)
            
            with open(os.path.join(CACHE_DIR, DOC_LENS_PKL), "rb") as f:
                self.doc_lengths = pickle.load(f)

        except FileNotFoundError:
            raise FileNotFoundError("Index files do not exist. Run the build command first.")







        
