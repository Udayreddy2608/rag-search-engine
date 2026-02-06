from search_utils import load_movies
import os
import pickle
import string
from nltk.stem import PorterStemmer
from collections import Counter

CACHE_DIR = "cache"
INDEX_PKL = "index.pkl"
DOCMAP_PKL = "docmap.pkl"
TF_PKL = "term_frequencies.pkl"


## TF-IDF = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

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
        self.stemmer = PorterStemmer()

    def __add_document(self, doc_id: str, text: str):
        translator = str.maketrans("", "", string.punctuation)
        text = text.lower().translate(translator)
        tokens = text.split()
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        self.term_frequencies[doc_id] = Counter(stemmed_tokens)
        for token in tokens:
            stemmed_token = self.stemmer.stem(token)
            if stemmed_token in self.index:
                self.index[stemmed_token].add(doc_id)
            else:
                self.index[stemmed_token] = {doc_id}

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
    
    def save(self):
        self.build()
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, INDEX_PKL), "wb") as f:
            pickle.dump(self.index, f)

        with open(os.path.join(CACHE_DIR, DOCMAP_PKL), "wb") as f:
            pickle.dump(self.docmap, f)
        
        with open(os.path.join(CACHE_DIR, TF_PKL), "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        try:
            with open(os.path.join(CACHE_DIR, INDEX_PKL), "rb") as f:
                self.index = pickle.load(f)

            with open(os.path.join(CACHE_DIR, DOCMAP_PKL), "rb") as f:
                self.docmap = pickle.load(f)
            
            with open(os.path.join(CACHE_DIR, TF_PKL), "rb") as f:
                self.term_frequencies = pickle.load(f)

        except FileNotFoundError:
            raise FileNotFoundError("Index files do not exist. Run the build command first.")







        
