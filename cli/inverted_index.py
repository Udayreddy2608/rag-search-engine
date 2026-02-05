from search_utils import load_movies
import os
import pickle
import string
from nltk.stem import PorterStemmer

CACHE_DIR = "cache"
INDEX_PKL = "index.pkl"
DOCMAP_PKL = "docmap.pkl"

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[str]] = {}
        self.docmap: dict[str, dict] = {}
        self.stemmer = PorterStemmer()

    def __add_document(self, doc_id: str, text: str):
        translator = str.maketrans("", "", string.punctuation)
        text = text.lower().translate(translator)
        tokens = text.split()
        for token in tokens:
            stemmed_token = self.stemmer.stem(token)
            if stemmed_token in self.index:
                self.index[stemmed_token].add(doc_id)
            else:
                self.index[stemmed_token] = {doc_id}

    def get_documents(self, term: str) -> list[str]:
        stemmed_term = self.stemmer.stem(term.lower())
        return sorted(self.index.get(stemmed_term, []))

    
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

    def load(self):
        try:
            with open(os.path.join(CACHE_DIR, INDEX_PKL), "rb") as f:
                self.index = pickle.load(f)

            with open(os.path.join(CACHE_DIR, DOCMAP_PKL), "rb") as f:
                self.docmap = pickle.load(f)

        except FileNotFoundError:
            raise FileNotFoundError("Index files do not exist. Run the build command first.")







        
