from search_utils import load_movies
import os
import pickle

CACHE_DIR = "cache"

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[str]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: str, text: str):
        tokens = text.lower().split()

        for token in tokens:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

    def get_documents(self, term: str) -> list[str]:
        return sorted(self.index.get(term.lower(), []))

    
    def build(self):
        movies = load_movies()
        for movie in movies:
            full_title = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id=movie['id'],text= full_title)
            self.docmap[movie['id']] = movie
    
    def save(self):
        self.build()
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)

        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)






        
