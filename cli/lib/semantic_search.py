from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CACHE_STORAGE = "cache"
EMBEDDING_FILE = "movie_embeddings.npy"

class SemanticSearch:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str) -> list[float]:
        if not text:
            raise ValueError("Input text cannot be empty.")
        return self.model.encode(text)
    
    def build_embeddings(self, documents: list[str]) -> np:
        try:
            self.documents = documents
            movie_strings = []
            for document in self.documents:
                self.document_map[document['id']] = document
                movie_strings.append(f"{document['title']}: {document['description']}")
            self.embeddings = self.model.encode(movie_strings,show_progress_bar=True)
            with open(f"{CACHE_STORAGE}/{EMBEDDING_FILE}", "wb") as f:
                np.save(f, self.embeddings)
        except Exception as e:
            print(f"Error building embeddings: {e}")

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for idx, document in enumerate(documents):
            self.document_map[idx] = document
        try:
            self.embeddings = np.load(f"{CACHE_STORAGE}/{EMBEDDING_FILE}")

            if len(self.embeddings) != len(documents):
                print("Cache size mismatch. Rebuilding embeddings...")
                self.build_embeddings(documents)

            else:
                print("Embeddings loaded from cache.")

            return self.embeddings
        except FileNotFoundError:
            print("Cache not found. Building embeddings...")
            self.build_embeddings(documents)
            return self.embeddings


    def search(self, query: str, limit: int = 5) -> list[dict]:
        try:
            assert len(self.embeddings) == len(self.document_map)
            print(f"Searching for: '{query}' with limit {limit}")
            embeddings = self.embeddings
            if self.embeddings is None:
                raise ValueError("Embeddings not loaded. Please load or create embeddings first.")
            print("Generating query embedding...")
            query_embedding = self.generate_embedding(query)
            print("Query embedding generated")

            cosine_sims = []
            for idx, doc_embedding in enumerate(embeddings):
                score = cosine_similarity(query_embedding, doc_embedding)
                document = self.document_map.get(idx)
                if document:
                    cosine_sims.append((score, document))
            cosine_sims.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, document in cosine_sims[:limit]:
                results.append({
                    "score": score,
                    "title": document['title'],
                    "description": document['description']
                })
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

def verify_model(model_name: str) -> bool:
    try:
        semantic_search = SemanticSearch(model_name)
        model_name = semantic_search.model
        print(f"Model loaded: {model_name}")
        print(f"Max sequence length: {model_name.max_seq_length}")
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")

def embed_text(text: str) -> list[float]:
    try:    
        if not text:
            raise ValueError("Input text cannot be empty.")
        semantic_search = SemanticSearch()
        embedding = semantic_search.generate_embedding(text)
        print(f"Text: {text}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"Error embedding text: {e}")

def embed_query_text(query: str) -> list[float]:
    try:    
        if not query:
            raise ValueError("Input query cannot be empty.")
        semantic_search = SemanticSearch()
        embedding = semantic_search.generate_embedding(query)
        print(f"Query: {query}")
        print(f"First 5 dimensions: {embedding[:5]}")
        print(f"Shape: {embedding.shape}")
        return embedding
    except Exception as e:
        print(f"Error embedding query text: {e}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)