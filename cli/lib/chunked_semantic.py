from lib.semantic_search import SemanticSearch, cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import json
import os


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CACHE_DIR = "cache"
NP_CHUNK_EMBEDDINGS = "chunk_embeddings.npy"
SCORE_PRECISION = 4

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = EMBEDDING_MODEL) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[dict] = None
        self.model = SentenceTransformer(model_name)

    def __semantic_chunking(self, text: str, max_chunk_size: int = 4, overlap: int = 1) -> list[str]:
        text = text.strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        sentence_length = len(sentences)
        step = max_chunk_size - overlap
        for i in range(0, sentence_length, step):
            chunk = " ".join(sentences[i:i+max_chunk_size])
            chunks.append(chunk)
            if i + max_chunk_size >= sentence_length:
                break
        return chunks

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks: list[str] = []
        chunk_metadata: list[dict] = []
        for idx, document in enumerate(documents):
            if document['description']:
                self.document_map[document['id']] = document
                document_chunks = self.__semantic_chunking(document['description'], max_chunk_size=4, overlap=1)
                chunks.extend(document_chunks)
                for chunk_idx, chunk in enumerate(document_chunks):
                    chunk_metadata.append({
                        "move_idx": document['id'],
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(document_chunks),
                    })

        self.chunk_metadata = chunk_metadata
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        with open(f"{CACHE_DIR}/{NP_CHUNK_EMBEDDINGS}", "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open(f"{CACHE_DIR}/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        try:
            if os.path.exists(f"{CACHE_DIR}/{NP_CHUNK_EMBEDDINGS}") and \
            os.path.exists(f"{CACHE_DIR}/chunk_metadata.json"):

                with open(f"{CACHE_DIR}/{NP_CHUNK_EMBEDDINGS}", "rb") as f:
                    self.chunk_embeddings = np.load(f)

                with open(f"{CACHE_DIR}/chunk_metadata.json", "r") as f:
                    metadata = json.load(f)
                    self.chunk_metadata = metadata["chunks"]

                self.document_map = {doc["id"]: doc for doc in documents}

                return self.chunk_embeddings

            raise FileNotFoundError("Cache files not found")

        except Exception as e:
            print(f"Error loading chunk embeddings: {e}")
            print("Rebuilding chunk embeddings...")
            return self.build_chunk_embeddings(documents)

    
    def search_chunks(self, query: str, limit: int = 10):
        try:
            query_embeddings = self.model.encode(query)
            chunk_scores = []
            movie_idx_scores = {}
            if self.chunk_embeddings is None or self.chunk_metadata is None:
                self.load_or_create_chunk_embeddings(self.documents)

            for idx, chunk_embedding in enumerate(self.chunk_embeddings):
                score = cosine_similarity(query_embeddings, chunk_embedding)
                metadata = self.chunk_metadata[idx]
                chunk_scores.append({
                    "chunk_idx": idx, 
                    "move_idx": metadata["move_idx"],
                    "score": score,
                })
            
            for score in chunk_scores:
                if score['move_idx'] not in movie_idx_scores or score['score'] > movie_idx_scores[score['move_idx']]['score']:
                    movie_idx_scores[score['move_idx']] = score
            
            chunk_scores.sort(key=lambda x: x["score"], reverse=True)
            top_chunks = chunk_scores[:limit]
            results = []
            for chunk in top_chunks:
                document = self.document_map[chunk['move_idx']]
                results.append({
                    "id": chunk['move_idx'],
                    "title": document['title'],
                    "document": document['description'][:100],
                    "metadata": self.chunk_metadata[chunk['chunk_idx']] or {},
                    "score": round(chunk['score'], SCORE_PRECISION)
                })
            return results
        except Exception as e:
            print(f"Error during chunk search: {e}")
            return []