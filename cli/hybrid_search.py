from inverted_index import InvertedIndex
from lib.chunked_semantic import ChunkedSemanticSearch
import os

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        # if not os.path.exists(self.idx.index_path):
        #     self.idx.build()
        #     self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        try:
            bm25_results = self._bm25_search(query, limit=500 * limit)
            semantic_results = self.semantic_search.search_chunks(query, limit=500 * limit)
            results = {}
            
            bm25_scores = [res['score'] for res in bm25_results]
            normalized_bm25_scores = normalize_scores(bm25_scores)
            
            for i, res in enumerate(bm25_results):
                doc_id = res['id']
                bm25_score = normalized_bm25_scores[i]
                results[doc_id] = res
                results[doc_id]['bm25_score'] = bm25_score
                results[doc_id]['combined_score'] = hybrid_score(bm25_score, 0.0, alpha=alpha)
            
            for result in semantic_results:
                doc_id = result['id']
                semantic_score = result['score']
                
                if doc_id in results:
                    bm25_score = results[doc_id]['bm25_score']
                    combined_score = hybrid_score(bm25_score, semantic_score, alpha=alpha)
                    results[doc_id]['combined_score'] = combined_score
                    results[doc_id]['semantic_score'] = semantic_score
                else:
                    combined_score = hybrid_score(0.0, semantic_score, alpha=alpha)
                    results[doc_id] = result
                    results[doc_id]['combined_score'] = combined_score
                    results[doc_id]['semantic_score'] = semantic_score
            
            sorted_results = sorted(results.values(), key=lambda x: x['combined_score'], reverse=True)
            return sorted_results[:limit]
        except Exception as e:
            print(f"Error during weighted search: {e}")
            return []

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_scores(scores):
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]