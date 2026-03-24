# RAG Search Engine - Not VIBE CODED
A Python-based search engine implementing both **keyword-based** and **semantic search** capabilities using inverted indexing, BM25 ranking, and transformer-based embeddings.

## 🚀 Features

- **Keyword Search**: Fast inverted index-based search with BM25 ranking
- **Semantic Search**: Neural embedding-based search using Sentence Transformers
- **Chunked Semantic Search**: Advanced semantic search with document chunking for improved relevance
- **Caching**: Efficient caching of embeddings and indices for faster subsequent searches
- **CLI Tools**: Easy-to-use command-line interfaces for both search methods

## 📋 Requirements

- Python >= 3.11
- Dependencies:
  - `nltk` (3.9.1)
  - `numpy` (>= 2.4.2)
  - `sentence-transformers` (>= 5.2.2)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Udayreddy2608/rag-search-engine.git
cd rag-search-engine
```

2. Install dependencies using `uv` (recommended) or `pip`:
```bash
# Using uv
uv sync

# Or using pip
pip install -e .
```

## 📚 Project Structure

```
rag-search-engine/
├── cli/
│   ├── lib/
│   │   ├── semantic_search.py      # Base semantic search implementation
│   │   └── chunked_semantic.py     # Chunked semantic search with overlap
│   ├── inverted_index.py           # Inverted index & BM25 implementation
│   ├── keyword_search.py           # Keyword search utilities
│   ├── keyword_search_cli.py       # CLI for keyword search
│   ├── semantic_search_cli.py      # CLI for semantic search
│   └── search_utils.py             # Shared utilities
├── cache/                          # Cached embeddings and indices
├── data/                           # Data files
└── pyproject.toml
```

## 💡 Usage

### Keyword Search

Search using keyword matching and BM25 ranking:

```bash
python cli/keyword_search_cli.py
```

### Semantic Search

Search using neural embeddings (all-MiniLM-L6-v2 model):

```bash
python cli/semantic_search_cli.py
```

## 🔧 How It Works

### Keyword Search (BM25)

1. **Inverted Index**: Builds an inverted index mapping tokens to documents
2. **Stemming**: Uses Porter Stemmer for word normalization
3. **BM25 Ranking**: Ranks results using the BM25 algorithm with configurable parameters:
   - `k1 = 1.5` (term frequency saturation)
   - `b = 0.75` (length normalization)

### Semantic Search

1. **Embeddings**: Uses Sentence Transformers (all-MiniLM-L6-v2) to encode documents and queries
2. **Cosine Similarity**: Compares query embeddings with document embeddings
3. **Chunking**: Supports semantic chunking with overlap for long documents
   - Default: 4 sentences per chunk with 1 sentence overlap
4. **Caching**: Stores embeddings in numpy format for fast retrieval

## 🧩 Key Components

### InvertedIndex Class

- Builds and manages inverted index
- Implements BM25 scoring
- Handles term frequency and document length normalization
- Provides persistence via pickle serialization

### SemanticSearch Class

- Encodes documents using Sentence Transformers
- Computes cosine similarity for ranking
- Supports embedding caching

### ChunkedSemanticSearch Class

- Extends semantic search with document chunking
- Implements overlapping windowed chunking
- Maintains chunk metadata for result aggregation
- Selects best-matching chunk per document

## 📝 Example

```python
from cli.inverted_index import InvertedIndex

# Build index
idx = InvertedIndex()
idx.save()

# Search
results = idx.bm25_search("action movies", limit=5)
for result in results:
    print(f"{result['title']}: {result['score']}")
```

## 🔍 Advanced Features

- **Stop Word Filtering**: Removes common words for better relevance
- **Stemming**: Normalizes words to root forms
- **Score Rounding**: Configurable precision for similarity scores
- **Metadata Tracking**: Chunk-level metadata for debugging and analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Made with 🖤 by Uday

---

*This project was built as part of learning Retrieval-Augmented Generation (RAG) techniques and search engine implementation.*
