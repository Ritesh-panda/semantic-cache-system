"""
Production FAISS vector index for semantic search over the 20 Newsgroups corpus.

Justification:
1. HNSW provides sub-millisecond approximate nearest neighbor search with strong recall.
2. L2 normalization enables cosine similarity using inner product search.
3. IndexIDMap preserves document IDs so search results map directly to corpus entries.
"""

import time
import json
import numpy as np
import faiss
from pathlib import Path

from app.embedding_layer.embedding_model import load_embeddings


INDEX_PATH = Path("data/news_index.faiss")
META_PATH = Path("data/index_meta.json")

DIMENSION = 384
HNSW_M = 32
EF_CONSTRUCTION = 100
EF_SEARCH = 50


def build_index() -> faiss.Index:
    """
    Build FAISS HNSW index from embeddings and persist it to disk.
    """

    print("Loading embeddings...")
    embeddings = load_embeddings()

    print("Normalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)

    start_time = time.time()

    print("Building FAISS HNSW index...")

    base_index = faiss.IndexHNSWFlat(DIMENSION, HNSW_M)
    base_index.hnsw.efConstruction = EF_CONSTRUCTION
    base_index.hnsw.efSearch = EF_SEARCH

    index = faiss.IndexIDMap(base_index)

    ids = np.arange(embeddings.shape[0]).astype(np.int64)

    index.add_with_ids(embeddings, ids)

    build_time = time.time() - start_time

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    metadata = {
        "dimension": DIMENSION,
        "vectors": int(embeddings.shape[0]),
        "index_type": "HNSW",
        "M": HNSW_M,
        "ef_construction": EF_CONSTRUCTION,
        "ef_search": EF_SEARCH
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nIndex Build Complete")
    print("---------------------")
    print(f"Vectors indexed: {metadata['vectors']}")
    print(f"Dimension: {metadata['dimension']}")
    print(f"Build Time: {build_time:.2f} sec")
    print(f"Saved index → {INDEX_PATH}")

    return index


def load_index() -> faiss.Index:
    """
    Load FAISS index from disk.
    """

    if not INDEX_PATH.exists():
        print("Index not found. Building new index...")
        return build_index()

    print("Loading FAISS index from disk...")
    index = faiss.read_index(str(INDEX_PATH))

    return index


def search(query_embedding: np.ndarray, k: int = 10):
    """
    Perform ANN search using the FAISS index.
    Returns distances and indices.
    """

    index = load_index()

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    query_embedding = query_embedding.astype(np.float32)

    faiss.normalize_L2(query_embedding)

    start = time.time()

    distances, indices = index.search(query_embedding, k)

    latency = (time.time() - start) * 1000

    print(f"Search latency: {latency:.3f} ms")

    return distances, indices


if __name__ == "__main__":

    print("\n--- FAISS Index Test ---\n")

    index = load_index()

    embeddings = load_embeddings()

    random_id = np.random.randint(0, embeddings.shape[0])
    query = embeddings[random_id]

    print(f"Running test search with doc id: {random_id}")

    distances, indices = search(query, k=10)

    print("\nTop 10 Results")
    print("----------------")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"{i+1}. Doc ID: {idx} | Score: {dist:.4f}")